"""The click application: every tool of the configured MCP server becomes a subcommand.

The group is lazy. Nothing is loaded when the process starts; the config is read, and the
tool list taken from the cache or the server, only when click asks for the subcommands, which
happens after the group's own options are parsed. ``--config``, ``-v`` and ``--refresh-tools``
are eager and ``--help`` is not, so ``--help --config other.yaml`` lists the tools of the
other config.

Exit codes: 0 on success, 1 when the tool reports an error, 2 for a usage or config error,
3 when the server cannot be reached or does not speak MCP, 4 when a server needs a sign-in
that is missing, expired or refused.

The MCP SDK, httpx and anyio are imported only by the code that contacts a server or prints a
tool's result. ``--help`` from a fresh cache, ``schema``, ``config`` and a consumer's extra
commands start without them, which saves most of the start-up time.
"""

from __future__ import annotations

import difflib
import inspect
import json
import logging
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import click

from mcp_to_cli_kit import config_view, redaction
from mcp_to_cli_kit.arguments import CompiledTool, compile_tool, kebab_case
from mcp_to_cli_kit.cache import CachedTools, SchemaCache, default_cache_dir
from mcp_to_cli_kit.client_defaults import (
    DEFAULT_RETRY_DELAYS,
    KIT_DISTRIBUTION,
    client_header_value,
    distribution_version,
)
from mcp_to_cli_kit.config import (
    BUILTIN_COMMANDS,
    CliConfig,
    ConfigError,
    GroupMount,
    LoadedConfig,
    PrefixMount,
    ResolvedServer,
    ToolGroup,
    check_consumer_keys,
    config_path_variable,
    default_config_path,
    expand_url,
    load_config,
)
from mcp_to_cli_kit.lazy import EXIT_INTERRUPTED as EXIT_INTERRUPTED
from mcp_to_cli_kit.lazy import LazyKitGroup, _standalone, run
from mcp_to_cli_kit.logs import configure_logging
from mcp_to_cli_kit.output import context_flags, finish, mode_options, render, resolve_mode

if TYPE_CHECKING:
    from click.shell_completion import CompletionItem

if TYPE_CHECKING:
    from mcp import types

    from mcp_to_cli_kit.client import PortalProvider, ServerClient
    from mcp_to_cli_kit.oauth import TokenStore

logger = logging.getLogger(__name__)

# The result type is a string, so the alias exists at run time without importing the SDK.
ResultClassifier = Callable[[str, str, "types.CallToolResult"], bool]
"""Called with the server name, the tool name and a result the server did not mark as an error.

Returning True treats the result as an error: it is printed to stderr and the exit code is 1.
For servers that report failures as ordinary text.
"""

ConfigLayer = Callable[[click.Context, Mapping[str, Any]], Mapping[str, Any]]
"""Called once per invocation with the bin's group context and the consumer keys the config file holds.

Returns a config layer in the shape of the file, merged over the code default and under the file's
own keys and the environment. The context's ``params`` hold the global options, the consumer's
``extra_params`` included. Raising ``ConfigError`` ends the command with exit code 2.
"""

EXIT_TOOL_ERROR = 1
EXIT_USAGE = 2
EXIT_TRANSPORT = 3
EXIT_AUTH = 4

_RUNTIME_KEY = "mcp_to_cli_kit.runtime"

# Only at the start of an error, so a tool whose ordinary output mentions such a phrase keeps the cache.
_UNKNOWN_TOOL = re.compile(
    r"^\s*(?:error[^:\n]{0,60}:\s*)?(?:unknown tool|no such tool|tool not found)\b", re.IGNORECASE
)

# JSON-RPC "method not found" and "invalid params", the codes servers use for an unknown tool.
_UNKNOWN_TOOL_CODES = frozenset({-32601, -32602})


class ToolFailure(click.ClickException):
    """The server rejected the call. Printed as one ``Error:`` line."""

    exit_code = EXIT_TOOL_ERROR


class ConfigFailure(click.ClickException):
    """The configuration is missing or invalid. Printed as one ``Error:`` line."""

    exit_code = EXIT_USAGE


class UnresolvedServer(ConfigFailure):
    """A server cannot be contacted because an environment variable its config refers to is not set.

    Like a transport failure, it only affects the commands that need that server.
    """


class TransportFailure(click.ClickException):
    """The server could not be reached or did not speak MCP. Printed as one ``Error:`` line."""

    exit_code = EXIT_TRANSPORT


class AuthFailure(click.ClickException):
    """A server needs a sign-in that is missing, expired or refused. Printed as one ``Error:`` line."""

    exit_code = EXIT_AUTH


@dataclass(frozen=True)
class App:
    """What a consumer hands the kit: everything that does not come from the config."""

    bin_name: str
    default_config: Mapping[str, Any] | CliConfig | None = None
    distribution: str | None = None
    result_classifier: ResultClassifier | None = None
    retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS
    extra_params: tuple[click.Parameter, ...] = ()
    extra_commands: tuple[click.Command, ...] = ()
    consumer_keys: frozenset[str] = frozenset()
    config_layer: ConfigLayer | None = None

    @property
    def bin_version(self) -> str:
        return distribution_version(self.distribution, self.bin_name)

    @property
    def reserved_commands(self) -> frozenset[str]:
        """The command names no tool may take: the kit's built-ins and the consumer's extra commands."""
        return BUILTIN_COMMANDS | {command.name for command in self.extra_commands if command.name}

    def extra_command(self, name: str) -> click.Command | None:
        return next((command for command in self.extra_commands if command.name == name), None)


@dataclass(frozen=True)
class ToolEntry:
    """One tool of one server, under the command name it got."""

    server: str
    command: str
    tool: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.tool["name"])


def _normalized(name: str) -> str:
    return name.replace("-", "_")


@dataclass
class ToolIndex:
    """Tool commands by command name, with the tool's own spellings as aliases.

    Exact aliases are looked up before the spellings with ``-`` turned into ``_``, so a tool's own
    name always finds that tool, whatever order the tools were added in.
    """

    entries: dict[str, ToolEntry] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    normalized_aliases: dict[str, str] = field(default_factory=dict)
    reserved: frozenset[str] = BUILTIN_COMMANDS

    def add(self, server: str, local_name: str, tool: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
        """Add a tool under the kebab case of ``local_name``; returns the command name, None when it is taken.

        A tool whose command would be a reserved name (a built-in command such as ``schema`` or
        ``config``, or a consumer's extra command) gets a ``-tool`` suffix, at every level.
        """
        command = kebab_case(local_name)
        if command in self.reserved:
            command = f"{command}-tool"
        if command in self.entries:
            command = local_name
        if command in self.entries:
            logger.warning("Skipping the tool %r of %s: its command name is taken", tool.get("name"), server)
            return None
        self.entries[command] = ToolEntry(server, command, tool)
        for alias in (local_name, *aliases):
            self.aliases.setdefault(alias, command)
            self.normalized_aliases.setdefault(_normalized(alias), command)
        return command

    def find_exact(self, name: str) -> ToolEntry | None:
        """A command name or an exact alias."""
        if name in self.entries:
            return self.entries[name]
        alias = self.aliases.get(name)
        return self.entries.get(alias) if alias is not None else None

    def find_normalized(self, name: str) -> ToolEntry | None:
        """An alias spelled with ``-`` and ``_`` swapped."""
        alias = self.normalized_aliases.get(_normalized(name))
        return self.entries.get(alias) if alias is not None else None

    def find(self, name: str) -> ToolEntry | None:
        return self.find_exact(name) or self.find_normalized(name)


@dataclass
class ServerTools(ToolIndex):
    """The tool commands of one server, before any mount prefix.

    A ``strip_prefixes`` entry is first stripped from the tool names (see ``_local_names``); the
    tool's own name stays an alias, and ``hidden_tools`` still lists the tools by their own names.
    Tool groups match the stripped names. Tools whose names start with a ``tool_groups`` prefix
    live in that group, under their name without the prefix; the rest stay here, at the server's
    mount level. The longest matching prefix wins. A grouped tool's full name still finds it from
    the mount level, as an alias. Hidden tools are left out before grouping, so they are in no
    group either.

    Attributes:
        groups: The tool groups by group name, each with its tools.
        clashes: Names that are both an ungrouped tool command and a tool group.
    """

    groups: dict[str, ToolIndex] = field(default_factory=dict)
    clashes: list[str] = field(default_factory=list)
    grouped_aliases: dict[str, tuple[str, str]] = field(default_factory=dict)
    normalized_grouped_aliases: dict[str, tuple[str, str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        server: str,
        tools: list[dict[str, Any]],
        hidden: list[str],
        tool_groups: list[ToolGroup],
        strip_prefixes: Sequence[str] = (),
        reserved: frozenset[str] = BUILTIN_COMMANDS,
    ) -> ServerTools:
        index = cls(groups={group.group: ToolIndex(reserved=reserved) for group in tool_groups}, reserved=reserved)
        by_prefix = sorted(tool_groups, key=lambda group: len(group.prefix), reverse=True)
        group_names = [group.group for group in tool_groups]
        for name, local, tool in _local_names(server, tools, hidden, strip_prefixes, group_names):
            aliases = (name,) if local != name else ()
            group = next((g for g in by_prefix if local.startswith(g.prefix) and len(local) > len(g.prefix)), None)
            if group is None:
                index.add(server, local, tool, aliases)
                continue
            command = index.groups[group.group].add(server, local[len(group.prefix) :], tool, (local, *aliases))
            if command is not None:
                for alias in (local, *aliases):
                    index.grouped_aliases.setdefault(alias, (group.group, command))
                    index.normalized_grouped_aliases.setdefault(_normalized(alias), (group.group, command))
        index.clashes = sorted(set(index.entries) & set(index.groups))
        return index

    def find(self, name: str) -> ToolEntry | None:
        """Exact names before normalized spellings, the mount level before the tool groups."""
        return (
            self.find_exact(name)
            or self._grouped(self.grouped_aliases.get(name))
            or self.find_normalized(name)
            or self._grouped(self.normalized_grouped_aliases.get(_normalized(name)))
        )

    def _grouped(self, target: tuple[str, str] | None) -> ToolEntry | None:
        return self.groups[target[0]].entries.get(target[1]) if target is not None else None

    def find_in_group(self, group: str, name: str) -> ToolEntry | None:
        tools = self.groups.get(group)
        return tools.find(name) if tools is not None else None


def _local_names(
    server: str,
    tools: list[dict[str, Any]],
    hidden: list[str],
    strip_prefixes: Sequence[str],
    group_names: Sequence[str] = (),
) -> list[tuple[str, str, dict[str, Any]]]:
    """Each visible tool's own name and the name its command is built from, with ``strip_prefixes`` applied.

    The first configured prefix the name starts with is stripped, unless nothing would be left.
    ``hidden`` lists tools by their own names. A stripped name whose command would clash with
    another tool's command, or with one of the server's tool groups, is not stripped, and the
    clash is logged as a warning, so neither side shadows the other. A tool that keeps its full
    name can clash anew, so this repeats until nothing changes; the outcome does not depend on
    the order the server lists its tools in.
    """
    hidden_names = set(hidden)
    stripped: list[tuple[str, str, dict[str, Any]]] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in hidden_names:
            continue
        prefix = next((p for p in strip_prefixes if name.startswith(p) and len(name) > len(p)), "")
        stripped.append((name, name[len(prefix) :], tool))

    reserved = set(group_names)
    kept: dict[str, list[str]] = {}  # tools that keep their full name, with what they would clash with
    while True:
        commands: dict[str, list[str]] = {}
        for name, local, _ in stripped:
            commands.setdefault(kebab_case(name if name in kept else local), []).append(name)
        changed = False
        for name, local, _ in stripped:
            if local == name or name in kept:
                continue
            command = kebab_case(local)
            others = [other for other in commands[command] if other != name]
            if others or command in reserved:
                kept[name] = others if others else [f"the tool group {command}"]
                changed = True
        if not changed:
            break

    for name in sorted(kept):
        local = next(local for other, local, _ in stripped if other == name)
        logger.warning(
            "Not stripping the prefix from the tool %r of %s: %r would clash with %s",
            name,
            server,
            local,
            ", ".join(sorted(other if other.startswith("the tool group") else repr(other) for other in kept[name])),
        )
    return [(name, name if name in kept else local, tool) for name, local, tool in stripped]


@dataclass
class TopLevelListing:
    """What the bin's own help lists: the tools at the top level, the group mounts and the servers that failed."""

    tools: dict[str, ToolEntry] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    unresolved: set[str] = field(default_factory=set)
    unauthenticated: set[str] = field(default_factory=set)


@dataclass
class Runtime:
    """The state of one invocation: the config, the clients and the tool lists.

    The clients, the token store and the event loop behind them are created on first use, so an
    invocation that never contacts a server never imports the MCP SDK.
    """

    app: App
    loaded: LoadedConfig
    refresh: bool
    command_path: str = ""
    """How the user calls the kit group, such as ``host mcp`` when it is mounted under a host CLI; for hints."""
    clients: dict[str, ServerClient] = field(default_factory=dict)
    cache: SchemaCache = field(init=False)
    _portal: PortalProvider | None = None
    _tokens: TokenStore | None = None
    _tools: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _live: set[str] = field(default_factory=set)
    _indexes: dict[str, ServerTools] = field(default_factory=dict)
    _listing: TopLevelListing | None = None

    def __post_init__(self) -> None:
        self.command_path = self.command_path or self.app.bin_name
        self.cache = SchemaCache(default_cache_dir(self.app.bin_name))

    @property
    def tokens(self) -> TokenStore:
        if self._tokens is None:
            from mcp_to_cli_kit.oauth import TokenStore, default_token_dir

            self._tokens = TokenStore(default_token_dir(self.app.bin_name))
        return self._tokens

    def client(self, server: str) -> ServerClient:
        """The client of a server, created on first use."""
        if server not in self.clients:
            from mcp import types

            from mcp_to_cli_kit.client import PortalProvider, ServerClient

            if self._portal is None:
                self._portal = PortalProvider()
            version = self.app.bin_version
            self.clients[server] = ServerClient(
                server,
                self.config.servers[server],
                client_header_value(self.app.bin_name, version),
                self._portal,
                self.app.retry_delays,
                client_info=types.Implementation(name=self.app.bin_name, version=version),
                token_store=self.tokens,
                login_command=self.login_command(server),
            )
        return self.clients[server]

    def check_extra_commands(self) -> None:
        """Fail when a group mount or a top-level tool group takes the name of one of the consumer's extra commands."""
        for name in [*self.group_mounts(), *self.top_level_tool_groups()]:
            if self.app.extra_command(name) is not None:
                raise ConfigFailure(
                    f"The group {name!r} has the name of a command of {self.command_path}. "
                    "Give the group mount or tool group another name."
                )

    def oauth_servers(self) -> list[str]:
        """The servers that sign in with OAuth, in config order."""
        return [name for name, server in self.config.servers.items() if server.auth is not None]

    def login_command(self, server: str) -> str:
        """The command that signs in to a server; it names the server only when the bin has several to sign in to."""
        several = len([name for name, config in self.loaded.config.servers.items() if config.auth is not None]) > 1
        return f"{self.command_path} auth login" + (f" {server}" if several else "")

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        if self._portal is not None:
            self._portal.__exit__(None, None, None)
        # The expanded secrets of this invocation; a later one in the same process registers its own.
        redaction.clear()

    def tools(self, server: str) -> list[dict[str, Any]]:
        """The tools of a server: from a fresh cache, else from the server, else from an expired cache."""
        if server in self._tools:
            return self._tools[server]
        cached = self._load_cache(server)
        if cached is not None and cached.fresh and not self.refresh:
            logger.debug("Using the cached tool list of %s", server)
            self._tools[server] = cached.tools
            return cached.tools
        return self._fetch(server, fallback=cached.tools if cached is not None else None)

    def _fetch(self, server: str, fallback: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        from mcp_to_cli_kit.client import ServerRejectedError, TransportError
        from mcp_to_cli_kit.oauth import AuthError

        client = self.client(server)
        self._live.add(server)
        try:
            tools = client.list_tools()
        except (TransportError, ServerRejectedError, ConfigError, AuthError) as error:
            if fallback is None:
                if isinstance(error, ConfigError):
                    raise UnresolvedServer(str(error)) from error
                if isinstance(error, AuthError):
                    raise AuthFailure(f"cannot list the tools of {server}: {error}") from error
                raise TransportFailure(f"cannot list the tools of {client.display_url} ({server}): {error}") from error
            click.echo(
                f"{self.command_path}: warning: cannot refresh the tool list of {server} ({error}); "
                "using the expired cached list.",
                err=True,
            )
            self._tools[server] = fallback
            return fallback
        self.cache.save(server, client.server.url, tools, client.cache_url())
        self._tools[server] = tools
        return tools

    def _load_cache(self, server: str) -> CachedTools | None:
        """The cache entry of a server, told apart by its URL; reading it needs no client."""
        client = self.clients.get(server)
        if client is not None:
            return self.cache.load(server, client.server.url, client.cache_url())
        config = self.config.servers[server]
        return self.cache.load(server, config.url, expand_url(config))

    def server_tools(self, server: str) -> ServerTools:
        """A server's tools by their command name within the server, without any mount prefix."""
        if server not in self._indexes:
            self._indexes[server] = self._index(server, self.tools(server))
        return self._indexes[server]

    def _index(self, server: str, tools: list[dict[str, Any]]) -> ServerTools:
        config = self.config.servers[server]
        index = ServerTools.build(
            server, tools, config.hidden_tools, config.tool_groups, config.strip_prefixes, self.app.reserved_commands
        )
        if index.clashes:
            names = ", ".join(repr(name) for name in index.clashes)
            raise ConfigFailure(
                f"Server {server}: {names} is both a tool and a tool group. "
                "Rename the tool group, or hide the tool with hidden_tools."
            )
        return index

    def find_in_server(self, server: str, name: str, group: str | None = None) -> ToolEntry | None:
        """Find a tool of one server, or of one of its tool groups, by command name, tool name or snake case.

        A miss in a cached list fetches the list once.
        """

        def lookup() -> ToolEntry | None:
            tools = self.server_tools(server)
            return tools.find(name) if group is None else tools.find_in_group(group, name)

        entry = lookup()
        if entry is not None or server in self._live:
            return entry
        logger.debug("No command %r in the cached tool list of %s; fetching it again", name, server)
        self._tools.pop(server, None)
        self._indexes.pop(server, None)
        cached = self._load_cache(server)
        self._fetch(server, fallback=cached.tools if cached is not None else None)
        return lookup()

    def top_level_tool_groups(self) -> dict[str, tuple[str, ToolGroup]]:
        """The tool groups of root and prefix mounted servers, by top-level name. Known without a request.

        A prefix mount puts its prefix in front of its tool groups too, as in front of its tools.
        """
        groups = {}
        for server, prefix in self.top_level_servers():
            for group in self.config.servers[server].tool_groups:
                groups[prefix + group.group] = (server, group)
        return groups

    @property
    def config(self) -> CliConfig:
        return self.loaded.config

    def group_mounts(self) -> dict[str, str]:
        """The servers mounted as a subcommand group, by group name. Known without any request."""
        return {
            server.mount.group: name
            for name, server in self.config.servers.items()
            if isinstance(server.mount, GroupMount)
        }

    def top_level_servers(self) -> list[tuple[str, str]]:
        """The servers whose tools are top-level commands, with their command prefix ("" for the root mount)."""
        mounted = []
        for name, server in self.config.servers.items():
            if server.mount == "root":
                mounted.append((name, ""))
            elif isinstance(server.mount, PrefixMount):
                mounted.append((name, server.mount.prefix))
        return mounted

    def offline_tools(self, server: str, *, fresh_only: bool) -> ServerTools | None:
        """A server's tools without any request: the list already loaded, else its cache, else None.

        With ``fresh_only`` an expired cache, or any cache under ``--refresh-tools``, counts as missing.
        """
        if server in self._indexes:
            return self._indexes[server]
        if server in self._tools:
            return self.server_tools(server)
        if fresh_only and self.refresh:
            return None
        cached = self._load_cache(server)
        if cached is None or (fresh_only and not cached.fresh):
            return None
        index = self._index(server, cached.tools)
        if cached.fresh:
            self._tools[server] = cached.tools
            self._indexes[server] = index
        return index

    def candidates(self, name: str) -> list[tuple[str, str]]:
        """The top-level servers that could define a command name, with their prefixes."""
        return [
            (server, prefix)
            for server, prefix in self.top_level_servers()
            if name.startswith(prefix) and name != prefix
        ]

    def resolve_top_level(self, name: str) -> ToolEntry | None:
        """Find a top-level tool command, asking only the servers that could define that name.

        The first pass uses the lists already loaded or freshly cached and sends nothing. The
        servers are asked only when that is not conclusive: those whose list is stale or missing
        when one server has the name, and every candidate when nobody has it.

        Raises:
            ConfigFailure: Two servers define the same command name.
            TransportFailure: The name was not found, and a server that might define it is unreachable.
        """
        candidates = self.candidates(name)
        found: dict[str, ToolEntry] = {}
        unknown: list[tuple[str, str]] = []
        for server, prefix in candidates:
            tools = self.offline_tools(server, fresh_only=True)
            if tools is None:
                unknown.append((server, prefix))
                continue
            entry = tools.find(name[len(prefix) :])
            if entry is not None:
                found[server] = replace(entry, command=prefix + entry.command)

        to_ask = unknown if found else candidates
        unchecked: list[tuple[str, TransportFailure | UnresolvedServer | AuthFailure]] = []
        for server, prefix in to_ask:
            if server in found:
                continue
            try:
                entry = self.find_in_server(server, name[len(prefix) :])
            except (TransportFailure, UnresolvedServer, AuthFailure) as error:
                unchecked.append((server, error))
                continue
            if entry is not None:
                found[server] = replace(entry, command=prefix + entry.command)

        if len(found) > 1:
            raise _collision(name, [server for server in self.config.servers if server in found])
        if found:
            for server, failure in unchecked:
                click.echo(
                    f"{self.command_path}: warning: could not check server {server} for a conflicting tool "
                    f"({failure.format_message()})",
                    err=True,
                )
            return next(iter(found.values()))
        if unchecked:
            message = "\n".join(failure.format_message() for _, failure in unchecked)
            if any(isinstance(failure, UnresolvedServer) for _, failure in unchecked):
                raise ConfigFailure(message)
            if any(isinstance(failure, AuthFailure) for _, failure in unchecked):
                raise AuthFailure(message)
            raise TransportFailure(message)
        return None

    def check_group_name(self, group: str, server: str) -> None:
        """Fail when a top-level tool has the same name as a group mount.

        Judged from what is known without a request: loaded lists and caches, even expired ones.
        """
        for other, prefix in self.candidates(group):
            if other == server:
                continue  # a server's own clashes are found when its list is indexed
            tools = self.offline_tools(other, fresh_only=False)
            if tools is not None and tools.find(group[len(prefix) :]) is not None:
                raise _collision(group, [server, other])

    def top_level_listing(self) -> TopLevelListing:
        """Every top-level command, from the caches where possible.

        A server that cannot be listed is reported in the listing, not raised.
        """
        if self._listing is not None:
            return self._listing
        listing = TopLevelListing()
        owners: dict[str, str] = {}
        for group, server in self.group_mounts().items():
            mount = self.config.servers[server].mount
            assert isinstance(mount, GroupMount)
            owners[group] = server
            listing.groups[group] = mount.help or f"The tools of the {server} server."
        for name, (server, tool_group) in self.top_level_tool_groups().items():
            owners[name] = server
            listing.groups[name] = _tool_group_help(server, tool_group)
        for server, prefix in self.top_level_servers():
            try:
                tools = self.server_tools(server)
            except (TransportFailure, UnresolvedServer, AuthFailure) as error:
                listing.unavailable[server] = error.format_message()
                if isinstance(error, UnresolvedServer):
                    listing.unresolved.add(server)
                if isinstance(error, AuthFailure):
                    listing.unauthenticated.add(server)
                continue
            for command, entry in tools.entries.items():
                full = prefix + command
                if full in owners:
                    raise _collision(full, [owners[full], server])
                owners[full] = server
                listing.tools[full] = replace(entry, command=full)
        self._listing = listing
        return listing

    def forget_tools(self, server: str) -> None:
        """Drop a server's cache after it said a tool does not exist."""
        self.cache.drop(server)


def build_cli(
    bin_name: str,
    default_config: Mapping[str, Any] | CliConfig | None = None,
    *,
    distribution: str | None = None,
    result_classifier: ResultClassifier | None = None,
    retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
    extra_params: Sequence[click.Parameter] = (),
    extra_commands: Sequence[click.Command] = (),
    consumer_keys: Iterable[str] = (),
    config_layer: ConfigLayer | None = None,
) -> click.Group:
    """Build the click group of a kit CLI.

    Args:
        bin_name: The executable's name. It names the config and cache directories and the env prefix.
        default_config: The built-in configuration, in the shape of the config file.
        distribution: The consumer's distribution, for the version in the audit header. Without it,
            the distribution that installed the ``bin_name`` console script is used.
        result_classifier: Reclassifies successful results as errors; see ``ResultClassifier``.
        retry_delays: The pauses before each retry of a request that failed in transit.
        extra_params: Options of the consumer's own, added to the bin's global options, such as
            an account selector. Make them eager to have their values in ``ctx.params`` before
            the tools are listed. A name or option string the kit already uses is a ValueError.
        extra_commands: Commands of the consumer's own, listed and run at the top level next to the
            built-ins and the tools. They run without loading the config or the tool lists. A tool
            whose command would take one of their names gets a ``-tool`` suffix. A name of a
            built-in command, or one used twice, is a ValueError.
        consumer_keys: Top-level keys of the config file that the consumer reads itself, such as
            its accounts. The kit does not validate them, never reads them from the environment,
            and ``config`` lists them by name only. ``help`` or ``servers`` is a ValueError.
        config_layer: Computes a config layer from those keys; see ``ConfigLayer``.
    """
    _check_extra_commands(extra_commands)
    check_consumer_keys(set(consumer_keys))
    app = App(
        bin_name,
        default_config,
        distribution,
        result_classifier,
        retry_delays,
        tuple(extra_params),
        tuple(extra_commands),
        frozenset(consumer_keys),
        config_layer,
    )
    return KitGroup(app)


def _check_extra_commands(commands: Sequence[click.Command]) -> None:
    names: set[str] = set()
    for command in commands:
        if not command.name:
            raise ValueError("An extra command needs a name.")
        if command.name in BUILTIN_COMMANDS:
            raise ValueError(f"The extra command {command.name!r} has the name of a built-in command.")
        if command.name in names:
            raise ValueError(f"There are two extra commands named {command.name!r}.")
        names.add(command.name)


def main(
    bin_name: str,
    default_config: Mapping[str, Any] | CliConfig | None = None,
    *,
    distribution: str | None = None,
    result_classifier: ResultClassifier | None = None,
    extra_params: Sequence[click.Parameter] = (),
    extra_commands: Sequence[click.Command] = (),
    consumer_keys: Iterable[str] = (),
    config_layer: ConfigLayer | None = None,
    args: Sequence[str] | None = None,
) -> NoReturn:
    """Run a kit CLI; the entry point a consumer's console script calls. See ``build_cli`` for the arguments."""
    cli = build_cli(
        bin_name,
        default_config,
        distribution=distribution,
        result_classifier=result_classifier,
        extra_params=extra_params,
        extra_commands=extra_commands,
        consumer_keys=consumer_keys,
        config_layer=config_layer,
    )
    run(cli, prog_name=bin_name, args=args)


def _verbosity_callback(ctx: click.Context, param: click.Parameter, value: int) -> int:
    configure_logging(value)
    return value


def _version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    group = ctx.command
    assert isinstance(group, KitGroup)
    app = group.app
    click.echo(f"{app.bin_name} {app.bin_version} ({KIT_DISTRIBUTION} {distribution_version(KIT_DISTRIBUTION)})")
    ctx.exit(0)


def _kit_context(ctx: click.Context) -> tuple[click.Context, KitGroup]:
    """The context of the nearest ``KitGroup`` around a command: the root, or a host CLI's subcommand."""
    current: click.Context | None = ctx
    while current is not None:
        if isinstance(current.command, KitGroup):
            return current, current.command
        if isinstance(current.command, LazyKitGroup):
            # A context a host made for the stand-in itself, as click's to_info_dict() does.
            real = current.command.real()
            if isinstance(real, KitGroup):
                return current, real
        current = current.parent
    raise RuntimeError(f"{ctx.command_path} does not run inside a kit group")


def _runtime_owner(kit_ctx: click.Context) -> click.Context:
    """The context whose closing closes the runtime.

    Shell completion closes each context it resolves a subcommand with before it resolves the next
    one or completes an option, and keeps only the root context open throughout. A runtime made
    while it completes then closes with the root; otherwise with the kit group's own context.
    """
    return kit_ctx.find_root() if kit_ctx.resilient_parsing else kit_ctx


def _load(kit_ctx: click.Context, group: KitGroup) -> LoadedConfig:
    """Load the configuration of the kit group that runs in ``kit_ctx``, with the consumer's config layer."""
    app = group.app
    config_layer = app.config_layer

    def layer(consumer: Mapping[str, Any]) -> Mapping[str, Any]:
        assert config_layer is not None
        return config_layer(kit_ctx, consumer)

    try:
        return load_config(
            app.bin_name,
            default=app.default_config,
            config_path=kit_ctx.params.get("config"),
            command=kit_ctx.command_path,
            consumer_keys=app.consumer_keys,
            layer=layer if config_layer is not None else None,
        )
    except ConfigError as error:
        raise ConfigFailure(str(error)) from error


def _runtime(ctx: click.Context) -> Runtime:
    kit_ctx, group = _kit_context(ctx)
    # Click shares one meta dict across the whole context tree, a host CLI's contexts included.
    existing = kit_ctx.meta.get(_RUNTIME_KEY)
    if isinstance(existing, Runtime):
        return existing
    runtime = Runtime(
        group.app,
        _load(kit_ctx, group),
        refresh=bool(kit_ctx.params.get("refresh_tools")),
        command_path=kit_ctx.command_path,
    )
    kit_ctx.meta[_RUNTIME_KEY] = runtime

    def close() -> None:
        # A closed runtime must not serve a later lookup; see _CompletingGroup.
        if kit_ctx.meta.get(_RUNTIME_KEY) is runtime:
            del kit_ctx.meta[_RUNTIME_KEY]
        runtime.close()

    _runtime_owner(kit_ctx).call_on_close(close)
    runtime.check_extra_commands()
    return runtime


def _collision(name: str, servers: list[str]) -> ConfigFailure:
    return ConfigFailure(
        f"The command {name!r} is defined by more than one server: {', '.join(servers)}. "
        "Give one of them a group or prefix mount, or hide the tool with hidden_tools."
    )


def _tool_group_help(server: str, group: ToolGroup) -> str:
    return group.help or f"The {server} tools whose names start with {group.prefix}."


class _CompletingGroup(click.Group):
    """A group that may list tools while the shell completes its subcommands.

    Click completes a group's subcommands after it has closed every context, the root included,
    so a runtime made for the listing would never be closed, and its event loop thread would keep
    the process alive.
    """

    def shell_complete(self, ctx: click.Context, incomplete: str) -> list[CompletionItem]:
        try:
            return super().shell_complete(ctx, incomplete)
        finally:
            _runtime_owner(_kit_context(ctx)[0]).close()


class KitGroup(_CompletingGroup):
    """The bin's top-level command, whose subcommands are the tools."""

    def __init__(self, app: App) -> None:
        params: list[click.Parameter] = [
            click.Option(
                ["--config", "config"],
                metavar="PATH",
                is_eager=True,
                envvar=config_path_variable(app.bin_name),
                show_envvar=True,
                help=f"Read this config file instead of {_home_relative(default_config_path(app.bin_name))}.",
            ),
            click.Option(
                ["-v", "--verbose", "verbose"],
                count=True,
                is_eager=True,
                callback=_verbosity_callback,
                help="Log what the CLI does to stderr; -vv also logs the HTTP traffic.",
            ),
            click.Option(
                ["--refresh-tools", "refresh_tools"],
                is_flag=True,
                is_eager=True,
                help="Fetch the tool lists of the servers the command uses, ignoring the cache (kept for an hour).",
            ),
            *mode_options(),
            click.Option(
                ["--version"],
                is_flag=True,
                expose_value=False,
                is_eager=True,
                callback=_version_callback,
                help="Print the version and exit.",
            ),
        ]
        _check_extra_params(params, app.extra_params)
        params.extend(app.extra_params)
        super().__init__(name=app.bin_name, params=params, no_args_is_help=True)
        self.app = app

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        """Click's standalone handling, except that Ctrl-C exits 130 instead of 1; see ``lazy.run()``.

        Logging is set up here, not when the group is built, so building the CLI has no side
        effects on the process; the ``-v`` callback adjusts it afterwards.
        """
        configure_logging(0)
        if not standalone_mode:
            return super().main(args, prog_name, complete_var, standalone_mode=False, **extra)
        _standalone(
            self, lambda: super(KitGroup, self).main(args, prog_name, complete_var, standalone_mode=False, **extra)
        )

    def make_context(
        self, info_name: str | None, args: list[str], parent: click.Context | None = None, **extra: Any
    ) -> click.Context:
        """Set up logging when the group is used under a host CLI too, where ``main()`` never runs.

        The host's own commands keep the host's logging: only using this group changes it.
        """
        configure_logging(0)
        return super().make_context(info_name, args, parent, **extra)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Close the context when parsing fails, so the runtime it may already hold is closed too.

        ``--help`` lists the tools while the group's own arguments are parsed, before click
        enters the context, so click would never run the close callbacks of a failed listing;
        the open sessions and their event loop thread would then keep the process alive.
        """
        try:
            return super().parse_args(ctx, args)
        except BaseException:
            ctx.close()
            raise

    def get_help_option(self, ctx: click.Context) -> click.Option | None:
        # Not eager, so the eager --config and --refresh-tools are already known when the help lists the tools.
        option = super().get_help_option(ctx)
        if option is not None:
            option.is_eager = False
        return option

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        config = _runtime(ctx).config
        servers = "an MCP server" if len(config.servers) == 1 else "several MCP servers"
        self.help = config.help or (
            f"Call the tools of {servers}. Each tool is a subcommand; "
            f"run '{ctx.command_path} TOOL --help' for its options."
        )
        super().format_help_text(ctx, formatter)

    def list_commands(self, ctx: click.Context) -> list[str]:
        listing = self._listing(ctx)
        extras = [command.name for command in self.app.extra_commands if command.name]
        return sorted([*listing.tools, *listing.groups, *_top_level_builtins(_runtime(ctx)), *extras])

    def _listing(self, ctx: click.Context) -> TopLevelListing:
        runtime = _runtime(ctx)
        listing = runtime.top_level_listing()
        if listing.unavailable and not listing.tools and not listing.groups:
            # Nothing at all to show: the failures are the answer, as an error.
            message = "\n".join(listing.unavailable.values())
            if listing.unresolved:
                raise ConfigFailure(message)
            raise AuthFailure(message) if listing.unauthenticated else TransportFailure(message)
        return listing

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        extra = self.app.extra_command(cmd_name)
        if extra is not None:
            return extra
        if cmd_name == SchemaCommand.NAME:
            return SchemaCommand()
        if cmd_name == ConfigCommand.NAME:
            return ConfigCommand()
        if cmd_name == AuthGroup.NAME:
            return AuthGroup()
        runtime = _runtime(ctx)
        groups = runtime.group_mounts()
        if cmd_name in groups:
            # A group mount needs no request; the other servers are only checked from what is known.
            runtime.check_group_name(cmd_name, groups[cmd_name])
            return MountGroup(groups[cmd_name], cmd_name, runtime.config.servers[groups[cmd_name]].mount)
        tool_groups = runtime.top_level_tool_groups()
        if cmd_name in tool_groups:
            server, tool_group = tool_groups[cmd_name]
            runtime.check_group_name(cmd_name, server)
            return ToolGroupCommand(server, tool_group, cmd_name)
        entry = runtime.resolve_top_level(cmd_name)
        return ToolCommand(entry) if entry is not None else None

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        return _resolve_with_hint(self, ctx, args, ctx.command_path)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        listing = self._listing(ctx)
        if listing.groups:
            with formatter.section("Groups"):
                formatter.write_dl(sorted(listing.groups.items()))
        rows = [(name, _first_line(entry.tool.get("description"))) for name, entry in sorted(listing.tools.items())]
        if rows:
            with formatter.section("Tools"):
                formatter.write_dl(rows)
        extras = [
            (command.name, command.get_short_help_str(limit=formatter.width)) for command in self.app.extra_commands
        ]
        _write_builtins(formatter, top_level=True, auth=bool(_runtime(ctx).oauth_servers()), extra=extras)
        if listing.unavailable:
            with formatter.section("Unavailable servers"):
                formatter.write_dl(sorted(listing.unavailable.items()))


def _check_extra_params(own: Sequence[click.Parameter], extra: Sequence[click.Parameter]) -> None:
    """Refuse consumer options that would take the name or an option string of a kit option, or of each other."""
    names: set[str] = {"help", *(param.name for param in own if param.name)}
    flags: set[str] = {"--help", *(flag for param in own for flag in [*param.opts, *param.secondary_opts])}
    for param in extra:
        taken = [flag for flag in [*param.opts, *param.secondary_opts] if flag in flags]
        if param.name in names or taken:
            what = ", ".join(taken) if taken else repr(param.name)
            raise ValueError(f"The extra parameter {what} clashes with a global option of the kit or another extra.")
        if param.name:
            names.add(param.name)
        flags.update([*param.opts, *param.secondary_opts])


class MountGroup(_CompletingGroup):
    """The subcommand under which a group-mounted server's tools live. Only that server is contacted."""

    def __init__(self, server: str, name: str, mount: Any) -> None:
        help_text = getattr(mount, "help", None) or f"The tools of the {server} server."
        super().__init__(name=name, help=help_text, params=mode_options(), no_args_is_help=True)
        self.server = server

    def list_commands(self, ctx: click.Context) -> list[str]:
        tools = _runtime(ctx).server_tools(self.server)
        return sorted([*tools.entries, *tools.groups, SchemaCommand.NAME])

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name == SchemaCommand.NAME:
            return SchemaCommand()
        runtime = _runtime(ctx)
        for tool_group in runtime.config.servers[self.server].tool_groups:
            if tool_group.group == cmd_name:
                return ToolGroupCommand(self.server, tool_group, cmd_name)
        entry = runtime.find_in_server(self.server, cmd_name)
        return ToolCommand(entry) if entry is not None else None

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        return _resolve_with_hint(self, ctx, args, ctx.command_path)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        runtime = _runtime(ctx)
        tools = runtime.server_tools(self.server)
        groups = [
            (group.group, _tool_group_help(self.server, group))
            for group in runtime.config.servers[self.server].tool_groups
        ]
        if groups:
            with formatter.section("Groups"):
                formatter.write_dl(sorted(groups))
        _write_tools(formatter, tools.entries)
        _write_builtins(formatter)


class ToolGroupCommand(_CompletingGroup):
    """The tools of one server whose names start with a ``tool_groups`` prefix."""

    def __init__(self, server: str, tool_group: ToolGroup, name: str) -> None:
        super().__init__(
            name=name, help=_tool_group_help(server, tool_group), params=mode_options(), no_args_is_help=True
        )
        self.server = server
        self.tool_group = tool_group.group

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted([*_runtime(ctx).server_tools(self.server).groups[self.tool_group].entries, SchemaCommand.NAME])

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name == SchemaCommand.NAME:
            return SchemaCommand()
        entry = _runtime(ctx).find_in_server(self.server, cmd_name, group=self.tool_group)
        return ToolCommand(entry) if entry is not None else None

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        return _resolve_with_hint(self, ctx, args, ctx.command_path)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _write_tools(formatter, _runtime(ctx).server_tools(self.server).groups[self.tool_group].entries)
        _write_builtins(formatter)


class SchemaCommand(click.Command):
    """``schema TOOL``: print a tool's input schema, resolved like the tool command itself.

    The schema comes from the tool list, so with a warm cache nothing is sent. It is not server
    text, so machine mode prints it as compact JSON and pretty mode indents it.
    """

    NAME = "schema"
    HELP = "Print a tool's input schema as JSON."

    def __init__(self) -> None:
        super().__init__(
            name=self.NAME,
            help=f"{self.HELP} TOOL is any name the tool's own command accepts.",
            params=[click.Argument(["tool"], metavar="TOOL"), *mode_options()],
        )

    def invoke(self, ctx: click.Context) -> None:
        parent = ctx.parent
        assert parent is not None and isinstance(parent.command, click.Group)
        group = parent.command
        name = str(ctx.params["tool"])
        target = group.get_command(parent, name) if name != self.NAME else None
        if isinstance(target, click.Group):
            raise click.UsageError(
                f"{name!r} is a group, not a tool. Run '{parent.command_path} {name} schema TOOL'.", ctx
            )
        if not isinstance(target, ToolCommand):
            tools = [command for command in group.list_commands(parent) if command != self.NAME]
            close = difflib.get_close_matches(name, tools, n=3, cutoff=0.6)
            hint = (
                f" Did you mean {', '.join(close)}?" if close else f" Run '{parent.command_path} --help' for the list."
            )
            raise click.UsageError(f"No such tool {name!r}.{hint}", ctx)

        schema = target.entry.tool.get("inputSchema") or {}
        mode = resolve_mode(*context_flags(ctx))
        indent = 2 if mode == "pretty" else None
        _write(json.dumps(schema, ensure_ascii=False, indent=indent) + "\n", "stdout")


def _top_level_builtins(runtime: Runtime) -> list[str]:
    """The built-in commands at the top level; ``auth`` only when a server signs in with OAuth."""
    return [name for name in BUILTIN_COMMANDS if name != AuthGroup.NAME or runtime.oauth_servers()]


def _write_builtins(
    formatter: click.HelpFormatter,
    *,
    top_level: bool = False,
    auth: bool = False,
    extra: Sequence[tuple[str | None, str]] = (),
) -> None:
    rows = [(SchemaCommand.NAME, SchemaCommand.HELP)]
    rows += [(name, text) for name, text in extra if name is not None]
    if top_level:
        rows.append((ConfigCommand.NAME, ConfigCommand.HELP))
    if auth:
        rows.append((AuthGroup.NAME, AuthGroup.HELP))
    with formatter.section("Commands"):
        formatter.write_dl(sorted(rows))


def _write_tools(formatter: click.HelpFormatter, entries: Mapping[str, ToolEntry]) -> None:
    rows = [(name, _first_line(entries[name].tool.get("description"))) for name in sorted(entries)]
    if rows:
        with formatter.section("Tools"):
            formatter.write_dl(rows)


def _resolve_with_hint(
    group: click.Group, ctx: click.Context, args: list[str], help_path: str
) -> tuple[str | None, click.Command | None, list[str]]:
    name = click.utils.make_str(args[0])
    command = group.get_command(ctx, name)
    if command is None and not ctx.resilient_parsing:
        close = difflib.get_close_matches(name, group.list_commands(ctx), n=3, cutoff=0.6)
        hint = f" Did you mean {', '.join(close)}?" if close else f" Run '{help_path} --help' for the list."
        ctx.fail(f"No such command {name!r}.{hint}")
    return command.name if command is not None else None, command, args[1:]


class ConfigCommand(click.Command):
    """``config``: print the effective configuration and where it came from, with secrets masked.

    It reads the configuration only: no tool list, no cache and no request, so it also works
    offline and with unset ``${NAME}`` variables, which it shows unexpanded.
    """

    NAME = "config"
    HELP = "Print the effective configuration as JSON, with secrets masked."

    def __init__(self) -> None:
        super().__init__(
            name=self.NAME,
            help=(
                f"{self.HELP} Also shows the config file, whether it was loaded, the environment "
                "variable prefix and the names of the environment variables that were applied. "
                "${NAME} references are shown unexpanded. Nothing is sent to any server."
            ),
            params=[*mode_options()],
        )

    def invoke(self, ctx: click.Context) -> None:
        loaded = _load(*_kit_context(ctx))
        mode = resolve_mode(*context_flags(ctx))
        indent = 2 if mode == "pretty" else None
        _write(json.dumps(config_view.describe(loaded), ensure_ascii=False, indent=indent) + "\n", "stdout")


class AuthGroup(click.Group):
    """``auth``: sign in to the servers that use OAuth, sign out, and show who is signed in.

    Signing in is only ever done here. A tool command never opens a browser: without a usable
    sign-in it exits with ``EXIT_AUTH``, so an AI agent gets an answer at once instead of a
    command that waits for a person.
    """

    NAME = "auth"
    HELP = "Sign in to the servers that use OAuth, sign out, or show the sign-in state."

    def __init__(self) -> None:
        from mcp_to_cli_kit.oauth import DEFAULT_LOGIN_TIMEOUT_SECONDS

        super().__init__(name=self.NAME, help=self.HELP, no_args_is_help=True)
        server = click.Argument(["server"], required=False, metavar="[SERVER]")
        self.add_command(
            click.Command(
                "login",
                callback=_login,
                help=(
                    "Sign in to a server through the browser and store its tokens. Without SERVER, sign in to every "
                    "server that uses OAuth. The sign-in address is also printed, to open by hand. Commands refresh "
                    "the tokens by themselves afterwards, until the authorization server ends the sign-in."
                ),
                params=[
                    server,
                    click.Option(
                        ["--no-browser"],
                        is_flag=True,
                        help="Only print the sign-in address; do not open a browser.",
                    ),
                    click.Option(
                        ["--timeout"],
                        type=click.FloatRange(min=1),
                        default=DEFAULT_LOGIN_TIMEOUT_SECONDS,
                        show_default=True,
                        help="Seconds to wait for the sign-in in the browser.",
                    ),
                ],
            )
        )
        self.add_command(
            click.Command(
                "logout",
                callback=_logout,
                help=(
                    "Delete the stored tokens of a server, or of every server that uses OAuth. With --revoke, also "
                    "revoke the sign-in at the authorization server. Google revokes it for every app that uses the "
                    "same OAuth client, so other tools signed in with that client are signed out too."
                ),
                params=[
                    click.Argument(["server"], required=False, metavar="[SERVER]"),
                    click.Option(
                        ["--revoke"], is_flag=True, help="Also revoke the sign-in at the authorization server."
                    ),
                ],
            )
        )
        self.add_command(
            click.Command(
                "status",
                callback=_status,
                help=(
                    "Print the sign-in state of each server that uses OAuth as JSON, without the tokens: who is "
                    "signed in, since when, and until when. signed_in_until is null when the authorization server "
                    "did not say how long its refresh token lasts. Prints one line, or indented JSON with --full, "
                    "unless --json or --pretty says otherwise. Nothing is sent to any server. Exits 4 when a server "
                    "is not signed in."
                ),
                params=[
                    click.Argument(["server"], required=False, metavar="[SERVER]"),
                    click.Option(
                        ["--full"],
                        is_flag=True,
                        help="Also print when each token expires, the scopes, the client, the issuer, the token file.",
                    ),
                    *mode_options(),
                ],
            )
        )


def _oauth_targets(server: str | None) -> tuple[Runtime, list[str]]:
    ctx = click.get_current_context()
    runtime = _runtime(ctx)
    names = runtime.oauth_servers()
    if server is None:
        if not names:
            raise click.UsageError("No configured server signs in with OAuth.", ctx)
        return runtime, names
    if server not in runtime.config.servers:
        raise click.UsageError(f"No server named {server!r}. Servers: {', '.join(runtime.config.servers)}.", ctx)
    if server not in names:
        raise click.UsageError(f"The server {server!r} does not sign in with OAuth (it has no auth key).", ctx)
    return runtime, [server]


def _resolved(runtime: Runtime, name: str) -> ResolvedServer:
    try:
        resolved = runtime.client(name).resolve()
    except ConfigError as error:
        raise ConfigFailure(str(error)) from error
    assert resolved.auth is not None
    return resolved


def _login(server: str | None, no_browser: bool, timeout: float) -> None:
    from mcp_to_cli_kit import oauth
    from mcp_to_cli_kit.oauth import AuthError, AuthTransportError

    runtime, names = _oauth_targets(server)
    for name in names:
        resolved = _resolved(runtime, name)
        assert resolved.auth is not None
        try:
            tokens = oauth.login(
                name,
                resolved.url,
                resolved.auth,
                runtime.tokens,
                echo=lambda line: click.echo(line, err=True),
                browser=not no_browser,
                wait_timeout=timeout,
                http_timeout=runtime.config.servers[name].timeout,
            )
        except AuthError as error:
            raise AuthFailure(f"signing in to {name} failed: {error}") from error
        except AuthTransportError as error:
            raise TransportFailure(f"signing in to {name} failed: {error}") from error
        who = f" as {tokens.identity}" if tokens.identity else ""
        click.echo(f"Signed in to {name}{who}.", err=True)


def _logout(server: str | None, revoke: bool) -> None:
    from mcp_to_cli_kit import oauth
    from mcp_to_cli_kit.oauth import AuthError, AuthTransportError

    runtime, names = _oauth_targets(server)
    for name in names:
        tokens = runtime.tokens.load(name)
        if tokens is None:
            click.echo(f"Not signed in to {name}.", err=True)
            continue
        if revoke:
            resolved = _resolved(runtime, name)
            assert resolved.auth is not None
            try:
                if not oauth.revoke(tokens, resolved.auth, runtime.config.servers[name].timeout):
                    click.echo(f"The authorization server of {name} offers no revocation; only signing out.", err=True)
            except AuthError as error:
                raise AuthFailure(f"revoking the sign-in to {name} failed: {error}") from error
            except AuthTransportError as error:
                raise TransportFailure(f"revoking the sign-in to {name} failed: {error}") from error
        runtime.tokens.delete(name)
        click.echo(f"Signed out of {name}.", err=True)


def _status(server: str | None, full: bool, machine_output: bool, pretty_output: bool) -> None:
    ctx = click.get_current_context()
    runtime, names = _oauth_targets(server)
    servers = []
    for name in names:
        resolved = _resolved(runtime, name)
        assert resolved.auth is not None
        kind = resolved.auth.config.bearer
        stored = runtime.tokens.load(name)
        tokens = stored if stored is not None and stored.is_for(resolved.url, resolved.auth) else None
        entry: dict[str, Any] = {
            "server": name,
            "signed_in": tokens is not None and tokens.is_signed_in(kind),
            "identity": tokens.identity if tokens is not None else None,
            "signed_in_at": _iso_time(tokens.signed_in_at) if tokens is not None else None,
            "signed_in_until": _iso_time(tokens.signed_in_until(kind)) if tokens is not None else None,
        }
        if full:
            entry |= {
                "bearer": kind,
                "access_token_expires_at": _iso_time(tokens.expires_at) if tokens is not None else None,
                "id_token_expires_at": _iso_time(tokens.id_token_expires_at) if tokens is not None else None,
                "has_refresh_token": tokens is not None and tokens.refresh_token is not None,
                "refresh_token_expires_at": _iso_time(tokens.refresh_expires_at) if tokens is not None else None,
                "obtained_at": _iso_time(tokens.obtained_at) if tokens is not None else None,
                "scopes": tokens.scope.split() if tokens is not None and tokens.scope else [],
                "client_id": resolved.auth.client_id,
                "issuer": tokens.issuer if tokens is not None else resolved.auth.config.authorization_server,
                "resource": tokens.resource if tokens is not None else None,
                "token_file": str(runtime.tokens.path(name)),
            }
        servers.append(entry)
    # The short status fits one line for a person too; the full one is long enough to read only indented.
    mode = resolve_mode(*context_flags(ctx), "pretty" if full else "machine")
    _write(json.dumps({"servers": servers}, indent=2 if mode == "pretty" else None) + "\n", "stdout")
    if not all(entry["signed_in"] for entry in servers):
        ctx.exit(EXIT_AUTH)


def _iso_time(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="seconds") if epoch is not None else None


class ToolCommand(click.Command):
    """One tool as a command; its options are compiled from the tool's input schema."""

    def __init__(self, entry: ToolEntry) -> None:
        self.entry = entry
        self.compiled: CompiledTool = compile_tool(entry.tool.get("inputSchema"))
        description = entry.tool.get("description")
        help_text = _keep_paragraphs(description if isinstance(description, str) else "")
        if entry.tool.get("outputSchema"):
            help_text += "\n\nReturns JSON: the tool declares an output schema."
        super().__init__(
            name=entry.command,
            params=[*self.compiled.options(), *mode_options()],
            help=help_text or None,
            no_args_is_help=False,
        )

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        try:
            return super().parse_args(ctx, args)
        except click.UsageError as error:
            raise self._with_summary(error, ctx) from error

    def _with_summary(self, error: click.UsageError, ctx: click.Context) -> click.UsageError:
        # format_message() includes click's "Did you mean" hint, which belongs before the list.
        return click.UsageError(f"{error.format_message()}\n\n{self.compiled.summary()}", error.ctx or ctx)

    def invoke(self, ctx: click.Context) -> None:
        mode = resolve_mode(*context_flags(ctx))
        try:
            arguments = self.compiled.arguments(ctx.params)
        except click.UsageError as error:
            raise self._with_summary(error, ctx) from error
        problems = self.compiled.validate(arguments)
        if problems:
            invalid = click.UsageError("invalid arguments:\n  " + "\n  ".join(problems), ctx)
            raise self._with_summary(invalid, ctx)

        from mcp_to_cli_kit.client import ServerRejectedError, TransportError
        from mcp_to_cli_kit.oauth import AuthError

        runtime = _runtime(ctx)
        client = runtime.client(self.entry.server)
        logger.debug("Calling %s on %s with %s", self.entry.name, self.entry.server, arguments)
        try:
            result = client.call_tool(self.entry.name, arguments)
        except ConfigError as error:
            raise ConfigFailure(str(error)) from error
        except AuthError as error:
            raise AuthFailure(str(error)) from error
        except TransportError as error:
            raise TransportFailure(str(error)) from error
        except ServerRejectedError as error:
            if error.code in _UNKNOWN_TOOL_CODES and _UNKNOWN_TOOL.search(str(error)):
                runtime.forget_tools(self.entry.server)
            raise ToolFailure(f"the server rejected the call: {error}") from error

        is_error = result.isError
        classifier = runtime.app.result_classifier
        if not is_error and classifier is not None:
            is_error = bool(classifier(self.entry.server, self.entry.name, result))

        text = render(result, mode)
        if is_error:
            if _UNKNOWN_TOOL.search(_first_text(result)):
                runtime.forget_tools(self.entry.server)
            _write(finish(text, mode, _is_tty("stderr")) or "the tool reported an error without a message\n", "stderr")
            ctx.exit(EXIT_TOOL_ERROR)
        _write(finish(text, mode, _is_tty("stdout")), "stdout")


def _first_text(result: types.CallToolResult) -> str:
    from mcp import types

    for item in result.content:
        if isinstance(item, types.TextContent):
            return item.text
    return ""


def _is_tty(name: str) -> bool:
    stream = sys.stdout if name == "stdout" else sys.stderr
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _write(text: str, name: str) -> None:
    if not text:
        return
    stream = sys.stdout if name == "stdout" else sys.stderr
    stream.flush()
    binary = getattr(stream, "buffer", None)
    if binary is None:
        stream.write(text)
    else:
        binary.write(text.encode("utf-8"))
        binary.flush()


def _home_relative(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _first_line(description: Any) -> str:
    if not isinstance(description, str):
        return ""
    for line in description.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def _keep_paragraphs(text: str) -> str:
    """Mark every paragraph with click's no-rewrap marker, so lists and examples keep their lines."""
    text = inspect.cleandoc(text.replace("\f", ""))
    if not text:
        return ""
    paragraphs = re.split(r"\n[ \t]*\n", text)
    return "\n\n".join(f"\b\n{paragraph}" for paragraph in paragraphs)
