"""The click application: every tool of the configured MCP server becomes a subcommand.

The group is lazy. Nothing is loaded when the process starts; the config is read, and the
tool list taken from the cache or the server, only when click asks for the subcommands, which
happens after the group's own options are parsed. ``--config``, ``-v`` and ``--refresh-tools``
are eager and ``--help`` is not, so ``--help --config other.yaml`` lists the tools of the
other config.

Exit codes: 0 on success, 1 when the tool reports an error, 2 for a usage or config error,
3 when the server cannot be reached or does not speak MCP.
"""

from __future__ import annotations

import difflib
import inspect
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NoReturn

import click
from mcp import types

from mcp_to_cli_kit import config_view, redaction
from mcp_to_cli_kit.arguments import CompiledTool, compile_tool, kebab_case
from mcp_to_cli_kit.cache import CachedTools, SchemaCache, default_cache_dir
from mcp_to_cli_kit.client import (
    DEFAULT_RETRY_DELAYS,
    KIT_DISTRIBUTION,
    PortalProvider,
    ServerClient,
    ServerRejectedError,
    TransportError,
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
    ToolGroup,
    default_config_path,
    load_config,
)
from mcp_to_cli_kit.output import OutputMode, finish, render, resolve_mode
from mcp_to_cli_kit.redaction import RedactingFilter

logger = logging.getLogger(__name__)

ResultClassifier = Callable[[str, str, types.CallToolResult], bool]
"""Called with the server name, the tool name and a result the server did not mark as an error.

Returning True treats the result as an error: it is printed to stderr and the exit code is 1.
For servers that report failures as ordinary text.
"""

EXIT_TOOL_ERROR = 1
EXIT_USAGE = 2
EXIT_TRANSPORT = 3
EXIT_INTERRUPTED = 130

_RUNTIME_KEY = "mcp_to_cli_kit.runtime"

# Only at the start of an error, so a tool whose ordinary output mentions such a phrase keeps the cache.
_UNKNOWN_TOOL = re.compile(
    r"^\s*(?:error[^:\n]{0,60}:\s*)?(?:unknown tool|no such tool|tool not found)\b", re.IGNORECASE
)

# JSON-RPC "method not found" and "invalid params", the codes servers use for an unknown tool.
_UNKNOWN_TOOL_CODES = frozenset({-32601, -32602})

_SILENT = logging.CRITICAL + 10

_LIBRARY_LOGGERS = ("mcp", "httpx", "httpcore")

_KIT_HANDLER = "_mcp_to_cli_kit_handler"


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


@dataclass(frozen=True)
class App:
    """What a consumer hands the kit: everything that does not come from the config."""

    bin_name: str
    default_config: Mapping[str, Any] | CliConfig | None = None
    distribution: str | None = None
    result_classifier: ResultClassifier | None = None
    retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS

    @property
    def bin_version(self) -> str:
        return distribution_version(self.distribution, self.bin_name)


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

    def add(self, server: str, local_name: str, tool: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
        """Add a tool under the kebab case of ``local_name``; returns the command name, None when it is taken.

        A tool whose command would be a built-in command (``schema`` or ``config``) gets a ``-tool``
        suffix, at every level.
        """
        command = kebab_case(local_name)
        if command in BUILTIN_COMMANDS:
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
    ) -> ServerTools:
        index = cls(groups={group.group: ToolIndex() for group in tool_groups})
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


@dataclass
class Runtime:
    """The state of one invocation: the config, the clients and the tool lists."""

    app: App
    loaded: LoadedConfig
    refresh: bool
    portal: PortalProvider = field(default_factory=PortalProvider)
    clients: dict[str, ServerClient] = field(default_factory=dict)
    cache: SchemaCache = field(init=False)
    _tools: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _live: set[str] = field(default_factory=set)
    _indexes: dict[str, ServerTools] = field(default_factory=dict)
    _listing: TopLevelListing | None = None

    def __post_init__(self) -> None:
        self.cache = SchemaCache(default_cache_dir(self.app.bin_name))
        version = self.app.bin_version
        header = client_header_value(self.app.bin_name, version)
        client_info = types.Implementation(name=self.app.bin_name, version=version)
        for name, server in self.loaded.config.servers.items():
            self.clients[name] = ServerClient(
                name, server, header, self.portal, self.app.retry_delays, client_info=client_info
            )

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        self.portal.__exit__(None, None, None)
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
        client = self.clients[server]
        self._live.add(server)
        try:
            tools = client.list_tools()
        except (TransportError, ServerRejectedError, ConfigError) as error:
            if fallback is None:
                if isinstance(error, ConfigError):
                    raise UnresolvedServer(str(error)) from error
                raise TransportFailure(f"cannot list the tools of {client.display_url} ({server}): {error}") from error
            click.echo(
                f"{self.app.bin_name}: warning: cannot refresh the tool list of {server} ({error}); "
                "using the expired cached list.",
                err=True,
            )
            self._tools[server] = fallback
            return fallback
        self.cache.save(server, client.server.url, tools, client.cache_url())
        self._tools[server] = tools
        return tools

    def _load_cache(self, server: str) -> CachedTools | None:
        client = self.clients[server]
        return self.cache.load(server, client.server.url, client.cache_url())

    def server_tools(self, server: str) -> ServerTools:
        """A server's tools by their command name within the server, without any mount prefix."""
        if server not in self._indexes:
            self._indexes[server] = self._index(server, self.tools(server))
        return self._indexes[server]

    def _index(self, server: str, tools: list[dict[str, Any]]) -> ServerTools:
        config = self.config.servers[server]
        index = ServerTools.build(server, tools, config.hidden_tools, config.tool_groups, config.strip_prefixes)
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
        unchecked: list[tuple[str, TransportFailure | UnresolvedServer]] = []
        for server, prefix in to_ask:
            if server in found:
                continue
            try:
                entry = self.find_in_server(server, name[len(prefix) :])
            except (TransportFailure, UnresolvedServer) as error:
                unchecked.append((server, error))
                continue
            if entry is not None:
                found[server] = replace(entry, command=prefix + entry.command)

        if len(found) > 1:
            raise _collision(name, [server for server in self.config.servers if server in found])
        if found:
            for server, failure in unchecked:
                click.echo(
                    f"{self.app.bin_name}: warning: could not check server {server} for a conflicting tool "
                    f"({failure.format_message()})",
                    err=True,
                )
            return next(iter(found.values()))
        if unchecked:
            message = "\n".join(failure.format_message() for _, failure in unchecked)
            if any(isinstance(failure, UnresolvedServer) for _, failure in unchecked):
                raise ConfigFailure(message)
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
            except (TransportFailure, UnresolvedServer) as error:
                listing.unavailable[server] = error.format_message()
                if isinstance(error, UnresolvedServer):
                    listing.unresolved.add(server)
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
) -> click.Group:
    """Build the click group of a kit CLI.

    Args:
        bin_name: The executable's name. It names the config and cache directories and the env prefix.
        default_config: The built-in configuration, in the shape of the config file.
        distribution: The consumer's distribution, for the version in the audit header. Without it,
            the distribution that installed the ``bin_name`` console script is used.
        result_classifier: Reclassifies successful results as errors; see ``ResultClassifier``.
        retry_delays: The pauses before each retry of a request that failed in transit.
    """
    app = App(bin_name, default_config, distribution, result_classifier, retry_delays)
    return KitGroup(app)


def main(
    bin_name: str,
    default_config: Mapping[str, Any] | CliConfig | None = None,
    *,
    distribution: str | None = None,
    result_classifier: ResultClassifier | None = None,
    args: Sequence[str] | None = None,
) -> NoReturn:
    """Run a kit CLI; the entry point a consumer's console script calls."""
    cli = build_cli(bin_name, default_config, distribution=distribution, result_classifier=result_classifier)
    cli.main(args=list(args) if args is not None else None, prog_name=bin_name)
    raise SystemExit(0)  # pragma: no cover - click's standalone mode always exits


def configure_logging(verbosity: int) -> None:
    """``-v`` shows the kit's debug messages, ``-vv`` also the MCP SDK's and httpx's. All on stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    # Expanded ${NAME} values and the URL's userinfo and query never reach the log, -vv included.
    handler.addFilter(RedactingFilter())

    kit = logging.getLogger("mcp_to_cli_kit")
    kit.handlers = [handler]
    kit.propagate = False
    kit.setLevel(logging.DEBUG if verbosity >= 1 else logging.WARNING)

    for name in _LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        library.handlers = [handler] if verbosity >= 1 else [logging.NullHandler()]
        library.propagate = False
        library.setLevel(logging.DEBUG if verbosity >= 2 else logging.WARNING if verbosity == 1 else _SILENT)

    # The SDK also logs through the root logger ("Unhandled exception in receive loop"). Without
    # a handler there, Python's last-resort handler would print it with a traceback to stderr.
    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _KIT_HANDLER, False)]:
        root.removeHandler(existing)
    root_handler: logging.Handler = logging.StreamHandler(sys.stderr) if verbosity >= 1 else logging.NullHandler()
    root_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_handler.addFilter(RedactingFilter())
    root_handler.setLevel(logging.WARNING)
    setattr(root_handler, _KIT_HANDLER, True)
    root.addHandler(root_handler)


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


def _mode_options() -> list[click.Option]:
    return [
        click.Option(
            ["--json", "machine_output"],
            is_flag=True,
            default=False,
            help="Machine output: print the server's result exactly as sent, as under an AI agent.",
        ),
        click.Option(
            ["--pretty", "pretty_output"],
            is_flag=True,
            default=False,
            help="Pretty output: indent JSON results. The default unless an AI agent is detected.",
        ),
    ]


def _mode(ctx: click.Context) -> OutputMode | None:
    machine = bool(ctx.params.get("machine_output"))
    pretty = bool(ctx.params.get("pretty_output"))
    if machine and pretty:
        raise click.UsageError("--json and --pretty exclude each other.", ctx)
    return "machine" if machine else "pretty" if pretty else None


def _modes(ctx: click.Context) -> list[OutputMode | None]:
    """The output-mode flags from the tool's own options outwards, the innermost first."""
    modes = []
    current: click.Context | None = ctx
    while current is not None:
        modes.append(_mode(current))
        current = current.parent
    return modes


def _runtime(ctx: click.Context) -> Runtime:
    root = ctx.find_root()
    runtime = root.meta.get(_RUNTIME_KEY)
    if isinstance(runtime, Runtime):
        return runtime
    group = root.command
    assert isinstance(group, KitGroup)
    try:
        loaded = load_config(
            group.app.bin_name, default=group.app.default_config, config_path=root.params.get("config")
        )
    except ConfigError as error:
        raise ConfigFailure(str(error)) from error
    runtime = Runtime(group.app, loaded, refresh=bool(root.params.get("refresh_tools")))
    root.meta[_RUNTIME_KEY] = runtime
    root.call_on_close(runtime.close)
    return runtime


def _collision(name: str, servers: list[str]) -> ConfigFailure:
    return ConfigFailure(
        f"The command {name!r} is defined by more than one server: {', '.join(servers)}. "
        "Give one of them a group or prefix mount, or hide the tool with hidden_tools."
    )


def _tool_group_help(server: str, group: ToolGroup) -> str:
    return group.help or f"The {server} tools whose names start with {group.prefix}."


class KitGroup(click.Group):
    """The bin's top-level command, whose subcommands are the tools."""

    def __init__(self, app: App) -> None:
        params: list[click.Parameter] = [
            click.Option(
                ["--config", "config"],
                metavar="PATH",
                is_eager=True,
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
            *_mode_options(),
            click.Option(
                ["--version"],
                is_flag=True,
                expose_value=False,
                is_eager=True,
                callback=_version_callback,
                help="Print the version and exit.",
            ),
        ]
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
        """Click's standalone handling, except that Ctrl-C exits 130 instead of 1.

        Logging is set up here, not when the group is built, so building the CLI has no side
        effects on the process; the ``-v`` callback adjusts it afterwards.
        """
        configure_logging(0)
        if not standalone_mode:
            return super().main(args, prog_name, complete_var, standalone_mode=False, **extra)
        try:
            # Click itself still handles a closed pipe (EPIPE) in this mode.
            result = super().main(args, prog_name, complete_var, standalone_mode=False, **extra)
        except click.ClickException as error:
            error.show()
            sys.exit(error.exit_code)
        except (click.Abort, KeyboardInterrupt) as error:
            if isinstance(error.__cause__, EOFError):
                # End of input at a prompt, which click treats as an abort with exit 1.
                click.echo("Aborted!", err=True)
                sys.exit(1)
            click.echo("Interrupted.", err=True)
            sys.exit(EXIT_INTERRUPTED)
        sys.exit(result if isinstance(result, int) else 0)

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
            f"run '{self.app.bin_name} TOOL --help' for its options."
        )
        super().format_help_text(ctx, formatter)

    def list_commands(self, ctx: click.Context) -> list[str]:
        listing = self._listing(ctx)
        return sorted([*listing.tools, *listing.groups, *BUILTIN_COMMANDS])  # schema and config

    def _listing(self, ctx: click.Context) -> TopLevelListing:
        runtime = _runtime(ctx)
        listing = runtime.top_level_listing()
        if listing.unavailable and not listing.tools and not listing.groups:
            # Nothing at all to show: the failures are the answer, as an error.
            message = "\n".join(listing.unavailable.values())
            raise ConfigFailure(message) if listing.unresolved else TransportFailure(message)
        return listing

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name == SchemaCommand.NAME:
            return SchemaCommand()
        if cmd_name == ConfigCommand.NAME:
            return ConfigCommand()
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
        return _resolve_with_hint(self, ctx, args, self.app.bin_name)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        listing = self._listing(ctx)
        if listing.groups:
            with formatter.section("Groups"):
                formatter.write_dl(sorted(listing.groups.items()))
        rows = [(name, _first_line(entry.tool.get("description"))) for name, entry in sorted(listing.tools.items())]
        if rows:
            with formatter.section("Tools"):
                formatter.write_dl(rows)
        _write_builtins(formatter, top_level=True)
        if listing.unavailable:
            with formatter.section("Unavailable servers"):
                formatter.write_dl(sorted(listing.unavailable.items()))


class MountGroup(click.Group):
    """The subcommand under which a group-mounted server's tools live. Only that server is contacted."""

    def __init__(self, server: str, name: str, mount: Any) -> None:
        help_text = getattr(mount, "help", None) or f"The tools of the {server} server."
        super().__init__(name=name, help=help_text, params=_mode_options(), no_args_is_help=True)
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


class ToolGroupCommand(click.Group):
    """The tools of one server whose names start with a ``tool_groups`` prefix."""

    def __init__(self, server: str, tool_group: ToolGroup, name: str) -> None:
        super().__init__(
            name=name, help=_tool_group_help(server, tool_group), params=_mode_options(), no_args_is_help=True
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
            params=[click.Argument(["tool"], metavar="TOOL"), *_mode_options()],
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
        mode = resolve_mode(*_modes(ctx))
        indent = 2 if mode == "pretty" else None
        _write(json.dumps(schema, ensure_ascii=False, indent=indent) + "\n", "stdout")


def _write_builtins(formatter: click.HelpFormatter, *, top_level: bool = False) -> None:
    rows = [(SchemaCommand.NAME, SchemaCommand.HELP)]
    if top_level:
        rows.append((ConfigCommand.NAME, ConfigCommand.HELP))
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
            params=[*_mode_options()],
        )

    def invoke(self, ctx: click.Context) -> None:
        root = ctx.find_root()
        group = root.command
        assert isinstance(group, KitGroup)
        try:
            loaded = load_config(
                group.app.bin_name, default=group.app.default_config, config_path=root.params.get("config")
            )
        except ConfigError as error:
            raise ConfigFailure(str(error)) from error
        mode = resolve_mode(*_modes(ctx))
        indent = 2 if mode == "pretty" else None
        _write(json.dumps(config_view.describe(loaded), ensure_ascii=False, indent=indent) + "\n", "stdout")


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
            params=[*self.compiled.options(), *_mode_options()],
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
        mode = resolve_mode(*_modes(ctx))
        try:
            arguments = self.compiled.arguments(ctx.params)
        except click.UsageError as error:
            raise self._with_summary(error, ctx) from error
        problems = self.compiled.validate(arguments)
        if problems:
            invalid = click.UsageError("invalid arguments:\n  " + "\n  ".join(problems), ctx)
            raise self._with_summary(invalid, ctx)

        runtime = _runtime(ctx)
        client = runtime.clients[self.entry.server]
        logger.debug("Calling %s on %s with %s", self.entry.name, self.entry.server, arguments)
        try:
            result = client.call_tool(self.entry.name, arguments)
        except ConfigError as error:
            raise ConfigFailure(str(error)) from error
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
