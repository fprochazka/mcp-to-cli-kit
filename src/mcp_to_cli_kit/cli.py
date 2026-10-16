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
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import click
from mcp import types

from mcp_to_cli_kit.arguments import CompiledTool, compile_tool, kebab_case
from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir
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
from mcp_to_cli_kit.config import CliConfig, ConfigError, LoadedConfig, default_config_path, load_config
from mcp_to_cli_kit.output import OutputMode, finish, render, resolve_mode

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
    _entries: dict[str, ToolEntry] | None = None
    _aliases: dict[str, str] = field(default_factory=dict)

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

    def tools(self, server: str) -> list[dict[str, Any]]:
        """The tools of a server: from a fresh cache, else from the server, else from an expired cache."""
        if server in self._tools:
            return self._tools[server]
        client = self.clients[server]
        cached = self.cache.load(server, client.server.url)
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
        except (TransportError, ServerRejectedError) as error:
            if fallback is None:
                raise TransportFailure(f"cannot list the tools of {client.display_url}: {error}") from error
            click.echo(
                f"{self.app.bin_name}: warning: cannot refresh the tool list ({error}); using the expired cached list.",
                err=True,
            )
            self._tools[server] = fallback
            return fallback
        self.cache.save(server, client.server.url, tools)
        self._tools[server] = tools
        return tools

    def entries(self) -> dict[str, ToolEntry]:
        """Every tool command, by command name."""
        if self._entries is None:
            self._entries = {}
            self._aliases = {}
            for server in self.loaded.config.servers:
                self._add_entries(server)
        return self._entries

    def _add_entries(self, server: str) -> None:
        assert self._entries is not None
        hidden = set(self.loaded.config.servers[server].hidden_tools)
        for tool in self.tools(server):
            name = tool.get("name")
            if not isinstance(name, str) or not name or name in hidden:
                continue
            command = kebab_case(name)
            if command in self._entries:
                command = name
            if command in self._entries:
                logger.warning("Skipping the tool %r of %s: its command name is taken", name, server)
                continue
            self._entries[command] = ToolEntry(server, command, tool)
            for alias in (name, name.replace("-", "_")):
                self._aliases.setdefault(alias, command)

    def find(self, name: str) -> ToolEntry | None:
        """Find a tool by its command name, its own name or its snake case. A miss refreshes a cached list once."""
        entry = self._lookup(name)
        if entry is not None:
            return entry
        stale = [server for server in self.loaded.config.servers if server not in self._live]
        if not stale:
            return None
        logger.debug("No command %r in the cached tool lists; fetching them again", name)
        for server in stale:
            self._tools.pop(server, None)
            cached = self.cache.load(server, self.clients[server].server.url)
            self._fetch(server, fallback=cached.tools if cached is not None else None)
        self._entries = None
        return self._lookup(name)

    def _lookup(self, name: str) -> ToolEntry | None:
        entries = self.entries()
        if name in entries:
            return entries[name]
        alias = self._aliases.get(name) or self._aliases.get(name.replace("-", "_"))
        return entries.get(alias) if alias is not None else None

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
    unsupported = _unsupported(loaded.config)
    if unsupported:
        raise ConfigFailure(unsupported)
    runtime = Runtime(group.app, loaded, refresh=bool(root.params.get("refresh_tools")))
    root.meta[_RUNTIME_KEY] = runtime
    root.call_on_close(runtime.close)
    return runtime


def _unsupported(config: CliConfig) -> str | None:
    if len(config.servers) > 1:
        return "Several servers in one bin are not supported yet."
    for name, server in config.servers.items():
        if server.mount != "root":
            return f"Server {name}: only the root mount is supported yet."
        if server.tool_groups:
            return f"Server {name}: tool_groups are not supported yet."
    return None


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
                help="Fetch the tool list from the server instead of the cache (kept for an hour).",
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

    def get_help_option(self, ctx: click.Context) -> click.Option | None:
        # Not eager, so the eager --config and --refresh-tools are already known when the help lists the tools.
        option = super().get_help_option(ctx)
        if option is not None:
            option.is_eager = False
        return option

    def format_help_text(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        configured = _runtime(ctx).loaded.config.help
        self.help = configured or (
            "Call the tools of an MCP server. Each tool is a subcommand; "
            f"run '{self.app.bin_name} TOOL --help' for its options."
        )
        super().format_help_text(ctx, formatter)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_runtime(ctx).entries())

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        entry = _runtime(ctx).find(cmd_name)
        if entry is None:
            return None
        return ToolCommand(entry)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        name = click.utils.make_str(args[0])
        command = self.get_command(ctx, name)
        if command is None and not ctx.resilient_parsing:
            close = difflib.get_close_matches(name, self.list_commands(ctx), n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(close)}?" if close else f" Run '{self.app.bin_name} --help' for the list."
            ctx.fail(f"No such command {name!r}.{hint}")
        return command.name if command is not None else None, command, args[1:]

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        entries = _runtime(ctx).entries()
        rows = [(name, _first_line(entries[name].tool.get("description"))) for name in sorted(entries)]
        if rows:
            with formatter.section("Tools"):
                formatter.write_dl(rows)


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
        mode = resolve_mode(_mode(ctx), _mode(ctx.find_root()))
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
