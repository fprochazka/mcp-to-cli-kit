"""Helpers for a host CLI that mounts a kit CLI as one of its subcommands, such as ``example mcp``.

This module imports only click, so a host CLI can import it at start-up without loading the MCP
SDK, httpx, pydantic, jsonschema or PyYAML. They are loaded when the mounted group is first used:
``example --help`` and the host's own commands never pay for them.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import click

EXIT_INTERRUPTED = 130


class LazyKitGroup(click.Group):
    """A subcommand that stands in for a kit CLI and builds it only when the subcommand is used.

    The help and short help are static, so the host's ``--help`` lists the subcommand without
    building anything. When click makes the subcommand's context (to run it, show its help or
    complete it), the real group is built once by ``factory`` and takes over: the context belongs
    to the real group, which then runs the invocation itself.

    ``factory`` usually imports ``build_cli`` inside its body, so that the import of the kit's
    MCP side also waits until then::

        def factory() -> click.Group:
            from mcp_to_cli_kit.cli import build_cli

            return build_cli("example-mcp", DEFAULT_CONFIG, distribution="example-tool")
    """

    def __init__(
        self,
        name: str,
        help: str,
        factory: Callable[[], click.Group],
        *,
        short_help: str | None = None,
        hidden: bool = False,
    ) -> None:
        super().__init__(name=name, help=help, short_help=short_help, hidden=hidden)
        self._factory = factory
        self._group: click.Group | None = None

    def real(self) -> click.Group:
        """The real group, built on the first call."""
        if self._group is None:
            self._group = self._factory()
        return self._group

    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        return self.real().make_context(info_name, args, parent=parent, **extra)

    def invoke(self, ctx: click.Context) -> Any:
        return self.real().invoke(ctx)

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        return self.real().get_params(ctx)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return self.real().list_commands(ctx)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        return self.real().get_command(ctx, cmd_name)


def mount(
    host: click.Group,
    name: str,
    help: str,
    factory: Callable[[], click.Group],
    *,
    short_help: str | None = None,
    hidden: bool = False,
) -> LazyKitGroup:
    """Add a ``LazyKitGroup`` called ``name`` to ``host`` and return it."""
    group = LazyKitGroup(name, help, factory, short_help=short_help, hidden=hidden)
    host.add_command(group)
    return group


def run(group: click.Command, *, prog_name: str, args: Sequence[str] | None = None) -> NoReturn:
    """Run a CLI as a console script does: a kit CLI, or a host CLI with a kit CLI mounted in it.

    Like click's standalone mode: a command's return value is ignored, and only ``ctx.exit()``,
    ``click.exceptions.Exit`` or ``SystemExit`` set the exit code; an error prints one ``Error:``
    line and exits with its code; an abort (a declined ``confirm(abort=True)``, end of input at a
    prompt) prints ``Aborted!`` and exits 1. Unlike it, Ctrl-C prints ``Interrupted.`` and exits
    with 130, as a shell expects. Reads ``sys.argv`` when ``args`` is None.

    Logging is left to the host; a mounted kit group sets up its own when it is used.
    """
    _standalone(
        group,
        lambda: group.main(args=list(args) if args is not None else None, prog_name=prog_name, standalone_mode=False),
    )


def _standalone(group: click.Command, call: Callable[[], Any]) -> NoReturn:
    """Run ``call``, which runs ``group.main`` with ``standalone_mode=False``, and exit as standalone mode would."""
    # Without standalone mode, click's main returns either the command's return value or the code
    # of an Exit, and only these tell which: a return value must not become the exit code.
    returned = False
    invoke = group.invoke
    own_invoke = "invoke" in vars(group)

    def tracking_invoke(ctx: click.Context) -> Any:
        nonlocal returned
        result = invoke(ctx)
        returned = True
        return result

    group.invoke = tracking_invoke  # type: ignore[method-assign]
    try:
        # Click itself still handles a closed pipe (EPIPE) in this mode.
        result = call()
    except click.ClickException as error:
        error.show()
        sys.exit(error.exit_code)
    except click.Abort as error:
        # Click's main turns Ctrl-C into an Abort raised from the KeyboardInterrupt; a declined
        # confirmation, and end of input at a prompt, are aborts without one.
        if isinstance(error.__cause__, KeyboardInterrupt):
            _interrupted()
        click.echo("Aborted!", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        _interrupted()
    finally:
        if own_invoke:
            group.invoke = invoke  # type: ignore[method-assign]
        else:
            del group.invoke
    sys.exit(0 if returned or not isinstance(result, int) else result)


def _interrupted() -> NoReturn:
    click.echo("Interrupted.", err=True)
    sys.exit(EXIT_INTERRUPTED)
