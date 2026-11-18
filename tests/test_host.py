"""A kit CLI mounted as a subcommand of a host CLI: ``host mcp TOOL``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import click
import pytest
from click.shell_completion import BashComplete
from click.testing import CliRunner, Result
from conftest import BIN
from fixture_auth import CLIENT_ID, ProtectedServer
from fixture_server import RunningServer

from mcp_to_cli_kit.cache import default_cache_dir
from mcp_to_cli_kit.cli import build_cli
from mcp_to_cli_kit.lazy import LazyKitGroup, mount

Host = Callable[..., Result]


def host_group() -> click.Group:
    @click.group()
    def host() -> None:
        """A host CLI with its own commands."""

    @host.command()
    def hello() -> None:
        """Say hello."""
        click.echo("hello")

    return host


def kit(url: str) -> click.Group:
    return build_cli(BIN, {"servers": {"example": {"url": url}}}, retry_delays=(0.0, 0.0))


@pytest.fixture
def host(json_server: RunningServer) -> Host:
    """Run ``host`` with the kit CLI of the JSON fixture server mounted as ``host mcp``."""

    def run(args: Sequence[str], *, kit_cli: click.Group | None = None) -> Result:
        group = host_group()
        group.add_command(kit_cli if kit_cli is not None else kit(json_server.url), name="mcp")
        return CliRunner().invoke(group, list(args), catch_exceptions=False)

    return run


class TestMountedGroup:
    def test_help_lists_the_tools_under_the_mounted_path(self, host: Host) -> None:
        result = host(["mcp", "--help"])

        assert result.exit_code == 0, result.stderr
        assert "Usage: host mcp [OPTIONS] COMMAND [ARGS]..." in result.stdout
        assert "run 'host mcp TOOL --help' for its options" in " ".join(result.stdout.split())
        assert "search-items" in result.stdout

    def test_a_tool_is_called(self, host: Host, json_server: RunningServer) -> None:
        result = host(["mcp", "search-items", "--query", "red"])

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == {"status": "ok"}
        assert json_server.recorder.calls()[0]["arguments"] == {"query": "red"}

    def test_the_kit_keeps_its_own_storage(self, host: Host) -> None:
        host(["mcp", "--help"])

        assert (default_cache_dir(BIN) / "example.json").exists()

    def test_config_reads_the_mounted_groups_options(self, host: Host, tmp_path: Path) -> None:
        other = tmp_path / "other.yaml"
        other.write_text("servers:\n  example:\n    timeout: 7\n", encoding="utf-8")

        result = host(["mcp", "--config", str(other), "config"])

        assert result.exit_code == 0, result.stderr
        shown = json.loads(result.stdout)
        assert shown["config_file"] == str(other)
        assert shown["config_file_loaded"] is True
        assert shown["config"]["servers"]["example"]["timeout"] == 7
        assert shown["env_prefix"] == "EXAMPLE_TOOL_"

    def test_an_unknown_command_hints_the_mounted_help(self, host: Host) -> None:
        result = host(["mcp", "zzzz"])

        assert result.exit_code == 2
        assert "No such command 'zzzz'. Run 'host mcp --help' for the list." in result.stderr

    def test_a_tool_usage_error_names_the_mounted_path(self, host: Host) -> None:
        result = host(["mcp", "search-items"])

        assert result.exit_code == 2
        assert "host mcp search-items" in result.stderr

    def test_the_sign_in_hint_names_the_mounted_path(self, host: Host, protected_server: ProtectedServer) -> None:
        config = {
            "servers": {"example": {"url": protected_server.url, "auth": {"type": "oauth", "client_id": CLIENT_ID}}}
        }

        result = host(["mcp", "search-items", "--query", "x"], kit_cli=build_cli(BIN, config))

        assert result.exit_code == 4
        assert "Run 'host mcp auth login' in a terminal" in result.stderr

    def test_a_missing_configuration_names_the_mounted_path(self, host: Host) -> None:
        result = host(["mcp", "--help"], kit_cli=build_cli(BIN))

        assert result.exit_code == 2
        assert "Error: host mcp has no configuration." in result.stderr

    def test_the_host_commands_still_work(self, host: Host) -> None:
        result = host(["hello"])

        assert result.exit_code == 0
        assert result.stdout == "hello\n"


ROOT_SCRIPT = """
import sys
from mcp_to_cli_kit.cli import main
main({bin!r}, {{"servers": {{"example": {{"url": {url!r}}}}}}})
"""

MOUNTED_SCRIPT = """
import click
from mcp_to_cli_kit.cli import build_cli

@click.group()
def host():
    "A host CLI."

host.add_command(build_cli({bin!r}, {{"servers": {{"example": {{"url": {url!r}}}}}}}), name="mcp")
host(prog_name="host")
"""


COMPLETIONS = [
    pytest.param("sea", "plain,search-items\n", id="tool name"),
    pytest.param("search-items --q", "plain,--query\n", id="tool option"),
    pytest.param("search-items ", "\n", id="tool argument"),
    pytest.param("config --j", "plain,--json\n", id="built-in option"),
]


def complete(code: str, home: Path, var: str, words: str) -> subprocess.CompletedProcess[str]:
    """Ask a CLI in a fresh process to complete ``words``, as bash does."""
    current = len(words.split()) - (0 if words.endswith(" ") else 1)
    env = {**os.environ, "HOME": str(home), var: "bash_complete", "COMP_WORDS": words, "COMP_CWORD": str(current)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize(
    ("script", "prefix", "var"),
    [(ROOT_SCRIPT, BIN, "_EXAMPLE_TOOL_COMPLETE"), (MOUNTED_SCRIPT, "host mcp", "_HOST_COMPLETE")],
    ids=["root", "mounted"],
)
@pytest.mark.parametrize(("words", "expected"), COMPLETIONS)
def test_completion_with_an_empty_cache_exits(
    json_server: RunningServer, home: Path, script: str, prefix: str, var: str, words: str, expected: str
) -> None:
    """Run as a real process: the runtime created to list the tools must be closed, or the process hangs."""
    completed = complete(script.format(bin=BIN, url=json_server.url), home, var, f"{prefix} {words}")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == expected


HEAVY_MODULES = ("mcp", "httpx", "pydantic", "yaml", "jsonschema", "anyio", "mcp_to_cli_kit.cli")


class CountingFactory:
    """Builds the kit CLI of a server and counts the builds."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.calls = 0

    def __call__(self) -> click.Group:
        self.calls += 1
        return kit(self.url)


@pytest.fixture
def factory(json_server: RunningServer) -> CountingFactory:
    return CountingFactory(json_server.url)


@pytest.fixture
def lazy_host(factory: CountingFactory) -> click.Group:
    group = host_group()
    mount(group, "mcp", "Call the tools of the example MCP server.", factory)
    return group


def invoke_lazy(group: click.Group, args: Sequence[str]) -> Result:
    return CliRunner().invoke(group, list(args), catch_exceptions=False)


class TestLazyKitGroup:
    def test_host_help_lists_the_group_without_building_it(
        self, lazy_host: click.Group, factory: CountingFactory
    ) -> None:
        result = invoke_lazy(lazy_host, ["--help"])

        assert result.exit_code == 0
        assert "mcp    Call the tools of the example MCP server." in result.stdout
        assert "hello  Say hello." in result.stdout
        assert factory.calls == 0

    def test_host_commands_do_not_build_it(self, lazy_host: click.Group, factory: CountingFactory) -> None:
        result = invoke_lazy(lazy_host, ["hello"])

        assert result.exit_code == 0
        assert factory.calls == 0

    def test_help_lists_the_tools(self, lazy_host: click.Group, factory: CountingFactory) -> None:
        result = invoke_lazy(lazy_host, ["mcp", "--help"])

        assert result.exit_code == 0, result.stderr
        assert "Usage: host mcp [OPTIONS] COMMAND [ARGS]..." in result.stdout
        assert "search-items" in result.stdout
        assert factory.calls == 1

    def test_no_arguments_show_the_help(self, lazy_host: click.Group) -> None:
        result = invoke_lazy(lazy_host, ["mcp"])

        assert "search-items" in result.output

    def test_config(self, lazy_host: click.Group) -> None:
        result = invoke_lazy(lazy_host, ["mcp", "config"])

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout)["env_prefix"] == "EXAMPLE_TOOL_"

    def test_a_tool_is_called(self, lazy_host: click.Group, json_server: RunningServer) -> None:
        result = invoke_lazy(lazy_host, ["mcp", "search-items", "--query", "red"])

        assert result.exit_code == 0, result.stderr
        assert json_server.recorder.calls()[0]["arguments"] == {"query": "red"}

    def test_an_unknown_command_hints_the_mounted_help(self, lazy_host: click.Group) -> None:
        result = invoke_lazy(lazy_host, ["mcp", "zzzz"])

        assert result.exit_code == 2
        assert "No such command 'zzzz'. Run 'host mcp --help' for the list." in result.stderr

    def test_the_group_is_built_once(self, factory: CountingFactory) -> None:
        group = LazyKitGroup("mcp", "The tools.", factory)

        assert group.real() is group.real()
        assert factory.calls == 1

    @pytest.mark.parametrize("cached", [True, False], ids=["cached", "fetched"])
    def test_completion_of_tool_names(self, lazy_host: click.Group, cached: bool) -> None:
        if cached:
            invoke_lazy(lazy_host, ["mcp", "--help"])

        completion = BashComplete(lazy_host, {}, "host", "_HOST_COMPLETE")
        names = [item.value for item in completion.get_completions(["mcp"], "search")]

        assert names == ["search-items"]

    def test_completion_of_the_host_does_not_build_it(self, lazy_host: click.Group, factory: CountingFactory) -> None:
        completion = BashComplete(lazy_host, {}, "host", "_HOST_COMPLETE")
        names = [item.value for item in completion.get_completions([], "")]

        assert names == ["hello", "mcp"]
        assert factory.calls == 0


def lazy_host_script(url: str) -> str:
    """A host CLI run through ``run()`` in a fresh interpreter, which reports the heavy modules it loaded."""
    return f"""
import atexit, sys
import click
from mcp_to_cli_kit.lazy import mount, run

atexit.register(lambda: print("loaded:", [m for m in {HEAVY_MODULES!r} if m in sys.modules], file=sys.stderr))

def factory():
    from mcp_to_cli_kit.cli import build_cli
    return build_cli({BIN!r}, {{"servers": {{"example": {{"url": {url!r}}}}}}})

@click.group()
def host():
    "A host CLI."

@host.command()
def hello():
    click.echo("hello")

@host.command()
def returns():
    return 3

@host.command()
@click.pass_context
def exits(ctx):
    ctx.exit(3)

@host.command()
def confirm():
    click.confirm("Go on?", abort=True)

@host.command()
def prompt():
    click.prompt("Name")

@host.command()
def loggers():
    import logging
    library = logging.getLogger("httpx")
    print([library.level, len(library.handlers), library.propagate, len(logging.getLogger().handlers)])

mount(host, "mcp", "Call the tools.", factory)
run(host, prog_name="host")
"""


def run_script(code: str, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True, text=True, timeout=60)


class TestImportCost:
    def test_importing_the_host_helpers_loads_no_mcp_dependency(self, home: Path) -> None:
        code = f"import sys, mcp_to_cli_kit.lazy; print([m for m in {HEAVY_MODULES!r} if m in sys.modules])"

        completed = run_script(code, home)

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "[]\n"

    @pytest.mark.parametrize("args", [["--help"], ["hello"]])
    def test_the_host_runs_without_loading_the_kit(
        self, json_server: RunningServer, home: Path, args: list[str]
    ) -> None:
        completed = run_script(lazy_host_script(json_server.url), home, *args)

        assert completed.returncode == 0, completed.stderr
        assert "loaded: []" in completed.stderr

    def test_the_mounted_group_loads_the_kit_when_called(self, json_server: RunningServer, home: Path) -> None:
        completed = run_script(lazy_host_script(json_server.url), home, "mcp", "--help")

        assert completed.returncode == 0, completed.stderr
        assert "search-items" in completed.stdout
        assert "'mcp'" in completed.stderr

    def test_run_exits_with_the_kit_exit_codes(self, json_server: RunningServer, home: Path) -> None:
        script = lazy_host_script(json_server.url)

        assert run_script(script, home, "mcp", "fail", "--message", "boom").returncode == 1
        assert run_script(script, home, "mcp", "zzzz").returncode == 2
        assert run_script(script, home, "nosuch").returncode == 2

    @pytest.mark.parametrize(("words", "expected"), COMPLETIONS)
    def test_completion_with_an_empty_cache_exits(
        self, json_server: RunningServer, home: Path, words: str, expected: str
    ) -> None:
        completed = complete(lazy_host_script(json_server.url), home, "_HOST_COMPLETE", f"host mcp {words}")

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == expected


class TestRun:
    @pytest.mark.parametrize(("command", "code"), [("returns", 0), ("exits", 3)])
    def test_only_an_exit_sets_the_exit_code(
        self, json_server: RunningServer, home: Path, command: str, code: int
    ) -> None:
        completed = run_script(lazy_host_script(json_server.url), home, command)

        assert completed.returncode == code, completed.stderr

    @pytest.mark.parametrize(("command", "answer"), [("confirm", "n\n"), ("prompt", "")])
    def test_an_abort_exits_1(self, json_server: RunningServer, home: Path, command: str, answer: str) -> None:
        env = {**os.environ, "HOME": str(home)}
        completed = subprocess.run(
            [sys.executable, "-c", lazy_host_script(json_server.url), command],
            env=env,
            input=answer,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert completed.returncode == 1
        assert "Aborted!" in completed.stderr
        assert "Interrupted." not in completed.stderr

    def test_the_host_keeps_its_own_logging(self, json_server: RunningServer, home: Path) -> None:
        completed = run_script(lazy_host_script(json_server.url), home, "loggers")

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == "[0, 0, True, 0]\n"


def test_info_dict_lists_the_tools_of_a_lazy_group(lazy_host: click.Group, factory: CountingFactory) -> None:
    with click.Context(lazy_host, info_name="host") as ctx:
        info = lazy_host.to_info_dict(ctx)

    assert "search-items" in info["commands"]["mcp"]["commands"]
    assert factory.calls == 1
