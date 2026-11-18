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
from click.testing import CliRunner, Result
from conftest import BIN
from fixture_auth import CLIENT_ID, ProtectedServer
from fixture_server import RunningServer

from mcp_to_cli_kit.cache import default_cache_dir
from mcp_to_cli_kit.cli import build_cli

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
