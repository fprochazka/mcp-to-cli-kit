"""Commands that need no server start without importing the MCP SDK, httpx or anyio.

Each case runs in a fresh interpreter, because the test process has long imported them all.
No timing is asserted; the modules that were loaded are.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import BIN

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir

URL = "https://mcp.example.com/mcp"

HEAVY = ("mcp", "httpx", "httpcore", "anyio")

SCRIPT = f"""
import json
import sys

import click

from mcp_to_cli_kit.lazy import mount, run

if sys.argv[3] == "import-client":
    import mcp_to_cli_kit.client  # noqa: F401 - the control case


@click.command("whoami", help="A consumer's command.")
def whoami():
    click.echo("me")


def kit():
    from mcp_to_cli_kit.cli import build_cli

    org = click.Option(["--org"], is_eager=True)
    config = {{"servers": {{"example": {{"url": {URL!r}}}}}}}
    return build_cli({BIN!r}, config, extra_params=[org], extra_commands=[whoami])


if sys.argv[3] == "host":
    cli = click.Group("host")
    mount(cli, "mcp", "The tools.", kit)
else:
    cli = kit()
try:
    run(cli, prog_name="host" if sys.argv[3] == "host" else {BIN!r}, args=json.loads(sys.argv[1]))
except SystemExit as exit:
    if exit.code:
        raise
with open(sys.argv[2], "w") as out:
    json.dump(sorted(name for name in {HEAVY!r} if name in sys.modules), out)
"""

TOOL = {
    "name": "search_items",
    "description": "Search the items.",
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


@pytest.fixture(autouse=True)
def warm_cache(home: Path) -> None:
    SchemaCache(default_cache_dir(BIN)).save("example", URL, [TOOL])


def loaded_modules(tmp_path: Path, args: list[str], *, control: bool = False, host: bool = False) -> list[str]:
    out = tmp_path / "modules.json"
    variant = "import-client" if control else "host" if host else "-"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, json.dumps(args), str(out), variant],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    loaded: list[str] = json.loads(out.read_text(encoding="utf-8"))
    return loaded


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["--org", "work", "--help"],
        ["search-items", "--help"],
        ["schema", "search-items"],
        ["config"],
        ["whoami"],
        ["--org", "work", "whoami"],
    ],
)
def test_no_sdk_without_a_request(tmp_path: Path, args: list[str]) -> None:
    assert loaded_modules(tmp_path, args) == []


@pytest.mark.parametrize(
    "args",
    [["--help"], ["mcp", "--help"], ["mcp", "-v", "config"], ["mcp", "schema", "search-items"], ["mcp", "whoami"]],
)
def test_no_sdk_under_a_host_cli(tmp_path: Path, args: list[str]) -> None:
    assert loaded_modules(tmp_path, args, host=True) == []


def test_the_check_sees_the_sdk_when_it_is_imported(tmp_path: Path) -> None:
    # httpx imports httpcore only when it sends a request.
    assert loaded_modules(tmp_path, ["whoami"], control=True) == ["anyio", "httpx", "mcp"]
