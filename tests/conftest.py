"""Shared fixtures: the fixture MCP servers, an empty home directory and a shell without an AI agent."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from fixture_server import RunningServer, start

from mcp_to_cli_kit import agent as agent_module
from mcp_to_cli_kit.agent import AGENT_ENV_VARS, GENERIC_AGENT_ENV_VARS
from mcp_to_cli_kit.cli import ResultClassifier, build_cli

BIN = "example-tool"

ALL_AGENT_VARS = [name for _, variables in AGENT_ENV_VARS for name in variables] + GENERIC_AGENT_ENV_VARS


@pytest.fixture(scope="session")
def json_server() -> Iterator[RunningServer]:
    """A server that answers with plain application/json."""
    server = start(json_response=True)
    yield server
    server.stop()


@pytest.fixture(scope="session")
def sse_server() -> Iterator[RunningServer]:
    """A server that answers with an SSE stream."""
    server = start(json_response=False)
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An empty home directory, no XDG directory overrides and no AI agent, for every test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The kit honors these, and CI runners set them: without this, tests would read and write
    # the runner's real config and cache. Tests of the XDG behavior set them explicitly.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    for variable in ALL_AGENT_VARS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(agent_module, "DEVIN_MARKER", Path("/nonexistent-devin-marker"))
    return home


@pytest.fixture(autouse=True)
def reset_servers(request: pytest.FixtureRequest) -> None:
    """Forget what earlier tests sent to the servers they used."""
    for name in ("json_server", "sse_server"):
        if name in request.fixturenames:
            server: RunningServer = request.getfixturevalue(name)
            server.recorder.reset()


Invoke = Callable[..., Result]


@pytest.fixture
def invoke(json_server: RunningServer) -> Invoke:
    """Run the CLI against the JSON fixture server (or ``url=...``) and return click's result."""

    def run(
        args: Sequence[str],
        *,
        url: str | None = None,
        default: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        input: str | bytes | None = None,
        classifier: ResultClassifier | None = None,
        distribution: str | None = None,
    ) -> Result:
        config = default if default is not None else {"servers": {"example": {"url": url or json_server.url}}}
        cli = build_cli(BIN, config, retry_delays=(0.0, 0.0), result_classifier=classifier, distribution=distribution)
        return CliRunner().invoke(cli, list(args), env=env, input=input, catch_exceptions=False)

    return run
