"""What a consumer adds to the kit's bin: global options of its own."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result
from conftest import BIN
from fixture_server import RunningServer

from mcp_to_cli_kit.cli import build_cli


def org_option() -> click.Option:
    return click.Option(["-o", "--org", "org"], envvar="EXAMPLE_ORG", is_eager=True, help="The account to use.")


def run(server: RunningServer, args: Sequence[str], env: dict[str, str] | None = None, **kwargs: Any) -> Result:
    cli = build_cli(BIN, {"servers": {"example": {"url": server.url}}}, retry_delays=(0.0, 0.0), **kwargs)
    return CliRunner().invoke(cli, list(args), env=env, catch_exceptions=False)


class TestExtraParams:
    def test_an_extra_option_is_a_global_option(self, json_server: RunningServer) -> None:
        result = run(json_server, ["--help"], extra_params=[org_option()])

        assert result.exit_code == 0, result.stderr
        assert "-o, --org TEXT" in result.stdout and "The account to use." in result.stdout
        assert "search-items" in result.stdout

    @pytest.mark.parametrize("args", [["-o", "work", "no-args"], ["--org=work", "no-args"]])
    def test_a_tool_runs_with_the_option_before_it(self, json_server: RunningServer, args: list[str]) -> None:
        result = run(json_server, args, extra_params=[org_option()])

        assert result.exit_code == 0, result.stderr

    def test_without_extra_params_the_option_is_unknown(self, json_server: RunningServer) -> None:
        result = run(json_server, ["--org", "work", "no-args"])

        assert result.exit_code == 2
        assert "No such option" in result.stderr and "--org" in result.stderr

    @pytest.mark.parametrize(
        "param",
        [
            click.Option(["--config"]),
            click.Option(["-v", "--loud"]),
            click.Option(["--json", "as_json"]),
            click.Option(["--other", "refresh_tools"]),
            click.Option(["--help-me", "help"]),
        ],
    )
    def test_a_clash_with_a_kit_option_is_refused(self, param: click.Parameter) -> None:
        with pytest.raises(ValueError, match="clashes with a global option"):
            build_cli(BIN, {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}}, extra_params=[param])

    def test_two_extras_with_the_same_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="clashes"):
            build_cli(
                BIN,
                {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}},
                extra_params=[org_option(), click.Option(["--account", "org"])],
            )
