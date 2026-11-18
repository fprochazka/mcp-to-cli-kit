"""What a consumer adds to the kit's bin: global options and commands of its own."""

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


def run(
    server: RunningServer,
    args: Sequence[str],
    env: dict[str, str] | None = None,
    default: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Result:
    config = default if default is not None else {"servers": {"example": {"url": server.url}}}
    cli = build_cli(BIN, config, retry_delays=(0.0, 0.0), **kwargs)
    return CliRunner().invoke(cli, list(args), env=env, catch_exceptions=False)


@click.command("whoami", help="Print the selected account and the config option.")
@click.pass_context
def whoami(ctx: click.Context) -> None:
    assert ctx.parent is not None
    click.echo(f"org={ctx.parent.params.get('org')} config={ctx.parent.params.get('config')}")


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


class TestExtraCommands:
    def test_listed_under_commands_with_the_built_ins(self, json_server: RunningServer) -> None:
        result = run(json_server, ["--help"], extra_commands=[whoami])

        assert result.exit_code == 0, result.stderr
        commands = result.stdout.split("Commands:")[1]
        assert "whoami" in commands and "Print the selected account" in commands
        assert "schema" in commands and "config" in commands

    def test_runs_with_the_global_options(self, json_server: RunningServer) -> None:
        result = run(
            json_server,
            ["--org", "work", "--config", "/nonexistent.yaml", "whoami"],
            extra_params=[org_option()],
            extra_commands=[whoami],
        )

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "org=work config=/nonexistent.yaml\n"

    def test_runs_without_the_config_or_a_server(self, json_server: RunningServer) -> None:
        # The file does not exist and the default is invalid: any tool command would exit 2.
        broken = {"servers": {"example": {"url": "not a url"}}}

        result = run(json_server, ["--config", "/nonexistent.yaml", "whoami"], default=broken, extra_commands=[whoami])

        assert result.exit_code == 0, result.stderr
        assert json_server.recorder.requests == []

    def test_a_tool_of_the_same_name_gets_a_suffix(self, json_server: RunningServer) -> None:
        clash = click.Command("no-args", callback=lambda: click.echo("the consumer's command"))

        listed = run(json_server, ["--help"], extra_commands=[clash])
        own = run(json_server, ["no-args"], extra_commands=[clash])
        tool = run(json_server, ["no-args-tool"], extra_commands=[clash])

        assert "no-args-tool" in listed.stdout
        assert own.stdout == "the consumer's command\n"
        assert tool.exit_code == 0, tool.stderr
        assert "the consumer's command" not in tool.stdout

    def test_a_group_mount_of_the_same_name_is_a_config_error(self, json_server: RunningServer) -> None:
        default = {"servers": {"example": {"url": json_server.url, "mount": {"group": "whoami"}}}}

        result = run(json_server, ["--help"], default=default, extra_commands=[whoami])

        assert result.exit_code == 2
        assert "'whoami' has the name of a command of example-tool" in result.stderr

    @pytest.mark.parametrize("name", ["schema", "config", "auth"])
    def test_a_built_in_name_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="built-in command"):
            build_cli(
                BIN,
                {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}},
                extra_commands=[click.Command(name)],
            )

    def test_two_commands_of_the_same_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="two extra commands"):
            build_cli(
                BIN,
                {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}},
                extra_commands=[whoami, click.Command("whoami")],
            )
