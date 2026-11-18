"""What a consumer adds to the kit's bin: global options, commands and config keys of its own."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import click
import pytest
import yaml
from click.testing import CliRunner, Result
from conftest import BIN
from fixture_server import RunningServer

from mcp_to_cli_kit.cli import build_cli
from mcp_to_cli_kit.config import ConfigError, default_config_path

SECRET = "acct-s3cr3t-key-value"


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


ACCOUNT_KEYS = frozenset({"orgs", "default_org"})


def accounts_layer(calls: list[Mapping[str, Any]] | None = None) -> Any:
    """A consumer's layer: the headers of the account --org or default_org selects."""

    def layer(ctx: click.Context, consumer: Mapping[str, Any]) -> Mapping[str, Any]:
        if calls is not None:
            calls.append(dict(consumer))
        orgs = consumer.get("orgs") or {}
        name = ctx.params.get("org") or consumer.get("default_org")
        if name is None:
            return {}
        if name not in orgs:
            raise ConfigError(f"No account {name!r}. Accounts: {', '.join(sorted(orgs))}.")
        account = orgs[name]
        headers = {"Authorization": f"Bearer {account['api_key']}"}
        if "org_id" in account:
            headers["X-Org-Id"] = account["org_id"]
        return {"servers": {"example": {"headers": headers}}}

    return layer


def write_config(data: dict[str, Any]) -> None:
    path = default_config_path(BIN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def accounts(server: RunningServer) -> dict[str, Any]:
    return {
        "default_org": "work",
        "orgs": {"work": {"api_key": SECRET, "org_id": "org-1111"}, "personal": {"api_key": "personal-key-0001"}},
    }


def sent(server: RunningServer, header: str) -> set[str | None]:
    return {request.headers.get(header) for request in server.recorder.requests}


class TestConsumerConfig:
    def consumer_run(self, server: RunningServer, args: Sequence[str], **kwargs: Any) -> Result:
        return run(
            server,
            args,
            extra_params=[org_option()],
            consumer_keys=ACCOUNT_KEYS,
            config_layer=kwargs.pop("config_layer", accounts_layer()),
            **kwargs,
        )

    def test_the_default_account_sends_its_headers(self, json_server: RunningServer) -> None:
        write_config(accounts(json_server))

        result = self.consumer_run(json_server, ["no-args"])

        assert result.exit_code == 0, result.stderr
        assert sent(json_server, "authorization") == {f"Bearer {SECRET}"}
        assert sent(json_server, "x-org-id") == {"org-1111"}

    @pytest.mark.parametrize(("args", "env"), [(["--org", "personal"], None), ([], {"EXAMPLE_ORG": "personal"})])
    def test_the_option_selects_another_account(
        self, json_server: RunningServer, args: list[str], env: dict[str, str] | None
    ) -> None:
        write_config(accounts(json_server))

        result = self.consumer_run(json_server, [*args, "no-args"], env=env)

        assert result.exit_code == 0, result.stderr
        assert sent(json_server, "authorization") == {"Bearer personal-key-0001"}
        assert sent(json_server, "x-org-id") == {None}

    def test_the_layer_runs_once_per_invocation(self, json_server: RunningServer) -> None:
        write_config(accounts(json_server))
        calls: list[Mapping[str, Any]] = []

        result = self.consumer_run(json_server, ["no-args"], config_layer=accounts_layer(calls))

        assert result.exit_code == 0, result.stderr
        assert calls == [accounts(json_server)]

    def test_without_a_file_the_layer_gets_nothing(self, json_server: RunningServer) -> None:
        calls: list[Mapping[str, Any]] = []

        result = self.consumer_run(json_server, ["no-args"], config_layer=accounts_layer(calls))

        assert result.exit_code == 0, result.stderr
        assert calls == [{}]
        assert sent(json_server, "authorization") == {None}

    def test_a_config_error_of_the_layer_exits_2(self, json_server: RunningServer) -> None:
        write_config(accounts(json_server))

        result = self.consumer_run(json_server, ["--org", "missing", "no-args"])

        assert result.exit_code == 2
        assert "No account 'missing'. Accounts: personal, work." in result.stderr
        assert json_server.recorder.requests == []

    def test_the_file_and_the_environment_win_over_the_layer(self, json_server: RunningServer) -> None:
        data = accounts(json_server)
        data["servers"] = {"example": {"headers": {"X-Org-Id": "from-file"}}}
        write_config(data)

        result = self.consumer_run(
            json_server, ["no-args"], env={"EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION": "Bearer from-env"}
        )

        assert result.exit_code == 0, result.stderr
        assert sent(json_server, "authorization") == {"Bearer from-env"}
        assert sent(json_server, "x-org-id") == {"from-file"}

    def test_consumer_keys_are_not_read_from_the_environment(self, json_server: RunningServer) -> None:
        calls: list[Mapping[str, Any]] = []

        result = self.consumer_run(
            json_server,
            ["-v", "no-args"],
            env={"EXAMPLE_TOOL_DEFAULT_ORG": "work", "EXAMPLE_TOOL_ORGS": "{}"},
            config_layer=accounts_layer(calls),
        )

        assert result.exit_code == 0, result.stderr
        assert calls == [{}]
        assert "Ignoring EXAMPLE_TOOL_DEFAULT_ORG" in result.stderr

    def test_config_lists_the_consumer_keys_by_name_only(self, json_server: RunningServer) -> None:
        write_config(accounts(json_server))

        result = self.consumer_run(json_server, ["config", "--json"])

        assert result.exit_code == 0, result.stderr
        assert SECRET not in result.stdout + result.stderr
        shown = json.loads(result.stdout)
        assert shown["consumer_keys"] == ["default_org", "orgs"]
        assert shown["config"]["servers"]["example"]["headers"]["Authorization"] == f"Bearer ***{SECRET[-4:]}"
        assert "orgs" not in shown["config"]

    def test_config_without_consumer_keys_shows_no_such_field(self, json_server: RunningServer) -> None:
        result = run(json_server, ["config", "--json"])

        assert "consumer_keys" not in json.loads(result.stdout)

    def test_without_consumer_keys_the_file_keys_are_invalid(self, json_server: RunningServer) -> None:
        write_config(accounts(json_server))

        result = run(json_server, ["no-args"])

        assert result.exit_code == 2
        assert "Extra inputs are not permitted" in result.stderr

    @pytest.mark.parametrize("key", ["servers", "help"])
    def test_a_kit_key_cannot_be_a_consumer_key(self, key: str) -> None:
        with pytest.raises(ValueError, match="kit's own keys"):
            build_cli(BIN, {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}}, consumer_keys=[key])
