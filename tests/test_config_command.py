"""The config command: the effective configuration and its sources, with secrets masked, offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import BIN, Invoke
from fixture_server import RunningServer, free_port_url

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir
from mcp_to_cli_kit.config import default_config_path
from mcp_to_cli_kit.config_view import mask

SECRET = "s3cr3t-literal-value"

URL = "https://mcp.example.com/mcp"


@pytest.fixture(autouse=True)
def no_test_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EXAMPLE_TOKEN", "EXAMPLE_HOST"):
        monkeypatch.delenv(name, raising=False)


def show(invoke: Invoke, server: dict[str, Any], *args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = invoke(["config", "--json", *args], default={"servers": {"example": server}}, env=env)
    assert result.exit_code == 0, result.stderr
    assert SECRET not in result.stdout + result.stderr
    shown: dict[str, Any] = json.loads(result.stdout)
    return shown


class TestOutput:
    def test_the_sources_and_the_merged_config(self, invoke: Invoke) -> None:
        shown = show(invoke, {"url": URL})

        assert shown["config_file"] == str(default_config_path(BIN))
        assert shown["config_file_loaded"] is False
        assert shown["env_prefix"] == "EXAMPLE_TOOL_"
        assert shown["env_applied"] == []
        server = shown["config"]["servers"]["example"]
        assert server["url"] == URL
        assert server["mount"] == "root" and server["timeout"] == 30.0 and server["strip_prefixes"] == []

    def test_a_loaded_file(self, invoke: Invoke, tmp_path: Path) -> None:
        path = tmp_path / "other.yaml"
        path.write_text(yaml.safe_dump({"servers": {"example": {"timeout": 5}}}), encoding="utf-8")

        result = invoke(["--config", str(path), "config", "--json"], default={"servers": {"example": {"url": URL}}})

        assert result.exit_code == 0
        assert result.stderr == ""
        data = json.loads(result.stdout)
        assert data["config_file"] == str(path) and data["config_file_loaded"] is True
        assert data["config"]["servers"]["example"]["timeout"] == 5

    def test_env_variables_are_listed_by_name_only(self, invoke: Invoke) -> None:
        env = {
            "EXAMPLE_TOOL_HELP": "From env",
            "EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION": f"Bearer {SECRET}",
            "EXAMPLE_TOOL_UNRELATED": "ignored",
        }

        shown = show(invoke, {"url": URL}, env=env)

        assert shown["env_applied"] == ["EXAMPLE_TOOL_HELP", "EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION"]
        assert shown["config"]["help"] == "From env"
        assert shown["config"]["servers"]["example"]["headers"]["Authorization"] == "Bearer " + mask(SECRET)

    def test_machine_mode_is_compact(self, invoke: Invoke) -> None:
        result = invoke(["config", "--json"], default={"servers": {"example": {"url": URL}}})

        assert result.stdout.count("\n") == 1

    @pytest.mark.parametrize("args", [["config", "--pretty"], ["config"]])
    def test_pretty_mode_is_indented(self, invoke: Invoke, args: list[str]) -> None:
        result = invoke(args, default={"servers": {"example": {"url": URL}}})

        assert result.exit_code == 0
        assert '\n  "config_file": ' in result.stdout
        json.loads(result.stdout)

    def test_an_agent_gets_machine_mode(self, invoke: Invoke) -> None:
        result = invoke(["config"], default={"servers": {"example": {"url": URL}}}, env={"CLAUDECODE": "1"})

        assert result.stdout.count("\n") == 1

    def test_the_group_level_flag_works_too(self, invoke: Invoke) -> None:
        result = invoke(["--json", "config"], default={"servers": {"example": {"url": URL}}})

        assert result.stdout.count("\n") == 1

    def test_an_invalid_config_exits_2(self, invoke: Invoke) -> None:
        result = invoke(["config"], default={"servers": {"example": {"url": "ftp://x"}}})

        assert result.exit_code == 2
        assert "Invalid configuration" in result.stderr


class TestMasking:
    def test_templates_are_shown_unexpanded(self, invoke: Invoke) -> None:
        server = {"url": "https://${EXAMPLE_HOST}/mcp", "headers": {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}}

        shown = show(invoke, server, env={"EXAMPLE_TOKEN": SECRET, "EXAMPLE_HOST": SECRET})

        example = shown["config"]["servers"]["example"]
        assert example["url"] == "https://${EXAMPLE_HOST}/mcp"
        assert example["headers"] == {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}

    def test_unset_variables_are_fine(self, invoke: Invoke) -> None:
        shown = show(invoke, {"url": "https://${EXAMPLE_HOST}/mcp"})

        assert shown["config"]["servers"]["example"]["url"] == "https://${EXAMPLE_HOST}/mcp"

    @pytest.mark.parametrize(
        ("header", "value", "expected"),
        [
            ("Authorization", f"Bearer {SECRET}", "Bearer " + mask(SECRET)),
            ("X-Api-Key", SECRET, mask(SECRET)),
            ("X-Auth-Token", "short", "***"),
            ("Cookie", f"session={SECRET}", mask(f"session={SECRET}")),
            ("X-Client-Secret", f"{SECRET}${{EXAMPLE_TOKEN}}", mask(SECRET) + "${EXAMPLE_TOKEN}"),
            ("X-Session", SECRET, mask(SECRET)),
            ("X-Custom", "plain", "***"),
            ("Accept", "application/json", "application/json"),
            ("content-type", "application/json", "application/json"),
            ("User-Agent", "probe/1.0", "probe/1.0"),
            ("X-Mcp-Client", "probe/1.0", "probe/1.0"),
        ],
    )
    def test_literal_header_values(self, invoke: Invoke, header: str, value: str, expected: str) -> None:
        shown = show(invoke, {"url": URL, "headers": {header: value}})

        assert shown["config"]["servers"]["example"]["headers"][header] == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (f"Bearer {SECRET}", "Bearer " + mask(SECRET)),
            (f"basic {SECRET}", "basic " + mask(SECRET)),
            (f"Token {SECRET}", "Token " + mask(SECRET)),
            (f"Digest {SECRET}", "Digest " + mask(SECRET)),
            (f"Bot {SECRET}", "Bot " + mask(SECRET)),
            (f"ApiKey {SECRET}", "ApiKey " + mask(SECRET)),
            ("Bearer ${EXAMPLE_TOKEN}", "Bearer ${EXAMPLE_TOKEN}"),
            ("Lettersonlysecretword ${EXAMPLE_TOKEN}", mask("Lettersonlysecretword ") + "${EXAMPLE_TOKEN}"),
            ("correct horse battery staple", mask("correct horse battery staple")),
            ("Bearer", "***"),
            ("Bearer ", "***"),
        ],
    )
    def test_the_auth_scheme_word(self, invoke: Invoke, value: str, expected: str) -> None:
        shown = show(invoke, {"url": URL, "headers": {"Authorization": value}})

        assert shown["config"]["servers"]["example"]["headers"]["Authorization"] == expected
        assert "Lettersonlysecretword" not in json.dumps(shown) and "horse" not in json.dumps(shown)

    @pytest.mark.parametrize(("length", "expected"), [(11, "***"), (12, "***" + "q9Wx")])
    def test_the_12_character_boundary(self, invoke: Invoke, length: int, expected: str) -> None:
        value = "Z" * (length - 4) + "q9Wx"

        shown = show(invoke, {"url": URL, "headers": {"X-Api-Key": value}})

        assert shown["config"]["servers"]["example"]["headers"]["X-Api-Key"] == expected

    def test_masked_in_pretty_mode_and_from_the_file(self, invoke: Invoke, tmp_path: Path) -> None:
        path = tmp_path / "secret.yaml"
        file = {"servers": {"example": {"url": f"https://u:{SECRET}@h.example/mcp", "headers": {"X-Session": SECRET}}}}
        path.write_text(yaml.safe_dump(file), encoding="utf-8")

        result = invoke(["--config", str(path), "config", "--pretty"], default={"servers": {"example": {"url": URL}}})

        assert result.exit_code == 0
        assert SECRET not in result.stdout + result.stderr
        assert '\n  "config_file": ' in result.stdout
        example = json.loads(result.stdout)["config"]["servers"]["example"]
        assert example["headers"] == {"X-Session": mask(SECRET)}
        assert example["url"] == f"https://{mask('u:' + SECRET)}@h.example/mcp"

    def test_mask_reveals_the_last_4_characters_of_long_values_only(self) -> None:
        assert mask(SECRET) == "***" + SECRET[-4:]
        assert mask("eleven-char") == "***"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (f"https://user:{SECRET}@mcp.example.com/mcp", f"https://{mask('user:' + SECRET)}@mcp.example.com/mcp"),
            (f"https://{SECRET}@mcp.example.com/mcp", f"https://{mask(SECRET)}@mcp.example.com/mcp"),
            (f"https://mcp.example.com/mcp#{SECRET}", f"https://mcp.example.com/mcp#{mask(SECRET)}"),
            ("https://mcp.example.com:8443/mcp/v1", "https://mcp.example.com:8443/mcp/v1"),
            (f"https://mcp.example.com/mcp?token={SECRET}&x=", f"https://mcp.example.com/mcp?token={mask(SECRET)}&x="),
            (
                "https://u:${EXAMPLE_TOKEN}@h/mcp?t=${EXAMPLE_TOKEN}#${EXAMPLE_TOKEN}",
                "https://***${EXAMPLE_TOKEN}@h/mcp?t=${EXAMPLE_TOKEN}#${EXAMPLE_TOKEN}",
            ),
            ("https://${EXAMPLE_TOKEN}@h:${EXAMPLE_HOST}/mcp", "https://${EXAMPLE_TOKEN}@h:${EXAMPLE_HOST}/mcp"),
            (f"https://h/mcp?{SECRET}", f"https://h/mcp?{mask(SECRET)}"),
            (f"${{EXAMPLE_HOST}}?token={SECRET}", "${EXAMPLE_HOST}" + mask(f"?token={SECRET}")),
            ("${EXAMPLE_HOST}/mcp", "${EXAMPLE_HOST}/mcp"),
        ],
    )
    def test_urls(self, invoke: Invoke, url: str, expected: str) -> None:
        shown = show(invoke, {"url": url}, env={"EXAMPLE_TOKEN": SECRET})

        assert shown["config"]["servers"]["example"]["url"] == expected

    @pytest.mark.parametrize(
        "url",
        [f"https://u:ab#{SECRET}@h.example/mcp", f"https://h.example:{SECRET}/mcp", "https://h.example:99999/mcp"],
    )
    def test_a_malformed_port_is_a_config_error(self, invoke: Invoke, url: str) -> None:
        result = invoke(["config"], default={"servers": {"example": {"url": url}}})

        assert result.exit_code == 2
        assert "numeric port" in result.stderr
        assert SECRET not in result.stdout + result.stderr


class TestOffline:
    def test_no_request_and_no_cache(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["config"], default={"servers": {"example": {"url": json_server.url}}})

        assert result.exit_code == 0
        assert json_server.recorder.requests == []
        assert not default_cache_dir(BIN).exists()

    def test_with_an_unreachable_server(self, invoke: Invoke) -> None:
        result = invoke(["config"], default={"servers": {"example": {"url": free_port_url()}}})

        assert result.exit_code == 0

    def test_config_help_needs_no_server(self, invoke: Invoke) -> None:
        result = invoke(["config", "--help"], default={"servers": {"example": {"url": free_port_url()}}})

        assert result.exit_code == 0
        assert "Print the effective configuration" in result.stdout


class TestReservedName:
    def test_listed_in_the_top_level_help(self, invoke: Invoke) -> None:
        result = invoke(["--help"])

        commands = result.stdout.split("Commands:")[1]
        assert "\n  config " in commands and "\n  schema " in commands

    def test_not_listed_inside_a_group(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(
            ["ex", "--help"], default={"servers": {"example": {"url": json_server.url, "mount": {"group": "ex"}}}}
        )

        assert "\n  config " not in result.stdout.split("Commands:")[1]

    def test_a_tool_named_config_inside_a_group_mount_is_renamed(self, invoke: Invoke) -> None:
        url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("example", url, [{"name": "config", "inputSchema": {}}])
        default = {"servers": {"example": {"url": url, "mount": {"group": "ex"}}}}

        listed = invoke(["ex", "--help"], default=default)
        schema = invoke(["ex", "schema", "config-tool"], default=default)

        assert "\n  config-tool\n" in listed.stdout.split("Tools:")[1]
        assert schema.exit_code == 0

    def test_a_tool_named_config_is_renamed(self, invoke: Invoke) -> None:
        url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("example", url, [{"name": "config", "inputSchema": {}}])
        default = {"servers": {"example": {"url": url}}}

        listed = invoke(["--help"], default=default)
        schema = invoke(["schema", "config-tool"], default=default)

        assert "\n  config-tool\n" in listed.stdout.split("Tools:")[1]
        assert schema.exit_code == 0

    @pytest.mark.parametrize(
        "server",
        [
            {"url": URL, "mount": {"group": "config"}},
            {"url": URL, "tool_groups": [{"prefix": "c_", "group": "config"}]},
        ],
    )
    def test_a_group_named_config_is_a_config_error(self, invoke: Invoke, server: dict[str, Any]) -> None:
        result = invoke(["config"], default={"servers": {"example": server}})

        assert result.exit_code == 2
        assert "'config' is a built-in command name" in result.stderr
