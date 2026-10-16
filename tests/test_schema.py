"""The built-in ``schema TOOL`` command."""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import BIN, Invoke
from fixture_server import RunningServer, free_port_url

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir


def cached_schema(server: str, url: str, tool: str) -> dict[str, Any]:
    cached = SchemaCache(default_cache_dir(BIN)).load(server, url)
    assert cached is not None
    schema = next(item["inputSchema"] for item in cached.tools if item["name"] == tool)
    assert isinstance(schema, dict)
    return schema


class TestTopLevel:
    def test_prints_the_cached_schema_indented(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["schema", "search-items"])

        assert result.exit_code == 0
        schema = cached_schema("example", json_server.url, "search_items")
        assert json.loads(result.stdout) == schema
        assert result.stdout == json.dumps(schema, ensure_ascii=False, indent=2) + "\n"
        assert schema["properties"]["startTime"]["anyOf"]

    @pytest.mark.parametrize("args", [["--json", "schema", "search-items"], ["schema", "search-items", "--json"]])
    def test_machine_mode_is_compact(self, invoke: Invoke, json_server: RunningServer, args: list[str]) -> None:
        result = invoke(args)

        assert result.stdout.count("\n") == 1
        assert json.loads(result.stdout) == cached_schema("example", json_server.url, "search_items")

    def test_agent_gets_compact_json(self, invoke: Invoke) -> None:
        assert invoke(["schema", "no-args"], env={"CLAUDECODE": "1"}).stdout.count("\n") == 1

    @pytest.mark.parametrize("name", ["search-items", "search_items"])
    def test_accepts_the_tool_s_aliases(self, invoke: Invoke, name: str) -> None:
        result = invoke(["schema", name])

        assert result.exit_code == 0
        assert "query" in json.loads(result.stdout)["properties"]

    def test_warm_schema_sends_nothing(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["--help"])
        json_server.recorder.reset()

        assert invoke(["schema", "search-items"]).exit_code == 0
        assert json_server.recorder.requests == []

    def test_unknown_tool_suggests(self, invoke: Invoke) -> None:
        result = invoke(["schema", "serch-items"])

        assert result.exit_code == 2
        assert "No such tool 'serch-items'. Did you mean search-items?" in result.stderr

    def test_listed_in_the_help(self, invoke: Invoke) -> None:
        help_text = invoke(["--help"]).stdout

        assert "Commands:" in help_text
        assert "schema" in help_text.split("Commands:")[1]

    def test_own_help(self, invoke: Invoke) -> None:
        result = invoke(["schema", "--help"])

        assert result.exit_code == 0
        assert "TOOL" in result.stdout


class TestGroups:
    def test_inside_a_group_mount(self, invoke: Invoke, json_server: RunningServer, sse_server: RunningServer) -> None:
        default = {
            "servers": {"alpha": {"url": json_server.url}, "beta": {"url": sse_server.url, "mount": {"group": "beta"}}}
        }

        result = invoke(["beta", "schema", "svc-a-get"], default=default)

        assert result.exit_code == 0
        assert json.loads(result.stdout) == cached_schema("beta", sse_server.url, "svc_a_get")
        assert json_server.recorder.requests == []

    def test_inside_a_tool_group(self, invoke: Invoke, json_server: RunningServer) -> None:
        groups = [{"prefix": "svc_a_", "group": "svc-a"}]
        default = {"servers": {"example": {"url": json_server.url, "tool_groups": groups}}}

        for args in (["svc-a", "schema", "get"], ["svc-a", "schema", "svc_a_get"]):
            result = invoke(args, default=default)
            assert result.exit_code == 0
            assert "item_id" in json.loads(result.stdout)["properties"]

        help_text = invoke(["svc-a", "--help"], default=default).stdout
        assert "schema" in help_text.split("Commands:")[1]

    def test_a_group_is_not_a_tool(self, invoke: Invoke, json_server: RunningServer) -> None:
        groups = [{"prefix": "svc_a_", "group": "svc-a"}]
        default = {"servers": {"example": {"url": json_server.url, "tool_groups": groups}}}

        result = invoke(["schema", "svc-a"], default=default)

        assert result.exit_code == 2
        assert "'svc-a' is a group, not a tool. Run 'example-tool svc-a schema TOOL'." in " ".join(
            result.stderr.split()
        )


class TestNameCollisions:
    def test_a_tool_named_schema_becomes_schema_tool(self, invoke: Invoke) -> None:
        url = free_port_url()
        input_schema = {"type": "object", "properties": {"table": {"type": "string"}}}
        SchemaCache(default_cache_dir(BIN)).save("example", url, [{"name": "schema", "inputSchema": input_schema}])

        listed = invoke(["--help"], url=url)
        tool_help = invoke(["schema-tool", "--help"], url=url)
        printed = invoke(["schema", "schema-tool"], url=url)

        assert "schema-tool" in listed.stdout.split("Tools:")[1]
        assert "--table TEXT" in tool_help.stdout
        assert json.loads(printed.stdout) == input_schema

    @pytest.mark.parametrize(
        "server",
        [
            {"mount": {"group": "schema"}},
            {"tool_groups": [{"prefix": "x_", "group": "schema"}]},
        ],
    )
    def test_groups_cannot_be_named_schema(self, invoke: Invoke, server: dict[str, Any]) -> None:
        result = invoke(["--help"], default={"servers": {"example": {"url": free_port_url(), **server}}})

        assert result.exit_code == 2
        assert "'schema' is a built-in command name" in result.stderr
