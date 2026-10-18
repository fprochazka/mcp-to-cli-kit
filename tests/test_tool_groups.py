"""A server's tools grouped into subcommands by name prefix, in every mount mode, and hidden tools.

The fixture servers have ``svc_a_list``, ``svc_a_get`` and ``svc_b_search`` next to ungrouped
tools such as ``no_args``.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import BIN, Invoke
from fixture_server import RunningServer, free_port_url

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir

TOOL_GROUPS = [
    {"prefix": "svc_a_", "group": "svc-a", "help": "Service A."},
    {"prefix": "svc_b_", "group": "svc-b"},
]

WARM_CALL = ["initialize", "notifications/initialized", "tools/call"]


def server(url: str, **extra: Any) -> dict[str, Any]:
    return {"url": url, "tool_groups": TOOL_GROUPS, **extra}


class TestRootMount:
    @pytest.fixture
    def default(self, json_server: RunningServer) -> dict[str, Any]:
        return {"servers": {"example": server(json_server.url)}}

    def test_help_lists_groups_and_ungrouped_tools(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        groups, tools = result.stdout.split("Groups:")[1].split("Tools:")
        assert "svc-a" in groups and "Service A." in groups
        assert "svc-b" in groups and "The example tools whose names start with svc_b_." in groups
        assert "\n  no-args " in tools
        assert "svc-a-list" not in result.stdout

    def test_group_help_lists_its_tools_without_the_prefix(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["svc-a", "--help"], default=default)

        assert result.exit_code == 0
        assert "\n  get " in result.stdout and "\n  list " in result.stdout
        assert "search" not in result.stdout

    def test_call_in_a_group(self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer) -> None:
        result = invoke(["svc-a", "get", "--item-id", "7"], default=default)

        assert result.exit_code == 0
        assert json_server.recorder.calls() == [{"name": "svc_a_get", "arguments": {"item_id": "7"}}]

    @pytest.mark.parametrize(
        "args", [["svc-a", "svc_a_list"], ["svc-a", "svc-a-list"], ["svc-a", "list"], ["svc_a_list"]]
    )
    def test_the_tool_name_is_an_alias(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer, args: list[str]
    ) -> None:
        assert invoke(args, default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "svc_a_list", "arguments": {}}]

    def test_ungrouped_tool_stays_at_the_top(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer
    ) -> None:
        assert invoke(["no-args"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_unknown_command_in_a_group(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["svc-a", "lst"], default=default)

        assert result.exit_code == 2
        assert "Did you mean list?" in result.stderr

    def test_warm_call_in_a_group_sends_only_the_call(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer
    ) -> None:
        invoke(["--help"], default=default)
        json_server.recorder.reset()

        assert invoke(["svc-a", "list"], default=default).exit_code == 0
        assert json_server.recorder.rpc_methods() == WARM_CALL

    def test_the_longest_prefix_wins(self, invoke: Invoke, json_server: RunningServer) -> None:
        groups = [{"prefix": "svc_", "group": "svc"}, {"prefix": "svc_a_", "group": "svc-a"}]
        default = {"servers": {"example": {"url": json_server.url, "tool_groups": groups}}}

        svc = invoke(["svc", "--help"], default=default)

        assert "\n  b-search " in svc.stdout
        assert "a-list" not in svc.stdout
        assert "\n  list " in invoke(["svc-a", "--help"], default=default).stdout


class TestOtherMounts:
    def test_inside_a_group_mount(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, mount={"group": "ex"})}}

        mount_help = invoke(["ex", "--help"], default=default)
        result = invoke(["ex", "svc-b", "search", "--query", "q"], default=default)

        assert "Groups:" in mount_help.stdout and "svc-a" in mount_help.stdout
        assert "\n  no-args " in mount_help.stdout
        assert "svc-a-list" not in mount_help.stdout
        assert result.exit_code == 0
        assert json_server.recorder.calls() == [{"name": "svc_b_search", "arguments": {"query": "q"}}]

    def test_behind_a_prefix_mount(self, invoke: Invoke, json_server: RunningServer, sse_server: RunningServer) -> None:
        default = {
            "servers": {
                "alpha": {"url": json_server.url, "hidden_tools": ["svc_a_list", "svc_a_get", "svc_b_search"]},
                "beta": server(sse_server.url, mount={"prefix": "b-"}),
            }
        }

        top = invoke(["--help"], default=default)
        result = invoke(["b-svc-a", "list"], default=default)

        assert "b-svc-a" in top.stdout.split("Groups:")[1].split("Tools:")[0]
        assert result.exit_code == 0
        assert sse_server.recorder.calls() == [{"name": "svc_a_list", "arguments": {}}]

    def test_warm_call_in_a_group_does_not_contact_the_other_server(
        self, invoke: Invoke, json_server: RunningServer, sse_server: RunningServer
    ) -> None:
        default = {
            "servers": {
                "alpha": server(json_server.url),
                "beta": {"url": sse_server.url, "mount": {"prefix": "b-"}},
            }
        }
        invoke(["--help"], default=default)
        json_server.recorder.reset()
        sse_server.recorder.reset()

        assert invoke(["svc-a", "list"], default=default).exit_code == 0
        assert invoke(["b-svc-a-list"], default=default).exit_code == 0

        assert json_server.recorder.rpc_methods() == WARM_CALL
        assert sse_server.recorder.rpc_methods() == WARM_CALL


class TestCollisions:
    def test_tool_group_named_like_a_tool_of_the_same_server(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {
            "servers": {"example": {"url": json_server.url, "tool_groups": [{"prefix": "x_", "group": "no-args"}]}}
        }

        result = invoke(["--help"], default=default)

        assert result.exit_code == 2
        assert "Server example: 'no-args' is both a tool and a tool group" in result.stderr

    def test_tool_group_named_like_another_server_s_tool(self, invoke: Invoke, json_server: RunningServer) -> None:
        beta_url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("beta", beta_url, [{"name": "inventory", "inputSchema": {}}])
        default = {
            "servers": {
                "alpha": {"url": json_server.url, "tool_groups": [{"prefix": "svc_a_", "group": "inventory"}]},
                "beta": {"url": beta_url},
            }
        }

        for args in (["inventory", "list"], ["--help"]):
            result = invoke(args, default=default)
            assert result.exit_code == 2
            assert "'inventory' is defined by more than one server: alpha, beta" in result.stderr
        assert json_server.recorder.calls() == []

    def test_tool_group_named_like_another_server_s_group(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {
            "servers": {
                "alpha": {"url": json_server.url, "tool_groups": [{"prefix": "svc_a_", "group": "beta"}]},
                "beta": {"url": free_port_url(), "mount": {"group": "beta"}},
            }
        }

        result = invoke(["--help"], default=default)

        assert result.exit_code == 2
        assert "duplicate top-level command name" in result.stderr and ": beta" in result.stderr

    def test_prefixed_tool_group_against_a_group_mount(self, invoke: Invoke) -> None:
        default = {
            "servers": {
                "alpha": {"url": free_port_url(), "mount": {"prefix": "x-"}, "tool_groups": TOOL_GROUPS},
                "beta": {"url": free_port_url(), "mount": {"group": "x-svc-a"}},
            }
        }

        result = invoke(["--help"], default=default)

        assert result.exit_code == 2
        assert "duplicate top-level command name" in result.stderr and "x-svc-a" in result.stderr


class TestHiddenTools:
    def test_hidden_tool_is_not_listed_and_cannot_be_called(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": {"url": json_server.url, "hidden_tools": ["no_args"]}}}

        listed = invoke(["--help"], default=default)
        called = invoke(["no-args"], default=default)

        assert "\n  no-args " not in listed.stdout
        assert called.exit_code == 2
        assert "No such command 'no-args'" in called.stderr
        assert json_server.recorder.calls() == []

    def test_hidden_tool_inside_a_tool_group(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, hidden_tools=["svc_a_get"])}}

        group_help = invoke(["svc-a", "--help"], default=default)
        called = invoke(["svc-a", "get", "--item-id", "1"], default=default)
        by_name = invoke(["svc_a_get", "--item-id", "1"], default=default)

        assert "\n  list " in group_help.stdout and "\n  get " not in group_help.stdout
        assert called.exit_code == 2 and by_name.exit_code == 2
        assert json_server.recorder.calls() == []

    def test_group_whose_tools_are_all_hidden_is_still_listed_but_empty(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        default = {"servers": {"example": server(json_server.url, hidden_tools=["svc_b_search"])}}

        top = invoke(["--help"], default=default)
        group_help = invoke(["svc-b", "--help"], default=default)

        assert "svc-b" in top.stdout
        assert group_help.exit_code == 0
        assert "Tools:" not in group_help.stdout
