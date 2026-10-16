"""Several MCP servers in one bin: group and prefix mounts.

The JSON fixture server plays "alpha" and the SSE one "beta". Both have the same tools, so
every tool name collides unless the mounts keep them apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import BIN, Invoke
from fixture_server import RAW_JSON, RunningServer, free_port_url

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir


def config(alpha: dict[str, Any], beta: dict[str, Any]) -> dict[str, Any]:
    return {"servers": {"alpha": alpha, "beta": beta}}


@pytest.fixture
def alpha(json_server: RunningServer) -> RunningServer:
    return json_server


@pytest.fixture
def beta(sse_server: RunningServer) -> RunningServer:
    return sse_server


class TestGroupMounts:
    @pytest.fixture
    def default(self, alpha: RunningServer, beta: RunningServer) -> dict[str, Any]:
        return config(
            {"url": alpha.url, "mount": {"group": "alpha", "help": "The alpha tools."}},
            {"url": beta.url, "mount": {"group": "beta"}},
        )

    def test_top_level_help_needs_no_request(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        assert "Groups:" in result.stdout
        assert "The alpha tools." in result.stdout
        assert "The tools of the beta server." in result.stdout
        assert alpha.recorder.requests == [] and beta.recorder.requests == []

    def test_a_call_contacts_only_its_server(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        result = invoke(["beta", "no-args"], default=default)

        assert result.exit_code == 0
        assert alpha.recorder.requests == []
        assert beta.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_same_tool_name_stays_distinct(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        invoke(["alpha", "search-items", "--query", "a"], default=default)
        invoke(["beta", "search-items", "--query", "b"], default=default)

        assert alpha.recorder.calls() == [{"name": "search_items", "arguments": {"query": "a"}}]
        assert beta.recorder.calls() == [{"name": "search_items", "arguments": {"query": "b"}}]

    def test_group_help_lists_its_tools(self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer) -> None:
        result = invoke(["beta", "--help"], default=default)

        assert result.exit_code == 0
        assert "search-items" in result.stdout
        assert alpha.recorder.requests == []

    def test_group_without_a_command_shows_its_help(self, invoke: Invoke, default: dict[str, Any]) -> None:
        assert "search-items" in invoke(["beta"], default=default).output

    def test_unknown_command_in_a_group(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["beta", "serch-items"], default=default)

        assert result.exit_code == 2
        assert "Did you mean search-items?" in result.stderr

    def test_output_flag_on_the_group(self, invoke: Invoke, default: dict[str, Any]) -> None:
        assert invoke(["beta", "--json", "raw-json"], default=default).stdout_bytes == RAW_JSON.encode()
        assert invoke(["beta", "--json", "raw-json", "--pretty"], default=default).stdout.startswith("{\n")

    def test_each_server_has_its_own_cache(self, invoke: Invoke, default: dict[str, Any], home: Path) -> None:
        invoke(["alpha", "--help"], default=default)
        assert (home / ".cache" / BIN / "alpha.json").exists()
        assert not (home / ".cache" / BIN / "beta.json").exists()

        invoke(["beta", "--help"], default=default)
        assert (home / ".cache" / BIN / "beta.json").exists()

    def test_refresh_touches_only_the_server_used(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        invoke(["alpha", "--help"], default=default)
        invoke(["beta", "--help"], default=default)
        alpha.recorder.reset()
        beta.recorder.reset()

        invoke(["--refresh-tools", "beta", "no-args"], default=default)

        assert alpha.recorder.requests == []
        assert beta.recorder.rpc_methods() == ["initialize", "notifications/initialized", "tools/list", "tools/call"]

    def test_each_server_gets_its_own_headers_and_the_same_client_header(
        self, invoke: Invoke, alpha: RunningServer, beta: RunningServer
    ) -> None:
        default = config(
            {"url": alpha.url, "mount": {"group": "alpha"}, "headers": {"X-Api-Key": "a"}},
            {"url": beta.url, "mount": {"group": "beta"}, "headers": {"X-Api-Key": "b"}},
        )

        invoke(["alpha", "no-args"], default=default)
        invoke(["beta", "no-args"], default=default)

        assert {request.headers["x-api-key"] for request in alpha.recorder.requests} == {"a"}
        assert {request.headers["x-api-key"] for request in beta.recorder.requests} == {"b"}
        client_headers = {
            request.headers["x-mcp-client"] for request in alpha.recorder.requests + beta.recorder.requests
        }
        assert len(client_headers) == 1

    def test_unreachable_group_server_does_not_affect_the_other(self, invoke: Invoke, alpha: RunningServer) -> None:
        default = config(
            {"url": alpha.url, "mount": {"group": "alpha"}},
            {"url": free_port_url(), "mount": {"group": "beta"}},
        )

        assert invoke(["alpha", "no-args"], default=default).exit_code == 0
        failed = invoke(["beta", "no-args"], default=default)
        assert failed.exit_code == 3
        assert "(beta)" in failed.stderr


class TestPrefixMounts:
    @pytest.fixture
    def default(self, alpha: RunningServer, beta: RunningServer) -> dict[str, Any]:
        return config({"url": alpha.url}, {"url": beta.url, "mount": {"prefix": "b-"}})

    def test_help_lists_root_and_prefixed_tools(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        assert "\n  no-args " in result.stdout
        assert "\n  b-no-args " in result.stdout

    def test_calls_route_by_prefix(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        assert invoke(["b-no-args"], default=default).exit_code == 0
        assert invoke(["no-args"], default=default).exit_code == 0

        assert alpha.recorder.calls() == [{"name": "no_args", "arguments": {}}]
        assert beta.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_prefixed_tool_name_alias(self, invoke: Invoke, default: dict[str, Any], beta: RunningServer) -> None:
        assert invoke(["b-no_args"], default=default).exit_code == 0
        assert beta.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_unprefixed_call_does_not_contact_the_prefixed_server(
        self, invoke: Invoke, default: dict[str, Any], beta: RunningServer
    ) -> None:
        result = invoke(["no-args"], default=default)

        assert result.exit_code == 0
        assert beta.recorder.requests == []

    def test_warm_calls_send_nothing_to_the_other_server(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        invoke(["--help"], default=default)
        alpha.recorder.reset()
        beta.recorder.reset()

        assert invoke(["no-args"], default=default).exit_code == 0
        assert beta.recorder.requests == []

        alpha.recorder.reset()
        assert invoke(["b-no-args"], default=default).exit_code == 0
        assert alpha.recorder.requests == []

    def test_refresh_tools_in_prefix_mode(
        self, invoke: Invoke, default: dict[str, Any], alpha: RunningServer, beta: RunningServer
    ) -> None:
        invoke(["--help"], default=default)
        alpha.recorder.reset()
        beta.recorder.reset()

        assert invoke(["--refresh-tools", "b-no-args"], default=default).exit_code == 0

        assert beta.recorder.rpc_methods() == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
        assert "tools/call" not in alpha.recorder.rpc_methods()

    def test_unavailable_server_is_listed_and_the_rest_still_shown(self, invoke: Invoke, alpha: RunningServer) -> None:
        default = config({"url": alpha.url}, {"url": free_port_url(), "mount": {"prefix": "b-"}})

        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        assert "\n  no-args " in result.stdout
        assert "Unavailable servers:" in result.stdout
        assert "beta" in result.stdout.split("Unavailable servers:")[1]
        assert "cannot connect" in " ".join(result.stdout.split())

    def test_expired_cache_of_one_server_is_used_when_it_is_down(
        self, invoke: Invoke, alpha: RunningServer, home: Path
    ) -> None:
        url = free_port_url()
        cache = SchemaCache(default_cache_dir(BIN))
        cache.save("beta", url, [{"name": "old_tool", "inputSchema": {}}])
        content = json.loads(cache.path("beta").read_text(encoding="utf-8"))
        content["fetched_at"] -= 7200
        cache.path("beta").write_text(json.dumps(content), encoding="utf-8")

        result = invoke(["--help"], default=config({"url": alpha.url}, {"url": url, "mount": {"prefix": "b-"}}))

        assert result.exit_code == 0
        assert "b-old-tool" in result.stdout and "\n  no-args " in result.stdout
        assert "cannot refresh the tool list of beta" in result.stderr


class TestCollisions:
    def test_two_root_servers_with_the_same_tool(
        self, invoke: Invoke, alpha: RunningServer, beta: RunningServer
    ) -> None:
        default = config({"url": alpha.url}, {"url": beta.url})

        for args, name in ((["no-args"], "no-args"), (["--help"], "")):
            result = invoke(args, default=default)
            assert result.exit_code == 2
            assert f"{name}' is defined by more than one server: alpha, beta" in result.stderr
        assert alpha.recorder.calls() == [] and beta.recorder.calls() == []

    def test_hidden_tool_resolves_the_collision(
        self, invoke: Invoke, alpha: RunningServer, beta: RunningServer
    ) -> None:
        default = config({"url": alpha.url}, {"url": beta.url, "hidden_tools": ["no_args"]})

        assert invoke(["no-args"], default=default).exit_code == 0
        assert alpha.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_group_mount_against_a_root_tool(self, invoke: Invoke, alpha: RunningServer, beta: RunningServer) -> None:
        default = config({"url": alpha.url}, {"url": beta.url, "mount": {"group": "no-args"}})

        result = invoke(["--help"], default=default)

        assert result.exit_code == 2
        assert "'no-args' is defined by more than one server: beta, alpha" in result.stderr

    def test_unreachable_candidate_is_reported(self, invoke: Invoke, alpha: RunningServer) -> None:
        default = config({"url": alpha.url}, {"url": free_port_url()})

        result = invoke(["no-args"], default=default)

        assert result.exit_code == 0
        assert "warning: could not check server beta for a conflicting tool" in result.stderr
        assert alpha.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_prefixed_tool_against_a_root_tool(self, invoke: Invoke, alpha: RunningServer) -> None:
        url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("beta", url, [{"name": "json", "inputSchema": {}}])
        default = config({"url": alpha.url}, {"url": url, "mount": {"prefix": "raw-"}})

        result = invoke(["raw-json"], default=default)

        assert result.exit_code == 2
        assert "'raw-json' is defined by more than one server: alpha, beta" in result.stderr
        assert alpha.recorder.calls() == []

    def test_overlapping_prefixes(self, invoke: Invoke) -> None:
        cache = SchemaCache(default_cache_dir(BIN))
        alpha_url, beta_url = free_port_url(), free_port_url()
        cache.save("alpha", alpha_url, [{"name": "b-x", "inputSchema": {}}])
        cache.save("beta", beta_url, [{"name": "x", "inputSchema": {}}])
        default = config({"url": alpha_url, "mount": {"prefix": "a-"}}, {"url": beta_url, "mount": {"prefix": "a-b-"}})

        for args in (["a-b-x"], ["--help"]):
            result = invoke(args, default=default)
            assert result.exit_code == 2
            assert "'a-b-x' is defined by more than one server: alpha, beta" in result.stderr

    def test_group_against_a_root_tool_on_a_direct_call(self, invoke: Invoke, beta: RunningServer) -> None:
        alpha_url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("alpha", alpha_url, [{"name": "no_args", "inputSchema": {}}])
        default = config({"url": alpha_url}, {"url": beta.url, "mount": {"group": "no-args"}})

        result = invoke(["no-args", "search-items", "--query", "q"], default=default)

        assert result.exit_code == 2
        assert "'no-args' is defined by more than one server: beta, alpha" in result.stderr
        assert beta.recorder.requests == []

    def test_every_unavailable_server_is_reported(self, invoke: Invoke) -> None:
        result = invoke(["--help"], default=config({"url": free_port_url()}, {"url": free_port_url()}))

        assert result.exit_code == 3
        assert "(alpha)" in result.stderr and "(beta)" in result.stderr

    def test_disjoint_tools_do_not_collide(self, invoke: Invoke, alpha: RunningServer, beta: RunningServer) -> None:
        names = ["no_args", "search_items", "fail", "big", "multi", "plain_text", "raw_json", "structured"]
        names += ["wrapped_text", "slow", "error_as_text", "svc_a_list", "svc_a_get", "svc_b_search"]
        default = config(
            {"url": alpha.url, "hidden_tools": [n for n in names if n != "no_args"]},
            {"url": beta.url, "hidden_tools": [n for n in names if n != "raw_json"]},
        )

        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        assert "no-args" in result.stdout and "raw-json" in result.stdout
