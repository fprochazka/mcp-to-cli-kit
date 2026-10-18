"""Prefixes stripped from a server's tool names, in every mount mode, with tool groups and with collisions.

The fixture servers have ``vendor-x_list``, ``vendor-x_no_args`` (which clashes with ``no_args``
once stripped) and ``vendor_x-svc_c_find`` next to the other tools.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from conftest import BIN, Invoke
from fixture_server import RunningServer, free_port_url

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir
from mcp_to_cli_kit.cli import ServerTools
from mcp_to_cli_kit.config import ConfigError, ToolGroup, load_config

STRIP = ["vendor-x_", "vendor_x-"]

WARM_CALL = ["initialize", "notifications/initialized", "tools/call"]

CLASH_WARNING = (
    "Not stripping the prefix from the tool 'vendor-x_no_args' of example: 'no_args' would clash with 'no_args'"
)


def server(url: str, **extra: Any) -> dict[str, Any]:
    return {"url": url, "strip_prefixes": STRIP, **extra}


def tools_section(stdout: str) -> str:
    return stdout.split("Tools:")[1].split("Commands:")[0]


class TestRootMount:
    @pytest.fixture
    def default(self, json_server: RunningServer) -> dict[str, Any]:
        return {"servers": {"example": server(json_server.url)}}

    def test_help_lists_the_stripped_names(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        tools = tools_section(result.stdout)
        assert "\n  list " in tools and "\n  svc-c-find " in tools
        assert "vendor-x-list" not in tools and "vendor-x-svc-c-find" not in tools

    def test_call_by_the_stripped_name(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer
    ) -> None:
        result = invoke(["list"], default=default)

        assert result.exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor-x_list", "arguments": {}}]

    @pytest.mark.parametrize("name", ["vendor-x_list", "vendor-x-list", "vendor_x_list"])
    def test_the_original_name_is_an_alias(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer, name: str
    ) -> None:
        assert invoke([name], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor-x_list", "arguments": {}}]

    def test_the_second_prefix_is_stripped_too(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer
    ) -> None:
        assert invoke(["svc-c-find", "--query", "q"], default=default).exit_code == 0
        assert invoke(["vendor_x-svc_c_find", "--query", "r"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [
            {"name": "vendor_x-svc_c_find", "arguments": {"query": "q"}},
            {"name": "vendor_x-svc_c_find", "arguments": {"query": "r"}},
        ]

    def test_schema_by_the_stripped_name(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["--pretty", "schema", "svc-c-find"], default=default)

        assert result.exit_code == 0
        assert '"query"' in result.stdout

    @pytest.mark.parametrize("name", ["list", "vendor-x_list"])
    def test_warm_call_sends_only_the_call(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer, name: str
    ) -> None:
        invoke(["--help"], default=default)
        json_server.recorder.reset()

        assert invoke([name], default=default).exit_code == 0
        assert json_server.recorder.rpc_methods() == WARM_CALL

    def test_without_strip_prefixes_the_names_are_kept(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["--help"], default={"servers": {"example": {"url": json_server.url}}})

        tools = tools_section(result.stdout)
        assert "\n  vendor-x-list " in tools and "\n  list " not in tools

    def test_the_first_matching_prefix_wins(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, strip_prefixes=["vendor", "vendor-x_"])}}

        result = invoke(["x-list"], default=default)

        assert result.exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor-x_list", "arguments": {}}]

    def test_a_name_equal_to_the_prefix_is_kept(self, invoke: Invoke) -> None:
        url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("example", url, [{"name": "vendor-x_", "inputSchema": {}}])

        result = invoke(["--help"], default={"servers": {"example": server(url)}})

        assert result.exit_code == 0
        assert "\n  vendor-x\n" in tools_section(result.stdout)


class TestCollisions:
    @pytest.fixture
    def default(self, json_server: RunningServer) -> dict[str, Any]:
        return {"servers": {"example": server(json_server.url)}}

    def test_a_clash_keeps_the_original_name_and_warns(self, invoke: Invoke, default: dict[str, Any]) -> None:
        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        tools = tools_section(result.stdout)
        assert "\n  no-args " in tools and "\n  vendor-x-no-args " in tools
        assert CLASH_WARNING in result.stderr

    def test_both_clashing_tools_stay_callable(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer
    ) -> None:
        assert invoke(["no-args"], default=default).exit_code == 0
        assert invoke(["vendor-x-no-args"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [
            {"name": "no_args", "arguments": {}},
            {"name": "vendor-x_no_args", "arguments": {}},
        ]

    def test_hiding_one_side_resolves_the_clash(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, hidden_tools=["no_args"])}}

        result = invoke(["no-args"], default=default)

        assert result.exit_code == 0
        assert "Not stripping" not in result.stderr
        assert json_server.recorder.calls() == [{"name": "vendor-x_no_args", "arguments": {}}]

    def test_two_prefixes_stripping_to_the_same_name(self, invoke: Invoke) -> None:
        url = free_port_url()
        tools = [{"name": "vendor-x_ping", "inputSchema": {}}, {"name": "vendor_x-ping", "inputSchema": {}}]
        SchemaCache(default_cache_dir(BIN)).save("example", url, tools)

        result = invoke(["--help"], default={"servers": {"example": server(url)}})

        assert result.exit_code == 0
        assert "\n  ping " not in tools_section(result.stdout)
        assert "'vendor-x_ping'" in result.stderr and "'vendor_x-ping'" in result.stderr

    def test_a_clash_with_another_server_is_a_config_error(self, invoke: Invoke, json_server: RunningServer) -> None:
        beta_url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("beta", beta_url, [{"name": "list", "inputSchema": {}}])
        default = {"servers": {"alpha": server(json_server.url), "beta": {"url": beta_url}}}

        for args in (["list"], ["--help"]):
            result = invoke(args, default=default)
            assert result.exit_code == 2
            assert "'list' is defined by more than one server: alpha, beta" in result.stderr
        assert json_server.recorder.calls() == []

    def test_a_clash_with_a_tool_group_name_keeps_the_original_name(self, invoke: Invoke) -> None:
        url = free_port_url()
        SchemaCache(default_cache_dir(BIN)).save("example", url, [{"name": "vendor-x_svc-c", "inputSchema": {}}])
        groups = [{"prefix": "zz_", "group": "svc-c"}]

        result = invoke(["--help"], default={"servers": {"example": server(url, tool_groups=groups)}})

        assert result.exit_code == 0
        assert "\n  vendor-x-svc-c\n" in tools_section(result.stdout)
        assert "'svc-c' would clash with the tool group svc-c" in result.stderr

    def test_a_kept_name_that_clashes_again_is_resolved_in_any_order(self) -> None:
        tools: list[dict[str, Any]] = [{"name": name, "inputSchema": {}} for name in ("p_a", "a", "p_p-a")]

        mappings = {
            tuple(sorted((command, entry.name) for command, entry in index.entries.items()))
            for order in itertools.permutations(tools)
            for index in [ServerTools.build("example", list(order), [], [], ["p_"])]
        }

        assert mappings == {(("a", "a"), ("p-a", "p_a"), ("p-p-a", "p_p-a"))}

    def test_the_own_name_finds_its_tool_in_any_order(self) -> None:
        tools: list[dict[str, Any]] = [{"name": name, "inputSchema": {}} for name in ("p_a", "p-a")]

        for order in itertools.permutations(tools):
            index = ServerTools.build("example", list(order), [], [], ["p_"])
            found = {name: entry.name for name in ("p_a", "p-a", "a") if (entry := index.find(name)) is not None}
            assert found == {"p_a": "p_a", "p-a": "p-a", "a": "p_a"}

    def test_the_own_name_finds_its_grouped_tool_in_any_order(self) -> None:
        tools: list[dict[str, Any]] = [{"name": name, "inputSchema": {}} for name in ("p_g_x", "g-x")]
        groups = [ToolGroup(prefix="g_", group="g")]

        for order in itertools.permutations(tools):
            index = ServerTools.build("example", list(order), [], groups, ["p_"])
            assert (entry := index.find("p_g_x")) is not None and entry.name == "p_g_x"
            assert (entry := index.find("g-x")) is not None and entry.name == "g-x"

    def test_a_stripped_name_equal_to_schema(self, invoke: Invoke) -> None:
        url = free_port_url()
        tool = {"name": "vendor-x_schema", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
        SchemaCache(default_cache_dir(BIN)).save("example", url, [tool])
        default = {"servers": {"example": server(url)}}

        listed = invoke(["--help"], default=default)
        by_command = invoke(["--json", "schema", "schema-tool"], default=default)
        by_name = invoke(["--json", "schema", "vendor-x_schema"], default=default)

        assert "\n  schema-tool\n" in tools_section(listed.stdout)
        assert by_command.exit_code == 0 and by_name.exit_code == 0
        assert '"q"' in by_command.stdout and by_command.stdout == by_name.stdout


class TestHiddenTools:
    def test_hidden_by_its_own_name(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, hidden_tools=["vendor-x_list"])}}

        listed = invoke(["--help"], default=default)
        called = invoke(["vendor-x_list"], default=default)

        assert "\n  list " not in tools_section(listed.stdout)
        assert called.exit_code == 2
        assert json_server.recorder.calls() == []

    def test_the_stripped_name_does_not_hide_it(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, hidden_tools=["list"])}}

        assert invoke(["list"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor-x_list", "arguments": {}}]


class TestToolGroups:
    GROUPS = [{"prefix": "svc_c_", "group": "svc-c", "help": "Service C."}]

    @pytest.fixture
    def default(self, json_server: RunningServer) -> dict[str, Any]:
        return {"servers": {"example": server(json_server.url, tool_groups=self.GROUPS)}}

    def test_groups_match_the_stripped_name(self, invoke: Invoke, default: dict[str, Any]) -> None:
        top = invoke(["--help"], default=default)
        group = invoke(["svc-c", "--help"], default=default)

        assert "svc-c" in top.stdout.split("Groups:")[1].split("Tools:")[0]
        assert "svc-c-find" not in top.stdout
        assert "\n  find " in group.stdout

    @pytest.mark.parametrize(
        "args",
        [
            ["svc-c", "find"],
            ["svc-c", "svc_c_find"],
            ["svc-c", "vendor_x-svc_c_find"],
            ["svc_c_find"],
            ["vendor_x-svc_c_find"],
            ["vendor-x-svc-c-find"],
        ],
    )
    def test_every_name_reaches_the_tool(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer, args: list[str]
    ) -> None:
        assert invoke([*args, "--query", "q"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor_x-svc_c_find", "arguments": {"query": "q"}}]

    def test_a_group_prefix_does_not_match_the_original_name(self, invoke: Invoke, json_server: RunningServer) -> None:
        groups = [{"prefix": "vendor_x-svc_c_", "group": "svc-c"}]
        default = {"servers": {"example": server(json_server.url, tool_groups=groups)}}

        result = invoke(["svc-c", "--help"], default=default)

        assert result.exit_code == 0
        assert "Tools:" not in result.stdout

    def test_warm_call_in_a_group_sends_only_the_call(
        self, invoke: Invoke, default: dict[str, Any], json_server: RunningServer
    ) -> None:
        invoke(["--help"], default=default)
        json_server.recorder.reset()

        assert invoke(["svc-c", "find", "--query", "q"], default=default).exit_code == 0
        assert json_server.recorder.rpc_methods() == WARM_CALL


class TestOtherMounts:
    def test_group_mount(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, mount={"group": "ex"})}}

        listed = invoke(["ex", "--help"], default=default)

        assert "\n  list " in tools_section(listed.stdout)
        assert invoke(["ex", "list"], default=default).exit_code == 0
        assert invoke(["ex", "vendor-x_list"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor-x_list", "arguments": {}}] * 2

    def test_prefix_mount(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": server(json_server.url, mount={"prefix": "v-"})}}

        listed = invoke(["--help"], default=default)

        assert "\n  v-list " in tools_section(listed.stdout)
        assert invoke(["v-list"], default=default).exit_code == 0
        assert invoke(["v-vendor-x_list"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor-x_list", "arguments": {}}] * 2

    def test_prefix_mount_with_tool_groups(self, invoke: Invoke, json_server: RunningServer) -> None:
        groups = [{"prefix": "svc_c_", "group": "svc-c"}]
        default = {"servers": {"example": server(json_server.url, mount={"prefix": "v-"}, tool_groups=groups)}}

        assert invoke(["v-svc-c", "find", "--query", "q"], default=default).exit_code == 0
        assert json_server.recorder.calls() == [{"name": "vendor_x-svc_c_find", "arguments": {"query": "q"}}]

    @pytest.mark.parametrize(
        ("args", "name"),
        [(["ex", "list"], "vendor-x_list"), (["ex", "vendor-x_list"], "vendor-x_list"), (["v-list"], "vendor-x_list")],
    )
    def test_warm_call_sends_only_the_call(
        self, invoke: Invoke, json_server: RunningServer, args: list[str], name: str
    ) -> None:
        mount = {"group": "ex"} if args[0] == "ex" else {"prefix": "v-"}
        default = {"servers": {"example": server(json_server.url, mount=mount)}}
        invoke([args[0], "--help"] if args[0] == "ex" else ["--help"], default=default)
        json_server.recorder.reset()

        assert invoke(args, default=default).exit_code == 0
        assert json_server.recorder.rpc_methods() == WARM_CALL
        assert json_server.recorder.calls() == [{"name": name, "arguments": {}}]


class TestConfig:
    def load(self, server_config: dict[str, Any], **env: str) -> Any:
        default = {"servers": {"alpha": {"url": "https://mcp.example.com/mcp", **server_config}}}
        return load_config(BIN, default=default, environ=env)

    def test_defaults_to_none(self) -> None:
        assert self.load({}).config.servers["alpha"].strip_prefixes == []

    def test_from_the_environment_as_json(self) -> None:
        loaded = self.load({}, EXAMPLE_TOOL_SERVERS__ALPHA__STRIP_PREFIXES='["a_", "b-"]')

        assert loaded.config.servers["alpha"].strip_prefixes == ["a_", "b-"]

    def test_from_the_file(self, home: Path) -> None:
        path = home / "config.yaml"
        path.write_text("servers:\n  alpha:\n    strip_prefixes: [c_]\n", encoding="utf-8")

        loaded = load_config(
            BIN, default={"servers": {"alpha": {"url": "https://mcp.example.com/mcp"}}}, config_path=path, environ={}
        )

        assert loaded.config.servers["alpha"].strip_prefixes == ["c_"]

    @pytest.mark.parametrize(
        ("value", "message"), [(["a_", "a_"], "duplicate strip_prefixes entry"), ([""], "at least 1 character")]
    )
    def test_invalid(self, value: list[str], message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            self.load({"strip_prefixes": value})

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ('["a_", "a_"]', "duplicate strip_prefixes entry"),
            ('[""]', "at least 1 character"),
            ("a_", "strip_prefixes"),
            ("[a_", "STRIP_PREFIXES is not valid JSON"),
        ],
    )
    def test_invalid_from_the_environment(self, value: str, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            self.load({}, EXAMPLE_TOOL_SERVERS__ALPHA__STRIP_PREFIXES=value)
