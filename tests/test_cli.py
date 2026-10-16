"""End-to-end tests: the CLI against a real MCP server on a localhost port."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from conftest import BIN, Invoke
from fixture_server import CZECH, RAW_JSON, RunningServer, free_port_url

from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir
from mcp_to_cli_kit.cli import Runtime, build_cli


def sent_arguments(server: RunningServer) -> dict[str, object]:
    calls = server.recorder.calls()
    assert len(calls) == 1, calls
    arguments = calls[0]["arguments"]
    assert isinstance(arguments, dict)
    return arguments


class TestDiscovery:
    def test_help_lists_every_tool(self, invoke: Invoke) -> None:
        result = invoke(["--help"])

        assert result.exit_code == 0
        for command in ("search-items", "raw-json", "multi", "fail", "no-args"):
            assert command in result.stdout

    def test_no_arguments_shows_the_help(self, invoke: Invoke) -> None:
        result = invoke([])

        assert "search-items" in result.output

    def test_tool_help_is_complete(self, invoke: Invoke) -> None:
        result = invoke(["search-items", "--help"])

        assert result.exit_code == 0
        assert "The END of the description." in result.stdout
        assert "        - newest\n" not in result.stdout, "the docstring indentation is removed"
        assert "- newest\n" in result.stdout, "list lines keep their own lines"
        assert "[alpha|beta|gamma]" in result.stdout
        assert "--start-time TEXT" in result.stdout
        assert "--startTime" not in result.stdout, "aliases are not shown"
        assert "--verbose / --no-verbose" in result.stdout
        flat = " ".join(result.stdout.split())
        assert "Minimum: 1." in flat
        assert "Server default: 10." in flat
        assert "[required]" in result.stdout

    def test_output_schema_is_announced(self, invoke: Invoke) -> None:
        assert "Returns JSON" in invoke(["structured", "--help"]).stdout
        assert "Returns JSON" not in invoke(["raw-json", "--help"]).stdout

    def test_tool_name_as_alias(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["no_args"])

        assert result.exit_code == 0
        assert json_server.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_unknown_command_suggests(self, invoke: Invoke) -> None:
        result = invoke(["search-itmes"])

        assert result.exit_code == 2
        assert "Did you mean search-items?" in result.stderr

    def test_version(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["--version"])

        assert result.exit_code == 0
        assert result.stdout.startswith(f"{BIN} 0.1.0 (mcp-to-cli-kit ")
        assert json_server.recorder.requests == []


class TestArgumentsOnTheWire:
    def test_camel_case_name_is_sent_unchanged_by_any_spelling(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        for flag in ("--start-time", "--startTime", "--start_time"):
            json_server.recorder.reset()
            assert invoke(["search-items", "--query", "q", flag, "2026-01-01"]).exit_code == 0
            assert sent_arguments(json_server) == {"query": "q", "startTime": "2026-01-01"}

    def test_defaults_are_not_sent(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q"])

        assert sent_arguments(json_server) == {"query": "q"}

    def test_false_can_be_sent(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--no-verbose", "--strict"])

        assert sent_arguments(json_server) == {"query": "q", "verbose": False, "strict": True}

    def test_nullable_boolean_false(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--no-strict"])

        assert sent_arguments(json_server) == {"query": "q", "strict": False}

    def test_repeated_array_option(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--tags", "a,b", "--tags", "c"])

        assert sent_arguments(json_server)["tags"] == ["a,b", "c"]

    def test_array_as_json_literal(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--tags", '["a", "b"]'])

        assert sent_arguments(json_server)["tags"] == ["a", "b"]

    def test_array_of_objects(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--points", '{"x": 1, "y": 2}', "--points", '{"x": 3, "y": 4}'])

        assert sent_arguments(json_server)["points"] == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    def test_object_from_a_ref(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--filter", '{"field": "f", "value": 1}'])

        assert sent_arguments(json_server)["filter"] == {"field": "f", "value": 1}

    def test_all_of_object(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--combined", '{"a": 5}'])

        assert sent_arguments(json_server)["combined"] == {"a": 5}

    def test_all_of_required_is_checked(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["search-items", "--query", "q", "--combined", '{"b": 5}'])

        assert result.exit_code == 2
        assert "--combined: 'a' is a required property" in result.stderr
        assert json_server.recorder.calls() == []

    def test_typed_values(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--limit", "5", "--ratio", "0.5", "--kind", "beta"])

        assert sent_arguments(json_server) == {"query": "q", "limit": 5, "ratio": 0.5, "kind": "beta"}

    def test_parameter_named_json(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--query", "q", "--arg-json", "value"])

        assert sent_arguments(json_server) == {"query": "q", "json": "value"}

    def test_input_json_merges_with_options(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--input-json", '{"query": "q", "strict": false}', "--limit", "3"])

        assert sent_arguments(json_server) == {"query": "q", "strict": False, "limit": 3}

    def test_input_json_from_stdin(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["search-items", "--input-json", "-"], input='{"query": "from stdin"}')

        assert sent_arguments(json_server) == {"query": "from stdin"}

    def test_input_json_conflict(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["search-items", "--input-json", '{"query": "a"}', "--query", "b"])

        assert result.exit_code == 2
        assert "given both" in result.stderr
        assert json_server.recorder.calls() == []

    def test_arg_file(self, invoke: Invoke, json_server: RunningServer, tmp_path: Path) -> None:
        body = tmp_path / "body.md"
        body.write_text("# Title\n\nLong *markdown* body, ž.\n", encoding="utf-8")

        invoke(["search-items", "--arg-file", f"query={body}", "--arg-file", "tags=-"], input='["x", "y"]')

        assert sent_arguments(json_server) == {"query": "# Title\n\nLong *markdown* body, ž.\n", "tags": ["x", "y"]}

    def test_arg_file_by_option_spelling(self, invoke: Invoke, json_server: RunningServer, tmp_path: Path) -> None:
        value = tmp_path / "start.txt"
        value.write_text("2026", encoding="utf-8")

        invoke(["search-items", "--query", "q", "--arg-file", f"--start-time={value}"])

        assert sent_arguments(json_server) == {"query": "q", "startTime": "2026"}


class TestUsageErrors:
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--limit", "abc"], "'abc' is not an integer"),
            (["--limit", "0"], "--limit: 0 is less than the minimum of 1"),
            (["--kind", "delta"], "'delta' is not one of"),
            (["--filter", "{not json"], "is not a JSON object"),
            (["--options", "[1]"], "expected a JSON object, got list"),
            (["--points", "[{bad"], "is not a valid JSON array"),
            (["--ratio", "nan"], "is not a finite number"),
        ],
    )
    def test_bad_values_exit_2_and_send_nothing(
        self, invoke: Invoke, json_server: RunningServer, args: list[str], message: str
    ) -> None:
        result = invoke(["search-items", "--query", "q", *args])

        assert result.exit_code == 2
        assert message in result.stderr
        assert json_server.recorder.calls() == []

    def test_missing_required_lists_the_parameters(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["search-items", "--limit", "3"])

        assert result.exit_code == 2
        assert "missing required --query ('query')" in result.stderr
        assert "Parameters:" in result.stderr
        assert "--kind [alpha|beta|gamma]" in result.stderr
        assert json_server.recorder.calls() == []

    def test_unknown_option_lists_the_parameters_after_the_hint(self, invoke: Invoke) -> None:
        result = invoke(["search-items", "--qeury", "q"])

        assert result.exit_code == 2
        stderr = result.stderr
        assert "No such option '--qeury'" in stderr
        assert "Did you mean '--query'?" in stderr
        assert stderr.index("Did you mean") < stderr.index("Parameters:")
        assert "--query TEXT" in stderr and "[required]" in stderr
        assert "--verbose / --no-verbose" in stderr

    def test_json_and_pretty_exclude_each_other(self, invoke: Invoke) -> None:
        result = invoke(["raw-json", "--json", "--pretty"])

        assert result.exit_code == 2
        assert "exclude each other" in result.stderr


class TestOutput:
    @pytest.mark.parametrize("server_name", ["json_server", "sse_server"])
    def test_agent_output_is_byte_identical(
        self, invoke: Invoke, request: pytest.FixtureRequest, server_name: str
    ) -> None:
        server: RunningServer = request.getfixturevalue(server_name)

        result = invoke(["raw-json"], url=server.url, env={"CLAUDECODE": "1"})

        assert result.exit_code == 0
        assert result.stdout_bytes == RAW_JSON.encode("utf-8")

    def test_json_flag_forces_machine_mode_at_both_levels(self, invoke: Invoke) -> None:
        assert invoke(["--json", "raw-json"]).stdout_bytes == RAW_JSON.encode()
        assert invoke(["raw-json", "--json"]).stdout_bytes == RAW_JSON.encode()

    def test_pretty_flag_wins_over_the_agent(self, invoke: Invoke) -> None:
        result = invoke(["raw-json", "--pretty"], env={"CLAUDECODE": "1"})

        assert result.stdout == json.dumps(json.loads(RAW_JSON), indent=2, ensure_ascii=False) + "\n"

    def test_tool_flag_wins_over_the_group_flag(self, invoke: Invoke) -> None:
        assert invoke(["--json", "raw-json", "--pretty"]).stdout.startswith("{\n")

    def test_human_output_is_valid_indented_json_with_readable_text(self, invoke: Invoke) -> None:
        result = invoke(["structured"])

        assert json.loads(result.stdout) == {"name": CZECH, "count": 3}
        assert CZECH in result.stdout
        assert result.stdout.startswith('{\n  "name"')

    def test_long_lines_are_never_wrapped(self, invoke: Invoke) -> None:
        result = invoke(["big", "--size", "5000"], env={"COLUMNS": "40"})

        lines = result.stdout.splitlines()
        assert max(len(line) for line in lines) > 1000
        assert json.loads(result.stdout)

    def test_plain_text_is_printed_as_is(self, invoke: Invoke) -> None:
        assert invoke(["plain-text"]).stdout == CZECH + "\nsecond line\n"
        assert invoke(["plain-text"], env={"CLAUDECODE": "1"}).stdout == CZECH + "\nsecond line"

    def test_several_items_become_an_array(self, invoke: Invoke) -> None:
        machine = invoke(["multi"], env={"CLAUDECODE": "1"})
        pretty = invoke(["multi"])

        expected = [{"total": 2}, "Total results: 2, showing 2.", [1, "ž"]]
        assert machine.stdout == '[{"total": 2},"Total results: 2, showing 2.",[1, "ž"]]'
        assert json.loads(machine.stdout) == expected
        assert json.loads(pretty.stdout) == expected

    def test_wrapped_text_prints_the_text(self, invoke: Invoke) -> None:
        assert invoke(["wrapped-text"], env={"CLAUDECODE": "1"}).stdout == "just text"

    @pytest.mark.parametrize("server_name", ["json_server", "sse_server"])
    def test_result_over_one_mebibyte(self, invoke: Invoke, request: pytest.FixtureRequest, server_name: str) -> None:
        server: RunningServer = request.getfixturevalue(server_name)

        result = invoke(["big", "--size", str(3 * 1024 * 1024)], url=server.url, env={"CLAUDECODE": "1"})

        assert result.exit_code == 0
        assert len(result.stdout_bytes) > 1024 * 1024
        assert json.loads(result.stdout)["data"].startswith(CZECH)


class TestErrors:
    def test_tool_error_exits_1_on_stderr(self, invoke: Invoke) -> None:
        result = invoke(["fail", "--message", "it broke"])

        assert result.exit_code == 1
        assert result.stdout == ""
        assert "it broke" in result.stderr

    def test_classifier_turns_a_result_into_an_error(self, invoke: Invoke) -> None:
        seen = []

        def classifier(server: str, tool: str, result: object) -> bool:
            seen.append((server, tool))
            return True

        result = invoke(["error-as-text"], classifier=classifier)

        assert result.exit_code == 1
        assert "403 Forbidden" in result.stderr
        assert seen == [("example", "error_as_text")]

    def test_classifier_is_not_asked_about_real_errors(self, invoke: Invoke) -> None:
        def classifier(server: str, tool: str, result: object) -> bool:
            raise AssertionError("not called for isError results")

        assert invoke(["fail", "--message", "x"], classifier=classifier).exit_code == 1

    def test_unreachable_server_exits_3_with_one_line(self, invoke: Invoke) -> None:
        url = free_port_url()

        result = invoke(["--help"], url=url)

        assert result.exit_code == 3
        assert "cannot list the tools of http://127.0.0.1" in result.stderr
        assert "cannot connect" in result.stderr
        assert "Traceback" not in result.stderr
        assert "No such command" not in result.stderr

    def test_http_status_is_reported_with_a_redacted_url(self, invoke: Invoke, json_server: RunningServer) -> None:
        json_server.recorder.fail_with = [500]
        url = json_server.url.replace("http://", "http://user:secret@") + "?token=secret"

        result = invoke(["--help"], url=url)

        assert result.exit_code == 3
        assert "HTTP 500 Internal Server Error from http://127.0.0.1" in result.stderr
        assert "secret" not in result.stderr
        assert "Traceback" not in result.stderr
        assert "post_writer" not in result.stderr
        assert len(result.stderr.strip().splitlines()) == 1

    def test_unknown_tool_drops_the_cache(self, invoke: Invoke, json_server: RunningServer) -> None:
        cache = SchemaCache(default_cache_dir(BIN))
        cache.save("example", json_server.url, [{"name": "ghost", "inputSchema": {"type": "object"}}])

        result = invoke(["ghost"])

        assert result.exit_code == 1
        assert "Unknown tool" in result.stderr
        assert not cache.path("example").exists()

    def test_missing_config_exits_2_and_names_the_path(self, home: Path) -> None:
        result = CliRunner().invoke(build_cli(BIN), ["--help"])

        assert result.exit_code == 2
        assert str(home / ".config" / BIN / "config.yaml") in result.stderr


class TestSessions:
    def test_cold_call_uses_one_session(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["no-args"])

        assert json_server.recorder.rpc_methods() == [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
        ]

    def test_warm_call_skips_the_tool_list(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["--help"])
        json_server.recorder.reset()

        invoke(["no-args"])

        assert json_server.recorder.rpc_methods() == ["initialize", "notifications/initialized", "tools/call"]

    def test_warm_help_sends_nothing(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["--help"])
        json_server.recorder.reset()

        invoke(["--help"])
        invoke(["search-items", "--help"])

        assert json_server.recorder.requests == []

    def test_refresh_tools(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["--help"])
        json_server.recorder.reset()

        invoke(["--refresh-tools", "no-args"])

        assert json_server.recorder.rpc_methods() == [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
        ]

    def test_cache_file_is_per_server_and_holds_no_url(self, invoke: Invoke, home: Path) -> None:
        invoke(["--help"])

        path = home / ".cache" / BIN / "example.json"
        content = json.loads(path.read_text(encoding="utf-8"))
        assert {tool["name"] for tool in content["tools"]} >= {"search_items", "no_args"}
        assert "127.0.0.1" not in path.read_text(encoding="utf-8")

    def test_audit_header(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["no-args"])

        headers = {request.headers.get("x-mcp-client") for request in json_server.recorder.requests}
        assert len(headers) == 1
        value = headers.pop()
        assert value is not None
        assert value.startswith(f"{BIN}/0.1.0 (mcp-to-cli-kit ")

    def test_client_info_names_the_bin(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["no-args"], distribution="click")

        initialize = next(request for request in json_server.recorder.requests if request.rpc_method == "initialize")
        assert initialize.params is not None
        assert initialize.params["clientInfo"] == {"name": BIN, "version": metadata.version("click")}

    def test_audit_header_with_the_consumer_distribution(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["no-args"], distribution="click")

        version = metadata.version("click")
        assert json_server.recorder.requests[0].headers["x-mcp-client"].startswith(f"{BIN}/{version} (")

    def test_config_header_is_sent(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": {"url": json_server.url, "headers": {"X-Api-Key": "k"}}}}

        invoke(["no-args"], default=default)

        assert all(request.headers.get("x-api-key") == "k" for request in json_server.recorder.requests)


class TestStaleCache:
    def write_expired_cache(self, url: str) -> SchemaCache:
        cache = SchemaCache(default_cache_dir(BIN))
        cache.save("example", url, [{"name": "old_tool", "description": "From the cache.", "inputSchema": {}}])
        content = json.loads(cache.path("example").read_text(encoding="utf-8"))
        content["fetched_at"] = time.time() - 7200
        cache.path("example").write_text(json.dumps(content), encoding="utf-8")
        return cache

    def test_expired_cache_is_used_when_the_server_is_down(self, invoke: Invoke) -> None:
        url = free_port_url()
        self.write_expired_cache(url)

        result = invoke(["--help"], url=url)

        assert result.exit_code == 0
        assert "old-tool" in result.stdout
        assert "warning: cannot refresh the tool list" in result.stderr

    def test_expired_cache_is_refreshed_when_the_server_is_up(self, invoke: Invoke, json_server: RunningServer) -> None:
        self.write_expired_cache(json_server.url)

        result = invoke(["--help"])

        assert "old-tool" not in result.stdout
        assert "search-items" in result.stdout
        assert result.stderr == ""

    def test_unknown_command_in_a_fresh_cache_fetches_again(self, invoke: Invoke, json_server: RunningServer) -> None:
        SchemaCache(default_cache_dir(BIN)).save("example", json_server.url, [{"name": "other", "inputSchema": {}}])

        result = invoke(["no-args"])

        assert result.exit_code == 0
        assert json_server.recorder.rpc_methods() == [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
        ]


class TestRetries:
    def test_tool_list_is_retried_on_503(self, invoke: Invoke, json_server: RunningServer) -> None:
        json_server.recorder.fail_with = [503, 503]

        result = invoke(["no-args"])

        assert result.exit_code == 0
        assert json_server.recorder.rpc_methods().count("initialize") == 3

    def test_retries_are_limited(self, invoke: Invoke, json_server: RunningServer) -> None:
        json_server.recorder.fail_with = [502, 502, 502]

        result = invoke(["--help"])

        assert result.exit_code == 3
        assert "HTTP 502 Bad Gateway" in result.stderr
        assert "(after 3 attempts)" in result.stderr

    def test_500_is_not_retried(self, invoke: Invoke, json_server: RunningServer) -> None:
        json_server.recorder.fail_with = [500]

        assert invoke(["--help"]).exit_code == 3
        assert json_server.recorder.rpc_methods() == ["initialize"]

    def test_tool_call_is_not_retried_after_it_was_sent(self, invoke: Invoke, json_server: RunningServer) -> None:
        invoke(["--help"])
        json_server.recorder.reset()
        json_server.recorder.fail_with = [503]
        json_server.recorder.fail_method = "tools/call"

        result = invoke(["no-args"])

        assert result.exit_code == 3
        assert "HTTP 503" in result.stderr
        assert json_server.recorder.rpc_methods().count("tools/call") == 1

    def test_tool_call_session_is_retried_when_initialize_fails(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        invoke(["--help"])
        json_server.recorder.reset()
        json_server.recorder.fail_with = [503]
        json_server.recorder.fail_method = "initialize"

        result = invoke(["no-args"])

        assert result.exit_code == 0
        assert json_server.recorder.rpc_methods().count("tools/call") == 1

    def test_connect_errors_are_retried(self, invoke: Invoke) -> None:
        result = invoke(["-v", "--help"], url=free_port_url())

        assert result.exit_code == 3
        assert "retry 2" in result.stderr
        assert "(after 3 attempts)" in result.stderr


class TestConfigAndLogging:
    def test_config_option_works_before_and_after_help(
        self, json_server: RunningServer, tmp_path: Path, home: Path
    ) -> None:
        config = tmp_path / "other.yaml"
        config.write_text(yaml.safe_dump({"servers": {"other": {"url": json_server.url}}}), encoding="utf-8")
        cli = build_cli(BIN)

        for args in (["--config", str(config), "--help"], ["--help", "--config", str(config)]):
            result = CliRunner().invoke(cli, args)
            assert result.exit_code == 0, result.output
            assert "search-items" in result.stdout

        assert (home / ".cache" / BIN / "other.json").exists()

    def test_verbose_shows_ignored_environment_variables(self, invoke: Invoke) -> None:
        result = invoke(["-v", "no-args"], env={"EXAMPLE_TOOL_COLOR": "red"})

        assert "Ignoring EXAMPLE_TOOL_COLOR" in result.stderr

    def test_quiet_without_verbose(self, invoke: Invoke) -> None:
        result = invoke(["no-args"], env={"EXAMPLE_TOOL_COLOR": "red"})

        assert result.stderr == ""


class TestBrokenResponses:
    def config(self, server: RunningServer, read_timeout: float = 100) -> dict[str, object]:
        return {"servers": {"example": {"url": server.url, "sse_read_timeout": read_timeout}}}

    def warm(self, invoke: Invoke, server: RunningServer) -> None:
        invoke(["--help"], url=server.url)
        server.recorder.reset()

    def test_json_response_dropped_after_the_headers_fails_at_once(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        self.warm(invoke, json_server)
        json_server.recorder.break_mode = "drop"
        json_server.recorder.fail_method = "tools/call"

        started = time.monotonic()
        result = invoke(["no-args"], default=self.config(json_server))

        assert result.exit_code == 3
        assert time.monotonic() - started < 1
        assert "request to http://127.0.0.1" in result.stderr
        assert "peer closed connection" in result.stderr
        assert len(result.stderr.strip().splitlines()) == 1, "no SDK log lines or tracebacks"
        assert json_server.recorder.rpc_methods().count("tools/call") == 1, "a call is not retried once sent"

    def test_json_response_cut_in_the_body_fails_at_once(self, invoke: Invoke, json_server: RunningServer) -> None:
        self.warm(invoke, json_server)
        json_server.recorder.break_mode = "cut"
        json_server.recorder.fail_method = "tools/call"

        started = time.monotonic()
        result = invoke(["no-args"], default=self.config(json_server))

        assert result.exit_code == 3
        assert time.monotonic() - started < 1

    def test_dropped_list_never_reuses_the_broken_session(self, invoke: Invoke, json_server: RunningServer) -> None:
        json_server.recorder.break_mode = "drop"
        json_server.recorder.fail_method = "tools/list"

        started = time.monotonic()
        result = invoke(["--help"])

        assert result.exit_code == 3
        assert time.monotonic() - started < 1
        assert json_server.recorder.rpc_methods().count("tools/list") == 1

    @pytest.mark.parametrize("mode", ["drop", "cut"])
    def test_sse_stream_cut_ends_with_the_read_timeout(
        self, invoke: Invoke, sse_server: RunningServer, mode: str
    ) -> None:
        self.warm(invoke, sse_server)
        sse_server.recorder.break_mode = mode  # type: ignore[assignment]
        sse_server.recorder.fail_method = "tools/call"

        started = time.monotonic()
        result = invoke(["no-args"], default=self.config(sse_server, read_timeout=1))

        assert result.exit_code == 3, result.stderr
        assert time.monotonic() - started < 10
        assert "Traceback" not in result.stderr

    @pytest.mark.parametrize("server_name", ["json_server", "sse_server"])
    def test_slow_tool_past_the_read_timeout_exits_3(
        self, invoke: Invoke, request: pytest.FixtureRequest, server_name: str
    ) -> None:
        server: RunningServer = request.getfixturevalue(server_name)
        self.warm(invoke, server)

        result = invoke(["slow", "--seconds", "5"], default=self.config(server, read_timeout=0.5))

        assert result.exit_code == 3
        assert "timed out after 0.5s waiting for http://127.0.0.1" in result.stderr
        assert server.recorder.rpc_methods().count("tools/call") == 1


def test_ctrl_c_exits_130_at_once(json_server: RunningServer, home: Path) -> None:
    code = (
        "import sys; from mcp_to_cli_kit.cli import main; "
        f"main({BIN!r}, {{'servers': {{'example': {{'url': {json_server.url!r}}}}}}}, args=sys.argv[1:])"
    )
    env = {**os.environ, "HOME": str(home)}
    process = subprocess.Popen(
        [sys.executable, "-c", code, "slow", "--seconds", "30"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while "tools/call" not in json_server.recorder.rpc_methods():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "the call never reached the server"
            time.sleep(0.05)

        interrupted = time.monotonic()
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=10)
        elapsed = time.monotonic() - interrupted
    finally:
        process.kill()

    assert process.returncode == 130, stderr
    assert elapsed < 1.5
    assert "Interrupted." in stderr
    assert "Traceback" not in stderr


class TestProcessBehaviour:
    def test_building_the_cli_leaves_logging_alone(self) -> None:
        loggers = [logging.getLogger(name) for name in ("", "mcp_to_cli_kit", "mcp", "httpx")]
        before = [(logger.level, list(logger.handlers), logger.propagate) for logger in loggers]

        build_cli(BIN, {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}})

        assert [(logger.level, list(logger.handlers), logger.propagate) for logger in loggers] == before

    @pytest.mark.parametrize(
        ("raised", "code", "message"), [(KeyboardInterrupt, 130, "Interrupted."), (EOFError, 1, "Aborted!")]
    )
    def test_interrupt_and_end_of_input(
        self, invoke: Invoke, monkeypatch: pytest.MonkeyPatch, raised: type[BaseException], code: int, message: str
    ) -> None:
        def interrupted(self: object) -> None:
            raise raised()

        monkeypatch.setattr(Runtime, "top_level_listing", interrupted)

        result = invoke(["--help"])

        assert result.exit_code == code
        assert message in result.stderr

    def test_closed_pipe_is_quiet(self, json_server: RunningServer, home: Path) -> None:
        code = (
            "import sys; from mcp_to_cli_kit.cli import main; "
            f"main({BIN!r}, {{'servers': {{'example': {{'url': {json_server.url!r}}}}}}}, args=sys.argv[1:])"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code, "big", "--size", "4000000"],
            env={**os.environ, "HOME": str(home)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        process.stdout.read(10)
        process.stdout.close()
        stderr = process.stderr.read().decode()
        process.wait(timeout=20)

        assert "Traceback" not in stderr
        assert "BrokenPipeError" not in stderr


class TestXdgDirectories:
    def test_config_and_cache_follow_xdg(
        self, json_server: RunningServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_home = tmp_path / "xdg-config"
        cache_home = tmp_path / "xdg-cache"
        (config_home / BIN).mkdir(parents=True)
        (config_home / BIN / "config.yaml").write_text(
            yaml.safe_dump({"servers": {"example": {"url": json_server.url}}}), encoding="utf-8"
        )
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

        result = CliRunner().invoke(build_cli(BIN), ["--help"])

        assert result.exit_code == 0, result.output
        assert (cache_home / BIN / "example.json").exists()

    def test_relative_xdg_paths_are_ignored(self, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")

        result = CliRunner().invoke(build_cli(BIN), ["--help"])

        assert str(home / ".config" / BIN / "config.yaml") in result.stderr


class TestUnknownToolDetection:
    def test_ordinary_error_mentioning_an_unknown_tool_keeps_the_cache(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        invoke(["--help"])

        result = invoke(["fail", "--message", "lookup failed: unknown tool name in the query"])

        assert result.exit_code == 1
        assert SchemaCache(default_cache_dir(BIN)).path("example").exists()


def test_rich_is_never_imported(json_server: RunningServer, tmp_path: Path, home: Path) -> None:
    """A fake rich package that fails on import is first on the path; every output path must still work."""
    fake = tmp_path / "fake-packages" / "rich"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("raise ImportError('rich must not be imported')\n", encoding="utf-8")
    code = (
        "import sys; from mcp_to_cli_kit.cli import main; "
        f"main({BIN!r}, {{'servers': {{'example': {{'url': {json_server.url!r}}}}}}}, args=sys.argv[1:])"
    )
    env = {**os.environ, "HOME": str(home), "PYTHONPATH": str(fake.parent)}

    for args in (["--help"], ["multi"], ["multi", "--json"], ["structured"], ["fail", "--message", "x"]):
        completed = subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True, text=True)
        assert "rich must not be imported" not in completed.stderr, args
        assert completed.returncode in (0, 1), (args, completed.stderr)
