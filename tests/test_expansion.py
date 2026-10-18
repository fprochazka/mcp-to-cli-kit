"""``${NAME}`` references to environment variables in a server's url and headers.

The references are expanded only when a server is contacted, so an unset variable fails only
the commands that need that server, and a warm cache still answers ``--help`` and ``schema``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import BIN, Invoke
from fixture_server import RunningServer, free_port_url

from mcp_to_cli_kit import redaction
from mcp_to_cli_kit.cache import SchemaCache, default_cache_dir
from mcp_to_cli_kit.client import redact_url
from mcp_to_cli_kit.config import (
    ConfigError,
    ServerConfig,
    default_config_path,
    expand_server,
    load_config,
    parse_template,
    template_variables,
)

SECRET = "s3cr3t-value"

URL = "https://mcp.example.com/mcp"


@pytest.fixture(autouse=True)
def no_test_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EXAMPLE_TOKEN", "EXAMPLE_HOST", "EXAMPLE_BASE", "EXAMPLE_OTHER"):
        monkeypatch.delenv(name, raising=False)


def host_of(server: RunningServer) -> str:
    return server.url.removeprefix("http://").removesuffix("/mcp")


def with_token(url: str, **extra: Any) -> dict[str, Any]:
    return {"servers": {"example": {"url": url, "headers": {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}, **extra}}}


def authorizations(server: RunningServer) -> set[str | None]:
    return {request.headers.get("authorization") for request in server.recorder.requests}


class TestTemplateSyntax:
    @pytest.mark.parametrize(
        ("value", "env", "expected"),
        [
            ("Bearer ${EXAMPLE_TOKEN}", {"EXAMPLE_TOKEN": "t"}, "Bearer t"),
            ("${A}${B}-${A}", {"A": "1", "B": "2"}, "12-1"),
            ("$${A}", {}, "${A}"),
            ("x$${A}y${A}", {"A": "1"}, "x${A}y1"),
            ("$5 and $ and $$ and $A", {}, "$5 and $ and $$ and $A"),
            ("trailing $", {}, "trailing $"),
            ("empty ${A}.", {"A": ""}, "empty ."),
        ],
    )
    def test_expansion(self, value: str, env: dict[str, str], expected: str) -> None:
        resolved = expand_server("alpha", ServerConfig(url=URL, headers={"X-Value": value}), env)

        assert resolved.headers["X-Value"] == expected

    @pytest.mark.parametrize("value", ["a ${", "${A", "x${A-B}", "${1A}", "${}", "${ A }"])
    def test_invalid_reference_is_a_config_error_at_load(self, value: str) -> None:
        default = {"servers": {"alpha": {"url": URL, "headers": {"X-Value": SECRET + value}}}}

        with pytest.raises(ConfigError, match="header X-Value") as raised:
            load_config(BIN, default=default, environ={})

        assert SECRET not in str(raised.value)

    def test_invalid_reference_in_the_url(self) -> None:
        with pytest.raises(ConfigError, match=r"servers\.alpha\.url.*without a closing"):
            load_config(BIN, default={"servers": {"alpha": {"url": "https://${HOST/mcp"}}}, environ={})

    def test_template_variables(self) -> None:
        assert template_variables("${A}/$${B}/${C}/${A}") == ["A", "C"]
        assert parse_template("plain") == ["plain"]


class TestUrlValidation:
    @pytest.mark.parametrize("url", ["https://${EXAMPLE_HOST}/mcp", "${EXAMPLE_BASE}", "${EXAMPLE_BASE}/mcp"])
    def test_a_template_is_accepted_at_load(self, url: str) -> None:
        loaded = load_config(BIN, default={"servers": {"alpha": {"url": url}}}, environ={})

        assert loaded.config.servers["alpha"].url == url

    def test_a_literal_url_is_still_checked_at_load(self) -> None:
        with pytest.raises(ConfigError, match="got the scheme 'ftp'"):
            load_config(BIN, default={"servers": {"alpha": {"url": "ftp://$${x}/mcp"}}}, environ={})

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (f"ftp://host/mcp?token={SECRET}", "must be an http:// or https:// URL."),
            (f"{SECRET}://host/mcp", "must be an http:// or https:// URL."),
            (f"{SECRET}", "without a scheme"),
            ("https:///mcp", "with a host"),
            (f"http://127.0.0.1:{SECRET}/mcp", "with a valid host and numeric port"),
            (f"http://127.0.0.1/mcp\n{SECRET}", "is not a valid URL"),
        ],
    )
    def test_the_expanded_url_is_checked_at_use(self, value: str, message: str) -> None:
        with pytest.raises(ConfigError) as raised:
            expand_server("alpha", ServerConfig(url="${EXAMPLE_BASE}"), {"EXAMPLE_BASE": value})

        assert str(raised.value).startswith("Server alpha: url, once EXAMPLE_BASE is expanded, ")
        assert message in str(raised.value)
        assert SECRET not in str(raised.value)

    def test_a_literal_scheme_is_named(self) -> None:
        with pytest.raises(ConfigError, match="got the scheme 'ftp'"):
            expand_server("alpha", ServerConfig(url="ftp://${EXAMPLE_HOST}/mcp"), {"EXAMPLE_HOST": "h"})

    def test_a_scheme_with_a_reference_is_not_named(self) -> None:
        with pytest.raises(ConfigError) as raised:
            expand_server("alpha", ServerConfig(url="${EXAMPLE_BASE}://h/mcp"), {"EXAMPLE_BASE": SECRET})

        assert "got the scheme" not in str(raised.value) and SECRET not in str(raised.value)

    def test_a_variable_as_the_port(self) -> None:
        template = ServerConfig(url="http://127.0.0.1:${EXAMPLE_PORT}/mcp")

        assert expand_server("alpha", template, {"EXAMPLE_PORT": "8443"}).url == "http://127.0.0.1:8443/mcp"
        with pytest.raises(
            ConfigError,
            match="once EXAMPLE_PORT is expanded, must be an http:// or https:// URL with a valid host and numeric",
        ) as raised:
            expand_server("alpha", template, {"EXAMPLE_PORT": SECRET})
        assert SECRET not in str(raised.value)

    @pytest.mark.parametrize("value", [f"12/{SECRET}", f"x?{SECRET}", f"x#{SECRET}", f"x/@{SECRET}"])
    def test_a_value_that_moves_text_into_another_part(self, value: str) -> None:
        template = ServerConfig(url="http://user:${EXAMPLE_TOKEN}@mcp.example.com/mcp")

        with pytest.raises(ConfigError, match="once EXAMPLE_TOKEN is expanded, (is not|must be)") as raised:
            expand_server("alpha", template, {"EXAMPLE_TOKEN": value})

        assert SECRET not in str(raised.value)

    def test_a_path_variable_may_hold_slashes(self) -> None:
        resolved = expand_server("alpha", ServerConfig(url="https://h/${EXAMPLE_BASE}"), {"EXAMPLE_BASE": "v1/mcp"})

        assert resolved.url == "https://h/v1/mcp"

    def test_a_template_url_is_redacted_as_written(self) -> None:
        assert redact_url("https://u:${PASS}@${HOST}:8443/mcp?token=${TOKEN}") == "https://${HOST}:8443/mcp"


class TestHeaderValues:
    @pytest.mark.parametrize(
        "value",
        [f"{SECRET}\n", f"{SECRET}\r\nX-Injected: 1", f"{SECRET}\x00", f"{SECRET}\x01", f"{SECRET}\x7f", f"{SECRET}é"],
    )
    def test_control_and_non_ascii_characters_are_rejected(self, value: str) -> None:
        server = ServerConfig(url=URL, headers={"Authorization": "Bearer ${EXAMPLE_TOKEN}"})

        with pytest.raises(ConfigError) as raised:
            expand_server("alpha", server, {"EXAMPLE_TOKEN": value})

        assert str(raised.value) == (
            "Server alpha: header Authorization (from EXAMPLE_TOKEN) holds a control or non-ASCII character."
        )

    def test_a_call_with_a_newline_in_the_token_exits_2(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["-vv", "no-args"], default=with_token(json_server.url), env={"EXAMPLE_TOKEN": f"{SECRET}\n"})

        assert result.exit_code == 2
        assert "header Authorization (from EXAMPLE_TOKEN) holds a control or non-ASCII character" in result.stderr
        assert SECRET not in result.stderr + result.stdout
        assert json_server.recorder.requests == []


class TestUnsetVariable:
    def test_the_error_names_the_server_the_key_and_the_variable(self) -> None:
        server = ServerConfig(url=URL, headers={"Authorization": "Bearer ${EXAMPLE_TOKEN}${EXAMPLE_OTHER}"})

        with pytest.raises(ConfigError) as raised:
            expand_server("alpha", server, {"EXAMPLE_OTHER": SECRET})

        assert str(raised.value) == (
            "Server alpha: header Authorization uses the environment variable EXAMPLE_TOKEN, which is not set."
        )

    def test_several_unset_variables(self) -> None:
        with pytest.raises(ConfigError, match="variables EXAMPLE_HOST, EXAMPLE_BASE, which are not set"):
            expand_server("alpha", ServerConfig(url="https://${EXAMPLE_HOST}/${EXAMPLE_BASE}"), {})

    def test_a_call_fails_with_exit_2_and_sends_nothing(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["no-args"], default=with_token(json_server.url))

        assert result.exit_code == 2
        assert result.stderr == (
            "Error: Server example: header Authorization uses the environment variable EXAMPLE_TOKEN, "
            "which is not set.\n"
        )
        assert json_server.recorder.requests == []

    def test_a_url_variable_with_a_cold_cache(self, invoke: Invoke) -> None:
        result = invoke(["--help"], default={"servers": {"example": {"url": "http://${EXAMPLE_HOST}/mcp"}}})

        assert result.exit_code == 2
        assert "Server example: url uses the environment variable EXAMPLE_HOST, which is not set." in result.stderr

    def test_only_the_server_that_needs_it_fails(
        self, invoke: Invoke, json_server: RunningServer, sse_server: RunningServer
    ) -> None:
        default = {
            "servers": {
                "alpha": {"url": json_server.url},
                "beta": {"url": "http://${EXAMPLE_HOST}/mcp", "mount": {"group": "b"}},
            }
        }

        assert invoke(["--help"], default=default).exit_code == 0
        assert invoke(["no-args"], default=default).exit_code == 0
        beta = invoke(["b", "no-args"], default=default)

        assert beta.exit_code == 2
        assert "Server beta: url uses the environment variable EXAMPLE_HOST" in beta.stderr
        assert json_server.recorder.calls() == [{"name": "no_args", "arguments": {}}]
        assert sse_server.recorder.requests == []

    def test_an_unresolved_root_server_is_listed_as_unavailable(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        default = {
            "servers": {
                "alpha": {"url": json_server.url, "mount": {"prefix": "a-"}},
                "beta": {"url": "http://${EXAMPLE_HOST}/mcp", "mount": {"prefix": "b-"}},
            }
        }

        result = invoke(["--help"], default=default)

        assert result.exit_code == 0
        assert "a-no-args" in result.stdout
        unavailable = result.stdout.split("Unavailable servers:")[1]
        assert "beta" in unavailable and "EXAMPLE_HOST" in unavailable

    def test_a_root_name_on_an_unresolved_server(self, invoke: Invoke) -> None:
        result = invoke(["anything"], default={"servers": {"example": {"url": "http://${EXAMPLE_HOST}/mcp"}}})

        assert result.exit_code == 2
        assert "EXAMPLE_HOST" in result.stderr

    def test_a_root_tool_while_another_root_server_is_unresolved(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        default = {"servers": {"alpha": {"url": json_server.url}, "beta": {"url": "http://${EXAMPLE_HOST}/mcp"}}}

        result = invoke(["no-args"], default=default)

        assert result.exit_code == 0
        assert "warning: could not check server beta for a conflicting tool" in result.stderr
        assert "EXAMPLE_HOST, which is not set" in result.stderr
        assert json_server.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    @pytest.mark.parametrize("args", [["--help"], ["no-args"]])
    def test_unreachable_and_unresolved_together(self, invoke: Invoke, args: list[str]) -> None:
        default = {"servers": {"alpha": {"url": free_port_url()}, "beta": {"url": "http://${EXAMPLE_HOST}/mcp"}}}

        result = invoke(args, default=default)

        assert result.exit_code == 2
        assert "(alpha)" in result.stderr and "cannot connect" in result.stderr
        assert "Server beta: url uses the environment variable EXAMPLE_HOST" in result.stderr


class TestLayers:
    def test_from_the_code_default(self, invoke: Invoke, json_server: RunningServer) -> None:
        result = invoke(["no-args"], default=with_token(json_server.url), env={"EXAMPLE_TOKEN": "t-code"})

        assert result.exit_code == 0
        assert authorizations(json_server) == {"Bearer t-code"}

    def test_from_the_file(self, invoke: Invoke, json_server: RunningServer) -> None:
        path = default_config_path(BIN)
        path.parent.mkdir(parents=True)
        file = {"servers": {"example": {"headers": {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}}}}
        path.write_text(yaml.safe_dump(file), encoding="utf-8")

        result = invoke(
            ["no-args"], default={"servers": {"example": {"url": json_server.url}}}, env={"EXAMPLE_TOKEN": "t-file"}
        )

        assert result.exit_code == 0
        assert authorizations(json_server) == {"Bearer t-file"}

    def test_from_the_environment(self, invoke: Invoke, json_server: RunningServer) -> None:
        env = {
            "EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION": "Bearer ${EXAMPLE_TOKEN}",
            "EXAMPLE_TOKEN": "t-env",
        }

        result = invoke(["no-args"], default={"servers": {"example": {"url": json_server.url}}}, env=env)

        assert result.exit_code == 0
        assert authorizations(json_server) == {"Bearer t-env"}

    def test_the_url_from_a_template(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": {"url": "http://${EXAMPLE_HOST}/mcp"}}}

        result = invoke(["no-args"], default=default, env={"EXAMPLE_HOST": host_of(json_server)})

        assert result.exit_code == 0
        assert json_server.recorder.calls() == [{"name": "no_args", "arguments": {}}]

    def test_an_escaped_reference_is_sent_literally(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": {"url": json_server.url, "headers": {"X-Literal": "a$${EXAMPLE_TOKEN}"}}}}

        assert invoke(["no-args"], default=default, env={"EXAMPLE_TOKEN": "t"}).exit_code == 0
        assert {request.headers.get("x-literal") for request in json_server.recorder.requests} == {"a${EXAMPLE_TOKEN}"}

    def test_other_values_are_not_expanded(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"help": "Help with ${EXAMPLE_TOKEN}.", "servers": {"example": {"url": json_server.url}}}

        result = invoke(["--help"], default=default, env={"EXAMPLE_TOKEN": "t"})

        assert "Help with ${EXAMPLE_TOKEN}." in result.stdout


class TestOffline:
    def warm(self, invoke: Invoke, default: dict[str, Any], server: RunningServer) -> None:
        assert invoke(["--help"], default=default, env={"EXAMPLE_TOKEN": "t"}).exit_code == 0
        server.recorder.reset()

    def test_help_and_schema_work_from_a_warm_cache_with_the_variable_unset(
        self, invoke: Invoke, json_server: RunningServer
    ) -> None:
        default = with_token(json_server.url)
        self.warm(invoke, default, json_server)

        help_result = invoke(["--help"], default=default)
        tool_help = invoke(["search-items", "--help"], default=default)
        schema = invoke(["schema", "no-args"], default=default)

        assert help_result.exit_code == 0 and "no-args" in help_result.stdout
        assert tool_help.exit_code == 0 and "--query" in tool_help.stdout
        assert schema.exit_code == 0
        assert json_server.recorder.requests == []

    def test_help_from_a_warm_cache_with_a_url_variable_unset(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = {"servers": {"example": {"url": "http://${EXAMPLE_HOST}/mcp"}}}
        assert invoke(["--help"], default=default, env={"EXAMPLE_HOST": host_of(json_server)}).exit_code == 0
        json_server.recorder.reset()

        result = invoke(["--help"], default=default)

        assert result.exit_code == 0 and "no-args" in result.stdout
        assert json_server.recorder.requests == []

    def test_an_expired_cache_is_used_with_a_warning(self, invoke: Invoke, json_server: RunningServer) -> None:
        default = with_token(json_server.url)
        self.warm(invoke, default, json_server)
        cache = SchemaCache(default_cache_dir(BIN))
        content = json.loads(cache.path("example").read_text(encoding="utf-8"))
        content["fetched_at"] = time.time() - 7200
        cache.path("example").write_text(json.dumps(content), encoding="utf-8")

        result = invoke(["--help"], default=default)

        assert result.exit_code == 0 and "no-args" in result.stdout
        assert "warning: cannot refresh the tool list of example" in result.stderr
        assert "EXAMPLE_TOKEN, which is not set" in result.stderr
        assert json_server.recorder.requests == []


class TestCacheKey:
    TEMPLATE = "https://${EXAMPLE_HOST}/mcp"

    def test_the_expanded_url_must_match_when_it_is_known(self, home: Path) -> None:
        cache = SchemaCache(default_cache_dir(BIN))
        cache.save("example", self.TEMPLATE, [{"name": "a"}], "https://one.example.com/mcp")

        assert cache.load("example", self.TEMPLATE, "https://one.example.com/mcp") is not None
        assert cache.load("example", self.TEMPLATE, "https://two.example.com/mcp") is None
        assert cache.load("example", self.TEMPLATE, None) is not None
        assert cache.load("example", "https://${EXAMPLE_OTHER}/mcp", None) is None

    def test_pointing_the_variable_elsewhere_fetches_again(
        self, invoke: Invoke, json_server: RunningServer, sse_server: RunningServer
    ) -> None:
        default = {"servers": {"example": {"url": "http://${EXAMPLE_HOST}/mcp"}}}
        invoke(["--help"], default=default, env={"EXAMPLE_HOST": host_of(json_server)})

        invoke(["--help"], default=default, env={"EXAMPLE_HOST": host_of(sse_server)})

        assert "tools/list" in sse_server.recorder.rpc_methods()

    def test_a_literal_url_keeps_the_old_cache_format(self) -> None:
        cache = SchemaCache(default_cache_dir(BIN))
        cache.save("example", URL, [{"name": "a"}], URL)

        content = json.loads(cache.path("example").read_text(encoding="utf-8"))

        assert "resolved_url_sha256" not in content
        assert cache.load("example", URL) is not None


class TestRedaction:
    def test_errors_show_the_template_never_the_values(self, invoke: Invoke, json_server: RunningServer) -> None:
        json_server.recorder.fail_with = [500]
        url = "http://${EXAMPLE_HOST}/mcp?token=${EXAMPLE_TOKEN}"
        env = {"EXAMPLE_HOST": host_of(json_server), "EXAMPLE_TOKEN": SECRET}

        result = invoke(["--help"], default={"servers": {"example": {"url": url}}}, env=env)

        assert result.exit_code == 3
        assert "HTTP 500 Internal Server Error from http://${EXAMPLE_HOST}/mcp" in result.stderr
        assert SECRET not in result.stderr

    def test_a_header_secret_is_never_in_an_error(self, invoke: Invoke) -> None:
        result = invoke(["--help"], default=with_token(free_port_url()), env={"EXAMPLE_TOKEN": SECRET})

        assert result.exit_code == 3
        assert SECRET not in result.stderr and SECRET not in result.stdout


class TestVerboseLog:
    def test_minus_vv_shows_no_expanded_value(self, invoke: Invoke, json_server: RunningServer) -> None:
        url = "http://user:${EXAMPLE_OTHER}@${EXAMPLE_HOST}/mcp?token=${EXAMPLE_BASE}"
        default = {"servers": {"example": {"url": url, "headers": {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}}}}
        env = {
            "EXAMPLE_HOST": host_of(json_server),
            "EXAMPLE_TOKEN": f"header-{SECRET}",
            "EXAMPLE_BASE": f"query-{SECRET}",
            "EXAMPLE_OTHER": f"password-{SECRET}",
        }

        result = invoke(["-vv", "no-args"], default=default, env=env)

        assert result.exit_code == 0
        output = result.stdout + result.stderr
        assert "httpx" in output and "***" in output, "the log was not written or not redacted"
        assert SECRET not in output
        assert {request.headers.get("authorization") for request in json_server.recorder.requests} != {None}

    def test_minus_vv_on_a_failure(self, invoke: Invoke) -> None:
        url = free_port_url().replace("/mcp", "/mcp?token=${EXAMPLE_TOKEN}")

        result = invoke(
            ["-vv", "--help"], default={"servers": {"example": {"url": url}}}, env={"EXAMPLE_TOKEN": SECRET}
        )

        assert result.exit_code == 3
        assert SECRET not in result.stdout + result.stderr

    @pytest.mark.parametrize(
        ("password", "query"),
        [
            (f"pass word@{SECRET}", f"a b@{SECRET}"),
            (f'x"<>|{SECRET}é', f'x"<>|{SECRET}é'),
            (f"%40{SECRET}", f"%7C{SECRET}"),
        ],
    )
    def test_minus_vv_shows_no_encoded_form(
        self, invoke: Invoke, json_server: RunningServer, password: str, query: str
    ) -> None:
        url = f"http://user:${{EXAMPLE_OTHER}}@{host_of(json_server)}/mcp?token=${{EXAMPLE_BASE}}"

        result = invoke(
            ["-vv", "no-args"],
            default={"servers": {"example": {"url": url}}},
            env={"EXAMPLE_OTHER": password, "EXAMPLE_BASE": query},
        )

        assert result.exit_code == 0, result.stderr
        assert "httpx" in result.stderr
        assert SECRET not in result.stdout + result.stderr


class TestWhatIsRegistered:
    def test_host_port_and_path_values_are_not_registered(self) -> None:
        server = ServerConfig(url="https://${EXAMPLE_HOST}:${EXAMPLE_PORT}/${EXAMPLE_BASE}?q=${EXAMPLE_TOKEN}")
        env = {
            "EXAMPLE_HOST": "user-host",
            "EXAMPLE_PORT": "8443",
            "EXAMPLE_BASE": "user_search",
            "EXAMPLE_TOKEN": SECRET,
        }

        expand_server("alpha", server, env)

        assert (
            redaction.redact("Unknown tool: user_search on user-host:8443")
            == "Unknown tool: user_search on user-host:8443"
        )
        assert redaction.redact(f"token {SECRET}") == "token ${EXAMPLE_TOKEN}"

    def test_header_userinfo_and_query_values_are_registered(self) -> None:
        server = ServerConfig(
            url="https://${EXAMPLE_OTHER}@mcp.example.com/mcp?k=v#${EXAMPLE_BASE}",
            headers={"Authorization": "Bearer ${EXAMPLE_TOKEN}"},
        )
        env = {"EXAMPLE_OTHER": f"u:{SECRET}-pw", "EXAMPLE_TOKEN": f"{SECRET}-hdr", "EXAMPLE_BASE": f"{SECRET}-frag"}

        expand_server("alpha", server, env)

        assert SECRET not in redaction.redact(f"{SECRET}-pw {SECRET}-hdr u:{SECRET}-pw {SECRET}-frag")

    def test_the_runtime_forgets_them_when_it_closes(self, invoke: Invoke, json_server: RunningServer) -> None:
        url = f"{json_server.url}?token=${{EXAMPLE_TOKEN}}"

        assert (
            invoke(["no-args"], default={"servers": {"example": {"url": url}}}, env={"EXAMPLE_TOKEN": SECRET}).exit_code
            == 0
        )

        assert redaction.redact(SECRET) == SECRET
