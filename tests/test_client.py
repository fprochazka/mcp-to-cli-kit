"""Unit tests for the client helpers that need no server."""

from __future__ import annotations

import concurrent.futures

import anyio
import httpx
import pytest
from fixture_server import free_port_url

from mcp_to_cli_kit.client import (
    FALLBACK_VERSION,
    PortalProvider,
    TransportError,
    _HttpTrace,
    _retry_before_answer,
    _retry_on_connect,
    _Session,
    client_header_value,
    describe_failure,
    distribution_version,
    redact_url,
)
from mcp_to_cli_kit.config import ServerConfig

URL = "https://mcp.example.com/mcp"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:pass@mcp.example.com:8443/mcp/abc?token=x#frag", "https://mcp.example.com:8443/mcp/abc"),
        ("http://mcp.example.com", "http://mcp.example.com"),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert redact_url(url) == expected


def test_distribution_version() -> None:
    assert distribution_version("no-such-distribution-xyz") == FALLBACK_VERSION
    assert distribution_version(None, "no-such-bin-xyz") == FALLBACK_VERSION
    assert distribution_version(None, None) == FALLBACK_VERSION
    assert distribution_version("mcp-to-cli-kit") != ""


def test_client_header_value() -> None:
    value = client_header_value("example-tool", "1.2.3")

    assert value.startswith("example-tool/1.2.3 (mcp-to-cli-kit ")
    assert value.endswith(")")


def status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", URL + "?token=secret")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("failed", request=request, response=response)


class TestDescribeFailure:
    def test_status_inside_nested_groups(self) -> None:
        error = BaseExceptionGroup("outer", [ExceptionGroup("inner", [ValueError("noise"), status_error(503)])])

        described = describe_failure(error, redact_url(URL))

        assert str(described) == f"HTTP 503 Service Unavailable from {URL}"
        assert (described.kind, described.status) == ("status", 503)

    def test_auth_hint(self) -> None:
        assert "credentials" in str(describe_failure(status_error(401), URL))

    def test_connect_error(self) -> None:
        described = describe_failure(ExceptionGroup("g", [httpx.ConnectError("refused")]), URL)

        assert described.kind == "connect"
        assert str(described) == f"cannot connect to {URL}: refused"

    def test_timeout(self) -> None:
        assert describe_failure(httpx.ReadTimeout("slow"), URL).kind == "timeout"

    def test_unknown_error_is_one_line(self) -> None:
        described = describe_failure(RuntimeError("first line\nsecond line"), URL)

        assert str(described) == f"talking to {URL} failed: RuntimeError: first line"


class TestRetryPolicy:
    @pytest.mark.parametrize(
        ("error", "before_answer", "on_connect"),
        [
            (TransportError("x", "connect"), True, True),
            (TransportError("x", "timeout"), True, False),
            (TransportError("x", "status", 502), True, False),
            (TransportError("x", "status", 503), True, False),
            (TransportError("x", "status", 504), True, False),
            (TransportError("x", "status", 500), False, False),
            (TransportError("x", "status", 429), False, False),
            (TransportError("x", "protocol"), False, False),
        ],
    )
    def test_policy(self, error: TransportError, before_answer: bool, on_connect: bool) -> None:
        assert _retry_before_answer(error) is before_answer
        assert _retry_on_connect(error) is on_connect


class TestHttpTrace:
    def test_only_the_latest_post_counts(self) -> None:
        trace = _HttpTrace()

        async def scenario() -> None:
            post = httpx.Request("POST", URL)
            await trace.on_request(post)
            await trace.on_response(httpx.Response(404, request=post))
            assert trace.status == 404

            await trace.on_response(httpx.Response(405, request=httpx.Request("GET", URL)))
            assert trace.status == 404, "the GET stream does not count"

            await trace.on_request(httpx.Request("POST", URL))
            assert trace.status is None, "a new POST clears the old status"

        anyio.run(scenario)


def test_cancel_before_the_session_started_takes_effect() -> None:
    server = ServerConfig(url=free_port_url())
    with PortalProvider() as provider:
        session = _Session(provider(), server.url, server, {})
        session.cancel()

        concurrent.futures.wait([session.finished], timeout=2)
        assert session.finished.done()
        assert not session.usable
