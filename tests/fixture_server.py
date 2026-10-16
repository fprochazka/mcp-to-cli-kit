"""An MCP server for the end-to-end tests, run with uvicorn on a localhost port.

Its tools cover the schema constructs the argument compiler must handle. A recorder in front
of the app keeps every JSON-RPC request, so tests assert what went over the wire, and can make
the next requests fail with an HTTP status.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import TextContent
from pydantic import BaseModel, Field, WithJsonSchema

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
AsgiApp = Callable[[Scope, Receive, Send], Awaitable[None]]

CZECH = "Příliš žluťoučký kůň úpěl ďábelské ódy"

RAW_JSON = '{"b": 1,  "a": [1.10, "ž"], "nested": {"x": null}}'

ALL_OF_SCHEMA = {
    "allOf": [
        {"type": "object", "properties": {"a": {"type": "integer"}}},
        {"required": ["a"]},
    ],
    "description": "An object built with allOf.",
}


class Point(BaseModel):
    """A point."""

    x: int
    y: int


class Filter(BaseModel):
    """A filter on one field."""

    field: str
    value: int


@dataclass
class Recorded:
    """One HTTP request the server received."""

    http_method: str
    rpc_method: str | None
    params: dict[str, Any] | None
    headers: dict[str, str]


@dataclass
class Recorder:
    """Records requests and injects failures in front of an ASGI app."""

    app: AsgiApp
    sse: bool
    requests: list[Recorded] = field(default_factory=list)
    # HTTP statuses returned instead of the app's answer, consumed one per matching request.
    fail_with: list[int] = field(default_factory=list)
    fail_method: str | None = None
    # "drop": send the headers of a 200 and close; "cut": also part of the body. For fail_method only.
    break_mode: Literal["drop", "cut"] | None = None

    def reset(self) -> None:
        self.requests.clear()
        self.fail_with.clear()
        self.fail_method = None
        self.break_mode = None

    def rpc_methods(self) -> list[str]:
        return [request.rpc_method for request in self.requests if request.rpc_method is not None]

    def calls(self) -> list[dict[str, Any]]:
        return [request.params or {} for request in self.requests if request.rpc_method == "tools/call"]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break

        rpc_method = None
        params = None
        if body:
            payload = json.loads(body)
            if isinstance(payload, dict):
                rpc_method = payload.get("method")
                params = payload.get("params")
        headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
        self.requests.append(Recorded(scope["method"], rpc_method, params, headers))

        if self.fail_with and (self.fail_method is None or self.fail_method == rpc_method):
            status = self.fail_with.pop(0)
            await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"injected failure"})
            return

        if self.break_mode is not None and self.fail_method == rpc_method:
            await self._break(send)
            return

        sent = False

        async def replay() -> Message:
            # The body once, then the real channel, which reports the client's disconnect when it comes.
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _break(self, send: Send) -> None:
        """Start a 200 response and lose the connection, as a crashing proxy or server would."""
        content_type = b"text/event-stream" if self.sse else b"application/json"
        headers = [(b"content-type", content_type)]
        if not self.sse:
            headers.append((b"content-length", b"500"))
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        if self.break_mode == "cut":
            partial = b'event: message\ndata: {"jsonrpc": "2.0", "id"' if self.sse else b'{"jsonrpc": "2.0", "id"'
            await send({"type": "http.response.body", "body": partial, "more_body": True})
        raise ConnectionAbortedError("the fixture server dropped the connection")


def build_server(json_response: bool) -> FastMCP:
    """The example server with every tool the tests use."""
    mcp = FastMCP("example", stateless_http=True, json_response=json_response)

    @mcp.tool(structured_output=False)
    def search_items(
        query: Annotated[str, Field(description="What to search for.")],
        startTime: Annotated[str | None, Field(description="ISO timestamp to start from.")] = None,
        kind: Annotated[Literal["alpha", "beta", "gamma"] | None, Field(description="The kind.")] = None,
        limit: Annotated[int, Field(ge=1, le=100, description="How many results.")] = 10,
        tags: list[str] | None = None,
        points: list[Point] | None = None,
        filter: Filter | None = None,
        options: dict[str, Any] | None = None,
        verbose: bool = False,
        strict: bool | None = None,
        json: str | None = None,
        ratio: float | None = None,
        combined: Annotated[dict[str, Any] | None, WithJsonSchema(ALL_OF_SCHEMA)] = None,
    ) -> str:
        """Search the items.

        Long description that must never be cut. Allowed sort orders are listed here:
        - newest
        - oldest

        The END of the description."""
        return '{"status": "ok"}'

    @mcp.tool(structured_output=False)
    def fail(message: str) -> str:
        """Always fails with the given message."""
        raise ToolError(message)

    @mcp.tool(structured_output=False)
    def big(size: int) -> str:
        """Return a JSON document of about size bytes."""
        chunk = CZECH + " "
        text = (chunk * (size // len(chunk.encode()) + 1))[: size // 2]
        return '{"data": ' + json.dumps(text, ensure_ascii=False) + "}"

    @mcp.tool(structured_output=False)
    def multi() -> list[TextContent]:
        """Return several content items."""
        return [
            TextContent(type="text", text='{"total": 2}'),
            TextContent(type="text", text="Total results: 2, showing 2."),
            TextContent(type="text", text='[1, "ž"]'),
        ]

    @mcp.tool(structured_output=False)
    def plain_text() -> str:
        """Return text that is not JSON."""
        return CZECH + "\nsecond line"

    @mcp.tool(structured_output=False)
    def raw_json() -> str:
        """Return JSON with spacing and number formats a re-serialization would change."""
        return RAW_JSON

    @mcp.tool()
    def structured() -> dict[str, Any]:
        """Return structured content."""
        return {"name": CZECH, "count": 3}

    @mcp.tool()
    def wrapped_text() -> str:
        """Return a plain string, which the server wraps as structured {"result": ...}."""
        return "just text"

    @mcp.tool(structured_output=False)
    async def slow(seconds: float) -> str:
        """Answer after the given number of seconds."""
        await asyncio.sleep(seconds)
        return '"slept"'

    @mcp.tool(structured_output=False)
    def no_args() -> str:
        """Take nothing."""
        return '"done"'

    @mcp.tool(structured_output=False)
    def error_as_text() -> str:
        """Report a failure as an ordinary result."""
        return "Error: HTTP 403 Forbidden for GET /resource"

    return mcp


@dataclass
class RunningServer:
    """A fixture server listening on localhost."""

    url: str
    recorder: Recorder
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def start(json_response: bool) -> RunningServer:
    """Start a fixture server on a free port."""
    mcp = build_server(json_response)
    recorder = Recorder(mcp.streamable_http_app(), sse=not json_response)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    # Connections the break modes or the slow tool leave open must not hold the shutdown up.
    config = uvicorn.Config(recorder, log_level="error", lifespan="on", ws="none", timeout_graceful_shutdown=1)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the fixture server did not start")
        time.sleep(0.01)
    return RunningServer(f"http://127.0.0.1:{port}/mcp", recorder, server, thread)


def free_port_url() -> str:
    """A URL on a localhost port nothing listens on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/mcp"
