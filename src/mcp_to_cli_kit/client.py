"""Talk to one MCP server over streamable HTTP, from synchronous click code.

One invocation of the CLI opens at most one MCP session per server, lazily, and keeps it open
until the CLI exits, so a ``tools/list`` on a cache miss and the ``tools/call`` that follows
share it. The session lives in a task on a background event loop (an anyio blocking portal);
the synchronous side hands it work and waits for either the result or the session's death,
whichever comes first, so an HTTP error raised inside the SDK's task group never leaves a
caller hanging.

The SDK's ``ClientSession.call_tool`` sends an extra ``tools/list`` after every successful
call to validate output schemas. This module sends ``tools/call`` through ``send_request``
instead, which skips that round trip: the CLI prints results, it does not validate them.

There is no fallback to the legacy SSE transport: it hid the real error and could run a tool
call twice.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import timedelta
from importlib import metadata
from typing import Any, Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

import anyio
import httpx
from anyio.from_thread import BlockingPortal
from anyio.streams.memory import MemoryObjectSendStream
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError

from mcp_to_cli_kit.config import ServerConfig

logger = logging.getLogger(__name__)

KIT_DISTRIBUTION = "mcp-to-cli-kit"

FALLBACK_VERSION = "0.1.0"

CLIENT_HEADER = "X-Mcp-Client"

# Pauses before the first and second retry. Two retries at most.
DEFAULT_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.5)

RETRYABLE_STATUS = frozenset({502, 503, 504})

# The request-timeout error code some servers use; the Python SDK itself uses 408.
JSONRPC_REQUEST_TIMEOUT = -32001

# How long closing a session may take before the CLI exits anyway.
_CLOSE_TIMEOUT_SECONDS = 5.0

# How long a cancelled session may take to unwind.
_CANCEL_TIMEOUT_SECONDS = 1.0

# After "Connection closed", how long to wait for the session to report the real cause.
_CAUSE_GRACE_SECONDS = 2.0

_T = TypeVar("_T")

FailureKind = Literal["connect", "timeout", "status", "protocol", "other"]


class TransportError(Exception):
    """The server could not be reached or did not speak MCP. The message is one line, safe to print.

    Attributes:
        kind: What went wrong, which decides whether a retry may help.
        status: The HTTP status, for ``kind == "status"``.
    """

    def __init__(self, message: str, kind: FailureKind, status: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


class ServerRejectedError(Exception):
    """The server answered a request with a JSON-RPC error. The message is safe to print.

    Attributes:
        code: The JSON-RPC error code.
    """

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def redact_url(url: str) -> str:
    """Strip the userinfo, query and fragment, which may carry secrets, from a URL."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "<invalid URL>"


def distribution_version(distribution: str | None, bin_name: str | None = None) -> str:
    """The version of a distribution, found by name or through the console script ``bin_name``.

    Returns:
        The version, or ``0.1.0`` when the distribution cannot be found.
    """
    try:
        if distribution is not None:
            return metadata.version(distribution)
        if bin_name is not None:
            for entry_point in metadata.entry_points(group="console_scripts", name=bin_name):
                if entry_point.dist is not None:
                    return entry_point.dist.version
    except metadata.PackageNotFoundError:
        pass
    return FALLBACK_VERSION


def client_header_value(bin_name: str, bin_version: str) -> str:
    """``<bin>/<version> (mcp-to-cli-kit <version>)``, the value of the audit header."""
    return f"{bin_name}/{bin_version} ({KIT_DISTRIBUTION} {distribution_version(KIT_DISTRIBUTION)})"


@dataclass
class _HttpTrace:
    """The status of the latest POST, to explain errors the SDK reports vaguely.

    Only POSTs count, and each new POST clears the previous status, so an old failure (or the
    GET stream's 405) never relabels a later error.
    """

    status: int | None = None
    reason: str = ""

    async def on_request(self, request: httpx.Request) -> None:
        if request.method == "POST":
            self.status = None
            self.reason = ""

    async def on_response(self, response: httpx.Response) -> None:
        if response.request.method == "POST" and response.status_code >= 400:
            self.status = response.status_code
            self.reason = response.reason_phrase


@dataclass
class _Work:
    run: Callable[[ClientSession], Awaitable[Any]]
    future: concurrent.futures.Future[Any] = field(default_factory=concurrent.futures.Future)


class _Session:
    """One MCP session running in a task of the portal's event loop."""

    def __init__(
        self,
        portal: BlockingPortal,
        url: str,
        server: ServerConfig,
        headers: dict[str, str],
        client_info: types.Implementation | None = None,
    ) -> None:
        self._client_info = client_info
        self._portal = portal
        self._url = url
        self._server = server
        self._headers = headers
        self.trace = _HttpTrace()
        self._send: MemoryObjectSendStream[_Work] | None = None
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._busy = False
        # The portal marks its futures as running, so Future.cancel() cannot stop the task; this scope can.
        self._scope: anyio.CancelScope | None = None
        self._cancel_requested = False
        self.transport_failure: Exception | None = None
        self.finished: concurrent.futures.Future[None] = portal.start_task_soon(self._serve)

    async def _serve(self) -> None:
        with anyio.CancelScope() as scope:
            self._scope = scope
            # A cancel() that came before the scope existed still counts.
            if self._cancel_requested:
                scope.cancel()
            await self._serve_until_closed()

    async def _on_message(self, message: Any) -> None:
        """Make a transport exception the SDK only forwards (like a JSON response cut after its headers) fatal.

        The SDK puts such an exception into the read stream. Without this the request would wait
        for the read timeout; raising here makes the SDK fail the pending request at once with
        "Connection closed", and the recorded exception says why.
        """
        if isinstance(message, Exception):
            self.transport_failure = message
            # The SDK's receive loop dies with this raise, so the session can never answer again.
            if self._scope is not None:
                self._scope.cancel()
            raise message
        await anyio.lowlevel.checkpoint()

    async def _serve_until_closed(self) -> None:
        send, receive = anyio.create_memory_object_stream[_Work](16)
        self._send = send
        self._ready.set_result(None)
        timeout = httpx.Timeout(self._server.timeout, read=self._server.sse_read_timeout)
        async with (
            receive,
            httpx.AsyncClient(
                headers=self._headers,
                timeout=timeout,
                event_hooks={"request": [self.trace.on_request], "response": [self.trace.on_response]},
            ) as http,
            streamable_http_client(self._url, http_client=http) as (read_stream, write_stream, _),
            # An SSE stream cut in the middle of a response is swallowed by the SDK without any
            # hook, so such a request ends only with this read timeout (an McpError 408).
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self._server.sse_read_timeout),
                message_handler=self._on_message,
                client_info=self._client_info,
            ) as session,
        ):
            await session.initialize()
            async for work in receive:
                try:
                    work.future.set_result(await work.run(session))
                except Exception as error:  # noqa: BLE001 - handed to the waiting caller
                    work.future.set_exception(error)

    def run(self, operation: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        """Run an operation on the session and wait for it, or for the session to fail.

        On Ctrl-C the session is cancelled at once, then the KeyboardInterrupt goes on.
        """
        self._busy = True
        try:
            return self._run(operation)
        except KeyboardInterrupt:
            self.cancel()
            raise
        finally:
            self._busy = False

    def _run(self, operation: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        concurrent.futures.wait([self._ready, self.finished], return_when=concurrent.futures.FIRST_COMPLETED)
        work = _Work(operation)
        if self._send is not None and not self.finished.done():
            # A session that died in between is noticed by the wait below.
            with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                self._portal.call(self._send.send, work)

        done, _ = concurrent.futures.wait([work.future, self.finished], return_when=concurrent.futures.FIRST_COMPLETED)
        if work.future in done:
            error = work.future.exception()
            if error is None:
                return work.future.result()  # type: ignore[no-any-return]
            if isinstance(error, McpError) and error.error.code == types.CONNECTION_CLOSED:
                if self.transport_failure is not None:
                    raise error  # described from transport_failure by the caller, without waiting
                # The session died under the request; its own failure says why.
                concurrent.futures.wait([self.finished], timeout=_CAUSE_GRACE_SECONDS)
                cause = _failure(self.finished) if self.finished.done() else None
                if cause is not None:
                    raise cause
            raise error

        cause = _failure(self.finished) or self.transport_failure
        if cause is None:
            raise TransportError("the MCP session ended before the request was answered", "protocol")
        raise cause

    @property
    def usable(self) -> bool:
        """Whether the session may take more work: running, and its receive loop not dead."""
        return not self.finished.done() and self.transport_failure is None and not self._cancel_requested

    def cancel(self) -> None:
        """Abandon the session and whatever request is in flight, waiting only briefly for it to unwind."""
        self._cancel_requested = True
        if self.finished.done():
            return
        scope = self._scope
        if scope is not None:
            with contextlib.suppress(Exception):
                self._portal.call(scope.cancel)
        concurrent.futures.wait([self.finished], timeout=_CANCEL_TIMEOUT_SECONDS)

    def close(self) -> None:
        """End the session, which also sends the DELETE that terminates it on the server.

        A session with a request still in flight (after Ctrl-C or an error) is cancelled instead
        of waited for.
        """
        if self._busy:
            self.cancel()
            return
        if self._send is not None and not self.finished.done():
            try:
                self._portal.call(self._send.aclose)
            except Exception as error:  # noqa: BLE001 - closing is best effort
                logger.debug("Closing the MCP session failed: %s", error)
        concurrent.futures.wait([self.finished], timeout=_CLOSE_TIMEOUT_SECONDS)
        if not self.finished.done():
            self.cancel()
        elif _failure(self.finished) is not None:
            logger.debug("The MCP session ended with an error: %s", _failure(self.finished))


def _failure(future: concurrent.futures.Future[None]) -> BaseException | None:
    """The exception a finished session task ended with, None when it ended normally or was cancelled."""
    if future.cancelled():
        return None
    return future.exception()


class ServerClient:
    """The MCP client of one configured server, for the lifetime of one CLI invocation."""

    def __init__(
        self,
        name: str,
        server: ServerConfig,
        client_header: str,
        portal: Callable[[], BlockingPortal],
        retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
        client_info: types.Implementation | None = None,
    ) -> None:
        self.name = name
        self.server = server
        # Sent in initialize as clientInfo; without it the SDK announces itself as "mcp".
        self._client_info = client_info
        self._headers = {CLIENT_HEADER: client_header, **server.headers}
        self._portal = portal
        self._retry_delays = retry_delays
        self._session: _Session | None = None

    @property
    def display_url(self) -> str:
        """The server URL without the parts that may carry secrets."""
        return redact_url(self.server.url)

    def list_tools(self) -> list[dict[str, Any]]:
        """Fetch every page of ``tools/list``, retrying on connect errors, timeouts and 502/503/504.

        Raises:
            TransportError: The server could not be reached, after the retries.
            ServerRejectedError: The server answered with a JSON-RPC error.
        """

        async def fetch(session: ClientSession) -> list[dict[str, Any]]:
            tools: list[dict[str, Any]] = []
            cursor: str | None = None
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
                result = await session.send_request(
                    types.ClientRequest(types.ListToolsRequest(params=params)), types.ListToolsResult
                )
                tools.extend(tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in result.tools)
                cursor = result.nextCursor
                if not cursor:
                    return tools

        return self._with_retries("tools/list", fetch, _retry_before_answer)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """Send ``tools/call``. It is retried only when the call itself provably never reached the server.

        Raises:
            TransportError: The server could not be reached, after the retries.
            ServerRejectedError: The server answered with a JSON-RPC error.
        """
        sent = [False]

        async def call(session: ClientSession) -> types.CallToolResult:
            sent[0] = True
            return await session.send_request(
                types.ClientRequest(
                    types.CallToolRequest(params=types.CallToolRequestParams(name=name, arguments=arguments))
                ),
                types.CallToolResult,
            )

        def policy(error: TransportError) -> bool:
            # The work only runs after initialize succeeded, so an unsent call failed in initialize.
            retry = _retry_on_connect(error) if sent[0] else _retry_before_answer(error)
            sent[0] = False
            return retry

        return self._with_retries("tools/call", call, policy)

    def close(self) -> None:
        """Close the session, if one was opened."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _with_retries(
        self,
        what: str,
        operation: Callable[[ClientSession], Awaitable[_T]],
        policy: Callable[[TransportError], bool],
    ) -> _T:
        attempt = 0
        while True:
            try:
                return self._run(operation)
            except TransportError as error:
                self.close()
                if not policy(error) or attempt >= len(self._retry_delays):
                    if attempt:
                        raise TransportError(
                            f"{error} (after {attempt + 1} attempts)", error.kind, error.status
                        ) from error
                    raise
                delay = self._retry_delays[attempt]
                attempt += 1
                logger.info("%s on %s failed (%s); retry %d in %.1fs", what, self.name, error, attempt, delay)
                time.sleep(delay)

    def _run(self, operation: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        if self._session is None or not self._session.usable:
            if self._session is not None:
                self._session.close()
            logger.debug("Opening an MCP session with %s (%s)", self.name, self.display_url)
            self._session = _Session(self._portal(), self.server.url, self.server, self._headers, self._client_info)
        session = self._session
        try:
            return session.run(operation)
        except McpError as error:
            raise self._describe_mcp_error(error, session.trace, session.transport_failure) from error
        except TransportError:
            raise
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt | SystemExit):
                raise
            raise describe_failure(error, self.display_url, session.trace) from error

    def _describe_mcp_error(
        self, error: McpError, trace: _HttpTrace, session_failure: Exception | None = None
    ) -> Exception:
        if error.error.code in (httpx.codes.REQUEST_TIMEOUT, JSONRPC_REQUEST_TIMEOUT):
            # The SDK's own read timeout (408), or a server that reports a timeout (-32001).
            return TransportError(
                f"timed out after {self.server.sse_read_timeout:g}s waiting for {self.display_url}", "timeout"
            )
        if trace.status is not None and error.error.code in (32600, -32600, types.CONNECTION_CLOSED):
            # The SDK turns a 404 into "Session terminated"; the HTTP status says more.
            return _status_error(trace.status, trace.reason, self.display_url)
        if error.error.code == types.CONNECTION_CLOSED:
            if session_failure is not None:
                return describe_failure(session_failure, self.display_url)
            return TransportError(f"{self.display_url} closed the connection", "protocol")
        return ServerRejectedError(f"{error.error.message} (JSON-RPC error {error.error.code})", error.error.code)


def _retry_before_answer(error: TransportError) -> bool:
    return error.kind in ("connect", "timeout") or (error.kind == "status" and error.status in RETRYABLE_STATUS)


def _retry_on_connect(error: TransportError) -> bool:
    return error.kind == "connect"


def describe_failure(error: BaseException, display_url: str, trace: _HttpTrace | None = None) -> TransportError:
    """Turn whatever the SDK raised, often an ``ExceptionGroup``, into one readable line."""
    leaves = list(_leaves(error))
    for leaf in leaves:
        if isinstance(leaf, httpx.HTTPStatusError):
            return _status_error(leaf.response.status_code, leaf.response.reason_phrase, display_url)
    for leaf in leaves:
        if isinstance(leaf, httpx.ConnectError | httpx.ConnectTimeout):
            return TransportError(f"cannot connect to {display_url}: {_short(leaf)}", "connect")
    for leaf in leaves:
        if isinstance(leaf, httpx.TimeoutException):
            return TransportError(f"timed out waiting for {display_url}", "timeout")
    for leaf in leaves:
        if isinstance(leaf, httpx.RequestError):
            return TransportError(f"request to {display_url} failed: {_short(leaf)}", "protocol")
    if trace is not None and trace.status is not None:
        return _status_error(trace.status, trace.reason, display_url)
    for leaf in leaves:
        if isinstance(leaf, McpError):
            return TransportError(f"{display_url}: {leaf.error.message}", "protocol")
    first = leaves[0] if leaves else error
    return TransportError(f"talking to {display_url} failed: {type(first).__name__}: {_short(first)}", "other")


def _status_error(status: int, reason: str, display_url: str) -> TransportError:
    message = f"HTTP {status} {reason} from {display_url}".replace("  ", " ")
    if status in (401, 403):
        message += " (the server refused the credentials; check the headers in the config)"
    elif status == 404:
        message += " (check the server URL in the config)"
    return TransportError(message, "status", status)


def _short(error: BaseException) -> str:
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return text[:300]


def _leaves(error: BaseException) -> Iterator[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        for inner in error.exceptions:
            yield from _leaves(inner)
        return
    if isinstance(error, asyncio.CancelledError):
        return
    yield error
    cause = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    if cause is not None and cause is not error:
        yield from _leaves(cause)


class PortalProvider(AbstractContextManager["PortalProvider"]):
    """Starts the background event loop on first use and stops it on exit."""

    def __init__(self) -> None:
        self._manager: AbstractContextManager[BlockingPortal] | None = None
        self._portal: BlockingPortal | None = None

    def __call__(self) -> BlockingPortal:
        if self._portal is None:
            self._manager = anyio.from_thread.start_blocking_portal()
            self._portal = self._manager.__enter__()
        return self._portal

    def __exit__(self, *exc_info: object) -> None:
        if self._manager is not None:
            self._manager.__exit__(None, None, None)
        self._manager = None
        self._portal = None
