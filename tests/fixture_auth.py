"""An MCP server behind OAuth and its authorization server, in one localhost app, for the OAuth tests.

The app serves the fixture MCP server (``fixture_server.build_server``) at ``/mcp`` behind a
bearer check that answers 401 with a ``WWW-Authenticate`` challenge, the protected resource
metadata, and a small authorization server: its metadata, ``/authorize`` (which approves at once
and redirects, as a user who clicks "Allow" would), ``/token`` and ``/revoke``. The switches on
``AuthorizationServer`` make each part misbehave the way the tests need.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlsplit

import uvicorn
from fixture_server import AsgiApp, Message, Receive, Recorder, Scope, Send, build_server

CLIENT_ID = "test-client.apps.example.com"

CLIENT_SECRET = "test-client-secret-value"

EMAIL = "user@example.com"


def id_token(exp: float, email: str = EMAIL) -> str:
    """An unsigned JWT with the claims the kit reads."""

    def part(content: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(content).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part({'exp': int(exp), 'email': email, 'aud': CLIENT_ID})}.signature"


@dataclass
class AuthorizationServer:
    """The state and the switches of the fake authorization server and the bearer check."""

    origin: str = ""
    accept: Literal["access_token", "id_token"] = "access_token"
    expires_in: int = 3600
    rotate_refresh_tokens: bool = False
    # Non-standard lifetimes of the refresh token added to every token response, such as
    # {"refresh_token_expires_in": 604800} as Google sends, or {"refresh_expires_in": 1800} as Keycloak does
    refresh_lifetime: dict[str, int] = field(default_factory=dict)
    issue_id_token: bool = True
    # error= in the redirect instead of a code
    deny: str | None = None
    tamper_state: bool = False
    # An OAuth error for the next refresh grants
    refresh_error: str | None = None
    # An HTTP status for the next token requests, consumed one per request
    token_status: list[int] = field(default_factory=list)
    # Valid bearers to reject with 401 anyway, consumed one per MCP request
    reject_next: int = 0
    challenge: bool = True
    publish_resource_metadata: bool = True
    code_challenge_methods: list[str] = field(default_factory=lambda: ["S256"])
    metadata_issuer: str | None = None

    authorize_requests: list[dict[str, str]] = field(default_factory=list)
    token_requests: list[dict[str, str]] = field(default_factory=list)
    revoke_requests: list[dict[str, str]] = field(default_factory=list)
    metadata_requests: list[str] = field(default_factory=list)
    bearers: list[str | None] = field(default_factory=list)

    codes: dict[str, dict[str, str]] = field(default_factory=dict)
    access_tokens: set[str] = field(default_factory=set)
    id_tokens: set[str] = field(default_factory=set)
    refresh_tokens: set[str] = field(default_factory=set)

    def reset(self) -> None:
        """Forget what earlier tests did and set every switch back."""
        self.__dict__.update(AuthorizationServer(origin=self.origin).__dict__)

    def refresh_grants(self) -> list[dict[str, str]]:
        return [request for request in self.token_requests if request.get("grant_type") == "refresh_token"]


@dataclass
class ProtectedServer:
    """The app listening on localhost."""

    url: str
    origin: str
    recorder: Recorder
    auth: AuthorizationServer
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


class App:
    """Routes the requests: OAuth endpoints here, the rest to the bearer check and the MCP app."""

    def __init__(self, mcp: AsgiApp, auth: AuthorizationServer) -> None:
        self.mcp = mcp
        self.auth = auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.mcp(scope, receive, send)
            return
        path: str = scope["path"]
        query = {key: values[0] for key, values in parse_qs(scope["query_string"].decode()).items()}
        auth = self.auth
        if path.startswith("/.well-known/"):
            auth.metadata_requests.append(path)
            await self._metadata(path, send)
        elif path == "/authorize":
            auth.authorize_requests.append(query)
            await self._authorize(query, send)
        elif path == "/token":
            form = await _form(receive)
            auth.token_requests.append(form)
            await self._token(form, send)
        elif path == "/revoke":
            form = await _form(receive)
            auth.revoke_requests.append(form)
            auth.refresh_tokens.discard(form.get("token", ""))
            auth.access_tokens.discard(form.get("token", ""))
            await _respond(send, 200, b"", "text/plain")
        else:
            await self._protected(scope, receive, send)

    async def _metadata(self, path: str, send: Send) -> None:
        origin = self.auth.origin
        if path in ("/.well-known/oauth-protected-resource/mcp", "/.well-known/oauth-protected-resource"):
            if not self.auth.publish_resource_metadata or path.endswith("/mcp"):
                await _respond(send, 404, b"not found", "text/plain")
                return
            await _json(
                send,
                200,
                {"resource": origin, "authorization_servers": [origin], "scopes_supported": ["openid", "email"]},
            )
        elif path == "/.well-known/oauth-authorization-server":
            await _json(
                send,
                200,
                {
                    "issuer": self.auth.metadata_issuer or origin,
                    "authorization_endpoint": f"{origin}/authorize",
                    "token_endpoint": f"{origin}/token",
                    "revocation_endpoint": f"{origin}/revoke",
                    "code_challenge_methods_supported": self.auth.code_challenge_methods,
                },
            )
        else:
            await _respond(send, 404, b"not found", "text/plain")

    async def _authorize(self, query: dict[str, str], send: Send) -> None:
        state = query.get("state", "")
        if self.auth.tamper_state:
            state = "tampered"
        if self.auth.deny is not None:
            params = {"error": self.auth.deny, "state": state}
        else:
            code = secrets.token_urlsafe(16)
            self.auth.codes[code] = query
            params = {"code": code, "state": state}
        location = f"{query['redirect_uri']}?{urlencode(params)}"
        await send({"type": "http.response.start", "status": 302, "headers": [(b"location", location.encode())]})
        await send({"type": "http.response.body", "body": b""})

    async def _token(self, form: dict[str, str], send: Send) -> None:
        auth = self.auth
        if auth.token_status:
            await _respond(send, auth.token_status.pop(0), b"injected failure", "text/plain")
            return
        if form.get("client_id") != CLIENT_ID or form.get("client_secret") != CLIENT_SECRET:
            await _json(send, 401, {"error": "invalid_client"})
            return
        grant = form.get("grant_type")
        if grant == "authorization_code":
            request = auth.codes.pop(form.get("code", ""), None)
            verifier = form.get("code_verifier", "").encode()
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier).digest()).rstrip(b"=").decode()
            if request is None or request["redirect_uri"] != form.get("redirect_uri"):
                await _json(send, 400, {"error": "invalid_grant", "error_description": "unknown code"})
                return
            if request["code_challenge"] != challenge:
                await _json(send, 400, {"error": "invalid_grant", "error_description": "PKCE verification failed"})
                return
            await _json(send, 200, self._issue(refresh=True))
        elif grant == "refresh_token":
            if auth.refresh_error is not None:
                await _json(send, 400, {"error": auth.refresh_error})
                return
            if form.get("refresh_token") not in auth.refresh_tokens:
                await _json(send, 400, {"error": "invalid_grant"})
                return
            if auth.rotate_refresh_tokens:
                auth.refresh_tokens.discard(form["refresh_token"])
            await _json(send, 200, self._issue(refresh=auth.rotate_refresh_tokens))
        else:
            await _json(send, 400, {"error": "unsupported_grant_type"})

    def _issue(self, *, refresh: bool) -> dict[str, Any]:
        auth = self.auth
        access = f"at-{secrets.token_urlsafe(16)}"
        auth.access_tokens.add(access)
        payload: dict[str, Any] = {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": auth.expires_in,
            "scope": "openid email",
            **auth.refresh_lifetime,
        }
        if auth.issue_id_token:
            token = id_token(time.time() + auth.expires_in)
            auth.id_tokens.add(token)
            payload["id_token"] = token
        if refresh:
            token = f"rt-{secrets.token_urlsafe(16)}"
            auth.refresh_tokens.add(token)
            payload["refresh_token"] = token
        return payload

    async def _protected(self, scope: Scope, receive: Receive, send: Send) -> None:
        auth = self.auth
        headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
        header = headers.get("authorization", "")
        bearer = header[len("Bearer ") :] if header.startswith("Bearer ") else None
        auth.bearers.append(bearer)
        accepted = auth.access_tokens if auth.accept == "access_token" else auth.id_tokens
        valid = bearer is not None and bearer in accepted
        if valid and auth.reject_next > 0:
            auth.reject_next -= 1
            valid = False
        if not valid:
            challenge = f'Bearer resource_metadata="{auth.origin}/.well-known/oauth-protected-resource", scope="openid"'
            extra = [(b"www-authenticate", challenge.encode())] if auth.challenge else []
            await send(
                {"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain"), *extra]}
            )
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.mcp(scope, receive, send)


async def _form(receive: Receive) -> dict[str, str]:
    body = b""
    while True:
        message: Message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    return {key: values[0] for key, values in parse_qs(body.decode()).items()}


async def _respond(send: Send, status: int, body: bytes, content_type: str) -> None:
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", content_type.encode())]})
    await send({"type": "http.response.body", "body": body})


async def _json(send: Send, status: int, content: dict[str, Any]) -> None:
    await _respond(send, status, json.dumps(content).encode(), "application/json")


def start() -> ProtectedServer:
    """Start the protected server on a free port."""
    mcp = build_server(json_response=True)
    recorder = Recorder(mcp.streamable_http_app(), sse=False)
    auth = AuthorizationServer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    auth.origin = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(
        App(recorder, auth), log_level="error", lifespan="on", ws="none", timeout_graceful_shutdown=1
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the protected fixture server did not start")
        time.sleep(0.01)
    return ProtectedServer(f"{auth.origin}/mcp", auth.origin, recorder, auth, server, thread)


def follow_sign_in(address: str) -> bool:
    """Stand in for the browser: follow the authorization server's redirect to the CLI's loopback listener."""
    import httpx

    parts = urlsplit(address)
    assert parts.scheme == "http", address
    with httpx.Client(follow_redirects=True, timeout=10) as http:
        response = http.get(address)
    assert response.status_code == 200, response.text
    return True
