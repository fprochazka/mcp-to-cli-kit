"""Sign in to an MCP server with OAuth 2.1 and keep its tokens fresh across invocations.

The flow follows the MCP authorization spec with a pre-registered client: the server's protected
resource metadata names the authorization server, whose metadata gives the endpoints; the user
signs in through the browser with the authorization code grant, PKCE (S256) and a loopback
redirect, and the RFC 8707 ``resource`` parameter goes with every grant.

Sign-in is an explicit command. Every other command only reads the stored tokens, refreshes
them when less than ``REFRESH_MARGIN_SECONDS`` remain, and fails with ``AuthError`` when there
is nothing to refresh, so a command run by an AI agent never waits for a browser.

The tokens of each server are a JSON file in ``$XDG_STATE_HOME/<bin>/oauth/`` (by default
``~/.local/state/<bin>/oauth/``), mode 0600. The file also keeps the token and revocation
endpoints, so a refresh is one request. It records which server URL and client it was issued
for; a file that does not match the configuration counts as not signed in, so a token is never
sent to another server. Invocations that run in parallel refresh under a file lock: one of them
refreshes, the others read its result.

The SDK's ``OAuthClientProvider`` is not used. It is built for a long-lived process: after a
restart it refreshes at ``<server origin>/token`` instead of the discovered endpoint, it loses
the expiry of stored tokens, it starts the browser flow inside whatever request got a 401, and
it cannot send an ID token.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import json
import logging
import os
import secrets
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, fields
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import httpx

from mcp_to_cli_kit import redaction
from mcp_to_cli_kit.config import ResolvedOAuth, xdg_directory
from mcp_to_cli_kit.files import write_private

if sys.platform != "win32":
    import fcntl

logger = logging.getLogger(__name__)

# Refresh this long before the token expires, so it cannot expire while a request is in flight.
REFRESH_MARGIN_SECONDS = 300

DEFAULT_LOGIN_TIMEOUT_SECONDS = 300.0

# How long to wait for another invocation's refresh before refreshing without the lock.
LOCK_TIMEOUT_SECONDS = 30.0

CALLBACK_PATH = "/callback"

# The token file format. A field that an older version can do without is added without a new version;
# the version changes only when an older version would misread the file.
_FORMAT_VERSION = 1

Bearer = Literal["access_token", "id_token"]


class AuthError(Exception):
    """Signing in failed, or the CLI is not signed in. The message is one line, safe to print, and says what to do."""


class AuthTransportError(Exception):
    """The authorization server or the metadata could not be reached. The message is safe to print.

    Attributes:
        status: The HTTP status, when the server answered with a 5xx or 429.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _GrantError(AuthError):
    """The token endpoint answered with an OAuth error."""

    def __init__(self, message: str, error: str) -> None:
        super().__init__(message)
        self.error = error


def _open_with_webbrowser(url: str) -> bool:
    # Without a display, webbrowser may start a text browser inside the terminal the CLI runs in.
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return webbrowser.open(url)


# Opens the sign-in address. Tests replace it with a function that follows the redirects itself.
open_browser: Callable[[str], bool] = _open_with_webbrowser


def _try_open_browser(address: str) -> bool:
    try:
        return open_browser(address)
    except Exception as error:  # noqa: BLE001 - the address is printed anyway
        logger.debug("Cannot open a browser: %s", error)
        return False


def default_token_dir(bin_name: str) -> Path:
    """Return ``$XDG_STATE_HOME/<bin>/oauth``, by default ``~/.local/state/<bin>/oauth``."""
    return xdg_directory("XDG_STATE_HOME", ".local/state") / bin_name / "oauth"


@dataclass(frozen=True)
class AuthorizationServer:
    """What discovery found: the endpoints, and what to ask for.

    Attributes:
        resource: The RFC 8707 resource identifier of the MCP server.
        scopes: The scopes to ask for, None to leave the parameter out.
    """

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str | None
    resource: str
    scopes: list[str] | None


@dataclass(frozen=True)
class StoredTokens:
    """The tokens of one server and what they were issued for. Holds secrets; never print it.

    Attributes:
        server_url_sha256: The hash of the expanded server URL the tokens were issued for.
        expires_at: When the access token expires, in seconds since the epoch; None when unknown.
        id_token_expires_at: When the ID token expires, from its ``exp`` claim.
        obtained_at: When the last token response came, at the sign-in or a refresh.
        identity: The ``email`` claim of the ID token, to tell the user who is signed in.
        refresh_expires_at: When the refresh token expires, from the non-standard
            ``refresh_token_expires_in`` (Google) or ``refresh_expires_in`` (Keycloak) of the token
            response; None when the response did not say, or there is no refresh token.
        signed_in_at: When the user signed in through the browser; refreshes keep it. None for
            tokens stored before the field existed.
    """

    server_url_sha256: str
    issuer: str
    client_id: str
    resource: str | None
    token_endpoint: str
    revocation_endpoint: str | None
    scope: str | None
    access_token: str
    id_token: str | None
    refresh_token: str | None
    expires_at: float | None
    id_token_expires_at: float | None
    obtained_at: float
    identity: str | None
    refresh_expires_at: float | None = None
    signed_in_at: float | None = None
    version: int = field(default=_FORMAT_VERSION)

    def bearer(self, kind: Bearer) -> str | None:
        """The token to send as the bearer."""
        return self.access_token if kind == "access_token" else self.id_token

    def expiry(self, kind: Bearer) -> float | None:
        """When the token sent as the bearer expires; None when unknown."""
        return self.expires_at if kind == "access_token" else self.id_token_expires_at

    def is_fresh(self, kind: Bearer, now: float | None = None) -> bool:
        """Whether the bearer exists and stays valid for at least ``REFRESH_MARGIN_SECONDS``."""
        if self.bearer(kind) is None:
            return False
        expiry = self.expiry(kind)
        now = time.time() if now is None else now
        return expiry is None or expiry - now > REFRESH_MARGIN_SECONDS

    def is_signed_in(self, kind: Bearer, now: float | None = None) -> bool:
        """Whether a command can get a bearer without a new sign-in: it is fresh, or a refresh token is left to get one.

        A refresh token counts until its known expiry. The authorization server can still end it
        earlier, which only the next refresh finds out.
        """
        now = time.time() if now is None else now
        if self.is_fresh(kind, now):
            return True
        return self.refresh_token is not None and (self.refresh_expires_at is None or self.refresh_expires_at > now)

    def signed_in_until(self, kind: Bearer) -> float | None:
        """When the sign-in ends unless the user signs in again.

        With a refresh token it is the refresh token's expiry, None when the authorization server
        did not say: then the sign-in lasts until the server ends it, for example after a
        revocation or a long time unused. Without a refresh token it is the bearer's expiry.
        """
        return self.refresh_expires_at if self.refresh_token is not None else self.expiry(kind)

    def is_for(self, url: str, auth: ResolvedOAuth) -> bool:
        """Whether the tokens were issued for this server URL, client and, when configured, issuer."""
        issuer = auth.config.authorization_server
        return (
            self.server_url_sha256 == url_hash(url)
            and self.client_id == auth.client_id
            and (issuer is None or _same_url(issuer, self.issuer))
        )

    def register_secrets(self) -> None:
        """Keep the tokens out of error messages and logs from now on."""
        for token in (self.access_token, self.id_token, self.refresh_token):
            if token:
                redaction.register(token, "***")


_FIELDS = frozenset(item.name for item in fields(StoredTokens))


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class TokenStore:
    """Reads and writes the token file of each server of one bin."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path(self, server: str) -> Path:
        """The token file of one server."""
        return self.directory / f"{server}.json"

    def load(self, server: str) -> StoredTokens | None:
        """The stored tokens of a server, or None when there are none or the file cannot be read."""
        path = self.path(server)
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            logger.debug("Ignoring unreadable token file %s: %s", path, error)
            return None
        try:
            if not isinstance(content, dict) or content.get("version") != _FORMAT_VERSION:
                raise TypeError("unknown format")
            # Fields a later version added are left out, so going back to an older version keeps the sign-in.
            tokens = StoredTokens(**{key: value for key, value in content.items() if key in _FIELDS})
            if not isinstance(tokens.access_token, str) or not isinstance(tokens.token_endpoint, str):
                raise TypeError("missing fields")
        except TypeError as error:
            logger.debug("Ignoring token file %s: %s", path, error)
            return None
        tokens.register_secrets()
        return tokens

    def save(self, server: str, tokens: StoredTokens) -> None:
        """Write a server's tokens atomically, readable by the owner only.

        Raises:
            AuthError: The file cannot be written.
        """
        tokens.register_secrets()
        try:
            write_private(self.path(server), json.dumps(asdict(tokens), indent=2) + "\n")
        except OSError as error:
            raise AuthError(f"cannot store the tokens in {self.path(server)}: {error.strerror or error}") from None

    def delete(self, server: str) -> bool:
        """Delete a server's tokens. Returns whether there were any."""
        path = self.path(server)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    @contextlib.contextmanager
    def lock(self, server: str) -> Iterator[None]:
        """Hold the lock of a server's token file, so parallel invocations refresh once.

        After ``LOCK_TIMEOUT_SECONDS`` the caller goes on without the lock: a second refresh is
        better than a command that hangs on a lock a stuck process holds.
        """
        if sys.platform == "win32":
            yield
            return
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(self.directory / f"{server}.json.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        logger.warning("Refreshing the tokens of %s without the lock another process holds", server)
                        break
                    time.sleep(0.05)
            yield
        finally:
            os.close(descriptor)  # closing releases the lock


class TokenManager:
    """The bearer of one server for one invocation, refreshed when needed."""

    def __init__(
        self,
        server: str,
        url: str,
        auth: ResolvedOAuth,
        store: TokenStore,
        login_command: str,
        http_timeout: float = 30.0,
    ) -> None:
        self.server = server
        self.url = url
        self.auth = auth
        self.store = store
        self.login_command = login_command
        self.http_timeout = http_timeout

    @property
    def kind(self) -> Bearer:
        return self.auth.config.bearer

    def bearer(self, rejected: str | None = None) -> str:
        """The bearer to send, refreshed first when it expires soon or the server rejected it.

        Args:
            rejected: A bearer the server answered with 401. It is refreshed unless another
                invocation already replaced it.

        Raises:
            AuthError: Not signed in, the sign-in expired, or the refresh was refused.
            AuthTransportError: The token endpoint could not be reached.
        """
        tokens = self._load()
        if rejected is None and tokens.is_fresh(self.kind):
            return self._use(tokens)
        with self.store.lock(self.server):
            tokens = self._load()
            current = tokens.bearer(self.kind)
            if tokens.is_fresh(self.kind) and current != rejected:
                return self._use(tokens)
            return self._use(self._refresh(tokens))

    def _load(self) -> StoredTokens:
        tokens = self.store.load(self.server)
        if tokens is None or not tokens.is_for(self.url, self.auth):
            raise AuthError(
                f"not signed in to {self.server}. Run '{self.login_command}' in a terminal; it opens a browser to "
                "sign in. An AI agent should ask the user to run it."
            )
        return tokens

    def _use(self, tokens: StoredTokens) -> str:
        bearer = tokens.bearer(self.kind)
        if bearer is None:
            raise AuthError(
                f"the authorization server of {self.server} issued no ID token. Ask for the openid scope in "
                "auth.scopes, or set auth.bearer to access_token."
            )
        return bearer

    def _refresh(self, tokens: StoredTokens) -> StoredTokens:
        if tokens.refresh_token is None:
            raise AuthError(f"the sign-in to {self.server} expired. Run '{self.login_command}' to sign in again.")
        logger.debug("Refreshing the tokens of %s", self.server)
        data = {"grant_type": "refresh_token", "refresh_token": tokens.refresh_token}
        if self.auth.config.resource and tokens.resource is not None:
            data["resource"] = tokens.resource
        try:
            with httpx.Client(timeout=self.http_timeout) as http:
                payload = _token_request(http, tokens.token_endpoint, data, self.auth)
        except _GrantError as error:
            if error.error == "invalid_grant":
                self.store.delete(self.server)
                raise AuthError(
                    f"the sign-in to {self.server} expired or was revoked. Run '{self.login_command}' to sign in again."
                ) from None
            raise
        refreshed = _tokens_from_response(payload, tokens, previous=tokens)
        self.store.save(self.server, refreshed)
        return refreshed


def login(
    server: str,
    url: str,
    auth: ResolvedOAuth,
    store: TokenStore,
    *,
    echo: Callable[[str], None],
    browser: bool = True,
    wait_timeout: float = DEFAULT_LOGIN_TIMEOUT_SECONDS,
    http_timeout: float = 30.0,
) -> StoredTokens:
    """Sign in through the browser and store the tokens.

    Args:
        server: The server's name, for messages and the token file.
        url: The expanded server URL.
        auth: The server's OAuth settings, expanded.
        echo: Prints a line for the user, on stderr.
        browser: Whether to open the sign-in address in a browser; it is printed either way.
        wait_timeout: How long to wait for the browser to come back to the loopback address.

    Raises:
        AuthError: Discovery found no usable authorization server, the user did not sign in, or
            the authorization server refused the grant.
        AuthTransportError: A server could not be reached.
    """
    with httpx.Client(timeout=http_timeout) as http:
        found = discover(http, url, auth)
        listener = _CallbackListener("127.0.0.1", auth.config.redirect_port, CALLBACK_PATH)
        request = _authorization_request(found, auth, f"http://127.0.0.1:{listener.port}{CALLBACK_PATH}")

        echo(f"Sign in to {server} in the browser. If it does not open, open this address:\n{request.address}")
        if browser and not _try_open_browser(request.address):
            echo("Could not open a browser; open the address above.")
        result = listener.wait(wait_timeout)

        if not secrets.compare_digest(result.get("state", ""), request.state):
            raise AuthError("the sign-in answer does not belong to this sign-in (the state differs); try again")
        if "error" in result:
            description = result.get("error_description")
            raise AuthError(f"the sign-in failed: {result['error']}" + (f" ({description})" if description else ""))
        code = result.get("code")
        if not code:
            raise AuthError("the sign-in answer carries no authorization code; try again")

        return _exchange_code(http, server, url, auth, store, request, code)


@dataclass(frozen=True)
class _AuthorizationRequest:
    """What one sign-in sends the browser to, and what it needs to exchange the code it gets back."""

    found: AuthorizationServer
    verifier: str
    state: str
    redirect_uri: str
    address: str


def _authorization_request(found: AuthorizationServer, auth: ResolvedOAuth, redirect_uri: str) -> _AuthorizationRequest:
    """Build the authorization address for a discovered authorization server, with a fresh PKCE verifier and state."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=")
    state = secrets.token_urlsafe(32)

    params = {
        "response_type": "code",
        "client_id": auth.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge.decode("ascii"),
        "code_challenge_method": "S256",
    }
    if found.scopes:
        params["scope"] = " ".join(found.scopes)
    if auth.config.resource:
        params["resource"] = found.resource
    separator = "&" if "?" in found.authorization_endpoint else "?"
    address = found.authorization_endpoint + separator + urlencode(params)
    return _AuthorizationRequest(
        found=found, verifier=verifier, state=state, redirect_uri=redirect_uri, address=address
    )


def _exchange_code(
    http: httpx.Client,
    server: str,
    url: str,
    auth: ResolvedOAuth,
    store: TokenStore,
    request: _AuthorizationRequest,
    code: str,
) -> StoredTokens:
    """Exchange the authorization code for tokens and store them.

    Raises:
        AuthError: The authorization server refused the grant.
        AuthTransportError: It could not be reached.
    """
    found = request.found
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": request.redirect_uri,
        "code_verifier": request.verifier,
    }
    if auth.config.resource:
        data["resource"] = found.resource
    payload = _token_request(http, found.token_endpoint, data, auth)

    base = StoredTokens(
        server_url_sha256=url_hash(url),
        issuer=found.issuer,
        client_id=auth.client_id,
        resource=found.resource if auth.config.resource else None,
        token_endpoint=found.token_endpoint,
        revocation_endpoint=found.revocation_endpoint,
        scope=" ".join(found.scopes) if found.scopes else None,
        access_token="",
        id_token=None,
        refresh_token=None,
        expires_at=None,
        id_token_expires_at=None,
        obtained_at=0.0,
        identity=None,
    )
    tokens = _tokens_from_response(payload, base, previous=None)
    store.save(server, tokens)
    return tokens


def revoke(tokens: StoredTokens, auth: ResolvedOAuth, http_timeout: float = 30.0) -> bool:
    """Revoke the grant at the authorization server (RFC 7009). Returns False when it has no revocation endpoint.

    Raises:
        AuthError: The authorization server refused the revocation.
        AuthTransportError: It could not be reached.
    """
    if tokens.revocation_endpoint is None:
        return False
    data = {"token": tokens.refresh_token or tokens.access_token, "client_id": auth.client_id}
    if auth.client_secret is not None:
        data["client_secret"] = auth.client_secret
    try:
        with httpx.Client(timeout=http_timeout) as http:
            response = http.post(tokens.revocation_endpoint, data=data)
    except httpx.HTTPError as error:
        raise AuthTransportError(f"cannot reach {_display(tokens.revocation_endpoint)}: {_short(error)}") from None
    if response.status_code >= 500:
        raise AuthTransportError(f"HTTP {response.status_code} from {_display(tokens.revocation_endpoint)}")
    if response.status_code >= 400:
        raise AuthError(f"{_display(tokens.revocation_endpoint)} refused the revocation: {_oauth_error(response)}")
    return True


def discover(http: httpx.Client, url: str, auth: ResolvedOAuth) -> AuthorizationServer:
    """Find the authorization server of an MCP server and what to ask it for.

    The server's 401 names its protected resource metadata, else the well-known addresses are
    tried, the one with the endpoint's path first. ``auth.authorization_server`` skips that
    step. The authorization server's metadata is then read from the RFC 8414 or the OpenID
    Connect address; it must name the expected issuer and support PKCE with S256.

    Raises:
        AuthError: The metadata is missing or does not fit.
        AuthTransportError: A server could not be reached, or answered with a 5xx.
    """
    challenge_scopes: list[str] | None = None
    resource = _canonical(url)
    supported_scopes: list[str] | None = None
    issuer = auth.config.authorization_server
    if issuer is None:
        metadata_url, challenge_scopes = _challenge(http, url)
        metadata = _protected_resource(http, url, metadata_url)
        servers = metadata.get("authorization_servers")
        if not isinstance(servers, list) or not servers or not isinstance(servers[0], str):
            raise AuthError(f"the protected resource metadata of {_display(url)} names no authorization server")
        issuer = servers[0]
        announced = metadata.get("resource")
        if isinstance(announced, str) and announced:
            if not _covers(announced, url):
                raise AuthError(f"the protected resource metadata is for {announced}, not for {_display(url)}")
            resource = announced
        scopes = metadata.get("scopes_supported")
        if isinstance(scopes, list) and all(isinstance(scope, str) for scope in scopes):
            supported_scopes = scopes or None

    server_metadata = _authorization_server_metadata(http, issuer)
    endpoints = {}
    for key in ("authorization_endpoint", "token_endpoint"):
        value = server_metadata.get(key)
        if not isinstance(value, str) or not value:
            raise AuthError(f"the metadata of {issuer} has no {key}")
        endpoints[key] = value
    if "S256" not in (server_metadata.get("code_challenge_methods_supported") or []):
        raise AuthError(f"{issuer} does not announce PKCE with S256, which the MCP authorization spec requires")
    revocation = server_metadata.get("revocation_endpoint")
    return AuthorizationServer(
        issuer=issuer,
        authorization_endpoint=endpoints["authorization_endpoint"],
        token_endpoint=endpoints["token_endpoint"],
        revocation_endpoint=revocation if isinstance(revocation, str) and revocation else None,
        resource=resource,
        scopes=auth.config.scopes or challenge_scopes or supported_scopes,
    )


def _challenge(http: httpx.Client, url: str) -> tuple[str | None, list[str] | None]:
    """The metadata address and scopes from the server's 401, or Nones. A GET is safe on an MCP endpoint."""
    try:
        with http.stream("GET", url, headers={"Accept": "text/event-stream"}) as response:
            if response.status_code != 401:
                return None, None
            header = response.headers.get("www-authenticate", "")
    except httpx.HTTPError as error:
        raise AuthTransportError(f"cannot reach {_display(url)}: {_short(error)}") from None
    params = _challenge_params(header)
    scopes = params.get("scope", "").split()
    return params.get("resource_metadata"), scopes or None


def _challenge_params(header: str) -> dict[str, str]:
    """The parameters of a ``Bearer`` challenge in a WWW-Authenticate header."""
    params: dict[str, str] = {}
    position = header.lower().find("bearer")
    if position < 0:
        return params
    rest = header[position + len("bearer") :]
    index = 0
    while index < len(rest):
        while index < len(rest) and rest[index] in " ,\t":
            index += 1
        equals = rest.find("=", index)
        if equals < 0:
            break
        name = rest[index:equals].strip().lower()
        index = equals + 1
        if index < len(rest) and rest[index] == '"':
            end = index + 1
            value = []
            while end < len(rest) and rest[end] != '"':
                if rest[end] == "\\" and end + 1 < len(rest):
                    end += 1
                value.append(rest[end])
                end += 1
            params.setdefault(name, "".join(value))
            index = end + 1
        else:
            end = rest.find(",", index)
            end = len(rest) if end < 0 else end
            params.setdefault(name, rest[index:end].strip())
            index = end
    return params


def _protected_resource(http: httpx.Client, url: str, from_challenge: str | None) -> dict[str, Any]:
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    candidates = [from_challenge] if from_challenge else []
    path = parts.path.rstrip("/")
    if path:
        candidates.append(f"{origin}/.well-known/oauth-protected-resource{path}")
    candidates.append(f"{origin}/.well-known/oauth-protected-resource")
    for candidate in candidates:
        metadata = _get_json(http, candidate)
        if metadata is not None:
            return metadata
    raise AuthError(
        f"{_display(url)} publishes no OAuth protected resource metadata; set auth.authorization_server in the config"
    )


def _authorization_server_metadata(http: httpx.Client, issuer: str) -> dict[str, Any]:
    parts = urlsplit(issuer)
    origin = f"{parts.scheme}://{parts.netloc}"
    path = parts.path.rstrip("/")
    if path:
        candidates = [
            f"{origin}/.well-known/oauth-authorization-server{path}",
            f"{origin}/.well-known/openid-configuration{path}",
            f"{origin}{path}/.well-known/openid-configuration",
        ]
    else:
        candidates = [f"{origin}/.well-known/oauth-authorization-server", f"{origin}/.well-known/openid-configuration"]
    for candidate in candidates:
        metadata = _get_json(http, candidate)
        if metadata is None:
            continue
        announced = metadata.get("issuer")
        if not isinstance(announced, str) or not _same_url(announced, issuer):
            raise AuthError(f"the metadata at {candidate} is for the issuer {announced!r}, not {issuer}")
        return metadata
    raise AuthError(f"{issuer} publishes no OAuth authorization server metadata")


def _get_json(http: httpx.Client, url: str) -> dict[str, Any] | None:
    """A JSON object from a metadata address, or None when it has none (any 4xx, or not JSON)."""
    try:
        response = http.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError as error:
        raise AuthTransportError(f"cannot reach {_display(url)}: {_short(error)}") from None
    if response.status_code >= 500 or response.status_code == 429:
        raise AuthTransportError(f"HTTP {response.status_code} from {_display(url)}", response.status_code)
    if response.status_code != 200:
        logger.debug("No metadata at %s: HTTP %d", url, response.status_code)
        return None
    try:
        content = response.json()
    except ValueError:
        logger.debug("No metadata at %s: not JSON", url)
        return None
    return content if isinstance(content, dict) else None


def _token_request(http: httpx.Client, endpoint: str, data: dict[str, str], auth: ResolvedOAuth) -> dict[str, Any]:
    form = {**data, "client_id": auth.client_id}
    if auth.client_secret is not None:
        form["client_secret"] = auth.client_secret
    try:
        response = http.post(endpoint, data=form, headers={"Accept": "application/json"})
    except httpx.HTTPError as error:
        raise AuthTransportError(f"cannot reach the token endpoint {_display(endpoint)}: {_short(error)}") from None
    if response.status_code >= 500 or response.status_code == 429:
        raise AuthTransportError(
            f"HTTP {response.status_code} from the token endpoint {_display(endpoint)}", response.status_code
        )
    if response.status_code != 200:
        raise _GrantError(f"the token endpoint refused the grant: {_oauth_error(response)}", _error_code(response))
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise AuthError(f"the token endpoint {_display(endpoint)} answered without an access token")
    return payload


def _tokens_from_response(payload: dict[str, Any], base: StoredTokens, previous: StoredTokens | None) -> StoredTokens:
    """The tokens a token response carries, on top of what ``base`` says they are for.

    A refresh response without a refresh token keeps the previous one; one with a new refresh
    token replaces it, as an authorization server that rotates them expects. The refresh token's
    expiry follows the same rule: a response that does not state it keeps the previous one only
    with the previous refresh token, as a new refresh token may have a lifetime of its own.
    ``signed_in_at`` is now at the sign-in and the previous one on a refresh.
    """
    now = time.time()
    expires_in = _positive_seconds(payload.get("expires_in"))
    refresh_expires_in = _positive_seconds(payload.get("refresh_token_expires_in", payload.get("refresh_expires_in")))
    id_token = payload.get("id_token") if isinstance(payload.get("id_token"), str) else None
    claims = _jwt_claims(id_token) if id_token else {}
    exp = claims.get("exp")
    email = claims.get("email")
    refresh = payload.get("refresh_token")
    scope = payload.get("scope")
    refresh_token = refresh if isinstance(refresh, str) else previous.refresh_token if previous else None
    if refresh_expires_in is not None:
        refresh_expires_at: float | None = now + refresh_expires_in
    elif previous is not None and refresh_token is not None and refresh_token == previous.refresh_token:
        refresh_expires_at = previous.refresh_expires_at
    else:
        refresh_expires_at = None
    tokens = StoredTokens(
        server_url_sha256=base.server_url_sha256,
        issuer=base.issuer,
        client_id=base.client_id,
        resource=base.resource,
        token_endpoint=base.token_endpoint,
        revocation_endpoint=base.revocation_endpoint,
        scope=scope if isinstance(scope, str) else base.scope,
        access_token=payload["access_token"],
        id_token=id_token,
        refresh_token=refresh_token,
        expires_at=now + expires_in if expires_in is not None else None,
        id_token_expires_at=float(exp) if isinstance(exp, int | float) else None,
        obtained_at=now,
        identity=email if isinstance(email, str) else previous.identity if previous else None,
        refresh_expires_at=refresh_expires_at if refresh_token is not None else None,
        signed_in_at=previous.signed_in_at if previous is not None else now,
    )
    tokens.register_secrets()
    return tokens


def _positive_seconds(value: Any) -> float | None:
    """A lifetime in seconds from a token response; None when absent, not a number, or not positive.

    Keycloak sends ``refresh_expires_in: 0`` for an offline token that does not expire.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return None
    return float(value)


def _jwt_claims(token: str) -> dict[str, Any]:
    """The payload of a JWT, unverified: only to know when it expires and whom it names."""
    try:
        payload = token.split(".")[1]
        decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _error_code(response: httpx.Response) -> str:
    try:
        content = response.json()
    except ValueError:
        return ""
    error = content.get("error") if isinstance(content, dict) else None
    return error if isinstance(error, str) else ""


def _oauth_error(response: httpx.Response) -> str:
    """``error (error_description)`` of an OAuth error answer, else its status. Never the request."""
    try:
        content = response.json()
    except ValueError:
        content = None
    if isinstance(content, dict) and isinstance(content.get("error"), str):
        description = content.get("error_description")
        suffix = f" ({str(description)[:200]})" if isinstance(description, str) and description else ""
        return redaction.redact(content["error"][:100] + suffix)
    return f"HTTP {response.status_code}"


def _canonical(url: str) -> str:
    """The server URL as an RFC 8707 resource: no query, no fragment, no trailing slash."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _covers(resource: str, url: str) -> bool:
    """Whether a protected resource identifier is the server URL or one of its parents."""
    resource = _canonical(resource)
    target = _canonical(url)
    return target == resource or target.startswith(resource + "/")


def _same_url(left: str, right: str) -> bool:
    return _canonical(left) == _canonical(right)


def _display(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _short(error: BaseException) -> str:
    text = str(error).strip()
    return (text.splitlines()[0] if text else type(error).__name__)[:300]


class _CallbackListener:
    """Waits on a loopback port for the browser to come back from the authorization server."""

    def __init__(self, host: str, port: int, path: str) -> None:
        self.result: dict[str, str] | None = None
        self._received = threading.Event()
        listener = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
                parts = urlsplit(self.path)
                if parts.path != path:
                    self.send_error(404)
                    return
                query = {name: values[0] for name, values in parse_qs(parts.query).items() if values}
                if listener.result is None:
                    listener.result = query
                    listener._received.set()
                if "error" in query:
                    text = f"The sign-in failed: {query['error']}. Go back to the terminal."
                else:
                    text = "The sign-in is complete. You can close this tab and go back to the terminal."
                body = f"<!doctype html><meta charset=utf-8><title>Sign-in</title><p>{html.escape(text)}</p>\n"
                encoded = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the signature http.server calls
                logger.debug("Sign-in listener: %s", redaction.redact(format % args))

        try:
            self._server = HTTPServer((host, port), Handler)
        except OSError as error:
            raise AuthError(
                f"cannot listen on {host}:{port} for the sign-in: {error.strerror or error}. "
                "Free the port, or change auth.redirect_port."
            ) from None
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()

    def wait(self, timeout: float) -> dict[str, str]:
        """The query of the first callback. The listener is closed afterwards in any case.

        Raises:
            AuthError: Nothing came back within ``timeout`` seconds.
        """
        try:
            if not self._received.wait(timeout):
                raise AuthError(f"timed out after {timeout:g}s waiting for the sign-in in the browser")
            assert self.result is not None
            return self.result
        finally:
            self._server.shutdown()
            self._server.server_close()
