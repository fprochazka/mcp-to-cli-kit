"""OAuth sign-in end to end: the auth commands, the bearer on every request, refresh, and the failures.

The tests run against ``fixture_auth``: an MCP server behind a bearer check and its authorization
server, on a localhost port. The browser is replaced by a function that follows the authorization
server's redirect to the CLI's real loopback listener.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from conftest import BIN, Invoke
from fixture_auth import CLIENT_ID, CLIENT_SECRET, EMAIL, ProtectedServer, follow_sign_in

from mcp_to_cli_kit import oauth
from mcp_to_cli_kit.config import ServerConfig, expand_server
from mcp_to_cli_kit.oauth import TokenManager, TokenStore, default_token_dir

Run = Callable[..., Result]


def server_config(server: ProtectedServer, **auth: Any) -> dict[str, Any]:
    settings = {"type": "oauth", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET, **auth}
    return {"url": server.url, "auth": {key: value for key, value in settings.items() if value is not None}}


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The addresses the CLI opened in the "browser", which signs in at once."""
    addresses: list[str] = []

    def browser(address: str) -> bool:
        addresses.append(address)
        return follow_sign_in(address)

    monkeypatch.setattr(oauth, "open_browser", browser)
    return addresses


@pytest.fixture
def run(invoke: Invoke, protected_server: ProtectedServer, opened: list[str]) -> Run:
    """Run the CLI with one OAuth server, ``example``; keyword arguments change its auth settings."""

    def run(args: list[str], env: dict[str, str] | None = None, **auth: Any) -> Result:
        return invoke(args, default={"servers": {"example": server_config(protected_server, **auth)}}, env=env)

    return run


def token_file() -> Path:
    return default_token_dir(BIN) / "example.json"


def stored() -> dict[str, Any]:
    content: dict[str, Any] = json.loads(token_file().read_text(encoding="utf-8"))
    return content


def expire() -> None:
    """Make the stored tokens expire, as if an hour had passed."""
    content = stored()
    content["expires_at"] = content["id_token_expires_at"] = time.time() - 10
    token_file().write_text(json.dumps(content), encoding="utf-8")


def secrets_of(server: ProtectedServer) -> list[str]:
    auth = server.auth
    return [CLIENT_SECRET, *auth.access_tokens, *auth.id_tokens, *auth.refresh_tokens]


def assert_no_secrets(result: Result, server: ProtectedServer) -> None:
    output = result.stdout + result.stderr
    for secret in secrets_of(server):
        assert secret not in output


def sign_in(run: Run, **auth: Any) -> None:
    result = run(["auth", "login"], **auth)
    assert result.exit_code == 0, result.stderr


class TestLogin:
    def test_signs_in_and_stores_the_tokens_for_the_owner_only(
        self, run: Run, protected_server: ProtectedServer, opened: list[str]
    ) -> None:
        result = run(["auth", "login"])

        assert result.exit_code == 0, result.stderr
        assert f"Signed in to example as {EMAIL}." in result.stderr
        assert len(opened) == 1 and opened[0] in result.stderr
        assert_no_secrets(result, protected_server)
        assert stat.S_IMODE(token_file().stat().st_mode) == 0o600
        assert stat.S_IMODE(token_file().parent.stat().st_mode) == 0o700
        assert stored()["refresh_token"] in protected_server.auth.refresh_tokens

    def test_the_grant_follows_the_mcp_authorization_spec(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)

        auth = protected_server.auth
        [authorize] = auth.authorize_requests
        assert authorize["client_id"] == CLIENT_ID
        assert authorize["code_challenge_method"] == "S256"
        assert authorize["redirect_uri"].startswith("http://127.0.0.1:")
        assert authorize["redirect_uri"].endswith("/callback")
        # The scope of the 401 challenge wins over scopes_supported.
        assert authorize["scope"] == "openid"
        assert authorize["resource"] == protected_server.origin
        [token] = auth.token_requests
        assert token["grant_type"] == "authorization_code"
        assert token["resource"] == protected_server.origin
        assert token["client_secret"] == CLIENT_SECRET and token["code_verifier"]

    def test_configured_scopes_win(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run, scopes=["email", "profile"])

        assert protected_server.auth.authorize_requests[0]["scope"] == "email profile"

    def test_without_a_challenge_the_well_known_addresses_are_tried(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        protected_server.auth.challenge = False

        sign_in(run)

        assert protected_server.auth.metadata_requests[:2] == [
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-protected-resource",
        ]
        assert protected_server.auth.authorize_requests[0]["scope"] == "openid email"

    def test_a_configured_authorization_server_needs_no_resource_metadata(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        protected_server.auth.challenge = False
        protected_server.auth.publish_resource_metadata = False

        missing = run(["auth", "login"])
        assert missing.exit_code == 4
        assert "set auth.authorization_server" in missing.stderr

        sign_in(run, authorization_server=protected_server.origin)
        assert protected_server.auth.authorize_requests[0]["resource"] == protected_server.url

    def test_the_resource_parameter_can_be_left_out(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run, resource=False)

        assert "resource" not in protected_server.auth.authorize_requests[0]
        assert "resource" not in protected_server.auth.token_requests[0]

    def test_no_browser_only_prints_the_address(self, run: Run, opened: list[str]) -> None:
        result = run(["auth", "login", "--no-browser", "--timeout", "1"])

        assert result.exit_code == 4
        assert opened == []
        assert "http://127.0.0.1" in result.stderr and "/authorize?" in result.stderr
        assert "timed out after 1s" in result.stderr

    def test_a_denied_consent_fails(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.deny = "access_denied"

        result = run(["auth", "login"])

        assert result.exit_code == 4
        assert "access_denied" in result.stderr
        assert not token_file().exists()

    def test_an_answer_with_another_state_is_refused(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.tamper_state = True

        result = run(["auth", "login"])

        assert result.exit_code == 4
        assert "state differs" in result.stderr
        assert protected_server.auth.token_requests == []

    def test_an_authorization_server_without_s256_is_refused(
        self, run: Run, protected_server: ProtectedServer, opened: list[str]
    ) -> None:
        protected_server.auth.code_challenge_methods = ["plain"]

        result = run(["auth", "login"])

        assert result.exit_code == 4
        assert "S256" in result.stderr
        assert opened == []

    def test_metadata_of_another_issuer_is_refused(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.metadata_issuer = "https://issuer.example.com"

        result = run(["auth", "login"])

        assert result.exit_code == 4
        assert "issuer" in result.stderr

    def test_a_server_without_oauth_cannot_sign_in(self, invoke: Invoke, json_server: Any) -> None:
        result = invoke(["auth", "login"])

        assert result.exit_code == 2
        assert "No configured server signs in with OAuth" in result.stderr


class TestBearer:
    def test_every_request_carries_the_access_token(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)

        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 0, result.stderr
        assert protected_server.recorder.calls()[0]["name"] == "search_items"
        bearers = [bearer for bearer in protected_server.auth.bearers if bearer is not None]
        assert bearers and set(bearers) == {stored()["access_token"]}

    def test_the_id_token_can_be_the_bearer(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.accept = "id_token"
        sign_in(run, bearer="id_token")

        result = run(["search-items", "--query", "x"], bearer="id_token")

        assert result.exit_code == 0, result.stderr
        assert protected_server.auth.bearers[-1] == stored()["id_token"]

    def test_not_signed_in_fails_at_once(self, run: Run, protected_server: ProtectedServer, opened: list[str]) -> None:
        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 4
        assert f"Run '{BIN} auth login' in a terminal" in result.stderr
        assert opened == [] and protected_server.auth.authorize_requests == []
        assert protected_server.recorder.requests == []

    def test_an_expired_token_is_refreshed_once(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        expire()

        first = run(["search-items", "--query", "x"])
        second = run(["search-items", "--query", "y"])

        assert first.exit_code == 0 and second.exit_code == 0, first.stderr + second.stderr
        [refresh] = protected_server.auth.refresh_grants()
        assert refresh["resource"] == protected_server.origin
        assert protected_server.auth.bearers[-1] == stored()["access_token"]
        assert stored()["expires_at"] > time.time() + 3000

    def test_a_rotated_refresh_token_is_stored(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.rotate_refresh_tokens = True
        sign_in(run)
        first = stored()["refresh_token"]
        expire()

        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 0, result.stderr
        assert stored()["refresh_token"] != first
        assert stored()["refresh_token"] in protected_server.auth.refresh_tokens

    def test_a_refused_token_is_refreshed_and_the_call_sent_once(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        sign_in(run)
        assert run(["search-items", "--query", "warm the cache"]).exit_code == 0
        protected_server.recorder.reset()
        protected_server.auth.reject_next = 1

        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 0, result.stderr
        assert len(protected_server.auth.refresh_grants()) == 1
        assert [call["arguments"] for call in protected_server.recorder.calls()] == [{"query": "x"}]

    def test_a_token_refused_after_a_refresh_fails(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        protected_server.auth.reject_next = 10

        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 4
        assert "refused the token of example even after a refresh" in result.stderr
        assert len(protected_server.auth.refresh_grants()) == 1

    def test_a_revoked_sign_in_deletes_the_tokens(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        expire()
        protected_server.auth.refresh_error = "invalid_grant"

        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 4
        assert "expired or was revoked" in result.stderr
        assert not token_file().exists()

    def test_an_unreachable_token_endpoint_keeps_the_tokens(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        expire()
        protected_server.auth.token_status = [503] * 5

        result = run(["search-items", "--query", "x"])

        assert result.exit_code == 3
        assert "token endpoint" in result.stderr
        assert token_file().exists()

    def test_tokens_of_another_client_are_not_sent(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)

        result = run(["search-items", "--query", "x"], client_id="another-client")

        assert result.exit_code == 4
        assert "not signed in to example" in result.stderr
        assert protected_server.auth.bearers == [None]  # only the login's discovery request

    def test_parallel_invocations_refresh_once(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        expire()
        server = ServerConfig.model_validate(server_config(protected_server))
        resolved = expand_server("example", server)
        auth = resolved.auth
        assert auth is not None
        store = TokenStore(default_token_dir(BIN))
        bearers: list[str] = []

        def refresh() -> None:
            manager = TokenManager("example", resolved.url, auth, store, "login")
            bearers.append(manager.bearer())

        threads = [threading.Thread(target=refresh) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert len(protected_server.auth.refresh_grants()) == 1
        assert len(set(bearers)) == 1 and len(bearers) == 4

    def test_secrets_stay_out_of_the_verbose_log(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        expire()

        result = run(["-vv", "search-items", "--query", "x"])

        assert result.exit_code == 0, result.stderr
        assert "/token" in result.stderr
        assert_no_secrets(result, protected_server)


DEFAULT_STATUS_FIELDS = ["server", "signed_in", "identity", "signed_in_at", "signed_in_until"]

FULL_STATUS_FIELDS = [
    *DEFAULT_STATUS_FIELDS,
    "bearer",
    "access_token_expires_at",
    "id_token_expires_at",
    "has_refresh_token",
    "refresh_token_expires_at",
    "obtained_at",
    "scopes",
    "client_id",
    "issuer",
    "resource",
    "token_file",
]


def status(run: Run, *args: str) -> tuple[int, dict[str, Any]]:
    result = run(["auth", "status", "--json", *args])
    [entry] = json.loads(result.stdout)["servers"]
    return result.exit_code, entry


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def edit_stored(change: Callable[[dict[str, Any]], object]) -> None:
    content = stored()
    change(content)
    token_file().write_text(json.dumps(content), encoding="utf-8")


class TestRefreshExpiry:
    @pytest.mark.parametrize("name", ["refresh_token_expires_in", "refresh_expires_in"])
    def test_the_refresh_token_lifetime_is_stored(self, run: Run, protected_server: ProtectedServer, name: str) -> None:
        protected_server.auth.refresh_lifetime = {name: 604800}
        before = time.time()

        sign_in(run)

        assert before + 604800 <= stored()["refresh_expires_at"] <= time.time() + 604800
        exit_code, entry = status(run)
        assert exit_code == 0
        assert entry["signed_in_until"] == datetime.fromtimestamp(stored()["refresh_expires_at"], UTC).isoformat(
            timespec="seconds"
        )

    def test_a_zero_lifetime_counts_as_unknown(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.refresh_lifetime = {"refresh_expires_in": 0}

        sign_in(run)

        assert stored()["refresh_expires_at"] is None
        assert status(run)[1]["signed_in_until"] is None

    def test_a_refresh_keeps_the_lifetime_of_the_kept_refresh_token(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        protected_server.auth.refresh_lifetime = {"refresh_token_expires_in": 604800}
        sign_in(run)
        signed_in = stored()
        protected_server.auth.refresh_lifetime = {}
        expire()

        assert run(["search-items", "--query", "x"]).exit_code == 0

        refreshed = stored()
        assert protected_server.auth.refresh_grants()
        assert refreshed["refresh_token"] == signed_in["refresh_token"]
        assert refreshed["refresh_expires_at"] == signed_in["refresh_expires_at"]
        assert refreshed["signed_in_at"] == signed_in["signed_in_at"]
        assert refreshed["obtained_at"] > signed_in["obtained_at"]

    def test_a_refresh_that_states_the_lifetime_replaces_it(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.refresh_lifetime = {"refresh_token_expires_in": 604800}
        sign_in(run)
        protected_server.auth.refresh_lifetime = {"refresh_token_expires_in": 1000}
        expire()

        assert run(["search-items", "--query", "x"]).exit_code == 0

        assert stored()["refresh_expires_at"] <= time.time() + 1000

    def test_a_rotated_refresh_token_without_a_lifetime_has_none(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        protected_server.auth.refresh_lifetime = {"refresh_token_expires_in": 604800}
        protected_server.auth.rotate_refresh_tokens = True
        sign_in(run)
        signed_in = stored()
        protected_server.auth.refresh_lifetime = {}
        expire()

        assert run(["search-items", "--query", "x"]).exit_code == 0

        refreshed = stored()
        assert refreshed["refresh_token"] != signed_in["refresh_token"]
        assert refreshed["refresh_expires_at"] is None
        assert refreshed["signed_in_at"] == signed_in["signed_in_at"]

    def test_an_expired_refresh_token_is_not_signed_in(self, run: Run, protected_server: ProtectedServer) -> None:
        protected_server.auth.refresh_lifetime = {"refresh_token_expires_in": 604800}
        sign_in(run)
        expire()
        edit_stored(lambda content: content.update(refresh_expires_at=time.time() - 10))

        exit_code, entry = status(run)

        assert exit_code == 4
        assert entry["signed_in"] is False and entry["signed_in_until"] is not None


class TestTokenFile:
    def test_a_file_from_before_the_refresh_expiry_still_signs_in(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        sign_in(run)
        edit_stored(lambda content: [content.pop(key) for key in ("refresh_expires_at", "signed_in_at")])
        expire()

        exit_code, entry = status(run)
        assert exit_code == 0
        assert entry["signed_in"] is True and entry["signed_in_at"] is None and entry["signed_in_until"] is None

        assert run(["search-items", "--query", "x"]).exit_code == 0, "the old file refreshes"
        assert stored()["version"] == 1 and stored()["signed_in_at"] is None

    def test_fields_of_a_later_version_are_ignored(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        edit_stored(lambda content: content.update(added_later="x"))

        assert status(run)[1]["signed_in"] is True
        assert run(["search-items", "--query", "x"]).exit_code == 0


class TestStatusAndLogout:
    def test_status_before_and_after_signing_in(self, run: Run, protected_server: ProtectedServer) -> None:
        before = run(["auth", "status", "--json"])
        assert before.exit_code == 4
        assert json.loads(before.stdout)["servers"] == [
            {"server": "example", "signed_in": False, "identity": None, "signed_in_at": None, "signed_in_until": None}
        ]

        started = time.time()
        sign_in(run)
        after = run(["auth", "status", "--json"])

        assert after.exit_code == 0, after.stderr
        [entry] = json.loads(after.stdout)["servers"]
        assert list(entry) == DEFAULT_STATUS_FIELDS
        assert entry["server"] == "example" and entry["signed_in"] is True and entry["identity"] == EMAIL
        assert started - 1 <= epoch(entry["signed_in_at"]) <= time.time()
        # A refresh token of unknown lifetime: the sign-in lasts until the authorization server ends it.
        assert entry["signed_in_until"] is None
        assert_no_secrets(after, protected_server)

    def test_status_prints_one_line_and_full_status_indented(self, run: Run) -> None:
        sign_in(run)
        # The same for a person and for an AI agent, unless a flag says otherwise.
        for env in (None, {"CLAUDECODE": "1"}):
            short = run(["auth", "status"], env=env)
            full = run(["auth", "status", "--full"], env=env)
            assert short.exit_code == 0 and full.exit_code == 0
            assert short.stdout.count("\n") == 1
            assert full.stdout.startswith('{\n  "servers"') and json.loads(full.stdout) == json.loads(
                run(["auth", "status", "--full", "--json"]).stdout
            )

        assert run(["auth", "status", "--pretty"]).stdout.startswith('{\n  "servers"')
        assert run(["auth", "status", "--full", "--json"]).stdout.count("\n") == 1

    def test_full_status(self, run: Run, protected_server: ProtectedServer) -> None:
        before = run(["auth", "status", "--full", "--json"])
        assert before.exit_code == 4
        [entry] = json.loads(before.stdout)["servers"]
        assert list(entry) == FULL_STATUS_FIELDS
        assert entry["has_refresh_token"] is False and entry["client_id"] == CLIENT_ID

        protected_server.auth.refresh_lifetime = {"refresh_token_expires_in": 604800}
        sign_in(run)
        after = run(["auth", "status", "--full", "--json"])

        assert after.exit_code == 0, after.stderr
        [entry] = json.loads(after.stdout)["servers"]
        assert list(entry) == FULL_STATUS_FIELDS
        assert entry["bearer"] == "access_token" and entry["has_refresh_token"] is True
        assert entry["signed_in_until"] == entry["refresh_token_expires_at"]
        assert 3000 < epoch(entry["access_token_expires_at"]) - time.time() <= 3600
        assert 3000 < epoch(entry["id_token_expires_at"]) - time.time() <= 3600
        assert entry["obtained_at"] == entry["signed_in_at"]
        assert entry["scopes"] == ["openid", "email"]
        assert entry["client_id"] == CLIENT_ID and entry["issuer"] == protected_server.origin
        assert entry["resource"] == protected_server.origin
        assert entry["token_file"] == str(token_file())
        assert_no_secrets(after, protected_server)

    def test_without_a_refresh_token_the_sign_in_lasts_as_long_as_the_bearer(
        self, run: Run, protected_server: ProtectedServer
    ) -> None:
        protected_server.auth.accept = "id_token"
        sign_in(run, bearer="id_token")
        edit_stored(lambda content: content.update(refresh_token=None, refresh_expires_at=None))

        result = run(["auth", "status", "--full", "--json"], bearer="id_token")

        [entry] = json.loads(result.stdout)["servers"]
        assert entry["signed_in"] is True and entry["has_refresh_token"] is False
        assert entry["signed_in_until"] == entry["id_token_expires_at"] is not None

        expire()
        assert run(["auth", "status"], bearer="id_token").exit_code == 4

    def test_logout_deletes_the_tokens(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)

        result = run(["auth", "logout"])

        assert result.exit_code == 0
        assert "Signed out of example." in result.stderr
        assert not token_file().exists()
        assert protected_server.auth.revoke_requests == []
        assert "Not signed in to example." in run(["auth", "logout"]).stderr

    def test_logout_can_revoke(self, run: Run, protected_server: ProtectedServer) -> None:
        sign_in(run)
        refresh_token = stored()["refresh_token"]

        result = run(["auth", "logout", "--revoke"])

        assert result.exit_code == 0, result.stderr
        [revoke] = protected_server.auth.revoke_requests
        assert revoke["token"] == refresh_token and revoke["client_id"] == CLIENT_ID
        assert not token_file().exists()

    def test_an_unknown_server_is_a_usage_error(self, run: Run) -> None:
        result = run(["auth", "status", "other"])

        assert result.exit_code == 2
        assert "No server named 'other'" in result.stderr


class TestConfig:
    def test_the_client_secret_is_masked(self, run: Run) -> None:
        result = run(["config", "--json"])

        assert result.exit_code == 0
        assert CLIENT_SECRET not in result.stdout
        auth = json.loads(result.stdout)["config"]["servers"]["example"]["auth"]
        assert auth["client_id"] == CLIENT_ID and auth["client_secret"].startswith("***")

    def test_the_secret_from_an_environment_variable(self, run: Run, protected_server: ProtectedServer) -> None:
        result = run(
            ["auth", "login"],
            env={"EXAMPLE_TOOL_SERVERS__EXAMPLE__AUTH__CLIENT_SECRET": CLIENT_SECRET},
            client_secret="x",
        )

        assert result.exit_code == 0, result.stderr

    def test_a_template_secret(self, run: Run) -> None:
        result = run(["auth", "login"], env={"OAUTH_SECRET": CLIENT_SECRET}, client_secret="${OAUTH_SECRET}")

        assert result.exit_code == 0, result.stderr

    def test_auth_and_an_authorization_header_exclude_each_other(
        self, invoke: Invoke, protected_server: ProtectedServer
    ) -> None:
        server = {**server_config(protected_server), "headers": {"authorization": "Bearer x"}}

        result = invoke(["config"], default={"servers": {"example": server}})

        assert result.exit_code == 2
        assert "auth and an Authorization header exclude each other" in result.stderr

    def test_an_unknown_auth_key_in_the_environment(self, run: Run) -> None:
        result = run(["config"], env={"EXAMPLE_TOOL_SERVERS__EXAMPLE__AUTH__CLIENT": "x"})

        assert result.exit_code == 2
        assert "Unknown auth key CLIENT" in result.stderr

    def test_the_help_lists_auth_only_for_oauth_servers(self, run: Run, invoke: Invoke, json_server: Any) -> None:
        sign_in(run)

        assert "auth " in run(["--help"]).stdout
        assert "auth " not in invoke(["--help"]).stdout


def test_the_token_directory_follows_xdg_state_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    assert default_token_dir(BIN) == tmp_path / BIN / "oauth"
    monkeypatch.delenv("XDG_STATE_HOME")
    assert default_token_dir(BIN) == Path(os.environ["HOME"]) / ".local" / "state" / BIN / "oauth"
