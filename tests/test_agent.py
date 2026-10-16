"""Tests for AI agent detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_to_cli_kit import agent as agent_module
from mcp_to_cli_kit.agent import AGENT_ENV_VARS, GENERIC_AGENT_ENV_VARS, detect_agent

ALL_AGENT_VARS = [name for _, variables in AGENT_ENV_VARS for name in variables] + GENERIC_AGENT_ENV_VARS


@pytest.fixture(autouse=True)
def no_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in a shell that announces no agent."""
    for variable in ALL_AGENT_VARS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(agent_module, "DEVIN_MARKER", Path("/nonexistent-devin-marker"))


class TestDetectAgent:
    """Tests for detect_agent()."""

    def test_nothing_set(self) -> None:
        assert detect_agent() is None

    def test_claude_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")

        assert detect_agent() == "Claude Code"

    def test_gemini_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_CLI", "1")

        assert detect_agent() == "Gemini CLI"

    def test_an_empty_value_still_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Presence of the variable is the signal, whatever it holds."""
        monkeypatch.setenv("CLINE_ACTIVE", "")

        assert detect_agent() == "Cline"

    def test_generic_agent_slug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT", "goose")

        assert detect_agent() == "Goose"

    def test_generic_slug_with_a_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Some shells set this versioned form."""
        monkeypatch.setenv("AI_AGENT", "claude-code_2-1-260_agent")

        assert detect_agent() == "Claude Code"

    def test_unknown_slug_is_title_cased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_AGENT", "foo-bar")

        assert detect_agent() == "Foo Bar"

    def test_empty_generic_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_AGENT", "")

        assert detect_agent() == "an AI agent"

    def test_harness_variable_beats_the_generic_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A user may set AI_AGENT by hand, so a harness's own variable wins."""
        monkeypatch.setenv("AI_AGENT", "foo-bar")
        monkeypatch.setenv("CURSOR_AGENT", "1")

        assert detect_agent() == "Cursor"

    def test_agent_beats_ai_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT", "amp")
        monkeypatch.setenv("AI_AGENT", "foo-bar")

        assert detect_agent() == "Amp"

    def test_devin_is_detected_by_a_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Devin sets no variable at all, only leaves a file behind."""
        monkeypatch.setattr(agent_module, "DEVIN_MARKER", tmp_path)

        assert detect_agent() == "Devin"

    def test_a_variable_a_human_also_has_is_not_a_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cursor's IDE terminal sets CURSOR_TRACE_ID for people too."""
        monkeypatch.setenv("CURSOR_TRACE_ID", "abc")
        monkeypatch.setenv("TERM_PROGRAM", "kiro")

        assert detect_agent() is None
