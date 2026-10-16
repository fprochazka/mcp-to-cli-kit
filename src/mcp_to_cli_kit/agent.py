"""Detection of the AI coding agent this process is running under.

The CLI prints a tool's result differently for an agent than for a person: an agent gets
the server's text verbatim, a person gets it pretty-printed. There is no standard for a
harness to announce itself, so this recognises the environment variables the known
harnesses set, plus the two generic ones (``AGENT`` and ``AI_AGENT``) that carry a slug.

The table below is the single place to extend. It was researched on 2026-09-05 from
official documentation, project sources and ``vercel/detect-agent``'s ``agents.json``.
Only Claude Code is verified first hand; the rest are best effort, and a harness that
announces nothing is deliberately left out rather than guessed at from something like
``TERM_PROGRAM``, which a human typing in the same terminal would also set.

Known to set nothing detectable: GitHub Copilot (the feature request is unshipped),
Aider, Windsurf/Cascade, and Warp's agent mode.
"""

from __future__ import annotations

import os
from pathlib import Path

# Harness-specific variables, checked in this order. The presence of the variable is
# the signal; its value is not read. These win over AGENT and AI_AGENT, which a user
# may well have set by hand.
AGENT_ENV_VARS: list[tuple[str, list[str]]] = [
    ("Claude Code", ["CLAUDECODE", "CLAUDE_CODE"]),
    ("Gemini CLI", ["GEMINI_CLI"]),
    ("Antigravity", ["ANTIGRAVITY_AGENT", "ANTIGRAVITY_CLI_ALIAS"]),
    ("Codex", ["CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED", "CODEX_CI", "CODEX_THREAD_ID"]),
    # Deliberately not CURSOR_TRACE_ID: the IDE sets it for a human's terminal too
    ("Cursor", ["CURSOR_AGENT", "CURSOR_CLI", "CURSOR_EXTENSION_HOST_ROLE"]),
    ("Cline", ["CLINE_ACTIVE"]),
    ("OpenCode", ["OPENCODE", "OPENCODE_CLIENT"]),
    ("Goose", ["GOOSE_TERMINAL"]),
    ("Augment", ["AUGMENT_AGENT"]),
    ("Junie", ["JUNIE_DATA", "JUNIE_SHIM_PATH"]),
    ("Kimi Code", ["KIMI_PLUGIN_ROOT"]),
    ("Grok Build", ["GROK_PLUGIN_ROOT", "GROK_PLUGIN_DATA"]),
    ("OpenClaw", ["OPENCLAW_SHELL"]),
    ("Trae", ["TRAE_AI_SHELL_ID"]),
    ("Pi", ["PI_CODING_AGENT"]),
]

# Generic variables carrying a slug, checked after the table above.
GENERIC_AGENT_ENV_VARS = ["AGENT", "AI_AGENT"]

# Slugs whose title-cased form would be wrong or ugly.
AGENT_SLUG_NAMES = {
    "claude-code": "Claude Code",
    "cursor-cli": "Cursor",
    "gemini-cli": "Gemini CLI",
    "amp": "Amp",
    "goose": "Goose",
    "codex": "Codex",
}

# Devin announces itself with a file rather than a variable.
DEVIN_MARKER = Path("/opt/.devin")

# Name used when something says an agent is running but not which one.
UNKNOWN_AGENT = "an AI agent"


def _name_from_slug(value: str) -> str:
    """Turn the value of a generic agent variable into a display name.

    The value is a slug that may carry a version, as in
    ``claude-code_2-1-260_agent``, so only the part before the first underscore names
    the agent.

    Args:
        value: The raw value of AGENT or AI_AGENT.

    Returns:
        The display name, or the fallback when the value names nothing.
    """
    slug = value.strip().split("_")[0].strip().lower()
    if not slug:
        return UNKNOWN_AGENT

    if slug in AGENT_SLUG_NAMES:
        return AGENT_SLUG_NAMES[slug]

    return " ".join(word.capitalize() for word in slug.split("-") if word) or UNKNOWN_AGENT


def detect_agent() -> str | None:
    """Detect the AI coding agent this process is running under.

    Returns:
        The agent's display name, or None when no agent is detected.
    """
    for name, variables in AGENT_ENV_VARS:
        if any(variable in os.environ for variable in variables):
            return name

    for variable in GENERIC_AGENT_ENV_VARS:
        value = os.environ.get(variable)
        if value is not None:
            return _name_from_slug(value)

    if DEVIN_MARKER.exists():
        return "Devin"

    return None
