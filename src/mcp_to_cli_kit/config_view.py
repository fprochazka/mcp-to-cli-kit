"""The effective configuration as the ``config`` command prints it, with its secrets masked.

What is masked, the same parts the log redaction treats as secrets:

- the literal text of every header value, except for the harmless headers in
  ``SHOWN_HEADERS``; a leading known auth scheme word (``Bearer``, ``Basic``, ``Token``,
  ``Digest``, ``Bot``, ``ApiKey``) followed by more text stays readable;
- a URL's whole userinfo (user and password), the values of its query and its fragment. The
  scheme, host, port and path are shown.

A masked value becomes ``***`` followed by its last 4 characters, or only ``***`` when it is
shorter than ``REVEAL_MIN_LENGTH``, so a short secret is not half shown. ``${NAME}`` references
are shown as written and never expanded, so no value from the environment is ever printed.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mcp_to_cli_kit.config import LoadedConfig, VariableReference, parse_template

MASK = "***"

REVEAL_MIN_LENGTH = 12

# Headers whose values are shown as written; every other header value is masked.
SHOWN_HEADERS = frozenset({"accept", "content-type", "user-agent", "x-mcp-client"})

# A known auth scheme word at the start of a header value, followed by whitespace and more text.
_AUTH_SCHEME = re.compile(r"^((?:Bearer|Basic|Token|Digest|Bot|ApiKey)[ \t]+)(.*)$", re.IGNORECASE | re.DOTALL)


def mask(value: str) -> str:
    """``***`` plus the last 4 characters, or only ``***`` for a value shorter than ``REVEAL_MIN_LENGTH``."""
    return MASK + value[-4:] if len(value) >= REVEAL_MIN_LENGTH else MASK


def describe(loaded: LoadedConfig) -> dict[str, Any]:
    """Where the configuration came from and the merged configuration itself, with secrets masked."""
    config = loaded.config.model_dump(mode="json")
    for server in config["servers"].values():
        server["url"] = _mask_url(server["url"])
        server["headers"] = {
            name: value if name.lower() in SHOWN_HEADERS else _mask_template(value, keep_scheme=True)
            for name, value in server["headers"].items()
        }
    return {
        "config_file": str(loaded.path),
        "config_file_loaded": loaded.file_loaded,
        "env_prefix": loaded.env_prefix,
        "env_applied": list(loaded.env_applied),
        "config": config,
    }


def _mask_template(value: str, *, keep_scheme: bool = False) -> str:
    """Mask the literal text of a template, keeping its ``${NAME}`` references as written."""
    parts = parse_template(value)
    shown = []
    first = parts[0] if parts else None
    if keep_scheme and isinstance(first, str) and (scheme := _AUTH_SCHEME.match(first)):
        rest = [scheme.group(2), *parts[1:]]
        if any(rest):  # the word alone stays masked: it may be the whole secret
            shown.append(scheme.group(1))
            parts = rest
    for part in parts:
        if isinstance(part, VariableReference):
            shown.append("${" + part.name + "}")
        elif part:
            shown.append(mask(part))
    return "".join(shown)


def _mask_url(url: str) -> str:
    """Mask the userinfo, the query values and the fragment of a URL or URL template; the rest stays as written."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return _mask_template(url)
    if not parts.scheme or not parts.netloc:
        # A URL that is a reference as a whole, or starts with one: only its literal text can be judged.
        return url if not _has_literal_secret_parts(url) else _mask_template(url)
    userinfo, at, host = parts.netloc.rpartition("@")
    netloc = f"{_mask_template(userinfo)}{at}{host}"
    query = "&".join(_mask_query_item(item) for item in parts.query.split("&")) if parts.query else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, _mask_template(parts.fragment)))


def _mask_query_item(item: str) -> str:
    name, equals, value = item.partition("=")
    if not equals:
        return _mask_template(item) if item else item
    return f"{name}={_mask_template(value)}" if value else item


def _has_literal_secret_parts(url: str) -> bool:
    """Whether the literal text of a URL that starts with a reference holds a ``?`` or an ``@``."""
    return any(isinstance(part, str) and ("?" in part or "@" in part) for part in parse_template(url))
