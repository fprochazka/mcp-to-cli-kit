"""The per-server cache of ``tools/list`` results, ``$XDG_CACHE_HOME/<bin>/<server>.json``.

Without ``XDG_CACHE_HOME`` the directory is ``~/.cache/<bin>``.

A cache entry is fresh for an hour. An expired entry is still kept: when the server cannot
be reached, the CLI falls back to it rather than failing to know its own commands. The entry
records a hash of the server URL, never the URL itself, which may carry a token; an entry
written for another URL is ignored.

A configured URL may be a template with ``${NAME}`` references. The entry then records a hash
of the template and one of the expanded URL. When the variables are set, both must match, so
pointing a variable at another server does not reuse the old server's tools; when one is not
set, the template alone decides, so ``--help`` still lists the tools offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_to_cli_kit.config import xdg_directory

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CachedTools:
    """A cache entry.

    Attributes:
        tools: The tools as ``tools/list`` returned them, one JSON object each.
        fetched_at: When they were fetched, in seconds since the epoch.
        fresh: Whether the entry is younger than the TTL.
    """

    tools: list[dict[str, Any]]
    fetched_at: float
    fresh: bool


def default_cache_dir(bin_name: str) -> Path:
    """Return ``$XDG_CACHE_HOME/<bin>``, by default ``~/.cache/<bin>``."""
    return xdg_directory("XDG_CACHE_HOME", ".cache") / bin_name


class SchemaCache:
    """Reads and writes the tool list of each server of one bin."""

    def __init__(self, directory: Path, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.directory = directory
        self.ttl_seconds = ttl_seconds

    def path(self, server: str) -> Path:
        """The cache file of one server."""
        return self.directory / f"{server}.json"

    def load(self, server: str, url: str, resolved_url: str | None = None) -> CachedTools | None:
        """Return the cached tools of a server, fresh or expired, or None when there is no usable entry.

        Args:
            server: The server's name.
            url: The URL as configured, a template or not.
            resolved_url: The URL with its variables expanded, or None when they cannot be.
        """
        path = self.path(server)
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            logger.debug("Ignoring unreadable cache file %s: %s", path, error)
            return None

        if (
            not isinstance(content, dict)
            or content.get("version") != _FORMAT_VERSION
            or content.get("url_sha256") != _url_hash(url)
            or (resolved_url not in (None, url) and content.get("resolved_url_sha256") != _url_hash(resolved_url))
            or not isinstance(content.get("tools"), list)
            or not isinstance(content.get("fetched_at"), int | float)
        ):
            logger.debug("Ignoring cache file %s: written for another URL or format", path)
            return None

        fetched_at = float(content["fetched_at"])
        age = time.time() - fetched_at
        tools = [tool for tool in content["tools"] if isinstance(tool, dict)]
        return CachedTools(tools=tools, fetched_at=fetched_at, fresh=0 <= age < self.ttl_seconds)

    def save(self, server: str, url: str, tools: list[dict[str, Any]], resolved_url: str | None = None) -> None:
        """Write a server's tools atomically. A failure to write is logged, never raised."""
        content: dict[str, Any] = {"version": _FORMAT_VERSION, "url_sha256": _url_hash(url)}
        if resolved_url is not None and resolved_url != url:
            content["resolved_url_sha256"] = _url_hash(resolved_url)
        content.update(fetched_at=time.time(), tools=tools)
        path = self.path(server)
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=f".{server}.", suffix=".tmp")
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(content, handle, ensure_ascii=False)
                os.replace(temporary, path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError as error:
            logger.warning("Cannot write the tool cache %s: %s", path, error)

    def drop(self, server: str) -> None:
        """Delete a server's entry, so the next run fetches the tools again."""
        try:
            self.path(server).unlink(missing_ok=True)
        except OSError as error:
            logger.debug("Cannot delete the tool cache %s: %s", self.path(server), error)


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()
