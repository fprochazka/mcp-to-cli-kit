"""What the MCP client announces about itself, and its retry defaults, importable without the MCP SDK.

The CLI needs these to build its group and print ``--version``; keeping them out of ``client``
lets it do so without loading the SDK.
"""

from __future__ import annotations

from importlib import metadata

KIT_DISTRIBUTION = "mcp-to-cli-kit"

FALLBACK_VERSION = "0.1.0"

CLIENT_HEADER = "X-Mcp-Client"

# Pauses before the first and second retry. Two retries at most.
DEFAULT_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.5)


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
