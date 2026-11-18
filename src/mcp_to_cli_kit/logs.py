"""Logging to stderr, set up without importing the MCP SDK, so a host CLI can run before the kit is loaded."""

from __future__ import annotations

import logging
import sys

from mcp_to_cli_kit.redaction import RedactingFilter

_SILENT = logging.CRITICAL + 10

_LIBRARY_LOGGERS = ("mcp", "httpx", "httpcore")

_KIT_HANDLER = "_mcp_to_cli_kit_handler"


def configure_logging(verbosity: int) -> None:
    """``-v`` shows the kit's debug messages, ``-vv`` also the MCP SDK's and httpx's. All on stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    # Expanded ${NAME} values and the URL's userinfo and query never reach the log, -vv included.
    handler.addFilter(RedactingFilter())

    kit = logging.getLogger("mcp_to_cli_kit")
    kit.handlers = [handler]
    kit.propagate = False
    kit.setLevel(logging.DEBUG if verbosity >= 1 else logging.WARNING)

    for name in _LIBRARY_LOGGERS:
        library = logging.getLogger(name)
        library.handlers = [handler] if verbosity >= 1 else [logging.NullHandler()]
        library.propagate = False
        library.setLevel(logging.DEBUG if verbosity >= 2 else logging.WARNING if verbosity == 1 else _SILENT)

    # The SDK also logs through the root logger ("Unhandled exception in receive loop"). Without
    # a handler there, Python's last-resort handler would print it with a traceback to stderr.
    root = logging.getLogger()
    for existing in [h for h in root.handlers if getattr(h, _KIT_HANDLER, False)]:
        root.removeHandler(existing)
    root_handler: logging.Handler = logging.StreamHandler(sys.stderr) if verbosity >= 1 else logging.NullHandler()
    root_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_handler.addFilter(RedactingFilter())
    root_handler.setLevel(logging.WARNING)
    setattr(root_handler, _KIT_HANDLER, True)
    root.addHandler(root_handler)
