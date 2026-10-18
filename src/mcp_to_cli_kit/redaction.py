"""Keep expanded secrets out of error messages and logs.

When a server's ``${NAME}`` references are expanded, the parts that carry secrets are
registered here with the text that replaces them: the values of the variables in its headers,
userinfo, query and fragment (replaced by the ``${NAME}`` they came from), and the userinfo,
password, query and fragment of the expanded URL (replaced by ``***``). Each is also registered in the percent-encoded
forms a URL log line shows. Host, port and path values are not secrets and are not registered.
Error messages of the transport and every log record written through the kit's handlers,
``-vv`` included, pass through ``redact``.

Values shorter than ``MIN_LENGTH`` are not registered: replacing every ``1`` or ``a`` in a
message would make it unreadable, and such a value is no secret.
"""

from __future__ import annotations

import logging
import threading
from urllib.parse import quote, quote_plus, unquote

MIN_LENGTH = 4

_lock = threading.Lock()
_replacements: dict[str, str] = {}

# The characters httpx leaves unencoded in a URL's userinfo, path and query.
_URL_SAFE = "!$&'()*+,;=:/?@[]|~"


def register(value: str, replacement: str) -> None:
    """Replace ``value`` with ``replacement`` wherever ``redact`` sees it from now on."""
    if len(value) < MIN_LENGTH or value == replacement:
        return
    with _lock:
        _replacements.setdefault(value, replacement)


def register_encoded(value: str, replacement: str) -> None:
    """Register a value as written, decoded, and in the percent-encoded forms URLs and logs use."""
    for form in (
        value,
        unquote(value),
        quote(value, safe=""),
        quote(value, safe=_URL_SAFE),
        quote(unquote(value), safe=_URL_SAFE),
        quote_plus(value),
    ):
        register(form, replacement)


def clear() -> None:
    """Forget every registered value, at the end of an invocation."""
    with _lock:
        _replacements.clear()


def redact(text: str) -> str:
    """The text with every registered value replaced, the longest values first."""
    with _lock:
        values = sorted(_replacements, key=len, reverse=True)
        replacements = dict(_replacements)
    for value in values:
        if value in text:
            text = text.replace(value, replacements[value])
    return text


class RedactingFilter(logging.Filter):
    """A handler filter that redacts the formatted message and the exception text of each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True
