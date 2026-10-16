"""Print a tool result, for an AI agent or a script (machine mode) or for a person (pretty mode).

Machine mode prints what the server sent without re-serializing it: a single text item is
written byte for byte, and no newline is added unless stdout is a terminal. Pretty mode
re-indents JSON with ``indent=2`` and keeps non-ASCII characters readable; text that is not
JSON is printed as it is. Nothing here wraps lines, in either mode.

Which content is printed:

- One item: that item. A text item is its text; any other item is its JSON form.
- Several items: one JSON array. An item whose text parses as JSON is embedded as that value
  (in machine mode, as the server's own text), any other text item as a string.
- ``structuredContent`` is printed instead of the content, unless the content is just its
  serialization: a single text item that parses to the same value, or the ``{"result": ...}``
  wrapper some servers put around a plain text result.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mcp import types

from mcp_to_cli_kit.agent import detect_agent

OutputMode = Literal["machine", "pretty"]


def resolve_mode(*flags: OutputMode | None) -> OutputMode:
    """The first explicit flag wins; without one, an AI agent gets machine mode and a person pretty mode."""
    for flag in flags:
        if flag is not None:
            return flag
    return "machine" if detect_agent() is not None else "pretty"


def render(result: types.CallToolResult, mode: OutputMode) -> str:
    """The text to print for a result, without the trailing newline decision."""
    content = list(result.content)
    structured = result.structuredContent
    if structured is not None and not _content_is_serialized(content, structured):
        return _dump(structured, mode)

    if not content:
        return ""

    if len(content) == 1:
        item = content[0]
        if isinstance(item, types.TextContent):
            if mode == "machine":
                return item.text
            parsed = _try_json(item.text)
            return item.text if parsed is _NOT_JSON else _dump(parsed, mode)
        return _dump(_item_json(item), mode)

    if mode == "machine":
        return "[" + ",".join(_machine_array_item(item) for item in content) + "]"
    return _dump([_pretty_array_item(item) for item in content], mode)


def finish(text: str, mode: OutputMode, is_tty: bool) -> str:
    """Add the final newline, except in machine mode on a pipe, where the bytes stay the server's."""
    if text == "" or text.endswith("\n"):
        return text
    if mode == "machine" and not is_tty:
        return text
    return text + "\n"


class _NotJson:
    pass


_NOT_JSON: Any = _NotJson()


def _try_json(text: str) -> Any:
    """Parse strict JSON: NaN and Infinity, which Python would accept, are not JSON."""
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError:
        return _NOT_JSON


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _content_is_serialized(content: list[types.ContentBlock], structured: dict[str, Any]) -> bool:
    if len(content) != 1 or not isinstance(content[0], types.TextContent):
        return False
    text = content[0].text
    parsed = _try_json(text)
    if parsed is not _NOT_JSON and parsed == structured:
        return True
    if set(structured) == {"result"}:
        wrapped = structured["result"]
        return text == wrapped or (parsed is not _NOT_JSON and parsed == wrapped)
    return False


def _item_json(item: types.ContentBlock) -> Any:
    return item.model_dump(mode="json", by_alias=True, exclude_none=True)


def _machine_array_item(item: types.ContentBlock) -> str:
    if isinstance(item, types.TextContent):
        if _try_json(item.text) is not _NOT_JSON:
            return item.text.strip()
        return json.dumps(item.text, ensure_ascii=False)
    return json.dumps(_item_json(item), ensure_ascii=False)


def _pretty_array_item(item: types.ContentBlock) -> Any:
    if isinstance(item, types.TextContent):
        parsed = _try_json(item.text)
        return item.text if parsed is _NOT_JSON else parsed
    return _item_json(item)


def _dump(value: Any, mode: OutputMode) -> str:
    if mode == "machine":
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, indent=2)
