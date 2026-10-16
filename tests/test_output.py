"""Unit tests for how results are printed."""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp import types
from pydantic import AnyUrl

from mcp_to_cli_kit.output import finish, render, resolve_mode


def text(value: str) -> types.TextContent:
    return types.TextContent(type="text", text=value)


def result(*content: types.ContentBlock, structured: dict[str, Any] | None = None) -> types.CallToolResult:
    return types.CallToolResult(content=list(content), structuredContent=structured)


class TestMode:
    def test_flags_win_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")

        assert resolve_mode("pretty", "machine") == "pretty"
        assert resolve_mode(None, "pretty") == "pretty"
        assert resolve_mode(None, None) == "machine"

    def test_a_person_gets_pretty(self) -> None:
        assert resolve_mode(None, None) == "pretty"


class TestSingleItem:
    raw = '{"b": 1,  "a": [1.10]}'

    def test_machine_is_verbatim(self) -> None:
        assert render(result(text(self.raw)), "machine") == self.raw

    def test_pretty_reindents(self) -> None:
        assert render(result(text(self.raw)), "pretty") == '{\n  "b": 1,\n  "a": [\n    1.1\n  ]\n}'

    def test_pretty_keeps_non_ascii(self) -> None:
        assert render(result(text('{"a": "\\u017e"}')), "pretty") == '{\n  "a": "ž"\n}'

    def test_text_that_is_not_json(self) -> None:
        assert render(result(text("Total: 3\n- a")), "pretty") == "Total: 3\n- a"

    def test_nan_is_not_json(self) -> None:
        assert render(result(text("NaN")), "pretty") == "NaN"

    def test_non_text_item(self) -> None:
        image = types.ImageContent(type="image", data="aGk=", mimeType="image/png")

        assert json.loads(render(result(image), "machine")) == {
            "type": "image",
            "data": "aGk=",
            "mimeType": "image/png",
        }

    def test_empty(self) -> None:
        assert render(result(), "machine") == ""


class TestSeveralItems:
    items = (text('{"a": 1}'), text("plain"), text(" [1, 2] "))

    def test_machine_embeds_the_server_text(self) -> None:
        assert render(result(*self.items), "machine") == '[{"a": 1},"plain",[1, 2]]'

    def test_pretty_is_the_same_array_indented(self) -> None:
        rendered = render(result(*self.items), "pretty")

        assert json.loads(rendered) == [{"a": 1}, "plain", [1, 2]]
        assert rendered.startswith("[\n  {")

    def test_machine_array_is_valid_json_with_non_text_items(self) -> None:
        link = types.ResourceLink(type="resource_link", uri=AnyUrl("file:///x"), name="x")

        assert json.loads(render(result(text("a"), link), "machine"))[1]["uri"] == "file:///x"


class TestStructuredContent:
    def test_structured_wins_over_a_summary(self) -> None:
        rendered = render(result(text("2 rows"), structured={"rows": [1, 2]}), "machine")

        assert rendered == '{"rows": [1, 2]}'

    def test_serialized_structured_keeps_the_server_text(self) -> None:
        raw = '{"rows":[1,2]}'

        assert render(result(text(raw), structured={"rows": [1, 2]}), "machine") == raw

    def test_wrapped_plain_result_prints_the_text(self) -> None:
        assert render(result(text("hello"), structured={"result": "hello"}), "machine") == "hello"

    def test_structured_alone(self) -> None:
        assert render(result(structured={"a": "ž"}), "pretty") == '{\n  "a": "ž"\n}'


class TestFinish:
    @pytest.mark.parametrize(
        ("text_value", "mode", "tty", "expected"),
        [
            ("x", "machine", False, "x"),
            ("x", "machine", True, "x\n"),
            ("x", "pretty", False, "x\n"),
            ("x\n", "pretty", True, "x\n"),
            ("", "pretty", True, ""),
        ],
    )
    def test_newline(self, text_value: str, mode: Any, tty: bool, expected: str) -> None:
        assert finish(text_value, mode, tty) == expected
