"""Unit tests for compiling input schemas into options and option values into arguments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from mcp_to_cli_kit.arguments import CompiledTool, compile_tool, kebab_case, shape_of


def command_for(schema: dict[str, Any]) -> tuple[click.Command, CompiledTool, list[dict[str, Any]]]:
    """A click command that records the arguments it would send."""
    compiled = compile_tool(schema)
    sent: list[dict[str, Any]] = []

    def callback(**values: Any) -> None:
        arguments = compiled.arguments(values)
        problems = compiled.validate(arguments)
        if problems:
            raise click.UsageError("; ".join(problems))
        sent.append(arguments)

    return click.Command("tool", params=compiled.options(), callback=callback), compiled, sent


def run(schema: dict[str, Any], *args: str, input: str | None = None) -> tuple[int, list[dict[str, Any]], str]:
    command, _, sent = command_for(schema)
    result = CliRunner().invoke(command, list(args), input=input)
    return result.exit_code, sent, result.output


class TestNames:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("startTime", "start-time"),
            ("start_time", "start-time"),
            ("HTTPCode", "http-code"),
            ("already-kebab", "already-kebab"),
            ("x", "x"),
            ("__", "arg"),
        ],
    )
    def test_kebab_case(self, name: str, expected: str) -> None:
        assert kebab_case(name) == expected

    def test_flag_and_aliases(self) -> None:
        compiled = compile_tool({"type": "object", "properties": {"applicationName": {"type": "string"}}})

        parameter = compiled.parameters[0]
        assert parameter.flag == "--application-name"
        assert parameter.aliases == ("--applicationName", "--application_name")

    @pytest.mark.parametrize("flag", ["--application-name", "--applicationName", "--application_name"])
    def test_every_spelling_sends_the_schema_name(self, flag: str) -> None:
        schema = {"type": "object", "properties": {"applicationName": {"type": "string"}}}

        assert run(schema, flag, "v")[1] == [{"applicationName": "v"}]

    def test_builtin_names_get_a_prefix(self) -> None:
        schema = {"type": "object", "properties": {name: {"type": "string"} for name in ("json", "pretty", "help")}}
        compiled = compile_tool(schema)

        assert [parameter.flag for parameter in compiled.parameters] == ["--arg-json", "--arg-pretty", "--arg-help"]
        assert all(parameter.aliases == () for parameter in compiled.parameters)
        assert run(schema, "--arg-help", "x")[1] == [{"help": "x"}]

    @pytest.mark.parametrize("order", [("fooBar", "foo_bar"), ("foo_bar", "fooBar")])
    def test_own_name_beats_another_property_s_alias_in_any_order(self, order: tuple[str, str]) -> None:
        types = {"fooBar": {"type": "string"}, "foo_bar": {"type": "integer"}}
        schema = {"type": "object", "properties": {name: types[name] for name in order}}
        compiled = compile_tool(schema)

        flags = {parameter.name: (parameter.flag, parameter.aliases) for parameter in compiled.parameters}
        assert flags == {"fooBar": ("--foo-bar", ("--fooBar",)), "foo_bar": ("--foo-bar-2", ("--foo_bar",))}
        assert run(schema, "--foo_bar", "3", "--fooBar", "a")[1] == [{"fooBar": "a", "foo_bar": 3}]

    @pytest.mark.parametrize("order", [("verbose", "no_verbose"), ("no_verbose", "verbose")])
    def test_negative_boolean_form_is_reserved(self, order: tuple[str, str]) -> None:
        types = {"verbose": {"type": "boolean"}, "no_verbose": {"type": "string"}}
        schema = {"type": "object", "properties": {name: types[name] for name in order}}
        compiled = compile_tool(schema)

        flags = {parameter.name: parameter.flag for parameter in compiled.parameters}
        assert flags == {"verbose": "--verbose", "no_verbose": "--no-verbose-2"}
        assert run(schema, "--no-verbose", "--no_verbose", "x")[1] == [{"verbose": False, "no_verbose": "x"}]

    def test_boolean_alias_needs_its_negative_free(self) -> None:
        schema = {
            "type": "object",
            "properties": {"dryRun": {"type": "boolean"}, "no-dryRun": {"type": "string"}},
        }
        compiled = compile_tool(schema)

        dry_run = compiled.find("dryRun")
        assert dry_run is not None
        assert "--dryRun" not in dry_run.aliases, "--no-dryRun is the other property's own name"
        assert "--dry_run" in dry_run.aliases
        assert run(schema, "--no-dry-run", "--no-dryRun", "x")[1] == [{"dryRun": False, "no-dryRun": "x"}]

    def test_find_by_any_spelling(self) -> None:
        compiled = compile_tool({"type": "object", "properties": {"startTime": {"type": "string"}}})

        for name in ("startTime", "start-time", "--start-time", "start_time", "--startTime"):
            found = compiled.find(name)
            assert found is not None and found.name == "startTime"
        assert compiled.find("nope") is None


class TestShapes:
    root: dict[str, Any] = {
        "$defs": {
            "Point": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "Kind": {"enum": ["a", "b"]},
            "Alias": {"$ref": "#/$defs/Kind"},
        }
    }

    @pytest.mark.parametrize(
        ("schema", "kind", "nullable"),
        [
            ({"type": "string"}, "string", False),
            ({"type": ["integer", "null"]}, "integer", True),
            ({"anyOf": [{"type": "boolean"}, {"type": "null"}]}, "boolean", True),
            ({"oneOf": [{"type": "number"}, {"type": "null"}]}, "number", True),
            ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "any", False),
            ({"$ref": "#/$defs/Point"}, "object", False),
            ({"anyOf": [{"$ref": "#/$defs/Point"}, {"type": "null"}]}, "object", True),
            ({"$ref": "#/$defs/Alias"}, "enum", False),
            ({"allOf": [{"$ref": "#/$defs/Point"}]}, "object", False),
            ({"enum": ["a", None]}, "enum", True),
            ({"const": "fixed"}, "enum", False),
            ({"properties": {"a": {}}}, "object", False),
            ({}, "any", False),
            ({"$ref": "#/$defs/Missing"}, "any", False),
        ],
    )
    def test_kind(self, schema: dict[str, Any], kind: str, nullable: bool) -> None:
        shape = shape_of(self.root, schema)

        assert (shape.kind, shape.nullable) == (kind, nullable)

    def test_array_items(self) -> None:
        shape = shape_of(self.root, {"type": "array", "items": {"$ref": "#/$defs/Point"}})

        assert shape.kind == "array"
        assert shape.items is not None and shape.items.kind == "object"

    def test_cyclic_ref_terminates(self) -> None:
        root = {"$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}}

        assert shape_of(root, {"$ref": "#/$defs/A"}).kind == "any"

    def test_all_of_merges_properties_and_required(self) -> None:
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                {"properties": {"b": {"type": "string"}}, "required": ["b"]},
            ]
        }
        compiled = compile_tool(schema)

        assert [(parameter.name, parameter.required) for parameter in compiled.parameters] == [
            ("a", True),
            ("b", True),
        ]


class TestValues:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "default": 5},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "maybe": {"type": ["boolean", "null"]},
            "kind": {"enum": ["a", "b"]},
            "level": {"enum": [1, 2]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "ids": {"type": "array", "items": {"type": "integer"}},
            "shape": {"type": "object"},
            "either": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        },
    }

    def test_nothing_given_sends_nothing(self) -> None:
        assert run(self.schema)[1] == [{}]

    def test_types(self) -> None:
        code, sent, _ = run(self.schema, "--count", "3", "--ratio", "2", "--kind", "b", "--level", "2")

        assert code == 0
        assert sent == [{"count": 3, "ratio": 2, "kind": "b", "level": 2}]

    def test_booleans(self) -> None:
        assert run(self.schema, "--flag", "--no-maybe")[1] == [{"flag": True, "maybe": False}]
        assert run(self.schema, "--no-flag")[1] == [{"flag": False}]

    def test_array_items_are_typed(self) -> None:
        assert run(self.schema, "--ids", "1", "--ids", "2")[1] == [{"ids": [1, 2]}]

    def test_array_is_never_split_on_commas(self) -> None:
        assert run(self.schema, "--tags", "a,b")[1] == [{"tags": ["a,b"]}]

    def test_array_json_literal(self) -> None:
        assert run(self.schema, "--ids", "[1, 2]")[1] == [{"ids": [1, 2]}]

    def test_string_item_that_looks_like_broken_json_stays_text(self) -> None:
        assert run(self.schema, "--tags", "[draft")[1] == [{"tags": ["[draft"]}]

    def test_bad_array_item(self) -> None:
        code, _, output = run(self.schema, "--ids", "x")

        assert code == 2
        assert "'x' is not an integer" in output

    def test_object_must_be_valid_json(self) -> None:
        code, sent, output = run(self.schema, "--shape", "{oops")

        assert code == 2
        assert sent == []
        assert "is not a JSON object" in output

    def test_union_prefers_json_that_fits(self) -> None:
        assert run(self.schema, "--either", "5")[1] == [{"either": 5}]
        assert run(self.schema, "--either", "five")[1] == [{"either": "five"}]
        assert run(self.schema, "--either", "[1]")[1] == [{"either": "[1]"}], "JSON that fits no branch stays text"

    def test_minimum_is_validated(self) -> None:
        code, sent, output = run(self.schema, "--count", "0")

        assert code == 2
        assert sent == []
        assert "--count: 0 is less than the minimum of 1" in output

    def test_unknown_enum_value(self) -> None:
        code, _, output = run(self.schema, "--kind", "c")

        assert code == 2
        assert "'c' is not one of" in output

    def test_required_is_validated(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}

        code, _, output = run(schema)

        assert code == 2
        assert "missing required --q ('q')" in output

    def test_nested_required_from_a_ref(self) -> None:
        schema = {
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/P"}},
            "$defs": {"P": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
        }

        code, _, output = run(schema, "--p", "{}")

        assert code == 2
        assert "--p: 'x' is a required property" in output


class TestInputs:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"body": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
    }

    def test_input_json_literal(self) -> None:
        assert run(self.schema, "--input-json", '{"body": "x", "extra": 1}')[1] == [{"body": "x", "extra": 1}]

    def test_input_json_file(self, tmp_path: Path) -> None:
        file = tmp_path / "args.json"
        file.write_text('{"body": "from file"}', encoding="utf-8")

        assert run(self.schema, "--input-json", f"@{file}")[1] == [{"body": "from file"}]

    def test_input_json_must_be_an_object(self) -> None:
        code, _, output = run(self.schema, "--input-json", "[1]")

        assert code == 2
        assert "must hold a JSON object" in output

    def test_input_json_invalid(self) -> None:
        code, _, output = run(self.schema, "--input-json", "{")

        assert code == 2
        assert "is not valid JSON" in output

    def test_input_json_missing_file(self, tmp_path: Path) -> None:
        code, _, output = run(self.schema, "--input-json", f"@{tmp_path / 'none.json'}")

        assert code == 2
        assert "cannot read" in output

    def test_arg_file_keeps_the_content_exactly(self, tmp_path: Path) -> None:
        file = tmp_path / "body.md"
        file.write_text("line\n\n  indented\n", encoding="utf-8")

        assert run(self.schema, "--arg-file", f"body={file}")[1] == [{"body": "line\n\n  indented\n"}]

    def test_arg_file_from_stdin(self) -> None:
        assert run(self.schema, "--arg-file", "body=-", input="piped")[1] == [{"body": "piped"}]

    def test_arg_file_for_an_array_is_one_occurrence(self, tmp_path: Path) -> None:
        file = tmp_path / "tags.json"
        file.write_text('["a", "b"]', encoding="utf-8")

        assert run(self.schema, "--arg-file", f"tags={file}")[1] == [{"tags": ["a", "b"]}]

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--arg-file", "body"], "expected NAME=PATH"),
            (["--arg-file", "nope=x"], "no property named 'nope'"),
            (["--arg-file", "body=/nonexistent/file"], "cannot read"),
            (["--body", "a", "--arg-file", "body=-"], "given both as --body and with --arg-file"),
            (["--input-json", '{"body": "a"}', "--arg-file", "body=-"], "given both in --input-json"),
            (["--input-json", "-", "--arg-file", "tags=-"], "only one input can read stdin"),
        ],
    )
    def test_arg_file_errors(self, args: list[str], message: str) -> None:
        code, sent, output = run(self.schema, *args, input="{}")

        assert code == 2
        assert sent == []
        assert message in " ".join(output.split())


def test_stdin_conflict_is_found_before_stdin_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    class Untouchable:
        def read(self) -> str:
            raise AssertionError("stdin must not be read")

    monkeypatch.setattr(sys, "stdin", Untouchable())
    compiled = compile_tool(TestInputs.schema)
    values = {"input_json": "-", "arg_file": ("body=-",)}

    with pytest.raises(click.BadParameter, match="only one input can read stdin"):
        compiled.arguments(values)


class TestHelp:
    def test_help_is_never_truncated(self) -> None:
        description = "Start. " + " ".join(f"Rule {i} applies." for i in range(80)) + " End."
        schema = {"type": "object", "properties": {"sql": {"type": "string", "description": description}}}
        command, _, _ = command_for(schema)

        output = CliRunner().invoke(command, ["--help"], terminal_width=80).output

        flat = " ".join(output.split())
        assert "Rule 79 applies. End." in flat
        assert "Rule 0 applies." in flat

    def test_help_shows_allowed_items_and_constraints(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "metrics": {"type": "array", "items": {"enum": ["cpu", "io"]}, "description": "Metrics."},
                "step": {"type": "integer", "minimum": 0, "maximum": 60, "default": 10},
            },
        }
        command, compiled, _ = command_for(schema)

        flat = " ".join(CliRunner().invoke(command, ["--help"]).output.split())

        assert "Allowed items: cpu, io." in flat
        assert "Repeat the option for each item" in flat
        assert "Minimum: 0. Maximum: 60. Server default: 10." in flat
        assert "--metrics [cpu|io]" in flat
        summary = compiled.summary()
        assert "--metrics [cpu|io]" in summary and "[repeatable]" in summary and "allowed: cpu, io" in summary
        assert "minimum 0 maximum 60" in summary

    def test_summary_without_parameters(self) -> None:
        assert compile_tool(None).summary() == "This tool takes no parameters."
