"""Compile a tool's JSON input schema into click options, and the option values back into arguments.

The rules, chosen so that what a user types reaches the server unchanged or fails loudly:

- An option is named after the property in kebab case (``startTime`` becomes ``--start-time``),
  and the property's own name is what goes on the wire. The property's own spelling
  (``--startTime``) and its snake case (``--start_time``) are accepted as hidden aliases, unless
  another option already uses that spelling.
- An option the user leaves out is not sent. Schema defaults are shown in the help, never sent.
- A boolean is a ``--x/--no-x`` pair, so ``false`` can be sent.
- An ``enum`` of scalars is a fixed choice.
- An array is a repeatable option, one item per occurrence. A single occurrence holding a JSON
  array is taken as the whole array. Values are never split on commas.
- An object takes a JSON literal. Invalid JSON is a usage error, not a string.
- ``$ref`` to the schema's own ``$defs`` or ``definitions``, ``allOf`` and nullable ``anyOf`` /
  ``oneOf`` / ``type: [T, "null"]`` are resolved before an option is chosen. Anything else
  (a real union, a schema without a type) takes JSON when the value parses and fits the
  schema, and the raw string otherwise.
- ``--input-json`` passes the whole argument object as JSON, from a literal, ``@file`` or ``-``
  for stdin. ``--arg-file NAME=PATH`` reads one property's value from a file, or from stdin when
  the path is ``-``, exactly as if it had been typed after the property's option. Options given
  next to them are merged in; naming the same property twice is an error.
- The merged arguments are validated against the full schema with ``jsonschema`` before the
  call, which covers required properties, ``minimum`` / ``maximum`` and the rest.
- An option whose name collides with a built-in option (``--help``, ``--input-json``,
  ``--arg-file``, ``--json``, ``--pretty``) gets an ``arg-`` prefix.
- Help text is never truncated.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import click
import jsonschema
from jsonschema.exceptions import SchemaError
from jsonschema.protocols import Validator

Kind = Literal["boolean", "integer", "number", "string", "enum", "array", "object", "any"]

INPUT_JSON_OPTION = "--input-json"

ARG_FILE_OPTION = "--arg-file"

# Built into every tool command. The output-mode flags are added by the CLI, next to these.
BUILTIN_OPTIONS = frozenset({"--help", INPUT_JSON_OPTION, ARG_FILE_OPTION, "--json", "--pretty"})

# A property name that would turn into one of these is prefixed, so the built-in stays reachable.
COLLISION_PREFIX = "arg-"

_MAX_REF_DEPTH = 32

_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class SchemaShape:
    """What a (sub)schema accepts, after ``$ref``, ``allOf`` and nullability are resolved.

    Attributes:
        kind: The option type chosen for it.
        schema: The resolved schema, with the ``null`` alternative removed.
        nullable: Whether ``null`` is also allowed.
        enum: The allowed values, for ``kind == "enum"``.
        items: The shape of the items, for ``kind == "array"``.
    """

    kind: Kind
    schema: Mapping[str, Any]
    nullable: bool = False
    enum: tuple[Any, ...] = ()
    items: SchemaShape | None = None


@dataclass(frozen=True)
class Parameter:
    """One tool property as a click option.

    Attributes:
        name: The property name, sent on the wire unchanged.
        flag: The long option, ``--kebab-name``.
        aliases: Further accepted spellings, not shown in the help.
        dest: The click parameter name, a safe identifier unrelated to the property name.
        shape: What the property accepts.
        required: Whether the schema lists it as required.
        description: The property's description, if any.
    """

    name: str
    flag: str
    aliases: tuple[str, ...]
    dest: str
    shape: SchemaShape
    required: bool
    description: str | None


@dataclass
class CompiledTool:
    """A tool's input schema, ready to build a click command from and to turn values back into arguments."""

    schema: Mapping[str, Any]
    parameters: list[Parameter] = field(default_factory=list)
    input_json_dest: str = "input_json"
    arg_file_dest: str = "arg_file"

    def options(self) -> list[click.Parameter]:
        """Build the click options: the tool's properties first, then ``--input-json`` and ``--arg-file``."""
        options: list[click.Parameter] = [_build_option(self.schema, parameter) for parameter in self.parameters]
        options.append(
            click.Option(
                [INPUT_JSON_OPTION, self.input_json_dest],
                metavar="JSON|@FILE|-",
                help=(
                    "All arguments as one JSON object: a literal, @path to read a file, or - to read stdin. "
                    "Options given as well are merged in; naming the same property twice is an error."
                ),
            )
        )
        options.append(
            click.Option(
                [ARG_FILE_OPTION, self.arg_file_dest],
                metavar="NAME=PATH",
                multiple=True,
                help=(
                    "Read one property's value from a file, or from stdin when PATH is -, exactly as if it were "
                    "typed after the property's option. NAME is the option or the property name. Repeatable."
                ),
            )
        )
        return options

    def find(self, name: str) -> Parameter | None:
        """Find a parameter by its property name, its option or any alias, with or without the dashes."""
        wanted = name if name.startswith("--") else f"--{name}"
        for parameter in self.parameters:
            if name == parameter.name or wanted == parameter.flag or wanted in parameter.aliases:
                return parameter
        return None

    def arguments(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Turn the parsed click values into the tool's arguments, leaving out everything not given.

        Raises:
            click.BadParameter: A value cannot be converted or read, or a property is given twice.
        """
        arguments: dict[str, Any] = {}
        sources: dict[str, str] = {}
        raw_input = values.get(self.input_json_dest)
        arg_files = [self._parse_arg_file(spec) + (spec,) for spec in values.get(self.arg_file_dest) or ()]

        # Checked before anything is read, so a conflict never consumes stdin.
        stdin_users = [INPUT_JSON_OPTION] if raw_input in ("-", "@-") else []
        stdin_users += [f"{ARG_FILE_OPTION} {spec}" for _, path, spec in arg_files if path == "-"]
        if len(stdin_users) > 1:
            raise click.BadParameter(f"only one input can read stdin, got: {', '.join(stdin_users)}.")

        if raw_input is not None:
            for name, value in read_input_json(raw_input).items():
                arguments[name] = value
                sources[name] = INPUT_JSON_OPTION

        given: dict[str, tuple[Parameter, Any, str]] = {}
        for parameter in self.parameters:
            value = values.get(parameter.dest)
            if value is None or (isinstance(value, tuple) and len(value) == 0):
                continue
            given[parameter.name] = (parameter, value, parameter.flag)

        for parameter, path, spec in arg_files:
            if parameter.name in given:
                raise click.BadParameter(
                    f"{parameter.name!r} is given both as {given[parameter.name][2]} and with {ARG_FILE_OPTION}.",
                    param_hint=ARG_FILE_OPTION,
                )
            content = _read_arg_file(path)
            value = (content,) if parameter.shape.kind == "array" else content
            given[parameter.name] = (parameter, value, f"{ARG_FILE_OPTION} {spec}")

        for name, (parameter, value, source) in given.items():
            converted = convert_value(self.schema, parameter, value)
            if name in arguments:
                raise click.BadParameter(
                    f"{name!r} is given both in {sources[name]} and as {source}.", param_hint=parameter.flag
                )
            arguments[name] = converted
        return arguments

    def _parse_arg_file(self, spec: str) -> tuple[Parameter, str]:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise click.BadParameter(f"expected NAME=PATH, got {spec!r}.", param_hint=ARG_FILE_OPTION)
        parameter = self.find(name)
        if parameter is None:
            known = ", ".join(parameter.flag for parameter in self.parameters) or "none"
            raise click.BadParameter(f"no property named {name!r}. Known: {known}.", param_hint=ARG_FILE_OPTION)
        return parameter, path

    def summary(self) -> str:
        """The parameter list printed with a usage error, so no ``--help`` round trip is needed."""
        if not self.parameters:
            return "This tool takes no parameters."
        rows = [(_option_label(parameter), _summary_notes(parameter)) for parameter in self.parameters]
        width = min(max(len(left) for left, _ in rows), 40)
        lines = ["Parameters:"]
        for left, notes in rows:
            lines.append(f"  {left.ljust(width)}  {notes}".rstrip())
        return "\n".join(lines)

    def validate(self, arguments: Mapping[str, Any]) -> list[str]:
        """Validate the arguments against the full input schema.

        Returns:
            One readable message per problem, empty when the arguments are valid. A schema that
            is itself invalid is not held against the user, and validates everything.
        """
        validator = _validator_for(self.schema)
        if validator is None:
            return []
        messages = []
        for error in sorted(validator.iter_errors(dict(arguments)), key=lambda error: list(error.absolute_path)):
            messages.append(_describe_validation_error(self, error))
        return messages


def compile_tool(input_schema: Mapping[str, Any] | None) -> CompiledTool:
    """Compile a tool's ``inputSchema`` into options.

    Args:
        input_schema: The schema from ``tools/list``. None or an empty schema means no properties.
    """
    schema: Mapping[str, Any] = input_schema or {"type": "object"}
    root = _resolve(schema, schema)
    properties: Mapping[str, Any] = root.get("properties") or {}
    required = set(root.get("required") or [])

    shapes = {
        name: shape_of(schema, property_schema if isinstance(property_schema, Mapping) else {})
        for name, property_schema in properties.items()
    }
    flags = _allocate_flags(list(properties), {name for name, shape in shapes.items() if shape.kind == "boolean"})

    parameters = []
    for index, (name, property_schema) in enumerate(properties.items()):
        flag, aliases = flags[name]
        resolved = property_schema if isinstance(property_schema, Mapping) else {}
        description = _resolve(schema, resolved).get("description")
        if isinstance(description, str):
            description = inspect.cleandoc(description)
        parameters.append(
            Parameter(
                name=name,
                flag=flag,
                aliases=aliases,
                dest=f"arg_{index}",
                shape=shapes[name],
                required=name in required,
                description=description if isinstance(description, str) else None,
            )
        )

    return CompiledTool(schema=schema, parameters=parameters)


def _is_valid_flag(flag: str) -> bool:
    return re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9_-]*", flag) is not None


def kebab_case(name: str) -> str:
    """``startTime`` and ``start_time`` become ``start-time``; ``HTTPCode`` becomes ``http-code``."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", spaced)
    kebab = re.sub(r"[^A-Za-z0-9]+", "-", spaced).strip("-").lower()
    return kebab or "arg"


def _allocate_flags(names: list[str], booleans: set[str]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Give every property its option and aliases, independently of the order the schema lists them in.

    Three passes, each in sorted order: the kebab-case options of all properties, then each
    property's own name, then the snake-case forms. A spelling from an earlier pass always wins,
    so a property's own name beats another property's derived alias. A boolean also claims the
    ``--no-`` form of each spelling, and booleans go first in the first pass, so ``false`` can
    always be sent; a property whose option would be that ``--no-`` form is renamed.
    """
    used: set[str] = set(BUILTIN_OPTIONS)

    def spellings(flag: str, name: str) -> set[str]:
        return {flag, f"--no-{flag[2:]}"} if name in booleans else {flag}

    def free(flag: str, name: str) -> bool:
        return _is_valid_flag(flag) and not spellings(flag, name) & used

    primaries: dict[str, str] = {}
    for name in sorted(names, key=lambda item: (item not in booleans, item)):
        base = f"--{kebab_case(name)}"
        if spellings(base, name) & BUILTIN_OPTIONS:
            base = f"--{COLLISION_PREFIX}{kebab_case(name)}"
        flag = base
        counter = 2
        while not free(flag, name):
            flag = f"{base}-{counter}"
            counter += 1
        used |= spellings(flag, name)
        primaries[name] = flag

    # A boolean's alias must not take the --no- form that is another property's own name.
    own_names = {f"--{name}": name for name in names}

    def claimed_by_other(alias: str, name: str) -> bool:
        return any(own_names.get(spelling, name) != name for spelling in spellings(alias, name))

    def own_name(name: str) -> str:
        return f"--{name}"

    def snake_case(name: str) -> str:
        return f"--{kebab_case(name).replace('-', '_')}"

    aliases: dict[str, list[str]] = {name: [] for name in names}
    for derive in (own_name, snake_case):
        for name in sorted(names):
            alias = derive(name)
            if (
                alias != primaries[name]
                and alias not in aliases[name]
                and free(alias, name)
                and not claimed_by_other(alias, name)
            ):
                used |= spellings(alias, name)
                aliases[name].append(alias)

    return {name: (primaries[name], tuple(aliases[name])) for name in names}


def shape_of(root: Mapping[str, Any], schema: Mapping[str, Any]) -> SchemaShape:
    """Work out what a property schema accepts and which option type fits it."""
    resolved = _resolve(root, schema)
    nullable = False

    if "const" in resolved and "enum" not in resolved:
        resolved = {**resolved, "enum": [resolved["const"]]}

    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        values = [value for value in enum if value is not None]
        nullable = len(values) != len(enum)
        if values and all(isinstance(value, str | int | float | bool) for value in values):
            return SchemaShape("enum", resolved, nullable, enum=tuple(values))
        return SchemaShape("any", resolved, nullable)

    for union_key in ("anyOf", "oneOf"):
        branches = resolved.get(union_key)
        if isinstance(branches, list) and branches:
            resolved_branches = [_resolve(root, branch) for branch in branches if isinstance(branch, Mapping)]
            non_null = [branch for branch in resolved_branches if not _is_null_schema(branch)]
            nullable = len(non_null) != len(resolved_branches)
            if len(non_null) == 1:
                rest = {key: value for key, value in resolved.items() if key != union_key}
                inner = shape_of(root, {**rest, **non_null[0]})
                return SchemaShape(
                    inner.kind, inner.schema, True if nullable else inner.nullable, inner.enum, inner.items
                )
            return SchemaShape("any", resolved, nullable)

    declared = resolved.get("type")
    types: list[str]
    if isinstance(declared, list):
        types = [str(item) for item in declared]
    elif isinstance(declared, str):
        types = [declared]
    elif "properties" in resolved or "additionalProperties" in resolved:
        types = ["object"]
    elif "items" in resolved:
        types = ["array"]
    else:
        types = []

    if "null" in types:
        nullable = True
        types = [item for item in types if item != "null"]

    if len(types) != 1:
        return SchemaShape("any", resolved, nullable)

    kind = types[0]
    if kind == "array":
        items = resolved.get("items")
        item_shape = shape_of(root, items) if isinstance(items, Mapping) else SchemaShape("any", {})
        return SchemaShape("array", resolved, nullable, items=item_shape)
    if kind in ("boolean", "integer", "number", "string", "object"):
        return SchemaShape(kind, resolved, nullable)  # type: ignore[arg-type]
    return SchemaShape("any", resolved, nullable)


def _is_null_schema(schema: Mapping[str, Any]) -> bool:
    return schema.get("type") == "null" or schema.get("const", ...) is None or schema.get("enum") == [None]


def _resolve(root: Mapping[str, Any], schema: Mapping[str, Any], depth: int = 0) -> Mapping[str, Any]:
    """Inline a local ``$ref`` and merge ``allOf``, at the top level of ``schema`` only."""
    if depth > _MAX_REF_DEPTH:
        return schema

    current: Mapping[str, Any] = schema
    reference = current.get("$ref")
    if isinstance(reference, str):
        target = _lookup_ref(root, reference)
        if target is not None:
            siblings = {key: value for key, value in current.items() if key != "$ref"}
            current = {**_resolve(root, target, depth + 1), **siblings}

    parts = current.get("allOf")
    if isinstance(parts, list) and parts:
        merged: dict[str, Any] = {key: value for key, value in current.items() if key != "allOf"}
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            resolved_part = _resolve(root, part, depth + 1)
            for key, value in resolved_part.items():
                if key == "properties" and isinstance(value, Mapping):
                    existing = merged.get("properties") or {}
                    # Earlier definitions win; properties keep the order they are declared in.
                    merged["properties"] = {**existing, **{k: v for k, v in value.items() if k not in existing}}
                elif key == "required" and isinstance(value, list):
                    merged["required"] = list(dict.fromkeys([*(merged.get("required") or []), *value]))
                else:
                    merged.setdefault(key, value)
        current = merged

    return current


def _lookup_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any] | None:
    if not reference.startswith("#"):
        return None
    node: Any = root
    for token in reference[1:].split("/"):
        if token == "":
            continue
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    return node if isinstance(node, Mapping) else None


class ToolOption(click.Option):
    """An option whose help lists only its primary spelling, not the aliases."""

    def __init__(self, parameter: Parameter, declarations: list[str], **attrs: Any) -> None:
        super().__init__(declarations, **attrs)
        self.tool_parameter = parameter

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        record = super().get_help_record(ctx)
        if record is None:
            return None
        return _option_label(self.tool_parameter), record[1]


def _option_label(parameter: Parameter) -> str:
    if parameter.shape.kind == "boolean":
        return f"{parameter.flag} / --no-{parameter.flag[2:]}"
    return f"{parameter.flag} {_metavar(_display_shape(parameter.shape))}"


def _build_option(root: Mapping[str, Any], parameter: Parameter) -> click.Option:
    shape = parameter.shape
    help_text = _option_help(parameter)
    names = [parameter.flag, *parameter.aliases]

    if shape.kind == "boolean":
        pairs = [f"{name}/--no-{name[2:]}" for name in names]
        return ToolOption(parameter, [*pairs, parameter.dest], default=None, help=help_text)

    if shape.kind == "enum":
        choices = [_choice_label(value) for value in shape.enum]
        return ToolOption(
            parameter, [*names, parameter.dest], type=click.Choice(choices, case_sensitive=True), help=help_text
        )

    if shape.kind == "array":
        return ToolOption(
            parameter, [*names, parameter.dest], multiple=True, metavar=_metavar(shape.items), help=help_text
        )

    return ToolOption(parameter, [*names, parameter.dest], metavar=_metavar(shape), help=help_text)


def _display_shape(shape: SchemaShape) -> SchemaShape | None:
    return shape.items if shape.kind == "array" else shape


def _summary_notes(parameter: Parameter) -> str:
    shape = parameter.shape
    notes = []
    if parameter.required:
        notes.append("[required]")
    if shape.kind == "array":
        notes.append("[repeatable]")
        if shape.items is not None and shape.items.kind == "enum":
            notes.append("allowed: " + ", ".join(_choice_label(value) for value in shape.items.enum))
    for keyword in ("minimum", "maximum"):
        if keyword in shape.schema and isinstance(shape.schema[keyword], int | float):
            notes.append(f"{keyword} {shape.schema[keyword]}")
    return " ".join(notes)


def _choice_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _metavar(shape: SchemaShape | None) -> str:
    if shape is None:
        return "VALUE"
    return {
        "boolean": "BOOLEAN",
        "integer": "INTEGER",
        "number": "NUMBER",
        "string": "TEXT",
        "enum": "[" + "|".join(_choice_label(value) for value in shape.enum) + "]",
        "array": "JSON",
        "object": "JSON",
        "any": "VALUE",
    }[shape.kind]


def _option_help(parameter: Parameter) -> str:
    shape = parameter.shape
    parts: list[str] = []
    if parameter.description:
        parts.append(parameter.description.strip())

    details: list[str] = []
    if shape.kind == "array":
        item = shape.items
        if item is not None and item.kind == "enum":
            details.append("Allowed items: " + ", ".join(_choice_label(value) for value in item.enum) + ".")
        details.append("Repeat the option for each item, or pass the whole array as one JSON literal.")
    elif shape.kind == "object":
        details.append("A JSON object.")
    elif shape.kind == "any":
        details.append("JSON, or text when the value is not JSON.")

    for keyword, label in (
        ("minimum", "Minimum"),
        ("exclusiveMinimum", "Greater than"),
        ("maximum", "Maximum"),
        ("exclusiveMaximum", "Less than"),
        ("minLength", "Minimum length"),
        ("maxLength", "Maximum length"),
        ("minItems", "Minimum items"),
        ("maxItems", "Maximum items"),
        ("pattern", "Pattern"),
        ("format", "Format"),
    ):
        if keyword in shape.schema and not isinstance(shape.schema[keyword], bool | dict | list):
            details.append(f"{label}: {shape.schema[keyword]}.")

    if shape.schema.get("default") is not None:
        details.append(f"Server default: {json.dumps(shape.schema['default'], ensure_ascii=False)}.")
    if parameter.required:
        details.append("[required]")

    return " ".join([*parts, *details])


def convert_value(root: Mapping[str, Any], parameter: Parameter, value: Any) -> Any:
    """Convert what click parsed for one option into the JSON value to send.

    Raises:
        click.BadParameter: The value does not fit the option's type.
    """
    shape = parameter.shape
    try:
        if shape.kind == "array":
            return _convert_array(root, shape, tuple(value))
        return _convert_scalar(root, shape, value)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint=parameter.flag) from error


def _convert_array(root: Mapping[str, Any], shape: SchemaShape, values: tuple[str, ...]) -> list[Any]:
    item = shape.items or SchemaShape("any", {})
    if len(values) == 1 and values[0].lstrip().startswith("["):
        try:
            parsed = json.loads(values[0])
        except json.JSONDecodeError as error:
            if item.kind not in ("string", "any", "enum"):
                raise ValueError(f"{values[0]!r} is not a valid JSON array: {error.msg}.") from error
        else:
            if isinstance(parsed, list):
                return parsed
    return [_convert_scalar(root, item, value) for value in values]


def _convert_scalar(root: Mapping[str, Any], shape: SchemaShape, value: Any) -> Any:
    if shape.kind == "boolean":
        if isinstance(value, bool):
            return value
        return _parse_bool(str(value))
    if shape.kind == "enum":
        for allowed in shape.enum:
            if _choice_label(allowed) == value:
                return allowed
        raise ValueError(f"{value!r} is not one of {', '.join(_choice_label(v) for v in shape.enum)}.")
    if shape.kind == "integer":
        return _parse_integer(str(value))
    if shape.kind == "number":
        return _parse_number(str(value))
    if shape.kind == "string":
        return str(value)
    if shape.kind == "object":
        parsed = _parse_json(str(value), "a JSON object")
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object, got {type(parsed).__name__}.")
        return parsed
    if shape.kind == "array":
        parsed = _parse_json(str(value), "a JSON array")
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array, got {type(parsed).__name__}.")
        return parsed
    return _convert_any(root, shape, str(value))


def _convert_any(root: Mapping[str, Any], shape: SchemaShape, value: str) -> Any:
    """Prefer the JSON reading of the value when it fits the schema, the plain string otherwise."""
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        return value
    validator = _validator_for(_with_definitions(root, shape.schema))
    if validator is None or validator.is_valid(parsed):
        return parsed
    if validator.is_valid(value):
        return value
    return parsed


def _with_definitions(root: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    combined = dict(schema)
    for key in ("$defs", "definitions"):
        if key in root and key not in combined:
            combined[key] = root[key]
    return combined


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    raise ValueError(f"{value!r} is not a boolean; use true or false.")


def _parse_integer(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"{value!r} is not an integer.") from None


def _parse_number(value: str) -> int | float:
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{value!r} is not a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{value!r} is not a finite number.")
    return number


def _parse_json(value: str, what: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{value!r} is not {what}: {error.msg} at position {error.pos}.") from error


def _read_arg_file(path: str) -> str:
    """Read an ``--arg-file`` value exactly as stored, without stripping a trailing newline."""
    if path == "-":
        return sys.stdin.read()
    file = Path(path).expanduser()
    try:
        return file.read_text(encoding="utf-8")
    except OSError as error:
        raise click.BadParameter(
            f"cannot read {file}: {error.strerror or error}", param_hint=ARG_FILE_OPTION
        ) from error
    except UnicodeDecodeError as error:
        raise click.BadParameter(f"{file} is not UTF-8 text: {error.reason}", param_hint=ARG_FILE_OPTION) from error


def read_input_json(raw: str) -> dict[str, Any]:
    """Read the ``--input-json`` value: a JSON literal, ``@path`` for a file, or ``-`` / ``@-`` for stdin.

    Raises:
        click.BadParameter: The source cannot be read, or does not hold a JSON object.
    """
    source = "the value"
    text = raw
    if raw in ("-", "@-"):
        source = "stdin"
        text = sys.stdin.read()
    elif raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        source = str(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise click.BadParameter(
                f"cannot read {path}: {error.strerror or error}", param_hint=INPUT_JSON_OPTION
            ) from error
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise click.BadParameter(
            f"{source} is not valid JSON: {error.msg} at line {error.lineno} column {error.colno}.",
            param_hint=INPUT_JSON_OPTION,
        ) from error
    if not isinstance(parsed, dict):
        raise click.BadParameter(
            f"{source} must hold a JSON object of arguments, got {type(parsed).__name__}.", param_hint=INPUT_JSON_OPTION
        )
    return parsed


def _validator_for(schema: Mapping[str, Any]) -> Validator | None:
    try:
        validator_class = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
        validator_class.check_schema(dict(schema))
    except SchemaError:
        print("warning: the tool's input schema is invalid, arguments are not validated", file=sys.stderr)
        return None
    validator: Validator = validator_class(schema)
    return validator


def _describe_validation_error(tool: CompiledTool, error: jsonschema.ValidationError) -> str:
    path = list(error.absolute_path)
    flags = {parameter.name: parameter.flag for parameter in tool.parameters}

    if error.validator == "required" and not path:
        missing = _missing_property(error)
        if missing is not None:
            flag = flags.get(missing)
            return f"missing required {flag or repr(missing)}" + (f" ({missing!r})" if flag else "")

    if error.validator == "additionalProperties" and not path:
        return f"unexpected arguments: {error.message}"

    if path and isinstance(path[0], str):
        head = flags.get(path[0], repr(path[0]))
        rest = "".join(f"[{item!r}]" for item in path[1:])
        return f"{head}{rest}: {error.message}"
    return error.message


def _missing_property(error: jsonschema.ValidationError) -> str | None:
    match = re.match(r"^'(.+)' is a required property$", error.message)
    return match.group(1) if match else None


def signature(command_path: str, parameters: Sequence[Parameter]) -> str:
    """A one-line summary of a tool's options, required ones first, for usage errors."""
    required = [f"{parameter.flag} {_metavar(parameter.shape)}" for parameter in parameters if parameter.required]
    optional = [f"[{parameter.flag}]" for parameter in parameters if not parameter.required]
    return " ".join([command_path, *required, *optional])
