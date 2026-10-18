"""Configuration of a kit CLI: which MCP servers it talks to and how their tools are mounted.

The configuration comes from three layers, each overriding the one before it:

1. the default the consumer package passes in code, in the same shape as the file;
2. the YAML file, ``$XDG_CONFIG_HOME/<bin>/config.yaml`` (``~/.config/<bin>/config.yaml`` when
   the variable is unset) or the path given by ``--config``;
3. environment variables named ``<PREFIX><KEY>``, where the prefix is derived from the bin
   name (``my-tool`` becomes ``MY_TOOL_``) and nested keys are joined with ``__``.

The layers are merged as plain dictionaries before the model validates the result, so a
layer may set a single key of a single server and leave the rest alone. Lists and the
``mount`` key are replaced as a whole, never merged item by item, so overriding only the
group of a mount drops the help text the lower layer gave it. A layer can add servers and
change them, but it cannot remove a server that the code default defines.

Environment variables::

    MY_TOOL_HELP=...                         top-level help text
    MY_TOOL_SERVERS__ALPHA__URL=...          a key of the server "alpha"
    MY_TOOL_SERVERS__ALPHA__HEADERS__X_API_KEY=...
                                             a header; "_" becomes "-" unless the header
                                             already exists under another spelling
    MY_TOOL_SERVERS__ALPHA__HIDDEN_TOOLS='["a", "b"]'
    MY_TOOL_SERVERS__ALPHA__STRIP_PREFIXES='["vendor_"]'
                                             lists, objects and mounts are given as JSON
    MY_TOOL_SERVERS__BETA__MOUNT__GROUP=b    one key of a mount, which replaces the whole mount

The whole variable name, the prefix included, is matched case-insensitively, and a server
name matches with ``-`` and ``_`` treated as the same character. Server names therefore
cannot contain two separators in a row, which would read as the ``__`` nesting.

Values of ``url`` and of ``headers`` may refer to environment variables as ``${NAME}``
(``$${`` gives a literal ``${``; a ``$`` not followed by ``{`` stays as it is). The references
are kept as written when the config is loaded and expanded only when a server is contacted,
so an unset variable fails only the commands that need that server. See ``expand_server``.

Environment variables cannot add a server, only change one the default or the file defines.
A variable that names no configured server or no known key is ignored and logged at debug
level, so a short prefix does not break the CLI over an unrelated variable. A known server
with an unknown key, and invalid JSON in a value meant to be JSON, are errors.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mcp_to_cli_kit import redaction

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "config.yaml"

ENV_NESTING = "__"

# Single separators only: "a__b" or "a-_b" would read as the "__" nesting in an env variable name.
SERVER_NAME_PATTERN = r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$"

COMMAND_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"

# Names the kit reserves for its own commands: "schema" at every level where tools live, "config"
# at the top level. No group may take them, and a tool that would is renamed with a "-tool" suffix.
BUILTIN_COMMANDS = frozenset({"schema", "config"})


def _not_builtin(name: str) -> str:
    if name in BUILTIN_COMMANDS:
        raise ValueError(f"{name!r} is a built-in command name")
    return name


CommandName = Annotated[str, Field(pattern=COMMAND_NAME_PATTERN), AfterValidator(_not_builtin)]


class ConfigError(Exception):
    """The configuration is missing or invalid. The message is meant for the user."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GroupMount(_StrictModel):
    """Mount a server's tools under a subcommand of the bin."""

    group: CommandName
    help: str | None = None


class PrefixMount(_StrictModel):
    """Mount a server's tools at the top level of the bin, with a prefix on every command name."""

    prefix: Annotated[str, Field(min_length=1)]


Mount = Literal["root"] | GroupMount | PrefixMount


class ToolGroup(_StrictModel):
    """Put the tools whose names start with ``prefix`` under the subcommand ``group``.

    The prefix is stripped from the command names inside the group.
    """

    prefix: Annotated[str, Field(min_length=1)]
    group: CommandName
    help: str | None = None


class ServerConfig(_StrictModel):
    """One MCP server and where its tools appear in the bin.

    ``url`` and the ``headers`` values are templates that may refer to environment variables as
    ``${NAME}``; ``expand_server`` resolves them when the server is contacted.
    """

    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: Annotated[float, Field(gt=0)] = 30.0
    # Below the 120 s after which a coding agent's shell tool typically kills a command, so the
    # agent sees a timeout error rather than a killed process.
    sse_read_timeout: Annotated[float, Field(gt=0)] = 100.0
    mount: Mount = "root"
    # Stripped from the start of tool names before anything else; the first matching prefix wins.
    strip_prefixes: list[Annotated[str, Field(min_length=1)]] = Field(default_factory=list)
    tool_groups: list[ToolGroup] = Field(default_factory=list)
    hidden_tools: list[str] = Field(default_factory=list)

    @field_validator("url")
    @classmethod
    def _url_is_http(cls, url: str) -> str:
        parts = parse_template(url)
        # A URL with references is checked once they are expanded, when the server is contacted.
        if not any(isinstance(part, VariableReference) for part in parts):
            check_http_url(_join(parts, {}))
        return url

    @field_validator("headers")
    @classmethod
    def _headers_are_templates(cls, headers: dict[str, str]) -> dict[str, str]:
        for name, value in headers.items():
            try:
                parse_template(value)
            except ValueError as error:
                raise ValueError(f"header {name}: {error}") from None
        return headers

    @model_validator(mode="after")
    def _tool_groups_are_distinct(self) -> ServerConfig:
        _reject_duplicates("strip_prefixes entry", self.strip_prefixes)
        _reject_duplicates("tool_groups prefix", [group.prefix for group in self.tool_groups])
        _reject_duplicates("tool_groups group", [group.group for group in self.tool_groups])
        return self


class CliConfig(_StrictModel):
    """The whole configuration of one bin."""

    help: str | None = None
    servers: Annotated[dict[Annotated[str, Field(pattern=SERVER_NAME_PATTERN)], ServerConfig], Field(min_length=1)]

    @model_validator(mode="after")
    def _mounts_are_distinct(self) -> CliConfig:
        _reject_duplicates("server name", [_env_key(name) for name in self.servers], explain="once - and _ are equated")
        groups = [server.mount.group for server in self.servers.values() if isinstance(server.mount, GroupMount)]
        _reject_duplicates("mount group", groups)
        prefixes = [server.mount.prefix for server in self.servers.values() if isinstance(server.mount, PrefixMount)]
        _reject_duplicates("mount prefix", prefixes)
        # Group mounts and the tool groups of root and prefix mounted servers (behind the prefix)
        # all become top-level subcommands.
        top_level_tool_groups = [
            _top_level_prefix(server.mount) + group.group
            for server in self.servers.values()
            if not isinstance(server.mount, GroupMount)
            for group in server.tool_groups
        ]
        _reject_duplicates(
            "top-level command name", groups + top_level_tool_groups, explain="from group mounts and tool_groups"
        )
        return self


@dataclass(frozen=True)
class VariableReference:
    """A ``${NAME}`` in a config value."""

    name: str


@dataclass(frozen=True)
class ResolvedServer:
    """A server's ``url`` and ``headers`` with every ``${NAME}`` expanded. Holds secrets; never print it."""

    url: str
    headers: dict[str, str]


_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_template(value: str) -> list[str | VariableReference]:
    """Split a config value into literal text and ``${NAME}`` references.

    ``$${`` is a literal ``${``, and a ``$`` that does not start ``${`` is literal text.

    Raises:
        ValueError: A ``${`` has no closing ``}``, or what it encloses is not a variable name
            (letters, digits and ``_``, not starting with a digit). The message never repeats
            the value, which may be a secret.
    """
    parts: list[str | VariableReference] = []
    literal: list[str] = []
    position = 0
    while position < len(value):
        if value.startswith("$${", position):
            literal.append("${")
            position += 3
        elif value.startswith("${", position):
            end = value.find("}", position + 2)
            if end < 0:
                raise ValueError("has a ${ without a closing }; write $${ for a literal ${")
            name = value[position + 2 : end]
            if not _VARIABLE_NAME.fullmatch(name):
                raise ValueError(
                    "has a ${...} that does not enclose a variable name (letters, digits and _); "
                    "write $${ for a literal ${"
                )
            if literal:
                parts.append("".join(literal))
                literal = []
            parts.append(VariableReference(name))
            position = end + 1
        else:
            literal.append(value[position])
            position += 1
    if literal:
        parts.append("".join(literal))
    return parts


def template_variables(value: str) -> list[str]:
    """The names of the variables a config value refers to, in order, without duplicates."""
    names = [part.name for part in parse_template(value) if isinstance(part, VariableReference)]
    return list(dict.fromkeys(names))


def check_http_url(url: str, *, show_scheme: bool = True) -> None:
    """Raise ValueError unless the URL is http(s) with a host.

    The URL may carry a token, so the message never repeats more than its scheme, and not even
    that without ``show_scheme``.
    """
    scheme, separator, rest = url.partition("://")
    if not separator:
        raise ValueError("must be an http:// or https:// URL, got a value without a scheme")
    if scheme.lower() not in ("http", "https"):
        got = f", got the scheme {scheme[:16]!r}" if show_scheme else ""
        raise ValueError(f"must be an http:// or https:// URL{got}")
    if not re.match(r"^[^/\s?#]+", rest):
        raise ValueError("must be an http:// or https:// URL with a host")
    try:
        urlsplit(url).port  # noqa: B018 - reading it validates it
    except ValueError:
        raise ValueError("must be an http:// or https:// URL with a valid host and numeric port") from None


def expand_server(name: str, server: ServerConfig, environ: Mapping[str, str] | None = None) -> ResolvedServer:
    """Expand the ``${NAME}`` references in a server's ``url`` and ``headers``.

    Each expanded value, and the userinfo and query of the expanded URL, are registered with
    ``redaction``, so they are kept out of transport errors and logs from then on.

    Args:
        name: The server's name, for the error message.
        server: The server's configuration, with the references as written.
        environ: The environment to read, ``os.environ`` when None.

    Raises:
        ConfigError: A variable is not set; the expanded URL is not a valid http(s) URL, or a
            variable's value changes which part of the URL the text around it belongs to; or an
            expanded header value holds a control or non-ASCII character. The message names the
            server, the key and the variables, never a value.
    """
    environ = os.environ if environ is None else environ
    url = _expand(name, "url", server.url, environ)
    url_variables = template_variables(server.url)
    _check_expanded_url(name, server.url, url, url_variables, environ)

    headers = {}
    for header, value in server.headers.items():
        expanded = _expand(name, f"header {header}", value, environ)
        if any(ord(character) < 0x20 or ord(character) >= 0x7F for character in expanded):
            variables = template_variables(value)
            source = f" (from {', '.join(variables)})" if variables else ""
            raise ConfigError(f"Server {name}: header {header}{source} holds a control or non-ASCII character.")
        headers[header] = expanded

    _register_secrets(server, url, environ)
    return ResolvedServer(url=url, headers=headers)


def _register_secrets(server: ServerConfig, url: str, environ: Mapping[str, str]) -> None:
    """Register the parts of a server's expanded config that carry secrets with ``redaction``.

    Those are the values of the variables in the headers, the userinfo, the query and the
    fragment, and the userinfo, password, query and fragment of the expanded URL. The fragment is
    never sent, but the SDK and httpx log it. Host, port and path are left alone.
    """
    template = urlsplit(server.url)
    secret_templates = [*server.headers.values()]
    if template.scheme and template.netloc:
        secret_templates += [template.netloc.rpartition("@")[0], template.query, template.fragment]
    for value in secret_templates:
        for variable in template_variables(value):
            redaction.register_encoded(environ[variable], "${" + variable + "}")

    parts = urlsplit(url)
    userinfo = parts.netloc.rpartition("@")[0]
    secrets = [userinfo, userinfo.partition(":")[2], parts.query, parts.fragment]
    parsed = httpx.URL(url)
    secrets += [
        parsed.userinfo.decode("ascii", "replace"),
        parsed.password or "",
        parsed.query.decode("ascii", "replace"),
    ]
    for secret in secrets:
        redaction.register_encoded(secret, "***")


def _check_expanded_url(name: str, template: str, url: str, variables: list[str], environ: Mapping[str, str]) -> None:
    if not variables:
        return  # checked when the config was loaded
    prefix = f"Server {name}: url, once {', '.join(variables)} {'is' if len(variables) == 1 else 'are'} expanded,"
    template_scheme = template.partition("://")[0] if "://" in template else "${"
    try:
        check_http_url(url, show_scheme="${" not in template_scheme)
    except ValueError as error:
        raise ConfigError(f"{prefix} {error}.") from None
    try:
        httpx.URL(url)
        expanded_parts = tuple(urlsplit(url))
        template_parts = urlsplit(template)
    except (httpx.InvalidURL, ValueError):
        raise ConfigError(f"{prefix} is not a valid URL.") from None
    if template_parts.scheme and template_parts.netloc:
        expected = tuple(_join(parse_template(part), environ) for part in template_parts)
        if expanded_parts != expected:
            raise ConfigError(
                f"{prefix} is not the URL the template describes: a value holds a /, ?, # or @ "
                "that moves text into another part of the URL. Percent-encode it."
            )


def expand_url(server: ServerConfig, environ: Mapping[str, str] | None = None) -> str | None:
    """A server's expanded URL, or None when a variable it refers to is not set."""
    environ = os.environ if environ is None else environ
    parts = parse_template(server.url)
    if any(isinstance(part, VariableReference) and part.name not in environ for part in parts):
        return None
    return _join(parts, environ)


def _expand(server: str, key: str, value: str, environ: Mapping[str, str]) -> str:
    try:
        parts = parse_template(value)
    except ValueError as error:  # pragma: no cover - the model already validated the syntax
        raise ConfigError(f"Server {server}: {key} {error}.") from None
    missing = [part.name for part in parts if isinstance(part, VariableReference) and part.name not in environ]
    if missing:
        names = ", ".join(dict.fromkeys(missing))
        one = len(set(missing)) == 1
        raise ConfigError(
            f"Server {server}: {key} uses the environment {'variable' if one else 'variables'} {names}, "
            f"which {'is' if one else 'are'} not set."
        )
    return _join(parts, environ)


def _join(parts: list[str | VariableReference], environ: Mapping[str, str]) -> str:
    return "".join(environ[part.name] if isinstance(part, VariableReference) else part for part in parts)


@dataclass(frozen=True)
class LoadedConfig:
    """The validated configuration together with where it came from.

    Attributes:
        bin_name: The name of the executable, which also names the config and cache directories.
        config: The merged and validated configuration.
        path: The config file that was looked for.
        file_loaded: Whether that file existed and was read.
        env_prefix: The prefix of the environment variables that were applied.
        env_applied: The names of the environment variables that were applied, sorted.
    """

    bin_name: str
    config: CliConfig
    path: Path
    file_loaded: bool
    env_prefix: str
    env_applied: tuple[str, ...] = ()


def default_config_path(bin_name: str) -> Path:
    """Return ``$XDG_CONFIG_HOME/<bin>/config.yaml``, by default ``~/.config/<bin>/config.yaml``."""
    return xdg_directory("XDG_CONFIG_HOME", ".config") / bin_name / CONFIG_FILE_NAME


def xdg_directory(variable: str, fallback: str) -> Path:
    """An XDG base directory: the variable when it holds an absolute path, ``~/<fallback>`` otherwise."""
    value = os.environ.get(variable, "")
    if value and Path(value).is_absolute():
        return Path(value)
    return Path.home() / fallback


def resolve_config_path(bin_name: str, explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the config file to read: the ``--config`` value when given, the default path otherwise."""
    if explicit is None or str(explicit) == "":
        return default_config_path(bin_name)
    return Path(explicit).expanduser()


def env_prefix_for(bin_name: str) -> str:
    """Derive the environment variable prefix from the bin name: ``my-tool`` becomes ``MY_TOOL_``."""
    return _env_key(bin_name) + "_"


def load_config(
    bin_name: str,
    *,
    default: Mapping[str, Any] | CliConfig | None = None,
    config_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    env_prefix: str | None = None,
) -> LoadedConfig:
    """Merge the code default, the config file and the environment, and validate the result.

    Args:
        bin_name: The executable's name.
        default: The consumer's default configuration, in the shape of the file.
        config_path: The ``--config`` value; when given, the file must exist.
        environ: The environment to read, ``os.environ`` when None.
        env_prefix: Overrides the prefix derived from the bin name.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: The file is missing or malformed, or the merged configuration is invalid.
    """
    path = resolve_config_path(bin_name, config_path)
    prefix = env_prefix if env_prefix is not None else env_prefix_for(bin_name)
    environ = os.environ if environ is None else environ

    merged: dict[str, Any] = _as_dict(default) if default is not None else {}

    if path.is_dir():
        raise ConfigError(f"Config path {path} is a directory, not a file.")
    file_loaded = path.is_file()
    if file_loaded:
        merged = _merge_config(merged, _read_file(path))
    elif config_path is not None and str(config_path) != "":
        raise ConfigError(f"Config file {path} does not exist.")
    elif default is None:
        raise ConfigError(
            f"{bin_name} has no configuration. Create {path} with at least one server, for example:\n\n"
            + yaml.safe_dump({"servers": {"example": {"url": "https://mcp.example.com/mcp"}}}, sort_keys=False)
        )

    applied: list[str] = []
    merged = _merge_config(merged, _read_env(environ, prefix, merged, applied))

    try:
        config = CliConfig.model_validate(merged)
    except ValidationError as error:
        sources = ["the built-in default"] if default is not None else []
        if file_loaded:
            sources.append(str(path))
        sources.append(f"{prefix}* environment variables")
        raise ConfigError(f"Invalid configuration (from {', '.join(sources)}):\n{_format_errors(error)}") from error

    return LoadedConfig(
        bin_name=bin_name,
        config=config,
        path=path,
        file_loaded=file_loaded,
        env_prefix=prefix,
        env_applied=tuple(sorted(applied)),
    )


def save_config(path: str | os.PathLike[str], data: Mapping[str, Any] | CliConfig) -> Path:
    """Write a config file that may hold secrets, readable by its owner only.

    The file is written to a temporary file created with mode 0600 and then renamed over the
    target, so the secrets are never readable by others, not even for a moment.

    Args:
        path: The file to write. Missing directories are created with mode 0700. A symlink is
            followed, so the link stays in place and the file it points to is replaced.
        data: The configuration, in the shape of the file.

    Returns:
        The path written, with symlinks resolved.
    """
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    content = yaml.safe_dump(_as_dict(data), sort_keys=False, allow_unicode=True)

    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    os.chmod(target, 0o600)
    return target


def _as_dict(data: Mapping[str, Any] | CliConfig) -> dict[str, Any]:
    if isinstance(data, CliConfig):
        return data.model_dump(mode="json", exclude_unset=True)
    return {str(key): _deep_copy(value) for key, value in data.items()}


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_deep_copy(item) for item in value]
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    return value


def _read_file(path: Path) -> dict[str, Any]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Config file {path} is not valid YAML: {error}") from error
    except OSError as error:
        raise ConfigError(f"Cannot read config file {path}: {error}") from error

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"Config file {path} must hold a mapping at the top level, got {type(content).__name__}.")
    return content


def _merge_config(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two config layers. Servers merge key by key; a server's mount is replaced whole."""
    result = dict(base)
    for key, value in override.items():
        if key == "servers" and isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            servers = dict(result[key])
            for name, server in value.items():
                current = servers.get(name)
                if isinstance(server, Mapping) and isinstance(current, Mapping):
                    servers[name] = _merge_server(current, server)
                else:
                    servers[name] = _deep_copy(server)
            result[key] = servers
        else:
            result[key] = _deep_copy(value)
    return result


def _merge_server(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key != "mount" and isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _deep_copy(value)
    return result


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _deep_copy(value)
    return result


def _env_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


_SERVER_FIELDS = {_env_key(name): name for name in ServerConfig.model_fields}

_JSON_FIELDS = {"headers", "mount", "strip_prefixes", "tool_groups", "hidden_tools"}


def _read_env(
    environ: Mapping[str, str], prefix: str, merged: Mapping[str, Any], applied: list[str] | None = None
) -> dict[str, Any]:
    """Turn the ``<PREFIX>*`` variables into a config layer, matched against the layers below.

    The names of the variables that were used are appended to ``applied``.
    """
    applied = [] if applied is None else applied
    servers_below: Mapping[str, Any] = merged.get("servers") or {}
    if not isinstance(servers_below, Mapping):
        servers_below = {}
    server_names = {_env_key(str(name)): str(name) for name in servers_below}

    layer: dict[str, Any] = {}
    prefix = prefix.upper()
    for variable in sorted(environ):
        if not variable.upper().startswith(prefix):
            continue
        value = environ[variable]
        parts = variable[len(prefix) :].upper().split(ENV_NESTING)

        if parts == ["HELP"]:
            layer["help"] = value
            applied.append(variable)
            continue

        if parts[0] == "SERVERS" and len(parts) >= 3:
            if parts[1] not in server_names:
                logger.debug("Ignoring %s: no configured server is named %s", variable, parts[1].lower())
                continue
            server = server_names[parts[1]]
            field_path = parts[2:]
        else:
            logger.debug("Ignoring %s: it matches no configuration key", variable)
            continue

        server_below = servers_below.get(server)
        server_layer = layer.setdefault("servers", {}).setdefault(server, {})
        _set_server_field(variable, server_layer, field_path, value, server_below)
        applied.append(variable)

    return layer


def _set_server_field(
    variable: str, server_layer: dict[str, Any], field_path: list[str], value: str, server_below: Any
) -> None:
    field = _SERVER_FIELDS.get(field_path[0])
    if field is None:
        known = ", ".join(sorted(_SERVER_FIELDS))
        raise ConfigError(f"Unknown server key {field_path[0]} in {variable}. Known keys: {known}.")

    if len(field_path) == 1:
        server_layer[field] = _parse_json(variable, value) if field in _JSON_FIELDS else value
        return

    if field == "headers" and len(field_path) == 2:
        headers_below = server_below.get("headers") if isinstance(server_below, Mapping) else None
        server_layer.setdefault("headers", {})[_header_name(field_path[1], headers_below)] = value
        return

    if field == "mount" and len(field_path) == 2:
        server_layer["mount"] = {field_path[1].lower(): value}
        return

    raise ConfigError(f"{variable} nests deeper than {field} allows.")


def _header_name(env_name: str, headers_below: Any) -> str:
    if isinstance(headers_below, Mapping):
        for existing in headers_below:
            if _env_key(str(existing)) == env_name:
                return str(existing)
    return "-".join(word.capitalize() for word in env_name.split("_"))


def _parse_json(variable: str, value: str) -> Any:
    stripped = value.strip()
    if not stripped.startswith(("[", "{", '"')):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{variable} is not valid JSON: {error}") from error


def _top_level_prefix(mount: Mount) -> str:
    return mount.prefix if isinstance(mount, PrefixMount) else ""


def _reject_duplicates(what: str, values: Iterable[str], explain: str | None = None) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        suffix = f" ({explain})" if explain else ""
        raise ValueError(f"duplicate {what}{suffix}: {', '.join(duplicates)}")


def _format_errors(error: ValidationError) -> str:
    lines = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "(top level)"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)
