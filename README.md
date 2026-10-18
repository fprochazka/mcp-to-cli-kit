# mcp-to-cli-kit

A small library for turning one or more remote MCP servers into a typed command-line tool. Each MCP tool becomes a subcommand with real options compiled from its JSON input schema, so a shell user or an AI coding agent can call it without hand-writing JSON-RPC.

The kit owns the parts every such CLI needs and tends to get wrong when it is copied from project to project: the mapping from JSON schema to options, the configuration loading, the schema cache, and output that stays valid JSON when piped. A consumer package only supplies its bin name and a default configuration.

## Installation

Add the kit to the consumer package:

```bash
uv add mcp-to-cli-kit
```

## Building a consumer package

A consumer is a small Python package with one console script. It names the bin, gives the default configuration, and hands both to the kit.

`pyproject.toml`:

```toml
[project]
name = "example-tool"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["mcp-to-cli-kit"]

[project.scripts]
example-tool = "example_tool.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/example_tool"]
```

`src/example_tool/cli.py`:

```python
from typing import Any

from mcp_to_cli_kit.cli import main as kit_main

DEFAULT_CONFIG: dict[str, Any] = {
    "help": "Call the tools of the example MCP server.",
    "servers": {"example": {"url": "https://mcp.example.com/mcp"}},
}


def main() -> None:
    kit_main("example-tool", DEFAULT_CONFIG, distribution="example-tool")
```

`DEFAULT_CONFIG` has the same shape as the config file (see [Configuration](#configuration)), so a user's `~/.config/example-tool/config.yaml` can override any key of it.

`distribution` is the name of the consumer's own package. The kit reads its version from the package metadata and shows it in `--version`, in the `X-Mcp-Client: example-tool/<version> (mcp-to-cli-kit <version>)` header of every request, and in the `clientInfo` of the MCP `initialize` request. Without it, the kit looks for the package that installed the `example-tool` script, and falls back to `0.1.0`.

### Results reported as text

Some servers report a failure as an ordinary result instead of setting `isError`. A `result_classifier` turns such a result into an error: it is printed to stderr and the exit code is 1. The kit calls it only for results the server did not already mark as errors.

```python
from mcp import types
from mcp_to_cli_kit.cli import main as kit_main


def is_error(server: str, tool: str, result: types.CallToolResult) -> bool:
    first = next((item.text for item in result.content if isinstance(item, types.TextContent)), "")
    return first.startswith("Error: HTTP ")


def main() -> None:
    kit_main("example-tool", DEFAULT_CONFIG, distribution="example-tool", result_classifier=is_error)
```

### Developing the kit and a consumer together

To change the kit and a consumer at the same time, point the consumer at a local checkout of the kit as an editable path dependency, then install the consumer as a uv tool:

```toml
[tool.uv.sources]
mcp-to-cli-kit = { path = "../mcp-to-cli-kit", editable = true }
```

```console
$ uv tool install -e .
$ example-tool --version
example-tool 0.1.0 (mcp-to-cli-kit 0.1.dev7+g1d1eb86b6)
```

Because both installs are editable, a change in the kit or in the consumer takes effect without reinstalling. The kit takes its version from git tags, so a checkout between releases reports a development version such as `0.1.dev7+g1d1eb86b6`; the editable install records it when it is installed, so run `uv tool install -e .` again to refresh it.

## Usage

Every tool of the server is then a subcommand, named in kebab case (`search_items` becomes `search-items`; the tool's own name works too):

```console
$ example-tool --help                      # the tools, from a one-hour cache
$ example-tool search-items --help         # a tool's full description and options
$ example-tool schema search-items         # the tool's raw input schema
$ example-tool search-items --query "red" --limit 5 --tags a --tags b --no-verbose
```

### Configuration

The default from code is overridden by `~/.config/<bin>/config.yaml` (or the file given by `--config`), and that by environment variables with the bin's prefix, such as `EXAMPLE_TOOL_SERVERS__EXAMPLE__URL` or `EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION`. Without a default and without a file, the CLI says which file it expected. See `mcp_to_cli_kit/config.py` for the full rules.

```yaml
servers:
  example:
    url: https://mcp.example.com/mcp
    headers:
      Authorization: Bearer ...
    timeout: 30            # seconds to connect and to send a request
    sse_read_timeout: 100  # seconds to wait for an answer, streamed or not
```

### Environment variables in values

The `url` and the `headers` values may refer to environment variables as `${NAME}`, so a token can stay out of the config file and the code default:

```yaml
servers:
  example:
    url: https://${EXAMPLE_HOST}/mcp
    headers:
      Authorization: "Bearer ${EXAMPLE_TOKEN}"
```

- References are expanded when the CLI contacts the server, never when the config is loaded. An unset variable therefore fails only a command that needs that server, with a config error (exit code 2) that names the variable and the server, and never shows a value. An empty variable counts as set.
- `--help`, a tool's `--help` and `schema` still work offline from a fresh tool cache when the variable is unset. With an expired cache the CLI warns and uses it, as when the server is down.
- A template can come from any layer: the code default, the config file, or an environment variable such as `EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION='Bearer ${EXAMPLE_TOKEN}'`.
- `$${` is a literal `${`. A `$` not followed by `{` stays as it is. A `${` without a closing `}`, or one that does not enclose a variable name (letters, digits and `_`), is a config error when the config is loaded.
- Only `url` and the header values are expanded. Names (servers, mounts, tool groups, tools) and help texts are taken as written.
- A URL with references is checked once it is expanded: it must be a valid http(s) URL, and no value may move text into another part of the URL (a `/`, `?`, `#` or `@` in a password, for example), so percent-encode such characters in the variable. An expanded header value must not hold control or non-ASCII characters. Each of these is a config error that names the variable, never its value.
- Error messages show the URL as configured, with `${NAME}` unexpanded and without its userinfo and query. The parts that carry secrets (the values of the variables in the headers, and the userinfo, password, query and fragment of the expanded URL, also in their percent-encoded forms, if they are 4 characters or longer) are also replaced in transport errors and in the `-v` / `-vv` log. Host, port and path values are not secrets and are shown.
- The tool cache remembers a hash of the URL as configured and one of the expanded URL. Pointing a variable at another server fetches that server's tools; with the variable unset, the cache of the configured URL is used.

### Several servers in one bin

A bin can serve several MCP servers. Each server says where its tools appear with `mount`:

```yaml
servers:
  alpha:
    url: https://alpha.example.com/mcp      # mount: root, the default: tools at the top level
  beta:
    url: https://beta.example.com/mcp
    mount: {group: beta, help: The beta tools.}   # example-tool beta search-items ...
  gamma:
    url: https://gamma.example.com/mcp
    mount: {prefix: "g-"}                    # example-tool g-search-items ...
```

Each server keeps its own session, cache file (`<cache>/<bin>/<server>.json`), headers and retries, and only the servers a command needs are contacted: `example-tool beta search-items` never talks to alpha. The top-level `--help` lists the groups without asking their servers, and lists the root and prefixed tools from the caches; a server that cannot be listed is shown under "Unavailable servers" while the others are still listed.

Two servers must not produce the same command name. When the tool lists show that they do (two root servers with a tool of the same name, or a group named like a top-level tool), the CLI stops with a config error that names both servers; give one of them a group or prefix mount, or hide the tool with `hidden_tools`. Calling a group checks the other servers only from what is already known (their caches), so a clash with a server whose tools were never fetched surfaces in the top-level `--help`. When a command resolves on one server while another server that could also define it cannot be reached, the CLI runs it and warns that the other server could not be checked.

### Stripping prefixes from tool names

Some servers put their own name in front of every tool (`vendor-x_list`, `vendor-x_get`). `strip_prefixes` removes it from the command names:

```yaml
servers:
  example:
    url: https://mcp.example.com/mcp
    strip_prefixes: [vendor-x_, vendor_x-]   # vendor-x_list -> example-tool list
```

- The first prefix in the list that a tool name starts with is stripped. A name that equals the prefix is kept.
- It works in every mount mode and happens before tool groups and mounts: `tool_groups` prefixes match the stripped names, and a group or prefix mount applies to the stripped command (`prefix: "v-"` gives `example-tool v-list`).
- The tool's own name stays an alias (`example-tool vendor-x_list`), and `hidden_tools` still lists tools by their own names.
- When a stripped name would clash with another tool or a tool group of the same server (`vendor-x_list` next to a plain `list`), that tool keeps its full name and the CLI warns on stderr, so neither tool hides the other; hide one of them with `hidden_tools` to silence it. A clash with another server's tool is a config error, as for any other command name.

### Tool groups and hidden tools

A server with many tools can split them into subcommands by tool-name prefix:

```yaml
servers:
  example:
    url: https://mcp.example.com/mcp
    tool_groups:
      - {prefix: svc_a_, group: svc-a, help: Service A.}   # svc_a_list -> example-tool svc-a list
      - {prefix: svc_b_, group: svc-b}                     # svc_b_search -> example-tool svc-b search
    hidden_tools: [svc_a_delete]
```

- The prefix is stripped from the command name inside the group. The tool's own name still works as an alias, inside the group (`svc-a svc_a_list`) and at the server's mount level (`example-tool svc_a_list`). The longest matching prefix wins.
- Tools that match no prefix stay at the server's mount level.
- Tool groups sit where the server is mounted: at the top level for a root mount, under the group for a group mount (`example-tool ex svc-a list`), and behind the prefix for a prefix mount, because the prefix applies to everything the server puts at the top level (`prefix: "b-"` gives `example-tool b-svc-a list`).
- A tool group name must not clash with a tool command of the same server, with a group mount, or with another server's tool or tool group; each clash is a config error.
- `hidden_tools` lists tools by their own names. A hidden tool is not listed and cannot be called, inside a tool group as well; a tool group stays listed even when all its tools are hidden.

### Options

Each property of a tool's input schema becomes an option:

- The option is the kebab-case name (`--start-time`). The schema's own spelling (`--startTime`) and the snake case (`--start_time`) are accepted too, and the schema's spelling is what is sent.
- An option left out is not sent; the server applies its own default, which the help shows.
- Booleans are `--flag` / `--no-flag`, so `false` can be sent.
- Enums are fixed choices, listed in the help.
- Arrays are repeated options, one item each, or one JSON array. Values are never split on commas.
- Objects take a JSON literal; invalid JSON is an error, not a string.
- `--input-json JSON|@FILE|-` passes all arguments as one JSON object; `--arg-file NAME=PATH` reads one value from a file, or from stdin with `-`.
- Arguments are validated against the schema before the call. A usage error lists the tool's parameters, so no `--help` round trip is needed.

A property named like a built-in option (`help`, `input-json`, `arg-file`, `json`, `pretty`) gets an `--arg-` prefix.

### The schema command

`schema TOOL` prints a tool's raw JSON input schema, as the server declared it in `tools/list`. It exists at every level where tools live (`example-tool schema search-items`, `example-tool beta schema search-items`, `example-tool svc-a schema list`) and accepts every name the tool's own command accepts. It reads the cached tool list, so with a warm cache nothing is sent to the server. The schema is printed as compact JSON in machine mode and indented otherwise.

`schema` is a built-in command name: a tool whose command would be `schema` is renamed to `schema-tool`, and no group mount or tool group may be called `schema`.

### The config command

`example-tool config` prints the effective configuration as JSON: the code default, the config file and the environment variables merged, with every default filled in. Next to it, it shows the config file that was looked for (`config_file`), whether it was loaded (`config_file_loaded`), the environment variable prefix (`env_prefix`) and the names of the environment variables that were applied (`env_applied`), never their values.

```console
$ example-tool config --pretty
{
  "config_file": "/home/me/.config/example-tool/config.yaml",
  "config_file_loaded": true,
  "env_prefix": "EXAMPLE_TOOL_",
  "env_applied": ["EXAMPLE_TOOL_SERVERS__EXAMPLE__TIMEOUT"],
  "config": {"help": null, "servers": {"example": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ${EXAMPLE_TOKEN}"}, ...}}}
}
```

- It reads the configuration only. It loads no tool list, sends nothing to any server and needs no cache, so it works offline and with unset variables.
- `${NAME}` references are shown as written, never expanded.
- Literal text that may be a secret is masked as `***` plus its last 4 characters, or only `***` when it is shorter than 12 characters. These are masked:
  - every header value, except `Accept`, `Content-Type`, `User-Agent` and `X-Mcp-Client` (matched in any case), which are shown as written. A leading `Bearer`, `Basic`, `Token`, `Digest`, `Bot` or `ApiKey` followed by more text stays readable, as in `Bearer ***abcd`;
  - a URL's whole userinfo (user and password), its query values and its fragment.
- Shown as written: the URL's scheme, host, port and path, every other setting (timeouts, mounts, tool groups, hidden tools, prefixes, help texts), and all `${NAME}` references.
- The output follows the usual modes: compact under an AI agent or with `--json`, indented otherwise or with `--pretty`.

`config` is a reserved command name, like `schema`: a tool whose command would be `config` is renamed to `config-tool`, and no group mount or tool group may be called `config`.

### Output and exit codes

Under an AI coding agent (detected from its environment variables), or with `--json`, the result is printed exactly as the server sent it. Otherwise, or with `--pretty`, JSON is indented with `indent=2` and non-ASCII text stays readable. Several content items become one JSON array. Nothing is ever wrapped to the terminal width.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | The tool reported an error; its text is on stderr |
| 2 | Usage or configuration error |
| 3 | The server could not be reached or did not speak MCP |

Other global options: `--refresh-tools` fetches the tool list again, `-v` / `-vv` log to stderr, `--version`. Every request carries an `X-Mcp-Client: <bin>/<version> (mcp-to-cli-kit <version>)` header.

When the server cannot be reached, an expired tool cache is used with a warning. `initialize` and `tools/list` are retried twice on connect errors, timeouts and HTTP 502/503/504; a tool call is retried only when it never reached the server. Ctrl-C abandons the request in flight and exits with 130.

Config and cache live under `$XDG_CONFIG_HOME` and `$XDG_CACHE_HOME` when those are set, and under `~/.config` and `~/.cache` otherwise.

## Troubleshooting

- **A call ends with "timed out after 100s".** The server took longer than `sse_read_timeout`, which defaults to 100 seconds so that a coding agent's shell tool (often limited to 120 seconds) sees this error instead of killing the command. Raise `sse_read_timeout` for that server when its tools are legitimately slow. Timeouts exit with 3, never with 1, because the tool's own outcome is unknown.
- **A call hangs until that timeout although the server already failed.** When a server or proxy cuts a streamed (SSE) answer in the middle, the MCP SDK swallows the broken stream and offers no hook to notice it, so the request ends only with the read timeout. A plain JSON answer cut the same way fails at once.
- **Something else.** Run with `-v` to see what the CLI does, or `-vv` to also see the HTTP traffic and the MCP SDK's own log.

## License

MIT
