# mcp-to-cli-kit

A small library for turning one or more remote MCP servers into a typed command-line tool. Each MCP tool becomes a subcommand with real options compiled from its JSON input schema, so a shell user or an AI coding agent can call it without hand-writing JSON-RPC.

The kit owns the parts every such CLI needs and tends to get wrong when it is copied from project to project: the mapping from JSON schema to options, the configuration loading, the schema cache, and output that stays valid JSON when piped. A consumer package only supplies its bin name and a default configuration.

The kit is under development and not published yet.

## Usage

A consumer package is a console script that hands the kit its bin name and a default configuration:

```python
from mcp_to_cli_kit.cli import main as kit_main

DEFAULT = {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}}


def main() -> None:
    kit_main("example-tool", DEFAULT)
```

```toml
[project.scripts]
example-tool = "example_tool:main"
```

Every tool of the server is then a subcommand, named in kebab case (`search_items` becomes `search-items`; the tool's own name works too):

```console
$ example-tool --help                      # the tools, from a one-hour cache
$ example-tool search-items --help         # a tool's full description and options
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
