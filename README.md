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
requires-python = ">=3.12"
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

Hints such as `Run 'example-tool --help' for the list.` name the command the user typed. A consumer that runs the group from `build_cli()` itself, instead of through `main()`, gets the program name click detects from `sys.argv[0]` there, unless it passes `prog_name`.

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

### Mounting under a host CLI

The group `build_cli()` returns can also be a subcommand of a larger click CLI, so that `example mcp search-items ...` calls the tools next to the host's own commands. `mcp_to_cli_kit.lazy` has what the host needs, and imports only click:

```python
import click

from mcp_to_cli_kit.lazy import mount, run

DEFAULT_CONFIG = {"servers": {"example": {"url": "https://mcp.example.com/mcp"}}}


@click.group()
def example() -> None:
    """The example CLI."""


@example.command()
def hello() -> None:
    """A command of the host itself."""


def kit_cli() -> click.Group:
    from mcp_to_cli_kit.cli import build_cli

    return build_cli("example-mcp", DEFAULT_CONFIG, distribution="example-tool")


mount(example, "mcp", "Call the tools of the example MCP server.", kit_cli)


def main() -> None:
    run(example, prog_name="example")
```

- `mount(host, name, help, factory)` adds a `LazyKitGroup` to the host. Its help is static, so `example --help`, `example hello` and the completion of `example <TAB>` never call `factory()`. The factory runs once, when `example mcp ...` is used (to run a tool, show the help or complete a tool name), and the group it builds handles the rest. With `build_cli` imported inside the factory, as above, the MCP SDK, httpx, pydantic, jsonschema and PyYAML are not even imported before then, which saves about 0.2 s at every start of the host.
- `run(group, *, prog_name, args=None)` runs a click CLI the way the kit's own `main()` does, which is click's standalone mode except for Ctrl-C: an error prints one `Error:` line and exits with its code, an abort (a declined `confirm(abort=True)`, end of input at a prompt) prints `Aborted!` and exits 1, and Ctrl-C prints `Interrupted.` and exits 130 (click exits 1). A command's return value is ignored; `ctx.exit(code)` sets the exit code. Use it for the host, so that the mounted tools fail the same way as in a standalone kit CLI. It leaves logging alone: the mounted group sets up the kit's logging to stderr (and `-v`) only when it is used, so the host's own commands keep the host's logging.
- `LazyKitGroup(name, help, factory, *, short_help=None, hidden=False)` is the group `mount()` adds, for a host that registers its commands another way. `build_cli()`'s group also mounts directly with `host.add_command(build_cli(...), name="mcp")`, but then the kit is imported and built at every start.
- The kit's global options belong to the mounted group: `example mcp --config other.yaml search-items`, not `example --config other.yaml mcp search-items`.
- Hints and usage lines name the path the user typed: `Run 'example mcp --help' for the list.`, `Run 'example mcp auth login' in a terminal`.
- The bin name passed to `build_cli()` still names the config, cache and token directories, the environment variable prefix (`EXAMPLE_MCP_`), the `X-Mcp-Client` header and the MCP `clientInfo`, so the mounted group shares them with a standalone `example-mcp` script built from the same arguments.

### Global options and commands of the consumer's own

`extra_params` adds click options to the bin's global options, next to `--config`, `-v` and `--json`. Make such an option eager (`is_eager=True`), so its value is in the group's `ctx.params` before the tools are listed or a tool is resolved. An option whose name or option string the kit already uses is a `ValueError` when the CLI is built.

```python
import click

ORG = click.Option(["-o", "--org", "org"], envvar="EXAMPLE_TOOL_ORG", is_eager=True, help="The account to use.")


def main() -> None:
    kit_main("example-tool", DEFAULT_CONFIG, distribution="example-tool", extra_params=[ORG])
```

`extra_commands` adds click commands (or groups) to the top level, next to the tools. They are listed under "Commands" with `schema` and `config`, and run without loading the configuration or any tool list, so they work offline and with a broken config. They read the global options from their parent context (`ctx.parent.params["org"]`, `ctx.parent.params["config"]`); `mcp_to_cli_kit.config.resolve_config_path` finds the config file the kit reads, `read_config_file` reads it as written (an empty mapping when there is none, a `ConfigError` when it is broken), and `save_config` writes it back, readable by its owner only. A command should import its heavy dependencies inside its callback, so it starts fast. `mcp_to_cli_kit.output.echo_json(ctx, value)` prints JSON the way the built-in commands do: compact under an AI agent or with `--json`, indented otherwise or with `--pretty`. `mode_options()` gives a command its own `--json` and `--pretty`, next to the bin's global ones.

```python
@click.command("whoami", help="Print the selected account.")
@click.pass_context
def whoami(ctx: click.Context) -> None:
    click.echo(ctx.parent.params["org"])


def main() -> None:
    kit_main("example-tool", DEFAULT_CONFIG, distribution="example-tool", extra_params=[ORG], extra_commands=[whoami])
```

The names of extra commands are reserved like `schema` and `config`: a tool whose command would take one gets a `-tool` suffix, and a group mount or top-level tool group with such a name is a config error. An extra command named like a built-in command is a `ValueError` when the CLI is built.

### Settings of the consumer's own in the config file

A consumer may keep settings of its own in the bin's config file, such as several accounts, and turn them into kit configuration on each run. `consumer_keys` names those top-level keys, and `config_layer` computes a config layer from them:

```yaml
default_org: work
orgs:
  work: {api_key: key-1, org_id: "1111"}
  personal: {api_key: key-2}
servers:
  example:
    timeout: 60
```

```python
from collections.abc import Mapping
from typing import Any

import click
from mcp_to_cli_kit.config import ConfigError


def account_headers(ctx: click.Context, consumer: Mapping[str, Any]) -> Mapping[str, Any]:
    orgs = consumer.get("orgs") or {}
    name = ctx.params.get("org") or consumer.get("default_org")
    if name is None:
        return {}
    if name not in orgs:
        raise ConfigError(f"No account {name!r}. Accounts: {', '.join(sorted(orgs))}.")
    headers = {"Authorization": f"Bearer {orgs[name]['api_key']}"}
    if "org_id" in orgs[name]:
        headers["X-Org-Id"] = orgs[name]["org_id"]
    return {"servers": {"example": {"headers": headers}}}


def main() -> None:
    kit_main(
        "example-tool",
        DEFAULT_CONFIG,
        distribution="example-tool",
        extra_params=[ORG],
        consumer_keys={"orgs", "default_org"},
        config_layer=account_headers,
    )
```

- The consumer keys are taken out of the file before the kit validates it, so they do not fail as unknown keys. They are read from the file only, never from the code default or the environment.
- `config_layer` is called once per invocation, when a command first needs the configuration, with the bin's group context and the consumer keys the file holds (an empty mapping without a file). The context's `params` hold the global options, `extra_params` included.
- Its result has the shape of the file. It is merged over the code default and under the file's own keys and the environment variables, so `servers.example.headers` in the file or `EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION` still win. A header it leaves out is not sent.
- Raising `ConfigError` ends the command with exit code 2 and its message.
- `config` shows the merged result, with its header values masked as usual, and lists the consumer keys the file holds under `consumer_keys`, by name only, never their values.
- `help` and `servers` cannot be consumer keys.

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

The default from code is overridden by `~/.config/<bin>/config.yaml` (or the file given by `--config` or the `<PREFIX>CONFIG` environment variable, such as `EXAMPLE_TOOL_CONFIG`; the option wins), and that by environment variables with the bin's prefix, such as `EXAMPLE_TOOL_SERVERS__EXAMPLE__URL` or `EXAMPLE_TOOL_SERVERS__EXAMPLE__HEADERS__AUTHORIZATION`. Without a default and without a file, the CLI says which file it expected. See `mcp_to_cli_kit/config.py` for the full rules.

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
- Only `url`, the header values, `auth.client_id` and `auth.client_secret` are expanded. Names (servers, mounts, tool groups, tools) and help texts are taken as written.
- A URL with references is checked once it is expanded: it must be a valid http(s) URL, and no value may move text into another part of the URL (a `/`, `?`, `#` or `@` in a password, for example), so percent-encode such characters in the variable. An expanded header value must not hold control or non-ASCII characters. Each of these is a config error that names the variable, never its value.
- Error messages show the URL as configured, with `${NAME}` unexpanded and without its userinfo and query. The parts that carry secrets (the values of the variables in the headers, and the userinfo, password, query and fragment of the expanded URL, also in their percent-encoded forms, if they are 4 characters or longer) are also replaced in transport errors and in the `-v` / `-vv` log. So is the literal text of every header value that `config` masks (all headers but `Accept`, `Content-Type`, `User-Agent` and `X-Mcp-Client`, without a leading `Bearer` or other auth scheme word), for a token written into the header itself. Host, port and path values are not secrets and are shown.
- The tool cache remembers a hash of the URL as configured and one of the expanded URL. Pointing a variable at another server fetches that server's tools; with the variable unset, the cache of the configured URL is used.

### Signing in with OAuth

A server that follows the [MCP authorization spec](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) gets an `auth` key instead of an `Authorization` header. The kit needs a pre-registered OAuth client: it does not register one dynamically.

```yaml
servers:
  example:
    url: https://mcp.example.com/mcp
    auth:
      type: oauth
      client_id: 1234-abcd.apps.googleusercontent.com
      client_secret: GOCSPX-...    # optional; ${NAME} references work here and in client_id
      scopes: [openid, email]      # optional; default: the scopes the server announces
      bearer: access_token         # or id_token, for a proxy that wants an OpenID Connect ID token
      authorization_server: null   # optional: the issuer, for a server without protected resource metadata
      redirect_port: 0             # optional: a fixed loopback port, for a client registered with an exact redirect URI
      redirect_uri: null           # optional: the registered redirect URI, for a client without a loopback redirect
      resource: true               # optional: false leaves out the RFC 8707 resource parameter
```

```console
$ example-tool auth login     # opens the browser, then stores the tokens
$ example-tool auth login --paste   # the same, but you paste the URL the browser was sent to
$ example-tool auth status    # who is signed in, since when and until when, as one line of JSON; --full adds every token's expiry, indented
$ example-tool auth logout    # deletes the stored tokens; --revoke also revokes the sign-in
```

- Only `auth login` signs in. It finds the authorization server from the server's protected resource metadata, opens the browser with PKCE (S256) and a redirect to `http://127.0.0.1:<port>/callback` (or to a loopback `redirect_uri`), and waits up to 5 minutes (`--timeout`). It also prints the sign-in address; with `--no-browser` it only prints it. Without a server name it signs in to every server that uses OAuth.
- `redirect_uri` is the redirect URI the client is registered with. It is an `https://` URL with any host, or an `http://` URL of `127.0.0.1`, `[::1]` or `localhost` with an explicit port. A query is allowed, a fragment is not. It is not a `${NAME}` template, and it needs `redirect_port: 0`. With a loopback `redirect_uri`, `auth login` listens on its port and path. `[::1]` listens on the IPv6 loopback; `127.0.0.1` and `localhost` both listen on `127.0.0.1`, so a `localhost` redirect works only in a browser that resolves `localhost` to `127.0.0.1`.
- `auth login --paste` signs in without a listener. Use it when the client's only registered redirect URIs are another application's `https://` callbacks, or when the browser runs on another machine. After you sign in, copy the whole URL from the browser's address bar, or only the value of its `code` parameter, and paste it at the prompt. The CLI does not echo what you paste. A pasted URL must have the scheme, host, port and path of the redirect URI, and its `state` must belong to this sign-in. A bare code skips the state check, and the CLI warns about that; the code still works only with this sign-in's PKCE verifier. Codes are short-lived and work once, so paste the URL at once. Without `redirect_uri`, `--paste` uses `http://127.0.0.1:<redirect_port>/callback` and needs a non-zero `redirect_port`; the browser then shows a connection error, and the URL in its address bar is the one to paste. Plain `auth login` refuses an `https://` `redirect_uri`, and for such a server a command that is not signed in names `auth login --paste`.
- With another application's redirect URI, the authorization server sends the browser, with the code and the state, to that application. Its server sees the code, in its request log for example. It cannot redeem the code without the PKCE verifier, which never leaves the CLI, but the code is still a credential that reaches a party you did not choose. That page may also redirect at once, or show an error because you are signed in to that application. Then find the callback URL in the browser history, or turn on "Preserve log" in the Network tab of the browser's developer tools before you sign in, and copy the URL of the callback request.
- Every other command reads the stored tokens and refreshes them when less than 5 minutes remain. It never opens a browser: when there is no sign-in, or the authorization server ended it, the command exits with 4 and names the `auth login` command to run. An AI agent gets that answer at once and can ask the user to sign in.
- `auth status` prints, per server, `signed_in`, `identity`, `signed_in_at` (the browser sign-in) and `signed_in_until`: when the sign-in ends unless the user signs in again. With a refresh token that is the refresh token's expiry, which the kit knows only when the token response states it in the non-standard `refresh_token_expires_in` (Google) or `refresh_expires_in` (Keycloak); otherwise it is `null`, and the sign-in lasts until the authorization server ends it. Without a refresh token it is the bearer's expiry. `--full` adds `bearer`, `access_token_expires_at`, `id_token_expires_at`, `has_refresh_token`, `refresh_token_expires_at`, `obtained_at` (the last token response), `scopes`, `client_id`, `issuer`, `resource` and `token_file`. The short status is one line of JSON and the full one is indented, for a person and an AI agent alike; `--json` or `--pretty` overrides that. It never prints a token or the client secret, sends nothing to any server, and exits 4 when a server is not signed in.
- When the server answers 401 to a token the CLI considers valid, the CLI refreshes the token once and sends the request again. The request never reached the server, so this is safe for a tool call too.
- The tokens are stored in `$XDG_STATE_HOME/<bin>/oauth/<server>.json` (`~/.local/state/...` by default), readable by the owner only. The file records the server URL and the client it was issued for; after either changes in the config, the CLI counts as signed out and never sends the old token. Parallel commands refresh under a file lock, so one refresh serves them all.
- The client secret is masked in `config`, and the secret and every token are kept out of error messages and the `-v` / `-vv` log. Give the secret's environment variable a name outside the CLI's own (`EXAMPLE_TOOL_CONFIG`, `EXAMPLE_TOOL_HELP` and `EXAMPLE_TOOL_SERVERS__...`), for example `client_secret: ${EXAMPLE_MCP_CLIENT_SECRET}`.
- A Google "Desktop app" client is a public client: Google expects its secret to ship with the app, so a consumer can put it in its code default. With Google, `logout --revoke` revokes the sign-in of every app that uses the same client.

`auth` is a reserved command name, like `schema` and `config`. It shows in the help only when a server uses OAuth.

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

`example-tool config` prints the effective configuration as JSON: the code default, the config file and the environment variables merged, with every default filled in. Next to it, it shows the config file that was looked for (`config_file`), whether it was loaded (`config_file_loaded`), the environment variable prefix (`env_prefix`) and the names of the environment variables that were applied (`env_applied`), never their values, and for a bin with consumer keys, the names of those the file holds (`consumer_keys`).

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
  - a URL's whole userinfo (user and password), its query values and its fragment;
  - `auth.client_secret`.
- Shown as written: the URL's scheme, host, port and path, every other setting (timeouts, mounts, tool groups, hidden tools, prefixes, help texts), and all `${NAME}` references.
- The output follows the usual modes: compact under an AI agent or with `--json`, indented otherwise or with `--pretty`.

`config` and `auth` are reserved command names, like `schema`: a tool whose command would be one of them is renamed with a `-tool` suffix, and no group mount or tool group may take them.

### Output and exit codes

Under an AI coding agent (detected from its environment variables), or with `--json`, the result is printed exactly as the server sent it. Otherwise, or with `--pretty`, JSON is indented with `indent=2` and non-ASCII text stays readable. Several content items become one JSON array. Nothing is ever wrapped to the terminal width.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | The tool reported an error; its text is on stderr |
| 2 | Usage or configuration error |
| 3 | The server could not be reached or did not speak MCP |
| 4 | The server needs a sign-in that is missing, expired or refused; run `<bin> auth login` |

Other global options: `--refresh-tools` fetches the tool list again, `-v` / `-vv` log to stderr, `--version`. Every request carries an `X-Mcp-Client: <bin>/<version> (mcp-to-cli-kit <version>)` header.

When the server cannot be reached, an expired tool cache is used with a warning. `initialize` and `tools/list` are retried twice on connect errors, timeouts and HTTP 502/503/504; a tool call is retried only when it never reached the server. Ctrl-C abandons the request in flight and exits with 130.

Commands that need no server, `--help` and a tool's `--help` from a fresh cache, `schema`, `config` and a consumer's own commands, do not import the MCP SDK, httpx or anyio, so they start several times faster than a command that contacts a server.

Config, cache and OAuth tokens live under `$XDG_CONFIG_HOME`, `$XDG_CACHE_HOME` and `$XDG_STATE_HOME` when those are set, and under `~/.config`, `~/.cache` and `~/.local/state` otherwise.

## Troubleshooting

- **A call ends with "timed out after 100s".** The server took longer than `sse_read_timeout`, which defaults to 100 seconds so that a coding agent's shell tool (often limited to 120 seconds) sees this error instead of killing the command. Raise `sse_read_timeout` for that server when its tools are legitimately slow. Timeouts exit with 3, never with 1, because the tool's own outcome is unknown.
- **A call hangs until that timeout although the server already failed.** When a server or proxy cuts a streamed (SSE) answer in the middle, the MCP SDK swallows the broken stream and offers no hook to notice it, so the request ends only with the read timeout. A plain JSON answer cut the same way fails at once.
- **A command exits with 4 although `auth login` succeeded.** The server refused the token even after a refresh. If the server sits behind a proxy that wants another kind of token, switch `auth.bearer` between `access_token` and `id_token`; the stored sign-in holds both, so no new sign-in is needed. Otherwise the account may lack access to the server.
- **Something else.** Run with `-v` to see what the CLI does, or `-vv` to also see the HTTP traffic and the MCP SDK's own log.

## Developing

Install the dependencies, then run the same checks as CI:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest
```

CI runs the tests on Python 3.12 to 3.15.

### Releasing

The version comes from git tags through hatch-vcs; `pyproject.toml` holds none. A release is a `vX.Y.Z` tag on a `master` commit whose CI run is green:

```bash
git tag v0.5.0
git push origin v0.5.0
```

The tag runs `.github/workflows/publish.yml`, which builds the sdist and the wheel with `uv build` and uploads them to PyPI through trusted publishing (the `pypi` environment, no token). Check that PyPI serves the new version:

```bash
curl -s https://pypi.org/pypi/mcp-to-cli-kit/json | jq -r .info.version
```

Follow semantic versioning: a new feature that keeps every existing consumer working is a minor release, a fix is a patch release. Consumers pin a minor range such as `mcp-to-cli-kit>=0.5.0,<0.6`.

### Commit hooks

Configs, test fixtures and captured output can carry private identifiers such as internal server URLs, account names or paths. `scripts/hooks` holds a pre-commit and a commit-msg hook that refuse a commit when a staged line or the commit message matches a private identifier. The identifiers live in a file outside the repository, because listing them here would publish the very strings the hooks exist to keep out. Set it up once per clone:

```bash
git config core.hooksPath scripts/hooks
git config hooks.forbiddenPatternsFile ~/.config/git/forbidden-patterns.txt  # any path outside the repo
```

The file holds one extended regex per line, matched case-insensitively; blank lines and `#` comments are ignored. Until `hooks.forbiddenPatternsFile` points at a file with at least one pattern, both hooks refuse every commit and print how to fix it. The hooks check only the lines a commit adds.

## License

MIT
