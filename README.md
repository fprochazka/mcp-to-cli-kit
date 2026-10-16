# mcp-to-cli-kit

A small library for turning one or more remote MCP servers into a typed command-line tool. Each MCP tool becomes a subcommand with real options compiled from its JSON input schema, so a shell user or an AI coding agent can call it without hand-writing JSON-RPC.

The kit owns the parts every such CLI needs and tends to get wrong when it is copied from project to project: the mapping from JSON schema to options, the configuration loading, the schema cache, and output that stays valid JSON when piped. A consumer package only supplies its bin name and a default configuration.

The kit is under development and not published yet.

## License

MIT
