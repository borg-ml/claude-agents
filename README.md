# claude-agents

`claude-agents` is a native Rust runtime for Claude Code agent sessions. It
speaks Claude Code's `stream-json` protocol directly, including:

- streaming assistant, reasoning, tool-call, and tool-result events;
- permission and MCP elicitation control requests;
- session initialization and resume identifiers;
- live context-usage telemetry;
- steering and interruption controls; and
- safe pooled-process reuse across local turns.

It contains no Node.js or TypeScript runtime. Applications provide the Claude
binary command configuration and consume the typed event/control stream.

## Status

This crate is extracted from Borg's native Claude provider and is being
stabilized as an independent public API. The wire implementation is MIT
licensed; Claude Code itself remains Anthropic software and is not included.

## License

MIT. See [LICENSE](LICENSE).
