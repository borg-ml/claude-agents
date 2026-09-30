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

## Upstream compatibility

Protocol reviewed against upstream HEAD on 2026-09-30:

- [Python SDK 0.2.163](https://github.com/anthropics/claude-agent-sdk-python/commit/1ef6d8c71bb0e44a6b33fe61497864f21e17fdb7)
- [TypeScript SDK 0.3.286](https://github.com/anthropics/claude-agent-sdk-typescript/commit/0e21ebb22c36caca1220c34ae6c9fe8ddad40e47)
- Both target Claude Code **2.1.286**. The binary is supplied by the caller,
  not installed or upgraded by this crate.

The runtime requests SDK host session-state events and keeps the control
channel open until Claude reports idle, including follow-up turns woken by
background agents. Older CLIs without these events retain result/task-based
completion. `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS` bounds waits between turns
(default 600000 ms; `0` disables the limit). Expiry fails the turn and discards
its process rather than pooling a potentially active session. Caller environment
overrides/removals are respected. Priority `now` steering supports both the
separate-turn behavior; 2.1.286 joins the running turn only for human-origin
messages, which SDK steers do not claim.

This is protocol compatibility for the runtime's supported features, not full
Python/TypeScript SDK API parity. CLI options remain caller-owned; SDK hooks,
in-process SDK MCP servers, session-store helpers and prewarming are not
implemented. Unknown inbound control requests receive an explicit error.

## License

MIT. See [LICENSE](LICENSE).
