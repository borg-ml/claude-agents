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

Protocol source-reviewed against upstream HEAD on 2026-10-20:

- [Python SDK 0.2.168 (HEAD)](https://github.com/anthropics/claude-agent-sdk-python/commit/c01a99416a02805b0a050d058da51bde53f6df64)
- [TypeScript SDK 0.3.304](https://github.com/anthropics/claude-agent-sdk-typescript/commit/b8060fa5ac705b8f69ea32aafd21d32a9851000e)
- Both upstream HEADs target Claude Code **2.1.304** (Python's latest changelog
  release, 0.2.168, bundles 2.1.301). The binary is supplied by the caller,
  not installed or upgraded by this crate.

The 2.1.287–2.1.304 source review found no required wire-runtime changes: Python's
query, transport and message parser are unchanged. New task ancestry/run ids,
subagent `agent_id`, rate-limit metadata and initialization fields remain
available in raw provider events; subagent frames still carry
`parent_tool_use_id`. MCP elicitation `property_order` is preserved in the
interaction payload alongside the normalized `requestedSchema`, so hosts need
not rely on JSON object key order. New CLI defaults (including fork subagents
and permission mode) remain caller-owned; pass explicit configuration when
those defaults are not desired. Non-interactive CLI sessions omit thinking
text by default; callers needing reasoning events should configure
`--thinking-display summarized`.

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
