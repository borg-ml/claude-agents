//! Native Claude Code agent protocol and process runtime.
//!
//! The runtime speaks Claude Code's `stream-json` + `control_request` protocol
//! directly. It deliberately has no dependency on Borg or a JavaScript
//! runtime; callers provide the fully configured Claude command and consume
//! the typed event/control stream.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

mod stream;

/// A fully configured Claude Code child command.
#[derive(Debug, Clone)]
pub struct CommandSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub current_dir: PathBuf,
    pub environment: Vec<(String, String)>,
    pub environment_remove: Vec<String>,
}

/// Temporary home/configuration storage that stays alive for the duration of
/// a direct turn or for as long as a pooled process is retained.
#[derive(Debug, Clone)]
pub struct RuntimeDirectory {
    guard: std::sync::Arc<tempfile::TempDir>,
}

impl RuntimeDirectory {
    pub fn new() -> Result<Self> {
        Ok(Self {
            guard: std::sync::Arc::new(tempfile::tempdir()?),
        })
    }

    pub fn path(&self) -> &Path {
        self.guard.path()
    }
}

impl CommandSpec {
    fn into_command(self) -> tokio::process::Command {
        use std::process::Stdio;

        let mut command = tokio::process::Command::new(&self.program);
        command
            .args(self.args)
            .current_dir(self.current_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (key, value) in self.environment {
            command.env(key, value);
        }
        for key in self.environment_remove {
            command.env_remove(key);
        }
        isolate_async_process_from_terminal(&mut command);
        command
    }
}

/// Input for one Claude Code turn. The caller owns binary discovery, auth,
/// MCP setup, provider-channel environment, and CLI argument construction.
#[derive(Debug, Clone)]
pub struct ChatStreamRequest {
    pub prompt: String,
    pub attachments: Vec<PathBuf>,
    pub system_prompt: String,
    pub command: CommandSpec,
    pub runtime_directory: Option<RuntimeDirectory>,
    /// Stable configuration identity used to decide whether a pooled process
    /// can serve this turn.
    pub lifecycle_key: String,
}

/// Public name for the standalone SDK's request type.
pub type ClaudeRequest = ChatStreamRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatApprovalDecision {
    ApproveOnce,
    ApproveSession,
    Reject,
}

#[derive(Debug)]
pub enum ChatStreamControl {
    Steer {
        text: String,
        attachments: Vec<PathBuf>,
        ack: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
    Approval {
        approval_id: String,
        decision: ChatApprovalDecision,
    },
    ProviderInteractionResponse {
        interaction_id: String,
        response: Value,
    },
    Interrupt,
}

pub type ClaudeControl = ChatStreamControl;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ProviderCallUsage {
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub cached_input_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub context_tokens: Option<u64>,
    #[serde(default)]
    pub context_window_tokens: Option<u64>,
    #[serde(default)]
    pub cost_microusd: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    ProviderEvent {
        kind: String,
        payload: Value,
        raw_payload: Option<Value>,
        stream_channel: Option<String>,
        content_text: Option<String>,
        provider_item_id: Option<String>,
        tool_use_id: Option<String>,
        tool_name: Option<String>,
    },
    Delta(String),
    ReasoningDelta(String),
    Narration {
        text: String,
    },
    Phase {
        name: String,
        input: Value,
    },
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        output: String,
        is_error: bool,
        input: Option<Value>,
    },
    ApprovalRequested {
        approval_id: String,
        title: String,
        detail: String,
        command: Option<String>,
    },
    ProviderInteractionRequested {
        interaction_id: String,
        kind: String,
        title: String,
        detail: String,
        payload: Value,
    },
    Done {
        final_text: String,
        usage: Option<ProviderCallUsage>,
        session_id: Option<String>,
    },
    Failed {
        error: String,
    },
}

/// Native Rust runtime pool. A pooled process is only returned after a clean
/// terminal frame and completed interrupt/context cleanup.
#[derive(Clone, Default)]
pub struct ClaudePool {
    inner: std::sync::Arc<tokio::sync::Mutex<Option<PooledClaudeNative>>>,
}

pub type ClaudeAgentsPool = ClaudePool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalAgentPermission {
    FullAccess,
    Auto,
    Manual,
}

fn elapsed_millis_u64(started_at: std::time::Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

#[cfg(unix)]
fn isolate_async_process_from_terminal(command: &mut tokio::process::Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn isolate_async_process_from_terminal(_command: &mut tokio::process::Command) {}

fn prompt_text(prompt: &str, attachments: &[PathBuf]) -> String {
    if attachments.is_empty() {
        return prompt.to_string();
    }
    let list = attachments
        .iter()
        .map(|path| format!("- {}", path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{prompt}\n\nAttached files:\n{list}")
}

#[derive(Debug, Clone, Default)]
struct ProviderEventTelemetry {
    stream_channel: Option<String>,
    content_text: Option<String>,
    provider_item_id: Option<String>,
    tool_use_id: Option<String>,
    tool_name: Option<String>,
}

fn extract_text_block(item: &Value) -> Option<String> {
    if item.get("type").and_then(Value::as_str) != Some("text") {
        return None;
    }
    item.get("text")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn extract_tool_result_content(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| match item {
                        Value::String(text) => text.clone(),
                        other => other.to_string(),
                    })
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

fn provider_cost_usd_to_microusd(cost: f64) -> Option<u64> {
    if !cost.is_finite() || cost.is_sign_negative() {
        return None;
    }
    Some((cost * 1_000_000.0).round() as u64)
}

fn extract_claude_usage(envelope: &Value) -> ProviderCallUsage {
    let usage = envelope
        .get("usage")
        .or_else(|| envelope.pointer("/message/usage"));
    let input_tokens = usage
        .and_then(|value| value.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached_input_tokens = usage
        .and_then(|value| value.get("cache_read_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_creation_input_tokens = usage
        .and_then(|value| value.get("cache_creation_input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|value| value.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_tokens = input_tokens
        .saturating_add(cached_input_tokens)
        .saturating_add(cache_creation_input_tokens)
        .saturating_add(output_tokens);
    let cost_microusd = envelope
        .get("total_cost_usd")
        .and_then(Value::as_f64)
        .and_then(provider_cost_usd_to_microusd);
    ProviderCallUsage {
        duration_ms: envelope
            .get("duration_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        input_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        output_tokens,
        total_tokens,
        context_tokens: None,
        context_window_tokens: None,
        cost_microusd,
    }
}

/// Extract token and provider-reported cost fields from a Claude message or
/// result envelope.
pub fn extract_usage(envelope: &Value) -> ProviderCallUsage {
    extract_claude_usage(envelope)
}

fn summarize_claude_provider_event(value: &Value) -> Value {
    let mut out = serde_json::Map::new();
    for key in ["type", "subtype", "session_id"] {
        if let Some(value) = value.get(key).and_then(Value::as_str) {
            out.insert(key.to_string(), Value::String(value.to_string()));
        }
    }
    if let Some(result) = value.get("result").and_then(Value::as_str) {
        out.insert(
            "result_chars".to_string(),
            serde_json::json!(result.chars().count()),
        );
    }
    if let Some(content) = value
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_array)
    {
        out.insert(
            "content_blocks".to_string(),
            serde_json::json!(content.len()),
        );
        out.insert(
            "content_block_types".to_string(),
            serde_json::json!(
                content
                    .iter()
                    .filter_map(|block| block.get("type").and_then(Value::as_str))
                    .collect::<Vec<_>>()
            ),
        );
    }
    Value::Object(out)
}

fn classify_claude_provider_event(value: &Value) -> ProviderEventTelemetry {
    match value.get("type").and_then(Value::as_str).unwrap_or("") {
        "stream_event" => {
            let Some(event) = value.get("event") else {
                return ProviderEventTelemetry::default();
            };
            if event.get("type").and_then(Value::as_str) == Some("content_block_delta") {
                if let Some(delta) = event.get("delta") {
                    let (stream_channel, content_text) =
                        match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                            "text_delta" => (
                                Some("assistant_text".to_string()),
                                delta
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                            ),
                            "thinking_delta" => (
                                Some("reasoning".to_string()),
                                delta
                                    .get("thinking")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                            ),
                            _ => (None, None),
                        };
                    if stream_channel.is_some() {
                        return ProviderEventTelemetry {
                            stream_channel,
                            content_text,
                            provider_item_id: event
                                .get("index")
                                .and_then(Value::as_i64)
                                .map(|index| index.to_string()),
                            ..ProviderEventTelemetry::default()
                        };
                    }
                }
            }
            ProviderEventTelemetry {
                stream_channel: Some("provider_event".to_string()),
                ..ProviderEventTelemetry::default()
            }
        }
        "assistant" => {
            let content = value
                .get("message")
                .and_then(|message| message.get("content"))
                .and_then(Value::as_array);
            if let Some(tool) = content.and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            }) {
                return ProviderEventTelemetry {
                    stream_channel: Some("tool_call".to_string()),
                    provider_item_id: tool.get("id").and_then(Value::as_str).map(str::to_string),
                    tool_use_id: tool.get("id").and_then(Value::as_str).map(str::to_string),
                    tool_name: tool.get("name").and_then(Value::as_str).map(str::to_string),
                    content_text: tool.get("input").map(Value::to_string),
                };
            }
            ProviderEventTelemetry {
                stream_channel: Some("assistant_message".to_string()),
                content_text: content.map(|blocks| {
                    blocks
                        .iter()
                        .filter_map(extract_text_block)
                        .collect::<Vec<_>>()
                        .join("\n\n")
                }),
                ..ProviderEventTelemetry::default()
            }
        }
        "user" => {
            let content = value
                .get("message")
                .and_then(|message| message.get("content"))
                .and_then(Value::as_array);
            if let Some(tool_result) = content.and_then(|blocks| {
                blocks
                    .iter()
                    .find(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
            }) {
                return ProviderEventTelemetry {
                    stream_channel: Some("tool_result".to_string()),
                    tool_use_id: tool_result
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    content_text: Some(extract_tool_result_content(tool_result.get("content"))),
                    ..ProviderEventTelemetry::default()
                };
            }
            ProviderEventTelemetry::default()
        }
        "result" => ProviderEventTelemetry {
            stream_channel: Some("terminal".to_string()),
            content_text: value
                .get("result")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..ProviderEventTelemetry::default()
        },
        "error" => ProviderEventTelemetry {
            stream_channel: Some("error".to_string()),
            content_text: value
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string),
            ..ProviderEventTelemetry::default()
        },
        _ => ProviderEventTelemetry::default(),
    }
}

async fn receive_claude_control(
    controls: &mut Option<tokio::sync::mpsc::Receiver<ChatStreamControl>>,
) -> Option<ChatStreamControl> {
    match controls {
        Some(controls) => controls.recv().await,
        None => std::future::pending().await,
    }
}

fn truncate(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    match text.char_indices().nth(max) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// One line of the child's stdout.
///
/// The stream is a multiplexed demux, not a pure message stream: control
/// traffic is interleaved with SDK messages. Unrecognized frames must be
/// ignored rather than treated as errors — Anthropic adds frame types without
/// bumping the protocol.
#[derive(Debug)]
pub(crate) enum Frame {
    /// Reply to a request we sent, correlated by `request_id`.
    ControlResponse {
        request_id: String,
        result: ControlOutcome,
    },
    /// The CLI asking *us* something (permissions, elicitation, hooks).
    /// Every one must be answered or the turn hangs.
    ControlRequest { request_id: String, request: Value },
    /// Withdraws an in-flight inbound request; drop it without replying.
    ControlCancel { request_id: String },
    /// An SDK message for `ClaudeStreamState`.
    Message(Value),
    /// Keep-alives, transcript mirrors, and future frame types.
    Ignored,
}

#[derive(Debug)]
pub(crate) enum ControlOutcome {
    Success(Value),
    Error(String),
}

impl Frame {
    pub(crate) fn parse(line: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("failed to parse claude frame: {}", truncate(line, 200)))?;
        Ok(Self::from_value(value))
    }

    fn from_value(value: Value) -> Self {
        match value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "control_response" => {
                let response = match value.get("response") {
                    Some(response) => response,
                    None => return Frame::Ignored,
                };
                let request_id = response
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if request_id.is_empty() {
                    return Frame::Ignored;
                }
                // Success payloads may carry prompt-redelivery fields
                // (`pending_permission_requests`, `pending_user_dialog_requests`).
                // The SDK strips and ignores them; so do we.
                let result = match response.get("subtype").and_then(Value::as_str) {
                    Some("error") => ControlOutcome::Error(
                        response
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("claude control request failed")
                            .to_string(),
                    ),
                    _ => ControlOutcome::Success(
                        response.get("response").cloned().unwrap_or(Value::Null),
                    ),
                };
                Frame::ControlResponse { request_id, result }
            }
            "control_request" => {
                let request_id = value
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let request = value.get("request").cloned().unwrap_or(Value::Null);
                if request_id.is_empty() {
                    return Frame::Ignored;
                }
                Frame::ControlRequest {
                    request_id,
                    request,
                }
            }
            "control_cancel_request" => {
                let request_id = value
                    .get("request_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if request_id.is_empty() {
                    return Frame::Ignored;
                }
                Frame::ControlCancel { request_id }
            }
            "keep_alive" | "transcript_mirror" => Frame::Ignored,
            "" => Frame::Ignored,
            _ => Frame::Message(value),
        }
    }
}

/// A `control_request` we send to the CLI. `request_id` is opaque and
/// caller-generated; the CLI echoes it back for correlation.
#[derive(Debug, Serialize)]
pub(crate) struct OutboundControlRequest<'a> {
    #[serde(rename = "type")]
    pub(crate) frame_type: &'static str,
    pub(crate) request_id: &'a str,
    pub(crate) request: Value,
}

impl<'a> OutboundControlRequest<'a> {
    pub(crate) fn new(request_id: &'a str, request: Value) -> Self {
        Self {
            frame_type: "control_request",
            request_id,
            request,
        }
    }
}

/// Our reply to an inbound `control_request`.
pub(crate) fn control_response_success(request_id: &str, payload: Value) -> Value {
    serde_json::json!({
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": request_id,
            "response": payload,
        }
    })
}

pub(crate) fn control_response_error(request_id: &str, error: &str) -> Value {
    serde_json::json!({
        "type": "control_response",
        "response": {
            "subtype": "error",
            "request_id": request_id,
            "error": error,
        }
    })
}

// ---------------------------------------------------------------------------
// Control channel
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Translates between the public event/control vocabulary and the CLI's
/// control frames.
///
/// Deliberately I/O-free: every method returns the frames to write rather than
/// writing them, so the protocol logic is unit-testable without a subprocess.
/// The runner owns stdin and the actual byte pushing.
#[derive(Debug, Default)]
pub(crate) struct ControlChannel {
    /// Inbound `can_use_tool` requests awaiting a user decision. Holds the
    /// `permission_suggestions` needed to build `updatedPermissions` on a
    /// session-scoped approval, and the original `input` so an allow can echo
    /// it back as `updatedInput`.
    pending_approvals: HashMap<String, PendingApproval>,
    /// Inbound elicitations awaiting a response. The key is the SDK-shaped
    /// interaction id surfaced to the caller; the value is the wire control
    /// request id that must be used for the reply.
    pending_interactions: HashMap<String, String>,
    /// Outbound requests awaiting a `control_response`, by `request_id`.
    pending_requests: HashMap<String, OutboundKind>,
    /// Interrupt receipts must be followed through before a pooled process is
    /// reusable. Older CLIs without the receipt capability deliberately make
    /// the process non-reusable after an interrupt.
    interrupt_cleanup: InterruptCleanup,
}

#[derive(Debug, Clone)]
struct PendingApproval {
    tool_use_id: Option<String>,
    input: Value,
    suggestions: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutboundKind {
    Initialize,
    Interrupt,
    CancelAsyncMessage,
    StopTask,
    ContextUsage,
}

#[derive(Debug, Default)]
struct InterruptCleanup {
    awaiting_receipt: bool,
    cancel_queued: bool,
    pending_cancellations: usize,
    failed: bool,
}

/// What the runner should do with an inbound frame.
#[derive(Debug)]
pub(crate) enum Inbound {
    /// Surface to the caller; no immediate reply.
    Event(ChatStreamEvent),
    /// Reply immediately with this frame.
    Reply(Value),
    /// A reply to something we sent.
    Response {
        kind: OutboundKind,
        result: ControlOutcome,
    },
    Nothing,
}

impl ControlChannel {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Register an outbound request so its response can be correlated.
    pub(crate) fn begin_request(&mut self, request_id: &str, kind: OutboundKind) {
        self.pending_requests.insert(request_id.to_string(), kind);
    }

    pub(crate) fn handle_frame(&mut self, frame: Frame) -> Inbound {
        match frame {
            Frame::ControlRequest {
                request_id,
                request,
            } => self.handle_inbound_request(request_id, &request),
            Frame::ControlResponse { request_id, result } => {
                match self.pending_requests.remove(&request_id) {
                    Some(kind) => Inbound::Response { kind, result },
                    // A response we never asked for, or a duplicate. The SDK
                    // parks these; we have no use for one.
                    None => Inbound::Nothing,
                }
            }
            Frame::ControlCancel { request_id } => {
                // Withdrawn: drop the pending entry and send nothing. Replying
                // to a cancelled request is a protocol error.
                self.pending_approvals.remove(&request_id);
                self.pending_interactions
                    .retain(|_, wire_request_id| wire_request_id != &request_id);
                Inbound::Nothing
            }
            Frame::Message(_) | Frame::Ignored => Inbound::Nothing,
        }
    }

    fn handle_inbound_request(&mut self, request_id: String, request: &Value) -> Inbound {
        let subtype = request
            .get("subtype")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match subtype {
            "can_use_tool" => {
                let tool_name = request
                    .get("tool_name")
                    .and_then(Value::as_str)
                    .unwrap_or("a tool");
                let input = request.get("input").cloned().unwrap_or(Value::Null);
                self.pending_approvals.insert(
                    request_id.clone(),
                    PendingApproval {
                        tool_use_id: request
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        input: input.clone(),
                        suggestions: request.get("permission_suggestions").cloned(),
                    },
                );
                // Prefer the richer display fields and retain deterministic
                // wording for older Claude Code frames.
                let title = first_str(request, &["title", "display_name"])
                    .unwrap_or_else(|| format!("Use {tool_name}"));
                let detail = first_str(request, &["description", "decision_reason"])
                    .unwrap_or_else(|| format!("Claude requested permission to use {tool_name}."));
                let command = (tool_name == "Bash")
                    .then(|| input.get("command").and_then(Value::as_str))
                    .flatten()
                    .map(str::to_string);
                Inbound::Event(ChatStreamEvent::ApprovalRequested {
                    approval_id: request_id,
                    title,
                    detail,
                    command,
                })
            }
            "elicitation" => {
                let payload = normalize_elicitation_payload(request);
                let server = payload
                    .get("serverName")
                    .and_then(Value::as_str)
                    .unwrap_or("An MCP server");
                // Match the SDK: expose elicitationId when supplied, and
                // synthesize an interaction id for older URL/form requests
                // that omit it. Replies still use the control request id.
                let interaction_id = payload
                    .get("elicitationId")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                self.pending_interactions
                    .insert(interaction_id.clone(), request_id);
                let title = first_str(&payload, &["title", "displayName"])
                    .unwrap_or_else(|| format!("{server} requests input"));
                let detail = first_str(&payload, &["description", "message"])
                    .unwrap_or_else(|| "Claude needs additional input.".to_string());
                Inbound::Event(ChatStreamEvent::ProviderInteractionRequested {
                    interaction_id,
                    kind: "mcp_elicitation".to_string(),
                    title,
                    detail,
                    payload,
                })
            }
            // Every inbound request must be answered or the turn hangs. We do
            // not implement hooks or SDK MCP servers, so decline explicitly
            // rather than leaving the CLI waiting.
            other => Inbound::Reply(control_response_error(
                &request_id,
                &format!("claude-agents does not implement control request '{other}'"),
            )),
        }
    }

    /// Build the reply for a user's approval decision. Returns `None` if the
    /// approval is unknown (already cancelled, or a duplicate decision).
    pub(crate) fn approval_reply(
        &mut self,
        approval_id: &str,
        decision: ChatApprovalDecision,
    ) -> Option<Value> {
        let pending = self.pending_approvals.remove(approval_id)?;
        let payload = match decision {
            ChatApprovalDecision::Reject => serde_json::json!({
                "behavior": "deny",
                "message": "User denied this action.",
            }),
            ChatApprovalDecision::ApproveOnce | ChatApprovalDecision::ApproveSession => {
                let mut allow = serde_json::json!({
                    "behavior": "allow",
                    // Echo the input so callers can support approval flows
                    // that modify tool arguments.
                    "updatedInput": pending.input,
                });
                if decision == ChatApprovalDecision::ApproveSession {
                    if let Some(suggestions) = pending.suggestions {
                        allow["updatedPermissions"] = suggestions;
                    }
                }
                allow
            }
        };
        let mut payload = payload;
        if let Some(tool_use_id) = pending.tool_use_id {
            payload["toolUseID"] = Value::String(tool_use_id);
        }
        Some(control_response_success(approval_id, payload))
    }

    /// Build the reply for an elicitation response.
    pub(crate) fn interaction_reply(
        &mut self,
        interaction_id: &str,
        response: Value,
    ) -> Option<Value> {
        let request_id = self.pending_interactions.remove(interaction_id)?;
        Some(control_response_success(&request_id, response))
    }

    /// Deny everything still outstanding. Called when the child is going away,
    /// so a UI waiting on an approval does not hang forever.
    pub(crate) fn drain_pending(&mut self) -> Vec<Value> {
        let mut frames = Vec::new();
        for approval_id in self.pending_approvals.keys().cloned().collect::<Vec<_>>() {
            self.pending_approvals.remove(&approval_id);
            frames.push(control_response_success(
                &approval_id,
                serde_json::json!({
                    "behavior": "deny",
                    "message": "Claude session ended before permission was decided.",
                }),
            ));
        }
        for (_, request_id) in self.pending_interactions.drain() {
            frames.push(control_response_success(
                &request_id,
                serde_json::json!({"action": "cancel"}),
            ));
        }
        frames
    }

    #[cfg(any())]
    pub(crate) fn has_pending_approval(&self, approval_id: &str) -> bool {
        self.pending_approvals.contains_key(approval_id)
    }

    pub(crate) fn has_pending_context_usage(&self) -> bool {
        self.pending_requests
            .values()
            .any(|kind| *kind == OutboundKind::ContextUsage)
    }

    pub(crate) fn begin_interrupt(&mut self, supports_receipt: bool, cancel_queued: bool) {
        self.interrupt_cleanup = InterruptCleanup {
            awaiting_receipt: true,
            cancel_queued,
            pending_cancellations: 0,
            // Without an advertised receipt there is no safe way to prove
            // that queued user messages were removed before pool reuse.
            failed: !supports_receipt,
        };
    }

    pub(crate) fn has_pending_interrupt_cleanup(&self) -> bool {
        self.interrupt_cleanup.awaiting_receipt || self.interrupt_cleanup.pending_cancellations > 0
    }

    pub(crate) fn interrupt_cleanup_ready(&self) -> bool {
        !self.interrupt_cleanup.awaiting_receipt
            && self.interrupt_cleanup.pending_cancellations == 0
            && !self.interrupt_cleanup.failed
    }

    async fn handle_interrupt_receipt(
        &mut self,
        payload: Value,
        stdin: &mut tokio::process::ChildStdin,
    ) -> Result<()> {
        if !self.interrupt_cleanup.awaiting_receipt {
            return Ok(());
        }
        let Some(queued) = payload.get("still_queued").and_then(Value::as_array) else {
            self.interrupt_cleanup.awaiting_receipt = false;
            self.interrupt_cleanup.failed = true;
            return Ok(());
        };
        if self.interrupt_cleanup.cancel_queued
            && payload.get("cancelled").and_then(Value::as_array).is_none()
        {
            self.interrupt_cleanup.failed = true;
        }
        self.interrupt_cleanup.awaiting_receipt = false;
        for message_uuid in queued.iter().filter_map(Value::as_str) {
            let request_id = format!("claude-agents-cancel-{}", uuid::Uuid::new_v4());
            self.begin_request(&request_id, OutboundKind::CancelAsyncMessage);
            write_line(
                stdin,
                &serde_json::to_value(OutboundControlRequest::new(
                    &request_id,
                    serde_json::json!({
                        "subtype": "cancel_async_message",
                        "message_uuid": message_uuid,
                    }),
                ))?,
            )
            .await?;
            self.interrupt_cleanup.pending_cancellations += 1;
        }
        Ok(())
    }

    fn handle_cancel_async_response(&mut self, result: ControlOutcome) {
        if self.interrupt_cleanup.pending_cancellations == 0 {
            return;
        }
        self.interrupt_cleanup.pending_cancellations -= 1;
        match result {
            ControlOutcome::Success(payload)
                if payload.get("cancelled").and_then(Value::as_bool) == Some(true) => {}
            ControlOutcome::Success(_) | ControlOutcome::Error(_) => {
                self.interrupt_cleanup.failed = true;
            }
        }
    }

    fn fail_interrupt_cleanup(&mut self) {
        self.interrupt_cleanup.awaiting_receipt = false;
        self.interrupt_cleanup.failed = true;
    }
}

fn first_str(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .map(str::to_string)
    })
}

fn normalize_elicitation_payload(request: &Value) -> Value {
    let mut payload = request.clone();
    let Some(object) = payload.as_object_mut() else {
        return payload;
    };
    // `subtype` belongs to the control envelope, not the SDK's
    // ElicitationRequest object exposed to the callback/UI.
    object.remove("subtype");
    for (wire, sdk) in [
        ("mcp_server_name", "serverName"),
        ("elicitation_id", "elicitationId"),
        ("requested_schema", "requestedSchema"),
        ("display_name", "displayName"),
        ("additional_context", "additionalContext"),
    ] {
        if !object.contains_key(sdk) {
            if let Some(value) = object.remove(wire) {
                object.insert(sdk.to_string(), value);
            }
        } else {
            object.remove(wire);
        }
    }
    payload
}

/// Build a user message envelope for the Claude Code stream.
pub fn user_message(text: &str) -> Value {
    user_message_with_id(text).1
}

fn user_message_with_id(text: &str) -> (String, Value) {
    let id = uuid::Uuid::new_v4().to_string();
    let message = serde_json::json!({
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        "parent_tool_use_id": Value::Null,
        "session_id": "",
        "uuid": id,
    });
    (id, message)
}

fn message_belongs_to_turn(value: &Value, input_ids: &HashSet<String>) -> bool {
    if value.get("type").and_then(Value::as_str) != Some("result") {
        return true;
    }
    value
        .get("user_message_uuid")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .is_none_or(|id| input_ids.contains(id))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnMessageAction {
    Forward,
    Suppress,
    Terminal,
}

#[derive(Debug, Default)]
struct TurnMessageBoundary {
    live_background_tasks: HashSet<String>,
    observed_background_work: bool,
    awaiting_post_background_result: bool,
}

fn capture_init_capabilities(value: &Value, capabilities: &mut HashSet<String>) {
    if value.get("type").and_then(Value::as_str) != Some("system")
        || value.get("subtype").and_then(Value::as_str) != Some("init")
    {
        return;
    }
    if let Some(items) = value.get("capabilities").and_then(Value::as_array) {
        capabilities.clear();
        capabilities.extend(items.iter().filter_map(Value::as_str).map(str::to_string));
    }
}

impl TurnMessageBoundary {
    fn background_task_ids(&self) -> Vec<String> {
        self.live_background_tasks.iter().cloned().collect()
    }

    fn classify(&mut self, value: &Value, input_ids: &HashSet<String>) -> TurnMessageAction {
        if value.get("type").and_then(Value::as_str) == Some("system")
            && value.get("subtype").and_then(Value::as_str) == Some("background_tasks_changed")
        {
            self.live_background_tasks.clear();
            if let Some(tasks) = value.get("tasks").and_then(Value::as_array) {
                self.live_background_tasks
                    .extend(tasks.iter().filter_map(|task| {
                        task.get("task_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    }));
            }
            if !self.live_background_tasks.is_empty() {
                self.observed_background_work = true;
            }
            return TurnMessageAction::Forward;
        }

        if value.get("type").and_then(Value::as_str) != Some("result") {
            return TurnMessageAction::Forward;
        }

        if self.awaiting_post_background_result && self.live_background_tasks.is_empty() {
            // Result correlation is optional. After withholding a foreground
            // result, the next result at an empty live-task level is the
            // post-notification terminal even when the SDK omits its UUID.
            self.awaiting_post_background_result = false;
            return TurnMessageAction::Terminal;
        }

        if !message_belongs_to_turn(value, input_ids) {
            // A completion notification is an internal user message with its
            // own UUID. Its follow-up result is valid only when this turn
            // previously withheld a result while background work was live.
            return TurnMessageAction::Suppress;
        }

        if self.observed_background_work {
            self.awaiting_post_background_result = true;
            return TurnMessageAction::Suppress;
        }

        self.awaiting_post_background_result = false;
        TurnMessageAction::Terminal
    }
}

/// `initialize` handshake payload. `systemPrompt` is an array on the wire.
pub(crate) fn initialize_request(system_prompt: Option<&str>) -> Value {
    let mut request = serde_json::json!({"subtype": "initialize"});
    if let Some(prompt) = system_prompt.filter(|text| !text.trim().is_empty()) {
        request["systemPrompt"] = Value::Array(vec![Value::String(prompt.to_string())]);
    }
    request
}

/// Ask the CLI for live context telemetry after each assistant message.
async fn request_context_usage(
    channel: &mut ControlChannel,
    stdin: &mut tokio::process::ChildStdin,
) -> Result<()> {
    let request_id = format!("claude-agents-context-{}", uuid::Uuid::new_v4());
    channel.begin_request(&request_id, OutboundKind::ContextUsage);
    let result = write_line(
        stdin,
        &serde_json::to_value(OutboundControlRequest::new(
            &request_id,
            serde_json::json!({"subtype": "get_context_usage"}),
        ))?,
    )
    .await;
    if result.is_err() {
        channel.pending_requests.remove(&request_id);
    }
    result
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Public runtime
// ---------------------------------------------------------------------------

/// Run a single Claude Code turn on a caller-configured process.
pub async fn run(
    req: ChatStreamRequest,
    tx: tokio::sync::mpsc::Sender<ChatStreamEvent>,
    mut controls: Option<tokio::sync::mpsc::Receiver<ChatStreamControl>>,
) -> Result<()> {
    use crate::stream::ClaudeStreamState;

    let _runtime_directory = req.runtime_directory.clone();
    let started_at = std::time::Instant::now();
    let program = req.command.program.clone();
    let mut child = req
        .command
        .into_command()
        .spawn()
        .with_context(|| format!("failed to spawn {}", program.display()))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("claude stdin pipe missing"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("claude stdout pipe missing"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("claude stderr pipe missing"))?;

    let stderr_buf = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
    {
        let stderr_buf = stderr_buf.clone();
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stderr);
            let mut buf = String::new();
            loop {
                buf.clear();
                match reader.read_line(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => stderr_buf.lock().await.push_str(&buf),
                }
            }
        });
    }

    let mut channel = ControlChannel::new();
    let mut state = ClaudeStreamState::default();
    let init_id = format!("claude-agents-init-{}", uuid::Uuid::new_v4());
    channel.begin_request(&init_id, OutboundKind::Initialize);
    write_line(
        &mut stdin,
        &serde_json::to_value(OutboundControlRequest::new(
            &init_id,
            initialize_request(Some(&req.system_prompt)),
        ))?,
    )
    .await?;

    let (initial_input_id, initial_message) =
        user_message_with_id(&prompt_text(&req.prompt, &req.attachments));
    let mut input_ids = HashSet::from([initial_input_id]);
    write_line(&mut stdin, &initial_message).await?;

    let mut reader = tokio::io::BufReader::new(stdout);
    let mut line = String::new();
    let mut stream_ended = false;
    let mut terminal_seen = false;
    let mut capabilities = HashSet::new();
    let mut context_deadline = None;
    let mut interrupt_deadline = None;
    let mut turn_boundary = TurnMessageBoundary::default();

    while !stream_ended || channel.has_pending_interrupt_cleanup() {
        line.clear();
        let context_deadline_at = context_deadline;
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(
                context_deadline_at.unwrap_or_else(tokio::time::Instant::now)
            ), if terminal_seen && channel.has_pending_context_usage() => {
                tracing::warn!("claude context usage response timed out after the turn completed");
                stream_ended = true;
            }
            _ = tokio::time::sleep_until(
                interrupt_deadline.unwrap_or_else(tokio::time::Instant::now)
            ), if channel.has_pending_interrupt_cleanup() => {
                tracing::warn!("claude interrupt cleanup response timed out");
                channel.fail_interrupt_cleanup();
                stream_ended = true;
            }
            read = reader.read_line(&mut line) => {
                let read = read.context("failed reading claude stdout")?;
                if read == 0 {
                    break;
                }
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    continue;
                }
                match Frame::parse(trimmed)? {
                    Frame::Message(value) => {
                        capture_init_capabilities(&value, &mut capabilities);
                        let action = turn_boundary.classify(&value, &input_ids);
                        if action == TurnMessageAction::Suppress {
                            continue;
                        }
                        send_provider_event(&tx, &value).await;
                        let state_ended = state.handle_message(&value, &tx).await?;
                        if value.get("type").and_then(Value::as_str) == Some("assistant")
                            && !state_ended
                        {
                            if let Err(error) =
                                request_context_usage(&mut channel, &mut stdin).await
                            {
                                tracing::debug!(%error, "claude context usage request unavailable");
                            }
                        }
                        if state_ended {
                            stream_ended = true;
                        }
                        if action == TurnMessageAction::Terminal {
                            terminal_seen = true;
                            if channel.has_pending_context_usage() {
                                context_deadline = Some(
                                    tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                                );
                            } else if !channel.has_pending_interrupt_cleanup() {
                                stream_ended = true;
                            }
                        }
                    }
                    frame => match channel.handle_frame(frame) {
                        Inbound::Event(event) => {
                            if tx.send(event).await.is_err() {
                                stream_ended = true;
                            }
                        }
                        Inbound::Reply(reply) => write_line(&mut stdin, &reply).await?,
                        Inbound::Response { kind, result } => {
                            handle_control_response(kind, result, &mut channel, &mut stdin, &tx).await?;
                            if terminal_seen
                                && !channel.has_pending_context_usage()
                                && !channel.has_pending_interrupt_cleanup()
                            {
                                stream_ended = true;
                            }
                        }
                        Inbound::Nothing => {}
                    },
                }
            }
            control = receive_claude_control(&mut controls) => {
                match control {
                    Some(control) => {
                        if terminal_seen {
                            reject_late_native_control(control);
                        } else if !apply_control(
                            control,
                            &mut channel,
                            &mut stdin,
                            &mut input_ids,
                            &turn_boundary.background_task_ids(),
                            &capabilities,
                        ).await? {
                            stream_ended = true;
                        }
                        if channel.has_pending_interrupt_cleanup() {
                            interrupt_deadline = Some(
                                tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                            );
                        }
                    }
                    None => controls = None,
                }
            }
        }
    }

    for frame in channel.drain_pending() {
        let _ = write_line(&mut stdin, &frame).await;
    }
    let _ = stdin.shutdown().await;
    drop(stdin);
    let status = child.wait().await.context("failed waiting for claude")?;
    if state.emitted_failure {
        return Ok(());
    }
    if !status.success() {
        let stderr_text = stderr_buf.lock().await.clone();
        let suffix = if stderr_text.trim().is_empty() {
            String::new()
        } else {
            format!(": {}", stderr_text.trim())
        };
        let _ = tx
            .send(ChatStreamEvent::Failed {
                error: format!(
                    "claude exited with status {}{}",
                    status
                        .code()
                        .map(|code| code.to_string())
                        .unwrap_or_else(|| "?".to_string()),
                    suffix
                ),
            })
            .await;
        return Ok(());
    }
    let final_text = state
        .final_text
        .take()
        .unwrap_or_else(|| state.delta_accumulator.clone());
    let usage = Some(state.final_usage.unwrap_or_else(|| ProviderCallUsage {
        duration_ms: elapsed_millis_u64(started_at),
        ..ProviderCallUsage::default()
    }));
    let _ = tx
        .send(ChatStreamEvent::Done {
            final_text,
            usage,
            session_id: state.session_id.take(),
        })
        .await;
    Ok(())
}

struct PooledClaudeNative {
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    stdout: tokio::io::BufReader<tokio::process::ChildStdout>,
    stderr: std::sync::Arc<tokio::sync::Mutex<String>>,
    _runtime_directory: Option<RuntimeDirectory>,
    lifecycle_key: String,
    started: bool,
    capabilities: HashSet<String>,
    session_id: Option<String>,
    pending_steers: Vec<(String, Value)>,
}

async fn start_pooled_native(req: &ChatStreamRequest) -> Result<PooledClaudeNative> {
    let program = req.command.program.clone();
    let mut child = req
        .command
        .clone()
        .into_command()
        .spawn()
        .with_context(|| format!("failed to spawn {}", program.display()))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("pooled claude stdin pipe missing"))?;
    let stdout = tokio::io::BufReader::new(
        child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("pooled claude stdout pipe missing"))?,
    );
    let stderr = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
    let child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("pooled claude stderr pipe missing"))?;
    {
        let stderr = stderr.clone();
        tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(child_stderr);
            let mut line = String::new();
            while reader.read_line(&mut line).await.is_ok_and(|read| read > 0) {
                stderr.lock().await.push_str(&line);
                line.clear();
            }
        });
    }
    Ok(PooledClaudeNative {
        child,
        stdin,
        stdout,
        stderr,
        _runtime_directory: req.runtime_directory.clone(),
        lifecycle_key: req.lifecycle_key.clone(),
        started: false,
        capabilities: HashSet::new(),
        session_id: None,
        pending_steers: Vec::new(),
    })
}

/// Run a turn on a pooled Claude Code process. The process is retained only
/// after a correlated terminal frame and all cleanup receipts are complete.
pub async fn run_pooled(
    req: ChatStreamRequest,
    tx: tokio::sync::mpsc::Sender<ChatStreamEvent>,
    mut controls: Option<tokio::sync::mpsc::Receiver<ChatStreamControl>>,
    pool: ClaudePool,
) -> Result<()> {
    let mut guard = pool.inner.lock().await;
    let mut pooled = match guard.take() {
        Some(pooled) if pooled.lifecycle_key == req.lifecycle_key => pooled,
        Some(mut stale) => {
            let mut pending_steers = Vec::new();
            transfer_pending_steers(&mut stale.pending_steers, &mut pending_steers);
            let _ = stale.child.kill().await;
            let mut fresh = start_pooled_native(&req).await?;
            fresh.pending_steers = pending_steers;
            fresh
        }
        None => start_pooled_native(&req).await?,
    };

    let terminal = run_pooled_native_turn(
        &mut pooled,
        &req,
        &tx,
        &mut controls,
        std::time::Instant::now(),
    )
    .await;
    let reusable = match terminal {
        Ok(reusable) => reusable,
        Err(error) => {
            let _ = pooled.child.kill().await;
            return Err(error);
        }
    };
    if reusable {
        *guard = Some(pooled);
    } else {
        let _ = pooled.child.kill().await;
    }
    Ok(())
}

async fn run_pooled_native_turn(
    pooled: &mut PooledClaudeNative,
    req: &ChatStreamRequest,
    tx: &tokio::sync::mpsc::Sender<ChatStreamEvent>,
    controls: &mut Option<tokio::sync::mpsc::Receiver<ChatStreamControl>>,
    started_at: std::time::Instant,
) -> Result<bool> {
    use crate::stream::ClaudeStreamState;

    let mut channel = ControlChannel::new();
    if !pooled.started {
        let init_id = format!("claude-agents-init-{}", uuid::Uuid::new_v4());
        channel.begin_request(&init_id, OutboundKind::Initialize);
        write_line(
            &mut pooled.stdin,
            &serde_json::to_value(OutboundControlRequest::new(
                &init_id,
                initialize_request(Some(&req.system_prompt)),
            ))?,
        )
        .await?;
        pooled.started = true;
    }
    let (input_id, message) = user_message_with_id(&prompt_text(&req.prompt, &req.attachments));
    let mut input_ids = HashSet::from([input_id]);
    write_line(&mut pooled.stdin, &message).await?;
    for (input_id, message) in pooled.pending_steers.drain(..) {
        input_ids.insert(input_id);
        write_line(&mut pooled.stdin, &message).await?;
    }

    let mut state = ClaudeStreamState::default();
    let mut line = String::new();
    let mut terminal_seen = false;
    let mut abnormal_end = false;
    let mut context_deadline = None;
    let mut interrupt_deadline = None;
    let mut turn_boundary = TurnMessageBoundary::default();

    while !terminal_seen
        || channel.has_pending_context_usage()
        || channel.has_pending_interrupt_cleanup()
    {
        line.clear();
        let context_deadline_at = context_deadline;
        tokio::select! {
            biased;
            _ = tokio::time::sleep_until(
                context_deadline_at.unwrap_or_else(tokio::time::Instant::now)
            ), if terminal_seen && channel.has_pending_context_usage() => {
                tracing::warn!("pooled claude context usage response timed out after the turn completed");
                break;
            }
            _ = tokio::time::sleep_until(
                interrupt_deadline.unwrap_or_else(tokio::time::Instant::now)
            ), if channel.has_pending_interrupt_cleanup() => {
                tracing::warn!("pooled claude interrupt cleanup response timed out");
                channel.fail_interrupt_cleanup();
                break;
            }
            read = pooled.stdout.read_line(&mut line) => {
                let read = read.context("failed reading pooled claude stdout")?;
                if read == 0 {
                    let stderr = pooled.stderr.lock().await.clone();
                    if terminal_seen {
                        break;
                    }
                    bail!("pooled claude exited unexpectedly: {}", stderr.trim());
                }
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    continue;
                }
                match Frame::parse(trimmed)? {
                    Frame::Message(value) => {
                        capture_init_capabilities(&value, &mut pooled.capabilities);
                        let action = turn_boundary.classify(&value, &input_ids);
                        if action == TurnMessageAction::Suppress {
                            continue;
                        }
                        send_provider_event(tx, &value).await;
                        let state_ended = state.handle_message(&value, tx).await?;
                        if value.get("type").and_then(Value::as_str) == Some("assistant")
                            && !state_ended
                        {
                            if let Err(error) =
                                request_context_usage(&mut channel, &mut pooled.stdin).await
                            {
                                tracing::debug!(%error, "pooled claude context usage request unavailable");
                            }
                        }
                        if let Some(session_id) = state.session_id.as_ref() {
                            pooled.session_id = Some(session_id.clone());
                        }
                        if state_ended {
                            abnormal_end = true;
                        }
                        if state_ended || action == TurnMessageAction::Terminal {
                            terminal_seen = true;
                            if channel.has_pending_context_usage() {
                                context_deadline = Some(
                                    tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                                );
                            }
                        }
                    }
                    frame => match channel.handle_frame(frame) {
                        Inbound::Event(event) => {
                            if tx.send(event).await.is_err() {
                                return Ok(false);
                            }
                        }
                        Inbound::Reply(reply) => write_line(&mut pooled.stdin, &reply).await?,
                        Inbound::Response { kind, result } => {
                            handle_control_response(
                                kind,
                                result,
                                &mut channel,
                                &mut pooled.stdin,
                                tx,
                            ).await?;
                        }
                        Inbound::Nothing => {}
                    }
                }
            }
            control = receive_claude_control(controls), if controls.is_some() => {
                let Some(control) = control else {
                    *controls = None;
                    continue;
                };
                if terminal_seen {
                    queue_or_reject_native_control(pooled, control);
                } else if !apply_control(
                    control,
                    &mut channel,
                    &mut pooled.stdin,
                    &mut input_ids,
                    &turn_boundary.background_task_ids(),
                    &pooled.capabilities,
                ).await? {
                    terminal_seen = true;
                }
                if channel.has_pending_interrupt_cleanup() {
                    interrupt_deadline = Some(
                        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                    );
                }
            }
        }
    }

    while let Some(control) = controls
        .as_mut()
        .and_then(|receiver| receiver.try_recv().ok())
    {
        queue_or_reject_native_control(pooled, control);
    }
    for frame in channel.drain_pending() {
        let _ = write_line(&mut pooled.stdin, &frame).await;
    }
    if !terminal_seen || abnormal_end || !channel.interrupt_cleanup_ready() {
        return Ok(false);
    }
    if !state.emitted_failure {
        let final_text = state
            .final_text
            .take()
            .unwrap_or_else(|| state.delta_accumulator.clone());
        let usage = Some(state.final_usage.unwrap_or_else(|| ProviderCallUsage {
            duration_ms: elapsed_millis_u64(started_at),
            ..ProviderCallUsage::default()
        }));
        let _ = tx
            .send(ChatStreamEvent::Done {
                final_text,
                usage,
                session_id: state
                    .session_id
                    .take()
                    .or_else(|| pooled.session_id.clone()),
            })
            .await;
    }
    Ok(true)
}

async fn send_provider_event(tx: &tokio::sync::mpsc::Sender<ChatStreamEvent>, value: &Value) {
    let telemetry = classify_claude_provider_event(value);
    let _ = tx
        .send(ChatStreamEvent::ProviderEvent {
            kind: format!(
                "claude.{}",
                value
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message")
            ),
            payload: summarize_claude_provider_event(value),
            raw_payload: Some(value.clone()),
            stream_channel: telemetry.stream_channel,
            content_text: telemetry.content_text,
            provider_item_id: telemetry.provider_item_id,
            tool_use_id: telemetry.tool_use_id,
            tool_name: telemetry.tool_name,
        })
        .await;
}

fn transfer_pending_steers(from: &mut Vec<(String, Value)>, to: &mut Vec<(String, Value)>) {
    to.append(from);
}

fn queue_or_reject_native_control(pooled: &mut PooledClaudeNative, control: ChatStreamControl) {
    match control {
        ChatStreamControl::Steer {
            text,
            attachments,
            ack,
        } => {
            let (input_id, message) = user_message_with_id(&prompt_text(&text, &attachments));
            pooled.pending_steers.push((input_id, message));
            let _ = ack.send(Ok(()));
        }
        ChatStreamControl::Approval { .. }
        | ChatStreamControl::ProviderInteractionResponse { .. }
        | ChatStreamControl::Interrupt => {}
    }
}

fn reject_late_native_control(control: ChatStreamControl) {
    if let ChatStreamControl::Steer { ack, .. } = control {
        let _ = ack.send(Err(
            "Claude turn ended before the steer was delivered".to_string()
        ));
    }
}

async fn handle_control_response(
    kind: OutboundKind,
    result: ControlOutcome,
    channel: &mut ControlChannel,
    stdin: &mut tokio::process::ChildStdin,
    tx: &tokio::sync::mpsc::Sender<ChatStreamEvent>,
) -> Result<()> {
    match (kind, result) {
        (OutboundKind::Interrupt, ControlOutcome::Success(payload)) => {
            channel.handle_interrupt_receipt(payload, stdin).await?;
        }
        (OutboundKind::Interrupt, ControlOutcome::Error(error)) => {
            channel.fail_interrupt_cleanup();
            tracing::warn!(%error, "claude interrupt request failed");
        }
        (OutboundKind::CancelAsyncMessage, result) => {
            channel.handle_cancel_async_response(result);
        }
        (OutboundKind::ContextUsage, ControlOutcome::Success(payload)) => {
            let _ = tx
                .send(ChatStreamEvent::ProviderEvent {
                    kind: "claude.context_usage".to_string(),
                    payload: serde_json::json!({
                        "type": "borg_context_usage",
                        "total_tokens": payload.get("totalTokens"),
                        "context_window_tokens": payload.get("maxTokens"),
                        "raw_context_window_tokens": payload.get("rawMaxTokens"),
                        "model": payload.get("model"),
                        "categories": payload.get("categories"),
                    }),
                    raw_payload: Some(payload),
                    stream_channel: Some("usage".to_string()),
                    content_text: None,
                    provider_item_id: None,
                    tool_use_id: None,
                    tool_name: None,
                })
                .await;
        }
        (kind, ControlOutcome::Error(error)) => {
            tracing::warn!(?kind, %error, "claude control request failed");
        }
        _ => {}
    }
    Ok(())
}

async fn apply_control(
    control: ChatStreamControl,
    channel: &mut ControlChannel,
    stdin: &mut tokio::process::ChildStdin,
    input_ids: &mut HashSet<String>,
    background_task_ids: &[String],
    capabilities: &HashSet<String>,
) -> Result<bool> {
    match control {
        ChatStreamControl::Steer {
            text,
            attachments,
            ack,
        } => {
            let (input_id, message) = user_message_with_id(&prompt_text(&text, &attachments));
            let result = write_line(stdin, &message).await;
            if result.is_ok() {
                input_ids.insert(input_id);
            }
            let reply = result
                .as_ref()
                .map(|_| ())
                .map_err(|error| format!("{error:#}"));
            let _ = ack.send(reply);
            result?;
        }
        ChatStreamControl::Approval {
            approval_id,
            decision,
        } => {
            if let Some(reply) = channel.approval_reply(&approval_id, decision) {
                write_line(stdin, &reply).await?;
            }
        }
        ChatStreamControl::ProviderInteractionResponse {
            interaction_id,
            response,
        } => {
            if let Some(reply) = channel.interaction_reply(&interaction_id, response) {
                write_line(stdin, &reply).await?;
            }
        }
        ChatStreamControl::Interrupt => {
            if channel.has_pending_interrupt_cleanup() {
                return Ok(true);
            }
            let supports_receipt = capabilities.contains("interrupt_receipt_v1");
            let cancel_queued = capabilities.contains("interrupt_cancel_queued_v1");
            channel.begin_interrupt(supports_receipt, cancel_queued);
            for task_id in background_task_ids {
                let request_id = format!("claude-agents-stop-task-{}", uuid::Uuid::new_v4());
                channel.begin_request(&request_id, OutboundKind::StopTask);
                write_line(
                    stdin,
                    &serde_json::to_value(OutboundControlRequest::new(
                        &request_id,
                        serde_json::json!({"subtype": "stop_task", "task_id": task_id}),
                    ))?,
                )
                .await?;
            }
            let request_id = format!("claude-agents-interrupt-{}", uuid::Uuid::new_v4());
            channel.begin_request(&request_id, OutboundKind::Interrupt);
            let mut request = serde_json::json!({"subtype": "interrupt"});
            if cancel_queued {
                request["cancel_queued"] = Value::Bool(true);
            }
            write_line(
                stdin,
                &serde_json::to_value(OutboundControlRequest::new(&request_id, request))?,
            )
            .await?;
        }
    }
    Ok(true)
}

async fn write_line(stdin: &mut tokio::process::ChildStdin, value: &Value) -> Result<()> {
    stdin.write_all(&serde_json::to_vec(value)?).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn frame_parser_demultiplexes_control_and_message_frames() {
        assert!(matches!(
            Frame::parse(r#"{"type":"keep_alive"}"#).unwrap(),
            Frame::Ignored
        ));
        assert!(matches!(
            Frame::parse(r#"{"type":"assistant","message":{}}"#).unwrap(),
            Frame::Message(_)
        ));
        assert!(matches!(
            Frame::parse(r#"{"type":"control_response","response":{"subtype":"success","request_id":"r1","response":{"ok":true}}}"#).unwrap(),
            Frame::ControlResponse { request_id, result: ControlOutcome::Success(value) }
                if request_id == "r1" && value["ok"] == true
        ));
        assert!(Frame::parse("not json").is_err());
    }

    #[test]
    fn approval_and_elicitation_replies_preserve_wire_correlation() {
        let mut channel = ControlChannel::new();
        let approval = channel.handle_frame(Frame::ControlRequest {
            request_id: "approval-wire".to_string(),
            request: json!({
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "tool_use_id": "tool-1",
                "input": {"command": "cargo test"},
                "permission_suggestions": [{"type": "addRules", "rules": ["Bash"]}]
            }),
        });
        assert!(matches!(
            approval,
            Inbound::Event(ChatStreamEvent::ApprovalRequested {
                approval_id,
                command: Some(command),
                ..
            }) if approval_id == "approval-wire" && command == "cargo test"
        ));
        let reply = channel
            .approval_reply("approval-wire", ChatApprovalDecision::ApproveSession)
            .expect("approval reply");
        assert_eq!(reply["response"]["request_id"], "approval-wire");
        assert_eq!(reply["response"]["response"]["toolUseID"], "tool-1");
        assert_eq!(
            reply["response"]["response"]["updatedPermissions"][0]["type"],
            "addRules"
        );

        let interaction = channel.handle_frame(Frame::ControlRequest {
            request_id: "interaction-wire".to_string(),
            request: json!({
                "subtype": "elicitation",
                "mcp_server_name": "deploy",
                "elicitation_id": "interaction-1",
                "requested_schema": {"type": "object"}
            }),
        });
        assert!(matches!(
            interaction,
            Inbound::Event(ChatStreamEvent::ProviderInteractionRequested {
                interaction_id,
                payload,
                ..
            }) if interaction_id == "interaction-1"
                && payload["serverName"] == "deploy"
                && payload["requestedSchema"]["type"] == "object"
                && payload.get("subtype").is_none()
        ));
        let reply = channel
            .interaction_reply("interaction-1", json!({"action": "accept", "content": {}}))
            .expect("interaction reply");
        assert_eq!(reply["response"]["request_id"], "interaction-wire");
    }

    #[test]
    fn usage_extraction_accepts_assistant_and_result_envelopes() {
        let assistant = extract_usage(&json!({
            "type": "assistant",
            "message": {"usage": {
                "input_tokens": 11,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 5,
                "output_tokens": 3
            }}
        }));
        assert_eq!(assistant.total_tokens, 26);

        let result = extract_usage(&json!({
            "type": "result",
            "duration_ms": 250,
            "total_cost_usd": 0.012345,
            "usage": {"input_tokens": 20, "cache_read_input_tokens": 10, "output_tokens": 4}
        }));
        assert_eq!(result.duration_ms, 250);
        assert_eq!(result.total_tokens, 34);
        assert_eq!(result.cost_microusd, Some(12_345));
    }

    #[cfg(unix)]
    fn fake_request(_root: &Path, command: CommandSpec) -> ChatStreamRequest {
        ChatStreamRequest {
            prompt: "hello".to_string(),
            attachments: Vec::new(),
            system_prompt: "test system".to_string(),
            command,
            runtime_directory: Some(RuntimeDirectory::new().expect("runtime directory")),
            lifecycle_key: "stable-test-lifecycle".to_string(),
        }
    }

    #[cfg(unix)]
    fn fake_command(root: &Path) -> CommandSpec {
        let script = root.join("fake-claude");
        std::fs::write(
            &script,
            r##"#!/bin/sh
set -eu
turn=0
while IFS= read -r line; do
  case "$line" in
    *'"subtype":"initialize"'*)
      request_id=$(printf '%s\n' "$line" | sed -n 's/.*"request_id":"\([^"]*\)".*/\1/p')
      printf '{"type":"control_response","response":{"subtype":"success","request_id":"%s","response":{"capabilities":["interrupt_receipt_v1","interrupt_cancel_queued_v1"]}}}\n' "$request_id"
      ;;
    *'"subtype":"get_context_usage"'*)
      request_id=$(printf '%s\n' "$line" | sed -n 's/.*"request_id":"\([^"]*\)".*/\1/p')
      printf '{"type":"control_response","response":{"subtype":"success","request_id":"%s","response":{"totalTokens":3,"maxTokens":200000,"rawMaxTokens":200000,"model":"test-model"}}}\n' "$request_id"
      ;;
    *'"type":"user"'*)
      turn=$((turn + 1))
      printf '{"type":"system","subtype":"init","session_id":"session-test","capabilities":["interrupt_receipt_v1","interrupt_cancel_queued_v1"]}\n'
      printf '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"turn-%s"}}}\n' "$turn"
      printf '{"type":"assistant","session_id":"session-test","message":{"content":[{"type":"text","text":"turn-%s"}],"usage":{"input_tokens":1,"output_tokens":2}}}\n' "$turn"
      printf '{"type":"result","subtype":"success","result":"turn-%s","session_id":"session-test","usage":{"input_tokens":1,"output_tokens":2},"total_cost_usd":0.001}\n' "$turn"
      ;;
  esac
done
"##,
        )
        .expect("fake Claude runner");
        let mut permissions = std::fs::metadata(&script)
            .expect("fake Claude metadata")
            .permissions();
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(0o700);
        std::fs::set_permissions(&script, permissions).expect("fake Claude permissions");
        CommandSpec {
            program: script,
            args: Vec::new(),
            current_dir: root.to_path_buf(),
            environment: Vec::new(),
            environment_remove: Vec::new(),
        }
    }

    #[cfg(unix)]
    async fn done_text(mut events: tokio::sync::mpsc::Receiver<ChatStreamEvent>) -> String {
        while let Some(event) = events.recv().await {
            match event {
                ChatStreamEvent::Done { final_text, .. } => return final_text,
                ChatStreamEvent::Failed { error } => panic!("fake Claude failed: {error}"),
                _ => {}
            }
        }
        panic!("fake Claude ended without a terminal event");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn direct_and_pooled_runtimes_complete_native_stream_json_turns() {
        let root = tempfile::tempdir().expect("test root");
        let command = fake_command(root.path());

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        run(fake_request(root.path(), command.clone()), tx, None)
            .await
            .expect("direct runtime");
        assert_eq!(done_text(rx).await, "turn-1");

        let pool = ClaudePool::default();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        run_pooled(
            fake_request(root.path(), command.clone()),
            tx,
            None,
            pool.clone(),
        )
        .await
        .expect("first pooled runtime");
        assert_eq!(done_text(rx).await, "turn-1");

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        run_pooled(fake_request(root.path(), command), tx, None, pool)
            .await
            .expect("reused pooled runtime");
        assert_eq!(done_text(rx).await, "turn-2");
    }
}
