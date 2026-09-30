use anyhow::Result;
use serde_json::Value;
use tokio::sync::mpsc;

use super::{
    ChatStreamEvent, ProviderCallUsage, extract_claude_usage, extract_text_block,
    extract_tool_result_content,
};

#[derive(Default)]
pub(crate) struct ClaudeStreamState {
    pub(crate) saw_stream_text: bool,
    pub(crate) saw_stream_reasoning: bool,
    pub(crate) delta_accumulator: String,
    pub(crate) final_text: Option<String>,
    pub(crate) final_usage: Option<ProviderCallUsage>,
    pub(crate) emitted_failure: bool,
    /// Captured from the Claude Agent SDK's first "system" init
    /// message. Forwarded back to the caller on `Done` so the next
    /// turn can `resume: session_id`.
    pub(crate) session_id: Option<String>,
    /// Tokens of in-process subagent API calls, which the turn's `result`
    /// usage leaves out. Claude Code repeats a call's usage on every content
    /// block envelope, so each message id is counted once.
    subagent_usage: ProviderCallUsage,
    subagent_message_ids: std::collections::HashSet<String>,
}

impl ClaudeStreamState {
    pub(crate) async fn handle_message(
        &mut self,
        value: &Value,
        tx: &mpsc::Sender<ChatStreamEvent>,
    ) -> Result<bool> {
        if is_subagent_message(value) {
            if value.get("type").and_then(Value::as_str) == Some("assistant")
                && let Some(id) = value.pointer("/message/id").and_then(Value::as_str)
                && self.subagent_message_ids.insert(id.to_string())
            {
                let usage = extract_claude_usage(value);
                let total = &mut self.subagent_usage;
                total.input_tokens += usage.input_tokens;
                total.cached_input_tokens += usage.cached_input_tokens;
                total.cache_creation_input_tokens += usage.cache_creation_input_tokens;
                total.output_tokens += usage.output_tokens;
                total.total_tokens += usage.total_tokens;
            }
            return Ok(false);
        }
        let message_type = value.get("type").and_then(Value::as_str).unwrap_or("");
        match message_type {
            "stream_event" => {
                let event = match value.get("event") {
                    Some(event) => event,
                    None => return Ok(false),
                };
                if event.get("type").and_then(Value::as_str) != Some("content_block_delta") {
                    return Ok(false);
                }
                let delta = match event.get("delta") {
                    Some(delta) => delta,
                    None => return Ok(false),
                };
                match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            self.saw_stream_text = true;
                            self.delta_accumulator.push_str(text);
                            if tx
                                .send(ChatStreamEvent::Delta(text.to_string()))
                                .await
                                .is_err()
                            {
                                return Ok(true);
                            }
                        }
                    }
                    "thinking_delta" => {
                        if let Some(thinking) = delta.get("thinking").and_then(Value::as_str) {
                            self.saw_stream_reasoning = true;
                            if tx
                                .send(ChatStreamEvent::ReasoningDelta(thinking.to_string()))
                                .await
                                .is_err()
                            {
                                return Ok(true);
                            }
                        }
                    }
                    _ => {}
                }
            }
            "system" => {
                // Agent SDK emits a "system" init message at turn
                // start with session_id. Capture it; forwarded on
                // Done so the caller can `resume: session_id` next.
                if let Some(sid) = value.get("session_id").and_then(Value::as_str) {
                    if !sid.is_empty() {
                        self.session_id = Some(sid.to_string());
                    }
                }
            }
            "assistant" => {
                let usage = extract_claude_usage(value);
                if usage.duration_ms > 0 || usage.total_tokens > 0 || usage.cost_microusd.is_some()
                {
                    self.final_usage = Some(usage);
                }
                // Some SDK versions only surface session_id on the
                // assistant message envelope.
                if self.session_id.is_none() {
                    if let Some(sid) = value.get("session_id").and_then(Value::as_str) {
                        if !sid.is_empty() {
                            self.session_id = Some(sid.to_string());
                        }
                    }
                }
                // API failures are delivered as assistant envelopes by the
                // Agent SDK. Their text is user-facing, but the structured
                // `error` field means this turn failed and must not be treated
                // as a successful answer (which would auto-continue a goal).
                if let Some(kind) = value.get("error").and_then(Value::as_str) {
                    let message = value
                        .get("message")
                        .and_then(|message| message.get("content"))
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(extract_text_block)
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        })
                        .filter(|text| !text.trim().is_empty())
                        .unwrap_or_else(|| "Claude API request failed".to_string());
                    let status = value
                        .get("apiErrorStatus")
                        .or_else(|| value.get("api_error_status"))
                        .and_then(Value::as_u64);
                    let mut parts = vec![
                        format!("claude SDK API error: {message}"),
                        format!(r#""kind":"{kind}""#),
                    ];
                    if let Some(status) = status {
                        parts.push(format!(r#""status":{status}"#));
                    }
                    self.emitted_failure = true;
                    let _ = tx
                        .send(ChatStreamEvent::Failed {
                            error: parts.join(" "),
                        })
                        .await;
                    // Keep draining until the SDK's result envelope so a
                    // pooled runtime remains aligned for its next turn.
                    return Ok(false);
                }
                if let Some(content) = value
                    .get("message")
                    .and_then(|message| message.get("content"))
                    .and_then(Value::as_array)
                {
                    if !self.saw_stream_reasoning {
                        for block in content {
                            if block.get("type").and_then(Value::as_str) == Some("thinking") {
                                if let Some(thinking) =
                                    block.get("thinking").and_then(Value::as_str)
                                {
                                    if tx
                                        .send(ChatStreamEvent::ReasoningDelta(thinking.to_string()))
                                        .await
                                        .is_err()
                                    {
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    }
                    // Emit order matters for the consumer's bubble state:
                    // deltas first (progressive text), then the canonical
                    // Narration segment that replaces the delta
                    // accumulator, then tool calls (which flush the
                    // segment into a thinking breadcrumb). Sending
                    // Narration before the deltas doubles the segment in
                    // the live bubble.
                    if !self.saw_stream_text {
                        for block in content {
                            if block.get("type").and_then(Value::as_str) == Some("text") {
                                if let Some(text) = block.get("text").and_then(Value::as_str) {
                                    self.delta_accumulator.push_str(text);
                                    if tx
                                        .send(ChatStreamEvent::Delta(text.to_string()))
                                        .await
                                        .is_err()
                                    {
                                        return Ok(true);
                                    }
                                }
                            }
                        }
                    }
                    let segment = content
                        .iter()
                        .filter_map(extract_text_block)
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    if !segment.trim().is_empty()
                        && tx
                            .send(ChatStreamEvent::Narration { text: segment })
                            .await
                            .is_err()
                    {
                        return Ok(true);
                    }
                    for block in content {
                        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                            let id = block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            let name = block
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            if id.is_empty() || name.is_empty() {
                                tracing::warn!(
                                    has_id = !id.is_empty(),
                                    has_name = !name.is_empty(),
                                    "claude tool_use block missing id or name; skipping tool call"
                                );
                                continue;
                            }
                            let input = block.get("input").cloned().unwrap_or(Value::Null);
                            if tx
                                .send(ChatStreamEvent::ToolCall { id, name, input })
                                .await
                                .is_err()
                            {
                                return Ok(true);
                            }
                        }
                    }
                }
            }
            "user" => {
                if let Some(content) = value
                    .get("message")
                    .and_then(|message| message.get("content"))
                    .and_then(Value::as_array)
                {
                    for block in content {
                        if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                            continue;
                        }
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        if tool_use_id.is_empty() {
                            tracing::warn!(
                                "claude tool_result block missing tool_use_id; skipping tool result"
                            );
                            continue;
                        }
                        let output = extract_tool_result_content(block.get("content"));
                        let is_error = block
                            .get("is_error")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        if tx
                            .send(ChatStreamEvent::ToolResult {
                                tool_use_id,
                                output,
                                is_error,
                                input: None,
                            })
                            .await
                            .is_err()
                        {
                            return Ok(true);
                        }
                    }
                }
            }
            "result" => {
                let mut usage = extract_claude_usage(value);
                // `total_cost_usd` already includes subagents; their tokens do not.
                let subagents = std::mem::take(&mut self.subagent_usage);
                usage.input_tokens += subagents.input_tokens;
                usage.cached_input_tokens += subagents.cached_input_tokens;
                usage.cache_creation_input_tokens += subagents.cache_creation_input_tokens;
                usage.output_tokens += subagents.output_tokens;
                usage.total_tokens += subagents.total_tokens;
                if usage.duration_ms > 0 || usage.total_tokens > 0 || usage.cost_microusd.is_some()
                {
                    self.final_usage = Some(usage);
                }
                match value
                    .get("subtype")
                    .and_then(Value::as_str)
                    .unwrap_or("success")
                {
                    "success" if value.get("is_error").and_then(Value::as_bool) != Some(true) => {
                        // When the caller passed `outputFormat: { type: "json_schema", ... }`,
                        // the Agent SDK validates the final assistant output against the
                        // schema and emits the parsed object on `structured_output`. That
                        // value is authoritative. The free-text `result` field may still
                        // carry the raw model output and is only a presentation transcript.
                        let structured = value
                            .get("structured_output")
                            .filter(|value| !value.is_null());
                        if let Some(structured) = structured {
                            self.final_text = Some(structured.to_string());
                        } else {
                            self.final_text = value
                                .get("result")
                                .and_then(Value::as_str)
                                .map(ToString::to_string)
                                .or_else(|| {
                                    value.get("content").and_then(Value::as_array).map(|items| {
                                        items.iter().filter_map(extract_text_block).collect()
                                    })
                                });
                        }
                    }
                    subtype => {
                        // Non-success SDK result subtypes: error_during_execution,
                        // error_max_turns, error_max_budget_usd, error_max_structured_output_retries.
                        // Collect whatever diagnostic strings the SDK included.
                        let explicit_errors = value
                            .get("errors")
                            .and_then(Value::as_array)
                            .map(|items| {
                                items
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .collect::<Vec<_>>()
                                    .join("; ")
                            })
                            .filter(|joined| !joined.is_empty());
                        let error = value
                            .get("result")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                            .or(explicit_errors)
                            .or_else(|| {
                                value
                                    .get("error")
                                    .and_then(Value::as_str)
                                    .map(ToString::to_string)
                            })
                            .unwrap_or_else(|| format!("claude SDK returned subtype={subtype}"));
                        // Carry the structured fields (2.1.218+/2.1.223+) as
                        // the same `"key":value` tokens the `error` frame uses,
                        // so callers classify overload/rate-limit terminations
                        // without matching prose.
                        let mut parts = vec![format!("claude SDK {subtype}: {error}")];
                        if let Some(reason) = value.get("terminal_reason").and_then(Value::as_str) {
                            parts.push(format!(r#""terminal_reason":"{reason}""#));
                        }
                        if let Some(status) = value.get("api_error_status").and_then(Value::as_u64)
                        {
                            parts.push(format!(r#""status":{status}"#));
                        }
                        self.emitted_failure = true;
                        let _ = tx
                            .send(ChatStreamEvent::Failed {
                                error: parts.join(" "),
                            })
                            .await;
                        return Ok(true);
                    }
                }
            }
            "error" => {
                // Surface the structured `kind` / `status` fields so
                // `classify_provider_failure` can pick them up without
                // string matching.
                let message = value
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("claude SDK returned an error");
                let kind = value.get("kind").and_then(Value::as_str);
                let status = value.get("status").and_then(Value::as_u64);
                let mut parts: Vec<String> = vec![format!("claude SDK: {message}")];
                if let Some(kind) = kind {
                    parts.push(format!(r#""kind":"{kind}""#));
                }
                if let Some(status) = status {
                    parts.push(format!(r#""status":{status}"#));
                }
                self.emitted_failure = true;
                let _ = tx
                    .send(ChatStreamEvent::Failed {
                        error: parts.join(" "),
                    })
                    .await;
                return Ok(true);
            }
            _ => {}
        }
        Ok(false)
    }
}

/// Claude Code runs Agent-tool subagents in-process and reports their messages
/// on the same stream, tagged with the Agent call that owns them. They are the
/// subagent's own transcript, not this turn's text or tool calls; callers still
/// receive them as raw provider events.
pub(crate) fn is_subagent_message(value: &Value) -> bool {
    value
        .get("parent_tool_use_id")
        .is_some_and(|id| !id.is_null())
}

#[cfg(test)]
mod tests {
    use crate::ChatStreamEvent as StreamEvent;
    use serde_json::json as json_value;
    use tokio::sync::mpsc as stream_channel;

    use super::ClaudeStreamState as StreamState;
    #[tokio::test]
    async fn emits_streamed_thinking_separately_from_assistant_text() {
        let (tx, mut rx) = stream_channel::channel(4);
        let mut state = StreamState::default();
        state
            .handle_message(
                &json_value!({
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": "checking the source"
                        }
                    }
                }),
                &tx,
            )
            .await
            .unwrap();

        assert!(state.saw_stream_reasoning);
        assert!(matches!(
            rx.try_recv(),
            Ok(StreamEvent::ReasoningDelta(text)) if text == "checking the source"
        ));
    }

    #[tokio::test]
    async fn subagent_messages_are_not_the_parent_turn() {
        let (tx, mut rx) = stream_channel::channel(8);
        let mut state = StreamState::default();
        for message in [
            json_value!({
                "type": "assistant",
                "parent_tool_use_id": "toolu_agent",
                "message": {"content": [
                    {"type": "text", "text": "subagent narration"},
                    {"type": "tool_use", "id": "toolu_child", "name": "Bash", "input": {}}
                ]}
            }),
            json_value!({
                "type": "user",
                "parent_tool_use_id": "toolu_agent",
                "message": {"content": [
                    {"type": "tool_result", "tool_use_id": "toolu_child", "content": "done"}
                ]}
            }),
        ] {
            assert!(!state.handle_message(&message, &tx).await.unwrap());
        }
        assert!(rx.try_recv().is_err());

        state
            .handle_message(
                &json_value!({
                    "type": "assistant",
                    "parent_tool_use_id": null,
                    "message": {"content": [{"type": "text", "text": "parent"}]}
                }),
                &tx,
            )
            .await
            .unwrap();
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Delta(text)) if text == "parent"));
    }

    #[tokio::test]
    async fn subagent_tokens_join_the_turn_usage_once_per_api_call() {
        let (tx, _rx) = stream_channel::channel(8);
        let mut state = StreamState::default();
        let child_block = |block: serde_json::Value| {
            json_value!({
                "type": "assistant",
                "parent_tool_use_id": "toolu_agent",
                "message": {
                    "id": "msg_child",
                    "content": [block],
                    "usage": {"input_tokens": 3, "cache_read_input_tokens": 100, "output_tokens": 20}
                }
            })
        };
        // One API call reported on two content-block envelopes.
        for block in [
            json_value!({"type": "text", "text": "looking"}),
            json_value!({"type": "tool_use", "id": "toolu_child", "name": "Bash", "input": {}}),
        ] {
            state
                .handle_message(&child_block(block), &tx)
                .await
                .unwrap();
        }
        state
            .handle_message(
                &json_value!({
                    "type": "result",
                    "subtype": "success",
                    "result": "done",
                    "total_cost_usd": 0.5,
                    "usage": {"input_tokens": 4, "cache_read_input_tokens": 1000, "output_tokens": 50}
                }),
                &tx,
            )
            .await
            .unwrap();
        let usage = state.final_usage.expect("turn usage");
        assert_eq!(usage.input_tokens, 7);
        assert_eq!(usage.cached_input_tokens, 1100);
        assert_eq!(usage.output_tokens, 70);
        assert_eq!(usage.total_tokens, 1177);
        assert_eq!(usage.cost_microusd, Some(500_000));
    }

    #[tokio::test]
    async fn success_subtype_with_is_error_is_a_failure() {
        let (tx, mut rx) = stream_channel::channel(4);
        let mut state = StreamState::default();
        assert!(
            state
                .handle_message(
                    &json_value!({
                        "type": "result", "subtype": "success", "is_error": true,
                        "result": "API Error: service unavailable", "api_error_status": 503
                    }),
                    &tx
                )
                .await
                .unwrap()
        );
        assert!(state.emitted_failure);
        assert!(state.final_text.is_none());
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Failed { error })
            if error.contains("service unavailable") && error.contains(r#""status":503"#)));
    }

    #[tokio::test]
    async fn treats_assistant_api_error_as_a_terminal_failure() {
        let (tx, mut rx) = stream_channel::channel(4);
        let mut state = StreamState::default();
        let stop_reading = state
            .handle_message(
                &json_value!({
                    "type": "assistant",
                    "error": "rate_limit",
                    "apiErrorStatus": 429,
                    "message": {
                        "content": [{
                            "type": "text",
                            "text": "You've hit your monthly spend limit"
                        }]
                    }
                }),
                &tx,
            )
            .await
            .unwrap();

        assert!(!stop_reading);
        assert!(state.emitted_failure);
        assert!(matches!(
            rx.try_recv(),
            Ok(StreamEvent::Failed { error })
                if error.contains("monthly spend limit")
                    && error.contains(r#""kind":"rate_limit""#)
                    && error.contains(r#""status":429"#)
        ));
    }
}
