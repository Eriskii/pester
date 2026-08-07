//! Responses-API SSE parsing.
//!
//! The Responses stream frames typed events (`response.output_text.delta`,
//! `response.function_call_arguments.delta`, …) rather than chat-completion
//! choice deltas. Function-call argument deltas are keyed by the *item* id
//! while the harness tracks calls by their `call_id`, so [`ResponsesState`]
//! carries the item→call mapping established by each
//! `response.output_item.added`.

use std::collections::HashMap;

use bytes::BytesMut;
use futures::stream::{Stream, StreamExt};
use crate::core::Usage;
use serde_json::Value;

use crate::client::LlmError;
use crate::client::redact_key;
use crate::sse::{find_event_boundary, CompletionEvent, EventStream};

/// Per-stream parser state: the `function_call` item-id → `call_id` mapping.
pub(crate) struct ResponsesState {
    calls: HashMap<String, String>,
}

impl ResponsesState {
    pub(crate) fn new() -> Self {
        ResponsesState {
            calls: HashMap::new(),
        }
    }
}

/// Parses one SSE event (the text between blank-line boundaries) into zero or
/// more [`CompletionEvent`]s.
///
/// Unknown event types parse to nothing — the stream carries bookkeeping
/// (`response.created`, `response.metadata`, moderation payloads) the harness
/// has no use for. `data: [DONE]` and comment lines are skipped. An `error`
/// event or a `response.failed` surfaces as an `Err` with the API key
/// scrubbed.
pub(crate) fn parse_sse_event(
    event: &str,
    state: &mut ResponsesState,
    api_key: &str,
) -> Vec<Result<CompletionEvent, LlmError>> {
    let data: String = event
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|l| l.trim_start())
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() || data == "[DONE]" {
        return Vec::new();
    }
    let Ok(v) = serde_json::from_str::<Value>(&data) else {
        return Vec::new();
    };
    let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

    match typ {
        "response.output_text.delta" => delta_of(&v)
            .map(|d| vec![Ok(CompletionEvent::Content(d))])
            .unwrap_or_default(),
        "response.reasoning_summary_text.delta" => delta_of(&v)
            .map(|d| vec![Ok(CompletionEvent::Reasoning(d))])
            .unwrap_or_default(),
        // A summary part closed; render the boundary as a blank line so
        // multi-part summaries stay readable in the persisted trace.
        "response.reasoning_summary_text.done" => {
            vec![Ok(CompletionEvent::Reasoning("\n\n".into()))]
        }
        "response.output_item.added" => {
            let Some(item) = function_call_item(&v) else {
                return Vec::new();
            };
            state
                .calls
                .insert(item.item_id.clone(), item.call_id.clone());
            vec![Ok(CompletionEvent::ToolCallStart {
                id: item.call_id,
                name: item.name,
            })]
        }
        "response.function_call_arguments.delta" => {
            let (Some(item_id), Some(delta)) = (
                v.get("item_id").and_then(|x| x.as_str()),
                v.get("delta").and_then(|x| x.as_str()),
            ) else {
                return Vec::new();
            };
            let Some(call_id) = state.calls.get(item_id) else {
                return Vec::new();
            };
            vec![Ok(CompletionEvent::ToolCallArgsDelta {
                id: call_id.clone(),
                partial_json: delta.to_owned(),
            })]
        }
        "response.output_item.done" => function_call_item(&v)
            .map(|item| vec![Ok(CompletionEvent::ToolCallEnd { id: item.call_id })])
            .unwrap_or_default(),
        "response.completed" => v
            .get("response")
            .and_then(|r| r.get("usage"))
            .map(|u| vec![Ok(CompletionEvent::Usage(usage_from_responses(u)))])
            .unwrap_or_default(),
        "response.failed" | "response.incomplete" => {
            let msg = v
                .get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("response failed with no error detail");
            vec![Err(LlmError::Malformed(redact_key(msg, api_key)))]
        }
        "error" => {
            let msg = v
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("provider sent an error event with no message");
            vec![Err(LlmError::Malformed(redact_key(msg, api_key)))]
        }
        _ => Vec::new(),
    }
}

/// The fields of a `function_call` output item, when the event carries one.
struct FunctionCallItem {
    item_id: String,
    call_id: String,
    name: String,
}

fn function_call_item(v: &Value) -> Option<FunctionCallItem> {
    let item = v.get("item")?;
    if item.get("type").and_then(|t| t.as_str()) != Some("function_call") {
        return None;
    }
    Some(FunctionCallItem {
        item_id: item.get("id").and_then(|x| x.as_str())?.to_owned(),
        call_id: item.get("call_id").and_then(|x| x.as_str())?.to_owned(),
        name: item
            .get("name")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_owned(),
    })
}

fn delta_of(v: &Value) -> Option<String> {
    v.get("delta").and_then(|d| d.as_str()).map(|s| s.to_owned())
}

/// Maps the Responses `usage` block into [`Usage`].
///
/// `prompt_tokens` is `input_tokens` as reported (cached reads included),
/// `cached_tokens` is `input_tokens_details.cached_tokens`. The backend
/// reports no cache writes and no cost.
pub(crate) fn usage_from_responses(u: &Value) -> Usage {
    let get = |path: &[&str]| -> u32 {
        let mut cur = u;
        for p in path {
            match cur.get(p) {
                Some(v) => cur = v,
                None => return 0,
            }
        }
        cur.as_u64().unwrap_or(0) as u32
    };
    Usage {
        prompt_tokens: get(&["input_tokens"]),
        completion_tokens: get(&["output_tokens"]),
        total_tokens: get(&["total_tokens"]),
        cached_tokens: get(&["input_tokens_details", "cached_tokens"]),
        cache_write_tokens: 0,
        cost_usd: None,
    }
}

/// Converts a raw byte stream into [`CompletionEvent`]s, buffering across
/// chunk boundaries exactly like the chat-completions reader but with the
/// Responses event vocabulary.
pub(crate) fn byte_stream_to_events<S>(byte_stream: S, api_key: &str) -> EventStream
where
    S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
{
    let api_key = api_key.to_owned();
    let stream = async_stream::try_stream! {
        let mut buf = BytesMut::new();
        let mut state = ResponsesState::new();
        let mut byte_stream = Box::pin(byte_stream);
        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = chunk_result?;
            buf.extend_from_slice(&chunk);
            while let Some((end, boundary_len)) = find_event_boundary(&buf) {
                let raw = buf.split_to(end + boundary_len);
                let event_bytes = &raw[..raw.len() - boundary_len];
                let event_str = std::str::from_utf8(event_bytes).map_err(|e| {
                    LlmError::Malformed(format!("invalid utf-8 in SSE event: {e}"))
                })?;
                for event in parse_sse_event(event_str, &mut state, &api_key) {
                    yield event?;
                }
            }
        }
    };
    Box::pin(stream)
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
