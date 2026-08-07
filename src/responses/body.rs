//! Responses-API request-body construction.
//!
//! [`build_request_body`] serializes a storage transcript into the item-based
//! `input` array the Responses API takes: leading `System` messages become the
//! top-level `instructions` string, assistant `tool_uses` become
//! `function_call` items, and `Tool` results become `function_call_output`
//! items — so the harness-minted `additional_info` inbox exchange round-trips
//! as a synthetic call/output pair, which the Codex backend accepts even for
//! tools absent from the declared list.

use crate::core::{Message, ToolUse};
use serde_json::{json, Value};

use crate::client::{ReasoningConfig, ReasoningEffort, ToolChoice, ToolDef};

/// Builds the Responses-API request body for a model and transcript.
///
/// The body always carries `store: false` — the harness owns the transcript
/// and replays it in full every turn, exactly like the chat-completions path;
/// nothing is parked server-side. `reasoning` always rides with
/// `summary: "auto"` so reasoning progress streams back as summary deltas; an
/// absent config maps to effort `none` (a true non-thinking turn, zero
/// reasoning tokens). `session_id` rides as `prompt_cache_key` so the backend
/// can reuse prefix work between a conversation's turns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_request_body(
    model: &str,
    history: &[Message],
    session_id: Option<&str>,
    reasoning: Option<&ReasoningConfig>,
    tools: &[ToolDef],
    tool_choice: &ToolChoice,
    max_output_tokens: Option<u32>,
    streaming: bool,
) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(model));

    // Leading System messages fold into `instructions`; the rest become items.
    let split = history
        .iter()
        .position(|m| !matches!(m, Message::System { .. }))
        .unwrap_or(history.len());
    let instructions: Vec<&str> = history[..split].iter().map(|m| m.content()).collect();
    if !instructions.is_empty() {
        body.insert("instructions".into(), json!(instructions.join("\n\n")));
    }

    let mut input: Vec<Value> = Vec::new();
    for m in &history[split..] {
        push_items(&mut input, m);
    }
    body.insert("input".into(), Value::Array(input));

    if !tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(tools.iter().map(tool_def_value).collect()),
        );
        if let Some(choice) = tool_choice_value(tool_choice) {
            body.insert("tool_choice".into(), choice);
        }
    }

    body.insert(
        "reasoning".into(),
        json!({"effort": wire_effort(reasoning), "summary": "auto"}),
    );

    if let Some(cap) = max_output_tokens {
        body.insert("max_output_tokens".into(), json!(cap));
    }
    if let Some(s) = session_id {
        body.insert("prompt_cache_key".into(), json!(s));
    }

    body.insert("store".into(), json!(false));
    body.insert("include".into(), json!([]));
    body.insert("stream".into(), json!(streaming));

    Value::Object(body)
}

/// Appends the input items one storage [`Message`] expands to.
///
/// A `User` (or non-leading `System`) message is one `input_text` item. An
/// `Assistant` message is an `output_text` message item when it carries text,
/// followed by one `function_call` item per tool_use. A `Tool` result is one
/// `function_call_output` item keyed by its `tool_call_id`.
fn push_items(input: &mut Vec<Value>, m: &Message) {
    match m {
        Message::User {
            content, images, ..
        } => {
            let mut parts = vec![json!({"type": "input_text", "text": content})];
            for uri in images {
                parts.push(json!({"type": "input_image", "image_url": uri}));
            }
            input.push(json!({
                "role": "user",
                "content": parts,
            }));
        }
        Message::System { content, .. } => {
            input.push(json!({
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            }));
        }
        Message::Assistant {
            content, tool_uses, ..
        } => {
            if !content.is_empty() {
                input.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }));
            }
            for t in tool_uses {
                input.push(function_call_item(t));
            }
        }
        Message::Tool {
            content,
            tool_call_id,
            ..
        } => {
            input.push(json!({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": content,
            }));
        }
    }
}

/// The `function_call` item for one stored [`ToolUse`]. The stored id is the
/// wire `call_id` — for live calls that is the id the stream reported; for
/// harness-minted exchanges it is whatever id the mint fabricated, which the
/// backend accepts as long as the output item matches.
fn function_call_item(t: &ToolUse) -> Value {
    json!({
        "type": "function_call",
        "call_id": t.id,
        "name": t.name,
        "arguments": t.input.to_string(),
    })
}

/// The flattened Responses-API tool declaration for one [`ToolDef`] — no
/// nested `function` wrapper, unlike chat completions.
fn tool_def_value(t: &ToolDef) -> Value {
    json!({
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters_schema,
    })
}

/// Maps a [`ToolChoice`] to the Responses `tool_choice` value, returning
/// `None` for the implicit default (`Auto`).
fn tool_choice_value(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => None,
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Tool(name) => Some(json!({"type": "function", "name": name})),
    }
}

/// The wire effort string for an optional [`ReasoningConfig`].
///
/// Absent config (and an explicit `None` effort) is `"none"` — the Codex
/// models' true no-thinking mode. `Minimal` clamps to `"low"`: the backend's
/// ladder is none/low/medium/high/xhigh with no `minimal` rung.
fn wire_effort(reasoning: Option<&ReasoningConfig>) -> &'static str {
    match reasoning.and_then(|r| r.effort) {
        Some(ReasoningEffort::Xhigh) => "xhigh",
        Some(ReasoningEffort::High) => "high",
        Some(ReasoningEffort::Medium) => "medium",
        Some(ReasoningEffort::Low) | Some(ReasoningEffort::Minimal) => "low",
        Some(ReasoningEffort::None) | None => "none",
    }
}
