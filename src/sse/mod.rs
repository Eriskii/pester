//! SSE event vocabulary and stream construction for streaming completions.
//!
//! [`CompletionEvent`] is the unit a streaming LLM call yields. OpenAI-style
//! SSE frames each event as one or more `data: <json>` lines separated by
//! blank lines; a final `data: [DONE]` marks the end. The Responses-API
//! parser and byte-stream reader live in [`crate::responses`];
//! [`find_event_boundary`] is the shared blank-line splitter.

use futures::stream::{BoxStream, StreamExt};

use crate::client::{Completion, LlmError};
use crate::core::Usage;

/// One unit of a streaming completion.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionEvent {
    /// A delta of assistant content text.
    Content(String),
    /// A delta of assistant reasoning text.
    Reasoning(String),
    /// A new tool call is starting. Carries the wire id and the tool name.
    /// The dispatcher uses this to begin accumulating the call's partial
    /// JSON arguments and to publish a live "tool starting" indicator.
    ToolCallStart {
        /// Wire id for this tool call; matches subsequent delta and stop
        /// events.
        id: String,
        /// Name of the tool the model is calling.
        name: String,
    },
    /// A fragment of the JSON arguments for an in-flight tool call.
    ///
    /// Both providers stream tool arguments as a concatenation of partial
    /// JSON strings. The dispatcher accumulates them per `id` and parses on
    /// the matching `ToolCallEnd`.
    ToolCallArgsDelta {
        /// Wire id of the in-flight tool call.
        id: String,
        /// One fragment of the call's JSON arguments.
        partial_json: String,
    },
    /// The tool call with this `id` is complete; arguments are no longer
    /// streaming. The dispatcher parses the accumulated JSON and runs (or
    /// short-circuits) the tool.
    ToolCallEnd {
        /// Wire id of the now-complete tool call.
        id: String,
    },
    /// The token and cost accounting for the call, reported once when the
    /// provider sends it (typically on the last SSE event).
    Usage(Usage),
}

/// A boxed stream of completion events with `LlmError` failures.
pub type EventStream = BoxStream<'static, Result<CompletionEvent, LlmError>>;

/// Wraps an already-finished [`Completion`] as a stream that yields its
/// components in one batch.
///
/// Used by the default `LlmClient::stream` impl so callers that only know how
/// to produce a full completion still expose a streaming surface. Tool calls
/// are replayed as a `ToolCallStart` / `ToolCallArgsDelta` (carrying the full
/// serialized JSON arguments) / `ToolCallEnd` triple per call, mirroring the
/// shape a real streaming provider would emit.
pub fn one_shot_stream(completion: Completion) -> EventStream {
    let mut events: Vec<Result<CompletionEvent, LlmError>> = Vec::new();
    if !completion.content.is_empty() {
        events.push(Ok(CompletionEvent::Content(completion.content)));
    }
    if let Some(r) = completion.reasoning {
        if !r.is_empty() {
            events.push(Ok(CompletionEvent::Reasoning(r)));
        }
    }
    for call in completion.tool_uses {
        events.push(Ok(CompletionEvent::ToolCallStart {
            id: call.id.clone(),
            name: call.name,
        }));
        events.push(Ok(CompletionEvent::ToolCallArgsDelta {
            id: call.id.clone(),
            partial_json: call.input.to_string(),
        }));
        events.push(Ok(CompletionEvent::ToolCallEnd { id: call.id }));
    }
    if let Some(u) = completion.usage {
        events.push(Ok(CompletionEvent::Usage(u)));
    }
    futures::stream::iter(events).boxed()
}

/// Finds the next event boundary in `buf`, returning its start offset and its
/// byte length.
///
/// Used by the Responses byte-stream reader, which delimits events with a
/// blank line. A blank line is either `\n\n` (two
/// bytes) or `\r\n\r\n` (four bytes): CRLF-normalizing proxies, such as
/// Cloudflare in front of a provider, deliver the four-byte form. The function
/// returns whichever boundary begins first so a mixed stream parses cleanly.
pub(crate) fn find_event_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let lf = buf.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2));
    let crlf = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| (p, 4));
    match (lf, crlf) {
        (Some(lf), Some(crlf)) => Some(if lf.0 <= crlf.0 { lf } else { crlf }),
        (Some(lf), None) => Some(lf),
        (None, Some(crlf)) => Some(crlf),
        (None, None) => None,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
