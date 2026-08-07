//! Tests pinning the Responses-API SSE event parsing.

use crate::sse::CompletionEvent;

use super::{parse_sse_event, ResponsesState};

fn parse_all(events: &[&str]) -> Vec<CompletionEvent> {
    let mut state = ResponsesState::new();
    let mut out = Vec::new();
    for e in events {
        for r in parse_sse_event(e, &mut state, "key") {
            out.push(r.expect("event parses"));
        }
    }
    out
}

#[test]
fn output_text_delta_yields_content() {
    let events = parse_all(&[
        r#"data: {"type":"response.output_text.delta","delta":"Hel"}"#,
        r#"data: {"type":"response.output_text.delta","delta":"lo"}"#,
    ]);
    assert_eq!(
        events,
        vec![
            CompletionEvent::Content("Hel".into()),
            CompletionEvent::Content("lo".into()),
        ]
    );
}

#[test]
fn reasoning_summary_delta_yields_reasoning() {
    let events = parse_all(&[
        r#"data: {"type":"response.reasoning_summary_text.delta","delta":"**Drafting**"}"#,
        r#"data: {"type":"response.reasoning_summary_text.done","text":"**Drafting**"}"#,
    ]);
    assert_eq!(
        events,
        vec![
            CompletionEvent::Reasoning("**Drafting**".into()),
            // Part boundary renders as a blank line so multi-part summaries
            // stay readable in the persisted trace.
            CompletionEvent::Reasoning("\n\n".into()),
        ]
    );
}

#[test]
fn function_call_lifecycle_maps_item_id_to_call_id() {
    let events = parse_all(&[
        r#"data: {"type":"response.output_item.added","item":{"id":"fc_abc","type":"function_call","call_id":"call_1","name":"weather","arguments":""}}"#,
        r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_abc","delta":"{\"loc\""}"#,
        r#"data: {"type":"response.function_call_arguments.delta","item_id":"fc_abc","delta":":\"Osaka\"}"}"#,
        r#"data: {"type":"response.output_item.done","item":{"id":"fc_abc","type":"function_call","call_id":"call_1","name":"weather","arguments":"{\"loc\":\"Osaka\"}"}}"#,
    ]);
    assert_eq!(
        events,
        vec![
            CompletionEvent::ToolCallStart {
                id: "call_1".into(),
                name: "weather".into(),
            },
            CompletionEvent::ToolCallArgsDelta {
                id: "call_1".into(),
                partial_json: r#"{"loc""#.into(),
            },
            CompletionEvent::ToolCallArgsDelta {
                id: "call_1".into(),
                partial_json: r#":"Osaka"}"#.into(),
            },
            CompletionEvent::ToolCallEnd { id: "call_1".into() },
        ]
    );
}

#[test]
fn non_function_output_items_are_ignored() {
    let events = parse_all(&[
        r#"data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message"}}"#,
        r#"data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message"}}"#,
        r#"data: {"type":"response.metadata","metadata":{}}"#,
    ]);
    assert!(events.is_empty());
}

#[test]
fn completed_yields_usage() {
    let events = parse_all(&[
        r#"data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":40},"output_tokens":25,"output_tokens_details":{"reasoning_tokens":10},"total_tokens":125}}}"#,
    ]);
    assert_eq!(events.len(), 1);
    let CompletionEvent::Usage(u) = &events[0] else {
        panic!("expected usage, got {:?}", events[0]);
    };
    assert_eq!(u.prompt_tokens, 100);
    assert_eq!(u.completion_tokens, 25);
    assert_eq!(u.total_tokens, 125);
    assert_eq!(u.cached_tokens, 40);
    assert_eq!(u.cache_write_tokens, 0);
    assert_eq!(u.cost_usd, None);
}

#[test]
fn error_event_surfaces_as_err_with_key_redacted() {
    let mut state = ResponsesState::new();
    let results = parse_sse_event(
        r#"data: {"type":"error","message":"bad key sk-secret"}"#,
        &mut state,
        "sk-secret",
    );
    assert_eq!(results.len(), 1);
    let err = results.into_iter().next().unwrap().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("<redacted>"), "got: {msg}");
    assert!(!msg.contains("sk-secret"));
}

#[test]
fn response_failed_surfaces_as_err() {
    let mut state = ResponsesState::new();
    let results = parse_sse_event(
        r#"data: {"type":"response.failed","response":{"error":{"message":"overloaded"}}}"#,
        &mut state,
        "key",
    );
    assert_eq!(results.len(), 1);
    assert!(results.into_iter().next().unwrap().is_err());
}

#[test]
fn done_marker_and_unknown_events_yield_nothing() {
    let events = parse_all(&[
        "data: [DONE]",
        r#"data: {"type":"response.created","response":{}}"#,
        r#"data: {"type":"response.in_progress"}"#,
        ": keepalive comment",
    ]);
    assert!(events.is_empty());
}
