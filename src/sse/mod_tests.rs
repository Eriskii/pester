//! Tests for event-boundary detection and one-shot replay.

use super::*;
use futures::StreamExt;

// --- find_event_boundary (LF and CRLF) ---

#[test]
fn find_event_boundary_locates_lf_boundary() {
    let buf = b"data: a\n\ndata: b\n\n";
    let (pos, len) = find_event_boundary(buf).unwrap();
    assert_eq!(&buf[..pos], b"data: a");
    assert_eq!(len, 2);
}

#[test]
fn find_event_boundary_locates_crlf_boundary() {
    // CRLF-normalizing proxies (Cloudflare in front of providers) deliver
    // `\r\n\r\n`-delimited events. The boundary is four bytes wide.
    let buf = b"data: a\r\n\r\ndata: b\r\n\r\n";
    let (pos, len) = find_event_boundary(buf).unwrap();
    assert_eq!(&buf[..pos], b"data: a");
    assert_eq!(len, 4);
}

#[test]
fn find_event_boundary_takes_whichever_appears_first() {
    // A mixed buffer with an LF boundary before a later CRLF one finds the
    // LF one first.
    let buf = b"x\n\ny\r\n\r\n";
    let (pos, len) = find_event_boundary(buf).unwrap();
    assert_eq!(pos, 1);
    assert_eq!(len, 2);
}

// --- one_shot_stream ---

#[tokio::test]
async fn one_shot_yields_content_reasoning_and_usage_in_order() {
    let completion = Completion {
        content: "the answer".into(),
        reasoning: Some("thinking".into()),
        tool_uses: Vec::new(),
        usage: Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
            cached_tokens: 0,
            cache_write_tokens: 0,
            cost_usd: Some(0.001),
        }),
    };
    let mut stream = one_shot_stream(completion);
    let mut got = Vec::new();
    while let Some(item) = stream.next().await {
        got.push(item.unwrap());
    }
    assert_eq!(got.len(), 3);
    assert!(matches!(got[0], CompletionEvent::Content(_)));
    assert!(matches!(got[1], CompletionEvent::Reasoning(_)));
    assert!(matches!(got[2], CompletionEvent::Usage(_)));
}

#[tokio::test]
async fn one_shot_skips_absent_or_empty_fields() {
    let completion = Completion {
        content: String::new(),
        reasoning: Some(String::new()),
        tool_uses: Vec::new(),
        usage: None,
    };
    let mut stream = one_shot_stream(completion);
    assert!(stream.next().await.is_none());
}
