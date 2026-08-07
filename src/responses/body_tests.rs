//! Tests pinning the Responses-API request-body wire shape.

use crate::core::{Message, ToolUse};
use serde_json::json;

use super::body::build_request_body;
use crate::client::{ReasoningConfig, ReasoningEffort, ToolChoice, ToolDef};

fn build(history: &[Message]) -> serde_json::Value {
    build_request_body(
        "gpt-5.6-luna",
        history,
        None,
        None,
        &[],
        &ToolChoice::Auto,
        None,
        true,
    )
}

#[test]
fn leading_system_message_becomes_instructions() {
    let history = vec![Message::system("be terse"), Message::user("hi")];
    let body = build(&history);
    assert_eq!(body["instructions"], json!("be terse"));
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], json!("user"));
    assert_eq!(
        input[0]["content"],
        json!([{"type": "input_text", "text": "hi"}])
    );
}

#[test]
fn user_and_assistant_messages_map_to_items() {
    let history = vec![Message::user("hi"), Message::assistant("hello")];
    let body = build(&history);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[1]["type"], json!("message"));
    assert_eq!(input[1]["role"], json!("assistant"));
    assert_eq!(
        input[1]["content"],
        json!([{"type": "output_text", "text": "hello"}])
    );
}

#[test]
fn assistant_tool_uses_become_function_call_items() {
    let history = vec![
        Message::user("weather?"),
        Message::assistant_with_tools(
            "checking",
            vec![ToolUse {
                id: "call_1".into(),
                name: "weather".into(),
                input: json!({"location": "Osaka"}),
            }],
        ),
        Message::tool_result(r#"{"temperature_c": 20}"#, "call_1"),
    ];
    let body = build(&history);
    let input = body["input"].as_array().unwrap();
    // user message, assistant text message, function_call, function_call_output
    assert_eq!(input.len(), 4);
    assert_eq!(input[2]["type"], json!("function_call"));
    assert_eq!(input[2]["call_id"], json!("call_1"));
    assert_eq!(input[2]["name"], json!("weather"));
    assert_eq!(input[2]["arguments"], json!(r#"{"location":"Osaka"}"#));
    assert_eq!(input[3]["type"], json!("function_call_output"));
    assert_eq!(input[3]["call_id"], json!("call_1"));
    assert_eq!(input[3]["output"], json!(r#"{"temperature_c": 20}"#));
}

#[test]
fn tool_only_assistant_turn_emits_no_message_item() {
    let history = vec![Message::assistant_with_tools(
        "",
        vec![ToolUse {
            id: "call_2".into(),
            name: "additional_info".into(),
            input: json!({}),
        }],
    )];
    let body = build(&history);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], json!("function_call"));
}

#[test]
fn tools_declare_flattened() {
    let tools = vec![ToolDef {
        name: "weather".into(),
        description: "get weather".into(),
        parameters_schema: json!({"type": "object"}),
    }];
    let body = build_request_body(
        "m",
        &[Message::user("hi")],
        None,
        None,
        &tools,
        &ToolChoice::Auto,
        None,
        true,
    );
    assert_eq!(
        body["tools"],
        json!([{
            "type": "function",
            "name": "weather",
            "description": "get weather",
            "parameters": {"type": "object"},
        }])
    );
    // Auto stays implicit.
    assert!(body.get("tool_choice").is_none());
}

#[test]
fn tool_choice_maps() {
    let tools = vec![ToolDef {
        name: "t".into(),
        description: String::new(),
        parameters_schema: json!({}),
    }];
    let choice = |c: &ToolChoice| {
        build_request_body("m", &[], None, None, &tools, c, None, true)
            .get("tool_choice")
            .cloned()
    };
    assert_eq!(choice(&ToolChoice::Required), Some(json!("required")));
    assert_eq!(choice(&ToolChoice::None), Some(json!("none")));
    assert_eq!(
        choice(&ToolChoice::Tool("t".into())),
        Some(json!({"type": "function", "name": "t"}))
    );
}

#[test]
fn absent_reasoning_requests_effort_none() {
    let body = build(&[Message::user("hi")]);
    assert_eq!(body["reasoning"], json!({"effort": "none", "summary": "auto"}));
}

#[test]
fn reasoning_efforts_map_with_minimal_clamped_to_low() {
    let effort = |e: ReasoningEffort| {
        build_request_body(
            "m",
            &[],
            None,
            Some(&ReasoningConfig::with_effort(e)),
            &[],
            &ToolChoice::Auto,
            None,
            true,
        )["reasoning"]["effort"]
            .clone()
    };
    assert_eq!(effort(ReasoningEffort::Xhigh), json!("xhigh"));
    assert_eq!(effort(ReasoningEffort::High), json!("high"));
    assert_eq!(effort(ReasoningEffort::Medium), json!("medium"));
    assert_eq!(effort(ReasoningEffort::Low), json!("low"));
    // The Codex backend rejects "minimal"; the nearest supported level is low.
    assert_eq!(effort(ReasoningEffort::Minimal), json!("low"));
    assert_eq!(effort(ReasoningEffort::None), json!("none"));
}

#[test]
fn session_id_rides_as_prompt_cache_key() {
    let body = build_request_body(
        "m",
        &[],
        Some("conv-123"),
        None,
        &[],
        &ToolChoice::Auto,
        None,
        true,
    );
    assert_eq!(body["prompt_cache_key"], json!("conv-123"));
}

#[test]
fn store_false_and_stream_flag() {
    let body = build(&[]);
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["stream"], json!(true));
    let body = build_request_body("m", &[], None, None, &[], &ToolChoice::Auto, None, false);
    assert_eq!(body["stream"], json!(false));
}

#[test]
fn max_output_tokens_rides_when_set() {
    let body = build_request_body(
        "m",
        &[],
        None,
        None,
        &[],
        &ToolChoice::Auto,
        Some(4096),
        true,
    );
    assert_eq!(body["max_output_tokens"], json!(4096));
    let body = build(&[]);
    assert!(body.get("max_output_tokens").is_none());
}

#[test]
fn non_leading_system_message_maps_to_user_item() {
    let history = vec![
        Message::system("sys"),
        Message::user("hi"),
        Message::system("late note"),
    ];
    let body = build(&history);
    assert_eq!(body["instructions"], json!("sys"));
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[1]["role"], json!("user"));
    assert_eq!(
        input[1]["content"],
        json!([{"type": "input_text", "text": "late note"}])
    );
}
