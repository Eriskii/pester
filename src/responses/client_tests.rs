//! End-to-end tests for [`ResponsesClient`] over a wiremock server.
//!
//! Test names carry the `http_` prefix so network-restricted sandboxes can
//! `--skip http_` (wiremock binds an ephemeral loopback port per test).

use crate::core::Message;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use super::ResponsesClient;
use crate::client::{CompletionRequest, LlmClient};
use crate::config::ProviderConfig;

fn config(base_url: String, api_key: &str) -> ProviderConfig {
    ProviderConfig {
        model: "gpt-5.6-luna".into(),
        base_url,
        api_key: api_key.into(),
        timeout_seconds: 10,
        max_output_tokens: None,
    }
}

const SSE_BODY: &str = concat!(
    "data: {\"type\":\"response.created\",\"response\":{}}\n\n",
    "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"**Thinking**\"}\n\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
    "data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
    "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"location\\\":\\\"Osaka\\\"}\"}\n\n",
    "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"weather\",\"arguments\":\"{}\"}}\n\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":50,\"output_tokens\":10,\"total_tokens\":60}}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn http_complete_folds_stream_into_completion() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer sk-static"))
        .and(header("openai-beta", "responses=experimental"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(SSE_BODY, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = ResponsesClient::new(config(server.uri(), "sk-static"));
    let history = vec![Message::system("sys"), Message::user("hi")];
    let completion = client
        .complete(CompletionRequest::new(&history))
        .await
        .expect("completes");

    assert_eq!(completion.content, "Hello");
    assert_eq!(completion.reasoning.as_deref(), Some("**Thinking**"));
    assert_eq!(completion.tool_uses.len(), 1);
    assert_eq!(completion.tool_uses[0].id, "call_9");
    assert_eq!(completion.tool_uses[0].name, "weather");
    assert_eq!(
        completion.tool_uses[0].input,
        serde_json::json!({"location": "Osaka"})
    );
    let usage = completion.usage.expect("usage reported");
    assert_eq!(usage.prompt_tokens, 50);
    assert_eq!(usage.completion_tokens, 10);

    // The request body carried instructions and store: false.
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["instructions"], serde_json::json!("sys"));
    assert_eq!(body["store"], serde_json::json!(false));
    assert_eq!(body["stream"], serde_json::json!(true));
}

#[tokio::test]
async fn http_expired_codex_token_refreshes_and_retries() {
    let server = MockServer::start().await;

    // First call with the stale token: 401. Second call with the refreshed
    // token: the SSE reply.
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer tok-old"))
        .respond_with(ResponseTemplate::new(401).set_body_string("token_expired"))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer tok-new"))
        .and(header("chatgpt-account-id", "acct-1"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\ndata: [DONE]\n\n",
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(|req: &Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
            assert_eq!(body["grant_type"], "refresh_token");
            assert_eq!(body["refresh_token"], "ref-1");
            ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "tok-new",
            }))
        })
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let auth_path = dir.path().join("auth.json");
    std::fs::write(
        &auth_path,
        serde_json::json!({
            "tokens": {
                "access_token": "tok-old",
                "refresh_token": "ref-1",
                "account_id": "acct-1",
            },
        })
        .to_string(),
    )
    .unwrap();

    let api_key = format!("codex-auth-file:{}", auth_path.display());
    let mut client = ResponsesClient::new(config(server.uri(), &api_key));
    // Point the refresh grant at the mock server.
    if let super::auth::AuthSource::CodexFile(file) = &mut client.auth {
        file.set_token_url(format!("{}/oauth/token", server.uri()));
    }

    let history = vec![Message::user("hi")];
    let completion = client
        .complete(CompletionRequest::new(&history))
        .await
        .expect("retry after refresh succeeds");
    assert_eq!(completion.content, "ok");

    // The refreshed token landed back in the auth file.
    let on_disk: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&auth_path).unwrap()).unwrap();
    assert_eq!(on_disk["tokens"]["access_token"], "tok-new");
}

#[tokio::test]
async fn http_static_key_401_is_unauthorized_without_refresh() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;

    let client = ResponsesClient::new(config(server.uri(), "sk-bad"));
    let history = vec![Message::user("hi")];
    let err = client
        .complete(CompletionRequest::new(&history))
        .await
        .unwrap_err();
    assert!(matches!(err, crate::client::LlmError::Unauthorized));
}
