//! The Responses-API client.
//!
//! [`ResponsesClient`] speaks OpenAI's Responses API — the item-based
//! `input`/`output` wire shape — against either the platform
//! `/v1/responses` endpoint (static bearer key) or the ChatGPT Codex
//! backend (`codex-auth-file:` key pointing at a Codex CLI `auth.json`,
//! whose OAuth access token is re-read per call and refreshed on 401).
//! The client is stream-first: `complete` drives the SSE stream and folds
//! it, because the Codex backend serves streaming requests only.

mod auth;
mod body;
mod stream;

use std::collections::HashMap;

use crate::core::ToolUse;
use serde_json::{json, Value};
use tracing::{debug, warn};

use auth::{parse_api_key, AuthSource};
use body::build_request_body;
use stream::byte_stream_to_events;

use crate::client::{build_http, redact_key, Completion, CompletionRequest, LlmClient, LlmError};
use crate::config::ProviderConfig;
use crate::sse::{CompletionEvent, EventStream};

use futures::StreamExt;

/// A client for the OpenAI Responses API, including the ChatGPT Codex
/// backend.
pub struct ResponsesClient {
    config: ProviderConfig,
    http: reqwest::Client,
    auth: AuthSource,
    /// Serializes token refreshes so concurrent 401s produce one grant, not a
    /// stampede of competing rewrites of `auth.json`.
    refresh_lock: tokio::sync::Mutex<()>,
}

impl ResponsesClient {
    /// Builds a client from provider configuration. The api key's shape
    /// selects the auth mode: plain value → static bearer,
    /// `codex-auth-file:<path>` → Codex CLI auth file with refresh.
    pub fn new(config: ProviderConfig) -> Self {
        let http = build_http(config.timeout_seconds);
        let auth = parse_api_key(&config.api_key);
        ResponsesClient {
            config,
            http,
            auth,
            refresh_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Returns the provider configuration this client uses.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// The current bearer token and optional ChatGPT account id.
    fn credentials(&self) -> Result<(String, Option<String>), LlmError> {
        match &self.auth {
            AuthSource::Static(key) => Ok((key.clone(), None)),
            AuthSource::CodexFile(file) => {
                let creds = file.load()?;
                Ok((creds.access_token, creds.account_id))
            }
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/responses", self.config.base_url.trim_end_matches('/'))
    }

    fn request_body(&self, ctx: &CompletionRequest<'_>) -> Value {
        build_request_body(
            &self.config.model,
            ctx.history,
            ctx.session_id,
            ctx.reasoning.as_ref(),
            ctx.tools,
            &ctx.tool_choice,
            self.config.max_output_tokens,
            true,
        )
    }

    /// POSTs the streaming request once with the given credentials.
    async fn post(
        &self,
        body: &Value,
        session_id: Option<&str>,
        token: &str,
        account_id: Option<&str>,
    ) -> Result<reqwest::Response, LlmError> {
        let mut req = self
            .http
            .post(self.endpoint())
            .bearer_auth(token)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", "codex_cli_rs")
            .header("Accept", "text/event-stream")
            .json(body);
        if let Some(acct) = account_id {
            req = req.header("chatgpt-account-id", acct);
        }
        if let Some(sid) = session_id {
            req = req.header("session_id", sid);
        }
        Ok(req.send().await?)
    }

    /// Opens the SSE response, refreshing an expired Codex token once.
    async fn open_stream(&self, ctx: &CompletionRequest<'_>) -> Result<EventStream, LlmError> {
        if self.config.api_key.is_empty() {
            return Err(LlmError::MissingApiKey);
        }
        let body = self.request_body(ctx);
        let (mut token, mut account_id) = self.credentials()?;
        debug!(
            model = %self.config.model,
            messages = ctx.history.len(),
            tools = ctx.tools.len(),
            "requesting streaming responses completion"
        );

        let mut response = self
            .post(&body, ctx.session_id, &token, account_id.as_deref())
            .await?;

        if matches!(
            response.status(),
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        ) {
            if let AuthSource::CodexFile(file) = &self.auth {
                // One refresh, one retry. The lock keeps concurrent turns
                // from racing the grant; the re-load afterwards picks up a
                // refresh another turn already performed.
                let _guard = self.refresh_lock.lock().await;
                let creds = match file.load() {
                    Ok(c) if c.access_token != token => c,
                    _ => {
                        warn!("codex access token rejected; running the refresh grant");
                        file.refresh(&self.http).await?
                    }
                };
                token = creds.access_token;
                account_id = creds.account_id;
                drop(_guard);
                response = self
                    .post(&body, ctx.session_id, &token, account_id.as_deref())
                    .await?;
            }
        }

        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(LlmError::Unauthorized);
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Server {
                status: status.as_u16(),
                body: redact_key(&body, &token),
            });
        }

        Ok(byte_stream_to_events(response.bytes_stream(), &token))
    }
}

#[async_trait::async_trait]
impl LlmClient for ResponsesClient {
    /// Produces a full completion by driving the stream and folding it — the
    /// Codex backend serves streaming responses only, so there is no separate
    /// non-streaming wire call to make.
    async fn complete(&self, ctx: CompletionRequest<'_>) -> Result<Completion, LlmError> {
        let mut events = self.stream(ctx).await?;
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut usage = None;
        let mut order: Vec<(String, String)> = Vec::new(); // (id, name)
        let mut args: HashMap<String, String> = HashMap::new();
        while let Some(event) = events.next().await {
            match event? {
                CompletionEvent::Content(d) => content.push_str(&d),
                CompletionEvent::Reasoning(d) => reasoning.push_str(&d),
                CompletionEvent::ToolCallStart { id, name } => {
                    order.push((id.clone(), name));
                    args.insert(id, String::new());
                }
                CompletionEvent::ToolCallArgsDelta { id, partial_json } => {
                    args.entry(id).or_default().push_str(&partial_json);
                }
                CompletionEvent::ToolCallEnd { .. } => {}
                CompletionEvent::Usage(u) => usage = Some(u),
            }
        }
        let tool_uses: Vec<ToolUse> = order
            .into_iter()
            .map(|(id, name)| {
                let raw = args.remove(&id).unwrap_or_default();
                let input: Value = serde_json::from_str(&raw).unwrap_or_else(|err| {
                    warn!("responses stream produced malformed tool-call arguments: {err}");
                    json!({})
                });
                ToolUse { id, name, input }
            })
            .collect();
        let reasoning = reasoning.trim_end().to_owned();
        Ok(Completion {
            content,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            tool_uses,
            usage,
        })
    }

    async fn stream(&self, ctx: CompletionRequest<'_>) -> Result<EventStream, LlmError> {
        self.open_stream(&ctx).await
    }

    /// Returns the exact streaming-shape JSON body this client would POST.
    fn debug_request_body(&self, ctx: &CompletionRequest<'_>) -> Option<Value> {
        Some(self.request_body(ctx))
    }
}

#[cfg(test)]
#[path = "body_tests.rs"]
mod body_tests;

#[cfg(test)]
#[path = "client_tests.rs"]
mod client_tests;
