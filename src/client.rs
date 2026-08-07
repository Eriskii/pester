//! The provider client abstraction.
//!
//! [`LlmClient`] is the seam the rest of pester talks to: a transcript goes
//! in, an assistant reply comes out. The one real implementation is
//! [`crate::responses::ResponsesClient`], which speaks the OpenAI Responses
//! API (platform or ChatGPT Codex backend).

use std::sync::Arc;

use crate::core::{Message, ToolUse, Usage};
use serde::Serialize;
use serde_json::Value;

use crate::sse::{one_shot_stream, EventStream};

/// Errors that arise while requesting a completion.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The HTTP request never completes: DNS, connect, or timeout failure.
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// The provider rejects the bearer token (HTTP 401/403).
    #[error("provider rejected the API key (unauthorized)")]
    Unauthorized,

    /// The provider returns a non-success status outside the auth range.
    #[error("provider returned status {status}: {body}")]
    Server { status: u16, body: String },

    /// The response body is not the shape an OpenAI completion takes.
    #[error("malformed provider response: {0}")]
    Malformed(String),

    /// A real network call is attempted with no API key configured.
    #[error("no API key configured for the provider")]
    MissingApiKey,
}

/// How hard the model reasons before answering.
///
/// Mirrors OpenRouter's `reasoning.effort` levels. The variant serializes to
/// its lowercase name on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Maximum reasoning budget.
    Xhigh,
    /// High reasoning budget.
    High,
    /// Medium reasoning budget.
    Medium,
    /// Low reasoning budget.
    Low,
    /// Minimal reasoning budget.
    Minimal,
    /// No reasoning budget.
    None,
}

impl std::str::FromStr for ReasoningEffort {
    type Err = String;

    /// Parses the lowercase level names, case-insensitively. `off` is an
    /// alias for `none`. This is the canonical parser — the server flags
    /// and the leaves' `reasoning_default` field all resolve through it.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "xhigh" => Ok(ReasoningEffort::Xhigh),
            "high" => Ok(ReasoningEffort::High),
            "medium" => Ok(ReasoningEffort::Medium),
            "low" => Ok(ReasoningEffort::Low),
            "minimal" => Ok(ReasoningEffort::Minimal),
            "none" | "off" => Ok(ReasoningEffort::None),
            other => Err(format!(
                "unknown reasoning effort {other:?} (expected xhigh|high|medium|low|minimal|none)"
            )),
        }
    }
}

/// OpenRouter's `reasoning` request object.
///
/// `effort` selects a reasoning budget; `exclude` keeps reasoning out of the
/// response while still letting the model reason; `enabled` toggles reasoning
/// without an explicit budget. Each field is omitted from the wire when unset.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ReasoningConfig {
    /// The reasoning effort level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Whether to omit the reasoning text from the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    /// Whether reasoning is enabled without an explicit effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

impl ReasoningConfig {
    /// Builds a config that requests the given reasoning effort.
    pub fn with_effort(effort: ReasoningEffort) -> Self {
        ReasoningConfig {
            effort: Some(effort),
            ..Default::default()
        }
    }
}

/// One tool the model can call this turn.
///
/// `parameters_schema` is forwarded verbatim into the provider's tool
/// declarations: Anthropic's `input_schema`, OpenAI's `function.parameters`.
/// The harness reads each declaration off the tool registry and packages it
/// here.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    /// The unique name the model calls this tool by.
    pub name: String,
    /// A short, human-readable description of what the tool does.
    pub description: String,
    /// The JSON Schema for the argument the tool expects.
    pub parameters_schema: Value,
}

/// How the model picks which tool (if any) to call this turn.
///
/// `Auto` lets the model decide; `Required` forces *some* tool call but
/// leaves the choice to the model; `Tool(name)` pins the call to one
/// specific tool; `None` lists tools but forbids calling them this turn.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call a tool. The default.
    #[default]
    Auto,
    /// The model must call at least one tool from the declared list.
    Required,
    /// The model must call this specific tool.
    Tool(String),
    /// Tools are declared but the model is forbidden to call them.
    None,
}

/// The inputs to one completion call.
///
/// `history` is the transcript to continue. `session_id` is the sticky-routing
/// key for OpenRouter: passing a stable id keeps a multi-turn conversation on
/// one provider so its prompt cache stays warm. The daemon passes the
/// conversation id. `reasoning` opts the call into model reasoning. `tools`
/// declares the tools the model may call; an empty list disables tool use.
/// `tool_choice` directs the model's selection among the declared tools.
#[derive(Debug, Clone)]
pub struct CompletionRequest<'a> {
    /// The transcript to continue.
    pub history: &'a [Message],
    /// Sticky-routing key for OpenRouter; the daemon passes the conversation id.
    pub session_id: Option<&'a str>,
    /// Reasoning configuration; absent leaves reasoning off.
    pub reasoning: Option<ReasoningConfig>,
    /// Tools the model may call this turn. Empty disables tool use.
    pub tools: &'a [ToolDef],
    /// How the model chooses among declared tools.
    pub tool_choice: ToolChoice,
}

impl<'a> CompletionRequest<'a> {
    /// Builds a request for a transcript with no session id, no reasoning,
    /// and no tools declared.
    pub fn new(history: &'a [Message]) -> Self {
        CompletionRequest {
            history,
            session_id: None,
            reasoning: None,
            tools: &[],
            tool_choice: ToolChoice::Auto,
        }
    }

    /// Sets the sticky-routing session id.
    pub fn with_session_id(mut self, session_id: &'a str) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Requests reasoning at the given effort level.
    pub fn with_reasoning(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(ReasoningConfig::with_effort(effort));
        self
    }

    /// Declares the tools the model may call this turn.
    pub fn with_tools(mut self, tools: &'a [ToolDef]) -> Self {
        self.tools = tools;
        self
    }

    /// Sets how the model chooses among declared tools.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }
}

/// One assistant reply plus optional reasoning, tool calls, and token
/// accounting.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The assistant reply text (text-block content).
    pub content: String,
    /// The model's reasoning text, when the provider returns it.
    pub reasoning: Option<String>,
    /// Tool calls the assistant emitted on this turn. Empty when the model
    /// produced no tool calls.
    pub tool_uses: Vec<ToolUse>,
    /// Token usage for the call, absent when the provider does not report it.
    pub usage: Option<Usage>,
}

/// A client that turns a conversation transcript into an assistant reply.
///
/// Implementations supply `complete` for a one-shot reply. `stream` defaults to
/// driving `complete` to completion and emitting its three parts as one batch
/// via [`one_shot_stream`]; real providers override `stream` with a
/// token-by-token SSE feed so the daemon can forward chunks live to its
/// endpoints.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Produces an assistant reply for a conversation transcript.
    async fn complete(&self, ctx: CompletionRequest<'_>) -> Result<Completion, LlmError>;

    /// Produces an assistant reply as a stream of [`CompletionEvent`]s.
    ///
    /// The default impl collects the full reply via `complete` and wraps it as
    /// a single-batch stream. Implementations that speak a streaming protocol
    /// (such as OpenAI-style SSE) override this to yield content, reasoning,
    /// and usage as the provider sends them.
    async fn stream(&self, ctx: CompletionRequest<'_>) -> Result<EventStream, LlmError> {
        let completion = self.complete(ctx).await?;
        Ok(one_shot_stream(completion))
    }

    /// Returns the exact JSON request body this client would POST to the
    /// provider for `ctx` in its **streaming** shape — surfaced by the web
    /// UI for ground-truth inspection.
    ///
    /// Real providers override this; the default returns `None`.
    fn debug_request_body(&self, _ctx: &CompletionRequest<'_>) -> Option<Value> {
        None
    }
}

/// A shared, type-erased client handle that `Core` and friends hold.
pub type SharedLlmClient = Arc<dyn LlmClient>;

/// Builds a `reqwest::Client` with the configured request timeout.
///
/// The timeout bounds the whole request (connect plus read) so a stuck
/// provider connection can never hold the turn loop indefinitely. A builder
/// failure (which the timeout setting alone does not provoke) falls back to a
/// default client.
pub(crate) fn build_http(timeout_seconds: u64) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_seconds))
        .build()
        .unwrap_or_default()
}

/// Strips the API key from an error body so a key never leaks into logs.
pub(crate) fn redact_key(body: &str, key: &str) -> String {
    if key.is_empty() {
        return body.to_owned();
    }
    body.replace(key, "<redacted>")
}
