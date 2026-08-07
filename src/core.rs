//! Chat message types and token accounting.
//!
//! A [`Message`] is the in-storage unit of the conversation: a typed enum
//! where each variant carries exactly the fields its role can hold. The
//! request-body builder projects these into the provider wire shape.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

/// The author role of a message within a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A message from the human user.
    User,
    /// A message from the assistant.
    Assistant,
    /// A system prompt or instruction.
    System,
    /// A tool invocation result attached to a prior assistant `tool_use`.
    Tool,
}

/// One tool invocation emitted by the assistant on an assistant turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolUse {
    /// Wire identifier the matching tool_result pins to.
    pub id: String,
    /// Registry name of the tool the assistant called.
    pub name: String,
    /// The JSON input the assistant produced for the call.
    #[serde(default)]
    pub input: Value,
}

impl ToolUse {
    /// Builds a tool use with the given id, name, and input.
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

/// A single message in a conversation, internally tagged by `role`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// A turn authored by the human user.
    User {
        /// The displayable text of the message.
        content: String,
        /// Images attached to the message, as `data:` URIs (already
        /// downscaled by the endpoint). The wire builder emits one
        /// `input_image` part per entry.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<String>,
        /// The moment the message is created, in UTC.
        #[serde(with = "time::serde::rfc3339")]
        created_at: OffsetDateTime,
    },
    /// A turn authored by the assistant, carrying any tool calls plus an
    /// optional reasoning trace.
    Assistant {
        /// The displayable reply text. Empty when the turn is purely a tool
        /// call with no accompanying text.
        content: String,
        /// Tool calls the assistant emitted on this turn.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_uses: Vec<ToolUse>,
        /// The assistant's reasoning text on this turn, when the provider
        /// returned any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        /// The moment the message is created, in UTC.
        #[serde(with = "time::serde::rfc3339")]
        created_at: OffsetDateTime,
    },
    /// A tool result, pinned to the assistant `tool_use` it answers.
    Tool {
        /// The result body, serialized as a string.
        content: String,
        /// The id of the `tool_use` this message answers.
        tool_call_id: String,
        /// The moment the message is created, in UTC.
        #[serde(with = "time::serde::rfc3339")]
        created_at: OffsetDateTime,
    },
    /// A system prompt or instruction.
    System {
        /// The instruction text.
        content: String,
        /// The moment the message is created, in UTC.
        #[serde(with = "time::serde::rfc3339")]
        created_at: OffsetDateTime,
    },
}

impl Message {
    /// Builds a plain text message with the given role and content.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        match role {
            Role::User => Message::user(content),
            Role::Assistant => Message::assistant(content),
            Role::System => Message::system(content),
            Role::Tool => Message::tool_result(content, String::new()),
        }
    }

    /// Builds a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Message::user_with_images(content, Vec::new())
    }

    /// Builds a user message carrying attached images as `data:` URIs.
    pub fn user_with_images(content: impl Into<String>, images: Vec<String>) -> Self {
        Message::User {
            content: content.into(),
            images,
            created_at: OffsetDateTime::now_utc(),
        }
    }

    /// Builds an assistant message with no tool calls.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message::Assistant {
            content: content.into(),
            tool_uses: Vec::new(),
            reasoning: None,
            created_at: OffsetDateTime::now_utc(),
        }
    }

    /// Builds an assistant message carrying the given tool_uses.
    pub fn assistant_with_tools(content: impl Into<String>, tool_uses: Vec<ToolUse>) -> Self {
        Message::Assistant {
            content: content.into(),
            tool_uses,
            reasoning: None,
            created_at: OffsetDateTime::now_utc(),
        }
    }

    /// Builds a tool-result message pinned to the given `tool_use` id.
    pub fn tool_result(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Message::Tool {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
            created_at: OffsetDateTime::now_utc(),
        }
    }

    /// Builds a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: content.into(),
            created_at: OffsetDateTime::now_utc(),
        }
    }

    /// Attaches the provided reasoning text to an assistant message and
    /// returns it. An empty `text` normalizes to `None`.
    pub fn with_reasoning(mut self, text: impl AsRef<str>) -> Self {
        let text = text.as_ref();
        let normalized = if text.is_empty() {
            None
        } else {
            Some(text.to_owned())
        };
        match &mut self {
            Message::Assistant { reasoning, .. } => *reasoning = normalized,
            _ => debug_assert!(
                normalized.is_none(),
                "with_reasoning called on a non-assistant message"
            ),
        }
        self
    }

    /// The author role of this message.
    pub fn role(&self) -> Role {
        match self {
            Message::User { .. } => Role::User,
            Message::Assistant { .. } => Role::Assistant,
            Message::Tool { .. } => Role::Tool,
            Message::System { .. } => Role::System,
        }
    }

    /// The displayable text of this message.
    pub fn content(&self) -> &str {
        match self {
            Message::User { content, .. }
            | Message::Assistant { content, .. }
            | Message::Tool { content, .. }
            | Message::System { content, .. } => content,
        }
    }

    /// The tool calls this message emitted. Empty for non-assistant messages.
    pub fn tool_uses(&self) -> &[ToolUse] {
        match self {
            Message::Assistant { tool_uses, .. } => tool_uses,
            _ => &[],
        }
    }

    /// The id of the `tool_use` this message answers.
    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Message::Tool { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }

    /// The assistant's reasoning trace, when present.
    pub fn reasoning(&self) -> Option<&str> {
        match self {
            Message::Assistant { reasoning, .. } => reasoning.as_deref(),
            _ => None,
        }
    }

    /// The images attached to this message. Empty for everything but a
    /// user message carrying attachments.
    pub fn images(&self) -> &[String] {
        match self {
            Message::User { images, .. } => images,
            _ => &[],
        }
    }
}

/// One Discord user the agent talks with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserInfo {
    /// The Discord user id — the key for their conversation and their
    /// reminders.
    pub id: u64,
    /// The username, refreshed on every message.
    pub name: String,
}

/// Token counts and cost for one completion, including cache accounting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens billed for the request.
    pub prompt_tokens: u32,
    /// Completion tokens generated in the reply.
    pub completion_tokens: u32,
    /// Total tokens for the request.
    pub total_tokens: u32,
    /// Tokens served from the provider cache.
    pub cached_tokens: u32,
    /// Tokens written to the cache on this request.
    pub cache_write_tokens: u32,
    /// Per-request USD cost as reported by the provider when available.
    pub cost_usd: Option<f64>,
}

impl Usage {
    /// Accumulates another usage record into this one. `cost_usd` is
    /// `Some(a + b)` only when BOTH sides report a cost — a missing cost
    /// means "unreported", not "free", so `None` is poisonous.
    pub fn add(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_tokens += other.cached_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.cost_usd = match (self.cost_usd, other.cost_usd) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trips_through_json() {
        let m = Message::assistant_with_tools(
            "thinking...",
            vec![ToolUse::new("t_1", "read_file", json!({"path": "notes.md"}))],
        );
        let json_str = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&json_str).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn role_is_the_serde_tag() {
        let v = serde_json::to_value(Message::user("hi")).unwrap();
        assert_eq!(v["role"], json!("user"));
        assert_eq!(v["content"], json!("hi"));
    }

    #[test]
    fn empty_tool_fields_omit_on_the_wire() {
        let m = Message::assistant("hi");
        let v: Value = serde_json::to_value(&m).unwrap();
        assert!(v.get("tool_uses").is_none());
        assert!(v.get("reasoning").is_none());
    }

    #[test]
    fn with_reasoning_normalizes_empty_to_none() {
        let m = Message::assistant("hi").with_reasoning("");
        assert!(m.reasoning().is_none());
    }

    #[test]
    fn usage_add_poisons_cost_on_missing_side() {
        let mut a = Usage {
            cost_usd: Some(0.01),
            ..Default::default()
        };
        a.add(&Usage::default());
        assert_eq!(a.cost_usd, None);
    }

    #[test]
    fn usage_add_sums_numeric_fields() {
        let mut a = Usage {
            prompt_tokens: 100,
            completion_tokens: 10,
            total_tokens: 110,
            cached_tokens: 40,
            cache_write_tokens: 5,
            cost_usd: None,
        };
        a.add(&Usage {
            prompt_tokens: 200,
            completion_tokens: 20,
            total_tokens: 220,
            cached_tokens: 80,
            cache_write_tokens: 7,
            cost_usd: None,
        });
        assert_eq!(a.prompt_tokens, 300);
        assert_eq!(a.completion_tokens, 30);
        assert_eq!(a.total_tokens, 330);
        assert_eq!(a.cached_tokens, 120);
        assert_eq!(a.cache_write_tokens, 12);
    }
}
