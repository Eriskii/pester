//! The tool registry: what the agent can do besides talk.
//!
//! A [`Tool`] is a name, a description, a JSON-Schema parameter shape, and an
//! async `call`. The [`Registry`] owns the set, renders the `${tools}` doc
//! for the system prompt, produces the wire [`ToolDef`]s, and dispatches a
//! call by name. Every dispatch returns a JSON string — errors ride as
//! `{"error": "..."}` so the model always gets a well-formed tool result.

pub mod fs;
pub mod memory;
pub mod remind;
pub mod time;

use std::sync::Arc;

use serde_json::{json, Value};

use crate::client::ToolDef;
use crate::core::UserInfo;

/// Ambient context a tool call runs under: which user's conversation the
/// call happens in. Tools that key state to a user (reminders) read it;
/// the rest ignore it.
#[derive(Debug, Clone)]
pub struct ToolCtx {
    /// The user this conversation is with.
    pub user: UserInfo,
}

/// One capability the model can call.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// The unique name the model calls this tool by.
    fn name(&self) -> &'static str;
    /// The tool's designated emoji — the Discord endpoint reacts to the
    /// triggering message with the emojis of every tool a turn used.
    fn emoji(&self) -> &'static str;
    /// A short description of what the tool does, shown to the model.
    fn description(&self) -> &'static str;
    /// The JSON Schema for the tool's arguments.
    fn schema(&self) -> Value;
    /// Runs the tool. `Ok` carries the result value; `Err` carries a
    /// human-readable error the dispatcher wraps as `{"error": ...}`.
    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, String>;
}

/// The set of tools declared to the model.
pub struct Registry {
    tools: Vec<Arc<dyn Tool>>,
}

impl Registry {
    /// Builds a registry over the given tools.
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Registry { tools }
    }

    /// The wire declarations for every registered tool.
    pub fn defs(&self) -> Vec<ToolDef> {
        self.tools
            .iter()
            .map(|t| ToolDef {
                name: t.name().to_owned(),
                description: t.description().to_owned(),
                parameters_schema: t.schema(),
            })
            .collect()
    }

    /// The human-readable tool listing the system prompt substitutes for
    /// `${tools}`.
    pub fn doc(&self) -> String {
        let mut out = String::from("Available tools:\n");
        for t in &self.tools {
            out.push_str(&format!(
                "- {} {}: {}\n",
                t.emoji(),
                t.name(),
                t.description()
            ));
        }
        out
    }

    /// The designated emoji for a tool, when the name is known.
    pub fn emoji_for(&self, name: &str) -> Option<&'static str> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.emoji())
    }

    /// Runs the named tool and returns its result serialized as a JSON
    /// string. An unknown name or a tool error returns `{"error": ...}`.
    pub async fn dispatch(&self, name: &str, input: Value, ctx: &ToolCtx) -> String {
        let Some(tool) = self.tools.iter().find(|t| t.name() == name) else {
            return json!({"error": format!("unknown tool: {name}")}).to_string();
        };
        match tool.call(input, ctx).await {
            Ok(v) => v.to_string(),
            Err(e) => json!({"error": e}).to_string(),
        }
    }
}

/// Reads a required string field off a tool input.
pub(crate) fn str_field(input: &Value, name: &str) -> Result<String, String> {
    input
        .get(name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| format!("missing required string field `{name}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolCtx {
        ToolCtx {
            user: crate::core::UserInfo {
                id: 7,
                name: "tester".into(),
            },
        }
    }

    struct Echo;

    #[async_trait::async_trait]
    impl Tool for Echo {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn emoji(&self) -> &'static str {
            "🔧"
        }
        fn description(&self) -> &'static str {
            "echoes its input"
        }
        fn schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn dispatch_runs_the_named_tool() {
        let reg = Registry::new(vec![Arc::new(Echo)]);
        let out = reg.dispatch("echo", json!({"a": 1}), &ctx()).await;
        assert_eq!(out, r#"{"a":1}"#);
    }

    #[tokio::test]
    async fn dispatch_wraps_unknown_tools_as_error_json() {
        let reg = Registry::new(vec![Arc::new(Echo)]);
        let out = reg.dispatch("nope", json!({}), &ctx()).await;
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"].as_str().unwrap().contains("unknown tool"));
    }

    #[test]
    fn doc_lists_every_tool_with_its_emoji() {
        let reg = Registry::new(vec![Arc::new(Echo)]);
        let doc = reg.doc();
        assert!(doc.contains("- 🔧 echo: echoes its input"));
    }

    #[test]
    fn emoji_for_resolves_known_names_only() {
        let reg = Registry::new(vec![Arc::new(Echo)]);
        assert_eq!(reg.emoji_for("echo"), Some("🔧"));
        assert_eq!(reg.emoji_for("nope"), None);
    }

    #[test]
    fn defs_carry_name_description_schema() {
        let reg = Registry::new(vec![Arc::new(Echo)]);
        let defs = reg.defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
        assert_eq!(defs[0].parameters_schema["type"], "object");
    }
}
