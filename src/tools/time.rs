//! The clock tool.

use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;

use super::{Tool, ToolCtx};
use crate::data::local_now;

/// Reports the current time.
pub struct GetTime;

#[async_trait::async_trait]
impl Tool for GetTime {
    fn name(&self) -> &'static str {
        "get_time"
    }
    fn emoji(&self) -> &'static str {
        "🕐"
    }
    fn description(&self) -> &'static str {
        "Get the current date and time (server local time)."
    }
    fn schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let now = local_now();
        let iso = now
            .format(&Rfc3339)
            .map_err(|e| format!("formatting time: {e}"))?;
        Ok(json!({
            "time": iso,
            "weekday": now.weekday().to_string(),
        }))
    }
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

    #[tokio::test]
    async fn get_time_returns_rfc3339_and_weekday() {
        let out = GetTime.call(json!({}), &ctx()).await.unwrap();
        let iso = out["time"].as_str().unwrap();
        assert!(time::OffsetDateTime::parse(iso, &Rfc3339).is_ok(), "{iso}");
        assert!(!out["weekday"].as_str().unwrap().is_empty());
    }
}
