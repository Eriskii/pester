//! The day-memory tool: the agent's running note about today.

use std::sync::Arc;

use serde_json::{json, Value};

use super::{str_field, Tool, ToolCtx};
use crate::data::local_date;
use crate::daynote::DayNoteStore;

/// Overwrites the current user's rolling day note.
pub struct UpdateDayMemory {
    store: Arc<DayNoteStore>,
}

impl UpdateDayMemory {
    pub fn new(store: Arc<DayNoteStore>) -> Self {
        UpdateDayMemory { store }
    }
}

#[async_trait::async_trait]
impl Tool for UpdateDayMemory {
    fn name(&self) -> &'static str {
        "update_day_memory"
    }
    fn emoji(&self) -> &'static str {
        "📔"
    }
    fn description(&self) -> &'static str {
        "Overwrite my rolling day-memory for this conversation — my running record of today: what they ate and when, mood, health notes, events, promises. It stays in my context all day and is distilled into my permanent memory files (/memory/<name>/<date>.md) at midnight. I rewrite the whole note each time, keeping it current and complete."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The full replacement note for today."}
            },
            "required": ["content"]
        })
    }
    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, String> {
        let content = str_field(&input, "content")?;
        let today = local_date();
        self.store.set(ctx.user.id, content.clone(), &today);
        Ok(json!({"updated": today, "chars": content.len()}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::UserInfo;

    #[tokio::test]
    async fn update_writes_todays_note_for_the_ctx_user() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(DayNoteStore::new(dir.path().join("daynotes.json")));
        let tool = UpdateDayMemory::new(Arc::clone(&store));
        let ctx = ToolCtx {
            user: UserInfo {
                id: 9,
                name: "julia".into(),
            },
        };
        tool.call(json!({"content": "ate toast at 8:15"}), &ctx)
            .await
            .unwrap();
        assert_eq!(
            store.today(9, &local_date()),
            Some("ate toast at 8:15".into())
        );
    }
}
