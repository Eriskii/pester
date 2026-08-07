//! The agent's reminder tools: create, list, delete.
//!
//! All three share the [`ReminderStore`]. Creation always produces an
//! unlocked reminder; deletion refuses operator-locked entries, and the
//! error text says so plainly enough for the agent to relay.

use std::sync::Arc;

use serde_json::{json, Value};
use time::format_description::well_known::Rfc3339;

use super::{str_field, Tool, ToolCtx};
use crate::data::local_now;
use crate::reminders::{ReminderStore, Schedule};

/// Creates a reminder.
pub struct CreateReminder {
    store: Arc<ReminderStore>,
}

impl CreateReminder {
    pub fn new(store: Arc<ReminderStore>) -> Self {
        CreateReminder { store }
    }
}

#[async_trait::async_trait]
impl Tool for CreateReminder {
    fn name(&self) -> &'static str {
        "create_reminder"
    }
    fn emoji(&self) -> &'static str {
        "⏰"
    }
    fn description(&self) -> &'static str {
        "Schedule a reminder that wakes me later with a message. Exactly one of: `in_minutes` (one-off), `every_minutes` (recurring), or `daily_at` (\"HH:MM\" local, recurring). `times` caps how often a recurring one fires."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What I am reminded of when it fires."},
                "in_minutes": {"type": "integer", "minimum": 1, "description": "Fire once, this many minutes from now."},
                "every_minutes": {"type": "integer", "minimum": 1, "description": "Fire repeatedly at this interval."},
                "daily_at": {"type": "string", "description": "Fire every day at this local time, e.g. \"12:30\"."},
                "times": {"type": "integer", "minimum": 1, "description": "For recurring reminders: total number of fires. Omit for forever."}
            },
            "required": ["message"]
        })
    }
    async fn call(&self, input: Value, ctx: &ToolCtx) -> Result<Value, String> {
        let message = str_field(&input, "message")?;
        if message.trim().is_empty() {
            return Err("`message` is empty".into());
        }
        let in_minutes = input.get("in_minutes").and_then(|v| v.as_u64());
        let every_minutes = input.get("every_minutes").and_then(|v| v.as_u64());
        let daily_at = input.get("daily_at").and_then(|v| v.as_str());
        let times = input.get("times").and_then(|v| v.as_u64()).map(|n| n as u32);

        let given = [
            in_minutes.is_some(),
            every_minutes.is_some(),
            daily_at.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if given != 1 {
            return Err(
                "exactly one of `in_minutes`, `every_minutes`, or `daily_at` is required".into(),
            );
        }

        let (schedule, remaining) = if let Some(mins) = in_minutes {
            if times.is_some() {
                return Err("`times` does not apply to a one-off `in_minutes` reminder".into());
            }
            // A one-off is a recurrence of one: first fire in `mins`, then gone.
            (
                Schedule::EveryMinutes {
                    minutes: mins.max(1),
                },
                Some(1),
            )
        } else if let Some(mins) = every_minutes {
            (
                Schedule::EveryMinutes {
                    minutes: mins.max(1),
                },
                times,
            )
        } else {
            let (hour, minute) = parse_hhmm(daily_at.expect("checked above"))?;
            (Schedule::DailyAt { hour, minute }, times)
        };

        let reminder = self
            .store
            .create(message, schedule, remaining, ctx.user.id, local_now())
            .map_err(|e| e.to_string())?;
        Ok(json!({
            "created": {
                "id": reminder.id,
                "message": reminder.message,
                "schedule": reminder.schedule.describe(),
                "first_fire": reminder.next_fire.and_then(|t| t.format(&Rfc3339).ok()),
                "fires": reminder.remaining,
            }
        }))
    }
}

/// Lists every reminder, locked ones flagged.
pub struct ListReminders {
    store: Arc<ReminderStore>,
}

impl ListReminders {
    pub fn new(store: Arc<ReminderStore>) -> Self {
        ListReminders { store }
    }
}

#[async_trait::async_trait]
impl Tool for ListReminders {
    fn name(&self) -> &'static str {
        "list_reminders"
    }
    fn emoji(&self) -> &'static str {
        "📋"
    }
    fn description(&self) -> &'static str {
        "List every scheduled reminder. Locked ones are operator-set and cannot be deleted."
    }
    fn schema(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    async fn call(&self, _input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let reminders = self.store.list().map_err(|e| e.to_string())?;
        let entries: Vec<Value> = reminders
            .iter()
            .map(|r| {
                json!({
                    "id": r.id,
                    "message": r.message,
                    "locked": r.locked,
                    "schedule": r.schedule.describe(),
                    "next_fire": r.next_fire.and_then(|t| t.format(&Rfc3339).ok()),
                    "fires_left": r.remaining,
                })
            })
            .collect();
        Ok(json!({"reminders": entries}))
    }
}

/// Deletes an unlocked reminder.
pub struct DeleteReminder {
    store: Arc<ReminderStore>,
}

impl DeleteReminder {
    pub fn new(store: Arc<ReminderStore>) -> Self {
        DeleteReminder { store }
    }
}

#[async_trait::async_trait]
impl Tool for DeleteReminder {
    fn name(&self) -> &'static str {
        "delete_reminder"
    }
    fn emoji(&self) -> &'static str {
        "🗑️"
    }
    fn description(&self) -> &'static str {
        "Delete a reminder I created, by id. Operator-locked reminders cannot be deleted — not by me, not on request."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The reminder id to delete."}
            },
            "required": ["id"]
        })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let id = str_field(&input, "id")?;
        self.store.delete(&id).map_err(|e| e.to_string())?;
        Ok(json!({"deleted": id}))
    }
}

/// Marks an active reminder handled, ending its hourly nag.
pub struct SatisfyReminder {
    store: Arc<ReminderStore>,
}

impl SatisfyReminder {
    pub fn new(store: Arc<ReminderStore>) -> Self {
        SatisfyReminder { store }
    }
}

#[async_trait::async_trait]
impl Tool for SatisfyReminder {
    fn name(&self) -> &'static str {
        "satisfy_reminder"
    }
    fn emoji(&self) -> &'static str {
        "✅"
    }
    fn description(&self) -> &'static str {
        "Mark an active reminder as handled, by id. Until I call this, an active reminder re-fires every hour. I call it only once the reminder's purpose is genuinely fulfilled (e.g. she actually ate, with proof), not just acknowledged."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The reminder id to satisfy."}
            },
            "required": ["id"]
        })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let id = str_field(&input, "id")?;
        let outcome = self.store.satisfy(&id).map_err(|e| e.to_string())?;
        Ok(json!({"satisfied": id, "outcome": outcome}))
    }
}

/// Parses `"HH:MM"` into hour and minute.
fn parse_hhmm(s: &str) -> Result<(u8, u8), String> {
    let (h, m) = s
        .split_once(':')
        .ok_or_else(|| format!("`daily_at` must be \"HH:MM\", got {s:?}"))?;
    let hour: u8 = h
        .trim()
        .parse()
        .map_err(|_| format!("bad hour in {s:?}"))?;
    let minute: u8 = m
        .trim()
        .parse()
        .map_err(|_| format!("bad minute in {s:?}"))?;
    if hour > 23 || minute > 59 {
        return Err(format!("{s:?} is not a valid 24-hour time"));
    }
    Ok((hour, minute))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::core::UserInfo;

    fn ctx() -> ToolCtx {
        ToolCtx {
            user: UserInfo {
                id: 7,
                name: "sofia".into(),
            },
        }
    }

    fn tools() -> (
        tempfile::TempDir,
        CreateReminder,
        ListReminders,
        DeleteReminder,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ReminderStore::new(dir.path().join("reminders.json")));
        (
            dir,
            CreateReminder::new(Arc::clone(&store)),
            ListReminders::new(Arc::clone(&store)),
            DeleteReminder::new(store),
        )
    }

    #[tokio::test]
    async fn create_one_off_then_list_then_delete() {
        let (_d, create, list, delete) = tools();
        let out = create
            .call(json!({"message": "check the oven", "in_minutes": 20}), &ctx())
            .await
            .unwrap();
        let id = out["created"]["id"].as_str().unwrap().to_owned();
        assert_eq!(out["created"]["fires"], 1);

        let listed = list.call(json!({}), &ctx()).await.unwrap();
        assert_eq!(listed["reminders"][0]["message"], "check the oven");
        assert_eq!(listed["reminders"][0]["locked"], false);

        delete.call(json!({"id": id}), &ctx()).await.unwrap();
        let listed = list.call(json!({}), &ctx()).await.unwrap();
        assert_eq!(listed["reminders"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn create_daily_at_parses_the_time() {
        let (_d, create, list, _) = tools();
        create
            .call(json!({"message": "eat lunch", "daily_at": "12:30"}), &ctx())
            .await
            .unwrap();
        let listed = list.call(json!({}), &ctx()).await.unwrap();
        assert_eq!(listed["reminders"][0]["schedule"], "daily at 12:30");
    }

    #[tokio::test]
    async fn create_requires_exactly_one_schedule() {
        let (_d, create, _, _) = tools();
        let err = create.call(json!({"message": "x"}), &ctx()).await.unwrap_err();
        assert!(err.contains("exactly one"), "{err}");
        let err = create
            .call(json!({"message": "x", "in_minutes": 5, "daily_at": "10:00"}), &ctx())
            .await
            .unwrap_err();
        assert!(err.contains("exactly one"), "{err}");
    }

    #[tokio::test]
    async fn create_rejects_bad_times_of_day() {
        let (_d, create, _, _) = tools();
        for bad in ["25:00", "12:75", "noon", "12.30"] {
            let err = create
                .call(json!({"message": "x", "daily_at": bad}), &ctx())
                .await
                .unwrap_err();
            assert!(!err.is_empty(), "{bad} must be rejected");
        }
    }

    #[tokio::test]
    async fn delete_relays_the_locked_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reminders.json");
        std::fs::write(
            &path,
            json!([{
                "id": "lunch",
                "message": "eat",
                "locked": true,
                "schedule": {"type": "daily_at", "hour": 12, "minute": 30}
            }])
            .to_string(),
        )
        .unwrap();
        let delete = DeleteReminder::new(Arc::new(ReminderStore::new(path)));
        let err = delete.call(json!({"id": "lunch"}), &ctx()).await.unwrap_err();
        assert!(err.contains("locked"), "{err}");
        assert!(err.contains("cannot be deleted"), "{err}");
    }
}
