//! The reminder system: scheduled wake-ups that inject into the
//! conversation, and nag until satisfied.
//!
//! A [`Reminder`] is a message plus a [`Schedule`] — one-off, every-N-
//! minutes, daily, or weekly at a local time — with an optional
//! remaining-fire count. The store is `reminders.json` in the data root,
//! re-read on every operation so the operator can hand-edit it while
//! pester runs; entries the operator writes with `"locked": true` are
//! permanent: the agent's delete tool refuses them, so no amount of asking
//! makes them go away. Agent-created reminders are never locked.
//!
//! A fire makes the reminder **active**: unless the agent marks it handled
//! with its `satisfy_reminder` tool, it re-fires every
//! [`RETICK_MINUTES`] minutes — the nag loop. Satisfying clears the nag
//! until the next scheduled occurrence; a reminder with no future
//! occurrence retires once satisfied. The idle compactor checks for active
//! reminders before folding, so a conversation is never compacted and then
//! immediately re-reminded.
//!
//! [`spawn_scheduler`] ticks every 30 seconds and fires anything due: the
//! fire payload rides to the endpoint, which injects it into the
//! transcript as a synthetic `reminder` tool exchange and runs a turn —
//! the agent wakes, reads the reminder, and decides what to say. A
//! reminder that comes due while pester is down fires once at the next
//! startup tick.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::json;
use time::{Duration, OffsetDateTime, Time, Weekday};
use tracing::{info, warn};

use crate::data::local_now;
use crate::status::Status;
use crate::timestamp::format_prefix;

/// How long an unsatisfied active reminder waits before re-firing.
pub const RETICK_MINUTES: i64 = 60;

/// How a reminder repeats.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Schedule {
    /// Fires once at `next_fire`, then only reticks until satisfied.
    Once,
    /// Fires every `minutes` minutes.
    EveryMinutes {
        /// The recurrence period, in minutes.
        minutes: u64,
    },
    /// Fires every day at the given local time.
    DailyAt {
        /// Hour, 0–23, local time.
        hour: u8,
        /// Minute, 0–59.
        minute: u8,
    },
    /// Fires every week on the given day at the given local time.
    WeeklyAt {
        /// The weekday, lowercase English name (e.g. `"sunday"`).
        weekday: String,
        /// Hour, 0–23, local time.
        hour: u8,
        /// Minute, 0–59.
        minute: u8,
    },
}

impl Schedule {
    /// A short human-readable description, shown to the agent.
    pub fn describe(&self) -> String {
        match self {
            Schedule::Once => "one-off".into(),
            Schedule::EveryMinutes { minutes } => format!("every {minutes} minutes"),
            Schedule::DailyAt { hour, minute } => format!("daily at {hour:02}:{minute:02}"),
            Schedule::WeeklyAt {
                weekday,
                hour,
                minute,
            } => format!("weekly on {weekday} at {hour:02}:{minute:02}"),
        }
    }
}

/// One scheduled reminder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reminder {
    /// Unique id. Operator entries carry hand-written names; agent-created
    /// ones get a timestamp id.
    pub id: String,
    /// The message the agent wakes to.
    pub message: String,
    /// Whether the reminder is operator-locked. The agent cannot delete a
    /// locked reminder.
    #[serde(default)]
    pub locked: bool,
    /// The Discord user this reminder belongs to and fires at. Stamped
    /// automatically when the agent creates one mid-conversation; the
    /// operator writes it by hand on locked entries. Absent falls back to
    /// the endpoint's default target.
    #[serde(default)]
    pub user: Option<u64>,
    /// The firing schedule.
    pub schedule: Schedule,
    /// Scheduled fires left, for recurring schedules. `None` recurs
    /// forever. Reaching 0 ends the scheduled fires; the entry retires
    /// once satisfied.
    #[serde(default)]
    pub remaining: Option<u32>,
    /// The next *scheduled* fire moment. Absent in a hand-written entry
    /// (the scheduler fills it from the schedule on first sight) and on an
    /// entry with no scheduled fires left.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub next_fire: Option<OffsetDateTime>,
    /// Whether the reminder has fired and not yet been satisfied — the
    /// nagging state.
    #[serde(default)]
    pub active: bool,
    /// When the next unsatisfied re-fire happens, while active.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub next_retick: Option<OffsetDateTime>,
}

impl Reminder {
    /// Whether the entry has neither scheduled fires ahead nor an active
    /// nag — nothing left to do.
    fn retired(&self) -> bool {
        self.next_fire.is_none() && !self.active
    }
}

/// One fired reminder, ready for injection.
#[derive(Debug, Clone)]
pub struct Fired {
    /// The reminder it came from.
    pub reminder: Reminder,
    /// The JSON payload injected as the synthetic tool result.
    pub payload: String,
}

/// Errors from the reminder store.
#[derive(Debug, thiserror::Error)]
pub enum ReminderError {
    /// The store file exists but does not parse — mutations are refused so
    /// a hand-edit typo never wipes the operator's entries.
    #[error("reminders.json is malformed: {0}")]
    Malformed(String),
    /// An I/O failure touching the store file.
    #[error("reminder store I/O: {0}")]
    Io(#[from] std::io::Error),
    /// A delete aimed at a locked reminder.
    #[error("reminder `{0}` is locked by the operator and cannot be deleted")]
    Locked(String),
    /// A delete or satisfy aimed at an id that does not exist.
    #[error("no reminder with id `{0}`")]
    Unknown(String),
}

/// The on-disk reminder store. Every operation re-reads the file, so
/// operator hand-edits are picked up live.
pub struct ReminderStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl ReminderStore {
    /// Builds a store over the given file path.
    pub fn new(path: PathBuf) -> Self {
        ReminderStore {
            path,
            lock: Mutex::new(()),
        }
    }

    fn read(&self) -> Result<Vec<Reminder>, ReminderError> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| ReminderError::Malformed(e.to_string())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }

    fn write(&self, reminders: &[Reminder]) -> Result<(), ReminderError> {
        let bytes = serde_json::to_vec_pretty(reminders)
            .map_err(|e| ReminderError::Malformed(e.to_string()))?;
        std::fs::write(&self.path, bytes)?;
        Ok(())
    }

    /// Lists every reminder.
    pub fn list(&self) -> Result<Vec<Reminder>, ReminderError> {
        let _guard = self.lock.lock().unwrap();
        self.read()
    }

    /// Whether any active (fired, unsatisfied) reminder targets `user`.
    /// Untargeted reminders count for everyone.
    pub fn has_active(&self, user: u64) -> bool {
        self.list()
            .map(|rs| {
                rs.iter()
                    .any(|r| r.active && r.user.map(|u| u == user).unwrap_or(true))
            })
            .unwrap_or(false)
    }

    /// Creates an agent-authored (never locked) reminder, stamped with the
    /// user whose conversation created it, and returns it.
    pub fn create(
        &self,
        message: String,
        schedule: Schedule,
        remaining: Option<u32>,
        user: u64,
        now: OffsetDateTime,
    ) -> Result<Reminder, ReminderError> {
        let _guard = self.lock.lock().unwrap();
        let mut reminders = self.read()?;
        let reminder = Reminder {
            id: format!("r{}", now.unix_timestamp_nanos() / 1_000_000),
            message,
            locked: false,
            user: Some(user),
            remaining,
            next_fire: Some(first_fire(&schedule, now)),
            schedule,
            active: false,
            next_retick: None,
        };
        reminders.push(reminder.clone());
        self.write(&reminders)?;
        Ok(reminder)
    }

    /// Deletes a reminder by id. A locked reminder refuses.
    pub fn delete(&self, id: &str) -> Result<(), ReminderError> {
        let _guard = self.lock.lock().unwrap();
        let mut reminders = self.read()?;
        let Some(pos) = reminders.iter().position(|r| r.id == id) else {
            return Err(ReminderError::Unknown(id.to_owned()));
        };
        if reminders[pos].locked {
            return Err(ReminderError::Locked(id.to_owned()));
        }
        reminders.remove(pos);
        self.write(&reminders)?;
        Ok(())
    }

    /// Marks an active reminder handled: the hourly nag stops until the
    /// next scheduled occurrence. An entry with no scheduled fires left
    /// retires. Works on locked reminders too — satisfying is how the nag
    /// ends; only deletion is forbidden. Returns a description of what
    /// happens next.
    pub fn satisfy(&self, id: &str) -> Result<String, ReminderError> {
        let _guard = self.lock.lock().unwrap();
        let mut reminders = self.read()?;
        let Some(pos) = reminders.iter().position(|r| r.id == id) else {
            return Err(ReminderError::Unknown(id.to_owned()));
        };
        reminders[pos].active = false;
        reminders[pos].next_retick = None;
        let outcome = match reminders[pos].next_fire {
            Some(next) => format!(
                "satisfied; next scheduled fire {}",
                next.format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_default()
            ),
            None => "satisfied and retired (no scheduled fires left)".to_owned(),
        };
        if reminders[pos].retired() {
            reminders.remove(pos);
        }
        self.write(&reminders)?;
        Ok(outcome)
    }

    /// Fires everything due at `now` — scheduled fires and unsatisfied
    /// reticks — updating the store, and returns the fires.
    pub fn take_due(&self, now: OffsetDateTime) -> Result<Vec<Fired>, ReminderError> {
        let _guard = self.lock.lock().unwrap();
        let mut reminders = self.read()?;
        let mut fired = Vec::new();
        let mut keep = Vec::with_capacity(reminders.len());
        let mut changed = false;
        for mut r in reminders.drain(..) {
            // A hand-written entry with no next_fire gets one from its
            // schedule (a one-off with no time fires immediately).
            if r.next_fire.is_none() && !r.active && r.remaining != Some(0) {
                r.next_fire = Some(match r.schedule {
                    Schedule::Once => now,
                    _ => first_fire(&r.schedule, now),
                });
                changed = true;
            }

            let scheduled_due = r.next_fire.map(|t| t <= now).unwrap_or(false);
            let retick_due = r.active && r.next_retick.map(|t| t <= now).unwrap_or(false);

            if scheduled_due {
                changed = true;
                r.active = true;
                r.next_retick = Some(now + Duration::minutes(RETICK_MINUTES));
                // Reschedule the *scheduled* track.
                r.next_fire = match &r.schedule {
                    Schedule::Once => None,
                    Schedule::EveryMinutes { minutes } => {
                        Some(now + Duration::minutes(*minutes as i64))
                    }
                    Schedule::DailyAt { hour, minute } => Some(next_daily(now, *hour, *minute)),
                    Schedule::WeeklyAt {
                        weekday,
                        hour,
                        minute,
                    } => next_weekly(now, weekday, *hour, *minute),
                };
                if let Some(n) = r.remaining {
                    r.remaining = Some(n.saturating_sub(1));
                    if r.remaining == Some(0) {
                        r.next_fire = None;
                    }
                }
                fired.push(Fired {
                    payload: fire_payload(&r, now, false),
                    reminder: r.clone(),
                });
            } else if retick_due {
                changed = true;
                r.next_retick = Some(now + Duration::minutes(RETICK_MINUTES));
                fired.push(Fired {
                    payload: fire_payload(&r, now, true),
                    reminder: r.clone(),
                });
            }

            keep.push(r);
        }
        if changed {
            self.write(&keep)?;
        }
        Ok(fired)
    }
}

/// The first fire moment for a schedule created at `now`.
fn first_fire(schedule: &Schedule, now: OffsetDateTime) -> OffsetDateTime {
    match schedule {
        Schedule::Once => now,
        Schedule::EveryMinutes { minutes } => now + Duration::minutes(*minutes as i64),
        Schedule::DailyAt { hour, minute } => next_daily(now, *hour, *minute),
        Schedule::WeeklyAt {
            weekday,
            hour,
            minute,
        } => next_weekly(now, weekday, *hour, *minute).unwrap_or(now),
    }
}

/// The next occurrence of `hour:minute` local time strictly after `now`.
fn next_daily(now: OffsetDateTime, hour: u8, minute: u8) -> OffsetDateTime {
    let t = Time::from_hms(hour.min(23), minute.min(59), 0).expect("clamped to valid");
    let today = now.replace_time(t);
    if today > now {
        today
    } else {
        today + Duration::days(1)
    }
}

/// The next occurrence of `weekday` at `hour:minute` strictly after `now`.
/// An unparseable weekday yields `None` (the entry never schedules; the
/// scheduler logs it).
fn next_weekly(now: OffsetDateTime, weekday: &str, hour: u8, minute: u8) -> Option<OffsetDateTime> {
    let target = parse_weekday(weekday)?;
    let mut cand = next_daily(now, hour, minute);
    while cand.weekday() != target {
        cand += Duration::days(1);
    }
    Some(cand)
}

/// Parses a lowercase English weekday name.
pub fn parse_weekday(s: &str) -> Option<Weekday> {
    Some(match s.to_ascii_lowercase().as_str() {
        "monday" => Weekday::Monday,
        "tuesday" => Weekday::Tuesday,
        "wednesday" => Weekday::Wednesday,
        "thursday" => Weekday::Thursday,
        "friday" => Weekday::Friday,
        "saturday" => Weekday::Saturday,
        "sunday" => Weekday::Sunday,
        _ => return None,
    })
}

/// The JSON payload a fire injects as the synthetic tool result.
fn fire_payload(r: &Reminder, now: OffsetDateTime, retick: bool) -> String {
    let mut schedule = r.schedule.describe();
    if let Some(n) = r.remaining {
        schedule.push_str(&format!(", {n} scheduled fire(s) left"));
    }
    let status = if retick {
        "hourly re-fire: this reminder is still unsatisfied"
    } else {
        "scheduled fire"
    };
    json!({
        "reminder_fired": {
            "id": r.id,
            "message": r.message,
            "locked": r.locked,
            "user": r.user,
            "schedule": schedule,
            "status": status,
            "fired_at": format_prefix(now).trim(),
        },
        "note": format!(
            "This reminder is firing now. Unless I mark it handled with satisfy_reminder(id: {:?}), it re-fires every {RETICK_MINUTES} minutes. I only satisfy it once its purpose is genuinely fulfilled.",
            r.id
        )
    })
    .to_string()
}

/// Spawns the scheduler: every 30 seconds, fires whatever is due into
/// `deliver`.
pub fn spawn_scheduler(
    store: Arc<ReminderStore>,
    deliver: tokio::sync::mpsc::Sender<Fired>,
    status: Arc<Status>,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            match store.take_due(local_now()) {
                Ok(fired) => {
                    for f in fired {
                        info!(id = %f.reminder.id, "reminder fired");
                        status.log(format!(
                            "reminder fired: {} ({})",
                            f.reminder.id, f.reminder.message
                        ));
                        if deliver.send(f).await.is_err() {
                            warn!("reminder delivery channel closed");
                            return;
                        }
                    }
                }
                Err(e) => status.record_error(format!("reminder scheduler: {e}")),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    fn store() -> (tempfile::TempDir, ReminderStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = ReminderStore::new(dir.path().join("reminders.json"));
        (dir, s)
    }

    const NOW: OffsetDateTime = datetime!(2026-08-07 10:00:00 UTC);

    #[test]
    fn missing_file_is_an_empty_store() {
        let (_d, s) = store();
        assert!(s.list().unwrap().is_empty());
    }

    #[test]
    fn create_then_list_round_trips() {
        let (_d, s) = store();
        let r = s
            .create(
                "drink water".into(),
                Schedule::EveryMinutes { minutes: 60 },
                Some(3),
                7,
                NOW,
            )
            .unwrap();
        assert!(!r.locked, "agent-created reminders are never locked");
        assert_eq!(r.user, Some(7));
        assert_eq!(r.next_fire, Some(NOW + Duration::minutes(60)));
        let all = s.list().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].message, "drink water");
    }

    #[test]
    fn delete_removes_unlocked() {
        let (_d, s) = store();
        let r = s.create("x".into(), Schedule::Once, None, 7, NOW).unwrap();
        s.delete(&r.id).unwrap();
        assert!(s.list().unwrap().is_empty());
    }

    #[test]
    fn delete_refuses_locked_and_unknown() {
        let (_d, s) = store();
        std::fs::write(
            s.path.clone(),
            serde_json::json!([{
                "id": "lunch",
                "message": "time to eat lunch",
                "locked": true,
                "schedule": {"type": "daily_at", "hour": 12, "minute": 30}
            }])
            .to_string(),
        )
        .unwrap();
        assert!(matches!(
            s.delete("lunch").unwrap_err(),
            ReminderError::Locked(id) if id == "lunch"
        ));
        assert!(matches!(
            s.delete("nope").unwrap_err(),
            ReminderError::Unknown(_)
        ));
        assert_eq!(s.list().unwrap().len(), 1, "locked entry survives");
    }

    #[test]
    fn malformed_file_refuses_mutation_instead_of_wiping() {
        let (_d, s) = store();
        std::fs::write(&s.path, b"{ not json").unwrap();
        assert!(matches!(
            s.create("x".into(), Schedule::Once, None, 7, NOW),
            Err(ReminderError::Malformed(_))
        ));
        assert_eq!(std::fs::read(&s.path).unwrap(), b"{ not json");
    }

    #[test]
    fn once_fires_then_nags_until_satisfied() {
        let (_d, s) = store();
        let r = s
            .create("one shot".into(), Schedule::Once, None, 7, NOW)
            .unwrap();
        let fired = s.take_due(NOW + Duration::seconds(1)).unwrap();
        assert_eq!(fired.len(), 1);
        assert!(fired[0].payload.contains("one shot"));
        // Still present and active: it nags until satisfied.
        let all = s.list().unwrap();
        assert_eq!(all.len(), 1);
        assert!(all[0].active);
        assert_eq!(all[0].next_fire, None, "no more scheduled fires");
        // Satisfying retires it.
        let outcome = s.satisfy(&r.id).unwrap();
        assert!(outcome.contains("retired"), "{outcome}");
        assert!(s.list().unwrap().is_empty());
    }

    #[test]
    fn unsatisfied_reminder_reticks_hourly() {
        let (_d, s) = store();
        s.create("eat!".into(), Schedule::Once, None, 7, NOW).unwrap();
        assert_eq!(s.take_due(NOW).unwrap().len(), 1, "scheduled fire");
        // 30 minutes later: nothing.
        assert!(s
            .take_due(NOW + Duration::minutes(30))
            .unwrap()
            .is_empty());
        // 61 minutes later: the retick.
        let fired = s.take_due(NOW + Duration::minutes(61)).unwrap();
        assert_eq!(fired.len(), 1);
        assert!(fired[0].payload.contains("still unsatisfied"), "{}", fired[0].payload);
        // And again an hour after that.
        let fired = s.take_due(NOW + Duration::minutes(122)).unwrap();
        assert_eq!(fired.len(), 1);
    }

    #[test]
    fn satisfy_stops_the_reticks_until_the_next_scheduled_fire() {
        let (_d, s) = store();
        let r = s
            .create(
                "eat lunch".into(),
                Schedule::DailyAt { hour: 12, minute: 0 },
                None,
                7,
                datetime!(2026-08-07 10:00:00 UTC),
            )
            .unwrap();
        // Fires at noon.
        let fired = s.take_due(datetime!(2026-08-07 12:00:30 UTC)).unwrap();
        assert_eq!(fired.len(), 1);
        assert!(s.has_active(7));
        // Satisfied: no retick an hour later.
        let outcome = s.satisfy(&r.id).unwrap();
        assert!(outcome.contains("next scheduled fire"), "{outcome}");
        assert!(!s.has_active(7));
        assert!(s
            .take_due(datetime!(2026-08-07 13:30:00 UTC))
            .unwrap()
            .is_empty());
        // Tomorrow's scheduled fire still happens.
        let fired = s.take_due(datetime!(2026-08-08 12:00:30 UTC)).unwrap();
        assert_eq!(fired.len(), 1);
    }

    #[test]
    fn has_active_scopes_to_the_user() {
        let (_d, s) = store();
        s.create("for seven".into(), Schedule::Once, None, 7, NOW)
            .unwrap();
        s.take_due(NOW).unwrap();
        assert!(s.has_active(7));
        assert!(!s.has_active(8), "another user's nag is not mine");
    }

    #[test]
    fn recurring_counts_down_scheduled_fires_only() {
        let (_d, s) = store();
        let r = s
            .create(
                "hydrate".into(),
                Schedule::EveryMinutes { minutes: 10 },
                Some(2),
                7,
                NOW,
            )
            .unwrap();
        let t1 = NOW + Duration::minutes(10);
        assert_eq!(s.take_due(t1).unwrap().len(), 1);
        assert_eq!(s.list().unwrap()[0].remaining, Some(1));
        // A retick does NOT consume a scheduled fire.
        s.satisfy(&r.id).unwrap();
        let t2 = t1 + Duration::minutes(10);
        assert_eq!(s.take_due(t2).unwrap().len(), 1, "final scheduled fire");
        let all = s.list().unwrap();
        assert_eq!(all[0].remaining, Some(0));
        assert_eq!(all[0].next_fire, None);
        assert!(all[0].active, "still nagging after the last fire");
        // Satisfying now retires it.
        s.satisfy(&r.id).unwrap();
        assert!(s.list().unwrap().is_empty());
    }

    #[test]
    fn daily_next_fire_rolls_to_tomorrow_when_past() {
        assert_eq!(next_daily(NOW, 9, 30), datetime!(2026-08-08 09:30:00 UTC));
        assert_eq!(next_daily(NOW, 12, 30), datetime!(2026-08-07 12:30:00 UTC));
    }

    #[test]
    fn weekly_next_fire_lands_on_the_right_weekday() {
        // 2026-08-07 is a Friday.
        assert_eq!(NOW.weekday(), Weekday::Friday);
        let next = next_weekly(NOW, "sunday", 12, 0).unwrap();
        assert_eq!(next, datetime!(2026-08-09 12:00:00 UTC));
        assert_eq!(next.weekday(), Weekday::Sunday);
        // Later today still counts when the time is ahead.
        let next = next_weekly(NOW, "friday", 23, 0).unwrap();
        assert_eq!(next, datetime!(2026-08-07 23:00:00 UTC));
        // Earlier today rolls a full week.
        let next = next_weekly(NOW, "friday", 9, 0).unwrap();
        assert_eq!(next, datetime!(2026-08-14 09:00:00 UTC));
        assert!(next_weekly(NOW, "caturday", 9, 0).is_none());
    }

    #[test]
    fn hand_written_entry_without_next_fire_gets_scheduled_not_backfired() {
        let (_d, s) = store();
        std::fs::write(
            s.path.clone(),
            serde_json::json!([{
                "id": "lunch",
                "message": "eat lunch",
                "locked": true,
                "user": 7,
                "schedule": {"type": "daily_at", "hour": 12, "minute": 30}
            }])
            .to_string(),
        )
        .unwrap();
        assert!(s.take_due(NOW).unwrap().is_empty());
        let all = s.list().unwrap();
        assert_eq!(all[0].next_fire, Some(datetime!(2026-08-07 12:30:00 UTC)));
        let fired = s.take_due(datetime!(2026-08-07 12:31:00 UTC)).unwrap();
        assert_eq!(fired.len(), 1);
        assert!(fired[0].reminder.locked);
        let all = s.list().unwrap();
        assert!(all[0].locked, "locked survives every fire");
        assert!(all[0].active, "locked reminders nag too");
        assert_eq!(all[0].next_fire, Some(datetime!(2026-08-08 12:30:00 UTC)));
        // Satisfying a locked reminder is allowed — only deleting is not.
        s.satisfy("lunch").unwrap();
        assert_eq!(s.list().unwrap().len(), 1, "locked entry never disappears");
    }

    #[test]
    fn fire_payload_carries_message_lock_schedule_and_satisfy_note() {
        let (_d, s) = store();
        s.create(
            "hydrate".into(),
            Schedule::EveryMinutes { minutes: 5 },
            Some(2),
            7,
            NOW,
        )
        .unwrap();
        let fired = s.take_due(NOW + Duration::minutes(5)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&fired[0].payload).unwrap();
        assert_eq!(v["reminder_fired"]["message"], "hydrate");
        assert_eq!(v["reminder_fired"]["locked"], false);
        assert_eq!(v["reminder_fired"]["user"], 7);
        assert_eq!(v["reminder_fired"]["status"], "scheduled fire");
        assert!(v["note"].as_str().unwrap().contains("satisfy_reminder"));
    }
}
