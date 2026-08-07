//! The rolling day-memory: one running note per user, always in context,
//! distilled into permanent memory files at midnight.
//!
//! The agent maintains each user's note with its `update_day_memory` tool
//! — meals, mood, events, whatever the day accumulates. The note renders
//! into every turn as a `<day_memory>` block. Shortly after local
//! midnight the distiller takes every note whose date has passed, runs it
//! through the summarizer model, writes the digest into the shared file
//! jail at `/memory/<user name>/<date>.md` (so the agent can browse its
//! own archive with its file tools), and clears the note for the new day.
//!
//! The store is `daynotes.json` in the data root, hand-editable like
//! everything else.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tracing::warn;

/// One user's rolling note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayNote {
    /// The local date (`YYYY-MM-DD`) the note is about.
    pub date: String,
    /// The note body.
    pub note: String,
}

/// The on-disk day-note store: user id → today's note.
pub struct DayNoteStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl DayNoteStore {
    /// Builds a store over the given file path.
    pub fn new(path: PathBuf) -> Self {
        DayNoteStore {
            path,
            lock: Mutex::new(()),
        }
    }

    fn read(&self) -> HashMap<u64, DayNote> {
        std::fs::read(&self.path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    fn write(&self, notes: &HashMap<u64, DayNote>) {
        match serde_json::to_vec_pretty(notes) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(&self.path, bytes) {
                    warn!("persisting day notes: {e}");
                }
            }
            Err(e) => warn!("serializing day notes: {e}"),
        }
    }

    /// The user's note for `today`, when one exists. A note from an
    /// earlier date does not surface — it is awaiting distillation.
    pub fn today(&self, user: u64, today: &str) -> Option<String> {
        let _guard = self.lock.lock().unwrap();
        self.read()
            .get(&user)
            .filter(|n| n.date == today)
            .map(|n| n.note.clone())
    }

    /// Replaces the user's note for `today`.
    pub fn set(&self, user: u64, note: String, today: &str) {
        let _guard = self.lock.lock().unwrap();
        let mut notes = self.read();
        notes.insert(
            user,
            DayNote {
                date: today.to_owned(),
                note,
            },
        );
        self.write(&notes);
    }

    /// Every note whose date is before `today` — the distillation queue.
    pub fn stale(&self, today: &str) -> Vec<(u64, DayNote)> {
        let _guard = self.lock.lock().unwrap();
        self.read()
            .into_iter()
            .filter(|(_, n)| n.date.as_str() < today)
            .collect()
    }

    /// Removes a user's note, but only if it still carries `date` — a note
    /// rewritten for the new day mid-distill survives.
    pub fn remove_if_dated(&self, user: u64, date: &str) {
        let _guard = self.lock.lock().unwrap();
        let mut notes = self.read();
        if notes.get(&user).map(|n| n.date == date).unwrap_or(false) {
            notes.remove(&user);
            self.write(&notes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, DayNoteStore) {
        let dir = tempfile::tempdir().unwrap();
        let s = DayNoteStore::new(dir.path().join("daynotes.json"));
        (dir, s)
    }

    #[test]
    fn set_then_today_round_trips() {
        let (_d, s) = store();
        s.set(7, "ate toast".into(), "2026-08-07");
        assert_eq!(s.today(7, "2026-08-07"), Some("ate toast".into()));
        assert_eq!(s.today(8, "2026-08-07"), None);
    }

    #[test]
    fn yesterdays_note_does_not_surface_as_today() {
        let (_d, s) = store();
        s.set(7, "old news".into(), "2026-08-06");
        assert_eq!(s.today(7, "2026-08-07"), None);
    }

    #[test]
    fn stale_returns_only_past_dates() {
        let (_d, s) = store();
        s.set(7, "yesterday".into(), "2026-08-06");
        s.set(8, "today".into(), "2026-08-07");
        let stale = s.stale("2026-08-07");
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].0, 7);
        assert_eq!(stale[0].1.note, "yesterday");
    }

    #[test]
    fn remove_if_dated_spares_a_rewritten_note() {
        let (_d, s) = store();
        s.set(7, "yesterday".into(), "2026-08-06");
        // The new day's note lands before the distiller finishes.
        s.set(7, "fresh".into(), "2026-08-07");
        s.remove_if_dated(7, "2026-08-06");
        assert_eq!(s.today(7, "2026-08-07"), Some("fresh".into()));
        // And removes when the date still matches.
        s.remove_if_dated(7, "2026-08-07");
        assert_eq!(s.today(7, "2026-08-07"), None);
    }
}
