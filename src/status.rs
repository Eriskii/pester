//! Shared live status the web UI reads and the agent loop writes.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::Serialize;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::core::Usage;

/// How many recent activity lines the log ring keeps.
const LOG_CAPACITY: usize = 200;

/// A JSON-serializable snapshot of the agent's state.
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    /// What the agent is doing right now, e.g. `idle`, `thinking`,
    /// `tool:read_file`.
    pub state: String,
    /// The configured model slug.
    pub model: String,
    /// The moment the process started, RFC 3339.
    pub started_at: String,
    /// Completed turns since start.
    pub turns: u64,
    /// Messages currently in the transcript.
    pub transcript_len: usize,
    /// Accumulated token usage since start.
    pub usage: Usage,
    /// The most recent error, if any.
    pub last_error: Option<String>,
    /// Recent activity, newest last: `(rfc3339, line)`.
    pub log: Vec<(String, String)>,
}

/// The mutable live status.
pub struct Status {
    model: String,
    started_at: OffsetDateTime,
    inner: Mutex<Inner>,
}

struct Inner {
    state: String,
    turns: u64,
    transcript_len: usize,
    usage: Usage,
    last_error: Option<String>,
    log: VecDeque<(OffsetDateTime, String)>,
}

impl Status {
    /// Builds a status for the given model, starting `idle` now.
    pub fn new(model: impl Into<String>) -> Self {
        Status {
            model: model.into(),
            started_at: OffsetDateTime::now_utc(),
            inner: Mutex::new(Inner {
                state: "idle".into(),
                turns: 0,
                transcript_len: 0,
                usage: Usage::default(),
                last_error: None,
                log: VecDeque::new(),
            }),
        }
    }

    /// Sets the current state string.
    pub fn set_state(&self, state: impl Into<String>) {
        self.inner.lock().unwrap().state = state.into();
    }

    /// Records a completed turn and the current transcript length.
    pub fn record_turn(&self, transcript_len: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.turns += 1;
        inner.transcript_len = transcript_len;
    }

    /// Folds one completion's usage into the running total.
    pub fn add_usage(&self, usage: &Usage) {
        self.inner.lock().unwrap().usage.add(usage);
    }

    /// Records an error (also logged into the activity ring).
    pub fn record_error(&self, error: impl Into<String>) {
        let error = error.into();
        let mut inner = self.inner.lock().unwrap();
        push_log(&mut inner.log, format!("error: {error}"));
        inner.last_error = Some(error);
    }

    /// Appends one activity line to the ring.
    pub fn log(&self, line: impl Into<String>) {
        push_log(&mut self.inner.lock().unwrap().log, line.into());
    }

    /// Takes a snapshot for the web UI.
    pub fn snapshot(&self) -> Snapshot {
        let inner = self.inner.lock().unwrap();
        Snapshot {
            state: inner.state.clone(),
            model: self.model.clone(),
            started_at: fmt(self.started_at),
            turns: inner.turns,
            transcript_len: inner.transcript_len,
            usage: inner.usage.clone(),
            last_error: inner.last_error.clone(),
            log: inner.log.iter().map(|(t, l)| (fmt(*t), l.clone())).collect(),
        }
    }
}

fn push_log(log: &mut VecDeque<(OffsetDateTime, String)>, line: String) {
    if log.len() >= LOG_CAPACITY {
        log.pop_front();
    }
    log.push_back((OffsetDateTime::now_utc(), line));
}

fn fmt(t: OffsetDateTime) -> String {
    t.format(&Rfc3339).unwrap_or_else(|_| t.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reflects_writes() {
        let s = Status::new("m");
        s.set_state("thinking");
        s.record_turn(7);
        s.add_usage(&Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        });
        s.log("hello");
        s.record_error("boom");
        let snap = s.snapshot();
        assert_eq!(snap.state, "thinking");
        assert_eq!(snap.turns, 1);
        assert_eq!(snap.transcript_len, 7);
        assert_eq!(snap.usage.total_tokens, 15);
        assert_eq!(snap.last_error.as_deref(), Some("boom"));
        assert_eq!(snap.log.len(), 2);
    }

    #[test]
    fn log_ring_caps_its_length() {
        let s = Status::new("m");
        for i in 0..(LOG_CAPACITY + 10) {
            s.log(format!("line {i}"));
        }
        assert_eq!(s.snapshot().log.len(), LOG_CAPACITY);
    }
}
