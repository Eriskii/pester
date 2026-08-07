//! The turn loop and the compactor: user text in, assistant text out,
//! tools in between, and a rolling summary per user keeping each context
//! small.
//!
//! [`Agent`] owns one conversation **per Discord user**, each a rolling
//! `summary + tail` persisted to `conversations/<user id>.json`. Users
//! share everything else — the file jail, the reminder store, the system
//! prompts. Each rendered context opens with the system prompt (whose
//! `${user}` block names who the conversation is with), then that user's
//! `<conversation_summary>` block, then their verbatim tail.
//!
//! A turn appends the incoming user messages, then loops: assemble the
//! system prompt fresh from the data dir (hot-editable), call the model,
//! ship any reply text to the output channel, run any tool calls (each
//! dispatched with a [`ToolCtx`] naming the user, so reminders key
//! themselves automatically) and append their results, and go again until
//! the model answers without calling tools.
//!
//! Compaction folds the oldest part of a user's tail into their summary
//! via a separate zero-reasoning summarizer call. It fires from two
//! triggers: the **idle compactor** ([`Agent::idle_compact`], driven by a
//! timer in main) once a conversation has sat silent past the prompt-cache
//! TTL — the point where compacting is free, since the next turn pays a
//! full re-read either way — and the **length backstop** at the end of any
//! turn whose prompt crossed the configured token budget. Each successful
//! fold emits a notice so the endpoint can react 🧹. Compaction never runs
//! mid-turn and never touches the last [`TAIL_KEEP`] messages, so an
//! active conversation is never scuttled.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tracing::{info, warn};

use crate::client::{CompletionRequest, ReasoningEffort, SharedLlmClient, ToolDef};
use crate::core::{Message, Role, ToolUse, UserInfo};
use crate::data::{local_date, DataDir, PromptContext};
use crate::daynote::DayNoteStore;
use crate::reminders::ReminderStore;
use crate::status::Status;
use crate::tools::{Registry, ToolCtx};

/// Hard cap on model calls within one turn, so a tool loop cannot spin
/// forever.
const MAX_ITERATIONS: usize = 24;

/// Messages the compactor always leaves verbatim at the end of a tail.
const TAIL_KEEP: usize = 16;

/// The tail length below which the idle compactor does not bother — a
/// short conversation summarized into mush is worse than a short
/// conversation.
const IDLE_COMPACT_FLOOR: usize = 24;

/// One inbound user message: text plus any attached images as `data:`
/// URIs (already downscaled by the endpoint).
#[derive(Debug, Clone)]
pub struct UserMsg {
    /// The stamped message text.
    pub text: String,
    /// Attached images as `data:` URIs.
    pub images: Vec<String>,
}

impl UserMsg {
    /// A plain text message with no images.
    pub fn text(text: impl Into<String>) -> Self {
        UserMsg {
            text: text.into(),
            images: Vec::new(),
        }
    }
}

/// What starts a turn: user messages, or a fired reminder.
enum Seed {
    /// Ordinary inbound user messages.
    Users(Vec<UserMsg>),
    /// A reminder payload, injected as a synthetic `reminder` tool
    /// exchange (the Responses body builder round-trips harness-minted
    /// call/output pairs even for undeclared tool names).
    Reminder(String),
}

/// One observable event within a turn, in occurrence order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    /// A reply the endpoint posts into the conversation.
    Reply(String),
    /// A tool call just dispatched — the endpoint reacts to the triggering
    /// message with the tool's designated emoji.
    Tool {
        /// The tool's registry name.
        name: String,
        /// The tool's designated emoji.
        emoji: &'static str,
    },
}

/// One user's persisted conversation: identity, rolling summary, verbatim
/// tail.
#[derive(Debug, Serialize, Deserialize)]
struct Convo {
    /// Who this conversation is with.
    user: UserInfo,
    /// The rolling summary of everything already folded away. Empty until
    /// the first fold.
    #[serde(default)]
    summary: String,
    /// The verbatim message tail.
    messages: Vec<Message>,
}

/// One user's conversation plus its runtime accounting.
struct ConvoState {
    convo: Convo,
    /// Prompt tokens the most recent completion reported; feeds the
    /// length backstop.
    last_prompt_tokens: u32,
    /// The moment the last turn finished; feeds the idle compactor.
    last_activity: Instant,
}

impl ConvoState {
    fn new(user: UserInfo) -> Self {
        ConvoState {
            convo: Convo {
                user,
                summary: String::new(),
                messages: Vec::new(),
            },
            last_prompt_tokens: 0,
            last_activity: Instant::now(),
        }
    }
}

/// The agent's tuning: model identity, reasoning effort, and the
/// length-backstop budget.
#[derive(Debug, Clone)]
pub struct AgentSettings {
    /// The configured model slug (feeds `${model}` and the status page).
    pub model: String,
    /// The reasoning effort every turn requests.
    pub reasoning: ReasoningEffort,
    /// The prompt-token count past which a turn triggers the post-turn
    /// length-backstop fold.
    pub compact_tokens: u32,
}

/// Shared per-user state stores the agent consults each turn.
pub struct SharedStores {
    /// The rolling day-memory store.
    pub daynotes: Arc<DayNoteStore>,
    /// The reminder store — consulted so idle compaction never folds a
    /// conversation with an unsatisfied nag pending.
    pub reminders: Arc<ReminderStore>,
}

/// The multi-conversation agent: one context per user, shared tools.
pub struct Agent {
    llm: SharedLlmClient,
    data: DataDir,
    tools: Registry,
    status: Arc<Status>,
    model: String,
    reasoning: ReasoningEffort,
    compact_tokens: u32,
    daynotes: Arc<DayNoteStore>,
    reminders: Arc<ReminderStore>,
    convos: tokio::sync::Mutex<HashMap<u64, ConvoState>>,
    /// Fired once per successful fold so the endpoint can react 🧹.
    compact_notify: mpsc::Sender<()>,
}

impl Agent {
    /// Builds an agent, restoring every persisted conversation from the
    /// data dir. `compact_notify` receives one `()` per successful fold.
    pub fn new(
        llm: SharedLlmClient,
        data: DataDir,
        tools: Registry,
        status: Arc<Status>,
        settings: AgentSettings,
        stores: SharedStores,
        compact_notify: mpsc::Sender<()>,
    ) -> Self {
        let convos = load_convos(&data);
        Agent {
            llm,
            data,
            tools,
            status,
            model: settings.model,
            reasoning: settings.reasoning,
            compact_tokens: settings.compact_tokens,
            daynotes: stores.daynotes,
            reminders: stores.reminders,
            convos: tokio::sync::Mutex::new(convos),
            compact_notify,
        }
    }

    /// Runs one turn in `user`'s conversation: appends `user_texts` as
    /// user messages, then drives the model/tool loop, sending a
    /// [`TurnEvent`] into `out` for each reply and each tool call as they
    /// happen. Ends with the length-backstop compaction check.
    pub async fn turn(&self, user: UserInfo, msgs: Vec<UserMsg>, out: &mpsc::Sender<TurnEvent>) {
        self.seeded_turn(user, Seed::Users(msgs), out).await;
    }

    /// Runs a reminder-fired turn in `user`'s conversation: injects
    /// `payload` as a synthetic `reminder` tool exchange — an assistant
    /// tool call the harness minted plus its result — then drives the same
    /// model/tool loop. The model wakes seeing the reminder as a tool
    /// result and decides what to say.
    pub async fn reminder_turn(
        &self,
        user: UserInfo,
        payload: String,
        out: &mpsc::Sender<TurnEvent>,
    ) {
        self.seeded_turn(user, Seed::Reminder(payload), out).await;
    }

    /// The shared turn body over either seed.
    async fn seeded_turn(&self, user: UserInfo, seed: Seed, out: &mpsc::Sender<TurnEvent>) {
        let mut convos = self.convos.lock().await;
        let state = convos
            .entry(user.id)
            .or_insert_with(|| ConvoState::new(user.clone()));
        // The name (or its operator alias) may have changed since restore.
        state.convo.user = user.clone();
        match seed {
            Seed::Users(msgs) => {
                for m in msgs {
                    state
                        .convo
                        .messages
                        .push(Message::user_with_images(m.text, m.images));
                }
            }
            Seed::Reminder(payload) => {
                let call_id = format!(
                    "reminder_{}",
                    crate::data::local_now().unix_timestamp_nanos() / 1_000_000
                );
                state.convo.messages.push(Message::assistant_with_tools(
                    "",
                    vec![ToolUse::new(call_id.clone(), "reminder", serde_json::json!({}))],
                ));
                state.convo.messages.push(Message::tool_result(payload, call_id));
            }
        }
        self.persist(&state.convo);

        let tool_defs: Vec<ToolDef> = self.tools.defs();
        let tool_ctx = ToolCtx { user: user.clone() };
        let session_id = format!("pester-{}", user.id);
        for _ in 0..MAX_ITERATIONS {
            self.status.set_state("thinking");
            let system = match self.system_prompt(&state.convo.user) {
                Ok(s) => s,
                Err(e) => {
                    self.fail(out, format!("system prompt failed to render: {e}"))
                        .await;
                    break;
                }
            };
            let day_note = self.daynotes.today(user.id, &local_date());
            let history = render_history(system, &state.convo, day_note);

            let request = CompletionRequest::new(&history)
                .with_session_id(&session_id)
                .with_reasoning(self.reasoning)
                .with_tools(&tool_defs);
            let completion = match self.llm.complete(request).await {
                Ok(c) => c,
                Err(e) => {
                    self.fail(out, format!("model call failed: {e}")).await;
                    break;
                }
            };
            if let Some(usage) = &completion.usage {
                self.status.add_usage(usage);
                state.last_prompt_tokens = usage.prompt_tokens;
            }

            let assistant = Message::assistant_with_tools(
                completion.content.clone(),
                completion.tool_uses.clone(),
            )
            .with_reasoning(completion.reasoning.unwrap_or_default());
            state.convo.messages.push(assistant);

            if !completion.content.is_empty() {
                let _ = out.send(TurnEvent::Reply(completion.content.clone())).await;
            }

            if completion.tool_uses.is_empty() {
                self.persist(&state.convo);
                break;
            }
            for call in &completion.tool_uses {
                self.status.set_state(format!("tool:{}", call.name));
                self.status
                    .log(format!("[{}] tool call: {}", user.name, call.name));
                info!(tool = %call.name, user = %user.name, "dispatching tool call");
                if let Some(emoji) = self.tools.emoji_for(&call.name) {
                    let _ = out
                        .send(TurnEvent::Tool {
                            name: call.name.clone(),
                            emoji,
                        })
                        .await;
                }
                let result = self
                    .tools
                    .dispatch(&call.name, call.input.clone(), &tool_ctx)
                    .await;
                state.convo.messages.push(Message::tool_result(result, &call.id));
            }
            self.persist(&state.convo);
        }

        self.status.record_turn(state.convo.messages.len());
        self.status.set_state("idle");

        // The length backstop: a prompt past the budget folds now, right
        // after the reply already shipped — nobody is waiting, and the
        // one-off re-cache next turn is the price of the marathon.
        if state.last_prompt_tokens > self.compact_tokens {
            self.compact(state).await;
        }

        state.last_activity = Instant::now();
    }

    /// The idle-compactor entry point, called on a timer: folds every
    /// conversation that has sat silent for `idle_mins` with a tail long
    /// enough to be worth folding.
    pub async fn idle_compact(&self, idle_mins: u64) {
        let window = std::time::Duration::from_secs(idle_mins * 60);
        let mut convos = self.convos.lock().await;
        for state in convos.values_mut() {
            if state.last_activity.elapsed() < window
                || state.convo.messages.len() < IDLE_COMPACT_FLOOR
            {
                continue;
            }
            // The retick check comes before the compaction check: an
            // unsatisfied reminder keeps nagging hourly, and folding now
            // would put the very next re-fire into a freshly emptied
            // context. Leave the conversation intact until it is
            // satisfied.
            if self.reminders.has_active(state.convo.user.id) {
                continue;
            }
            if self.compact(state).await {
                // A fold counts as activity: the timer restarts so a
                // long-idle conversation folds once, not every tick.
                state.last_activity = Instant::now();
            }
        }
    }

    /// Folds everything but the last [`TAIL_KEEP`] messages of one
    /// conversation into its rolling summary via the summarizer. Returns
    /// whether a fold happened. A summarizer failure leaves the
    /// conversation untouched — messages are never dropped without a
    /// successful summary.
    async fn compact(&self, state: &mut ConvoState) -> bool {
        let split = split_point(&state.convo.messages, TAIL_KEEP);
        if split == 0 {
            return false;
        }
        self.status.set_state("compacting");
        let span_text = render_span(&state.convo.messages[..split]);
        let system = match self.summarizer_prompt() {
            Ok(s) => s,
            Err(e) => {
                self.status
                    .record_error(format!("summarizer prompt failed to render: {e}"));
                self.status.set_state("idle");
                return false;
            }
        };
        let prior = if state.convo.summary.is_empty() {
            "(none — this is the first fold)".to_owned()
        } else {
            state.convo.summary.clone()
        };
        let user_msg = format!(
            "This conversation is with {} (id {}).\n\nMy previous summary:\n{prior}\n\nThe messages being folded away:\n{span_text}",
            state.convo.user.name, state.convo.user.id
        );
        let history = [Message::system(system), Message::user(user_msg)];
        let session_id = format!("pester-summarizer-{}", state.convo.user.id);
        let request = CompletionRequest::new(&history).with_session_id(&session_id);
        match self.llm.complete(request).await {
            Ok(completion) if !completion.content.trim().is_empty() => {
                if let Some(usage) = &completion.usage {
                    self.status.add_usage(usage);
                }
                state.convo.summary = completion.content.trim().to_owned();
                state.convo.messages.drain(..split);
                self.persist(&state.convo);
                state.last_prompt_tokens = 0;
                self.status.log(format!(
                    "[{}] compacted {split} messages into a {}-char summary",
                    state.convo.user.name,
                    state.convo.summary.len()
                ));
                info!(folded = split, user = %state.convo.user.name, "compacted conversation");
                let _ = self.compact_notify.try_send(());
                self.status.set_state("idle");
                true
            }
            Ok(_) => {
                self.status
                    .record_error("summarizer returned an empty summary; fold skipped");
                self.status.set_state("idle");
                false
            }
            Err(e) => {
                self.status
                    .record_error(format!("summarizer call failed: {e}; fold skipped"));
                self.status.set_state("idle");
                false
            }
        }
    }

    /// Distills every day note whose date has passed into a permanent
    /// memory file in the shared jail (`/memory/<name>/<date>.md`) and
    /// clears it. A distiller failure leaves the note in place for the
    /// next attempt — a day is never dropped undistilled.
    pub async fn distill_daynotes(&self) {
        let today = local_date();
        for (uid, note) in self.daynotes.stale(&today) {
            let name = {
                let convos = self.convos.lock().await;
                convos
                    .get(&uid)
                    .map(|s| s.convo.user.name.clone())
                    .unwrap_or_else(|| format!("user-{uid}"))
            };
            let system = match self
                .data
                .daynote_prompt(&PromptContext::with_model(&self.model))
            {
                Ok(s) => s,
                Err(e) => {
                    self.status
                        .record_error(format!("daynote prompt failed to render: {e}"));
                    return;
                }
            };
            let user_msg = format!(
                "My day-memory for {} about {name}:\n\n{}",
                note.date, note.note
            );
            let history = [Message::system(system), Message::user(user_msg)];
            let request = CompletionRequest::new(&history);
            match self.llm.complete(request).await {
                Ok(completion) if !completion.content.trim().is_empty() => {
                    if let Some(usage) = &completion.usage {
                        self.status.add_usage(usage);
                    }
                    let dir = self.data.files_dir().join("memory").join(sanitize(&name));
                    if let Err(e) = std::fs::create_dir_all(&dir) {
                        self.status
                            .record_error(format!("creating memory dir: {e}"));
                        continue;
                    }
                    let path = dir.join(format!("{}.md", note.date));
                    if let Err(e) = std::fs::write(&path, completion.content.trim()) {
                        self.status
                            .record_error(format!("writing memory file: {e}"));
                        continue;
                    }
                    self.daynotes.remove_if_dated(uid, &note.date);
                    self.status.log(format!(
                        "[{name}] distilled day-memory for {} into memory/{}/{}.md",
                        note.date,
                        sanitize(&name),
                        note.date
                    ));
                    info!(user = %name, date = %note.date, "distilled day note");
                }
                Ok(_) => self
                    .status
                    .record_error("day-note distiller returned empty; keeping the note"),
                Err(e) => self
                    .status
                    .record_error(format!("day-note distiller failed: {e}; keeping the note")),
            }
        }
    }

    /// Renders the system prompt fresh from the data directory, with the
    /// `${user}` block naming who the conversation is with.
    fn system_prompt(&self, user: &UserInfo) -> Result<String, crate::data::DataError> {
        let ctx = PromptContext::with_model(&self.model)
            .with_tools_doc(self.tools.doc())
            .with_user_doc(format!(
                "I am speaking with {} (Discord user id {}). Reminders I create in this conversation attach to them automatically.",
                user.name, user.id
            ));
        self.data.system_prompt(&ctx)
    }

    /// Renders the summarizer prompt fresh from the data directory.
    fn summarizer_prompt(&self) -> Result<String, crate::data::DataError> {
        let ctx = PromptContext::with_model(&self.model);
        self.data.summarizer_prompt(&ctx)
    }

    /// Records an error and tells the conversation about it.
    async fn fail(&self, out: &mpsc::Sender<TurnEvent>, message: String) {
        warn!("{message}");
        self.status.record_error(&message);
        let _ = out.send(TurnEvent::Reply(format!("⚠ {message}"))).await;
    }

    /// Writes one conversation to disk; a failure logs and moves on (the
    /// in-memory conversation stays authoritative for the session).
    fn persist(&self, convo: &Convo) {
        let path = self.data.conversation_path(convo.user.id);
        match serde_json::to_vec_pretty(convo) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(&path, bytes) {
                    warn!("persisting conversation to {}: {e}", path.display());
                }
            }
            Err(e) => warn!("serializing conversation: {e}"),
        }
    }
}

/// Spawns the idle-compactor timer: checks every conversation once a
/// minute.
pub fn spawn_idle_compactor(agent: Arc<Agent>, idle_mins: u64) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(60));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            agent.idle_compact(idle_mins).await;
        }
    });
}

/// Spawns the midnight distiller: once a minute, any day note whose date
/// has passed is distilled into a permanent memory file and cleared.
pub fn spawn_midnight_distiller(agent: Arc<Agent>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(60));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            agent.distill_daynotes().await;
        }
    });
}

/// The wire transcript for one completion call: the system prompt, the
/// summary block (when a summary exists), the day-memory block (when a
/// note exists), then the verbatim tail.
fn render_history(system: String, convo: &Convo, day_note: Option<String>) -> Vec<Message> {
    let mut history = Vec::with_capacity(convo.messages.len() + 3);
    history.push(Message::system(system));
    if !convo.summary.is_empty() {
        history.push(Message::user(format!(
            "<conversation_summary>\nMy own rolling summary of the conversation before this point:\n\n{}\n</conversation_summary>",
            convo.summary
        )));
    }
    if let Some(note) = day_note.filter(|n| !n.is_empty()) {
        history.push(Message::user(format!(
            "<day_memory>\nMy running day-memory for today (maintained with update_day_memory):\n\n{note}\n</day_memory>"
        )));
    }
    history.extend(convo.messages.iter().cloned());
    history
}

/// Picks the fold boundary: `len - keep`, advanced past any tool results so
/// the remaining tail never opens with an orphaned `function_call_output`
/// (a tool result stays with the assistant call it answers — both fold, or
/// neither). Returns 0 when there is nothing to fold.
fn split_point(messages: &[Message], keep: usize) -> usize {
    if messages.len() <= keep {
        return 0;
    }
    let mut split = messages.len() - keep;
    while split < messages.len() && messages[split].role() == Role::Tool {
        split += 1;
    }
    if split >= messages.len() {
        return 0;
    }
    split
}

/// Renders a folded span as plain text for the summarizer: one block per
/// message, tool traffic abbreviated.
fn render_span(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role() {
            Role::User => {
                out.push_str(&format!("[user] {}\n", m.content()));
                let images = m.images().len();
                if images > 0 {
                    out.push_str(&format!("[user attached {images} image(s)]\n"));
                }
            }
            Role::Assistant => {
                if !m.content().is_empty() {
                    out.push_str(&format!("[me] {}\n", m.content()));
                }
                for t in m.tool_uses() {
                    out.push_str(&format!("[me → tool] {}({})\n", t.name, t.input));
                }
            }
            Role::Tool => {
                let body = m.content();
                let short: String = body.chars().take(300).collect();
                let ellipsis = if body.chars().count() > 300 { "…" } else { "" };
                out.push_str(&format!("[tool result] {short}{ellipsis}\n"));
            }
            Role::System => {}
        }
    }
    out
}

/// A filesystem-safe slug of a user name.
fn sanitize(name: &str) -> String {
    let slug: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if slug.is_empty() {
        "unnamed".into()
    } else {
        slug
    }
}

/// Restores every persisted conversation from `conversations/`.
fn load_convos(data: &DataDir) -> HashMap<u64, ConvoState> {
    let mut convos = HashMap::new();
    let Ok(entries) = std::fs::read_dir(data.conversations_dir()) else {
        return convos;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Convo>(&b).ok()) {
            Some(convo) => {
                convos.insert(
                    convo.user.id,
                    ConvoState {
                        convo,
                        last_prompt_tokens: 0,
                        last_activity: Instant::now(),
                    },
                );
            }
            None => warn!(
                "conversation at {} is malformed; skipping",
                path.display()
            ),
        }
    }
    convos
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use serde_json::json;

    use crate::client::{Completion, CompletionRequest, LlmClient, LlmError};
    use crate::data::DataDir;
    use crate::tools::Registry;

    fn tester() -> UserInfo {
        UserInfo {
            id: 42,
            name: "sofia".into(),
        }
    }

    /// A client that plays back a script of completions in order.
    struct Scripted {
        script: Mutex<Vec<Completion>>,
    }

    impl Scripted {
        fn new(script: Vec<Completion>) -> Self {
            Scripted {
                script: Mutex::new(script),
            }
        }
    }

    #[async_trait::async_trait]
    impl LlmClient for Scripted {
        async fn complete(&self, _ctx: CompletionRequest<'_>) -> Result<Completion, LlmError> {
            let mut script = self.script.lock().unwrap();
            if script.is_empty() {
                return Err(LlmError::Malformed("script exhausted".into()));
            }
            Ok(script.remove(0))
        }
    }

    fn reply(content: &str) -> Completion {
        Completion {
            content: content.into(),
            reasoning: None,
            tool_uses: Vec::new(),
            usage: None,
        }
    }

    fn agent_with(script: Vec<Completion>, root: &std::path::Path) -> (Agent, mpsc::Receiver<()>) {
        agent_with_budget(script, root, u32::MAX)
    }

    fn agent_with_budget(
        script: Vec<Completion>,
        root: &std::path::Path,
        compact_tokens: u32,
    ) -> (Agent, mpsc::Receiver<()>) {
        let data = DataDir::open(root).unwrap();
        let tools = Registry::new(vec![std::sync::Arc::new(crate::tools::time::GetTime)]);
        let stores = SharedStores {
            daynotes: Arc::new(DayNoteStore::new(data.daynotes_path())),
            reminders: Arc::new(ReminderStore::new(data.reminders_path())),
        };
        let (notify_tx, notify_rx) = mpsc::channel(8);
        let agent = Agent::new(
            Arc::new(Scripted::new(script)),
            data,
            tools,
            Arc::new(Status::new("test-model")),
            AgentSettings {
                model: "test-model".into(),
                reasoning: ReasoningEffort::None,
                compact_tokens,
            },
            stores,
            notify_tx,
        );
        (agent, notify_rx)
    }

    /// The messages of one user's conversation.
    async fn messages_of(agent: &Agent, uid: u64) -> Vec<Message> {
        agent.convos.lock().await[&uid].convo.messages.clone()
    }

    async fn summary_of(agent: &Agent, uid: u64) -> String {
        agent.convos.lock().await[&uid].convo.summary.clone()
    }

    /// Seeds a user's conversation with `n` alternating messages.
    async fn seed(agent: &Agent, user: UserInfo, n: usize) {
        let mut convos = agent.convos.lock().await;
        let state = convos
            .entry(user.id)
            .or_insert_with(|| ConvoState::new(user));
        state.convo.messages = chatter(n);
    }

    #[tokio::test]
    async fn plain_reply_ends_the_turn_and_ships_the_text() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![reply("hello there")], root.path());
        let (tx, mut rx) = mpsc::channel(8);
        agent.turn(tester(), vec![UserMsg::text("hi")], &tx).await;
        assert_eq!(
            rx.recv().await.unwrap(),
            TurnEvent::Reply("hello there".into())
        );
        let messages = messages_of(&agent, 42).await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role(), Role::User);
        assert_eq!(messages[1].role(), Role::Assistant);
    }

    #[tokio::test]
    async fn users_get_separate_conversations() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![reply("hi a"), reply("hi b")], root.path());
        let (tx, _rx) = mpsc::channel(8);
        let a = UserInfo { id: 1, name: "a".into() };
        let b = UserInfo { id: 2, name: "b".into() };
        agent.turn(a, vec![UserMsg::text("from a")], &tx).await;
        agent.turn(b, vec![UserMsg::text("from b")], &tx).await;
        let ma = messages_of(&agent, 1).await;
        let mb = messages_of(&agent, 2).await;
        assert_eq!(ma.len(), 2);
        assert_eq!(mb.len(), 2);
        assert_eq!(ma[0].content(), "from a");
        assert_eq!(mb[0].content(), "from b");
    }

    #[tokio::test]
    async fn tool_call_round_trips_and_the_loop_continues() {
        let root = tempfile::tempdir().unwrap();
        let tool_turn = Completion {
            content: String::new(),
            reasoning: None,
            tool_uses: vec![ToolUse::new("call_1", "get_time", json!({}))],
            usage: None,
        };
        let (agent, _n) = agent_with(vec![tool_turn, reply("it is now")], root.path());
        let (tx, mut rx) = mpsc::channel(8);
        agent.turn(tester(), vec![UserMsg::text("what time is it")], &tx).await;
        assert_eq!(
            rx.recv().await.unwrap(),
            TurnEvent::Tool {
                name: "get_time".into(),
                emoji: "🕐"
            },
            "the tool event precedes the reply"
        );
        assert_eq!(rx.recv().await.unwrap(), TurnEvent::Reply("it is now".into()));
        let messages = messages_of(&agent, 42).await;
        // user, assistant(tool_use), tool result, assistant reply
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].role(), Role::Tool);
        assert_eq!(messages[2].tool_call_id(), Some("call_1"));
        assert!(messages[2].content().contains("time"));
    }

    #[tokio::test]
    async fn model_error_reports_to_the_channel() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![], root.path());
        let (tx, mut rx) = mpsc::channel(8);
        agent.turn(tester(), vec![UserMsg::text("hi")], &tx).await;
        let TurnEvent::Reply(out) = rx.recv().await.unwrap() else {
            panic!("expected a reply event");
        };
        assert!(out.starts_with('⚠'), "{out}");
    }

    #[tokio::test]
    async fn conversation_persists_and_restores_across_agents() {
        let root = tempfile::tempdir().unwrap();
        {
            let (agent, _n) = agent_with(vec![reply("first")], root.path());
            let (tx, _rx) = mpsc::channel(8);
            agent.turn(tester(), vec![UserMsg::text("hi")], &tx).await;
        }
        let (agent, _n) = agent_with(vec![reply("second")], root.path());
        let messages = messages_of(&agent, 42).await;
        assert_eq!(messages.len(), 2, "restored the persisted tail");
        assert_eq!(messages[1].content(), "first");
    }

    #[tokio::test]
    async fn reminder_turn_mints_the_synthetic_exchange_and_replies() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![reply("time to eat!!")], root.path());
        let (tx, mut rx) = mpsc::channel(8);
        agent
            .reminder_turn(
                tester(),
                r#"{"reminder_fired":{"message":"lunch"}}"#.into(),
                &tx,
            )
            .await;
        assert_eq!(
            rx.recv().await.unwrap(),
            TurnEvent::Reply("time to eat!!".into())
        );
        let messages = messages_of(&agent, 42).await;
        // minted assistant tool call, minted result, real reply
        assert_eq!(messages.len(), 3);
        let mint = &messages[0];
        assert_eq!(mint.role(), Role::Assistant);
        assert_eq!(mint.tool_uses().len(), 1);
        assert_eq!(mint.tool_uses()[0].name, "reminder");
        let result = &messages[1];
        assert_eq!(result.role(), Role::Tool);
        assert_eq!(
            result.tool_call_id(),
            Some(mint.tool_uses()[0].id.as_str()),
            "result pins to the minted call"
        );
        assert!(result.content().contains("lunch"));
        assert_eq!(messages[2].content(), "time to eat!!");
    }

    // --- the compactor ---

    /// A conversation of `n` alternating user/assistant messages.
    fn chatter(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user(format!("user line {i}"))
                } else {
                    Message::assistant(format!("reply {i}"))
                }
            })
            .collect()
    }

    #[test]
    fn split_point_keeps_the_tail_and_folds_the_rest() {
        let messages = chatter(40);
        let split = split_point(&messages, 16);
        assert_eq!(split, 24);
    }

    #[test]
    fn split_point_returns_zero_when_nothing_to_fold() {
        assert_eq!(split_point(&chatter(10), 16), 0);
        assert_eq!(split_point(&chatter(16), 16), 0);
        assert_eq!(split_point(&[], 16), 0);
    }

    #[test]
    fn split_point_never_strands_a_tool_result_at_the_tail_head() {
        // 20 messages where the natural split (index 4) lands on a run of
        // tool results: the split advances past them so they fold together
        // with the assistant call they answer.
        let mut messages = chatter(4);
        messages.push(Message::assistant_with_tools(
            "",
            vec![ToolUse::new("c1", "read_file", json!({}))],
        ));
        messages.push(Message::tool_result("{}", "c1"));
        messages.push(Message::tool_result("{}", "c2"));
        messages.extend(chatter(16));
        let split = split_point(&messages, 16);
        assert!(
            messages[split].role() != Role::Tool,
            "tail must not open with a tool result"
        );
        assert_eq!(split, 7);
    }

    #[tokio::test]
    async fn idle_compact_folds_into_a_summary_and_notifies() {
        let root = tempfile::tempdir().unwrap();
        let (agent, mut notify) =
            agent_with(vec![reply("THE SUMMARY of everything so far")], root.path());
        seed(&agent, tester(), 40).await;
        agent.idle_compact(0).await;
        assert_eq!(summary_of(&agent, 42).await, "THE SUMMARY of everything so far");
        let messages = messages_of(&agent, 42).await;
        assert_eq!(messages.len(), 16, "tail kept verbatim");
        assert_eq!(messages[0].content(), "user line 24");
        assert!(notify.try_recv().is_ok(), "compact notice fired");
    }

    #[tokio::test]
    async fn idle_compact_folds_each_idle_user_separately() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![reply("SUM A"), reply("SUM B")], root.path());
        seed(&agent, UserInfo { id: 1, name: "a".into() }, 40).await;
        seed(&agent, UserInfo { id: 2, name: "b".into() }, 40).await;
        agent.idle_compact(0).await;
        let sums = [summary_of(&agent, 1).await, summary_of(&agent, 2).await];
        assert!(sums.contains(&"SUM A".to_owned()));
        assert!(sums.contains(&"SUM B".to_owned()));
    }

    #[tokio::test]
    async fn idle_compact_respects_the_floor() {
        let root = tempfile::tempdir().unwrap();
        let (agent, mut notify) = agent_with(vec![reply("should not run")], root.path());
        seed(&agent, tester(), 20).await; // under IDLE_COMPACT_FLOOR
        agent.idle_compact(0).await;
        assert!(summary_of(&agent, 42).await.is_empty());
        assert_eq!(messages_of(&agent, 42).await.len(), 20);
        assert!(notify.try_recv().is_err());
    }

    #[tokio::test]
    async fn idle_compact_waits_out_the_idle_window() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![reply("too soon")], root.path());
        seed(&agent, tester(), 40).await;
        agent.idle_compact(60).await; // last_activity is ~now
        assert!(
            summary_of(&agent, 42).await.is_empty(),
            "no fold before the idle window"
        );
        assert_eq!(messages_of(&agent, 42).await.len(), 40);
    }

    #[tokio::test]
    async fn failed_summarizer_leaves_the_conversation_untouched() {
        let root = tempfile::tempdir().unwrap();
        let (agent, mut notify) = agent_with(vec![], root.path()); // script exhausted → error
        seed(&agent, tester(), 40).await;
        agent.idle_compact(0).await;
        assert!(summary_of(&agent, 42).await.is_empty());
        assert_eq!(
            messages_of(&agent, 42).await.len(),
            40,
            "no message is ever dropped without a summary"
        );
        assert!(notify.try_recv().is_err());
    }

    #[tokio::test]
    async fn second_fold_carries_the_prior_summary_into_the_prompt() {
        let root = tempfile::tempdir().unwrap();
        // Capture the summarizer's input by scripting a client that records.
        struct Recording {
            seen: Mutex<Vec<String>>,
        }
        #[async_trait::async_trait]
        impl LlmClient for Recording {
            async fn complete(
                &self,
                ctx: CompletionRequest<'_>,
            ) -> Result<Completion, LlmError> {
                self.seen
                    .lock()
                    .unwrap()
                    .push(ctx.history.last().unwrap().content().to_owned());
                Ok(Completion {
                    content: "NEW SUMMARY".into(),
                    reasoning: None,
                    tool_uses: Vec::new(),
                    usage: None,
                })
            }
        }
        let recording = Arc::new(Recording {
            seen: Mutex::new(Vec::new()),
        });
        let data = DataDir::open(root.path()).unwrap();
        let stores = SharedStores {
            daynotes: Arc::new(DayNoteStore::new(data.daynotes_path())),
            reminders: Arc::new(ReminderStore::new(data.reminders_path())),
        };
        let (notify_tx, _notify_rx) = mpsc::channel(8);
        let agent = Agent::new(
            Arc::clone(&recording) as SharedLlmClient,
            data,
            Registry::new(vec![]),
            Arc::new(Status::new("m")),
            AgentSettings {
                model: "m".into(),
                reasoning: ReasoningEffort::None,
                compact_tokens: u32::MAX,
            },
            stores,
            notify_tx,
        );
        {
            let mut convos = agent.convos.lock().await;
            let state = convos
                .entry(42)
                .or_insert_with(|| ConvoState::new(tester()));
            state.convo.summary = "OLD SUMMARY".into();
            state.convo.messages = chatter(40);
        }
        agent.idle_compact(0).await;
        let seen = recording.seen.lock().unwrap();
        assert!(seen[0].contains("OLD SUMMARY"), "prior summary rides along");
        assert!(seen[0].contains("user line 0"), "folded span rides along");
        assert!(seen[0].contains("sofia"), "summarizer knows whose convo it is");
    }

    #[tokio::test]
    async fn rendered_history_carries_the_summary_block() {
        let convo = Convo {
            user: tester(),
            summary: "WHAT CAME BEFORE".into(),
            messages: chatter(2),
        };
        let history = render_history("SYS".into(), &convo, Some("she ate toast".into()));
        assert_eq!(history.len(), 5);
        assert_eq!(history[0].role(), Role::System);
        assert!(history[1].content().contains("<conversation_summary>"));
        assert!(history[1].content().contains("WHAT CAME BEFORE"));
        assert!(history[2].content().contains("<day_memory>"));
        assert!(history[2].content().contains("she ate toast"));
        assert_eq!(history[3].content(), "user line 0");
    }

    #[tokio::test]
    async fn system_prompt_names_the_user() {
        let root = tempfile::tempdir().unwrap();
        let (agent, _n) = agent_with(vec![], root.path());
        let prompt = agent.system_prompt(&tester()).unwrap();
        assert!(prompt.contains("sofia"), "{prompt}");
        assert!(prompt.contains("42"), "{prompt}");
    }

    #[tokio::test]
    async fn length_backstop_folds_after_an_oversized_turn() {
        let root = tempfile::tempdir().unwrap();
        let big_reply = Completion {
            content: "long conversation continues".into(),
            reasoning: None,
            tool_uses: Vec::new(),
            usage: Some(crate::core::Usage {
                prompt_tokens: 5000,
                completion_tokens: 10,
                total_tokens: 5010,
                ..Default::default()
            }),
        };
        let (agent, mut notify) = agent_with_budget(
            vec![big_reply, reply("BACKSTOP SUMMARY")],
            root.path(),
            1000, // budget far below the reported 5000
        );
        seed(&agent, tester(), 38).await;
        let (tx, _rx) = mpsc::channel(8);
        agent.turn(tester(), vec![UserMsg::text("another message")], &tx).await;
        assert_eq!(summary_of(&agent, 42).await, "BACKSTOP SUMMARY");
        assert_eq!(messages_of(&agent, 42).await.len(), 16);
        assert!(notify.try_recv().is_ok());
    }
}
