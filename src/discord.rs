//! The Discord endpoint: DMs in, DMs out, one conversation per user.
//!
//! A serenity gateway client forwards direct messages into an mpsc work
//! queue — guild messages are ignored, and when an allowed user is
//! configured, DMs from anyone else are dropped (with the sender's id
//! logged so an operator can pin it). Operator aliases
//! (`PESTER_USER_ALIASES`) replace Discord usernames at the gateway, so
//! the aliased name is what the agent, the reminders, and the logs all
//! see.
//!
//! A worker task drains the queue: DM batches group per author (each
//! author's messages run as a turn in *their* conversation), replies post
//! back into the author's DM channel split at Discord's 2000-character
//! cap, and the triggering message collects the emoji of each tool the
//! turn used. Fired reminders resolve their target user's DM channel from
//! the persisted per-user map (`dm_channels.json`, refreshed on every DM)
//! — or create a fresh DM — and run as reminder turns with no reaction
//! target.
//!
//! DM content rides outside the privileged message-content intent, so the
//! gateway subscribes to `DIRECT_MESSAGES` only.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serenity::all::{
    ChannelId, Context, EventHandler, GatewayIntents, Message, MessageId, ReactionType, Ready,
    UserId,
};
use serenity::Client;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::agent::{Agent, TurnEvent, UserMsg};
use crate::core::UserInfo;
use crate::data::{local_now, DataDir};
use crate::reminders::Fired;
use crate::status::Status;
use crate::timestamp::prepend_prefix;

/// Discord's per-message character cap.
const DISCORD_MESSAGE_CAP: usize = 2000;

/// The reaction marking a compaction fold on the most recent message.
const COMPACT_EMOJI: &str = "🧹";

/// The most recent inbound DM's location, shared between the gateway
/// handler (writer) and the compact-notice reactor (reader).
type LastMessage = Arc<std::sync::Mutex<Option<(ChannelId, MessageId)>>>;

/// One inbound DM: who sent it, where to answer, the message to react to,
/// the stamped text, and any attached images as `data:` URIs.
struct Inbound {
    author: UserInfo,
    channel: ChannelId,
    message: MessageId,
    text: String,
    images: Vec<String>,
}

/// The longest attachment the endpoint downloads, in bytes.
const MAX_ATTACHMENT_BYTES: u32 = 20 * 1024 * 1024;

/// The bounding box images are downscaled into before riding the
/// transcript as JPEG `data:` URIs.
const IMAGE_MAX_EDGE: u32 = 1024;

/// Downloads and downscales a message's image attachments into `data:`
/// URIs. Non-image or unprocessable attachments come back as text notes.
async fn collect_images(msg: &Message) -> (Vec<String>, Vec<String>) {
    let mut images = Vec::new();
    let mut notes = Vec::new();
    for att in &msg.attachments {
        let is_image = att
            .content_type
            .as_deref()
            .map(|c| c.starts_with("image/"))
            .unwrap_or(false);
        if !is_image {
            notes.push(format!("[sent a non-image attachment: {}]", att.filename));
            continue;
        }
        if att.size > MAX_ATTACHMENT_BYTES {
            notes.push(format!(
                "[sent an image too large to process: {}]",
                att.filename
            ));
            continue;
        }
        match att.download().await {
            Ok(bytes) => {
                match tokio::task::spawn_blocking(move || shrink_to_data_uri(&bytes)).await {
                    Ok(Ok(uri)) => images.push(uri),
                    Ok(Err(e)) => {
                        warn!(file = %att.filename, "processing image attachment: {e}");
                        notes.push(format!(
                            "[sent an image that could not be processed: {}]",
                            att.filename
                        ));
                    }
                    Err(e) => warn!("image task panicked: {e}"),
                }
            }
            Err(e) => {
                warn!(file = %att.filename, "downloading attachment: {e}");
                notes.push(format!(
                    "[sent an image that could not be downloaded: {}]",
                    att.filename
                ));
            }
        }
    }
    (images, notes)
}

/// Decodes an image, bounds it to [`IMAGE_MAX_EDGE`], and re-encodes it as
/// a JPEG `data:` URI.
fn shrink_to_data_uri(bytes: &[u8]) -> Result<String, String> {
    use base64::Engine;
    let img = image::load_from_memory(bytes).map_err(|e| format!("decode: {e}"))?;
    let img = img.thumbnail(IMAGE_MAX_EDGE, IMAGE_MAX_EDGE);
    let mut jpeg = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut jpeg);
    img.to_rgb8()
        .write_to(&mut cursor, image::ImageFormat::Jpeg)
        .map_err(|e| format!("encode: {e}"))?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
    Ok(format!("data:image/jpeg;base64,{b64}"))
}

/// One unit for the agent worker: a DM or a fired reminder.
enum Work {
    /// An inbound DM.
    Dm(Inbound),
    /// A fired reminder, delivered by the scheduler.
    Reminder(Fired),
}

struct Handler {
    allowed_user: Option<UserId>,
    aliases: HashMap<u64, String>,
    inbound: mpsc::Sender<Work>,
    status: Arc<Status>,
    last_message: LastMessage,
}

/// Decides whether a message reaches the agent: it must be a DM (no guild),
/// not authored by a bot, and — when an allowed user is set — authored by
/// that user.
fn accepts(is_dm: bool, author_is_bot: bool, author: UserId, allowed: Option<UserId>) -> bool {
    is_dm && !author_is_bot && allowed.map(|u| u == author).unwrap_or(true)
}

#[serenity::async_trait]
impl EventHandler for Handler {
    async fn ready(&self, _ctx: Context, ready: Ready) {
        info!(user = %ready.user.name, "discord gateway connected; DM me");
        self.status
            .log(format!("discord connected as {} (DM mode)", ready.user.name));
    }

    async fn message(&self, _ctx: Context, msg: Message) {
        let is_dm = msg.guild_id.is_none();
        if !accepts(is_dm, msg.author.bot, msg.author.id, self.allowed_user) {
            if is_dm && !msg.author.bot {
                info!(
                    author_id = %msg.author.id,
                    author = %msg.author.name,
                    "ignoring DM from unallowed user"
                );
            }
            return;
        }
        // The operator alias, when set, replaces the Discord username
        // everywhere downstream.
        let author = UserInfo {
            id: msg.author.id.get(),
            name: self
                .aliases
                .get(&msg.author.id.get())
                .cloned()
                .unwrap_or_else(|| msg.author.name.clone()),
        };
        info!(author_id = %author.id, author = %author.name, "DM received");
        *self.last_message.lock().unwrap() = Some((msg.channel_id, msg.id));
        let (images, notes) = collect_images(&msg).await;
        let mut body = msg.content.clone();
        for note in &notes {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(note);
        }
        let stamped = prepend_prefix(local_now(), &format!("<{}> {}", author.name, body));
        self.status.log(format!(
            "<{}> {}{}",
            author.name,
            body,
            if images.is_empty() {
                String::new()
            } else {
                format!(" [{} image(s)]", images.len())
            }
        ));
        let inbound = Inbound {
            author,
            channel: msg.channel_id,
            message: msg.id,
            text: stamped,
            images,
        };
        if self.inbound.send(Work::Dm(inbound)).await.is_err() {
            warn!("agent worker is gone; dropping inbound message");
        }
    }
}

/// Runs the Discord endpoint forever: gateway, agent worker, the reminder
/// pump, and the compact-notice reactor (🧹 on the most recent message per
/// fold).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    token: String,
    allowed_user: Option<u64>,
    aliases: HashMap<u64, String>,
    agent: Arc<Agent>,
    status: Arc<Status>,
    data: DataDir,
    mut compact_notices: mpsc::Receiver<()>,
    mut reminder_fires: mpsc::Receiver<Fired>,
) -> Result<(), serenity::Error> {
    let (inbound_tx, inbound_rx) = mpsc::channel::<Work>(256);
    let last_message: LastMessage = Arc::default();

    let intents = GatewayIntents::DIRECT_MESSAGES;
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler {
            allowed_user: allowed_user.map(UserId::new),
            aliases: aliases.clone(),
            inbound: inbound_tx.clone(),
            status: Arc::clone(&status),
            last_message: Arc::clone(&last_message),
        })
        .await?;

    let http = Arc::clone(&client.http);
    tokio::spawn(worker(
        inbound_rx,
        agent,
        Arc::clone(&http),
        status,
        data,
        allowed_user,
        aliases,
    ));

    // The reminder pump: fired reminders join the same work queue as DMs,
    // so a fire never races an in-flight turn.
    tokio::spawn(async move {
        while let Some(fired) = reminder_fires.recv().await {
            if inbound_tx.send(Work::Reminder(fired)).await.is_err() {
                return;
            }
        }
    });

    tokio::spawn(async move {
        while compact_notices.recv().await.is_some() {
            let target = *last_message.lock().unwrap();
            let Some((channel, message)) = target else {
                continue;
            };
            let reaction = ReactionType::Unicode(COMPACT_EMOJI.to_owned());
            if let Err(e) = http.create_reaction(channel, message, &reaction).await {
                warn!("compact reaction failed: {e}");
            }
        }
    });

    client.start().await
}

/// One user's slice of a drained DM batch.
struct UserBatch {
    author: UserInfo,
    channel: ChannelId,
    trigger: MessageId,
    msgs: Vec<UserMsg>,
}

/// Drains the work queue. DM batches group per author and run as turns in
/// that author's conversation; fired reminders run as reminder turns into
/// their user's DM channel.
async fn worker(
    mut inbound: mpsc::Receiver<Work>,
    agent: Arc<Agent>,
    http: Arc<serenity::http::Http>,
    status: Arc<Status>,
    data: DataDir,
    allowed_user: Option<u64>,
    aliases: HashMap<u64, String>,
) {
    while let Some(work) = inbound.recv().await {
        let mut batches: Vec<UserBatch> = Vec::new();
        let mut reminders: Vec<Fired> = Vec::new();
        let absorb = |work: Work, batches: &mut Vec<UserBatch>, reminders: &mut Vec<Fired>| {
            match work {
                Work::Dm(m) => {
                    let msg = UserMsg {
                        text: m.text,
                        images: m.images,
                    };
                    if let Some(b) = batches.iter_mut().find(|b| b.author.id == m.author.id) {
                        b.trigger = m.message;
                        b.msgs.push(msg);
                    } else {
                        batches.push(UserBatch {
                            author: m.author,
                            channel: m.channel,
                            trigger: m.message,
                            msgs: vec![msg],
                        });
                    }
                }
                Work::Reminder(fired) => reminders.push(fired),
            }
        };
        absorb(work, &mut batches, &mut reminders);
        while let Ok(more) = inbound.try_recv() {
            absorb(more, &mut batches, &mut reminders);
        }

        for batch in batches {
            remember_dm_channel(&data, &batch.author, batch.channel);
            let _ = batch.channel.broadcast_typing(&http).await;
            let out = spawn_output(
                Arc::clone(&http),
                Arc::clone(&status),
                batch.channel,
                Some(batch.trigger),
            );
            agent.turn(batch.author, batch.msgs, &out.tx).await;
            out.finish().await;
        }
        for fired in reminders {
            let Some((user, channel)) =
                reminder_target(&data, allowed_user, &aliases, &http, &fired).await
            else {
                warn!(
                    id = %fired.reminder.id,
                    "reminder fired but no target user/DM channel is known; skipping"
                );
                status.record_error(format!(
                    "reminder {} fired with no known target",
                    fired.reminder.id
                ));
                continue;
            };
            let _ = channel.broadcast_typing(&http).await;
            let out = spawn_output(Arc::clone(&http), Arc::clone(&status), channel, None);
            agent.reminder_turn(user, fired.payload, &out.tx).await;
            out.finish().await;
        }
    }
}

/// The output pump for one turn: posts replies, reacts tool emojis when a
/// trigger message exists.
struct Output {
    tx: mpsc::Sender<TurnEvent>,
    task: tokio::task::JoinHandle<()>,
}

impl Output {
    /// Closes the channel and waits for the pump to drain.
    async fn finish(self) {
        drop(self.tx);
        let _ = self.task.await;
    }
}

fn spawn_output(
    http: Arc<serenity::http::Http>,
    status: Arc<Status>,
    channel: ChannelId,
    trigger: Option<MessageId>,
) -> Output {
    let (tx, mut rx) = mpsc::channel::<TurnEvent>(64);
    let task = tokio::spawn(async move {
        let mut reacted: Vec<&'static str> = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                TurnEvent::Reply(text) => {
                    status.log(format!("pester: {text}"));
                    for chunk in split_for_discord(&text) {
                        if let Err(e) = channel.say(&http, chunk).await {
                            warn!("posting to discord: {e}");
                        }
                    }
                }
                TurnEvent::Tool { name, emoji } => {
                    let Some(trigger) = trigger else {
                        continue;
                    };
                    if reacted.contains(&emoji) {
                        continue;
                    }
                    reacted.push(emoji);
                    let reaction = ReactionType::Unicode(emoji.to_owned());
                    if let Err(e) = http.create_reaction(channel, trigger, &reaction).await {
                        warn!(tool = %name, "reacting on discord: {e}");
                    }
                }
            }
        }
    });
    Output { tx, task }
}

/// One row of the persisted per-user DM channel map.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DmEntry {
    name: String,
    channel: u64,
}

fn load_channels(data: &DataDir) -> HashMap<u64, DmEntry> {
    std::fs::read(data.dm_channels_path())
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Persists a user's DM channel (and current name) so reminders can fire
/// into it across restarts.
fn remember_dm_channel(data: &DataDir, user: &UserInfo, channel: ChannelId) {
    let mut channels = load_channels(data);
    let fresh = DmEntry {
        name: user.name.clone(),
        channel: channel.get(),
    };
    let stale = channels
        .get(&user.id)
        .map(|e| e.name != fresh.name || e.channel != fresh.channel)
        .unwrap_or(true);
    if stale {
        channels.insert(user.id, fresh);
        match serde_json::to_vec_pretty(&channels) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(data.dm_channels_path(), bytes) {
                    warn!("persisting DM channel map: {e}");
                }
            }
            Err(e) => warn!("serializing DM channel map: {e}"),
        }
    }
}

/// Resolves who a fired reminder targets and which channel it fires into:
/// the reminder's own user, else the sole known user, else the allowed
/// user; the persisted channel, else a fresh DM.
async fn reminder_target(
    data: &DataDir,
    allowed_user: Option<u64>,
    aliases: &HashMap<u64, String>,
    http: &serenity::http::Http,
    fired: &Fired,
) -> Option<(UserInfo, ChannelId)> {
    let channels = load_channels(data);
    let uid = fired
        .reminder
        .user
        .or_else(|| {
            let mut keys = channels.keys();
            match (keys.next(), keys.next()) {
                (Some(&only), None) => Some(only),
                _ => None,
            }
        })
        .or(allowed_user)?;
    let name = channels
        .get(&uid)
        .map(|e| e.name.clone())
        .or_else(|| aliases.get(&uid).cloned())
        .unwrap_or_else(|| format!("user {uid}"));
    let user = UserInfo { id: uid, name };
    if let Some(entry) = channels.get(&uid) {
        return Some((user, ChannelId::new(entry.channel)));
    }
    match UserId::new(uid).create_dm_channel(http).await {
        Ok(channel) => Some((user, channel.id)),
        Err(e) => {
            warn!("creating DM channel for reminder: {e}");
            None
        }
    }
}

/// Splits `text` into chunks under the Discord message cap, preferring
/// newline boundaries, then spaces, then a hard split. Empty input yields no
/// chunks.
fn split_for_discord(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut rest = text.trim();
    while !rest.is_empty() {
        if rest.chars().count() <= DISCORD_MESSAGE_CAP {
            chunks.push(rest.to_owned());
            break;
        }
        // The byte index of the cap-th char bounds the searchable window.
        let hard = rest
            .char_indices()
            .nth(DISCORD_MESSAGE_CAP)
            .map(|(i, _)| i)
            .unwrap_or(rest.len());
        let window = &rest[..hard];
        let cut = window
            .rfind('\n')
            .or_else(|| window.rfind(' '))
            .filter(|&i| i > 0)
            .unwrap_or(hard);
        chunks.push(rest[..cut].trim_end().to_owned());
        rest = rest[cut..].trim_start();
    }
    chunks.retain(|c| !c.is_empty());
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- the DM gate ---

    #[test]
    fn accepts_dms_from_anyone_when_unrestricted() {
        assert!(accepts(true, false, UserId::new(1), None));
        assert!(accepts(true, false, UserId::new(2), None));
    }

    #[test]
    fn accepts_only_the_allowed_user_when_restricted() {
        let allowed = Some(UserId::new(7));
        assert!(accepts(true, false, UserId::new(7), allowed));
        assert!(!accepts(true, false, UserId::new(8), allowed));
    }

    #[test]
    fn rejects_guild_messages_and_bots() {
        assert!(!accepts(false, false, UserId::new(1), None), "guild message");
        assert!(!accepts(true, true, UserId::new(1), None), "bot author");
    }

    // --- the DM channel map ---

    fn data() -> (tempfile::TempDir, DataDir) {
        let dir = tempfile::tempdir().unwrap();
        let data = DataDir::open(dir.path()).unwrap();
        (dir, data)
    }

    #[test]
    fn dm_channel_map_round_trips() {
        let (_d, data) = data();
        let user = UserInfo {
            id: 7,
            name: "sofia".into(),
        };
        remember_dm_channel(&data, &user, ChannelId::new(999));
        let channels = load_channels(&data);
        assert_eq!(channels[&7].channel, 999);
        assert_eq!(channels[&7].name, "sofia");
    }

    #[test]
    fn dm_channel_map_tracks_renames() {
        let (_d, data) = data();
        let user = UserInfo {
            id: 7,
            name: "old".into(),
        };
        remember_dm_channel(&data, &user, ChannelId::new(999));
        let renamed = UserInfo {
            id: 7,
            name: "new".into(),
        };
        remember_dm_channel(&data, &renamed, ChannelId::new(999));
        assert_eq!(load_channels(&data)[&7].name, "new");
    }

    // --- message splitting ---

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(split_for_discord("hello"), vec!["hello"]);
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(split_for_discord("").is_empty());
        assert!(split_for_discord("   \n ").is_empty());
    }

    #[test]
    fn long_text_splits_under_the_cap_and_loses_nothing() {
        let text = "word ".repeat(1000); // 5000 chars
        let chunks = split_for_discord(&text);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.chars().count() <= DISCORD_MESSAGE_CAP, "{}", c.len());
        }
        let rejoined: Vec<&str> = chunks
            .iter()
            .flat_map(|c| c.split_whitespace())
            .collect();
        assert_eq!(rejoined.len(), 1000);
        assert!(rejoined.iter().all(|w| *w == "word"));
    }

    #[test]
    fn split_prefers_newline_boundaries() {
        let text = format!("{}\n{}", "a".repeat(1500), "b".repeat(1500));
        let chunks = split_for_discord(&text);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].chars().all(|c| c == 'a'));
        assert!(chunks[1].chars().all(|c| c == 'b'));
    }

    #[test]
    fn unsplittable_text_hard_splits_at_the_cap() {
        let text = "x".repeat(4500);
        let chunks = split_for_discord(&text);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.chars().count() <= DISCORD_MESSAGE_CAP));
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 4500);
    }

    #[test]
    fn multibyte_text_splits_on_char_boundaries() {
        let text = "é".repeat(4100);
        let chunks = split_for_discord(&text);
        assert!(chunks.iter().all(|c| c.chars().count() <= DISCORD_MESSAGE_CAP));
        assert_eq!(
            chunks.iter().map(|c| c.chars().count()).sum::<usize>(),
            4100
        );
    }
}
