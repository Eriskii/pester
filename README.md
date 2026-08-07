# pester

A tiny agent harness that lives in Discord DMs, with a status web UI. The
model is driven through the OpenAI Responses API — either the platform
endpoint with an `sk-...` key or the ChatGPT Codex backend through a Codex
CLI `auth.json` (OAuth tokens re-read per call, refreshed on 401).

## Running

```
cp .env.example .env   # set PESTER_DISCORD_TOKEN (and PESTER_DISCORD_USER to lock DMs to you)
cargo run
```

DMing a bot requires sharing a server with it (or a user-install). Invite
it to any mutual server with
`https://discord.com/oauth2/authorize?client_id=<app-id>&scope=bot`, then
open a DM.

With a logged-in Codex CLI on the machine, the provider needs zero config.
The status page is at http://127.0.0.1:8130.

## Users

Each Discord user gets a wholly separate conversation (transcript,
rolling summary, compaction clock) under `data/conversations/<id>.json`;
everything else — the file jail, the reminders, the prompts — is shared.
The system prompt's `${user}` block names who the conversation is with,
and `PESTER_USER_ALIASES="id:Name,..."` replaces Discord usernames with
operator-chosen names everywhere. Image attachments are downscaled and
ride the transcript as `input_image` parts, so the model sees them.

## Layout

- `data/system/` — the system prompt. `prompt.md` is the master template:
  `${name}` substitutes `system/<name>.md`, `{# ... #}` comments strip,
  reserved names are `${timestamp_note}`, `${tools}`, `${model}`. Re-read
  every turn, so edits are live.
- `data/files/` — the agent's whole filesystem. Its `read_file` /
  `write_file` / `list_files` tools resolve every path inside this jail;
  `..` is rejected outright and a leading `/` means this root.
- `data/transcript.json` — the conversation, restored on restart.

Tools: `read_file`, `write_file`, `list_files`, `get_time`,
`create_reminder`, `list_reminders`, `delete_reminder`.

## Reminders

`data/reminders.json` — hot-editable while running; the scheduler ticks
every 30 s. A fire wakes the agent with the message injected as a
synthetic tool result; the agent decides what to say into the DM. The DM
channel persists to `data/dm_channel`, so reminders fire across restarts.

The agent creates and deletes its own reminders by tool. Entries the
operator hand-writes with `"locked": true` are permanent — the delete
tool refuses them, no matter who asks. Omit `next_fire`; the scheduler
computes it. Example:

```json
[
  {
    "id": "lunch",
    "message": "Lunchtime. Make sure she actually eats something.",
    "locked": true,
    "schedule": { "type": "daily_at", "hour": 12, "minute": 30 }
  },
  {
    "id": "dinner",
    "message": "Dinner check-in.",
    "locked": true,
    "schedule": { "type": "daily_at", "hour": 18, "minute": 30 }
  }
]
```

Schedules: `{"type": "once"}` (fires immediately unless `next_fire` is
set), `{"type": "every_minutes", "minutes": N}`,
`{"type": "daily_at", "hour": H, "minute": M}`, and
`{"type": "weekly_at", "weekday": "sunday", "hour": H, "minute": M}` (all
server-local time). `"remaining": N` caps scheduled fires on a recurring
entry; `"user": <id>` targets the fire.

A fire makes the reminder **active**: unless the agent marks it handled
with `satisfy_reminder`, it re-fires every hour — the nag loop. The idle
compactor skips any conversation with an active reminder, so a context is
never folded and then immediately re-reminded.

## Day-memory

The agent keeps one rolling note per user (`update_day_memory`) — meals,
mood, events — always visible in its context as a `<day_memory>` block.
Shortly after local midnight each note is distilled by the model into
`files/memory/<name>/<date>.md` inside the agent's own file jail (so it
can browse its archive with `read_file`) and cleared for the new day.

## Testing

```
cargo test
```

The Responses client tests spin wiremock servers on loopback; skip them in
network-restricted sandboxes with `cargo test -- --skip http_`.
