//! The data directory: a root holding `system/` and `files/`.
//!
//! `system/` holds the master template `prompt.md` and the operator's
//! fragment files; `files/` is the agent's whole world — every path the
//! agent's file tools accept resolves inside it. Opening the directory
//! creates any missing subfolder and seeds the default prompt templates
//! when absent. Templates and fragments are re-read from disk on every
//! render, so they are hot-editable at runtime.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use tracing::info;

use crate::template;
use crate::timestamp::TIMESTAMP_NOTE;

/// The master template file name under `system/`.
pub const PROMPT_FILE: &str = "prompt.md";

/// The compaction-summarizer template file name under `system/`.
pub const SUMMARY_PROMPT_FILE: &str = "summary_prompt.md";

/// The day-memory distiller template file name under `system/`.
pub const DAYNOTE_PROMPT_FILE: &str = "daynote_prompt.md";

/// Maximum nesting depth for `${name}` substitutions. Catches cyclic
/// fragments (`identity.md` referencing `${identity}`, etc.).
const MAX_SUBSTITUTION_DEPTH: usize = 8;

/// The default master template, seeded on first open.
const DEFAULT_PROMPT: &str = "\
{# This is the master system-prompt template. `${name}` substitutes
   system/<name>.md (fragments may reference each other); `{# ... #}`
   comments are stripped. Reserved names: ${timestamp_note} explains the
   message timestamp prefix, ${tools} lists the declared tools, ${model}
   is the configured model slug, ${user} identifies who this conversation
   is with. Edits take effect on the next turn. #}
${identity}

${user}

${tools}

${timestamp_note}
";

/// The default identity fragment, seeded on first open.
const DEFAULT_IDENTITY: &str = "\
I am Pester, a small agent living in Discord DMs. My replies go straight
into the conversation, so I keep them conversational and sized for chat.
I have a private file area for notes and working state; paths I use with
my file tools are rooted there.
";

/// The default compaction-summarizer template, seeded on first open.
const DEFAULT_SUMMARY_PROMPT: &str = "\
I am Pester's memory. I fold the older part of a conversation into a
compact rolling summary so the conversation can continue with a short
context. I am given my previous summary (possibly empty) and a span of
messages being folded away; I produce the new summary, and my output is
the summary alone — no preamble, no commentary.

Rules:
- Durable facts come first and survive every fold verbatim: names, who
  the person is, preferences, running jokes, ongoing projects, promises
  I made, anything with a date. I carry these forward unchanged unless
  the conversation contradicts them; I do not re-paraphrase them.
- After the facts, a brief account of how the conversation has gone and
  where it stands: open threads, the current topic, unfinished business.
- I write in first person, plain prose, under about 400 words.
- I keep timestamps on anything time-sensitive.
";

/// The default day-memory distiller template, seeded on first open.
const DEFAULT_DAYNOTE_PROMPT: &str = "\
I am Pester's night archivist. I receive my rolling day-memory about one
person — the running note I kept through the day — and I distill it into
a permanent memory entry for that date. I keep everything that matters
later: what they ate and when, health notes, mood, appointments made or
needed, promises, notable events. I drop chatter and repetition. Plain
prose or terse bullet lines, first person, no preamble — my output is
the memory entry alone.
";

/// Errors that arise while opening or reading the data directory.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// An I/O failure touching a specific path.
    #[error("I/O error at {1}: {0}")]
    Io(#[source] io::Error, PathBuf),

    /// The path exists but is not a directory.
    #[error("not a directory: {0}")]
    NotADirectory(PathBuf),

    /// A system file's contents are not valid UTF-8.
    #[error("non-UTF-8 contents in {0}")]
    Utf8(PathBuf),

    /// The template references a `${name}` that resolves to neither a
    /// reserved built-in nor a fragment file.
    #[error("system prompt references unknown name `{0}`")]
    UnknownSubstitution(String),

    /// A `{# ... #}` comment opened but never closed.
    #[error("unterminated `{{#` comment at line {line}")]
    UnterminatedComment { line: usize },

    /// A `${...}` substitution is malformed: no closing `}`, an empty name,
    /// or an illegal character in the name.
    #[error("malformed `${{...}}` substitution at line {line}: {detail}")]
    MalformedSubstitution { line: usize, detail: String },

    /// Fragment expansion nested deeper than the renderer permits, most
    /// often because a fragment references itself (directly or via a cycle).
    #[error(
        "system prompt substitution recursed past the limit ({max}); check for cyclic references"
    )]
    SubstitutionTooDeep {
        /// The maximum nesting depth the renderer permits.
        max: usize,
    },
}

/// Per-call inputs the system prompt resolver consults for built-in
/// substitutions. New built-ins land here as fields, never as new function
/// parameters.
#[derive(Debug, Clone, Default)]
pub struct PromptContext {
    /// The configured model slug. Backs `${model}`.
    pub model: String,
    /// A human-readable listing of the declared tools, prebuilt by the
    /// registry. Backs `${tools}`.
    pub tools_doc: String,
    /// The block identifying who this conversation is with, prebuilt by
    /// the agent. Backs `${user}`. Empty when no user is in scope (the
    /// summarizer, for instance).
    pub user_doc: String,
}

impl PromptContext {
    /// Builds a context that fills `${model}` with the given slug.
    pub fn with_model(model: impl Into<String>) -> Self {
        PromptContext {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Sets the tool inventory text the template substitutes for `${tools}`.
    pub fn with_tools_doc(mut self, tools_doc: impl Into<String>) -> Self {
        self.tools_doc = tools_doc.into();
        self
    }

    /// Sets the user-identity block the template substitutes for `${user}`.
    pub fn with_user_doc(mut self, user_doc: impl Into<String>) -> Self {
        self.user_doc = user_doc.into();
        self
    }
}

/// The data directory: a root plus its cached subpaths.
#[derive(Debug, Clone)]
pub struct DataDir {
    root: PathBuf,
    system: PathBuf,
    files: PathBuf,
    conversations: PathBuf,
}

impl DataDir {
    /// Opens (or creates) the data directory and its subfolders, seeding the
    /// default templates when absent. A root that exists as a file yields
    /// [`DataError::NotADirectory`].
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DataError> {
        let root = root.into();
        if root.exists() && !root.is_dir() {
            return Err(DataError::NotADirectory(root));
        }
        create_dir_logged(&root)?;

        let system = root.join("system");
        let files = root.join("files");
        let conversations = root.join("conversations");
        for sub in [&system, &files, &conversations] {
            create_dir_logged(sub)?;
        }

        for (name, body) in [
            (PROMPT_FILE, DEFAULT_PROMPT),
            ("identity.md", DEFAULT_IDENTITY),
            (SUMMARY_PROMPT_FILE, DEFAULT_SUMMARY_PROMPT),
            (DAYNOTE_PROMPT_FILE, DEFAULT_DAYNOTE_PROMPT),
        ] {
            let path = system.join(name);
            if !path.exists() {
                fs::write(&path, body).map_err(|e| DataError::Io(e, path.clone()))?;
                info!(path = %path.display(), "seeded default template");
            }
        }

        Ok(DataDir {
            root,
            system,
            files,
            conversations,
        })
    }

    /// The data directory root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The `system/` subfolder holding the master template and fragments.
    pub fn system_dir(&self) -> &Path {
        &self.system
    }

    /// The `files/` subfolder — the agent's jailed filesystem root.
    pub fn files_dir(&self) -> &Path {
        &self.files
    }

    /// The `conversations/` subfolder holding one persisted conversation
    /// per user.
    pub fn conversations_dir(&self) -> &Path {
        &self.conversations
    }

    /// The path one user's conversation persists to.
    pub fn conversation_path(&self, user_id: u64) -> PathBuf {
        self.conversations.join(format!("{user_id}.json"))
    }

    /// The path of the reminder store. Operator-editable by hand: entries
    /// written with `"locked": true` cannot be deleted by the agent.
    pub fn reminders_path(&self) -> PathBuf {
        self.root.join("reminders.json")
    }

    /// The path of the per-user DM channel map, so reminders can fire into
    /// the right conversation across restarts.
    pub fn dm_channels_path(&self) -> PathBuf {
        self.root.join("dm_channels.json")
    }

    /// The path of the rolling day-memory store.
    pub fn daynotes_path(&self) -> PathBuf {
        self.root.join("daynotes.json")
    }

    /// Renders the master template `system/prompt.md` into the full system
    /// prompt: comments stripped, every `${name}` resolved (including those
    /// inside fragments), and runs of three or more newlines collapsed to
    /// two. Re-reads `prompt.md` and its fragments on every call so they can
    /// be hot-edited. A `${name}` resolves reserved built-ins first
    /// (`timestamp_note`, `tools`, `model`), then a fragment file under
    /// `system/` (`name.md`, or `name.ext` literally when `name` carries a
    /// dot).
    pub fn system_prompt(&self, ctx: &PromptContext) -> Result<String, DataError> {
        self.render_template_file(PROMPT_FILE, ctx)
    }

    /// Renders the compaction-summarizer template
    /// `system/summary_prompt.md` through the same pipeline as the main
    /// prompt, so it can reference fragments and built-ins.
    pub fn summarizer_prompt(&self, ctx: &PromptContext) -> Result<String, DataError> {
        self.render_template_file(SUMMARY_PROMPT_FILE, ctx)
    }

    /// Renders the day-memory distiller template
    /// `system/daynote_prompt.md` through the same pipeline.
    pub fn daynote_prompt(&self, ctx: &PromptContext) -> Result<String, DataError> {
        self.render_template_file(DAYNOTE_PROMPT_FILE, ctx)
    }

    /// Reads one template file under `system/` and runs the full render
    /// pipeline: comments stripped, `${name}` resolved recursively, blank
    /// runs collapsed.
    fn render_template_file(
        &self,
        file_name: &str,
        ctx: &PromptContext,
    ) -> Result<String, DataError> {
        let prompt_path = self.system.join(file_name);
        let template =
            fs::read_to_string(&prompt_path).map_err(|e| DataError::Io(e, prompt_path.clone()))?;
        let mut out = self.render_with_depth(&template, ctx, 0)?;
        collapse_blank_runs(&mut out);
        Ok(out)
    }

    /// Renders one template body at a given nesting depth. Each fragment
    /// expansion re-enters with `depth + 1`; past [`MAX_SUBSTITUTION_DEPTH`]
    /// the recursion bails out as [`DataError::SubstitutionTooDeep`].
    fn render_with_depth(
        &self,
        body: &str,
        ctx: &PromptContext,
        depth: usize,
    ) -> Result<String, DataError> {
        if depth > MAX_SUBSTITUTION_DEPTH {
            return Err(DataError::SubstitutionTooDeep {
                max: MAX_SUBSTITUTION_DEPTH,
            });
        }
        let resolver = |name: &str| -> Result<String, DataError> {
            self.resolve_with_depth(ctx, name, depth + 1)
        };
        template::render(body, &resolver)
    }

    /// Resolves a single `${name}` at the given depth. Reserved built-ins
    /// return their values directly; `tools` stays verbatim on purpose (tool
    /// descriptions must never be interpreted as template syntax); fragments
    /// are read from `system/` and themselves rendered.
    fn resolve_with_depth(
        &self,
        ctx: &PromptContext,
        name: &str,
        depth: usize,
    ) -> Result<String, DataError> {
        match name {
            "timestamp_note" => Ok(TIMESTAMP_NOTE.to_owned()),
            "tools" => Ok(ctx.tools_doc.clone()),
            "model" => Ok(ctx.model.clone()),
            "user" => Ok(ctx.user_doc.clone()),
            _ => {
                let body = self.read_fragment(name)?;
                self.render_with_depth(&body, ctx, depth)
            }
        }
    }

    /// Reads the fragment file backing `${name}` as UTF-8 text.
    ///
    /// A `name` carrying a dot maps to `system/{name}` literally; otherwise
    /// `.md` is appended. Hidden files, directories, and symlinks to
    /// non-files are not eligible; an absent or ineligible target yields
    /// [`DataError::UnknownSubstitution`].
    fn read_fragment(&self, name: &str) -> Result<String, DataError> {
        let file_name = if name.contains('.') {
            name.to_owned()
        } else {
            format!("{name}.md")
        };
        if file_name.starts_with('.') {
            return Err(DataError::UnknownSubstitution(name.to_owned()));
        }
        let path = self.system.join(&file_name);
        // `metadata` follows symlinks, so a symlink to a non-file is rejected.
        match fs::metadata(&path) {
            Ok(meta) if meta.is_file() => {
                let bytes = fs::read(&path).map_err(|e| DataError::Io(e, path.clone()))?;
                String::from_utf8(bytes).map_err(|_| DataError::Utf8(path))
            }
            _ => Err(DataError::UnknownSubstitution(name.to_owned())),
        }
    }
}

/// The server's UTC offset, captured once at startup — before the async
/// runtime spawns threads, after which the OS lookup refuses to run.
static LOCAL_OFFSET: std::sync::OnceLock<time::UtcOffset> = std::sync::OnceLock::new();

/// Captures the local UTC offset. Called first thing in `main`, while the
/// process is still single-threaded; later calls are no-ops.
pub fn init_local_offset() {
    let offset = time::UtcOffset::current_local_offset().unwrap_or_else(|_| {
        tracing::warn!("local time offset unavailable; treating local time as UTC");
        time::UtcOffset::UTC
    });
    let _ = LOCAL_OFFSET.set(offset);
}

/// The server's local time: UTC shifted by the startup-captured offset.
/// (A process that runs across a DST switch keeps the old offset until
/// restart.)
pub fn local_now() -> time::OffsetDateTime {
    let offset = LOCAL_OFFSET
        .get()
        .copied()
        .or_else(|| time::UtcOffset::current_local_offset().ok())
        .unwrap_or(time::UtcOffset::UTC);
    time::OffsetDateTime::now_utc().to_offset(offset)
}

/// Today's local date as `YYYY-MM-DD`.
pub fn local_date() -> String {
    let d = local_now().date();
    format!("{:04}-{:02}-{:02}", d.year(), u8::from(d.month()), d.day())
}

/// Creates a directory if it is missing, logging each creation at `info!`.
fn create_dir_logged(path: &Path) -> Result<(), DataError> {
    if path.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(path).map_err(|e| DataError::Io(e, path.to_path_buf()))?;
    info!(path = %path.display(), "created data directory");
    Ok(())
}

/// Collapses runs of three or more consecutive newlines down to exactly two
/// so empty substitutions do not leave large vertical gaps.
fn collapse_blank_runs(text: &mut String) {
    let mut out = String::with_capacity(text.len());
    let mut newlines = 0usize;
    for c in text.chars() {
        if c == '\n' {
            newlines += 1;
            if newlines <= 2 {
                out.push('\n');
            }
        } else {
            newlines = 0;
            out.push(c);
        }
    }
    *text = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Overwrites `prompt.md` with `body` so a test drives the resolver
    /// directly.
    fn write_prompt(data: &DataDir, body: &str) {
        std::fs::write(data.system_dir().join(PROMPT_FILE), body).unwrap();
    }

    #[test]
    fn open_creates_subfolders_and_seeds_templates() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        assert!(data.system_dir().is_dir());
        assert!(data.files_dir().is_dir());
        assert!(data.system_dir().join(PROMPT_FILE).is_file());
        assert!(data.system_dir().join("identity.md").is_file());
    }

    #[test]
    fn open_does_not_clobber_an_existing_template() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        write_prompt(&data, "custom");
        let data = DataDir::open(root.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(data.system_dir().join(PROMPT_FILE)).unwrap(),
            "custom"
        );
    }

    #[test]
    fn open_errors_when_root_is_a_file() {
        let parent = tempfile::tempdir().unwrap();
        let file = parent.path().join("not-a-dir");
        std::fs::write(&file, b"i am a file").unwrap();
        let err = DataDir::open(&file).unwrap_err();
        assert!(matches!(err, DataError::NotADirectory(p) if p == file));
    }

    #[test]
    fn default_prompt_renders_with_a_default_context() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        let out = data.system_prompt(&PromptContext::default()).unwrap();
        assert!(out.contains("I am Pester"), "{out}");
        assert!(out.contains(TIMESTAMP_NOTE), "{out}");
        assert!(!out.contains("${"), "{out}");
        assert!(!out.contains("{#"), "{out}");
    }

    #[test]
    fn built_ins_resolve_to_context_values() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        write_prompt(&data, "m=${model} t=${tools}");
        let ctx = PromptContext::with_model("gpt-5.6-luna").with_tools_doc("TOOLDOC");
        assert_eq!(
            data.system_prompt(&ctx).unwrap(),
            "m=gpt-5.6-luna t=TOOLDOC"
        );
    }

    #[test]
    fn fragments_resolve_recursively() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        std::fs::write(data.system_dir().join("inner.md"), "inner value").unwrap();
        std::fs::write(data.system_dir().join("outer.md"), "<${inner}>").unwrap();
        write_prompt(&data, "${outer}");
        assert_eq!(
            data.system_prompt(&PromptContext::default()).unwrap(),
            "<inner value>"
        );
    }

    #[test]
    fn fragment_may_reference_a_built_in() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        std::fs::write(
            data.system_dir().join("character.md"),
            "I run on ${model}.",
        )
        .unwrap();
        write_prompt(&data, "${character}");
        let ctx = PromptContext::with_model("foo/bar-1.0");
        assert_eq!(data.system_prompt(&ctx).unwrap(), "I run on foo/bar-1.0.");
    }

    #[test]
    fn cyclic_fragment_reference_errors_as_too_deep() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        std::fs::write(data.system_dir().join("loop.md"), "${loop}").unwrap();
        write_prompt(&data, "${loop}");
        let err = data.system_prompt(&PromptContext::default()).unwrap_err();
        assert!(matches!(err, DataError::SubstitutionTooDeep { .. }));
    }

    #[test]
    fn unknown_substitution_errors() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        write_prompt(&data, "${nope}");
        let err = data.system_prompt(&PromptContext::default()).unwrap_err();
        assert!(matches!(err, DataError::UnknownSubstitution(n) if n == "nope"));
    }

    #[test]
    fn reserved_built_in_wins_over_a_shadowing_fragment() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        std::fs::write(data.system_dir().join("timestamp_note.md"), "decoy").unwrap();
        write_prompt(&data, "${timestamp_note}");
        assert_eq!(
            data.system_prompt(&PromptContext::default()).unwrap(),
            TIMESTAMP_NOTE
        );
    }

    #[test]
    fn hidden_fragment_files_are_not_eligible() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        std::fs::write(data.system_dir().join(".secret.md"), "leaked").unwrap();
        write_prompt(&data, "${.secret.md}");
        let err = data.system_prompt(&PromptContext::default()).unwrap_err();
        assert!(matches!(err, DataError::UnknownSubstitution(_)));
    }

    #[test]
    fn blank_runs_collapse_after_empty_substitutions() {
        let root = tempfile::tempdir().unwrap();
        let data = DataDir::open(root.path()).unwrap();
        std::fs::write(data.system_dir().join("a.md"), "A").unwrap();
        std::fs::write(data.system_dir().join("b.md"), "B").unwrap();
        write_prompt(&data, "${a}\n\n${tools}\n\n${b}");
        assert_eq!(
            data.system_prompt(&PromptContext::default()).unwrap(),
            "A\n\nB"
        );
    }
}
