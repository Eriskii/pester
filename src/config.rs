//! Configuration read from the environment.
//!
//! Pester speaks exactly one wire format — the OpenAI Responses API — so the
//! provider config is flat: model, base URL, API key (or
//! `codex-auth-file:<path>` for the ChatGPT Codex backend), timeout, and an
//! optional output-token cap. The endpoint config carries the Discord token
//! and channel plus the web UI bind address.

use crate::client::ReasoningEffort;

/// Environment variable names.
pub const MODEL_VAR: &str = "PESTER_MODEL";
pub const BASE_URL_VAR: &str = "PESTER_BASE_URL";
pub const API_KEY_VAR: &str = "PESTER_API_KEY";
pub const TIMEOUT_SECONDS_VAR: &str = "PESTER_TIMEOUT_SECONDS";
pub const MAX_OUTPUT_TOKENS_VAR: &str = "PESTER_MAX_OUTPUT_TOKENS";
pub const REASONING_VAR: &str = "PESTER_REASONING";
pub const DATA_DIR_VAR: &str = "PESTER_DATA_DIR";
pub const DISCORD_TOKEN_VAR: &str = "PESTER_DISCORD_TOKEN";
pub const DISCORD_USER_VAR: &str = "PESTER_DISCORD_USER";
pub const WEB_ADDR_VAR: &str = "PESTER_WEB_ADDR";
pub const IDLE_COMPACT_MINS_VAR: &str = "PESTER_IDLE_COMPACT_MINS";
pub const COMPACT_TOKENS_VAR: &str = "PESTER_COMPACT_TOKENS";
pub const USER_ALIASES_VAR: &str = "PESTER_USER_ALIASES";

/// The default HTTP timeout, in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 600;

/// Errors that arise while reading configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A required environment variable is missing.
    #[error("missing required environment variable: {0}")]
    Missing(&'static str),
    /// An environment variable holds a value that does not parse.
    #[error("invalid value for {var}: {value:?}")]
    Invalid {
        /// The environment variable that holds the malformed value.
        var: &'static str,
        /// The malformed value itself.
        value: String,
    },
}

/// Connection settings for the model provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    /// The model slug, e.g. `gpt-5.6-luna`.
    pub model: String,
    /// The base URL the client posts to. `{base_url}/responses` is the
    /// endpoint.
    pub base_url: String,
    /// The bearer token, or `codex-auth-file:<path>` for the Codex CLI
    /// auth-file mode with OAuth refresh.
    pub api_key: String,
    /// The HTTP timeout, in seconds, applied to every provider call.
    pub timeout_seconds: u64,
    /// An optional cap on `max_output_tokens`.
    pub max_output_tokens: Option<u32>,
}

impl ProviderConfig {
    /// Reads provider configuration from the process environment. Model, base
    /// URL, and API key default to the ChatGPT Codex backend via the Codex
    /// CLI's auth file, so a machine with a logged-in Codex CLI needs zero
    /// provider config.
    pub fn from_env() -> Result<Self, ConfigError> {
        Ok(ProviderConfig {
            model: read_or(MODEL_VAR, "gpt-5.6-luna"),
            base_url: read_or(BASE_URL_VAR, "https://chatgpt.com/backend-api/codex"),
            api_key: read_or(API_KEY_VAR, "codex-auth-file:~/.codex/auth.json"),
            timeout_seconds: read_parsed_or(TIMEOUT_SECONDS_VAR, DEFAULT_TIMEOUT_SECONDS)?,
            max_output_tokens: read_parsed_opt(MAX_OUTPUT_TOKENS_VAR)?,
        })
    }
}

/// Everything else: data dir, reasoning default, Discord, web UI.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// The provider connection settings.
    pub provider: ProviderConfig,
    /// The reasoning effort every turn requests.
    pub reasoning: ReasoningEffort,
    /// The data directory root.
    pub data_dir: String,
    /// The Discord bot token.
    pub discord_token: String,
    /// The Discord user whose DMs the agent answers. `None` accepts DMs
    /// from anyone (the sender's id is logged so it can be pinned later).
    pub discord_user: Option<u64>,
    /// The address the status web UI binds.
    pub web_addr: String,
    /// Minutes of silence after which the idle compactor may fold the
    /// conversation. Past the provider's prompt-cache TTL, compacting while
    /// idle is free: the next turn pays a full re-read either way, so it may
    /// as well re-read a short prompt.
    pub idle_compact_mins: u64,
    /// The prompt-token count past which a turn triggers an immediate
    /// post-turn compaction, mid-conversation cache cost be damned.
    pub compact_tokens: u32,
    /// Operator aliases: Discord user id → the name the agent sees instead
    /// of the Discord username. `PESTER_USER_ALIASES="id:Name,id:Name"`.
    pub user_aliases: std::collections::HashMap<u64, String>,
}

impl AppConfig {
    /// Reads the full app configuration from the process environment. The
    /// caller loads `.env` first; this only reads variables already set.
    pub fn from_env() -> Result<Self, ConfigError> {
        let reasoning = match std::env::var(REASONING_VAR) {
            Ok(s) if !s.is_empty() => {
                s.parse::<ReasoningEffort>()
                    .map_err(|_| ConfigError::Invalid {
                        var: REASONING_VAR,
                        value: s,
                    })?
            }
            _ => ReasoningEffort::Medium,
        };
        Ok(AppConfig {
            provider: ProviderConfig::from_env()?,
            reasoning,
            data_dir: read_or(DATA_DIR_VAR, "data"),
            discord_token: read(DISCORD_TOKEN_VAR)?,
            discord_user: read_parsed_opt(DISCORD_USER_VAR)?,
            web_addr: read_or(WEB_ADDR_VAR, "127.0.0.1:8130"),
            idle_compact_mins: read_parsed_or(IDLE_COMPACT_MINS_VAR, 50)?,
            compact_tokens: read_parsed_or(COMPACT_TOKENS_VAR, 100_000)?,
            user_aliases: read_aliases()?,
        })
    }
}

/// Parses `PESTER_USER_ALIASES="id:Name,id:Name"` into the alias map.
fn read_aliases() -> Result<std::collections::HashMap<u64, String>, ConfigError> {
    let mut aliases = std::collections::HashMap::new();
    let raw = match std::env::var(USER_ALIASES_VAR) {
        Ok(s) if !s.is_empty() => s,
        _ => return Ok(aliases),
    };
    for pair in raw.split(',') {
        let invalid = || ConfigError::Invalid {
            var: USER_ALIASES_VAR,
            value: raw.clone(),
        };
        let (id, name) = pair.split_once(':').ok_or_else(invalid)?;
        let id: u64 = id.trim().parse().map_err(|_| invalid())?;
        let name = name.trim();
        if name.is_empty() {
            return Err(invalid());
        }
        aliases.insert(id, name.to_owned());
    }
    Ok(aliases)
}

fn read(var: &'static str) -> Result<String, ConfigError> {
    match std::env::var(var) {
        Ok(s) if !s.is_empty() => Ok(s),
        _ => Err(ConfigError::Missing(var)),
    }
}

fn read_or(var: &'static str, fallback: &str) -> String {
    match std::env::var(var) {
        Ok(s) if !s.is_empty() => s,
        _ => fallback.to_owned(),
    }
}

fn read_parsed_or<T: std::str::FromStr>(var: &'static str, fallback: T) -> Result<T, ConfigError> {
    match std::env::var(var) {
        Ok(s) if !s.is_empty() => s.parse::<T>().map_err(|_| ConfigError::Invalid {
            var,
            value: s,
        }),
        _ => Ok(fallback),
    }
}

fn read_parsed_opt<T: std::str::FromStr>(var: &'static str) -> Result<Option<T>, ConfigError> {
    match std::env::var(var) {
        Ok(s) if !s.is_empty() => s.parse::<T>().map(Some).map_err(|_| ConfigError::Invalid {
            var,
            value: s,
        }),
        _ => Ok(None),
    }
}
