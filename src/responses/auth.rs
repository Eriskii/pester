//! Auth for the Responses client: a static bearer token, or a Codex CLI
//! `auth.json` read per call and refreshed on expiry.
//!
//! The configured API key selects the mode by shape: a plain value is a
//! static bearer (the platform `/v1/responses` path), while
//! `codex-auth-file:<path>` points at a Codex CLI auth file — the ChatGPT
//! subscription path, whose access tokens expire and must be refreshed via
//! the OAuth refresh grant. The file is re-read on every request (cheap, and
//! it picks up refreshes the Codex CLI performs on its own), and a refresh
//! writes the new tokens back so the CLI and the daemon share one rotating
//! credential.

use std::path::PathBuf;
#[cfg(test)]
use std::path::Path;

use serde_json::Value;
use tracing::debug;

use crate::client::LlmError;

/// The api-key prefix that selects the Codex auth-file mode.
pub(crate) const AUTH_FILE_PREFIX: &str = "codex-auth-file:";

/// The OAuth client id the Codex CLI is registered under; the refresh grant
/// runs as this client.
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// The token endpoint the refresh grant posts to.
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Where the client's bearer token comes from.
pub(crate) enum AuthSource {
    /// A fixed API key used verbatim.
    Static(String),
    /// A Codex CLI auth file, read per call and refreshed on 401.
    CodexFile(CodexAuthFile),
}

/// Parses the configured api-key string into its auth source. A leading `~`
/// in the file path expands against `$HOME`.
pub(crate) fn parse_api_key(api_key: &str) -> AuthSource {
    match api_key.strip_prefix(AUTH_FILE_PREFIX) {
        Some(path) => AuthSource::CodexFile(CodexAuthFile::new(expand_tilde(path))),
        None => AuthSource::Static(api_key.to_owned()),
    }
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

/// The live credentials one load of the auth file yields.
#[derive(Debug, Clone)]
pub(crate) struct CodexCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub account_id: Option<String>,
}

/// A Codex CLI `auth.json` on disk.
pub(crate) struct CodexAuthFile {
    path: PathBuf,
    token_url: String,
}

impl CodexAuthFile {
    pub(crate) fn new(path: PathBuf) -> Self {
        CodexAuthFile {
            path,
            token_url: TOKEN_URL.to_owned(),
        }
    }

    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Points the refresh grant at a different token endpoint (tests).
    #[cfg(test)]
    pub(crate) fn set_token_url(&mut self, url: String) {
        self.token_url = url;
    }

    /// Reads the current tokens off disk.
    pub(crate) fn load(&self) -> Result<CodexCredentials, LlmError> {
        let text = std::fs::read_to_string(&self.path).map_err(|e| {
            LlmError::Malformed(format!(
                "cannot read codex auth.json at {}: {e}",
                self.path.display()
            ))
        })?;
        let v: Value = serde_json::from_str(&text).map_err(|e| {
            LlmError::Malformed(format!(
                "codex auth.json at {} is not valid JSON: {e}",
                self.path.display()
            ))
        })?;
        let tokens = v
            .get("tokens")
            .ok_or_else(|| LlmError::Malformed("codex auth.json has no tokens block".into()))?;
        let field = |name: &str| -> Result<String, LlmError> {
            tokens
                .get(name)
                .and_then(|x| x.as_str())
                .map(|s| s.to_owned())
                .ok_or_else(|| {
                    LlmError::Malformed(format!("codex auth.json tokens block has no {name}"))
                })
        };
        Ok(CodexCredentials {
            access_token: field("access_token")?,
            refresh_token: field("refresh_token")?,
            account_id: tokens
                .get("account_id")
                .and_then(|x| x.as_str())
                .map(|s| s.to_owned()),
        })
    }

    /// Runs the OAuth refresh grant and writes the rotated tokens back into
    /// the auth file, leaving every sibling field untouched.
    ///
    /// `last_refresh` is deliberately left as-is: this crate carries no
    /// timestamp formatter, and the Codex CLI treats a stale stamp as "refresh
    /// soon", which is harmless.
    pub(crate) async fn refresh(
        &self,
        http: &reqwest::Client,
    ) -> Result<CodexCredentials, LlmError> {
        let current = self.load()?;
        debug!(path = %self.path.display(), "refreshing codex access token");
        let response = http
            .post(&self.token_url)
            .json(&serde_json::json!({
                "client_id": CODEX_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": current.refresh_token,
                "scope": "openid profile email",
            }))
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::Server {
                status: status.as_u16(),
                // The body of a failed grant can echo the refresh token.
                body: body.replace(&current.refresh_token, "<redacted>"),
            });
        }
        let granted: Value = response
            .json()
            .await
            .map_err(|e| LlmError::Malformed(format!("refresh grant response: {e}")))?;
        let access_token = granted
            .get("access_token")
            .and_then(|x| x.as_str())
            .ok_or_else(|| LlmError::Malformed("refresh grant returned no access_token".into()))?
            .to_owned();

        // Rewrite the file: update the token fields, keep everything else.
        let text = std::fs::read_to_string(&self.path).map_err(|e| {
            LlmError::Malformed(format!("re-reading codex auth.json for update: {e}"))
        })?;
        let mut on_disk: Value = serde_json::from_str(&text)
            .map_err(|e| LlmError::Malformed(format!("codex auth.json went malformed: {e}")))?;
        let tokens = on_disk
            .get_mut("tokens")
            .and_then(|t| t.as_object_mut())
            .ok_or_else(|| LlmError::Malformed("codex auth.json lost its tokens block".into()))?;
        tokens.insert("access_token".into(), Value::String(access_token.clone()));
        let mut refresh_token = current.refresh_token.clone();
        if let Some(rt) = granted.get("refresh_token").and_then(|x| x.as_str()) {
            refresh_token = rt.to_owned();
            tokens.insert("refresh_token".into(), Value::String(rt.to_owned()));
        }
        if let Some(idt) = granted.get("id_token").and_then(|x| x.as_str()) {
            tokens.insert("id_token".into(), Value::String(idt.to_owned()));
        }
        std::fs::write(&self.path, on_disk.to_string()).map_err(|e| {
            LlmError::Malformed(format!(
                "writing refreshed tokens to {}: {e}",
                self.path.display()
            ))
        })?;

        Ok(CodexCredentials {
            access_token,
            refresh_token,
            account_id: current.account_id,
        })
    }
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
