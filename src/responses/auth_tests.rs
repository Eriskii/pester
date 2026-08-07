//! Tests for Responses-API auth resolution and the Codex auth-file reader.

use super::{parse_api_key, AuthSource, CodexAuthFile};

#[test]
fn plain_key_is_static_bearer() {
    let AuthSource::Static(k) = parse_api_key("sk-abc123") else {
        panic!("expected static");
    };
    assert_eq!(k, "sk-abc123");
}

#[test]
fn codex_auth_file_prefix_parses_to_file_source() {
    let AuthSource::CodexFile(f) = parse_api_key("codex-auth-file:/tmp/auth.json") else {
        panic!("expected file source");
    };
    assert_eq!(f.path().to_str().unwrap(), "/tmp/auth.json");
}

#[test]
fn tilde_expands_against_home() {
    let home = std::env::var("HOME").expect("HOME set in test env");
    let AuthSource::CodexFile(f) = parse_api_key("codex-auth-file:~/.codex/auth.json") else {
        panic!("expected file source");
    };
    assert_eq!(
        f.path().to_str().unwrap(),
        format!("{home}/.codex/auth.json")
    );
}

#[test]
fn reads_token_and_account_id_from_auth_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auth.json");
    std::fs::write(
        &path,
        serde_json::json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": "tok-live",
                "refresh_token": "ref-live",
                "account_id": "acct-1",
            },
            "last_refresh": "2026-06-03T03:55:20Z",
        })
        .to_string(),
    )
    .unwrap();

    let file = CodexAuthFile::new(path);
    let creds = file.load().expect("loads");
    assert_eq!(creds.access_token, "tok-live");
    assert_eq!(creds.refresh_token, "ref-live");
    assert_eq!(creds.account_id.as_deref(), Some("acct-1"));
}

#[test]
fn missing_file_is_a_clear_error() {
    let file = CodexAuthFile::new("/nonexistent/auth.json".into());
    let err = file.load().unwrap_err();
    assert!(err.to_string().contains("auth.json"), "got: {err}");
}

#[tokio::test]
async fn http_refresh_updates_tokens_in_place() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .and(wiremock::matchers::path("/oauth/token"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "tok-new",
            "refresh_token": "ref-new",
            "id_token": "id-new",
        })))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auth.json");
    std::fs::write(
        &path,
        serde_json::json!({
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": "unrelated",
            "tokens": {
                "access_token": "tok-old",
                "refresh_token": "ref-old",
                "id_token": "id-old",
                "account_id": "acct-1",
            },
        })
        .to_string(),
    )
    .unwrap();

    let mut file = CodexAuthFile::new(path.clone());
    file.set_token_url(format!("{}/oauth/token", server.uri()));
    let http = reqwest::Client::new();
    let creds = file.refresh(&http).await.expect("refresh succeeds");
    assert_eq!(creds.access_token, "tok-new");

    // The file is rewritten with the new tokens and untouched siblings.
    let on_disk: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(on_disk["tokens"]["access_token"], "tok-new");
    assert_eq!(on_disk["tokens"]["refresh_token"], "ref-new");
    assert_eq!(on_disk["tokens"]["id_token"], "id-new");
    assert_eq!(on_disk["tokens"]["account_id"], "acct-1");
    assert_eq!(on_disk["OPENAI_API_KEY"], "unrelated");
}

#[tokio::test]
async fn http_refresh_failure_is_an_error_and_leaves_file_alone() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("POST"))
        .respond_with(wiremock::ResponseTemplate::new(400).set_body_string("bad grant"))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auth.json");
    let original = serde_json::json!({
        "tokens": {"access_token": "tok-old", "refresh_token": "ref-old", "account_id": null},
    })
    .to_string();
    std::fs::write(&path, &original).unwrap();

    let mut file = CodexAuthFile::new(path.clone());
    file.set_token_url(format!("{}/oauth/token", server.uri()));
    let http = reqwest::Client::new();
    assert!(file.refresh(&http).await.is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
}
