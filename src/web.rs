//! The status web UI: one HTML page and one JSON endpoint.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{Html, Json};
use axum::routing::get;
use axum::Router;
use tracing::info;

use crate::status::Status;

/// Builds the router: `/` renders the status page, `/status.json` serves the
/// snapshot.
pub fn router(status: Arc<Status>) -> Router {
    Router::new()
        .route("/", get(page))
        .route("/status.json", get(json))
        .with_state(status)
}

/// Serves the web UI forever on `addr`.
pub async fn serve(status: Arc<Status>, addr: &str) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(addr, "web ui listening");
    axum::serve(listener, router(status)).await
}

async fn json(State(status): State<Arc<Status>>) -> Json<crate::status::Snapshot> {
    Json(status.snapshot())
}

async fn page(State(status): State<Arc<Status>>) -> Html<String> {
    let s = status.snapshot();
    let log_rows: String = s
        .log
        .iter()
        .rev()
        .map(|(t, l)| {
            format!(
                "<tr><td class=\"t\">{}</td><td>{}</td></tr>",
                escape(t),
                escape(l)
            )
        })
        .collect();
    let error = match &s.last_error {
        Some(e) => format!("<p class=\"err\">last error: {}</p>", escape(e)),
        None => String::new(),
    };
    let cached = if s.usage.prompt_tokens > 0 {
        format!(
            " ({}% cached)",
            s.usage.cached_tokens * 100 / s.usage.prompt_tokens.max(1)
        )
    } else {
        String::new()
    };
    Html(format!(
        r#"<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<title>pester</title>
<style>
body {{ font-family: monospace; max-width: 60rem; margin: 2rem auto; padding: 0 1rem;
        background: #14151a; color: #d8dce3; }}
h1 {{ font-size: 1.2rem; }} .state {{ color: #8fd18f; }} .err {{ color: #e08f8f; }}
table {{ border-collapse: collapse; width: 100%; }} td {{ padding: 2px 8px;
        border-bottom: 1px solid #24262e; vertical-align: top; }}
.t {{ color: #7a8091; white-space: nowrap; }}
dt {{ color: #7a8091; float: left; clear: left; width: 10rem; }} dd {{ margin: 0 0 4px 10rem; }}
</style></head><body>
<h1>pester <span class="state">[{state}]</span></h1>
<dl>
<dt>model</dt><dd>{model}</dd>
<dt>started</dt><dd>{started}</dd>
<dt>turns</dt><dd>{turns}</dd>
<dt>transcript</dt><dd>{tlen} messages</dd>
<dt>tokens</dt><dd>{prompt} in{cached} / {completion} out</dd>
</dl>
{error}
<h1>activity</h1>
<table>{log_rows}</table>
</body></html>"#,
        state = escape(&s.state),
        model = escape(&s.model),
        started = escape(&s.started_at),
        turns = s.turns,
        tlen = s.transcript_len,
        prompt = s.usage.prompt_tokens,
        completion = s.usage.completion_tokens,
    ))
}

/// Minimal HTML escaping for text nodes.
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_neutralizes_markup() {
        assert_eq!(escape("<b>&"), "&lt;b&gt;&amp;");
    }

    #[tokio::test]
    async fn page_renders_state_and_escapes_errors() {
        let status = Arc::new(Status::new("m"));
        status.record_error("<script>boom</script>");
        let Html(body) = page(State(status)).await;
        assert!(body.contains("[idle]"));
        assert!(!body.contains("<script>boom"));
        assert!(body.contains("&lt;script&gt;boom"));
    }
}
