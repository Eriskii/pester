//! Shared usage-object parsing for streaming and non-streaming completions.

use crate::core::Usage;

/// Builds a [`Usage`] from a JSON usage object, treating missing fields as
/// zero. Shared between the streaming parser and the non-streaming completion
/// parser so the accounting shape stays consistent.
pub(crate) fn usage_from_json(u: &serde_json::Value) -> Usage {
    let details = u.get("prompt_tokens_details");
    Usage {
        prompt_tokens: count(u, "prompt_tokens"),
        completion_tokens: count(u, "completion_tokens"),
        total_tokens: count(u, "total_tokens"),
        cached_tokens: details.map(|d| count(d, "cached_tokens")).unwrap_or(0),
        cache_write_tokens: details.map(|d| count(d, "cache_write_tokens")).unwrap_or(0),
        cost_usd: u.get("cost").and_then(|c| c.as_f64()),
    }
}

fn count(v: &serde_json::Value, key: &str) -> u32 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0) as u32
}
