//! Message timestamp prefixing.
//!
//! The server stamps each incoming message with the moment it arrived, in
//! the server's local time. The prefix is `[HH:MM:SS DD:MM:YY] ` (24-hour
//! clock, two-digit year) followed by the content verbatim. Display paths
//! that show a message back to its sender strip the prefix off again.

use time::OffsetDateTime;

/// The width, in bytes, of a well-formed prefix: `[HH:MM:SS DD:MM:YY] `.
const PREFIX_LEN: usize = 20;

/// The auto-generated note appended to the system prompt, explaining the
/// server-attached timestamp prefix to the model. Written in the model's
/// first person, like the operator's prompt files, and claiming nothing
/// about who the sender is — a shard's incoming message may come from a
/// human, another shard, or the harness.
///
/// The wording is fixed so the cached prefix of the system prompt stays stable
/// across turns.
pub const TIMESTAMP_NOTE: &str = "Server-attached timestamps: Messages reaching me are prefixed with `[HH:MM:SS DD:MM:YY] ` (24-hour clock, two-digit year, server local time). The server adds the prefix at the moment the message arrives; it is real, not a hallucination, and the sender does not see it in their own view. I treat it as ground-truth time.";

/// The prefix attached to user messages by the server.
///
/// Returns `[HH:MM:SS DD:MM:YY] ` for the given instant, with every component
/// zero-padded. The year is the last two digits.
pub fn format_prefix(when: OffsetDateTime) -> String {
    format!(
        "[{:02}:{:02}:{:02} {:02}:{:02}:{:02}] ",
        when.hour(),
        when.minute(),
        when.second(),
        when.day(),
        u8::from(when.month()),
        (when.year().rem_euclid(100)) as u8,
    )
}

/// Wraps a user message body with the timestamp prefix.
pub fn prepend_prefix(when: OffsetDateTime, body: &str) -> String {
    let mut out = format_prefix(when);
    out.push_str(body);
    out
}

/// Strips a leading prefix if one is present and well-formed; otherwise returns
/// the input unchanged.
///
/// A well-formed prefix matches `[HH:MM:SS DD:MM:YY] ` exactly, where each
/// component is digits of the right width. Anything else leaves the input
/// untouched.
pub fn strip_prefix(content: &str) -> &str {
    if has_prefix(content) {
        &content[PREFIX_LEN..]
    } else {
        content
    }
}

/// Recognizes a well-formed `[HH:MM:SS DD:MM:YY] ` prefix at the start of `s`.
///
/// Checks the layout byte by byte: brackets, colons, the space separator, the
/// closing bracket, the trailing space, and digits everywhere else. The digit
/// positions are not range-checked; the server is the only producer and emits
/// valid values.
fn has_prefix(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < PREFIX_LEN {
        return false;
    }
    // Layout of `[HH:MM:SS DD:MM:YY] `:
    //  index: 0123456789...
    //         [ H H : M M : S S _ D D : M M : Y Y ] _
    if b[0] != b'[' {
        return false;
    }
    let digits = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17];
    for &i in &digits {
        if !b[i].is_ascii_digit() {
            return false;
        }
    }
    let colons = [3, 6, 12, 15];
    for &i in &colons {
        if b[i] != b':' {
            return false;
        }
    }
    b[9] == b' ' && b[18] == b']' && b[19] == b' '
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    #[test]
    fn format_prefix_produces_exact_pattern() {
        // 2026-06-03 at 09:07:05 local.
        let when = datetime!(2026-06-03 09:07:05 UTC);
        assert_eq!(format_prefix(when), "[09:07:05 03:06:26] ");
    }

    #[test]
    fn format_prefix_zero_pads_every_component() {
        let when = datetime!(2007-01-02 03:04:05 UTC);
        assert_eq!(format_prefix(when), "[03:04:05 02:01:07] ");
    }

    #[test]
    fn prepend_round_trips_through_strip() {
        let when = datetime!(2026-12-31 23:59:59 UTC);
        for body in [
            "hello",
            "",
            "a message with [brackets] inside",
            "[10:20:30 01:02:03] looks like a prefix but is the body",
            "multi\nline\nbody",
        ] {
            let wrapped = prepend_prefix(when, body);
            assert_eq!(strip_prefix(&wrapped), body, "round trip for {body:?}");
        }
    }

    #[test]
    fn strip_is_noop_without_prefix() {
        assert_eq!(strip_prefix("plain text"), "plain text");
        assert_eq!(strip_prefix(""), "");
    }

    #[test]
    fn strip_is_noop_on_malformed_prefix() {
        // Letters where digits belong.
        assert_eq!(
            strip_prefix("[ab:cd:ef gh:ij:kl] body"),
            "[ab:cd:ef gh:ij:kl] body"
        );
        // Wrong separators.
        assert_eq!(
            strip_prefix("[09-07-05 03-06-26] body"),
            "[09-07-05 03-06-26] body"
        );
        // Missing trailing space.
        assert_eq!(
            strip_prefix("[09:07:05 03:06:26]body"),
            "[09:07:05 03:06:26]body"
        );
        // Missing closing bracket.
        assert_eq!(
            strip_prefix("[09:07:05 03:06:26  body"),
            "[09:07:05 03:06:26  body"
        );
    }

    #[test]
    fn strip_is_noop_on_partial_prefix() {
        assert_eq!(strip_prefix("[09:07:05"), "[09:07:05");
        assert_eq!(strip_prefix("["), "[");
    }

    #[test]
    fn strip_is_noop_on_wrong_width_components() {
        // Three-digit hour shifts the layout, so the colons land on digits.
        assert_eq!(
            strip_prefix("[009:7:5 3:6:26] body"),
            "[009:7:5 3:6:26] body"
        );
    }
}
