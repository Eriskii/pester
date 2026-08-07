//! The system-prompt template engine: comment stripping and `${name}`
//! substitution.
//!
//! A template is plain text carrying two constructs:
//!
//! - `{# ... #}` comments, stripped before substitution. Non-nested, may span
//!   multiple lines. An unterminated `{#` is an error.
//! - `${name}` substitutions, where `name` matches `[A-Za-z0-9_.-]+`. Each
//!   occurrence is replaced with the value the resolver returns for the bare
//!   name. There is no escape syntax and no caching: the resolver is called
//!   once per occurrence.
//!
//! [`render`] runs both passes and returns the resolved text. It does not touch
//! whitespace; callers collapse blank runs separately so empty substitutions do
//! not leave large gaps.

use crate::data::DataError;

/// Names the harness reserves; cannot be shadowed by fragment files.
pub const RESERVED_NAMES: &[&str] = &["timestamp_note", "tools", "model", "user"];

/// Renders a master template by stripping comments and resolving every
/// `${name}`.
///
/// Comments are stripped first, then each `${name}` is replaced with the value
/// the resolver returns for `name` (the bare name, without `${` or `}`). The
/// resolver runs once per occurrence. A resolver error propagates unchanged.
///
/// An unterminated `{#` yields [`DataError::UnterminatedComment`]; a `${` with
/// no closing `}` or an illegal name yields [`DataError::MalformedSubstitution`].
pub fn render(
    template: &str,
    resolver: &dyn Fn(&str) -> Result<String, DataError>,
) -> Result<String, DataError> {
    let stripped = strip_comments(template)?;
    substitute(&stripped, resolver)
}

/// Strips `{# ... #}` comments from `template`.
///
/// Comments do not nest; the first `#}` closes the comment opened by `{#`. An
/// opening `{#` with no matching `#}` yields [`DataError::UnterminatedComment`]
/// carrying the line on which the comment opened.
fn strip_comments(template: &str) -> Result<String, DataError> {
    let bytes = template.as_bytes();
    let mut out = String::with_capacity(template.len());
    let mut i = 0;
    let mut line = 1;
    while i < bytes.len() {
        if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'#' {
            let open_line = line;
            // Skip past `{#` and scan for the closing `#}`.
            i += 2;
            loop {
                if i >= bytes.len() {
                    return Err(DataError::UnterminatedComment { line: open_line });
                }
                if bytes[i] == b'#' && i + 1 < bytes.len() && bytes[i + 1] == b'}' {
                    i += 2;
                    break;
                }
                if bytes[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'\n' {
            line += 1;
        }
        // `template` is valid UTF-8 and we only branch on ASCII bytes, so every
        // non-ASCII byte is a continuation byte we copy through verbatim.
        out.push(bytes[i] as char);
        i += 1;
    }
    Ok(out)
}

/// Replaces every `${name}` in `text` with the resolver's value for `name`.
///
/// `name` matches `[A-Za-z0-9_.-]+`. A `${` with no closing `}`, an empty name,
/// or an illegal character in the name yields
/// [`DataError::MalformedSubstitution`] carrying the line of the `${`.
fn substitute(
    text: &str,
    resolver: &dyn Fn(&str) -> Result<String, DataError>,
) -> Result<String, DataError> {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut line = 1;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            let open_line = line;
            let name_start = i + 2;
            let mut j = name_start;
            while j < bytes.len() && bytes[j] != b'}' {
                let c = bytes[j];
                if !is_name_byte(c) {
                    return Err(DataError::MalformedSubstitution {
                        line: open_line,
                        detail: format!("illegal character {:?} in substitution name", c as char),
                    });
                }
                j += 1;
            }
            if j >= bytes.len() {
                return Err(DataError::MalformedSubstitution {
                    line: open_line,
                    detail: "missing closing `}`".to_owned(),
                });
            }
            if j == name_start {
                return Err(DataError::MalformedSubstitution {
                    line: open_line,
                    detail: "empty substitution name".to_owned(),
                });
            }
            let name = &text[name_start..j];
            out.push_str(&resolver(name)?);
            i = j + 1;
            continue;
        }
        if bytes[i] == b'\n' {
            line += 1;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    Ok(out)
}

/// Recognizes a byte allowed in a substitution name: `[A-Za-z0-9_.-]`.
fn is_name_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'-'
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A resolver that echoes the name in angle brackets, e.g. `foo` -> `<foo>`.
    fn echo(name: &str) -> Result<String, DataError> {
        Ok(format!("<{name}>"))
    }

    #[test]
    fn empty_template_renders_empty() {
        assert_eq!(render("", &echo).unwrap(), "");
    }

    #[test]
    fn no_substitutions_returns_text_verbatim() {
        let text = "plain text\nwith two lines and a $ and a { but no substitution";
        assert_eq!(render(text, &echo).unwrap(), text);
    }

    #[test]
    fn substitution_is_replaced_by_resolver_value() {
        assert_eq!(render("a ${name} b", &echo).unwrap(), "a <name> b");
    }

    #[test]
    fn resolver_is_called_once_per_occurrence() {
        let count = Cell::new(0);
        let counting = |name: &str| -> Result<String, DataError> {
            count.set(count.get() + 1);
            Ok(format!("[{name}]"))
        };
        let out = render("${a} ${a} ${b}", &counting).unwrap();
        assert_eq!(out, "[a] [a] [b]");
        // Three occurrences: no caching, so three calls.
        assert_eq!(count.get(), 3);
    }

    #[test]
    fn resolver_receives_the_bare_name() {
        let seen = Cell::new(String::new());
        let capture = |name: &str| -> Result<String, DataError> {
            seen.set(name.to_owned());
            Ok(String::new())
        };
        render("x ${character.md} y", &capture).unwrap();
        assert_eq!(seen.take(), "character.md");
    }

    #[test]
    fn unknown_substitution_error_propagates() {
        let failing = |name: &str| -> Result<String, DataError> {
            Err(DataError::UnknownSubstitution(name.to_owned()))
        };
        let err = render("before ${unknown} after", &failing).unwrap_err();
        assert!(matches!(err, DataError::UnknownSubstitution(n) if n == "unknown"));
    }

    #[test]
    fn comment_is_stripped() {
        assert_eq!(render("a {# hidden #}b", &echo).unwrap(), "a b");
    }

    #[test]
    fn multiline_comment_is_stripped() {
        assert_eq!(render("a {# multi\nline #}b", &echo).unwrap(), "a b");
    }

    #[test]
    fn comment_strips_before_substitution() {
        // A `${...}` inside a comment never reaches the resolver.
        let failing = |_: &str| -> Result<String, DataError> {
            panic!("resolver must not run for a commented-out substitution");
        };
        assert_eq!(render("{# ${nope} #}ok", &failing).unwrap(), "ok");
    }

    #[test]
    fn unterminated_comment_errors_with_line() {
        let err = render("line one\nline two {# open", &echo).unwrap_err();
        assert!(
            matches!(err, DataError::UnterminatedComment { line: 2 }),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_closing_brace_errors_with_line() {
        let err = render("ok\n${name no close", &echo).unwrap_err();
        match err {
            DataError::MalformedSubstitution { line, .. } => assert_eq!(line, 2),
            other => panic!("expected MalformedSubstitution, got {other:?}"),
        }
    }

    #[test]
    fn illegal_name_character_errors_with_line() {
        let err = render("a ${na me} b", &echo).unwrap_err();
        match err {
            DataError::MalformedSubstitution { line, .. } => assert_eq!(line, 1),
            other => panic!("expected MalformedSubstitution, got {other:?}"),
        }
    }

    #[test]
    fn empty_name_errors() {
        let err = render("${}", &echo).unwrap_err();
        assert!(
            matches!(err, DataError::MalformedSubstitution { line: 1, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn name_charset_allows_dots_dashes_underscores_digits() {
        let out = render("${a.b-c_d9}", &echo).unwrap();
        assert_eq!(out, "<a.b-c_d9>");
    }

    #[test]
    fn reserved_names_are_documented() {
        assert_eq!(RESERVED_NAMES, &["timestamp_note", "tools", "model", "user"]);
    }
}
