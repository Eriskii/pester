//! The agent's jailed filesystem tools: read, write, list.
//!
//! Every path the model supplies is interpreted relative to the data
//! directory's `files/` root — to the agent, `files/` *is* `/`. Resolution
//! is purely lexical: a leading `/` is stripped, `.` components are
//! dropped, and any `..` component is rejected outright, so a resolved
//! path is contained in the jail by construction. There is no traversal,
//! ever.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::{str_field, Tool, ToolCtx};

/// Resolves a model-supplied path against the jail root.
///
/// Accepts `notes.md`, `a/b/c.txt`, and `/a/b` (the leading slash is
/// meaningless — the agent thinks it is at `/`, which is the jail root).
/// Rejects any `..` component, empty paths that must name a file, and NUL
/// bytes. Purely lexical: the result is `root/<clean components>`.
fn resolve(root: &Path, raw: &str) -> Result<PathBuf, String> {
    if raw.contains('\0') {
        return Err("path contains a NUL byte".into());
    }
    let mut out = root.to_path_buf();
    for comp in raw.split('/') {
        match comp {
            "" | "." => continue,
            ".." => return Err(format!("path {raw:?} traverses upward; there is no upward")),
            c => out.push(c),
        }
    }
    Ok(out)
}

/// Like [`resolve`], but requires the path to name something (not the root).
fn resolve_file(root: &Path, raw: &str) -> Result<PathBuf, String> {
    let path = resolve(root, raw)?;
    if path == root {
        return Err("path names the root directory, not a file".into());
    }
    Ok(path)
}

/// The largest file `read_file` returns, in bytes.
const MAX_READ_BYTES: u64 = 512 * 1024;

/// Reads a UTF-8 file inside the jail.
pub struct ReadFile {
    root: PathBuf,
}

impl ReadFile {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        ReadFile { root: root.into() }
    }
}

#[async_trait::async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &'static str {
        "read_file"
    }
    fn emoji(&self) -> &'static str {
        "📖"
    }
    fn description(&self) -> &'static str {
        "Read a UTF-8 text file at the given absolute path."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to read, e.g. /notes/today.md"}
            },
            "required": ["path"]
        })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let raw = str_field(&input, "path")?;
        let path = resolve_file(&self.root, &raw)?;
        let meta = tokio::fs::metadata(&path)
            .await
            .map_err(|_| format!("no such file: {raw}"))?;
        if meta.is_dir() {
            return Err(format!("{raw} is a directory; use list_files"));
        }
        if meta.len() > MAX_READ_BYTES {
            return Err(format!(
                "{raw} is {} bytes, over the {MAX_READ_BYTES}-byte read limit",
                meta.len()
            ));
        }
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| format!("reading {raw}: {e}"))?;
        let content = String::from_utf8(bytes).map_err(|_| format!("{raw} is not UTF-8 text"))?;
        Ok(json!({"path": raw, "content": content}))
    }
}

/// Writes a UTF-8 file inside the jail, creating parent directories.
pub struct WriteFile {
    root: PathBuf,
}

impl WriteFile {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        WriteFile { root: root.into() }
    }
}

#[async_trait::async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &'static str {
        "write_file"
    }
    fn emoji(&self) -> &'static str {
        "📝"
    }
    fn description(&self) -> &'static str {
        "Write a UTF-8 text file at the given absolute path, creating parent directories and overwriting any existing file."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to write, e.g. /notes/today.md"},
                "content": {"type": "string", "description": "The full file content."}
            },
            "required": ["path", "content"]
        })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let raw = str_field(&input, "path")?;
        let content = str_field(&input, "content")?;
        let path = resolve_file(&self.root, &raw)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("creating parent directories for {raw}: {e}"))?;
        }
        tokio::fs::write(&path, content.as_bytes())
            .await
            .map_err(|e| format!("writing {raw}: {e}"))?;
        Ok(json!({"path": raw, "bytes": content.len()}))
    }
}

/// Lists a directory inside the jail.
pub struct ListFiles {
    root: PathBuf,
}

impl ListFiles {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        ListFiles { root: root.into() }
    }
}

#[async_trait::async_trait]
impl Tool for ListFiles {
    fn name(&self) -> &'static str {
        "list_files"
    }
    fn emoji(&self) -> &'static str {
        "📂"
    }
    fn description(&self) -> &'static str {
        "List a directory. Omit `path` (or pass \"/\") for the root. Directories are suffixed with /."
    }
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list; defaults to /."}
            }
        })
    }
    async fn call(&self, input: Value, _ctx: &ToolCtx) -> Result<Value, String> {
        let raw = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("/")
            .to_owned();
        let path = resolve(&self.root, &raw)?;
        let mut reader = tokio::fs::read_dir(&path)
            .await
            .map_err(|_| format!("no such directory: {raw}"))?;
        let mut entries = Vec::new();
        while let Ok(Some(entry)) = reader.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry
                .file_type()
                .await
                .map(|t| t.is_dir())
                .unwrap_or(false);
            entries.push(if is_dir { format!("{name}/") } else { name });
        }
        entries.sort();
        Ok(json!({"path": raw, "entries": entries}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolCtx {
        ToolCtx {
            user: crate::core::UserInfo {
                id: 7,
                name: "tester".into(),
            },
        }
    }

    // --- the jail contract: purely lexical, no upward traversal, ever ---

    #[test]
    fn resolve_treats_leading_slash_as_the_jail_root() {
        let root = Path::new("/jail");
        assert_eq!(resolve(root, "/etc/passwd").unwrap(), root.join("etc/passwd"));
    }

    #[test]
    fn resolve_accepts_bare_relative_paths() {
        let root = Path::new("/jail");
        assert_eq!(resolve(root, "notes.md").unwrap(), root.join("notes.md"));
        assert_eq!(resolve(root, "a/b/c.txt").unwrap(), root.join("a/b/c.txt"));
    }

    #[test]
    fn resolve_rejects_any_dotdot_component() {
        let root = Path::new("/jail");
        for raw in ["..", "../x", "a/../../x", "a/..", "/..", "a/b/../c"] {
            assert!(resolve(root, raw).is_err(), "{raw:?} must be rejected");
        }
    }

    #[test]
    fn resolve_skips_dot_and_empty_components() {
        let root = Path::new("/jail");
        assert_eq!(resolve(root, "./a//b/./c").unwrap(), root.join("a/b/c"));
    }

    #[test]
    fn resolve_rejects_nul_bytes() {
        assert!(resolve(Path::new("/jail"), "a\0b").is_err());
    }

    #[test]
    fn resolve_file_rejects_the_bare_root() {
        let root = Path::new("/jail");
        for raw in ["", "/", ".", "//", "./"] {
            assert!(resolve_file(root, raw).is_err(), "{raw:?} must be rejected");
        }
    }

    #[test]
    fn resolved_paths_always_stay_inside_the_jail() {
        // The property the whole module exists for.
        let root = Path::new("/jail");
        for raw in ["/a", "a/b", "/deep/ly/nes/ted", "....", "..hidden", "a.."] {
            let p = resolve(root, raw).unwrap();
            assert!(p.starts_with(root), "{raw:?} resolved outside the jail: {p:?}");
        }
    }

    #[test]
    fn dotdot_lookalikes_are_ordinary_names() {
        // `....`, `..a`, `a..` are legal file names, not traversal.
        let root = Path::new("/jail");
        assert_eq!(resolve(root, "..a").unwrap(), root.join("..a"));
        assert_eq!(resolve(root, "a..").unwrap(), root.join("a.."));
        assert_eq!(resolve(root, "....").unwrap(), root.join("...."));
    }

    // --- the tools over a real tempdir ---

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let w = WriteFile::new(dir.path());
        let r = ReadFile::new(dir.path());
        w.call(json!({"path": "/notes/today.md", "content": "hello jail"}), &ctx())
            .await
            .unwrap();
        let out = r.call(json!({"path": "notes/today.md"}), &ctx()).await.unwrap();
        assert_eq!(out["content"], "hello jail");
    }

    #[tokio::test]
    async fn write_refuses_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let w = WriteFile::new(dir.path());
        let err = w
            .call(json!({"path": "../escape.txt", "content": "nope"}), &ctx())
            .await
            .unwrap_err();
        assert!(err.contains("traverses upward"), "{err}");
        assert!(!dir.path().parent().unwrap().join("escape.txt").exists());
    }

    #[tokio::test]
    async fn read_refuses_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let r = ReadFile::new(dir.path());
        assert!(r.call(json!({"path": "/../etc/passwd"}), &ctx()).await.is_err());
    }

    #[tokio::test]
    async fn read_reports_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let r = ReadFile::new(dir.path());
        let err = r.call(json!({"path": "/nope.txt"}), &ctx()).await.unwrap_err();
        assert!(err.contains("no such file"), "{err}");
    }

    #[tokio::test]
    async fn read_refuses_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let r = ReadFile::new(dir.path());
        let err = r.call(json!({"path": "/sub"}), &ctx()).await.unwrap_err();
        assert!(err.contains("directory"), "{err}");
    }

    #[tokio::test]
    async fn list_shows_files_and_marks_directories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        let l = ListFiles::new(dir.path());
        let out = l.call(json!({}), &ctx()).await.unwrap();
        assert_eq!(out["entries"], json!(["a.txt", "sub/"]));
    }

    #[tokio::test]
    async fn list_defaults_to_the_root_and_accepts_slash() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "x").unwrap();
        let l = ListFiles::new(dir.path());
        let out = l.call(json!({"path": "/"}), &ctx()).await.unwrap();
        assert_eq!(out["entries"], json!(["a.txt"]));
    }

    #[tokio::test]
    async fn list_refuses_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let l = ListFiles::new(dir.path());
        assert!(l.call(json!({"path": ".."}), &ctx()).await.is_err());
    }

    #[tokio::test]
    async fn missing_fields_error_cleanly() {
        let dir = tempfile::tempdir().unwrap();
        let w = WriteFile::new(dir.path());
        let err = w.call(json!({"path": "/x"}), &ctx()).await.unwrap_err();
        assert!(err.contains("content"), "{err}");
    }
}
