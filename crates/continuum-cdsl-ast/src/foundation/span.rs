//! Source location tracking for error reporting and debugging.
//!
//! This module provides minimal source tracking structures optimized for size
//! and performance while still supporting rich error messages and IDE integration.
//!
//! # Design
//!
//! - `Span` — compact source location (12 bytes)
//! - `SourceMap` — manages all source files and provides lookup operations
//! - `SourceFile` — single source file with line indexing
//!
//! # Examples
//!
//! ```
//! # use continuum_cdsl::foundation::span::*;
//! # use std::path::PathBuf;
//! let mut map = SourceMap::new();
//! let file_id = map.add_file(PathBuf::from("test.cdsl"), "let x = 42\nlet y = 13".to_string());
//! let span = Span::new(file_id, 0, 10, 1);
//!
//! assert_eq!(map.file_path(&span).to_str(), Some("test.cdsl"));
//! assert_eq!(map.snippet(&span), "let x = 42");
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Compact source location reference.
///
/// Points to a byte range in a source file with cached line number.
/// Total size: 12 bytes (u16 + u32 + u32 + u16 + padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// Index into SourceMap.files
    pub file_id: u16,
    /// Byte offset of start position
    pub start: u32,
    /// Byte offset of end position (exclusive)
    pub end: u32,
    /// Cached line number (1-based) for the start position
    ///
    /// Cached here to avoid lookup in production error messages.
    pub start_line: u16,
}

/// Collection of all source files in a compilation.
///
/// Provides lookup operations for converting Spans into human-readable
/// locations and snippets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

/// A single source file with line indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    /// Absolute or relative path to this file
    pub path: PathBuf,
    /// Original source text
    ///
    /// May be dropped in production builds to save memory.
    pub source: String,
    /// Byte offsets of each line start
    ///
    /// line_starts[0] is always 0 (start of file).
    /// line_starts.len() == number of lines + 1 (includes EOF sentinel).
    pub line_starts: Vec<u32>,
}

impl Span {
    /// Create a new span.
    pub fn new(file_id: u16, start: u32, end: u32, start_line: u16) -> Self {
        Self {
            file_id,
            start,
            end,
            start_line,
        }
    }

    /// Create a zero-length span at the start of a file.
    pub fn zero(file_id: u16) -> Self {
        Self::new(file_id, 0, 0, 1)
    }

    /// Check if this span is zero-length.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Get the length of this span in bytes.
    ///
    /// # Panics
    /// Panics if end < start (malformed span).
    pub fn len(&self) -> u32 {
        assert!(
            self.end >= self.start,
            "malformed span: end ({}) < start ({})",
            self.end,
            self.start
        );
        self.end - self.start
    }

    /// Merge two spans (returns span covering both).
    ///
    /// Panics if spans are from different files.
    pub fn merge(&self, other: &Span) -> Span {
        assert_eq!(
            self.file_id, other.file_id,
            "cannot merge spans from different files"
        );
        Span {
            file_id: self.file_id,
            start: self.start.min(other.start),
            end: self.end.max(other.end),
            start_line: self.start_line.min(other.start_line),
        }
    }

    /// Extend this span to include another span.
    ///
    /// Panics if spans are from different files.
    pub fn extend(&mut self, other: &Span) {
        assert_eq!(
            self.file_id, other.file_id,
            "cannot extend span with span from different file"
        );
        self.start = self.start.min(other.start);
        self.end = self.end.max(other.end);
        self.start_line = self.start_line.min(other.start_line);
    }
}

impl SourceMap {
    /// Create an empty source map.
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// Add a source file and return its ID.
    ///
    /// The line_starts index is computed automatically.
    pub fn add_file(&mut self, path: PathBuf, source: String) -> u16 {
        let file_id = self.files.len();
        assert!(file_id < u16::MAX as usize, "too many source files");

        let line_starts = compute_line_starts(&source);
        self.files.push(SourceFile {
            path,
            source,
            line_starts,
        });

        file_id as u16
    }

    /// Get the source file for a span.
    pub fn file(&self, span: &Span) -> &SourceFile {
        &self.files[span.file_id as usize]
    }

    /// Get the file path for a span.
    pub fn file_path(&self, span: &Span) -> &Path {
        &self.files[span.file_id as usize].path
    }

    /// Get the source snippet for a span.
    pub fn snippet(&self, span: &Span) -> &str {
        let file = &self.files[span.file_id as usize];
        &file.source[span.start as usize..span.end as usize]
    }

    /// Get the (line, column) position for a span's start.
    ///
    /// Both line and column are 1-based.
    pub fn line_col(&self, span: &Span) -> (u32, u32) {
        let file = &self.files[span.file_id as usize];
        file.line_col(span.start)
    }

    /// Get the number of files in this map.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

impl SourceFile {
    /// Create a new source file with precomputed line starts.
    pub fn new(path: PathBuf, source: String) -> Self {
        let line_starts = compute_line_starts(&source);
        Self {
            path,
            source,
            line_starts,
        }
    }

    /// Get (line, column) for a byte offset.
    ///
    /// Both line and column are 1-based.
    ///
    /// # Panics
    /// Panics if offset is beyond EOF.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        assert!(
            offset <= self.source.len() as u32,
            "offset {} is beyond EOF (len = {})",
            offset,
            self.source.len()
        );

        // Binary search to find the line
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx,             // Exact match (start of line)
            Err(idx) => idx.max(1) - 1, // Falls within line idx-1
        };

        let line = (line_idx + 1) as u32; // 1-based line number
        let col = (offset - self.line_starts[line_idx]) + 1; // 1-based column

        (line, col)
    }

    /// Get the byte range for a given line number (1-based).
    ///
    /// Returns None if the line number is out of bounds.
    pub fn line_range(&self, line: u32) -> Option<(u32, u32)> {
        // Valid lines are 1..=(line_starts.len() - 1)
        // since line_starts[N-1] is the EOF sentinel
        if line == 0 || line as usize >= self.line_starts.len() {
            return None;
        }

        let line_idx = (line - 1) as usize;
        let start = self.line_starts[line_idx];
        let end = self.line_starts[line_idx + 1]; // Safe because we checked bounds above

        Some((start, end))
    }

    /// Get the text of a specific line (1-based).
    pub fn line_text(&self, line: u32) -> Option<&str> {
        let (start, end) = self.line_range(line)?;
        Some(&self.source[start as usize..end as usize])
    }

    /// Get the number of lines in this file.
    pub fn line_count(&self) -> usize {
        // line_starts includes EOF sentinel, so count is len - 1
        // Invariant: line_starts always has at least 1 element (byte 0)
        assert!(
            !self.line_starts.is_empty(),
            "line_starts invariant violated: empty array"
        );
        self.line_starts.len() - 1
    }
}

/// Compute byte offsets of line starts in source text.
///
/// Returns a Vec where:
/// - line_starts[0] is byte 0 (start of line 1)
/// - line_starts[i] is the start of line i+1
/// - line_starts[N-1] is EOF (sentinel for last line's end)
///
/// The number of lines is `line_starts.len() - 1`.
fn compute_line_starts(source: &str) -> Vec<u32> {
    let mut line_starts = vec![0]; // First line always starts at 0

    for (idx, ch) in source.char_indices() {
        if ch == '\n' {
            line_starts.push((idx + 1) as u32); // Next line starts after '\n'
        }
    }

    // Always add EOF sentinel (needed to compute the last line's range)
    if line_starts.last() != Some(&(source.len() as u32)) {
        line_starts.push(source.len() as u32);
    }

    line_starts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_creation() {
        let span = Span::new(0, 10, 20, 1);
        assert_eq!(span.file_id, 0);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
        assert_eq!(span.start_line, 1);
        assert_eq!(span.len(), 10);
        assert!(!span.is_empty());

        let empty = Span::zero(0);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_span_merge() {
        let span1 = Span::new(0, 10, 20, 1);
        let span2 = Span::new(0, 15, 30, 1);
        let merged = span1.merge(&span2);

        assert_eq!(merged.start, 10);
        assert_eq!(merged.end, 30);
        assert_eq!(merged.file_id, 0);
    }

    #[test]
    fn test_span_extend() {
        let mut span1 = Span::new(0, 10, 20, 1);
        let span2 = Span::new(0, 15, 30, 1);
        span1.extend(&span2);

        assert_eq!(span1.start, 10);
        assert_eq!(span1.end, 30);
    }

    #[test]
    fn test_compute_line_starts() {
        // Source without trailing newline
        let source = "line 1\nline 2\nline 3";
        let line_starts = compute_line_starts(source);
        // Lines start at: 0, 7, 14, EOF at 20
        assert_eq!(line_starts, vec![0, 7, 14, 20]);

        // Source with trailing newline
        let source_with_trailing = "line 1\nline 2\n";
        let line_starts = compute_line_starts(source_with_trailing);
        // Lines start at: 0, 7, EOF at 14 (after the newline, there's an empty line 3)
        assert_eq!(line_starts, vec![0, 7, 14]);
    }

    #[test]
    fn test_source_file_line_col() {
        let file = SourceFile::new(PathBuf::from("test.cdsl"), "hello\nworld\n".to_string());

        assert_eq!(file.line_col(0), (1, 1)); // 'h'
        assert_eq!(file.line_col(5), (1, 6)); // '\n'
        assert_eq!(file.line_col(6), (2, 1)); // 'w'
        assert_eq!(file.line_col(11), (2, 6)); // '\n'
    }

    #[test]
    fn test_source_file_line_range() {
        let file = SourceFile::new(PathBuf::from("test.cdsl"), "hello\nworld\n".to_string());

        assert_eq!(file.line_range(1), Some((0, 6))); // "hello\n"
        assert_eq!(file.line_range(2), Some((6, 12))); // "world\n"
        assert_eq!(file.line_range(3), None); // Out of bounds
    }

    #[test]
    fn test_source_file_line_text() {
        let file = SourceFile::new(PathBuf::from("test.cdsl"), "hello\nworld\n".to_string());

        assert_eq!(file.line_text(1), Some("hello\n"));
        assert_eq!(file.line_text(2), Some("world\n"));
        assert_eq!(file.line_text(3), None);
    }

    #[test]
    fn test_source_map() {
        let mut map = SourceMap::new();
        let file_id = map.add_file(
            PathBuf::from("test.cdsl"),
            "let x = 42\nlet y = 13".to_string(),
        );

        assert_eq!(map.file_count(), 1);

        let span = Span::new(file_id, 0, 10, 1);
        assert_eq!(map.snippet(&span), "let x = 42");
        assert_eq!(map.file_path(&span).to_str(), Some("test.cdsl"));
        assert_eq!(map.line_col(&span), (1, 1));
    }

    #[test]
    #[should_panic(expected = "malformed span")]
    fn test_span_len_panics_on_inverted() {
        let span = Span::new(0, 10, 5, 1); // end < start
        let _ = span.len();
    }

    #[test]
    #[should_panic(expected = "cannot merge spans from different files")]
    fn test_span_merge_panics_on_different_files() {
        let span1 = Span::new(0, 0, 1, 1);
        let span2 = Span::new(1, 0, 1, 1); // different file_id
        let _ = span1.merge(&span2);
    }

    #[test]
    #[should_panic(expected = "cannot extend span with span from different file")]
    fn test_span_extend_panics_on_different_files() {
        let mut span1 = Span::new(0, 0, 1, 1);
        let span2 = Span::new(1, 0, 1, 1); // different file_id
        span1.extend(&span2);
    }

    #[test]
    #[should_panic(expected = "beyond EOF")]
    fn test_source_file_line_col_panics_on_out_of_bounds() {
        let file = SourceFile::new(PathBuf::from("test.cdsl"), "abc".to_string());
        let _ = file.line_col(4); // offset beyond EOF
    }
}
