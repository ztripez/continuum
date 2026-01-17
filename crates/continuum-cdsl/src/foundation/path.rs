//! Path representation for namespaced identifiers
//!
//! Paths are dot-separated identifiers used throughout the DSL:
//! - `signal.atmosphere.temperature`
//! - `entity.stellar.planet`
//! - `fn.physics.gravity_force`
//!
//! The compiler uses Path extensively for name resolution and scoping.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A hierarchical path for identifiers in the DSL.
///
/// Paths are immutable and support efficient comparison and hashing.
/// They are used as keys in symbol tables and for name resolution.
///
/// # Examples
///
/// ```
/// # use continuum_cdsl::foundation::Path;
/// let path = Path::from("atmosphere.temperature");
/// assert_eq!(path.segments(), &["atmosphere", "temperature"]);
/// assert_eq!(path.to_string(), "atmosphere.temperature");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Path {
    segments: Vec<String>,
}

impl Path {
    /// Create a new path from a vector of segments.
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    /// Parse a path from a dot-separated string.
    pub fn from_str(s: &str) -> Self {
        Self {
            segments: s.split('.').map(String::from).collect(),
        }
    }

    /// Get the path segments.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Get the number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Get the first segment (namespace or root).
    pub fn first(&self) -> Option<&str> {
        self.segments.first().map(String::as_str)
    }

    /// Get the last segment (leaf name).
    pub fn last(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    /// Join segments with a separator.
    pub fn join(&self, sep: &str) -> String {
        self.segments.join(sep)
    }

    /// Append a segment to create a new path.
    pub fn append(&self, segment: impl Into<String>) -> Self {
        let mut new_segments = self.segments.clone();
        new_segments.push(segment.into());
        Self::new(new_segments)
    }

    /// Get the parent path (all segments except the last).
    ///
    /// Returns None if this is a single-segment path.
    pub fn parent(&self) -> Option<Self> {
        if self.segments.len() <= 1 {
            None
        } else {
            Some(Self::new(self.segments[..self.segments.len() - 1].to_vec()))
        }
    }

    /// Check if this path starts with another path.
    pub fn starts_with(&self, prefix: &Path) -> bool {
        self.segments.starts_with(&prefix.segments)
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.segments.join("."))
    }
}

impl From<&str> for Path {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

impl From<String> for Path {
    fn from(s: String) -> Self {
        Self::from_str(&s)
    }
}

impl PartialEq<&str> for Path {
    fn eq(&self, other: &&str) -> bool {
        self.to_string() == *other
    }
}

impl PartialEq<String> for Path {
    fn eq(&self, other: &String) -> bool {
        &self.to_string() == other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_creation() {
        let path = Path::from("a.b.c");
        assert_eq!(path.segments(), &["a", "b", "c"]);
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_path_display() {
        let path = Path::from("signal.temperature");
        assert_eq!(path.to_string(), "signal.temperature");
    }

    #[test]
    fn test_path_first_last() {
        let path = Path::from("a.b.c");
        assert_eq!(path.first(), Some("a"));
        assert_eq!(path.last(), Some("c"));
    }

    #[test]
    fn test_path_parent() {
        let path = Path::from("a.b.c");
        let parent = path.parent().unwrap();
        assert_eq!(parent.to_string(), "a.b");

        let single = Path::from("a");
        assert!(single.parent().is_none());
    }

    #[test]
    fn test_path_append() {
        let path = Path::from("a.b");
        let extended = path.append("c");
        assert_eq!(extended.to_string(), "a.b.c");
    }

    #[test]
    fn test_path_starts_with() {
        let path = Path::from("a.b.c.d");
        let prefix = Path::from("a.b");
        assert!(path.starts_with(&prefix));

        let non_prefix = Path::from("a.x");
        assert!(!path.starts_with(&non_prefix));
    }
}
