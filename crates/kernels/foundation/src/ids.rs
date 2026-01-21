//! Unique identifiers for Continuum entities
//!
//! All simulation entities are identified by typed string wrappers.
//! These ensure type safety and provide consistent serialization.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Represents a hierarchical path used for identifying simulation symbols.
///
/// Paths are composed of dot-separated segments (e.g., \"geophysics.elevation\").
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Path {
    /// Ordered segments of the hierarchical path.
    pub segments: Vec<String>,
}

impl Path {
    /// Creates a new Path from a list of segments.
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    /// Creates a new Path from a dot-separated string.
    pub fn from_path_str(s: &str) -> Self {
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

    /// Joins the path segments into a single string using the specified separator.
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
        Self::from_path_str(s)
    }
}

impl From<String> for Path {
    fn from(s: String) -> Self {
        Self::from_path_str(&s)
    }
}

impl PartialEq<&str> for Path {
    fn eq(&self, other: &&str) -> bool {
        self.segments.join(".") == *other
    }
}

impl PartialEq<String> for Path {
    fn eq(&self, other: &String) -> bool {
        &self.to_string() == other
    }
}

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub Path);

        impl $name {
            /// Creates a new identifier from a path.
            pub fn new(p: impl Into<Path>) -> Self {
                Self(p.into())
            }

            /// Returns the identifier as a string.
            pub fn as_str(&self) -> String {
                self.0.to_string()
            }

            /// Returns a reference to the underlying path.
            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(Path::from_path_str(s))
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(Path::from_path_str(&s))
            }
        }

        impl From<Path> for $name {
            fn from(p: Path) -> Self {
                Self(p)
            }
        }
    };
}

define_id!(
    /// Unique identifier for a signal
    SignalId
);

define_id!(
    /// Unique identifier for a stratum
    StratumId
);

define_id!(
    /// Unique identifier for an era
    EraId
);

define_id!(
    /// Unique identifier for a field
    FieldId
);

define_id!(
    /// Unique identifier for an operator
    OperatorId
);

define_id!(
    /// Unique identifier for an impulse
    ImpulseId
);

define_id!(
    /// Unique identifier for a fracture
    FractureId
);

define_id!(
    /// Unique identifier for a user-defined function
    FnId
);

define_id!(
    /// Unique identifier for an entity type (e.g., "stellar.moon")
    EntityId
);

define_id!(
    /// Unique identifier for an entity instance within an entity type (e.g., "luna")
    InstanceId
);

define_id!(
    /// Unique identifier for a chronicle (observer-only event recording)
    ChronicleId
);

define_id!(
    /// Unique identifier for a custom type definition
    TypeId
);

define_id!(
    /// Unique identifier for a member signal (per-entity state)
    /// Format: entity_path.signal_name (e.g., "human.person.age")
    MemberId
);

define_id!(
    /// Unique identifier for an analyzer (observer-only analysis query)
    AnalyzerId
);
