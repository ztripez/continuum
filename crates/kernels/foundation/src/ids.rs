//! Unique identifiers for Continuum entities
//!
//! All simulation entities are identified by typed string wrappers.
//! These ensure type safety and provide consistent serialization.

use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Path {
    pub segments: Vec<String>,
}

impl Path {
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments }
    }

    pub fn from_str(s: &str) -> Self {
        Self {
            segments: s.split('.').map(String::from).collect(),
        }
    }

    pub fn join(&self, sep: &str) -> String {
        self.segments.join(sep)
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

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub Path);

        impl $name {
            pub fn new(p: impl Into<Path>) -> Self {
                Self(p.into())
            }

            pub fn as_str(&self) -> String {
                self.0.to_string()
            }

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
                Self(Path::from_str(s))
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(Path::from_str(&s))
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
