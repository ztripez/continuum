//! Unique identifiers for Continuum entities
//!
//! All simulation entities are identified by typed string wrappers.
//! These ensure type safety and provide consistent serialization.

use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
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
