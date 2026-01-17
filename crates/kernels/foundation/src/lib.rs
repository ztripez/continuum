//! Continuum Foundation
//!
//! Core foundational utilities for the Continuum simulation engine.
//! Provides stable hashing, deterministic ID generation, and other
//! primitives required across crates.

pub mod coercion;
pub mod field;
pub mod ids;
pub mod matrix_ops;
pub mod operators;
pub mod primitives;
pub mod rng;
pub mod stable_hash;
pub mod tensor;
pub mod value;
pub mod vector_ops;

// Re-export ID types at crate root
pub use ids::{
    AnalyzerId, ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId,
    MemberId, OperatorId, Path, SignalId, StratumId, TypeId,
};

pub use coercion::{TypeCheckOp, TypeCheckResult, can_operate, type_shape};
pub use field::FieldSample;
pub use operators::{AggregateOp, BinaryOp, UnaryOp};
pub use primitives::{
    PRIMITIVE_TYPES, PrimitiveParamKind, PrimitiveParamSpec, PrimitiveShape, PrimitiveStorageClass,
    PrimitiveTypeDef, PrimitiveTypeId, primitive_type_by_name,
};
pub use value::{FromValue, IntoValue, Mat2, Mat3, Mat4, Quat, Value};

// Re-export stable hash items at crate root
pub use stable_hash::{
    FNV1A_OFFSET_BASIS_64, FNV1A_PRIME_64, fnv1a64, fnv1a64_mix, fnv1a64_path, fnv1a64_str,
};

// Re-export RNG at crate root
pub use rng::RngStream;

/// Simulation timestep in seconds.
///
/// Kernel functions that need dt take this as their last parameter.
pub type Dt = f64;

/// Stratum activation state within an era.
///
/// Strata can be configured to run at different rates or be paused entirely.
/// This allows multi-rate simulation where fast-changing phenomena (weather)
/// run every tick while slow phenomena (geology) run less frequently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StratumState {
    /// Executes every tick. Use for fast-changing phenomena.
    Active,
    /// Executes every N ticks. Use for slower phenomena.
    ActiveWithStride(u32),
    /// Paused entirely; state is preserved but not updated.
    Gated,
}

impl StratumState {
    /// Check if stratum should execute on given tick.
    pub fn is_eligible(&self, tick: u64) -> bool {
        match self {
            StratumState::Active => true,
            StratumState::ActiveWithStride(stride) => tick.is_multiple_of(*stride as u64),
            StratumState::Gated => false,
        }
    }
}

/// Unique identifier for a member signal family.
///
/// A member signal is a "family" of signals - one per entity instance.
/// For example, `member.human.person.age` identifies the family, while
/// a specific person's age is identified by `(signal_id, entity_index)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MemberSignalId {
    /// The entity type this signal belongs to
    pub entity_id: EntityId,
    /// The signal name within the entity
    pub signal_name: String,
}

impl MemberSignalId {
    /// Create a new member signal ID.
    pub fn new(entity_id: EntityId, signal_name: impl Into<String>) -> Self {
        Self {
            entity_id,
            signal_name: signal_name.into(),
        }
    }

    /// Parse from a dot-separated path like "human.person.age".
    ///
    /// The last component is the signal name, the rest form the entity path.
    pub fn from_path(path: &str) -> Option<Self> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.len() < 2 {
            return None;
        }

        let signal_name = parts[parts.len() - 1].to_string();
        let entity_path = parts[..parts.len() - 1].join(".");

        Some(Self {
            entity_id: entity_path.as_str().into(),
            signal_name,
        })
    }
}

impl std::fmt::Display for MemberSignalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "member.{}.{}", self.entity_id, self.signal_name)
    }
}
