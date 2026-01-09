//! Continuum Foundation
//!
//! Core foundational utilities for the Continuum simulation engine.
//! Provides stable hashing, deterministic ID generation, and other
//! primitives required across crates.

pub mod ids;
pub mod stable_hash;

// Re-export ID types at crate root
pub use ids::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, OperatorId,
    SignalId, StratumId,
};

// Re-export stable hash items at crate root
pub use stable_hash::{
    fnv1a64, fnv1a64_mix, fnv1a64_path, fnv1a64_str, FNV1A_OFFSET_BASIS_64, FNV1A_PRIME_64,
};

/// Simulation timestep in seconds.
///
/// Kernel functions that need dt take this as their last parameter.
pub type Dt = f64;
