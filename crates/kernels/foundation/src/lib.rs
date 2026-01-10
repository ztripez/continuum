//! Continuum Foundation
//!
//! Core foundational utilities for the Continuum simulation engine.
//! Provides stable hashing, deterministic ID generation, and other
//! primitives required across crates.

pub mod ids;
pub mod stable_hash;

// Re-export ID types at crate root
pub use ids::{
    ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId, MemberId,
    OperatorId, SignalId, StratumId, TypeId,
};

// Re-export stable hash items at crate root
pub use stable_hash::{
    fnv1a64, fnv1a64_mix, fnv1a64_path, fnv1a64_str, FNV1A_OFFSET_BASIS_64, FNV1A_PRIME_64,
};

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
