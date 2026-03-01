//! Unified storage for all simulation state.
//!
//! [`UnifiedStorage`] composes the storage backends into a single
//! owner with one `advance_tick()` path. Global signals are stored as
//! instance-count-1 SoA entries in `MemberSignalBuffer`, eliminating the
//! split between global and member signals at the storage level.
//!
//! # Sub-storages
//!
//! - [`MemberSignalBuffer`] — SoA double-buffered signals (both global and per-entity)
//! - [`EntityStorage`] — AoS double-buffered entity instance field data
//!
//! Phase executors receive `&mut UnifiedStorage` and access sub-storages
//! via public fields for split-borrow compatibility.

use crate::soa_storage::MemberSignalBuffer;
use crate::storage::EntityStorage;

/// Unified simulation state storage.
///
/// Owns all storage backends and provides a single `advance_tick()`
/// that advances all buffers atomically. Phase executors access individual
/// sub-storages via the public fields.
pub struct UnifiedStorage {
    /// SoA double-buffered signals (global + per-entity member signals).
    pub member_signals: MemberSignalBuffer,
    /// AoS double-buffered entity instance field data.
    pub entities: EntityStorage,
}

impl UnifiedStorage {
    /// Create empty unified storage.
    pub fn new() -> Self {
        Self {
            member_signals: MemberSignalBuffer::new(),
            entities: EntityStorage::default(),
        }
    }

    /// Advance all storage backends to the next tick.
    ///
    /// This is the single tick-advance path that replaces separate
    /// `advance_tick()` calls. Order is: entities, member signals.
    pub fn advance_tick(&mut self) {
        self.entities.advance_tick();
        self.member_signals.advance_tick();
    }
}

impl Default for UnifiedStorage {
    fn default() -> Self {
        Self::new()
    }
}
