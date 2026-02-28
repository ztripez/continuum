//! Unified storage for all simulation state.
//!
//! [`UnifiedStorage`] composes the three storage backends into a single
//! owner with one `advance_tick()` path. This eliminates the split between
//! global signals, member signals, and entity instances at the storage level.
//!
//! # Sub-storages
//!
//! - [`SignalStorage`] — double-buffered global signal values (key-value)
//! - [`MemberSignalBuffer`] — SoA double-buffered per-entity member signals
//! - [`EntityStorage`] — AoS double-buffered entity instance field data
//!
//! Phase executors receive `&mut UnifiedStorage` and access sub-storages
//! via public fields for split-borrow compatibility.

use crate::soa_storage::MemberSignalBuffer;
use crate::storage::{EntityStorage, SignalStorage};

/// Unified simulation state storage.
///
/// Owns all three storage backends and provides a single `advance_tick()`
/// that advances all buffers atomically. Phase executors access individual
/// sub-storages via the public fields.
pub struct UnifiedStorage {
    /// Double-buffered global signal values (key-value).
    pub signals: SignalStorage,
    /// SoA double-buffered per-entity member signals.
    pub member_signals: MemberSignalBuffer,
    /// AoS double-buffered entity instance field data.
    pub entities: EntityStorage,
}

impl UnifiedStorage {
    /// Create empty unified storage.
    pub fn new() -> Self {
        Self {
            signals: SignalStorage::default(),
            member_signals: MemberSignalBuffer::new(),
            entities: EntityStorage::default(),
        }
    }

    /// Advance all storage backends to the next tick.
    ///
    /// This is the single tick-advance path that replaces three separate
    /// `advance_tick()` calls. Order is: signals, entities, member signals.
    pub fn advance_tick(&mut self) {
        self.signals.advance_tick();
        self.entities.advance_tick();
        self.member_signals.advance_tick();
    }
}

impl Default for UnifiedStorage {
    fn default() -> Self {
        Self::new()
    }
}
