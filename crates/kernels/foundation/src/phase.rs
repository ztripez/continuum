//! Execution phases and capability tracking.
//!
//! This module defines the execution lifecycle phases and the capabilities
//! available in each phase. This enables compile-time validation of what
//! operations are allowed in which contexts.
//!
//! # Phases
//!
//! - **Initialization phases** (run once at startup):
//!   - `CollectConfig` — gather config values from scenario
//!   - `Initialize` — set initial signal values
//!   - `WarmUp` — run ticks until convergence
//!
//! - **Tick phases** (run every simulation step):
//!   - `Configure` — finalize per-tick execution context
//!   - `Collect` — gather inputs, apply impulses
//!   - `Resolve` — compute authoritative state
//!   - `Fracture` — detect and respond to tension
//!   - `Measure` — produce observations (fields)
//!   - `Assert` — validate invariants
//!
//! # Capabilities
//!
//! Capabilities represent what context is available during execution:
//! - `Scoping` — access to config/const values
//! - `Signals` — read signal values
//! - `Prev` — previous tick value
//! - `Current` — just-resolved value
//! - `Inputs` — accumulated inputs
//! - `Dt` — time step
//! - `Payload` — impulse payload
//! - `Emit` — emit to signal
//! - `Index` — entity self-reference
//!
//! # Examples
//!
//! ```
//! # use continuum_foundation::phase::*;
//! // Check if a phase is initialization
//! assert!(Phase::Initialize.is_init());
//! assert!(!Phase::Resolve.is_init());
//!
//! // Build a phase set
//! let phases = PhaseSet::empty()
//!     .with(Phase::Resolve)
//!     .with(Phase::Measure);
//! assert!(phases.contains(Phase::Resolve));
//! assert!(!phases.contains(Phase::Configure));
//!
//! // Build a capability set
//! let caps = CapabilitySet::empty()
//!     .with(Capability::Signals)
//!     .with(Capability::Dt);
//! assert!(caps.contains(Capability::Dt));
//! assert!(!caps.contains(Capability::Prev));
//! ```

use serde::{Deserialize, Serialize};

/// Execution phase in the simulation lifecycle.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Phase {
    // === Initialization phases (run once) ===
    /// Gather config values from scenario
    CollectConfig = 0,
    /// Set initial signal values from configs
    Initialize = 1,
    /// Run ticks until convergence (see WarmUpPolicy)
    WarmUp = 2,

    // === Tick phases (run every step) ===
    /// Finalize per-tick execution context
    Configure = 3,
    /// Gather inputs to signals, apply impulses
    Collect = 4,
    /// Compute authoritative state
    Resolve = 5,
    /// Detect and respond to tension
    Fracture = 6,
    /// Produce observations (fields)
    Measure = 7,
    /// Validate invariants (has Prev + Current for deltas)
    Assert = 8,
}

/// Context capability available during execution.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Capability {
    /// Access to config/const values
    Scoping = 0,
    /// Signal read access
    Signals = 1,
    /// Previous tick value
    Prev = 2,
    /// Just-resolved value
    Current = 3,
    /// Accumulated inputs
    Inputs = 4,
    /// Time step
    Dt = 5,
    /// Impulse payload
    Payload = 6,
    /// Emit to signal
    Emit = 7,
    /// Entity self-reference
    Index = 8,
    /// Field read access (observer-only)
    Fields = 9,
}

/// Bitset of phases (compact representation).
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PhaseSet(u16);

/// Bitset of capabilities (compact representation).
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct CapabilitySet(u16);

impl Phase {
    /// Total number of phases
    pub const COUNT: usize = 9;

    /// Check if this is an initialization phase (runs once at startup).
    pub const fn is_init(self) -> bool {
        matches!(self, Self::CollectConfig | Self::Initialize | Self::WarmUp)
    }

    /// Check if this is a tick phase (runs every simulation step).
    pub const fn is_tick(self) -> bool {
        !self.is_init()
    }

    /// Get the phase name as a static string.
    pub const fn name(self) -> &'static str {
        match self {
            Self::CollectConfig => "CollectConfig",
            Self::Initialize => "Initialize",
            Self::WarmUp => "WarmUp",
            Self::Configure => "Configure",
            Self::Collect => "Collect",
            Self::Resolve => "Resolve",
            Self::Fracture => "Fracture",
            Self::Measure => "Measure",
            Self::Assert => "Assert",
        }
    }
}

impl Capability {
    /// Total number of capabilities
    pub const COUNT: usize = 10;

    /// Get the capability name as a static string.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scoping => "Scoping",
            Self::Signals => "Signals",
            Self::Prev => "Prev",
            Self::Current => "Current",
            Self::Inputs => "Inputs",
            Self::Dt => "Dt",
            Self::Payload => "Payload",
            Self::Emit => "Emit",
            Self::Index => "Index",
            Self::Fields => "Fields",
        }
    }
}

impl PhaseSet {
    /// Create an empty phase set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create a phase set containing all phases.
    pub const fn all() -> Self {
        Self((1 << Phase::COUNT) - 1)
    }

    /// Create a phase set with a single phase.
    pub const fn single(phase: Phase) -> Self {
        Self(1 << phase as u16)
    }

    /// Add a phase to this set.
    pub const fn with(self, phase: Phase) -> Self {
        Self(self.0 | (1 << phase as u16))
    }

    /// Check if this set contains a phase.
    pub const fn contains(self, phase: Phase) -> bool {
        (self.0 & (1 << phase as u16)) != 0
    }

    /// Remove a phase from this set.
    pub const fn without(self, phase: Phase) -> Self {
        Self(self.0 & !(1 << phase as u16))
    }

    /// Check if this set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Get the number of phases in this set.
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Compute the union of two phase sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Compute the intersection of two phase sets.
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Check if this set is a subset of another.
    pub const fn is_subset_of(self, other: Self) -> bool {
        (self.0 & other.0) == self.0
    }
}

impl CapabilitySet {
    /// Create an empty capability set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Create a capability set containing all capabilities.
    pub const fn all() -> Self {
        Self((1 << Capability::COUNT) - 1)
    }

    /// Create a capability set with a single capability.
    pub const fn single(cap: Capability) -> Self {
        Self(1 << cap as u16)
    }

    /// Add a capability to this set.
    pub const fn with(self, cap: Capability) -> Self {
        Self(self.0 | (1 << cap as u16))
    }

    /// Check if this set contains a capability.
    pub const fn contains(self, cap: Capability) -> bool {
        (self.0 & (1 << cap as u16)) != 0
    }

    /// Remove a capability from this set.
    pub const fn without(self, cap: Capability) -> Self {
        Self(self.0 & !(1 << cap as u16))
    }

    /// Check if this set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Get the number of capabilities in this set.
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Compute the union of two capability sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Compute the intersection of two capability sets.
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Check if this set is a subset of another.
    pub const fn is_subset_of(self, other: Self) -> bool {
        (self.0 & other.0) == self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_classification() {
        assert!(Phase::CollectConfig.is_init());
        assert!(Phase::Initialize.is_init());
        assert!(Phase::WarmUp.is_init());

        assert!(Phase::Configure.is_tick());
        assert!(Phase::Collect.is_tick());
        assert!(Phase::Resolve.is_tick());
        assert!(Phase::Fracture.is_tick());
        assert!(Phase::Measure.is_tick());
        assert!(Phase::Assert.is_tick());
    }

    #[test]
    fn test_phase_names() {
        assert_eq!(Phase::Resolve.name(), "Resolve");
        assert_eq!(Phase::Measure.name(), "Measure");
    }

    #[test]
    fn test_capability_names() {
        assert_eq!(Capability::Signals.name(), "Signals");
        assert_eq!(Capability::Dt.name(), "Dt");
    }

    #[test]
    fn test_phase_set() {
        let empty = PhaseSet::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let set = PhaseSet::empty().with(Phase::Resolve).with(Phase::Measure);
        assert!(set.contains(Phase::Resolve));
        assert!(set.contains(Phase::Measure));
        assert!(!set.contains(Phase::Configure));
        assert_eq!(set.len(), 2);

        let set2 = set.without(Phase::Resolve);
        assert!(!set2.contains(Phase::Resolve));
        assert!(set2.contains(Phase::Measure));
        assert_eq!(set2.len(), 1);
    }

    #[test]
    fn test_phase_set_operations() {
        let set1 = PhaseSet::empty().with(Phase::Resolve).with(Phase::Measure);
        let set2 = PhaseSet::empty()
            .with(Phase::Measure)
            .with(Phase::Configure);

        let union = set1.union(set2);
        assert!(union.contains(Phase::Resolve));
        assert!(union.contains(Phase::Measure));
        assert!(union.contains(Phase::Configure));
        assert_eq!(union.len(), 3);

        let intersection = set1.intersection(set2);
        assert!(intersection.contains(Phase::Measure));
        assert!(!intersection.contains(Phase::Resolve));
        assert!(!intersection.contains(Phase::Configure));
        assert_eq!(intersection.len(), 1);

        assert!(set1.is_subset_of(union));
        assert!(!union.is_subset_of(set1));
    }

    #[test]
    fn test_capability_set() {
        let empty = CapabilitySet::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let set = CapabilitySet::empty()
            .with(Capability::Signals)
            .with(Capability::Dt);
        assert!(set.contains(Capability::Signals));
        assert!(set.contains(Capability::Dt));
        assert!(!set.contains(Capability::Prev));
        assert_eq!(set.len(), 2);

        let set2 = set.without(Capability::Signals);
        assert!(!set2.contains(Capability::Signals));
        assert!(set2.contains(Capability::Dt));
        assert_eq!(set2.len(), 1);
    }

    #[test]
    fn test_capability_set_operations() {
        let set1 = CapabilitySet::empty()
            .with(Capability::Signals)
            .with(Capability::Prev);
        let set2 = CapabilitySet::empty()
            .with(Capability::Prev)
            .with(Capability::Dt);

        let union = set1.union(set2);
        assert!(union.contains(Capability::Signals));
        assert!(union.contains(Capability::Prev));
        assert!(union.contains(Capability::Dt));
        assert_eq!(union.len(), 3);

        let intersection = set1.intersection(set2);
        assert!(intersection.contains(Capability::Prev));
        assert!(!intersection.contains(Capability::Signals));
        assert!(!intersection.contains(Capability::Dt));
        assert_eq!(intersection.len(), 1);

        assert!(set1.is_subset_of(union));
        assert!(!union.is_subset_of(set1));
    }

    #[test]
    fn test_phase_set_all() {
        let all = PhaseSet::all();
        assert_eq!(all.len(), Phase::COUNT);
        assert!(all.contains(Phase::CollectConfig));
        assert!(all.contains(Phase::Assert));
    }

    #[test]
    fn test_capability_set_all() {
        let all = CapabilitySet::all();
        assert_eq!(all.len(), Capability::COUNT);
        assert!(all.contains(Capability::Scoping));
        assert!(all.contains(Capability::Index));
    }
}
