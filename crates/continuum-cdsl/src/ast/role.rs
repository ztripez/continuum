// Role system - compile-time registry of role capabilities
//
// Roles define what a node *is* and what it can do in each phase.
// This is data-driven via ROLE_REGISTRY, not match-as-polymorphism.

use continuum_foundation::{Capability, CapabilitySet, Phase, PhaseSet};

/// Role identifier - indexing enum, NOT polymorphism
///
/// Each role represents a different kind of primitive (Signal, Field, etc).
/// The role determines:
/// - Which phases the node can execute in
/// - Which capabilities are available in each phase
/// - Whether reconstruction hints are allowed
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
#[repr(u8)]
pub enum RoleId {
    /// Signal - authoritative state resolution
    Signal = 0,
    /// Field - observer data emission
    Field = 1,
    /// Operator - causal transformations
    Operator = 2,
    /// Impulse - external inputs
    Impulse = 3,
    /// Fracture - tension detection
    Fracture = 4,
    /// Chronicle - observer-only interpretation
    Chronicle = 5,
}

impl RoleId {
    /// Total number of roles in the system
    pub const COUNT: usize = 6;

    /// Get the compile-time specification for this role
    #[inline]
    pub const fn spec(self) -> &'static RoleSpec {
        &ROLE_REGISTRY[self as usize]
    }
}

/// Role-specific data - makes invalid states unrepresentable
///
/// Each role can have its own associated data. This ensures that:
/// - Signals can't have reconstruction hints (only Fields have them)
/// - Impulses explicitly declare payload types
/// - Invalid combinations are compile errors
#[derive(Clone, Debug, PartialEq)]
pub enum RoleData {
    /// Signal - authoritative state resolution
    Signal,
    /// Field - observer data emission with optional reconstruction hint
    Field {
        /// Optional hint for how to reconstruct continuous field from samples
        reconstruction: Option<ReconstructionHint>,
    },
    /// Operator - causal transformations
    Operator,
    /// Impulse - external inputs with optional payload type
    Impulse {
        /// Type of payload accepted by this impulse (if any)
        payload: Option<crate::foundation::Type>,
    },
    /// Fracture - tension detection
    Fracture,
    /// Chronicle - observer-only interpretation
    Chronicle,
}

impl RoleData {
    /// Get the RoleId for this role
    pub fn id(&self) -> RoleId {
        match self {
            Self::Signal => RoleId::Signal,
            Self::Field { .. } => RoleId::Field,
            Self::Operator => RoleId::Operator,
            Self::Impulse { .. } => RoleId::Impulse,
            Self::Fracture => RoleId::Fracture,
            Self::Chronicle => RoleId::Chronicle,
        }
    }
}

/// Hint for reconstructing continuous field from discrete samples
///
/// Used by observer layer to interpolate field values between sample points.
/// Examples: Linear, Cubic, IDW (inverse distance weighted), Nearest, etc.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReconstructionHint {
    /// Linear interpolation
    Linear,
    /// Cubic interpolation
    Cubic,
    /// Inverse distance weighted
    Idw,
    /// Nearest neighbor
    Nearest,
}

/// Compile-time specification of role capabilities
///
/// This struct is entirely const - built at compile time via ROLE_REGISTRY.
/// Each role declares:
/// - Its name (for error messages)
/// - Which phases it's allowed to execute in
/// - Which capabilities are available in each phase
/// - Whether reconstruction hints are valid
pub struct RoleSpec {
    /// Role name (for error messages and debugging)
    pub name: &'static str,

    /// Which phases this role can execute in
    pub allowed_phases: PhaseSet,

    /// Capabilities available in each phase
    ///
    /// Indexed by Phase as usize. Empty set if phase not in allowed_phases.
    pub phase_capabilities: [CapabilitySet; Phase::COUNT],

    /// Whether this role supports reconstruction hints
    pub has_reconstruction: bool,
}

/// Compile-time registry of all role specifications
///
/// This is a static array indexed by RoleId as usize.
/// Zero runtime cost - all lookups inline to array access.
pub static ROLE_REGISTRY: [RoleSpec; RoleId::COUNT] = [
    // Signal - resolve authoritative state
    RoleSpec {
        name: "signal",
        allowed_phases: PhaseSet::empty().with(Phase::Resolve).with(Phase::Assert),
        phase_capabilities: phase_caps![
            Phase::Resolve => [Capability::Scoping, Capability::Signals, Capability::Prev, Capability::Inputs, Capability::Dt],
            Phase::Assert => [Capability::Scoping, Capability::Signals, Capability::Prev, Capability::Current, Capability::Dt]
        ],
        has_reconstruction: false,
    },
    // Field - emit observer data
    RoleSpec {
        name: "field",
        allowed_phases: PhaseSet::empty().with(Phase::Measure).with(Phase::Assert),
        phase_capabilities: phase_caps![
            Phase::Measure => [Capability::Scoping, Capability::Signals, Capability::Dt],
            Phase::Assert => [Capability::Scoping, Capability::Signals, Capability::Current, Capability::Dt]
        ],
        has_reconstruction: true,
    },
    // Operator - causal only, no observer phases
    RoleSpec {
        name: "operator",
        allowed_phases: PhaseSet::empty()
            .with(Phase::Configure)
            .with(Phase::Collect)
            .with(Phase::Resolve)
            .with(Phase::Fracture),
        phase_capabilities: phase_caps![
            Phase::Configure => [Capability::Scoping],
            Phase::Collect => [Capability::Scoping, Capability::Signals, Capability::Dt, Capability::Emit],
            Phase::Resolve => [Capability::Scoping, Capability::Signals, Capability::Prev, Capability::Inputs, Capability::Dt],
            Phase::Fracture => [Capability::Scoping, Capability::Signals, Capability::Dt]
        ],
        has_reconstruction: false,
    },
    // Impulse - external causal inputs
    RoleSpec {
        name: "impulse",
        allowed_phases: PhaseSet::empty().with(Phase::Collect),
        phase_capabilities: phase_caps![
            Phase::Collect => [Capability::Scoping, Capability::Signals, Capability::Dt, Capability::Payload, Capability::Emit]
        ],
        has_reconstruction: false,
    },
    // Fracture - emergent tension detection
    RoleSpec {
        name: "fracture",
        allowed_phases: PhaseSet::empty().with(Phase::Fracture).with(Phase::Assert),
        phase_capabilities: phase_caps![
            Phase::Fracture => [Capability::Scoping, Capability::Signals, Capability::Dt],
            Phase::Assert => [Capability::Scoping, Capability::Signals, Capability::Dt]
        ],
        has_reconstruction: false,
    },
    // Chronicle - observer-only, no DAG impact
    RoleSpec {
        name: "chronicle",
        allowed_phases: PhaseSet::empty().with(Phase::Measure),
        phase_capabilities: phase_caps![
            Phase::Measure => [Capability::Scoping, Capability::Signals, Capability::Dt]
        ],
        has_reconstruction: false,
    },
];

/// Macro for building phase_capabilities arrays at compile time
///
/// Usage:
/// ```ignore
/// phase_caps![
///     Phase::Resolve => [Capability::Scoping, Capability::Signals],
///     Phase::Assert => [Capability::Scoping, Capability::Current]
/// ]
/// ```
macro_rules! phase_caps {
    ($($phase:expr => [$($cap:expr),* $(,)?]),* $(,)?) => {{
        let mut caps = [CapabilitySet::empty(); Phase::COUNT];
        $(
            caps[$phase as usize] = CapabilitySet::empty()
                $(.with($cap))*;
        )*
        caps
    }};
}

// Re-export macro for use in this module
pub(crate) use phase_caps;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_id_spec_lookup() {
        // Verify each role can look up its spec
        assert_eq!(RoleId::Signal.spec().name, "signal");
        assert_eq!(RoleId::Field.spec().name, "field");
        assert_eq!(RoleId::Operator.spec().name, "operator");
        assert_eq!(RoleId::Impulse.spec().name, "impulse");
        assert_eq!(RoleId::Fracture.spec().name, "fracture");
        assert_eq!(RoleId::Chronicle.spec().name, "chronicle");
    }

    #[test]
    fn test_signal_allowed_phases() {
        let spec = RoleId::Signal.spec();
        assert!(spec.allowed_phases.contains(Phase::Resolve));
        assert!(spec.allowed_phases.contains(Phase::Assert));
        assert!(!spec.allowed_phases.contains(Phase::Measure));
        assert!(!spec.allowed_phases.contains(Phase::Collect));
    }

    #[test]
    fn test_field_allowed_phases() {
        let spec = RoleId::Field.spec();
        assert!(spec.allowed_phases.contains(Phase::Measure));
        assert!(spec.allowed_phases.contains(Phase::Assert));
        assert!(!spec.allowed_phases.contains(Phase::Resolve));
        assert!(!spec.allowed_phases.contains(Phase::Collect));
    }

    #[test]
    fn test_signal_resolve_capabilities() {
        let spec = RoleId::Signal.spec();
        let caps = spec.phase_capabilities[Phase::Resolve as usize];
        assert!(caps.contains(Capability::Scoping));
        assert!(caps.contains(Capability::Signals));
        assert!(caps.contains(Capability::Prev));
        assert!(caps.contains(Capability::Inputs));
        assert!(caps.contains(Capability::Dt));
        assert!(!caps.contains(Capability::Emit));
        assert!(!caps.contains(Capability::Payload));
    }

    #[test]
    fn test_field_measure_capabilities() {
        let spec = RoleId::Field.spec();
        let caps = spec.phase_capabilities[Phase::Measure as usize];
        assert!(caps.contains(Capability::Scoping));
        assert!(caps.contains(Capability::Signals));
        assert!(caps.contains(Capability::Dt));
        assert!(!caps.contains(Capability::Prev));
        assert!(!caps.contains(Capability::Current));
    }

    #[test]
    fn test_impulse_collect_capabilities() {
        let spec = RoleId::Impulse.spec();
        let caps = spec.phase_capabilities[Phase::Collect as usize];
        assert!(caps.contains(Capability::Scoping));
        assert!(caps.contains(Capability::Signals));
        assert!(caps.contains(Capability::Dt));
        assert!(caps.contains(Capability::Payload));
        assert!(caps.contains(Capability::Emit));
        assert!(!caps.contains(Capability::Prev));
    }

    #[test]
    fn test_reconstruction_hints() {
        assert!(RoleId::Field.spec().has_reconstruction);
        assert!(!RoleId::Signal.spec().has_reconstruction);
        assert!(!RoleId::Operator.spec().has_reconstruction);
        assert!(!RoleId::Impulse.spec().has_reconstruction);
        assert!(!RoleId::Fracture.spec().has_reconstruction);
        assert!(!RoleId::Chronicle.spec().has_reconstruction);
    }

    #[test]
    fn test_role_data_id() {
        assert_eq!(RoleData::Signal.id(), RoleId::Signal);
        assert_eq!(
            RoleData::Field {
                reconstruction: None
            }
            .id(),
            RoleId::Field
        );
        assert_eq!(
            RoleData::Field {
                reconstruction: Some(ReconstructionHint::Linear)
            }
            .id(),
            RoleId::Field
        );
        assert_eq!(RoleData::Operator.id(), RoleId::Operator);
        assert_eq!(RoleData::Impulse { payload: None }.id(), RoleId::Impulse);
        assert_eq!(RoleData::Fracture.id(), RoleId::Fracture);
        assert_eq!(RoleData::Chronicle.id(), RoleId::Chronicle);
    }

    #[test]
    fn test_role_count() {
        assert_eq!(RoleId::COUNT, 6);
        assert_eq!(ROLE_REGISTRY.len(), 6);
    }

    #[test]
    fn test_operator_phases() {
        let spec = RoleId::Operator.spec();
        assert!(spec.allowed_phases.contains(Phase::Configure));
        assert!(spec.allowed_phases.contains(Phase::Collect));
        assert!(spec.allowed_phases.contains(Phase::Resolve));
        assert!(spec.allowed_phases.contains(Phase::Fracture));
        assert!(!spec.allowed_phases.contains(Phase::Measure));
        assert!(!spec.allowed_phases.contains(Phase::Assert));
    }

    #[test]
    fn test_chronicle_observer_only() {
        let spec = RoleId::Chronicle.spec();
        // Chronicles only execute in Measure phase
        assert!(spec.allowed_phases.contains(Phase::Measure));
        assert!(!spec.allowed_phases.contains(Phase::Resolve));
        assert!(!spec.allowed_phases.contains(Phase::Collect));
        assert!(!spec.allowed_phases.contains(Phase::Fracture));
    }
}
