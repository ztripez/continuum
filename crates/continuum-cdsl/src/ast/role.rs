//! Role system - compile-time registry of role capabilities
//!
//! This module defines the role system that governs what each AST node can do.
//! Roles (Signal, Field, Operator, etc) determine:
//! - Which execution phases the node participates in
//! - Which capabilities (context features) are available in each phase
//! - What role-specific data the node can carry
//!
//! # Design Principles
//!
//! 1. **Data-driven, not polymorphism** - Role rules live in a static registry
//!    (`ROLE_REGISTRY`), not in match statements scattered through the compiler.
//!
//! 2. **Compile-time validation** - Phase and capability rules are enforced at
//!    compile time via const arrays and bitsets. Zero runtime cost.
//!
//! 3. **Invalid states unrepresentable** - `RoleData` is an enum where each variant
//!    carries only the data valid for that role (e.g., only Fields have reconstruction hints).
//!
//! # Role Specifications
//!
//! Each role has a `RoleSpec` entry in `ROLE_REGISTRY` that defines:
//! - `allowed_phases`: Which phases this role can execute in (e.g., Signal can execute in Resolve and Assert)
//! - `phase_capabilities`: For each phase, which capabilities are available (e.g., Signal in Resolve gets Scoping, Signals, Prev, Inputs, Dt)
//! - `has_reconstruction`: Whether reconstruction hints are valid (only true for Field)
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{RoleId, RoleData, ReconstructionHint};
//!
//! // Look up role capabilities
//! let spec = RoleId::Signal.spec();
//! assert!(spec.allowed_phases.contains(Phase::Resolve));
//!
//! // Create role data with role-specific fields
//! let field_role = RoleData::Field {
//!     reconstruction: Some(ReconstructionHint {
//!         domain: Domain::Cartesian,
//!         method: InterpolationMethod::Linear,
//!         boundary: BoundaryCondition::Clamp,
//!         conservative: false,
//!     }),
//! };
//! assert_eq!(field_role.id(), RoleId::Field);
//! ```

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
    ///
    /// Returns a static reference to the role's specification in the
    /// compile-time registry. This is a zero-cost operation - it compiles
    /// to a simple array index.
    ///
    /// # Returns
    ///
    /// Static reference to the RoleSpec for this role, containing:
    /// - Allowed execution phases
    /// - Capabilities available in each phase
    /// - Whether reconstruction hints are supported
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
    ///
    /// # Returns
    ///
    /// The RoleId enum value corresponding to this role data variant
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
/// Specifies how to interpolate field values between sample points during
/// observer queries. This is a compile-time hint - the observer layer may
/// choose the best available method based on capabilities.
///
/// # Structure
///
/// - `domain`: Distance metric and coordinate system
/// - `method`: Interpolation kernel to apply
/// - `boundary`: How to handle queries outside sample domain
/// - `conservative`: Whether to preserve integrals (for flux fields)
///
/// # Examples
///
/// ```rust,ignore
/// // Simple linear interpolation in Cartesian space
/// ReconstructionHint {
///     domain: Domain::Cartesian,
///     method: InterpolationMethod::Linear,
///     boundary: BoundaryCondition::Clamp,
///     conservative: false,
/// }
///
/// // Geodesic interpolation on sphere (for planetary fields)
/// ReconstructionHint {
///     domain: Domain::Spherical { radius: 6371e3 },
///     method: InterpolationMethod::NaturalNeighbor,
///     boundary: BoundaryCondition::NoBoundary,
///     conservative: false,
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ReconstructionHint {
    /// Distance metric and coordinate system
    pub domain: Domain,

    /// Interpolation kernel to apply
    pub method: InterpolationMethod,

    /// How to handle queries outside sample domain
    pub boundary: BoundaryCondition,

    /// Whether to preserve integrals (for flux fields like mass, energy)
    pub conservative: bool,
}

/// Domain determines distance metric and coordinate handling
#[derive(Clone, Debug, PartialEq)]
pub enum Domain {
    /// Euclidean distance in R^n
    Cartesian,

    /// Geodesic (great-circle) distance on sphere
    Spherical {
        /// Sphere radius in meters
        radius: f64,
    },
}

/// Interpolation kernel applied using domain's distance metric
#[derive(Clone, Debug, PartialEq)]
pub enum InterpolationMethod {
    // === Basic ===
    /// Closest sample (C0 discontinuous)
    NearestNeighbor,

    /// Linear blend by distance (C0 continuous)
    Linear,

    /// Cubic spline (C2 continuous)
    Cubic,

    // === Scattered Data (weighted) ===
    /// Inverse distance weighting
    Idw {
        /// Power parameter (typically 2.0)
        power: f64,
    },

    /// Radial basis functions
    Rbf {
        /// RBF kernel type
        kernel: RbfKernel,
    },

    /// Voronoi-based, C1 continuous
    NaturalNeighbor,

    // === Geostatistical ===
    /// Optimal for spatially correlated data
    Kriging {
        /// Variogram model
        variogram: Variogram,
    },

    // === Spectral (global) ===
    /// For spherical domains (global basis)
    SphericalHarmonics {
        /// Maximum degree of expansion
        max_degree: u32,
    },

    // === Local Approximation ===
    /// Moving least squares
    Mls {
        /// Polynomial degree
        degree: u8,
    },
}

/// RBF kernel types for radial basis function interpolation
#[derive(Clone, Debug, PartialEq)]
pub enum RbfKernel {
    /// Gaussian kernel: exp(-r²/ε²)
    Gaussian {
        /// Shape parameter ε
        epsilon: f64,
    },

    /// Multiquadric: sqrt(1 + (r/ε)²)
    Multiquadric {
        /// Shape parameter ε
        epsilon: f64,
    },

    /// Inverse multiquadric: 1/sqrt(1 + (r/ε)²)
    InverseMultiquadric {
        /// Shape parameter ε
        epsilon: f64,
    },

    /// Thin plate spline: r² ln(r)
    ThinPlateSpline,
}

/// Variogram model for kriging interpolation
#[derive(Clone, Debug, PartialEq)]
pub enum Variogram {
    /// Exponential variogram
    Exponential {
        /// Sill (plateau value)
        sill: f64,
        /// Range (correlation distance)
        range: f64,
        /// Nugget (measurement error variance)
        nugget: f64,
    },

    /// Gaussian variogram
    Gaussian {
        /// Sill (plateau value)
        sill: f64,
        /// Range (correlation distance)
        range: f64,
        /// Nugget (measurement error variance)
        nugget: f64,
    },

    /// Spherical variogram
    Spherical {
        /// Sill (plateau value)
        sill: f64,
        /// Range (correlation distance)
        range: f64,
        /// Nugget (measurement error variance)
        nugget: f64,
    },
}

/// Boundary condition for queries outside sample domain
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryCondition {
    // === Coordinate-based ===
    /// Clamp to nearest edge value
    Clamp,

    /// Periodic (e.g., longitude wrapping)
    Wrap,

    /// Reflect at boundary
    Mirror,

    /// Closed manifold (e.g., sphere has no boundary)
    NoBoundary,

    // === PDE boundary conditions ===
    /// Fixed value at boundary
    Dirichlet {
        /// Boundary value
        value: f64,
    },

    /// Fixed gradient at boundary
    Neumann {
        /// Boundary gradient
        gradient: f64,
    },

    /// Robin condition: α*u + β*∂u/∂n = γ
    Robin {
        /// Coefficient α
        alpha: f64,
        /// Coefficient β
        beta: f64,
    },
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
                reconstruction: Some(ReconstructionHint {
                    domain: Domain::Cartesian,
                    method: InterpolationMethod::Linear,
                    boundary: BoundaryCondition::Clamp,
                    conservative: false,
                })
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

    #[test]
    fn test_disallowed_phases_have_no_capabilities() {
        // Signal not allowed in Measure -> capabilities should be empty
        let spec = RoleId::Signal.spec();
        let measure_caps = spec.phase_capabilities[Phase::Measure as usize];
        assert!(measure_caps.is_empty());

        // Field not allowed in Resolve -> capabilities should be empty
        let spec = RoleId::Field.spec();
        let resolve_caps = spec.phase_capabilities[Phase::Resolve as usize];
        assert!(resolve_caps.is_empty());

        // Impulse not allowed in Resolve -> capabilities should be empty
        let spec = RoleId::Impulse.spec();
        let resolve_caps = spec.phase_capabilities[Phase::Resolve as usize];
        assert!(resolve_caps.is_empty());
    }

    #[test]
    fn test_fracture_capabilities() {
        let spec = RoleId::Fracture.spec();

        // Fracture phase capabilities
        let fracture_caps = spec.phase_capabilities[Phase::Fracture as usize];
        assert!(fracture_caps.contains(Capability::Scoping));
        assert!(fracture_caps.contains(Capability::Signals));
        assert!(fracture_caps.contains(Capability::Dt));
        assert!(!fracture_caps.contains(Capability::Emit));
        assert!(!fracture_caps.contains(Capability::Prev));

        // Assert phase capabilities
        let assert_caps = spec.phase_capabilities[Phase::Assert as usize];
        assert!(assert_caps.contains(Capability::Scoping));
        assert!(assert_caps.contains(Capability::Signals));
        assert!(assert_caps.contains(Capability::Dt));
    }

    #[test]
    fn test_chronicle_capabilities() {
        let spec = RoleId::Chronicle.spec();
        let measure_caps = spec.phase_capabilities[Phase::Measure as usize];
        assert!(measure_caps.contains(Capability::Scoping));
        assert!(measure_caps.contains(Capability::Signals));
        assert!(measure_caps.contains(Capability::Dt));
        assert!(!measure_caps.contains(Capability::Prev));
        assert!(!measure_caps.contains(Capability::Emit));
    }

    #[test]
    fn test_reconstruction_hint_variants() {
        // Test Domain variants
        let cartesian = Domain::Cartesian;
        let spherical = Domain::Spherical { radius: 6371e3 };
        assert_ne!(cartesian, spherical);

        // Test InterpolationMethod variants
        let linear = InterpolationMethod::Linear;
        let cubic = InterpolationMethod::Cubic;
        let _idw = InterpolationMethod::Idw { power: 2.0 };
        assert_ne!(linear, cubic);

        // Test BoundaryCondition variants
        let clamp = BoundaryCondition::Clamp;
        let wrap = BoundaryCondition::Wrap;
        assert_ne!(clamp, wrap);

        // Test full ReconstructionHint
        let hint = ReconstructionHint {
            domain: Domain::Cartesian,
            method: InterpolationMethod::NearestNeighbor,
            boundary: BoundaryCondition::Clamp,
            conservative: false,
        };
        assert!(!hint.conservative);
    }
}
