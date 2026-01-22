//! Structural declarations
//!
//! This module defines structural declarations that shape the compilation environment
//! rather than representing executable primitives. These include:
//!
//! - **Entity** - namespace + index type for per-entity primitives
//! - **Stratum** - execution lane with cadence
//! - **Era** - execution policy regime
//! - **Analyzer** - pure observer for post-hoc field analysis
//!
//! Unlike `Node<I>` primitives (signals, fields, operators), these declarations
//! define the structure and policy of the simulation without directly participating
//! in the execution DAG.

use crate::ast::declaration::Attribute;
use crate::ast::expr::TypedExpr;
use crate::foundation::{AnalyzerId, EntityId, EraId, FieldId, Path, Span, StratumId};

// =============================================================================
// Structural Declarations
// =============================================================================

/// Entity declaration - namespace + index type for per-entity primitives
///
/// Entities declare a type of thing that can have multiple instances.
/// Examples: plate, planet, star, person, city
///
/// An Entity creates an index type that parameterizes `Node<I>`:
/// - `Node<()>` — global primitive
/// - `Node<EntityId>` — per-entity primitive (member)
///
/// Any Role can be per-entity: Signal, Field, Fracture, Operator.
/// Impulse and Chronicle are always global.
///
/// # Entity Lifecycle
///
/// - Instance count is fixed at scenario initialization
/// - No runtime creation (spawn) — not yet supported
/// - No runtime destruction (destroy) — not yet supported
/// - Instance IDs are stable throughout simulation
/// - `prev` is always valid (no "newborn" entity edge case)
///
/// # Examples
///
/// ```cdsl
/// entity plate {
///     member area : Scalar<m2>         // Signal per plate
///     field stress : Scalar<Pa>        // Field per plate
///     fracture rift { ... }            // Fracture per plate
///     operator apply_friction { ... } // Operator per plate
/// }
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity type
    pub id: EntityId,

    /// Hierarchical path to this declaration
    pub path: Path,

    /// Source location for error messages
    pub span: Span,

    /// Documentation comment from source
    pub doc: Option<String>,

    /// Parsed attributes from source
    ///
    /// Raw attributes like `:count(100)`, etc.
    /// Processed during semantic analysis.
    pub attributes: Vec<Attribute>,
}

impl Entity {
    /// Create a new entity declaration
    pub fn new(id: EntityId, path: Path, span: Span) -> Self {
        Self {
            id,
            path,
            span,
            doc: None,
            attributes: Vec::new(),
        }
    }
}

/// Stratum declaration - execution lane with cadence
///
/// A Stratum defines a named execution lane with its own temporal cadence.
/// Strata enable multi-rate execution where different systems evolve at
/// different timescales.
///
/// # Strata and Time
///
/// - Time advances globally via ticks and dt
/// - Strata define *how often* logic runs relative to ticks
/// - Cadence = execute every N ticks (1 = every tick)
///
/// # Strata and Execution
///
/// - Execution graphs are constructed per (phase × stratum × era)
/// - Each stratum has its own DAG
/// - Strata may execute or be gated depending on the active era
///
/// # Examples
///
/// ```cdsl
/// stratum fast { cadence: 1 }      // every tick
/// stratum slow { cadence: 100 }    // every 100 ticks
///
/// signal temperature : Scalar<K> {
///     : stratum(fast)
///     resolve { ... }
/// }
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Stratum {
    /// Unique identifier for this stratum
    pub id: StratumId,

    /// Hierarchical path to this declaration
    pub path: Path,

    /// Execution cadence - execute every N ticks (1 = every tick)
    ///
    /// Extracted from `:stride(N)` or `:cadence(N)` attributes during semantic analysis.
    /// - None = not yet resolved (parser stage)
    /// - Some(n) = validated cadence value (semantic analysis stage)
    ///
    /// Semantic analysis will:
    /// - Extract from attributes
    /// - Default to 1 if absent
    /// - Error if present but invalid (non-literal, non-positive)
    /// - Validate cadence > 0
    pub cadence: Option<u32>,

    /// Source location for error messages
    pub span: Span,

    /// Documentation comment from source
    pub doc: Option<String>,

    /// Parsed attributes from source
    ///
    /// Raw attributes for semantic analysis to validate.
    /// Parser extracts cadence but preserves all attributes.
    pub attributes: Vec<Attribute>,
}

impl Stratum {
    /// Create a new stratum declaration
    ///
    /// Cadence is None at parser stage, will be resolved during semantic analysis.
    /// Semantic analysis validates:
    /// - Cadence is a positive integer literal
    /// - Defaults to 1 if no :stride/:cadence attribute present
    pub fn new(id: StratumId, path: Path, span: Span) -> Self {
        Self {
            id,
            path,
            cadence: None,
            span,
            doc: None,
            attributes: Vec::new(),
        }
    }

    /// Check if this stratum should execute on the given tick
    ///
    /// Requires cadence to be resolved (semantic analysis complete).
    /// Panics if cadence is None (indicates semantic analysis not run).
    pub fn is_eligible(&self, tick: u64) -> bool {
        let cadence = self
            .cadence
            .expect("Stratum cadence must be resolved before checking eligibility");
        tick % (cadence as u64) == 0
    }
}

/// Era declaration - execution policy regime
///
/// An Era defines a named execution regime that controls:
/// - Base timestep (dt)
/// - Which strata are active or gated
/// - Cadence overrides for active strata
/// - Transitions to other eras (signal-driven, deterministic)
///
/// # Era Membership and Activation
///
/// At any tick, exactly one era is active. The active era determines
/// execution policy. There is no implicit blending between eras.
///
/// # Era Transitions
///
/// Transitions are:
/// - Evaluated at tick boundaries
/// - Deterministic (depend only on resolved signals, not fields)
/// - Explicit (no implicit fallback)
/// - Evaluated in declaration order (first matching condition wins)
///
/// # Examples
///
/// ```cdsl
/// era formation {
///     : dt(1_000_000<yr>)
///     : strata(tectonics: active, climate: gated)
///     : transition(stable, when: mantle.temperature < 1500<K>)
/// }
///
/// era stable {
///     : dt(1000<yr>)
///     : strata(tectonics: active, climate: active)
/// }
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Era {
    /// Unique identifier for this era
    pub id: EraId,

    /// Hierarchical path to this declaration
    pub path: Path,

    /// Base timestep expression (may be constant or computed)
    pub dt: TypedExpr,

    /// Stratum activation policies for this era
    pub strata_policy: Vec<StratumPolicy>,

    /// Transitions to other eras
    ///
    /// Evaluated in declaration order. First matching condition wins.
    /// If multiple transitions could fire, this is deterministic but may
    /// indicate ambiguous era design.
    pub transitions: Vec<EraTransition>,

    /// Source location for error messages
    pub span: Span,

    /// Documentation comment from source
    pub doc: Option<String>,
}

impl Era {
    /// Create a new era declaration
    pub fn new(id: EraId, path: Path, dt: TypedExpr, span: Span) -> Self {
        Self {
            id,
            path,
            dt,
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            span,
            doc: None,
        }
    }

    /// Check if this era is terminal (has no outgoing transitions)
    pub fn is_terminal(&self) -> bool {
        self.transitions.is_empty()
    }
}

/// Stratum activation policy within an era
///
/// Controls whether a stratum executes during an era and at what cadence.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StratumPolicy {
    /// Which stratum this policy applies to
    pub stratum: StratumId,

    /// Whether the stratum is active in this era
    pub active: bool,

    /// Optional cadence override (if None, uses stratum's declared cadence)
    pub cadence_override: Option<u32>,
}

impl StratumPolicy {
    /// Create a new stratum policy
    pub fn new(stratum: StratumId, active: bool) -> Self {
        Self {
            stratum,
            active,
            cadence_override: None,
        }
    }

    /// Create an active stratum policy with cadence override
    pub fn with_cadence(stratum: StratumId, cadence: u32) -> Self {
        Self {
            stratum,
            active: true,
            cadence_override: Some(cadence),
        }
    }

    /// Create a gated (inactive) stratum policy
    pub fn gated(stratum: StratumId) -> Self {
        Self {
            stratum,
            active: false,
            cadence_override: None,
        }
    }
}

/// Era transition rule
///
/// Defines when and how to transition from one era to another.
/// Transitions are signal-driven and deterministic.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EraTransition {
    /// Target era to transition to
    pub target: EraId,

    /// Transition condition (evaluated over resolved signals)
    /// Must be Bool-typed expression
    pub condition: TypedExpr,

    /// Source location for error messages
    pub span: Span,
}

impl EraTransition {
    /// Create a new era transition
    pub fn new(target: EraId, condition: TypedExpr, span: Span) -> Self {
        Self {
            target,
            condition,
            span,
        }
    }
}

/// Analyzer declaration - pure observer for post-hoc field analysis
///
/// Analyzers are pure observers that run post-hoc on field snapshots.
/// They have no effect on causality and run outside the simulation DAG.
///
/// # Analyzer Capabilities
///
/// - Access fields through Lens handles
/// - Produce JSON-serializable output
/// - Declare validation rules (warnings/errors on field statistics)
/// - Cannot influence simulation state
///
/// # Field Access Patterns
///
/// - Aggregate statistics: `stats.mean(field.elevation)`
/// - Point queries: `field.temperature.at(lat: 45.0, lon: -122.0)`
/// - Raw samples: `field.elevation.samples()`
///
/// # Examples
///
/// ```cdsl
/// analyzer terra.elevation_stats {
///     : doc "Statistical summary of elevation distribution"
///     : requires(fields: [geophysics.elevation])
///     
///     compute {
///         {
///             mean: stats.mean(field.geophysics.elevation),
///             std: stats.std(field.geophysics.elevation),
///             min: stats.min(field.geophysics.elevation),
///             max: stats.max(field.geophysics.elevation),
///         }
///     }
///     
///     validate {
///         stats.min(field.geophysics.elevation) > -12000<m> : warn, "unrealistic ocean depth"
///         stats.max(field.geophysics.elevation) < 12000<m> : warn, "unrealistic mountain height"
///     }
/// }
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Analyzer {
    /// Unique identifier for this analyzer
    pub id: AnalyzerId,

    /// Hierarchical path to this declaration
    pub path: Path,

    /// Source location for error messages
    pub span: Span,

    /// Documentation comment from source
    pub doc: Option<String>,

    /// Field dependencies - which fields this analyzer requires
    ///
    /// Duplicates should be caught during validation phase.
    /// Order is preserved for deterministic processing.
    pub requires: Vec<FieldId>,

    /// Computation expression - produces JSON-serializable value
    pub compute: TypedExpr,

    /// Validation rules - assertions over field statistics
    pub validations: Vec<AnalyzerValidation>,
}

impl Analyzer {
    /// Create a new analyzer declaration
    pub fn new(id: AnalyzerId, path: Path, compute: TypedExpr, span: Span) -> Self {
        Self {
            id,
            path,
            span,
            doc: None,
            requires: Vec::new(),
            compute,
            validations: Vec::new(),
        }
    }
}

/// Analyzer validation rule
///
/// Defines a condition that should be checked against field data,
/// with configurable severity (warn, error, fatal).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AnalyzerValidation {
    /// Validation condition (Bool-typed expression over field statistics)
    pub condition: TypedExpr,

    /// Severity if condition fails
    pub severity: ValidationSeverity,

    /// Human-readable message describing the validation
    pub message: String,

    /// Source location for error messages
    pub span: Span,
}

impl AnalyzerValidation {
    /// Create a new analyzer validation rule
    pub fn new(
        condition: TypedExpr,
        severity: ValidationSeverity,
        message: impl Into<String>,
        span: Span,
    ) -> Self {
        Self {
            condition,
            severity,
            message: message.into(),
            span,
        }
    }
}

/// Validation severity levels
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ValidationSeverity {
    /// Warning - logged but execution continues
    Warn,

    /// Error - logged, may fail analysis depending on policy
    Error,

    /// Fatal - immediately halt analysis
    Fatal,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_creation() {
        let path = Path::from_path_str("test.plate");
        let span = Span::new(0, 0, 10, 1);
        let id = EntityId::new("test.plate");
        let entity = Entity::new(id.clone(), path.clone(), span);

        assert_eq!(entity.id, id);
        assert_eq!(entity.path, path);
        assert_eq!(entity.span, span);
        assert!(entity.doc.is_none());
    }

    #[test]
    fn test_stratum_creation() {
        let path = Path::from_path_str("test.fast");
        let span = Span::new(0, 0, 10, 1);
        let id = StratumId::new("test.fast");
        let mut stratum = Stratum::new(id.clone(), path.clone(), span);

        // Cadence is None until semantic analysis resolves it
        assert_eq!(stratum.cadence, None);

        // Simulate semantic analysis resolving cadence
        stratum.cadence = Some(1);

        assert_eq!(stratum.id, id);
        assert_eq!(stratum.path, path);
        assert_eq!(stratum.cadence, Some(1));
        assert!(stratum.is_eligible(0));
        assert!(stratum.is_eligible(1));
        assert!(stratum.is_eligible(100));
    }

    #[test]
    fn test_stratum_cadence() {
        let span = Span::new(0, 0, 10, 1);
        let mut slow = Stratum::new(StratumId::new("slow"), Path::from_path_str("slow"), span);

        // Simulate semantic analysis resolving cadence to 10
        slow.cadence = Some(10);

        assert!(slow.is_eligible(0));
        assert!(!slow.is_eligible(1));
        assert!(!slow.is_eligible(9));
        assert!(slow.is_eligible(10));
        assert!(slow.is_eligible(20));
    }

    #[test]
    fn test_era_creation() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::{KernelType, Shape, Type, Unit};

        let path = Path::from_path_str("test.formation");
        let span = Span::new(0, 0, 10, 1);
        let id = EraId::new("test.formation");

        // Create a simple dt expression (1000.0 seconds for testing)
        let dt_expr = TypedExpr {
            expr: ExprKind::Literal {
                value: 1000.0,
                unit: Some(Unit::seconds()),
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::seconds(),
                bounds: None,
            }),
            span,
        };

        let era = Era::new(id.clone(), path.clone(), dt_expr, span);

        assert_eq!(era.id, id);
        assert_eq!(era.path, path);
        assert!(era.is_terminal()); // No transitions yet
        assert!(era.strata_policy.is_empty());
    }

    #[test]
    fn test_stratum_policy() {
        let stratum_id = StratumId::new("test.fast");

        let active = StratumPolicy::new(stratum_id.clone(), true);
        assert!(active.active);
        assert!(active.cadence_override.is_none());

        let gated = StratumPolicy::gated(stratum_id.clone());
        assert!(!gated.active);
        assert!(gated.cadence_override.is_none());

        let overridden = StratumPolicy::with_cadence(stratum_id.clone(), 50);
        assert!(overridden.active);
        assert_eq!(overridden.cadence_override, Some(50));
    }

    #[test]
    fn test_era_transition() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::Type;

        let span = Span::new(0, 0, 10, 1);
        let target_era = EraId::new("stable");

        let condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0, // true represented as 1.0
                unit: None,
            },
            ty: Type::Bool,
            span,
        };

        let transition = EraTransition::new(target_era.clone(), condition, span);
        assert_eq!(transition.target, target_era);
    }

    #[test]
    fn test_analyzer_creation() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::{KernelType, Shape, Type, Unit};

        let path = Path::from_path_str("test.elevation_stats");
        let span = Span::new(0, 0, 10, 1);
        let id = AnalyzerId::new("test.elevation_stats");

        // Create a simple compute expression
        let compute = TypedExpr {
            expr: ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::dimensionless(),
                bounds: None,
            }),
            span,
        };

        let analyzer = Analyzer::new(id.clone(), path.clone(), compute, span);

        assert_eq!(analyzer.id, id);
        assert_eq!(analyzer.path, path);
        assert!(analyzer.requires.is_empty());
        assert!(analyzer.validations.is_empty());
    }

    #[test]
    fn test_analyzer_validation() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::Type;

        let span = Span::new(0, 0, 10, 1);

        let condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty: Type::Bool,
            span,
        };

        let validation =
            AnalyzerValidation::new(condition, ValidationSeverity::Warn, "test message", span);

        assert_eq!(validation.severity, ValidationSeverity::Warn);
        assert_eq!(validation.message, "test message");
    }

    #[test]
    fn test_validation_severity() {
        assert_ne!(ValidationSeverity::Warn, ValidationSeverity::Error);
        assert_ne!(ValidationSeverity::Error, ValidationSeverity::Fatal);
        assert_ne!(ValidationSeverity::Warn, ValidationSeverity::Fatal);
    }

    #[test]
    #[should_panic(expected = "cadence must be resolved")]
    fn test_stratum_is_eligible_panics_without_cadence() {
        let span = Span::new(0, 0, 10, 1);
        let stratum = Stratum::new(StratumId::new("test"), Path::from_path_str("test"), span);
        // cadence is None - documented to panic
        stratum.is_eligible(0);
    }

    #[test]
    fn test_era_is_not_terminal_with_transitions() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::{KernelType, Shape, Type, Unit};

        let span = Span::new(0, 0, 10, 1);
        let id = EraId::new("test.formation");
        let dt_expr = TypedExpr {
            expr: ExprKind::Literal {
                value: 1000.0,
                unit: Some(Unit::seconds()),
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::seconds(),
                bounds: None,
            }),
            span,
        };

        let mut era = Era::new(id, Path::from_path_str("test"), dt_expr, span);

        // Add a transition - era should not be terminal
        let condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty: Type::Bool,
            span,
        };
        era.transitions
            .push(EraTransition::new(EraId::new("stable"), condition, span));

        assert!(!era.is_terminal());
    }

    #[test]
    fn test_analyzer_validation_error_severity() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::Type;

        let span = Span::new(0, 0, 10, 1);
        let condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty: Type::Bool,
            span,
        };

        let validation = AnalyzerValidation::new(
            condition.clone(),
            ValidationSeverity::Error,
            "error message",
            span,
        );

        assert_eq!(validation.severity, ValidationSeverity::Error);
        assert_eq!(validation.message, "error message");
    }

    #[test]
    fn test_analyzer_validation_fatal_severity() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::Type;

        let span = Span::new(0, 0, 10, 1);
        let condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty: Type::Bool,
            span,
        };

        let validation =
            AnalyzerValidation::new(condition, ValidationSeverity::Fatal, "fatal message", span);

        assert_eq!(validation.severity, ValidationSeverity::Fatal);
        assert_eq!(validation.message, "fatal message");
    }
}
