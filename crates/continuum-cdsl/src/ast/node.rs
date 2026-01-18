//! Unified Node<I> structure
//!
//! This module defines `Node<I>`, the single structure that represents all
//! AST nodes throughout the compilation pipeline. Unlike traditional compilers
//! with separate AST and IR representations, Continuum uses one unified structure
//! that accumulates data through each compilation phase.
//!
//! # The Node<I> Design
//!
//! `Node<I>` is generic over an index type `I`:
//! - `Node<()>` - Global primitive (signal, field, operator at world scope)
//! - `Node<EntityId>` - Per-entity primitive (member)
//!
//! This makes the distinction between global and per-entity nodes type-safe and
//! explicit in the type system.
//!
//! # Lifecycle
//!
//! As the node flows through compilation phases, fields are added and cleared:
//!
//! 1. **Parsed** - Has `type_expr` and `execution_blocks` from source
//! 2. **Resolved** - `type_expr` cleared, `output` and `inputs` set
//! 3. **Validated** - `validation_errors` populated
//! 4. **Compiled** - `execution_blocks` cleared, `executions` and `reads` set
//!
//! This makes the pipeline state explicit: you can check `is_resolved()`,
//! `is_compiled()`, etc. to determine what phase the node has completed.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{Node, RoleData, EntityId};
//! use continuum_cdsl::foundation::{Path, Span, Type};
//!
//! // Create a global signal node
//! let mut signal = Node::new(
//!     Path::from_str("world.temperature"),
//!     Span::new(0, 0, 100, 1),
//!     RoleData::Signal,
//!     (), // global
//! );
//!
//! // After type resolution
//! signal.output = Some(Type::Kernel(KernelType { /*...*/ }));
//! assert!(signal.is_resolved());
//!
//! // Create a per-entity member
//! let member = Node::new(
//!     Path::from_str("plate.area"),
//!     Span::new(0, 0, 100, 1),
//!     RoleData::Signal,
//!     EntityId(Path::from_str("plate")),
//! );
//! assert_eq!(member.index, EntityId(Path::from_str("plate")));
//! ```

use crate::foundation::{AnalyzerId, EntityId, EraId, FieldId, Path, Span, StratumId, Type};
use std::path::PathBuf;

use super::block::BlockBody;
use super::declaration::{Attribute, ObserveBlock, WarmupBlock, WhenBlock};
use super::expr::TypedExpr;
use super::role::RoleData;

/// Index trait - marker for node indexing types
///
/// Implemented by:
/// - `()` for global primitives (signals, fields, operators at world scope)
/// - `EntityId` for per-entity primitives (members)
///
/// This is a simple marker trait - the type parameter I is enough to
/// distinguish global from per-entity nodes at compile time.
pub trait Index: Clone + PartialEq + std::fmt::Debug {}

impl Index for () {}

// Implement Index for foundation EntityId type
impl Index for EntityId {}

/// Unified node structure - everything flows through this
///
/// Node<I> is the single structure used throughout compilation:
/// - Parser produces Node<()> and Node<EntityId>
/// - Type resolution adds `output` and clears `type_expr`
/// - Validation adds `validation_errors`
/// - Compilation adds `executions` and `reads`, clears `execution_blocks`
///
/// The Index parameter I distinguishes:
/// - Node<()> = global primitive
/// - Node<EntityId> = per-entity primitive (member)
#[derive(Clone, Debug)]
pub struct Node<I: Index = ()> {
    // =========================================================================
    // Identity
    // =========================================================================
    /// Hierarchical path to this node (e.g., "terra.plate.velocity")
    pub path: Path,

    /// Source location for error messages
    pub span: Span,

    /// Source file this node was parsed from (if applicable)
    pub file: Option<PathBuf>,

    // =========================================================================
    // Documentation
    // =========================================================================
    /// Documentation comment from source
    pub doc: Option<String>,

    /// Short title for UI/diagnostics
    pub title: Option<String>,

    /// Symbol name (for exports, debugging)
    pub symbol: Option<String>,

    // =========================================================================
    // Role + Role-Specific Data
    // =========================================================================
    /// Role identifier and role-specific data
    ///
    /// This makes invalid states unrepresentable:
    /// - Only Fields can have reconstruction hints
    /// - Only Impulses declare payload types
    pub role: RoleData,

    // =========================================================================
    // Common Capabilities (used by multiple roles)
    // =========================================================================
    /// Scoping information (config/const references)
    pub scoping: Option<Scoping>,

    /// Assertions to validate after execution
    pub assertions: Vec<Assertion>,

    /// Compiled execution blocks (set after compilation)
    pub executions: Vec<Execution>,

    /// Stratum assignment (execution lane + cadence)
    pub stratum: Option<StratumId>,

    /// Output type (what this node produces)
    ///
    /// Set by type resolution. None before resolution.
    pub output: Option<Type>,

    /// Named input types (what this node receives)
    ///
    /// Set by type resolution. Each input has a name and type.
    /// For signals: inputs from collect/emit expressions
    /// For operators: depends on operator type
    /// For impulses: payload fields
    /// Empty before resolution.
    pub inputs: Vec<(String, Type)>,

    // =========================================================================
    // Indexing
    // =========================================================================
    /// Where this node lives
    ///
    /// - `I = ()` for global primitives
    /// - `I = EntityId` for per-entity primitives (members)
    pub index: I,

    // =========================================================================
    // Lifecycle Fields (cleared after consumption)
    // =========================================================================
    /// Parsed attributes from source (processed by semantic analysis)
    ///
    /// Raw attributes like `:title("...")`, `:stratum(fast)`, etc.
    /// Semantic analysis extracts these into proper fields (title, stratum, etc).
    /// Cleared after processing.
    pub attributes: Vec<Attribute>,

    /// Type expression from source (cleared after resolution)
    pub type_expr: Option<TypeExpr>,

    /// Warmup block for signals (optional)
    ///
    /// Contains warmup attributes and iterate expression.
    /// Only valid for Signal role.
    pub warmup: Option<WarmupBlock>,

    /// When block for fractures (optional)
    ///
    /// Contains condition expressions that trigger fracture detection.
    /// Only valid for Fracture role.
    pub when: Option<WhenBlock>,

    /// Observe block for chronicles (optional)
    ///
    /// Contains when/emit pairs for pattern detection.
    /// Only valid for Chronicle role.
    pub observe: Option<ObserveBlock>,

    /// Execution blocks from source (cleared after compilation)
    ///
    /// Map from phase name to block body (expression or statements).
    /// Compiler converts these to typed Execution structs with explicit phase enum values.
    /// Statement blocks are validated to only appear in effect phases during semantic analysis.
    pub execution_blocks: Vec<(String, BlockBody)>,

    /// Dependencies discovered during analysis
    ///
    /// Paths this node reads from. Set during compilation for DAG construction.
    pub reads: Vec<Path>,

    /// Validation errors found during semantic analysis
    pub validation_errors: Vec<ValidationError>,
}

impl<I: Index> Node<I> {
    /// Create a new node with minimal required fields
    ///
    /// All optional fields are initialized to None/empty. Lifecycle fields
    /// (`type_expr`, `execution_blocks`, etc) start empty - they will be
    /// populated by compilation phases.
    ///
    /// # Parameters
    ///
    /// - `path`: Hierarchical path to this node (e.g., "world.temperature")
    /// - `span`: Source location for error messages
    /// - `role`: Role data (what this node is + role-specific data)
    /// - `index`: Where this node lives (() = global, EntityId = per-entity)
    ///
    /// # Returns
    ///
    /// A new node in initial state (not resolved, not compiled)
    pub fn new(path: Path, span: Span, role: RoleData, index: I) -> Self {
        Self {
            path,
            span,
            file: None,
            doc: None,
            title: None,
            symbol: None,
            role,
            scoping: None,
            assertions: Vec::new(),
            executions: Vec::new(),
            stratum: None,
            output: None,
            inputs: Vec::new(),
            index,
            attributes: Vec::new(),
            type_expr: None,
            warmup: None,
            when: None,
            observe: None,
            execution_blocks: Vec::new(),
            reads: Vec::new(),
            validation_errors: Vec::new(),
        }
    }

    /// Get the role ID for this node
    ///
    /// # Returns
    ///
    /// The RoleId enum value for this node's role (Signal, Field, etc)
    pub fn role_id(&self) -> super::role::RoleId {
        self.role.id()
    }

    /// Check if this node has been type-resolved
    ///
    /// A node is considered resolved when:
    /// - `type_expr` has been cleared (consumed by type resolver)
    /// - `output` type has been set
    ///
    /// # Returns
    ///
    /// `true` if type resolution has completed, `false` otherwise
    pub fn is_resolved(&self) -> bool {
        self.type_expr.is_none() && self.output.is_some()
    }

    /// Check if this node has been compiled
    ///
    /// A node is considered compiled when:
    /// - `execution_blocks` have been cleared (consumed by compiler)
    /// - `executions` have been added (compiled execution blocks)
    ///
    /// # Returns
    ///
    /// `true` if compilation has completed, `false` otherwise
    pub fn is_compiled(&self) -> bool {
        self.execution_blocks.is_empty() && !self.executions.is_empty()
    }

    /// Check if this node has validation errors
    ///
    /// # Returns
    ///
    /// `true` if validation errors have been recorded, `false` otherwise
    pub fn has_errors(&self) -> bool {
        !self.validation_errors.is_empty()
    }
}

// =============================================================================
// Pipeline Trait Implementations
// =============================================================================
// These traits describe the data lifecycle of Node<I> as it flows through
// compilation passes. See ast/pipeline.rs for trait definitions.

use super::pipeline::{Compiled, Named, Parsed, Resolved, Validated};

impl<I: Index> Named for Node<I> {
    fn path(&self) -> &Path {
        &self.path
    }

    fn span(&self) -> Span {
        self.span
    }
}

impl<I: Index> Parsed for Node<I> {
    fn type_expr(&self) -> Option<&TypeExpr> {
        self.type_expr.as_ref()
    }

    fn execution_blocks(&self) -> &[(String, BlockBody)] {
        &self.execution_blocks
    }
}

impl<I: Index> Resolved for Node<I> {
    fn output(&self) -> Option<&Type> {
        self.output.as_ref()
    }

    fn inputs(&self) -> &[(String, Type)] {
        &self.inputs
    }
}

impl<I: Index> Validated for Node<I> {
    fn validation_errors(&self) -> &[ValidationError] {
        &self.validation_errors
    }
}

impl<I: Index> Compiled for Node<I> {
    fn executions(&self) -> &[Execution] {
        &self.executions
    }

    fn reads(&self) -> &[Path] {
        &self.reads
    }
}

// =============================================================================
// Structural Declarations
// =============================================================================
// These types define structure rather than execution. They are not Node<I>
// primitives but declarations that shape the compilation environment.

/// Entity declaration - namespace + index type for per-entity primitives
///
/// Entities declare a type of thing that can have multiple instances.
/// Examples: plate, planet, star, person, city
///
/// An Entity creates an index type that parameterizes Node<I>:
/// - Node<()> — global primitive
/// - Node<EntityId> — per-entity primitive (member)
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Warning - logged but execution continues
    Warn,

    /// Error - logged, may fail analysis depending on policy
    Error,

    /// Fatal - immediately halt analysis
    Fatal,
}

// =============================================================================
// Placeholder types for dependent structures
// =============================================================================
// These are minimal placeholder types that will be properly implemented in
// future phases. They exist to allow Node<I> to compile and be tested, but
// are not yet functional.

/// Scoping information for config/const resolution
///
/// This will contain mappings from names to config values and constants.
/// Will be implemented when the resolution phase is added.
///
/// **Current status:** Placeholder - not yet implemented
#[derive(Clone, Debug)]
pub struct Scoping {
    // This will hold config/const lookups when implemented
    #[doc(hidden)]
    _placeholder: (),
}

/// Assertion to validate invariants
///
/// Assertions validate conditions after execution completes.
/// Each assertion has a condition expression and error message.
///
/// **Current status:** Placeholder - not yet implemented
#[derive(Clone, Debug)]
pub struct Assertion {
    // This will hold assertion expression and message when implemented
    #[doc(hidden)]
    _placeholder: (),
}

/// Compiled execution block
///
/// An execution block contains the compiled code for a specific phase,
/// along with metadata about what it reads and emits.
///
/// **Current status:** Placeholder - not yet implemented
#[derive(Clone, Debug)]
pub struct Execution {
    // This will hold phase, body, and dependency info when implemented
    #[doc(hidden)]
    _placeholder: (),
}

// Re-export untyped AST types for use in Node
//
// These are defined in the untyped module and represent the parser output
// before type resolution transforms them into TypedExpr.
pub use super::untyped::TypeExpr;

/// Validation error from semantic analysis
///
/// Structured error produced during validation passes. Contains error
/// kind, message, source location, and hints for fixing.
///
/// **Current status:** Placeholder - not yet implemented
#[derive(Clone, Debug)]
pub struct ValidationError {
    // This will hold error kind, span, and hints when implemented
    #[doc(hidden)]
    _placeholder: (),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::block::BlockBody;
    use crate::ast::untyped::Expr;
    use crate::foundation::Span;

    #[test]
    fn test_node_creation_global() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        assert_eq!(node.path, path);
        assert_eq!(node.span, span);
        assert_eq!(node.role_id(), super::super::role::RoleId::Signal);
        assert_eq!(node.index, ());
        assert!(!node.is_resolved());
        assert!(!node.is_compiled());
        assert!(!node.has_errors());
    }

    #[test]
    fn test_node_creation_entity() {
        let path = Path::from_str("test.member");
        let span = Span::new(0, 0, 10, 1);
        let entity_id = EntityId::new("test.entity");
        let node = Node::new(path.clone(), span, RoleData::Signal, entity_id.clone());

        assert_eq!(node.path, path);
        assert_eq!(node.index, entity_id);
        assert_eq!(node.role_id(), super::super::role::RoleId::Signal);
    }

    #[test]
    fn test_field_with_reconstruction() {
        let path = Path::from_str("test.field");
        let span = Span::new(0, 0, 10, 1);
        let hint = super::super::role::ReconstructionHint {
            domain: super::super::role::Domain::Cartesian,
            method: super::super::role::InterpolationMethod::Linear,
            boundary: super::super::role::BoundaryCondition::Clamp,
            conservative: false,
        };
        let node = Node::new(
            path,
            span,
            RoleData::Field {
                reconstruction: Some(hint.clone()),
            },
            (),
        );

        assert_eq!(node.role_id(), super::super::role::RoleId::Field);
        match &node.role {
            RoleData::Field { reconstruction } => {
                assert!(reconstruction.is_some());
                assert_eq!(reconstruction.as_ref().unwrap(), &hint);
            }
            _ => panic!("Expected Field role"),
        }
    }

    #[test]
    fn test_impulse_with_payload() {
        let path = Path::from_str("test.impulse");
        let span = Span::new(0, 0, 10, 1);
        let payload_type = Type::Bool;
        let node = Node::new(
            path,
            span,
            RoleData::Impulse {
                payload: Some(payload_type.clone()),
            },
            (),
        );

        assert_eq!(node.role_id(), super::super::role::RoleId::Impulse);
        match &node.role {
            RoleData::Impulse { payload } => {
                assert_eq!(payload.as_ref(), Some(&payload_type));
            }
            _ => panic!("Expected Impulse role"),
        }
    }

    #[test]
    fn test_node_lifecycle_states() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // Initial state: not resolved, not compiled
        assert!(!node.is_resolved());
        assert!(!node.is_compiled());

        // After type resolution: type_expr cleared, output set
        node.type_expr = None;
        node.output = Some(Type::Bool);
        assert!(node.is_resolved());
        assert!(!node.is_compiled());

        // After compilation: execution_blocks cleared, executions set
        node.execution_blocks = Vec::new();
        node.executions = vec![Execution { _placeholder: () }];
        assert!(node.is_compiled());
    }

    #[test]
    fn test_node_validation_errors() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let mut node = Node::new(path, span, RoleData::Signal, ());

        assert!(!node.has_errors());

        node.validation_errors
            .push(ValidationError { _placeholder: () });
        assert!(node.has_errors());
    }

    #[test]
    fn test_different_roles() {
        let span = Span::new(0, 0, 10, 1);

        let signal = Node::new(Path::from_str("test.signal"), span, RoleData::Signal, ());
        assert_eq!(signal.role_id(), super::super::role::RoleId::Signal);

        let field = Node::new(
            Path::from_str("test.field"),
            span,
            RoleData::Field {
                reconstruction: None,
            },
            (),
        );
        assert_eq!(field.role_id(), super::super::role::RoleId::Field);

        let operator = Node::new(
            Path::from_str("test.operator"),
            span,
            RoleData::Operator,
            (),
        );
        assert_eq!(operator.role_id(), super::super::role::RoleId::Operator);

        let impulse = Node::new(
            Path::from_str("test.impulse"),
            span,
            RoleData::Impulse { payload: None },
            (),
        );
        assert_eq!(impulse.role_id(), super::super::role::RoleId::Impulse);

        let fracture = Node::new(
            Path::from_str("test.fracture"),
            span,
            RoleData::Fracture,
            (),
        );
        assert_eq!(fracture.role_id(), super::super::role::RoleId::Fracture);

        let chronicle = Node::new(
            Path::from_str("test.chronicle"),
            span,
            RoleData::Chronicle,
            (),
        );
        assert_eq!(chronicle.role_id(), super::super::role::RoleId::Chronicle);
    }

    #[test]
    fn test_entity_id_equality() {
        let entity1 = EntityId::new("entity1");
        let entity2 = EntityId::new("entity2");
        let entity1_dup = EntityId::new("entity1");

        assert_eq!(entity1, entity1_dup);
        assert_ne!(entity1, entity2);
    }

    #[test]
    fn test_node_lifecycle_boundaries() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // output set but type_expr still present -> not resolved
        node.type_expr = Some(TypeExpr::Bool);
        node.output = Some(Type::Bool);
        assert!(!node.is_resolved());

        // type_expr cleared but no output -> not resolved
        node.type_expr = None;
        node.output = None;
        assert!(!node.is_resolved());

        // execution_blocks empty but no executions -> not compiled
        node.execution_blocks = Vec::new();
        node.executions = Vec::new();
        assert!(!node.is_compiled());

        // executions present but execution_blocks not empty -> not compiled
        let test_expr = Expr::literal(0.0, None, span);
        let test_block = BlockBody::Expression(test_expr);
        node.execution_blocks = vec![("test".to_string(), test_block)];
        node.executions = vec![Execution { _placeholder: () }];
        assert!(!node.is_compiled());
    }

    #[test]
    fn test_node_default_initialization() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        // Verify all lifecycle fields start empty/None
        assert!(node.file.is_none());
        assert!(node.doc.is_none());
        assert!(node.title.is_none());
        assert!(node.symbol.is_none());
        assert!(node.scoping.is_none());
        assert!(node.assertions.is_empty());
        assert!(node.executions.is_empty());
        assert!(node.stratum.is_none());
        assert!(node.output.is_none());
        assert!(node.inputs.is_empty());
        assert!(node.type_expr.is_none());
        assert!(node.execution_blocks.is_empty());
        assert!(node.reads.is_empty());
        assert!(node.validation_errors.is_empty());
    }

    // =========================================================================
    // Structural Declaration Tests
    // =========================================================================

    #[test]
    fn test_entity_creation() {
        let path = Path::from_str("test.plate");
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
        let path = Path::from_str("test.fast");
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
        let mut slow = Stratum::new(StratumId::new("slow"), Path::from_str("slow"), span);

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
        use super::super::expr::ExprKind;
        use crate::foundation::{KernelType, Shape, Unit};

        let path = Path::from_str("test.formation");
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
        use super::super::expr::ExprKind;

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
        use super::super::expr::ExprKind;
        use crate::foundation::{KernelType, Shape, Unit};

        let path = Path::from_str("test.elevation_stats");
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
        use super::super::expr::ExprKind;

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
}
