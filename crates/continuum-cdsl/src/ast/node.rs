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

use crate::foundation::{AssertionSeverity, EntityId, Path, Span, StratumId, Type};
use std::path::PathBuf;

use super::block::{BlockBody, TypedStmt};
use super::declaration::{Attribute, ObserveBlock, WarmupBlock, WhenBlock};
use super::expr::TypedExpr;
use super::role::RoleData;

// Re-export structural declarations from structural module
pub use super::structural::{Entity, Stratum};

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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

    /// Aggregate signal and field dependencies for this node.
    ///
    /// Represents the union of all paths read by this node across all its
    /// execution contexts:
    /// 1. The `reads` list from every [`Execution`] block (e.g., `resolve`, `collect`).
    /// 2. All dependencies extracted from the condition expressions of [`Assertion`]s.
    ///
    /// These aggregate dependencies are used during structure validation (Phase 12)
    /// to build the high-level simulation graph and detect causal cycles between
    /// signals, members, and operators.
    ///
    /// Set during the execution block compilation pass ([`compile_execution_blocks`]).
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
// Structural Declarations (Re-exported from structural module)
// =============================================================================
// Types have been moved to ast/structural.rs to reduce file size.
// They are re-exported above for backward compatibility.

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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Scoping {
    // This will hold config/const lookups when implemented
    #[doc(hidden)]
    _placeholder: (),
}

/// Assertion to validate simulation invariants.
///
/// Assertions are non-causal checks that validate conditions during the
/// simulation lifecycle. They are typically executed during the [`Phase::Measure`]
/// phase but may also be checked during [`Phase::Resolve`] or [`Phase::Fracture`]
/// depending on their definition.
///
/// Each assertion consists of a boolean condition and metadata for error reporting.
/// If an assertion fails, it emits a structured fault. The simulation policy
/// determines if the failure is fatal, causes a halt, or allows continuation
/// based on the [`AssertionSeverity`].
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::{Assertion, AssertionSeverity};
/// use continuum_cdsl::ast::expr::TypedExpr;
/// use continuum_cdsl::foundation::Span;
///
/// // Create a simple assertion: temperature must be positive
/// let condition = ...; // TypedExpr evaluating to Type::Bool
/// let assertion = Assertion::new(
///     condition,
///     Some("Temperature must be positive".to_string()),
///     AssertionSeverity::Error,
///     span
/// );
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Assertion {
    /// The condition expression to validate.
    ///
    /// Must evaluate to [`Type::Bool`]. If the expression evaluates to `false`,
    /// the assertion is considered failed.
    pub condition: TypedExpr,

    /// Optional custom message to show when the assertion fails.
    ///
    /// This message is included in the structured fault emitted by the simulation.
    pub message: Option<String>,

    /// Severity level of the assertion failure.
    ///
    /// Determines the simulation's response to failure (e.g., warning, error,
    /// or fatal halt).
    pub severity: AssertionSeverity,

    /// Source location for error reporting and diagnostics.
    pub span: Span,
}

impl Assertion {
    /// Create a new assertion
    pub fn new(
        condition: TypedExpr,
        message: Option<String>,
        severity: AssertionSeverity,
        span: Span,
    ) -> Self {
        Self {
            condition,
            message,
            severity,
            span,
        }
    }
}

/// Body of an execution block in the Execution IR.
///
/// Differentiates between pure blocks (single expression) and effectful
/// blocks (statements). This distinction is critical for identifying
/// which phases an execution block can belong to (e.g., [`Phase::Resolve`]
/// blocks must be pure expressions).
///
/// # Variants
///
/// - [`ExecutionBody::Expr`]: A pure computation representing a single [`TypedExpr`].
///   Used in [`Phase::Resolve`], [`Phase::Measure`], [`Phase::Assert`], and [`Phase::Fracture`].
/// - [`ExecutionBody::Statements`]: An effectful computation containing a list of [`Stmt`].
///   Used in [`Phase::Collect`] and [`Phase::Apply`].
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::{ExecutionBody, TypedExpr, Stmt};
///
/// // A pure expression body for a Resolve phase
/// let body = ExecutionBody::Expr(typed_expr);
///
/// // An effectful statement body for a Collect phase
/// let body = ExecutionBody::Statements(vec![stmt1, stmt2]);
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ExecutionBody {
    /// Pure computation (Resolve, Measure, Assert, Fracture)
    Expr(TypedExpr),

    /// Effectful computation (Collect, Apply)
    ///
    /// Contains a list of typed statements.
    Statements(Vec<TypedStmt>),
}

/// Compiled execution block in the Execution IR.
///
/// An execution block contains the compiled code for a specific phase,
/// along with metadata about what it reads and emits. These blocks are
/// the primary input to the deterministic execution graph (DAG) builder.
///
/// # Lifecycle
///
/// `Execution` is created during the execution block compilation pass (Phase 12.5):
/// 1. Parser produces `Node.execution_blocks: Vec<(String, BlockBody)>`
/// 2. Execution compilation pass converts each block:
///    - Validates phase name against role's allowed phases (see [`RoleSpec::allowed_phases`])
///    - Type-checks the block body into an [`ExecutionBody`]
///    - Extracts signal/field reads as dependencies ([`Path`]s)
///    - Creates an [`Execution`] struct
/// 3. `Node.execution_blocks` is cleared, `Node.executions` populated
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::{Execution, ExecutionBody};
/// use continuum_cdsl::foundation::{Phase, Path, Span};
///
/// // Signal resolve block
/// let execution = Execution::new(
///     "resolve".to_string(),
///     Phase::Resolve,
///     ExecutionBody::Expr(typed_expr),
///     vec![Path::from("signal.temperature")],
///     span,
/// );
///
/// assert_eq!(execution.phase, Phase::Resolve);
/// assert_eq!(execution.reads.len(), 1);
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Execution {
    /// Name of the execution block (usually the phase name)
    pub name: String,

    /// Which phase this block executes in
    ///
    /// Determines when this execution runs relative to other operations.
    /// Must be one of the phases allowed by the node's role (see [`RoleSpec::allowed_phases`]).
    pub phase: crate::foundation::Phase,

    /// Compiled body of the execution block
    pub body: ExecutionBody,

    /// Signal/field dependencies specific to the execution block.
    ///
    /// Paths to signals and fields that the execution reads. These are used
    /// for fine-grained dependency analysis when building the deterministic
    /// execution graph (DAG) for each phase.
    ///
    /// These paths are extracted by recursively walking the expression tree in
    /// the [`ExecutionBody`]. The union of all `reads` across all execution
    /// blocks is also stored at the node level in [`Node::reads`].
    pub reads: Vec<Path>,

    /// Signal/field emission targets.
    ///
    /// Paths to signals and fields that this execution block may emit to.
    /// These are used for DAG construction to determine causal links and
    /// output dependencies.
    ///
    /// Set during the execution block compilation pass ([`compile_execution_blocks`][crate::resolve::blocks::compile_execution_blocks]).
    pub emits: Vec<Path>,

    /// Source location for error reporting
    ///
    /// Points to the execution block declaration in source code.
    /// Used for diagnostic messages when execution fails or validation errors occur.
    pub span: Span,
}

impl Execution {
    /// Create a new execution block
    ///
    /// # Parameters
    ///
    /// - `name`: Name of the block
    /// - `phase`: Which phase this executes in ([`Phase::Resolve`], [`Phase::Collect`], etc.)
    /// - `body`: The compiled [`ExecutionBody`]
    /// - `reads`: Dependencies extracted from the body
    /// - `emits`: Emission targets extracted from the body
    /// - `span`: Source location for error messages
    ///
    /// # Returns
    ///
    /// New execution block ready for DAG construction.
    pub fn new(
        name: String,
        phase: crate::foundation::Phase,
        body: ExecutionBody,
        reads: Vec<Path>,
        emits: Vec<Path>,
        span: Span,
    ) -> Self {
        Self {
            name,
            phase,
            body,
            reads,
            emits,
            span,
        }
    }
}

// Re-export untyped AST types for use in Node
//
// These are defined in the untyped module and represent the parser output
// before type resolution transforms them into TypedExpr.
pub use super::untyped::TypeExpr;

/// Validation error from semantic analysis.
///
/// Type alias to [`CompileError`](crate::error::CompileError) for validation
/// diagnostics. Validation errors are compile errors detected during the
/// validation pass (type checking, bounds validation, unit compatibility, etc.).
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::node::ValidationError;
/// use continuum_cdsl::error::ErrorKind;
/// use continuum_cdsl::foundation::Span;
///
/// let error = ValidationError::new(
///     ErrorKind::TypeMismatch,
///     span,
///     "expected Scalar<m>, found Scalar<s>".to_string(),
/// );
/// ```
pub type ValidationError = crate::error::CompileError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::block::BlockBody;
    use crate::ast::expr::{ExprKind, TypedExpr};
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
        let test_body = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        node.executions = vec![Execution::new(
            "resolve".to_string(),
            crate::foundation::Phase::Resolve,
            ExecutionBody::Expr(test_body),
            vec![],
            vec![],
            span,
        )];
        assert!(node.is_compiled());
    }

    #[test]
    fn test_node_validation_errors() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let mut node = Node::new(path, span, RoleData::Signal, ());

        assert!(!node.has_errors());

        node.validation_errors.push(ValidationError::new(
            crate::error::ErrorKind::TypeMismatch,
            Span::new(0, 0, 10, 1),
            "test error".to_string(),
        ));
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
        let test_body = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        node.executions = vec![Execution::new(
            "resolve".to_string(),
            crate::foundation::Phase::Resolve,
            ExecutionBody::Expr(test_body),
            vec![],
            vec![],
            span,
        )];
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

    #[test]
    fn test_execution_creation() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::{Phase, Type};

        let span = Span::new(0, 0, 10, 1);
        let body_expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::Bool, // Simplified for test
            span,
        );
        let body = ExecutionBody::Expr(body_expr);
        let reads = vec![Path::from_str("signal.temp"), Path::from_str("signal.prev")];

        let execution = Execution::new(
            "test".to_string(),
            Phase::Resolve,
            body,
            reads.clone(),
            vec![],
            span,
        );

        assert_eq!(execution.name, "test");
        assert_eq!(execution.phase, Phase::Resolve);
        assert_eq!(execution.span, span);
        assert_eq!(execution.reads.len(), 2);
        assert_eq!(execution.reads[0], Path::from_str("signal.temp"));
        assert_eq!(execution.reads[1], Path::from_str("signal.prev"));
    }

    #[test]
    fn test_execution_phases() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::{Phase, Type};

        let span = Span::new(0, 0, 10, 1);
        let body_expr = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );

        let phases = vec![
            Phase::Configure,
            Phase::Collect,
            Phase::Resolve,
            Phase::Fracture,
            Phase::Measure,
        ];

        for phase in phases {
            let body = ExecutionBody::Expr(body_expr.clone());
            let execution = Execution::new("test".to_string(), phase, body, vec![], vec![], span);
            assert_eq!(execution.phase, phase);
        }
    }

    #[test]
    fn test_execution_no_dependencies() {
        use crate::ast::expr::{ExprKind, TypedExpr};
        use crate::foundation::{Phase, Type};

        let span = Span::new(0, 0, 10, 1);
        let body_expr = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let body = ExecutionBody::Expr(body_expr);

        let execution = Execution::new(
            "test".to_string(),
            Phase::Resolve,
            body,
            vec![],
            vec![],
            span,
        );

        assert!(execution.reads.is_empty());
    }
}
