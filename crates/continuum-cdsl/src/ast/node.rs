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
//! 1. **Parsed** - Has `type_expr` and `execution_exprs` from source
//! 2. **Resolved** - `type_expr` cleared, `output` and `inputs` set
//! 3. **Validated** - `validation_errors` populated
//! 4. **Compiled** - `execution_exprs` cleared, `executions` and `reads` set
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

use crate::foundation::{Path, Span, Type};
use std::path::PathBuf;

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

/// Entity identifier for per-entity nodes
///
/// This will be used as the index type for members.
/// The wrapped Path points to the entity declaration.
///
/// # Examples
///
/// ```rust,ignore
/// let entity_id = EntityId(Path::from_str("plate"));
/// let member = Node::new(path, span, RoleData::Signal, entity_id);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct EntityId(
    /// Path to the entity declaration
    pub Path,
);

impl Index for EntityId {}

/// Unified node structure - everything flows through this
///
/// Node<I> is the single structure used throughout compilation:
/// - Parser produces Node<()> and Node<EntityId>
/// - Type resolution adds `output` and clears `type_expr`
/// - Validation adds `validation_errors`
/// - Compilation adds `executions` and `reads`, clears `execution_exprs`
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

    /// Input type (what this node receives)
    ///
    /// For impulses: type of payload
    /// For operators on entities: type of entity data
    /// None if no inputs.
    pub inputs: Option<Type>,

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
    /// Type expression from source (cleared after resolution)
    pub type_expr: Option<TypeExpr>,

    /// Execution expressions from source (cleared after compilation)
    ///
    /// Map from phase name to expression. Compiler converts these to
    /// typed Execution structs with explicit phase enum values.
    pub execution_exprs: Vec<(String, Expr)>,

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
    /// (`type_expr`, `execution_exprs`, etc) start empty - they will be
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
            inputs: None,
            index,
            type_expr: None,
            execution_exprs: Vec::new(),
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
    /// - `execution_exprs` have been cleared (consumed by compiler)
    /// - `executions` have been added (compiled execution blocks)
    ///
    /// # Returns
    ///
    /// `true` if compilation has completed, `false` otherwise
    pub fn is_compiled(&self) -> bool {
        self.execution_exprs.is_empty() && !self.executions.is_empty()
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

/// Stratum identifier (execution lane)
///
/// Identifies which execution stratum a node belongs to.
/// Strata control execution cadence and scheduling.
///
/// **Current status:** Minimal implementation - path-based ID only
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StratumId(
    /// Path to the stratum declaration
    pub Path,
);

/// Type expression from source (before resolution)
///
/// Represents a type as written in source code, before resolution
/// determines the actual Type. Examples: `Scalar<m/s>`, `Vec3<N>`, `Plate`.
///
/// **Current status:** Placeholder - not yet implemented
#[derive(Clone, Debug)]
pub struct TypeExpr {
    // This will hold unresolved type syntax when implemented
    #[doc(hidden)]
    _placeholder: (),
}

/// Expression from source (before compilation)
///
/// Represents an expression as parsed from source code, before type
/// checking and compilation. Will contain AST nodes for operators,
/// literals, function calls, etc.
///
/// **Current status:** Placeholder - not yet implemented
#[derive(Clone, Debug)]
pub struct Expr {
    // This will hold untyped expression tree when implemented
    #[doc(hidden)]
    _placeholder: (),
}

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
        let entity_path = Path::from_str("test.entity");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(
            path.clone(),
            span,
            RoleData::Signal,
            EntityId(entity_path.clone()),
        );

        assert_eq!(node.path, path);
        assert_eq!(node.index, EntityId(entity_path));
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

        // After compilation: execution_exprs cleared, executions set
        node.execution_exprs = Vec::new();
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
        let entity1 = EntityId(Path::from_str("entity1"));
        let entity2 = EntityId(Path::from_str("entity2"));
        let entity1_dup = EntityId(Path::from_str("entity1"));

        assert_eq!(entity1, entity1_dup);
        assert_ne!(entity1, entity2);
    }

    #[test]
    fn test_node_lifecycle_boundaries() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // output set but type_expr still present -> not resolved
        node.type_expr = Some(TypeExpr { _placeholder: () });
        node.output = Some(Type::Bool);
        assert!(!node.is_resolved());

        // type_expr cleared but no output -> not resolved
        node.type_expr = None;
        node.output = None;
        assert!(!node.is_resolved());

        // execution_exprs empty but no executions -> not compiled
        node.execution_exprs = Vec::new();
        node.executions = Vec::new();
        assert!(!node.is_compiled());

        // executions present but execution_exprs not empty -> not compiled
        node.execution_exprs = vec![("test".to_string(), Expr { _placeholder: () })];
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
        assert!(node.inputs.is_none());
        assert!(node.type_expr.is_none());
        assert!(node.execution_exprs.is_empty());
        assert!(node.reads.is_empty());
        assert!(node.validation_errors.is_empty());
    }
}
