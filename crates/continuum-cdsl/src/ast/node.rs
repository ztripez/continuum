// Unified Node<I> structure
//
// Everything is Node<I> with RoleData.
// No separate AST→IR copying - same struct flows through all passes.

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
/// EntityId(Path) points to the entity declaration.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct EntityId(pub Path);

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
    pub fn role_id(&self) -> super::role::RoleId {
        self.role.id()
    }

    /// Check if this node has been type-resolved
    pub fn is_resolved(&self) -> bool {
        self.type_expr.is_none() && self.output.is_some()
    }

    /// Check if this node has been compiled
    pub fn is_compiled(&self) -> bool {
        self.execution_exprs.is_empty() && !self.executions.is_empty()
    }

    /// Check if this node has validation errors
    pub fn has_errors(&self) -> bool {
        !self.validation_errors.is_empty()
    }
}

// =============================================================================
// Placeholder types for dependent structures
// =============================================================================
// These will be properly implemented in later phases.
// For now they're minimal stubs to make Node compile.

/// Scoping information for config/const resolution
///
/// TODO: Implement in scoping module (Phase 3.2)
#[derive(Clone, Debug)]
pub struct Scoping {
    // Placeholder
}

/// Assertion to validate invariants
///
/// TODO: Implement in assertion module (Phase 3.2)
#[derive(Clone, Debug)]
pub struct Assertion {
    // Placeholder
}

/// Compiled execution block
///
/// TODO: Implement in execution module (Phase 3.2)
#[derive(Clone, Debug)]
pub struct Execution {
    // Placeholder
}

/// Stratum identifier (execution lane)
///
/// TODO: Implement in stratum module
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StratumId(pub Path);

/// Type expression from source (before resolution)
///
/// TODO: Implement in expr module (Phase 4)
#[derive(Clone, Debug)]
pub struct TypeExpr {
    // Placeholder
}

/// Expression from source (before compilation)
///
/// TODO: Implement in expr module (Phase 4)
#[derive(Clone, Debug)]
pub struct Expr {
    // Placeholder
}

/// Validation error from semantic analysis
///
/// TODO: Implement in validation module
#[derive(Clone, Debug)]
pub struct ValidationError {
    // Placeholder
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
        let node = Node::new(
            path,
            span,
            RoleData::Field {
                reconstruction: Some(super::super::role::ReconstructionHint::Linear),
            },
            (),
        );

        assert_eq!(node.role_id(), super::super::role::RoleId::Field);
        match &node.role {
            RoleData::Field { reconstruction } => {
                assert!(reconstruction.is_some());
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
        node.executions = vec![Execution {}];
        assert!(node.is_compiled());
    }

    #[test]
    fn test_node_validation_errors() {
        let path = Path::from_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let mut node = Node::new(path, span, RoleData::Signal, ());

        assert!(!node.has_errors());

        node.validation_errors.push(ValidationError {});
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
}
