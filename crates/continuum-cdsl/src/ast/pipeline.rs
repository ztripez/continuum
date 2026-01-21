//! Pipeline traits for Node<I> lifecycle
//!
//! These traits form a supertrait hierarchy that describes the data lifecycle
//! of a Node<I> as it flows through compilation passes:
//!
//! ```text
//! Named → Parsed → Resolved → Validated → Compiled
//! ```
//!
//! Each trait adds new data that becomes available after a specific pass:
//! - **Named**: Identity (path, span) - available immediately after parsing
//! - **Parsed**: Syntax (type_expr, execution_blocks) - from parser
//! - **Resolved**: Semantics (output, inputs) - after type resolution
//! - **Validated**: Errors (validation_errors) - after validation pass
//! - **Compiled**: Execution (executions, reads) - after compilation
//!
//! # What is Node<I>?
//!
//! `Node<I>` is the unified structure for all DSL primitives (signals, fields,
//! operators, etc.). The generic parameter `I` specifies where the node lives:
//! - `Node<()>` - Global primitive (e.g., world-level signal)
//! - `Node<EntityId>` - Per-entity primitive (e.g., `plate.velocity`)
//!
//! `EntityId` is an identifier type for entity definitions (see `crate::ast::node::EntityId`)
//!
//! **Traits are read-only.** Mutation happens on the concrete Node<I> struct.
//! Pipeline functions take `&mut Node<I>`.
//!
//! # Example
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{Node, Named, Parsed, Resolved};
//!
//! fn process_parsed<T: Parsed>(node: &T) {
//!     println!("Processing: {}", node.path());  // Named
//!     if let Some(type_expr) = node.type_expr() {  // Parsed
//!         // ... resolve types
//!     }
//! }
//!
//! fn process_resolved<T: Resolved>(node: &T) {
//!     println!("Output type: {:?}", node.output());  // Resolved
//!     println!("Input types: {:?}", node.inputs());  // Resolved (named inputs)
//! }
//! ```

use super::block::BlockBody;
use super::node::{Execution, ValidationError};
use super::untyped::TypeExpr;
use crate::foundation::{Path, Span, Type};

/// Base trait - every node has identity
///
/// Provides access to the node's path and source location.
/// This is the base of the pipeline trait hierarchy.
pub trait Named {
    /// Get the hierarchical path to this node
    ///
    /// # Returns
    ///
    /// Reference to the node's hierarchical path (e.g., "world.temperature").
    fn path(&self) -> &Path;

    /// Get the source location for error messages
    ///
    /// # Returns
    ///
    /// Source span indicating file, line, and column range where this node
    /// was defined.
    fn span(&self) -> Span;
}

/// Parser output - has syntax from source
///
/// After parsing, nodes have type expressions and execution expressions
/// that will be processed by later passes.
pub trait Parsed: Named {
    /// Get the type expression from source (if present)
    ///
    /// This will be `None` after type resolution consumes it.
    ///
    /// # Returns
    ///
    /// `Some(&TypeExpr)` if the node has an unresolved type expression from source,
    /// `None` after type resolution completes (type expression is consumed).
    fn type_expr(&self) -> Option<&TypeExpr>;

    /// Get the execution blocks from source
    ///
    /// Map from phase name to block body (expression or statements).
    /// These will be cleared after compilation transforms them into
    /// typed Execution structs.
    ///
    /// # Returns
    ///
    /// Slice of (phase_name, block_body) pairs from source. Empty after
    /// compilation consumes them.
    fn execution_blocks(&self) -> &[(String, BlockBody)];
}

/// Type resolution complete - has semantic types
///
/// After type resolution, nodes have output and input types determined.
pub trait Resolved: Parsed {
    /// Get the output type (what this node produces)
    ///
    /// Returns `None` if type resolution hasn't completed yet.
    ///
    /// # Returns
    ///
    /// `Some(&Type)` if type resolution completed successfully, `None` if
    /// the node hasn't been type-resolved yet.
    fn output(&self) -> Option<&Type>;

    /// Get the named input types (what this node receives)
    ///
    /// Each input has a name and type. For signals, these are inputs from
    /// collect/emit expressions. For operators, depends on operator type.
    /// For impulses, these are payload fields.
    ///
    /// Returns empty slice before type resolution completes.
    ///
    /// # Returns
    ///
    /// Slice of (name, type) pairs representing named inputs to this node.
    /// Empty before type resolution.
    fn inputs(&self) -> &[(String, Type)];
}

/// Validation complete - has error diagnostics
///
/// After semantic validation, nodes may have validation errors recorded.
pub trait Validated: Resolved {
    /// Get validation errors found during semantic analysis
    ///
    /// Returns empty slice if no errors were found.
    ///
    /// # Returns
    ///
    /// Slice of validation errors. Empty if validation passed without errors.
    fn validation_errors(&self) -> &[ValidationError];
}

/// Compilation complete - has executable code
///
/// After compilation, nodes have execution blocks and dependency information.
pub trait Compiled: Validated {
    /// Get the compiled execution blocks
    ///
    /// Each execution has a phase, body, and metadata.
    ///
    /// # Returns
    ///
    /// Slice of compiled execution blocks for different phases. Empty before
    /// compilation.
    fn executions(&self) -> &[Execution];

    /// Get the dependency paths (what this node reads from)
    ///
    /// Used for DAG construction to determine execution order.
    ///
    /// # Returns
    ///
    /// Slice of paths to nodes this node depends on (reads from). Used for
    /// DAG construction. Empty before compilation.
    fn reads(&self) -> &[Path];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Node, RoleData};
    use crate::foundation::{Path, Span};

    #[test]
    fn test_named_trait() {
        let path = Path::from_path_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        // Named trait should be implemented
        let named: &dyn Named = &node;
        assert_eq!(named.path(), &path);
        assert_eq!(named.span(), span);
    }

    #[test]
    fn test_parsed_trait() {
        let path = Path::from_path_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        // Initially no type_expr or execution_blocks
        let parsed: &dyn Parsed = &node;
        assert!(parsed.type_expr().is_none());
        assert!(parsed.execution_blocks().is_empty());

        // Can still access Named methods through Parsed (supertrait)
        assert_eq!(parsed.path(), &path);
    }

    #[test]
    fn test_resolved_trait() {
        let path = Path::from_path_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        // Initially no output, empty inputs
        let resolved: &dyn Resolved = &node;
        assert!(resolved.output().is_none());
        assert!(resolved.inputs().is_empty());

        // Can access Parsed and Named methods through Resolved
        assert!(resolved.type_expr().is_none());
        assert_eq!(resolved.path(), &path);
    }

    #[test]
    fn test_validated_trait() {
        let path = Path::from_path_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        // Initially no validation errors
        let validated: &dyn Validated = &node;
        assert!(validated.validation_errors().is_empty());

        // Can access all previous traits through Validated
        assert!(validated.output().is_none());
        assert!(validated.type_expr().is_none());
        assert_eq!(validated.path(), &path);
    }

    #[test]
    fn test_compiled_trait() {
        let path = Path::from_path_str("test.signal");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path.clone(), span, RoleData::Signal, ());

        // Initially no executions or reads
        let compiled: &dyn Compiled = &node;
        assert!(compiled.executions().is_empty());
        assert!(compiled.reads().is_empty());

        // Can access all previous traits through Compiled
        assert!(compiled.validation_errors().is_empty());
        assert!(compiled.output().is_none());
        assert!(compiled.type_expr().is_none());
        assert_eq!(compiled.path(), &path);
    }

    #[test]
    fn test_trait_hierarchy() {
        let path = Path::from_path_str("test.operator");
        let span = Span::new(0, 0, 10, 1);
        let node = Node::new(path, span, RoleData::Operator, ());

        // Verify supertrait relationships work correctly
        fn accepts_named<T: Named>(_: &T) {}
        fn accepts_parsed<T: Parsed>(_: &T) {}
        fn accepts_resolved<T: Resolved>(_: &T) {}
        fn accepts_validated<T: Validated>(_: &T) {}
        fn accepts_compiled<T: Compiled>(_: &T) {}

        accepts_named(&node);
        accepts_parsed(&node);
        accepts_resolved(&node);
        accepts_validated(&node);
        accepts_compiled(&node);
    }

    #[test]
    fn test_generic_processing() {
        let signal = Node::new(
            Path::from_path_str("world.temperature"),
            Span::new(0, 0, 10, 1),
            RoleData::Signal,
            (),
        );

        let operator = Node::new(
            Path::from_path_str("update.temperature"),
            Span::new(0, 0, 10, 1),
            RoleData::Operator,
            (),
        );

        // Generic function that works with any Parsed node
        fn process_parsed<T: Parsed>(node: &T) -> &Path {
            node.path()
        }

        assert_eq!(
            process_parsed(&signal),
            &Path::from_path_str("world.temperature")
        );
        assert_eq!(
            process_parsed(&operator),
            &Path::from_path_str("update.temperature")
        );
    }
}
