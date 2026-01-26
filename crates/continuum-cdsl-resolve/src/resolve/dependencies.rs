//! Dependency extraction for execution blocks.
//!
//! This module implements the logic for identifying all external data
//! dependencies (signals, fields, config, constants) and structural
//! dependencies (entity set references) from expression trees.

use continuum_cdsl_ast::foundation::{Path, Type};
use continuum_cdsl_ast::{ExprKind, ExpressionVisitor, StatementVisitor, TypedExpr, TypedStmt};
use std::collections::HashSet;

/// Extracts signal and field paths from an expression tree.
///
/// Recursively walks the expression tree using [`DependencyVisitor`] to find all
/// path references that represent data reads. These paths are used during
/// Phase 13 (DAG Construction) to determine execution order and detect cycles.
///
/// # Determinism
///
/// Returns a tuple of `(reads, temporal_reads)` in **deterministic sorted order**
/// to satisfy the engine's core determinism invariant.
pub fn extract_dependencies(expr: &TypedExpr, current_node_path: &Path) -> (Vec<Path>, Vec<Path>) {
    let mut visitor = DependencyVisitor {
        paths: HashSet::new(),
        temporal_paths: HashSet::new(),
        current_node_path: current_node_path.clone(),
    };
    expr.walk(&mut visitor);

    use crate::resolve::utils::sort_unique;
    (
        sort_unique(visitor.paths),
        sort_unique(visitor.temporal_paths),
    )
}

/// Visitor that collects signal and field paths from an expression tree.
struct DependencyVisitor {
    /// Accumulated set of unique dependency paths.
    paths: HashSet<Path>,
    /// Accumulated set of unique temporal dependency paths (via 'prev').
    temporal_paths: HashSet<Path>,
    /// The path of the node whose execution block is being compiled.
    current_node_path: Path,
}

impl ExpressionVisitor for DependencyVisitor {
    fn visit_expr(&mut self, expr: &TypedExpr) {
        match &expr.expr {
            ExprKind::Signal(path) | ExprKind::Field(path) => {
                self.paths.insert(path.clone());
            }
            // Config and Const are not part of the signal dependency graph.
            // They are resolved statically during Configure phase and cannot
            // participate in signal resolution cycles.
            ExprKind::Config(_) | ExprKind::Const(_) => {
                // No dependency tracking needed
            }
            ExprKind::Aggregate { source, .. } | ExprKind::Fold { source, .. } => {
                // Iterating over an entity set is a read dependency on the entity's lifetime
                if let ExprKind::Entity(entity_id) = &source.expr {
                    self.paths
                        .insert(Path::from(entity_id.0.to_string().as_str()));
                }
            }
            ExprKind::Nearest { entity, .. }
            | ExprKind::Within { entity, .. }
            | ExprKind::Neighbors { entity, .. } => {
                self.paths.insert(Path::from(entity.0.to_string().as_str()));
            }
            ExprKind::Filter { source, .. } => {
                // Recursion handles source
            }
            ExprKind::Entity(entity_id) => {
                self.paths
                    .insert(Path::from(entity_id.0.to_string().as_str()));
            }
            ExprKind::FieldAccess { object, field } => {
                // BUG FIX (continuum-b8wv): Don't insert member paths when object is Prev.
                // Temporal self-references should not create cross-signal dependencies.
                if !matches!(object.expr, ExprKind::Prev) {
                    if let Type::User(type_id) = &object.ty {
                        // Accessing a member on a user type (entity or struct)
                        // In CDSL, members are identified by Entity.Member
                        self.paths
                            .insert(Path::from(type_id.to_string().as_str()).append(field));
                    }
                }
            }
            ExprKind::Index { entity, .. } => {
                // Index access creates dependency on the entity
                self.paths.insert(Path::from(entity.0.to_string().as_str()));
            }
            // Temporal tracking (continuum-ak0c): Capture self-referential temporal read
            ExprKind::Prev => {
                self.temporal_paths.insert(self.current_node_path.clone());
            }
            // Leaf values and binding structures that don't directly introduce
            // new path dependencies (dependencies are extracted from their sub-expressions
            // during the recursive walk)
            ExprKind::Literal { .. }
            | ExprKind::StringLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Vector(_)
            | ExprKind::Local(_)
            | ExprKind::Current
            | ExprKind::Inputs
            | ExprKind::Self_
            | ExprKind::Other
            | ExprKind::Payload
            | ExprKind::Let { .. }
            | ExprKind::Call { .. }
            | ExprKind::Struct { .. } => {}
        }
    }
}

/// Extracts side effects and read dependencies from a compiled statement.
///
/// Returns a tuple of `(reads, temporal_reads, emits)` used to build the [`Execution`] IR.
pub fn extract_stmt_dependencies(
    stmt: &TypedStmt,
    current_node_path: &Path,
) -> (Vec<Path>, Vec<Path>, Vec<Path>) {
    let mut visitor = StatementDependencyVisitor {
        current_node_path,
        reads: Vec::new(),
        temporal_reads: Vec::new(),
        emits: Vec::new(),
    };
    visitor.visit_stmt(stmt);
    (visitor.reads, visitor.temporal_reads, visitor.emits)
}

/// Visitor that extracts dependencies from statements and their expressions
struct StatementDependencyVisitor<'a> {
    current_node_path: &'a Path,
    reads: Vec<Path>,
    temporal_reads: Vec<Path>,
    emits: Vec<Path>,
}

impl<'a> StatementVisitor for StatementDependencyVisitor<'a> {
    fn visit_stmt(&mut self, stmt: &TypedStmt) {
        // Handle statement-level side effects (emits)
        match stmt {
            TypedStmt::SignalAssign { target, .. } => {
                self.emits.push(target.clone());
            }
            TypedStmt::FieldAssign { target, .. } => {
                self.emits.push(target.clone());
            }
            _ => {}
        }
        // Continue with default traversal to visit expressions
        self.walk_stmt(stmt);
    }

    fn visit_expr(&mut self, expr: &TypedExpr) {
        let (r, t) = extract_dependencies(expr, self.current_node_path);
        self.reads.extend(r);
        self.temporal_reads.extend(t);
    }
}
