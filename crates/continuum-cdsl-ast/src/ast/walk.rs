//! Expression tree walking utilities.
//!
//! Provides shared traversal logic to avoid duplicating recursive descent
//! across multiple validation and analysis passes.
//!
//! # Design
//!
//! - **Minimal API** - Single `walk_expr` function, not a trait hierarchy
//! - **Visitor pattern** - Caller provides `FnMut(&TypedExpr)` for node inspection
//! - **Pre-order traversal** - Visitor called before recursing into children
//! - **No context threading** - Visitor owns its state (simpler than passing context)
//!
//! # Why Not a Trait?
//!
//! A visitor trait would be over-engineered for this use case:
//! - We only need pre-order traversal
//! - All passes need the same traversal structure
//! - Closure-based API is simpler and more flexible
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::walk::walk_expr;
//!
//! // Count number of Call nodes
//! let mut call_count = 0;
//! walk_expr(&expr, &mut |node| {
//!     if matches!(node.expr, ExprKind::Call { .. }) {
//!         call_count += 1;
//!     }
//! });
//!
//! // Collect all accessed signals
//! let mut signals = Vec::new();
//! walk_expr(&expr, &mut |node| {
//!     if let ExprKind::Signal(id) = &node.expr {
//!         signals.push(*id);
//!     }
//! });
//! ```

use super::{ExprKind, TypedExpr};

/// Recursively walk an expression tree in pre-order, calling visitor for each node.
///
/// The visitor is called once for each node in the tree, starting with the root
/// before recursing into children. This enables validation passes to inspect nodes
/// and accumulate errors without duplicating traversal logic.
///
/// # Parameters
///
/// - `expr`: The expression tree to traverse
/// - `visitor`: Closure called for each node (receives `&TypedExpr`)
///
/// # Traversal Order
///
/// Pre-order (depth-first):
/// 1. Visit current node
/// 2. Recursively visit children (left-to-right)
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::walk::walk_expr;
/// use continuum_cdsl::ast::ExprKind;
///
/// // Find all Prev accesses
/// let mut has_prev = false;
/// walk_expr(&expr, &mut |node| {
///     if matches!(node.expr, ExprKind::Prev) {
///         has_prev = true;
///     }
/// });
/// ```
pub fn walk_expr<V>(expr: &TypedExpr, visitor: &mut V)
where
    V: FnMut(&TypedExpr),
{
    // Visit current node
    visitor(expr);

    // Recursively visit children based on expression kind
    match &expr.expr {
        // === Compound expressions with child nodes ===
        ExprKind::Call { args, .. } => {
            for arg in args {
                walk_expr(arg, visitor);
            }
        }

        ExprKind::Vector(elements) => {
            for elem in elements {
                walk_expr(elem, visitor);
            }
        }

        ExprKind::Let { value, body, .. } => {
            walk_expr(value, visitor);
            walk_expr(body, visitor);
        }

        ExprKind::Struct { fields, .. } => {
            for (_, field_expr) in fields {
                walk_expr(field_expr, visitor);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            walk_expr(object, visitor);
        }

        ExprKind::Aggregate { source, body, .. } => {
            walk_expr(source, visitor);
            walk_expr(body, visitor);
        }

        ExprKind::Fold {
            source, init, body, ..
        } => {
            walk_expr(source, visitor);
            walk_expr(init, visitor);
            walk_expr(body, visitor);
        }

        ExprKind::Filter { source, predicate } => {
            walk_expr(source, visitor);
            walk_expr(predicate, visitor);
        }

        ExprKind::Nearest { position, .. } => {
            walk_expr(position, visitor);
        }

        ExprKind::Within {
            position, radius, ..
        } => {
            walk_expr(position, visitor);
            walk_expr(radius, visitor);
        }

        ExprKind::Neighbors { instance, .. } => {
            walk_expr(instance, visitor);
        }

        // === Leaf nodes (no children to traverse) ===
        ExprKind::Literal { .. }
        | ExprKind::StringLiteral(_)
        | ExprKind::Local(_)
        | ExprKind::Signal(_)
        | ExprKind::Field(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Self_
        | ExprKind::Other
        | ExprKind::Payload
        | ExprKind::Entity(_) => {
            // No children to visit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{Span, Type};

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    fn unit_type() -> Type {
        Type::Unit
    }

    #[test]
    fn test_walk_leaf_node() {
        // Prev is a leaf - single node
        let expr = TypedExpr::new(ExprKind::Prev, unit_type(), test_span());

        let mut visit_count = 0;
        walk_expr(&expr, &mut |_| {
            visit_count += 1;
        });

        assert_eq!(visit_count, 1, "Leaf node should be visited exactly once");
    }

    #[test]
    fn test_walk_vector() {
        // Vector with 3 elements
        let expr = TypedExpr::new(
            ExprKind::Vector(vec![
                TypedExpr::new(ExprKind::Prev, unit_type(), test_span()),
                TypedExpr::new(ExprKind::Current, unit_type(), test_span()),
                TypedExpr::new(ExprKind::Inputs, unit_type(), test_span()),
            ]),
            unit_type(),
            test_span(),
        );

        // Should visit root + 3 elements = 4 nodes
        let mut visit_count = 0;
        walk_expr(&expr, &mut |_| {
            visit_count += 1;
        });

        assert_eq!(visit_count, 4, "Should visit Vector node + 3 elements");
    }

    #[test]
    fn test_walk_nested_let() {
        // let x = prev in current
        let expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(TypedExpr::new(ExprKind::Prev, unit_type(), test_span())),
                body: Box::new(TypedExpr::new(ExprKind::Current, unit_type(), test_span())),
            },
            unit_type(),
            test_span(),
        );

        // Should visit Let + value (Prev) + body (Current) = 3 nodes
        let mut visit_count = 0;
        walk_expr(&expr, &mut |_| {
            visit_count += 1;
        });

        assert_eq!(visit_count, 3, "Should visit Let + value + body");
    }

    #[test]
    fn test_walk_counts_capability_nodes() {
        // Vector with mixed capability nodes
        let expr = TypedExpr::new(
            ExprKind::Vector(vec![
                TypedExpr::new(ExprKind::Prev, unit_type(), test_span()),
                TypedExpr::new(ExprKind::Current, unit_type(), test_span()),
                TypedExpr::new(ExprKind::Inputs, unit_type(), test_span()),
            ]),
            unit_type(),
            test_span(),
        );

        // Count how many Prev/Current/Inputs nodes
        let mut capability_count = 0;
        walk_expr(&expr, &mut |node| {
            if matches!(
                node.expr,
                ExprKind::Prev | ExprKind::Current | ExprKind::Inputs
            ) {
                capability_count += 1;
            }
        });

        assert_eq!(capability_count, 3, "Should find 3 capability nodes");
    }

    #[test]
    fn test_walk_visitor_state() {
        // Verify visitor can accumulate state
        let expr = TypedExpr::new(
            ExprKind::Vector(vec![
                TypedExpr::new(ExprKind::Prev, unit_type(), test_span()),
                TypedExpr::new(ExprKind::Inputs, unit_type(), test_span()),
            ]),
            unit_type(),
            test_span(),
        );

        let mut has_prev = false;
        let mut has_inputs = false;

        walk_expr(&expr, &mut |node| {
            if matches!(node.expr, ExprKind::Prev) {
                has_prev = true;
            }
            if matches!(node.expr, ExprKind::Inputs) {
                has_inputs = true;
            }
        });

        assert!(has_prev, "Should detect Prev node");
        assert!(has_inputs, "Should detect Inputs node");
    }
}
