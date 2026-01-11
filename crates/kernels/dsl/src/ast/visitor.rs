//! Expression visitor pattern for AST traversal.
//!
//! This module provides two visitor traits for traversing expression trees:
//!
//! - [`ExprVisitor`] - Basic visitor without span information
//! - [`SpannedExprVisitor`] - Visitor with span information for each node
//!
//! Instead of duplicating match statements across validate.rs, lower.rs, etc.,
//! each use case implements the appropriate visitor trait.
//!
//! # Choosing a Visitor
//!
//! Use [`ExprVisitor`] when you only need to analyze expression structure
//! (e.g., checking if `dt_raw` is used).
//!
//! Use [`SpannedExprVisitor`] when you need source locations for each node
//! (e.g., LSP hover, go-to-definition, collecting references with positions).
//!
//! # Example: Basic Visitor
//!
//! ```ignore
//! use continuum_dsl::ast::{Expr, ExprVisitor};
//!
//! struct DtRawChecker { found: bool }
//!
//! impl ExprVisitor for DtRawChecker {
//!     fn visit_dt_raw(&mut self) -> bool {
//!         self.found = true;
//!         false // Stop traversal
//!     }
//! }
//!
//! let mut checker = DtRawChecker { found: false };
//! checker.walk(&expr);
//! if checker.found { /* ... */ }
//! ```
//!
//! # Example: Spanned Visitor for LSP
//!
//! ```ignore
//! use continuum_dsl::ast::{SpannedExprVisitor, Spanned, Expr, Path};
//! use std::ops::Range;
//!
//! struct RefCollector {
//!     refs: Vec<(Range<usize>, String)>,
//! }
//!
//! impl SpannedExprVisitor for RefCollector {
//!     fn visit_signal_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
//!         self.refs.push((span, path.to_string()));
//!         true // Continue traversal
//!     }
//! }
//!
//! let mut collector = RefCollector { refs: vec![] };
//! collector.walk(&spanned_expr);
//! // collector.refs now contains all signal references with their spans
//! ```

use super::{Expr, Path, Spanned};

/// Visitor trait for expression traversal.
///
/// All methods have default empty implementations, so visitors only need
/// to override the variants they care about. The `walk` method handles
/// recursion into child expressions.
///
/// Each `visit_*` method is called BEFORE recursing into children.
/// Return `false` from a visit method to skip visiting children of that node.
pub trait ExprVisitor {
    // === Leaf nodes (no children) ===

    /// Visit `dt_raw` keyword.
    fn visit_dt_raw(&mut self) -> bool {
        true
    }

    /// Visit a literal value.
    fn visit_literal(&mut self, _value: &super::Literal) -> bool {
        true
    }

    /// Visit a literal with unit annotation.
    fn visit_literal_with_unit(&mut self, _value: &super::Literal, _unit: &str) -> bool {
        true
    }

    /// Visit a path reference.
    fn visit_path(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `prev` keyword.
    fn visit_prev(&mut self) -> bool {
        true
    }

    /// Visit `prev.field` access.
    fn visit_prev_field(&mut self, _field: &str) -> bool {
        true
    }

    /// Visit `payload` keyword.
    fn visit_payload(&mut self) -> bool {
        true
    }

    /// Visit `payload.field` access.
    fn visit_payload_field(&mut self, _field: &str) -> bool {
        true
    }

    /// Visit `signal.path` reference.
    fn visit_signal_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `const.path` reference.
    fn visit_const_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `config.path` reference.
    fn visit_config_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `field.path` reference.
    fn visit_field_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `collected` keyword.
    fn visit_collected(&mut self) -> bool {
        true
    }

    /// Visit mathematical constant (pi, tau, e, etc).
    fn visit_math_const(&mut self, _mc: &super::MathConst) -> bool {
        true
    }

    /// Visit `self.field` in entity context.
    fn visit_self_field(&mut self, _field: &str) -> bool {
        true
    }

    /// Visit `entity.path` reference.
    fn visit_entity_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `other(entity.path)`.
    fn visit_other(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `pairs(entity.path)`.
    fn visit_pairs(&mut self, _path: &Path) -> bool {
        true
    }

    // === Compound nodes (have children that will be walked) ===

    /// Visit binary operation. Children: left, right.
    fn visit_binary(&mut self, _op: &super::BinaryOp) -> bool {
        true
    }

    /// Visit unary operation. Children: operand.
    fn visit_unary(&mut self, _op: &super::UnaryOp) -> bool {
        true
    }

    /// Visit function call. Children: function, args.
    fn visit_call(&mut self) -> bool {
        true
    }

    /// Visit method call. Children: object, args.
    fn visit_method_call(&mut self, _method: &str) -> bool {
        true
    }

    /// Visit field access. Children: object.
    fn visit_field_access(&mut self, _field: &str) -> bool {
        true
    }

    /// Visit let binding. Children: value, body.
    fn visit_let(&mut self, _name: &str) -> bool {
        true
    }

    /// Visit if expression. Children: condition, then_branch, else_branch.
    fn visit_if(&mut self) -> bool {
        true
    }

    /// Visit for loop. Children: iter, body.
    fn visit_for(&mut self, _var: &str) -> bool {
        true
    }

    /// Visit block. Children: all expressions in block.
    fn visit_block(&mut self) -> bool {
        true
    }

    /// Visit emit signal. Children: value.
    fn visit_emit_signal(&mut self, _target: &Path) -> bool {
        true
    }

    /// Visit emit field. Children: position, value.
    fn visit_emit_field(&mut self, _target: &Path) -> bool {
        true
    }

    /// Visit struct literal. Children: field values.
    fn visit_struct(&mut self) -> bool {
        true
    }

    /// Visit map. Children: sequence, function.
    fn visit_map(&mut self) -> bool {
        true
    }

    /// Visit fold. Children: sequence, init, function.
    fn visit_fold(&mut self) -> bool {
        true
    }

    /// Visit entity access. Children: instance.
    fn visit_entity_access(&mut self, _entity: &Path) -> bool {
        true
    }

    /// Visit aggregate. Children: body.
    fn visit_aggregate(&mut self, _op: &super::AggregateOp, _entity: &Path) -> bool {
        true
    }

    /// Visit filter. Children: predicate.
    fn visit_filter(&mut self, _entity: &Path) -> bool {
        true
    }

    /// Visit first. Children: predicate.
    fn visit_first(&mut self, _entity: &Path) -> bool {
        true
    }

    /// Visit nearest. Children: position.
    fn visit_nearest(&mut self, _entity: &Path) -> bool {
        true
    }

    /// Visit within. Children: position, radius.
    fn visit_within(&mut self, _entity: &Path) -> bool {
        true
    }

    // === Walk method ===

    /// Walk an expression tree, calling visit methods and recursing into children.
    fn walk(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }

    /// Walk a spanned expression.
    fn walk_spanned(&mut self, expr: &Spanned<Expr>) {
        self.walk(&expr.node);
    }
}

/// Walk an expression tree, calling visitor methods and recursing into children.
pub fn walk_expr<V: ExprVisitor + ?Sized>(visitor: &mut V, expr: &Expr) {
    match expr {
        // Leaf nodes
        Expr::DtRaw => {
            visitor.visit_dt_raw();
        }
        Expr::Literal(lit) => {
            visitor.visit_literal(lit);
        }
        Expr::LiteralWithUnit { value, unit } => {
            visitor.visit_literal_with_unit(value, unit);
        }
        Expr::Path(path) => {
            visitor.visit_path(path);
        }
        Expr::Prev => {
            visitor.visit_prev();
        }
        Expr::PrevField(field) => {
            visitor.visit_prev_field(field);
        }
        Expr::Payload => {
            visitor.visit_payload();
        }
        Expr::PayloadField(field) => {
            visitor.visit_payload_field(field);
        }
        Expr::SignalRef(path) => {
            visitor.visit_signal_ref(path);
        }
        Expr::ConstRef(path) => {
            visitor.visit_const_ref(path);
        }
        Expr::ConfigRef(path) => {
            visitor.visit_config_ref(path);
        }
        Expr::FieldRef(path) => {
            visitor.visit_field_ref(path);
        }
        Expr::Collected => {
            visitor.visit_collected();
        }
        Expr::MathConst(mc) => {
            visitor.visit_math_const(mc);
        }
        Expr::SelfField(field) => {
            visitor.visit_self_field(field);
        }
        Expr::EntityRef(path) => {
            visitor.visit_entity_ref(path);
        }
        Expr::Other(path) => {
            visitor.visit_other(path);
        }
        Expr::Pairs(path) => {
            visitor.visit_pairs(path);
        }

        // Compound nodes - call visitor then recurse if it returns true
        Expr::Binary { op, left, right } => {
            if visitor.visit_binary(op) {
                visitor.walk_spanned(left);
                visitor.walk_spanned(right);
            }
        }
        Expr::Unary { op, operand } => {
            if visitor.visit_unary(op) {
                visitor.walk_spanned(operand);
            }
        }
        Expr::Call { function, args } => {
            if visitor.visit_call() {
                visitor.walk_spanned(function);
                for arg in args {
                    visitor.walk_spanned(&arg.value);
                }
            }
        }
        Expr::MethodCall {
            object,
            method,
            args,
        } => {
            if visitor.visit_method_call(method) {
                visitor.walk_spanned(object);
                for arg in args {
                    visitor.walk_spanned(&arg.value);
                }
            }
        }
        Expr::FieldAccess { object, field } => {
            if visitor.visit_field_access(field) {
                visitor.walk_spanned(object);
            }
        }
        Expr::Let { name, value, body } => {
            if visitor.visit_let(name) {
                visitor.walk_spanned(value);
                visitor.walk_spanned(body);
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if visitor.visit_if() {
                visitor.walk_spanned(condition);
                visitor.walk_spanned(then_branch);
                if let Some(eb) = else_branch {
                    visitor.walk_spanned(eb);
                }
            }
        }
        Expr::For { var, iter, body } => {
            if visitor.visit_for(var) {
                visitor.walk_spanned(iter);
                visitor.walk_spanned(body);
            }
        }
        Expr::Block(exprs) => {
            if visitor.visit_block() {
                for e in exprs {
                    visitor.walk_spanned(e);
                }
            }
        }
        Expr::EmitSignal { target, value } => {
            if visitor.visit_emit_signal(target) {
                visitor.walk_spanned(value);
            }
        }
        Expr::EmitField {
            target,
            position,
            value,
        } => {
            if visitor.visit_emit_field(target) {
                visitor.walk_spanned(position);
                visitor.walk_spanned(value);
            }
        }
        Expr::Struct(fields) => {
            if visitor.visit_struct() {
                for (_, v) in fields {
                    visitor.walk_spanned(v);
                }
            }
        }
        Expr::Map { sequence, function } => {
            if visitor.visit_map() {
                visitor.walk_spanned(sequence);
                visitor.walk_spanned(function);
            }
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            if visitor.visit_fold() {
                visitor.walk_spanned(sequence);
                visitor.walk_spanned(init);
                visitor.walk_spanned(function);
            }
        }
        Expr::EntityAccess { entity, instance } => {
            if visitor.visit_entity_access(entity) {
                visitor.walk_spanned(instance);
            }
        }
        Expr::Aggregate { op, entity, body } => {
            if visitor.visit_aggregate(op, entity) {
                visitor.walk_spanned(body);
            }
        }
        Expr::Filter { entity, predicate } => {
            if visitor.visit_filter(entity) {
                visitor.walk_spanned(predicate);
            }
        }
        Expr::First { entity, predicate } => {
            if visitor.visit_first(entity) {
                visitor.walk_spanned(predicate);
            }
        }
        Expr::Nearest { entity, position } => {
            if visitor.visit_nearest(entity) {
                visitor.walk_spanned(position);
            }
        }
        Expr::Within {
            entity,
            position,
            radius,
        } => {
            if visitor.visit_within(entity) {
                visitor.walk_spanned(position);
                visitor.walk_spanned(radius);
            }
        }
    }
}

// === Spanned Expression Visitor ===

/// Visitor trait for span-aware expression traversal.
///
/// Similar to `ExprVisitor` but each visit method receives the span of the
/// expression being visited. This is useful for LSP features like hover,
/// go-to-definition, and reference finding.
///
/// # Example
///
/// ```ignore
/// use continuum_dsl::ast::{SpannedExprVisitor, Spanned, Expr, Path};
/// use std::ops::Range;
///
/// struct RefCollector {
///     refs: Vec<(Range<usize>, String)>,
/// }
///
/// impl SpannedExprVisitor for RefCollector {
///     fn visit_signal_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
///         self.refs.push((span, path.to_string()));
///         true
///     }
/// }
/// ```
pub trait SpannedExprVisitor {
    // === Leaf nodes (no children) ===

    /// Visit `dt_raw` keyword with its span.
    fn visit_dt_raw(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit a literal value with its span.
    fn visit_literal(&mut self, _span: std::ops::Range<usize>, _value: &super::Literal) -> bool {
        true
    }

    /// Visit a literal with unit annotation with its span.
    fn visit_literal_with_unit(
        &mut self,
        _span: std::ops::Range<usize>,
        _value: &super::Literal,
        _unit: &str,
    ) -> bool {
        true
    }

    /// Visit a path reference with its span.
    fn visit_path(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `prev` keyword with its span.
    fn visit_prev(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit `prev.field` access with its span.
    fn visit_prev_field(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit `payload` keyword with its span.
    fn visit_payload(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit `payload.field` access with its span.
    fn visit_payload_field(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit `signal.path` reference with its span.
    fn visit_signal_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `const.path` reference with its span.
    fn visit_const_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `config.path` reference with its span.
    fn visit_config_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `field.path` reference with its span.
    fn visit_field_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `collected` keyword with its span.
    fn visit_collected(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit mathematical constant with its span.
    fn visit_math_const(&mut self, _span: std::ops::Range<usize>, _mc: &super::MathConst) -> bool {
        true
    }

    /// Visit `self.field` in entity context with its span.
    fn visit_self_field(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit `entity.path` reference with its span.
    fn visit_entity_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `other(entity.path)` with its span.
    fn visit_other(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `pairs(entity.path)` with its span.
    fn visit_pairs(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    // === Compound nodes (have children that will be walked) ===

    /// Visit binary operation with its span.
    fn visit_binary(&mut self, _span: std::ops::Range<usize>, _op: &super::BinaryOp) -> bool {
        true
    }

    /// Visit unary operation with its span.
    fn visit_unary(&mut self, _span: std::ops::Range<usize>, _op: &super::UnaryOp) -> bool {
        true
    }

    /// Visit function call with its span.
    fn visit_call(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit method call with its span.
    fn visit_method_call(&mut self, _span: std::ops::Range<usize>, _method: &str) -> bool {
        true
    }

    /// Visit field access with its span.
    fn visit_field_access(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit let binding with its span.
    fn visit_let(&mut self, _span: std::ops::Range<usize>, _name: &str) -> bool {
        true
    }

    /// Visit if expression with its span.
    fn visit_if(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit for loop with its span.
    fn visit_for(&mut self, _span: std::ops::Range<usize>, _var: &str) -> bool {
        true
    }

    /// Visit block with its span.
    fn visit_block(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit emit signal with its span.
    fn visit_emit_signal(&mut self, _span: std::ops::Range<usize>, _target: &Path) -> bool {
        true
    }

    /// Visit emit field with its span.
    fn visit_emit_field(&mut self, _span: std::ops::Range<usize>, _target: &Path) -> bool {
        true
    }

    /// Visit struct literal with its span.
    fn visit_struct(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit map with its span.
    fn visit_map(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit fold with its span.
    fn visit_fold(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit entity access with its span.
    fn visit_entity_access(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit aggregate with its span.
    fn visit_aggregate(
        &mut self,
        _span: std::ops::Range<usize>,
        _op: &super::AggregateOp,
        _entity: &Path,
    ) -> bool {
        true
    }

    /// Visit filter with its span.
    fn visit_filter(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit first with its span.
    fn visit_first(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit nearest with its span.
    fn visit_nearest(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit within with its span.
    fn visit_within(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    // === Walk method ===

    /// Walk a spanned expression tree, calling visit methods with spans.
    fn walk(&mut self, expr: &Spanned<Expr>) {
        walk_spanned_expr(self, expr);
    }
}

/// Walk a spanned expression tree, calling visitor methods with spans.
pub fn walk_spanned_expr<V: SpannedExprVisitor + ?Sized>(visitor: &mut V, expr: &Spanned<Expr>) {
    let span = expr.span.clone();
    match &expr.node {
        // Leaf nodes
        Expr::DtRaw => {
            visitor.visit_dt_raw(span);
        }
        Expr::Literal(lit) => {
            visitor.visit_literal(span, lit);
        }
        Expr::LiteralWithUnit { value, unit } => {
            visitor.visit_literal_with_unit(span, value, unit);
        }
        Expr::Path(path) => {
            visitor.visit_path(span, path);
        }
        Expr::Prev => {
            visitor.visit_prev(span);
        }
        Expr::PrevField(field) => {
            visitor.visit_prev_field(span, field);
        }
        Expr::Payload => {
            visitor.visit_payload(span);
        }
        Expr::PayloadField(field) => {
            visitor.visit_payload_field(span, field);
        }
        Expr::SignalRef(path) => {
            visitor.visit_signal_ref(span, path);
        }
        Expr::ConstRef(path) => {
            visitor.visit_const_ref(span, path);
        }
        Expr::ConfigRef(path) => {
            visitor.visit_config_ref(span, path);
        }
        Expr::FieldRef(path) => {
            visitor.visit_field_ref(span, path);
        }
        Expr::Collected => {
            visitor.visit_collected(span);
        }
        Expr::MathConst(mc) => {
            visitor.visit_math_const(span, mc);
        }
        Expr::SelfField(field) => {
            visitor.visit_self_field(span, field);
        }
        Expr::EntityRef(path) => {
            visitor.visit_entity_ref(span, path);
        }
        Expr::Other(path) => {
            visitor.visit_other(span, path);
        }
        Expr::Pairs(path) => {
            visitor.visit_pairs(span, path);
        }

        // Compound nodes - call visitor then recurse if it returns true
        Expr::Binary { op, left, right } => {
            if visitor.visit_binary(span, op) {
                visitor.walk(left);
                visitor.walk(right);
            }
        }
        Expr::Unary { op, operand } => {
            if visitor.visit_unary(span, op) {
                visitor.walk(operand);
            }
        }
        Expr::Call { function, args } => {
            if visitor.visit_call(span) {
                visitor.walk(function);
                for arg in args {
                    visitor.walk(&arg.value);
                }
            }
        }
        Expr::MethodCall {
            object,
            method,
            args,
        } => {
            if visitor.visit_method_call(span, method) {
                visitor.walk(object);
                for arg in args {
                    visitor.walk(&arg.value);
                }
            }
        }
        Expr::FieldAccess { object, field } => {
            if visitor.visit_field_access(span, field) {
                visitor.walk(object);
            }
        }
        Expr::Let { name, value, body } => {
            if visitor.visit_let(span, name) {
                visitor.walk(value);
                visitor.walk(body);
            }
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if visitor.visit_if(span) {
                visitor.walk(condition);
                visitor.walk(then_branch);
                if let Some(eb) = else_branch {
                    visitor.walk(eb);
                }
            }
        }
        Expr::For { var, iter, body } => {
            if visitor.visit_for(span, var) {
                visitor.walk(iter);
                visitor.walk(body);
            }
        }
        Expr::Block(exprs) => {
            if visitor.visit_block(span) {
                for e in exprs {
                    visitor.walk(e);
                }
            }
        }
        Expr::EmitSignal { target, value } => {
            if visitor.visit_emit_signal(span, target) {
                visitor.walk(value);
            }
        }
        Expr::EmitField {
            target,
            position,
            value,
        } => {
            if visitor.visit_emit_field(span, target) {
                visitor.walk(position);
                visitor.walk(value);
            }
        }
        Expr::Struct(fields) => {
            if visitor.visit_struct(span) {
                for (_, v) in fields {
                    visitor.walk(v);
                }
            }
        }
        Expr::Map { sequence, function } => {
            if visitor.visit_map(span) {
                visitor.walk(sequence);
                visitor.walk(function);
            }
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            if visitor.visit_fold(span) {
                visitor.walk(sequence);
                visitor.walk(init);
                visitor.walk(function);
            }
        }
        Expr::EntityAccess { entity, instance } => {
            if visitor.visit_entity_access(span, entity) {
                visitor.walk(instance);
            }
        }
        Expr::Aggregate { op, entity, body } => {
            if visitor.visit_aggregate(span, op, entity) {
                visitor.walk(body);
            }
        }
        Expr::Filter { entity, predicate } => {
            if visitor.visit_filter(span, entity) {
                visitor.walk(predicate);
            }
        }
        Expr::First { entity, predicate } => {
            if visitor.visit_first(span, entity) {
                visitor.walk(predicate);
            }
        }
        Expr::Nearest { entity, position } => {
            if visitor.visit_nearest(span, entity) {
                visitor.walk(position);
            }
        }
        Expr::Within {
            entity,
            position,
            radius,
        } => {
            if visitor.visit_within(span, entity) {
                visitor.walk(position);
                visitor.walk(radius);
            }
        }
    }
}

// === Convenience visitors ===

/// Check if an expression contains `dt_raw`.
pub fn uses_dt_raw(expr: &Expr) -> bool {
    struct DtRawChecker {
        found: bool,
    }

    impl ExprVisitor for DtRawChecker {
        fn visit_dt_raw(&mut self) -> bool {
            self.found = true;
            false // Stop traversal once found
        }
    }

    let mut checker = DtRawChecker { found: false };
    checker.walk(expr);
    checker.found
}
