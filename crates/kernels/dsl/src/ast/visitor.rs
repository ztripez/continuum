//! Expression visitor pattern for AST traversal.
//!
//! This module provides two visitor traits for traversing expression trees:
//!
//! - [`ExprVisitor`] - Basic visitor without span information
//! - [`SpannedExprVisitor`] - Visitor with span information for each node
//!
//! Instead of duplicating match statements across validate.rs, lower.rs, etc.,
//! each use case implements the appropriate visitor trait.

use super::expr::{AggregateOp, BinaryOp, CallArg, Literal, MathConst, UnaryOp};
use super::items::{
    ApplyBlock, AssertBlock, AssertSeverity, Assertion, ChronicleDef, ConfigBlock, ConfigEntry,
    ConstBlock, ConstEntry, CountBounds, EntityDef, EraDef, FieldDef, FnDef, FnParam, FractureDef,
    ImpulseDef, MeasureBlock, MemberDef, ObserveBlock, ObserveHandler, OperatorBody, OperatorDef,
    OperatorPhase, PolicyBlock, ResolveBlock, SignalDef, StrataDef, StrataState, StrataStateKind,
    Topology, Transition, TypeDef, TypeField, ValueWithUnit, WarmupBlock, WorldDef,
};
use super::{
    CompilationUnit, Expr, Item, Path, Range, SeqConstraint, Spanned, TensorConstraint, TypeExpr,
};

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

    /// Visit `sim_time` keyword.
    fn visit_sim_time(&mut self) -> bool {
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

    /// Visit `signal.name` reference.
    fn visit_signal_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `const.name` reference.
    fn visit_const_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `config.name` reference.
    fn visit_config_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `field.name` reference.
    fn visit_field_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `collected` keyword.
    fn visit_collected(&mut self) -> bool {
        true
    }

    /// Visit mathematical constant (PI, TAU, etc.).
    fn visit_math_const(&mut self, _mc: &super::MathConst) -> bool {
        true
    }

    /// Visit `self.field` reference.
    fn visit_self_field(&mut self, _field: &str) -> bool {
        true
    }

    /// Visit `entity.name` reference.
    fn visit_entity_ref(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `other(entity.name)` reference.
    fn visit_other(&mut self, _path: &Path) -> bool {
        true
    }

    /// Visit `pairs(entity.name)` reference.
    fn visit_pairs(&mut self, _path: &Path) -> bool {
        true
    }

    // === Compound nodes (recursable) ===

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

    /// Visit vector literal. Children: elements.
    fn visit_vector(&mut self) -> bool {
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

/// Visitor trait for expression traversal with span information.
pub trait SpannedExprVisitor {
    // === Leaf nodes ===

    /// Visit `dt_raw` keyword.
    fn visit_dt_raw(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit `sim_time` keyword.
    fn visit_sim_time(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit literal value.
    fn visit_literal(&mut self, _span: std::ops::Range<usize>, _value: &super::Literal) -> bool {
        true
    }

    /// Visit literal with unit.
    fn visit_literal_with_unit(
        &mut self,
        _span: std::ops::Range<usize>,
        _value: &super::Literal,
        _unit: &str,
    ) -> bool {
        true
    }

    /// Visit path reference.
    fn visit_path(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `prev` keyword.
    fn visit_prev(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit `prev.field` access.
    fn visit_prev_field(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit `payload` keyword.
    fn visit_payload(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit `payload.field` access.
    fn visit_payload_field(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit `signal.name` reference.
    fn visit_signal_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `const.name` reference.
    fn visit_const_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `config.name` reference.
    fn visit_config_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `field.name` reference.
    fn visit_field_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `collected` keyword.
    fn visit_collected(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit math constant.
    fn visit_math_const(&mut self, _span: std::ops::Range<usize>, _mc: &super::MathConst) -> bool {
        true
    }

    /// Visit `self.field` reference.
    fn visit_self_field(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit `entity.name` reference.
    fn visit_entity_ref(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `other(entity.name)` reference.
    fn visit_other(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    /// Visit `pairs(entity.name)` reference.
    fn visit_pairs(&mut self, _span: std::ops::Range<usize>, _path: &Path) -> bool {
        true
    }

    // === Compound nodes ===

    /// Visit binary operation.
    fn visit_binary(&mut self, _span: std::ops::Range<usize>, _op: &super::BinaryOp) -> bool {
        true
    }

    /// Visit unary operation.
    fn visit_unary(&mut self, _span: std::ops::Range<usize>, _op: &super::UnaryOp) -> bool {
        true
    }

    /// Visit function call.
    fn visit_call(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit method call.
    fn visit_method_call(&mut self, _span: std::ops::Range<usize>, _method: &str) -> bool {
        true
    }

    /// Visit field access.
    fn visit_field_access(&mut self, _span: std::ops::Range<usize>, _field: &str) -> bool {
        true
    }

    /// Visit let binding.
    fn visit_let(&mut self, _span: std::ops::Range<usize>, _name: &str) -> bool {
        true
    }

    /// Visit if expression.
    fn visit_if(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit for loop.
    fn visit_for(&mut self, _span: std::ops::Range<usize>, _var: &str) -> bool {
        true
    }

    /// Visit block.
    fn visit_block(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit emit signal.
    fn visit_emit_signal(&mut self, _span: std::ops::Range<usize>, _target: &Path) -> bool {
        true
    }

    /// Visit emit field.
    fn visit_emit_field(&mut self, _span: std::ops::Range<usize>, _target: &Path) -> bool {
        true
    }

    /// Visit struct literal.
    fn visit_struct(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit vector literal.
    fn visit_vector(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit map.
    fn visit_map(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit fold.
    fn visit_fold(&mut self, _span: std::ops::Range<usize>) -> bool {
        true
    }

    /// Visit entity access.
    fn visit_entity_access(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit aggregate.
    fn visit_aggregate(
        &mut self,
        _span: std::ops::Range<usize>,
        _op: &super::AggregateOp,
        _entity: &Path,
    ) -> bool {
        true
    }

    /// Visit filter.
    fn visit_filter(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit first.
    fn visit_first(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit nearest.
    fn visit_nearest(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    /// Visit within.
    fn visit_within(&mut self, _span: std::ops::Range<usize>, _entity: &Path) -> bool {
        true
    }

    // === Walk methods ===

    /// Walk a spanned expression tree.
    fn walk(&mut self, expr: &Spanned<Expr>) {
        walk_spanned_expr(self, expr);
    }
}

/// Check if an expression tree uses the `dt_raw` keyword.
pub fn uses_dt_raw(expr: &Expr) -> bool {
    struct DtRawVisitor {
        found: bool,
    }
    impl ExprVisitor for DtRawVisitor {
        fn visit_dt_raw(&mut self) -> bool {
            self.found = true;
            false // Stop walking
        }
    }
    let mut visitor = DtRawVisitor { found: false };
    visitor.walk(expr);
    visitor.found
}

/// Walk an expression tree, calling visitor methods.
pub fn walk_expr<V: ExprVisitor + ?Sized>(visitor: &mut V, expr: &Expr) {
    match expr {
        // Leaf nodes
        Expr::DtRaw => {
            visitor.visit_dt_raw();
        }
        Expr::SimTime => {
            visitor.visit_sim_time();
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
                if let Some(else_expr) = else_branch {
                    visitor.walk_spanned(else_expr);
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
                for (_, val) in fields {
                    visitor.walk_spanned(val);
                }
            }
        }
        Expr::Vector(elems) => {
            if visitor.visit_vector() {
                for elem in elems {
                    visitor.walk_spanned(elem);
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

/// Walk a spanned expression tree, calling visitor methods with spans.
pub fn walk_spanned_expr<V: SpannedExprVisitor + ?Sized>(visitor: &mut V, expr: &Spanned<Expr>) {
    let span = expr.span.clone();
    match &expr.node {
        // Leaf nodes
        Expr::DtRaw => {
            visitor.visit_dt_raw(span);
        }
        Expr::SimTime => {
            visitor.visit_sim_time(span);
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
                if let Some(else_expr) = else_branch {
                    visitor.walk(else_expr);
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
                for (_, val) in fields {
                    visitor.walk(val);
                }
            }
        }
        Expr::Vector(elems) => {
            if visitor.visit_vector(span) {
                for elem in elems {
                    visitor.walk(elem);
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

pub trait AstVisitor {
    fn visit_compilation_unit(&mut self, unit: &CompilationUnit) {
        walk_compilation_unit(self, unit);
    }

    fn visit_item(&mut self, item: &Spanned<Item>) {
        walk_item(self, item);
    }

    fn visit_world_def(&mut self, def: &WorldDef) {
        walk_world_def(self, def);
    }

    fn visit_policy_block(&mut self, block: &PolicyBlock) {
        walk_policy_block(self, block);
    }

    fn visit_const_block(&mut self, block: &ConstBlock) {
        walk_const_block(self, block);
    }

    fn visit_const_entry(&mut self, entry: &ConstEntry) {
        walk_const_entry(self, entry);
    }

    fn visit_config_block(&mut self, block: &ConfigBlock) {
        walk_config_block(self, block);
    }

    fn visit_config_entry(&mut self, entry: &ConfigEntry) {
        walk_config_entry(self, entry);
    }

    fn visit_type_def(&mut self, def: &TypeDef) {
        walk_type_def(self, def);
    }

    fn visit_type_field(&mut self, field: &TypeField) {
        walk_type_field(self, field);
    }

    fn visit_fn_def(&mut self, def: &FnDef) {
        walk_fn_def(self, def);
    }

    fn visit_fn_param(&mut self, param: &FnParam) {
        walk_fn_param(self, param);
    }

    fn visit_strata_def(&mut self, def: &StrataDef) {
        walk_strata_def(self, def);
    }

    fn visit_era_def(&mut self, def: &EraDef) {
        walk_era_def(self, def);
    }

    fn visit_value_with_unit(&mut self, value: &ValueWithUnit) {
        walk_value_with_unit(self, value);
    }

    fn visit_strata_state(&mut self, state: &StrataState) {
        walk_strata_state(self, state);
    }

    fn visit_transition(&mut self, transition: &Transition) {
        walk_transition(self, transition);
    }

    fn visit_signal_def(&mut self, def: &SignalDef) {
        walk_signal_def(self, def);
    }

    fn visit_warmup_block(&mut self, block: &WarmupBlock) {
        walk_warmup_block(self, block);
    }

    fn visit_resolve_block(&mut self, block: &ResolveBlock) {
        walk_resolve_block(self, block);
    }

    fn visit_assert_block(&mut self, block: &AssertBlock) {
        walk_assert_block(self, block);
    }

    fn visit_assertion(&mut self, assertion: &Assertion) {
        walk_assertion(self, assertion);
    }

    fn visit_field_def(&mut self, def: &FieldDef) {
        walk_field_def(self, def);
    }

    fn visit_measure_block(&mut self, block: &MeasureBlock) {
        walk_measure_block(self, block);
    }

    fn visit_operator_def(&mut self, def: &OperatorDef) {
        walk_operator_def(self, def);
    }

    fn visit_operator_body(&mut self, body: &OperatorBody) {
        walk_operator_body(self, body);
    }

    fn visit_impulse_def(&mut self, def: &ImpulseDef) {
        walk_impulse_def(self, def);
    }

    fn visit_apply_block(&mut self, block: &ApplyBlock) {
        walk_apply_block(self, block);
    }

    fn visit_fracture_def(&mut self, def: &FractureDef) {
        walk_fracture_def(self, def);
    }

    fn visit_chronicle_def(&mut self, def: &ChronicleDef) {
        walk_chronicle_def(self, def);
    }

    fn visit_observe_block(&mut self, block: &ObserveBlock) {
        walk_observe_block(self, block);
    }

    fn visit_observe_handler(&mut self, handler: &ObserveHandler) {
        walk_observe_handler(self, handler);
    }

    fn visit_member_def(&mut self, def: &MemberDef) {
        walk_member_def(self, def);
    }

    fn visit_entity_def(&mut self, def: &EntityDef) {
        walk_entity_def(self, def);
    }

    fn visit_count_bounds(&mut self, bounds: &CountBounds) {
        walk_count_bounds(self, bounds);
    }

    fn visit_expr(&mut self, expr: &Spanned<Expr>) {
        walk_ast_expr(self, expr);
    }

    fn visit_call_arg(&mut self, arg: &CallArg) {
        walk_call_arg(self, arg);
    }

    fn visit_type_expr(&mut self, expr: &TypeExpr) {
        walk_type_expr(self, expr);
    }

    fn visit_spanned_type_expr(&mut self, expr: &Spanned<TypeExpr>) {
        self.visit_type_expr(&expr.node);
    }

    fn visit_spanned_path(&mut self, path: &Spanned<Path>) {
        self.visit_path(&path.node);
    }

    fn visit_spanned_literal(&mut self, literal: &Spanned<Literal>) {
        self.visit_literal(&literal.node);
    }

    fn visit_spanned_string(&mut self, value: &Spanned<String>) {
        self.visit_string(&value.node);
    }

    fn visit_string(&mut self, _value: &str) {}

    fn visit_spanned_u32(&mut self, value: &Spanned<u32>) {
        self.visit_u32(value.node);
    }

    fn visit_u32(&mut self, _value: u32) {}

    fn visit_spanned_f64(&mut self, value: &Spanned<f64>) {
        self.visit_f64(value.node);
    }

    fn visit_f64(&mut self, _value: f64) {}

    fn visit_path(&mut self, _path: &Path) {}

    fn visit_literal(&mut self, _literal: &Literal) {}

    fn visit_math_const(&mut self, _math_const: &MathConst) {}

    fn visit_binary_op(&mut self, _op: &BinaryOp) {}

    fn visit_unary_op(&mut self, _op: &UnaryOp) {}

    fn visit_aggregate_op(&mut self, _op: &AggregateOp) {}

    fn visit_range(&mut self, _range: &Range) {}

    fn visit_tensor_constraint(&mut self, _constraint: &TensorConstraint) {}

    fn visit_seq_constraint(&mut self, constraint: &SeqConstraint) {
        match constraint {
            SeqConstraint::Each(range) | SeqConstraint::Sum(range) => self.visit_range(range),
        }
    }

    fn visit_topology(&mut self, _topology: &Topology) {}

    fn visit_operator_phase(&mut self, _phase: &OperatorPhase) {}

    fn visit_strata_state_kind(&mut self, _kind: &StrataStateKind) {}

    fn visit_assert_severity(&mut self, _severity: &AssertSeverity) {}

    fn walk_compilation_unit(&mut self, unit: &CompilationUnit) {
        walk_compilation_unit(self, unit);
    }

    fn walk_item(&mut self, item: &Spanned<Item>) {
        walk_item(self, item);
    }

    fn walk_expr(&mut self, expr: &Spanned<Expr>) {
        walk_ast_expr(self, expr);
    }

    fn walk_type_expr(&mut self, expr: &TypeExpr) {
        walk_type_expr(self, expr);
    }

    fn walk_call_arg(&mut self, arg: &CallArg) {
        walk_call_arg(self, arg);
    }
}

pub fn walk_compilation_unit<V: AstVisitor + ?Sized>(visitor: &mut V, unit: &CompilationUnit) {
    for item in &unit.items {
        visitor.visit_item(item);
    }
}

pub fn walk_item<V: AstVisitor + ?Sized>(visitor: &mut V, item: &Spanned<Item>) {
    match &item.node {
        Item::WorldDef(def) => visitor.visit_world_def(def),
        Item::ConstBlock(block) => visitor.visit_const_block(block),
        Item::ConfigBlock(block) => visitor.visit_config_block(block),
        Item::TypeDef(def) => visitor.visit_type_def(def),
        Item::FnDef(def) => visitor.visit_fn_def(def),
        Item::StrataDef(def) => visitor.visit_strata_def(def),
        Item::EraDef(def) => visitor.visit_era_def(def),
        Item::SignalDef(def) => visitor.visit_signal_def(def),
        Item::FieldDef(def) => visitor.visit_field_def(def),
        Item::OperatorDef(def) => visitor.visit_operator_def(def),
        Item::ImpulseDef(def) => visitor.visit_impulse_def(def),
        Item::FractureDef(def) => visitor.visit_fracture_def(def),
        Item::ChronicleDef(def) => visitor.visit_chronicle_def(def),
        Item::EntityDef(def) => visitor.visit_entity_def(def),
        Item::MemberDef(def) => visitor.visit_member_def(def),
    }
}

pub fn walk_world_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &WorldDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(version) = &def.version {
        visitor.visit_spanned_string(version);
    }
    if let Some(policy) = &def.policy {
        visitor.visit_policy_block(policy);
    }
}

pub fn walk_policy_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &PolicyBlock) {
    for entry in &block.entries {
        visitor.visit_config_entry(entry);
    }
}

pub fn walk_const_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &ConstBlock) {
    for entry in &block.entries {
        visitor.visit_const_entry(entry);
    }
}

pub fn walk_const_entry<V: AstVisitor + ?Sized>(visitor: &mut V, entry: &ConstEntry) {
    visitor.visit_spanned_path(&entry.path);
    visitor.visit_spanned_literal(&entry.value);
    if let Some(unit) = &entry.unit {
        visitor.visit_spanned_string(unit);
    }
}

pub fn walk_config_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &ConfigBlock) {
    for entry in &block.entries {
        visitor.visit_config_entry(entry);
    }
}

pub fn walk_config_entry<V: AstVisitor + ?Sized>(visitor: &mut V, entry: &ConfigEntry) {
    visitor.visit_spanned_path(&entry.path);
    visitor.visit_spanned_literal(&entry.value);
    if let Some(unit) = &entry.unit {
        visitor.visit_spanned_string(unit);
    }
}

pub fn walk_type_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &TypeDef) {
    visitor.visit_spanned_string(&def.name);
    for field in &def.fields {
        visitor.visit_type_field(field);
    }
}

pub fn walk_type_field<V: AstVisitor + ?Sized>(visitor: &mut V, field: &TypeField) {
    visitor.visit_spanned_string(&field.name);
    visitor.visit_spanned_type_expr(&field.ty);
}

pub fn walk_fn_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &FnDef) {
    visitor.visit_spanned_path(&def.path);
    for generic in &def.generics {
        visitor.visit_spanned_string(generic);
    }
    for param in &def.params {
        visitor.visit_fn_param(param);
    }
    if let Some(return_type) = &def.return_type {
        visitor.visit_spanned_type_expr(return_type);
    }
    visitor.visit_expr(&def.body);
}

pub fn walk_fn_param<V: AstVisitor + ?Sized>(visitor: &mut V, param: &FnParam) {
    visitor.visit_spanned_string(&param.name);
    if let Some(ty) = &param.ty {
        visitor.visit_spanned_type_expr(ty);
    }
}

pub fn walk_strata_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &StrataDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(symbol) = &def.symbol {
        visitor.visit_spanned_string(symbol);
    }
    if let Some(stride) = &def.stride {
        visitor.visit_spanned_u32(stride);
    }
}

pub fn walk_era_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &EraDef) {
    visitor.visit_spanned_string(&def.name);
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(dt) = &def.dt {
        visitor.visit_value_with_unit(&dt.node);
    }
    for entry in &def.config_overrides {
        visitor.visit_config_entry(entry);
    }
    for state in &def.strata_states {
        visitor.visit_strata_state(state);
    }
    for transition in &def.transitions {
        visitor.visit_transition(transition);
    }
}

pub fn walk_value_with_unit<V: AstVisitor + ?Sized>(visitor: &mut V, value: &ValueWithUnit) {
    visitor.visit_literal(&value.value);
    visitor.visit_string(&value.unit);
}

pub fn walk_strata_state<V: AstVisitor + ?Sized>(visitor: &mut V, state: &StrataState) {
    visitor.visit_spanned_path(&state.strata);
    visitor.visit_strata_state_kind(&state.state);
}

pub fn walk_transition<V: AstVisitor + ?Sized>(visitor: &mut V, transition: &Transition) {
    visitor.visit_spanned_path(&transition.target);
    for condition in &transition.conditions {
        visitor.visit_expr(condition);
    }
}

pub fn walk_signal_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &SignalDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(ty) = &def.ty {
        visitor.visit_spanned_type_expr(ty);
    }
    if let Some(strata) = &def.strata {
        visitor.visit_spanned_path(strata);
    }
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(symbol) = &def.symbol {
        visitor.visit_spanned_string(symbol);
    }
    for entry in &def.local_consts {
        visitor.visit_const_entry(entry);
    }
    for entry in &def.local_config {
        visitor.visit_config_entry(entry);
    }
    if let Some(warmup) = &def.warmup {
        visitor.visit_warmup_block(warmup);
    }
    if let Some(resolve) = &def.resolve {
        visitor.visit_resolve_block(resolve);
    }
    if let Some(assertions) = &def.assertions {
        visitor.visit_assert_block(assertions);
    }
    for constraint in &def.tensor_constraints {
        visitor.visit_tensor_constraint(constraint);
    }
    for constraint in &def.seq_constraints {
        visitor.visit_seq_constraint(constraint);
    }
}

pub fn walk_warmup_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &WarmupBlock) {
    visitor.visit_spanned_u32(&block.iterations);
    if let Some(convergence) = &block.convergence {
        visitor.visit_spanned_f64(convergence);
    }
    visitor.visit_expr(&block.iterate);
}

pub fn walk_resolve_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &ResolveBlock) {
    visitor.visit_expr(&block.body);
}

pub fn walk_assert_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &AssertBlock) {
    for assertion in &block.assertions {
        visitor.visit_assertion(assertion);
    }
}

pub fn walk_assertion<V: AstVisitor + ?Sized>(visitor: &mut V, assertion: &Assertion) {
    visitor.visit_expr(&assertion.condition);
    visitor.visit_assert_severity(&assertion.severity);
    if let Some(message) = &assertion.message {
        visitor.visit_spanned_string(message);
    }
}

pub fn walk_field_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &FieldDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(ty) = &def.ty {
        visitor.visit_spanned_type_expr(ty);
    }
    if let Some(strata) = &def.strata {
        visitor.visit_spanned_path(strata);
    }
    if let Some(topology) = &def.topology {
        visitor.visit_topology(&topology.node);
    }
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(symbol) = &def.symbol {
        visitor.visit_spanned_string(symbol);
    }
    if let Some(measure) = &def.measure {
        visitor.visit_measure_block(measure);
    }
}

pub fn walk_measure_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &MeasureBlock) {
    visitor.visit_expr(&block.body);
}

pub fn walk_operator_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &OperatorDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(strata) = &def.strata {
        visitor.visit_spanned_path(strata);
    }
    if let Some(phase) = &def.phase {
        visitor.visit_operator_phase(&phase.node);
    }
    if let Some(body) = &def.body {
        visitor.visit_operator_body(body);
    }
    if let Some(assertions) = &def.assertions {
        visitor.visit_assert_block(assertions);
    }
}

pub fn walk_operator_body<V: AstVisitor + ?Sized>(visitor: &mut V, body: &OperatorBody) {
    match body {
        OperatorBody::Warmup(expr) => visitor.visit_expr(expr),
        OperatorBody::Collect(expr) => visitor.visit_expr(expr),
        OperatorBody::Measure(expr) => visitor.visit_expr(expr),
    }
}

pub fn walk_impulse_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &ImpulseDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(payload_type) = &def.payload_type {
        visitor.visit_spanned_type_expr(payload_type);
    }
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(symbol) = &def.symbol {
        visitor.visit_spanned_string(symbol);
    }
    for entry in &def.local_config {
        visitor.visit_config_entry(entry);
    }
    if let Some(apply) = &def.apply {
        visitor.visit_apply_block(apply);
    }
}

pub fn walk_apply_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &ApplyBlock) {
    visitor.visit_expr(&block.body);
}

pub fn walk_fracture_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &FractureDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(strata) = &def.strata {
        visitor.visit_spanned_path(strata);
    }
    for entry in &def.local_config {
        visitor.visit_config_entry(entry);
    }
    for condition in &def.conditions {
        visitor.visit_expr(condition);
    }
    if let Some(emit) = &def.emit {
        visitor.visit_expr(emit);
    }
}

pub fn walk_chronicle_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &ChronicleDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(observe) = &def.observe {
        visitor.visit_observe_block(observe);
    }
}

pub fn walk_observe_block<V: AstVisitor + ?Sized>(visitor: &mut V, block: &ObserveBlock) {
    for handler in &block.handlers {
        visitor.visit_observe_handler(handler);
    }
}

pub fn walk_observe_handler<V: AstVisitor + ?Sized>(visitor: &mut V, handler: &ObserveHandler) {
    visitor.visit_expr(&handler.condition);
    visitor.visit_spanned_path(&handler.event_name);
    for (name, expr) in &handler.event_fields {
        visitor.visit_spanned_string(name);
        visitor.visit_expr(expr);
    }
}

pub fn walk_member_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &MemberDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(ty) = &def.ty {
        visitor.visit_spanned_type_expr(ty);
    }
    if let Some(strata) = &def.strata {
        visitor.visit_spanned_path(strata);
    }
    if let Some(title) = &def.title {
        visitor.visit_spanned_string(title);
    }
    if let Some(symbol) = &def.symbol {
        visitor.visit_spanned_string(symbol);
    }
    for entry in &def.local_config {
        visitor.visit_config_entry(entry);
    }
    if let Some(initial) = &def.initial {
        visitor.visit_resolve_block(initial);
    }
    if let Some(resolve) = &def.resolve {
        visitor.visit_resolve_block(resolve);
    }
    if let Some(assertions) = &def.assertions {
        visitor.visit_assert_block(assertions);
    }
}

pub fn walk_entity_def<V: AstVisitor + ?Sized>(visitor: &mut V, def: &EntityDef) {
    visitor.visit_spanned_path(&def.path);
    if let Some(count_source) = &def.count_source {
        visitor.visit_spanned_path(count_source);
    }
    if let Some(bounds) = &def.count_bounds {
        visitor.visit_count_bounds(bounds);
    }
}

pub fn walk_count_bounds<V: AstVisitor + ?Sized>(visitor: &mut V, bounds: &CountBounds) {
    visitor.visit_u32(bounds.min);
    visitor.visit_u32(bounds.max);
}

pub fn walk_call_arg<V: AstVisitor + ?Sized>(visitor: &mut V, arg: &CallArg) {
    visitor.visit_expr(&arg.value);
    if let Some(name) = &arg.name {
        visitor.visit_string(name);
    }
}

pub fn walk_type_expr<V: AstVisitor + ?Sized>(visitor: &mut V, expr: &TypeExpr) {
    match expr {
        TypeExpr::Scalar { range, .. } => {
            if let Some(range) = range {
                visitor.visit_range(range);
            }
        }
        TypeExpr::Vector { magnitude, .. } => {
            if let Some(range) = magnitude {
                visitor.visit_range(range);
            }
        }
        TypeExpr::Quat { magnitude } => {
            if let Some(range) = magnitude {
                visitor.visit_range(range);
            }
        }
        TypeExpr::Tensor { constraints, .. } => {
            for constraint in constraints {
                visitor.visit_tensor_constraint(constraint);
            }
        }
        TypeExpr::Grid { element_type, .. } => {
            visitor.visit_type_expr(element_type);
        }
        TypeExpr::Seq {
            element_type,
            constraints,
        } => {
            visitor.visit_type_expr(element_type);
            for constraint in constraints {
                visitor.visit_seq_constraint(constraint);
            }
        }
        TypeExpr::Named(_) => {}
    }
}

pub fn walk_ast_expr<V: AstVisitor + ?Sized>(visitor: &mut V, expr: &Spanned<Expr>) {
    match &expr.node {
        Expr::Literal(literal) => visitor.visit_literal(literal),
        Expr::LiteralWithUnit { value, .. } => visitor.visit_literal(value),
        Expr::Path(path) => visitor.visit_path(path),
        Expr::Prev => {}
        Expr::PrevField(_) => {}
        Expr::DtRaw => {}
        Expr::SimTime => {}
        Expr::Payload => {}
        Expr::PayloadField(_) => {}
        Expr::SignalRef(path) => visitor.visit_path(path),
        Expr::ConstRef(path) => visitor.visit_path(path),
        Expr::ConfigRef(path) => visitor.visit_path(path),
        Expr::FieldRef(path) => visitor.visit_path(path),
        Expr::Collected => {}
        Expr::MathConst(math_const) => visitor.visit_math_const(math_const),
        Expr::SelfField(_) => {}
        Expr::EntityRef(path) => visitor.visit_path(path),
        Expr::Other(path) => visitor.visit_path(path),
        Expr::Pairs(path) => visitor.visit_path(path),
        Expr::Binary { op, left, right } => {
            visitor.visit_binary_op(op);
            visitor.visit_expr(left);
            visitor.visit_expr(right);
        }
        Expr::Unary { op, operand } => {
            visitor.visit_unary_op(op);
            visitor.visit_expr(operand);
        }
        Expr::Call { function, args } => {
            visitor.visit_expr(function);
            for arg in args {
                visitor.visit_call_arg(arg);
            }
        }
        Expr::MethodCall { object, args, .. } => {
            visitor.visit_expr(object);
            for arg in args {
                visitor.visit_call_arg(arg);
            }
        }
        Expr::FieldAccess { object, .. } => {
            visitor.visit_expr(object);
        }
        Expr::Let { value, body, .. } => {
            visitor.visit_expr(value);
            visitor.visit_expr(body);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            visitor.visit_expr(condition);
            visitor.visit_expr(then_branch);
            if let Some(else_branch) = else_branch {
                visitor.visit_expr(else_branch);
            }
        }
        Expr::For { iter, body, .. } => {
            visitor.visit_expr(iter);
            visitor.visit_expr(body);
        }
        Expr::Block(exprs) => {
            for expr in exprs {
                visitor.visit_expr(expr);
            }
        }
        Expr::EmitSignal { target, value } => {
            visitor.visit_path(target);
            visitor.visit_expr(value);
        }
        Expr::EmitField {
            target,
            position,
            value,
        } => {
            visitor.visit_path(target);
            visitor.visit_expr(position);
            visitor.visit_expr(value);
        }
        Expr::Struct(fields) => {
            for (_, value) in fields {
                visitor.visit_expr(value);
            }
        }
        Expr::Vector(elems) => {
            for elem in elems {
                visitor.visit_expr(elem);
            }
        }
        Expr::Map { sequence, function } => {
            visitor.visit_expr(sequence);
            visitor.visit_expr(function);
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            visitor.visit_expr(sequence);
            visitor.visit_expr(init);
            visitor.visit_expr(function);
        }
        Expr::EntityAccess { entity, instance } => {
            visitor.visit_path(entity);
            visitor.visit_expr(instance);
        }
        Expr::Aggregate { op, entity, body } => {
            visitor.visit_aggregate_op(op);
            visitor.visit_path(entity);
            visitor.visit_expr(body);
        }
        Expr::Filter { entity, predicate } => {
            visitor.visit_path(entity);
            visitor.visit_expr(predicate);
        }
        Expr::First { entity, predicate } => {
            visitor.visit_path(entity);
            visitor.visit_expr(predicate);
        }
        Expr::Nearest { entity, position } => {
            visitor.visit_path(entity);
            visitor.visit_expr(position);
        }
        Expr::Within {
            entity,
            position,
            radius,
        } => {
            visitor.visit_path(entity);
            visitor.visit_expr(position);
            visitor.visit_expr(radius);
        }
    }
}

pub trait AstTransformer {
    fn transform_compilation_unit(&mut self, unit: CompilationUnit) -> CompilationUnit {
        walk_compilation_unit_transform(self, unit)
    }

    fn transform_item(&mut self, item: Spanned<Item>) -> Spanned<Item> {
        walk_item_transform(self, item)
    }

    fn transform_world_def(&mut self, def: WorldDef) -> WorldDef {
        walk_world_def_transform(self, def)
    }

    fn transform_policy_block(&mut self, block: PolicyBlock) -> PolicyBlock {
        walk_policy_block_transform(self, block)
    }

    fn transform_const_block(&mut self, block: ConstBlock) -> ConstBlock {
        walk_const_block_transform(self, block)
    }

    fn transform_const_entry(&mut self, entry: ConstEntry) -> ConstEntry {
        walk_const_entry_transform(self, entry)
    }

    fn transform_config_block(&mut self, block: ConfigBlock) -> ConfigBlock {
        walk_config_block_transform(self, block)
    }

    fn transform_config_entry(&mut self, entry: ConfigEntry) -> ConfigEntry {
        walk_config_entry_transform(self, entry)
    }

    fn transform_type_def(&mut self, def: TypeDef) -> TypeDef {
        walk_type_def_transform(self, def)
    }

    fn transform_type_field(&mut self, field: TypeField) -> TypeField {
        walk_type_field_transform(self, field)
    }

    fn transform_fn_def(&mut self, def: FnDef) -> FnDef {
        walk_fn_def_transform(self, def)
    }

    fn transform_fn_param(&mut self, param: FnParam) -> FnParam {
        walk_fn_param_transform(self, param)
    }

    fn transform_strata_def(&mut self, def: StrataDef) -> StrataDef {
        walk_strata_def_transform(self, def)
    }

    fn transform_era_def(&mut self, def: EraDef) -> EraDef {
        walk_era_def_transform(self, def)
    }

    fn transform_value_with_unit(&mut self, value: ValueWithUnit) -> ValueWithUnit {
        walk_value_with_unit_transform(self, value)
    }

    fn transform_strata_state(&mut self, state: StrataState) -> StrataState {
        walk_strata_state_transform(self, state)
    }

    fn transform_transition(&mut self, transition: Transition) -> Transition {
        walk_transition_transform(self, transition)
    }

    fn transform_signal_def(&mut self, def: SignalDef) -> SignalDef {
        walk_signal_def_transform(self, def)
    }

    fn transform_warmup_block(&mut self, block: WarmupBlock) -> WarmupBlock {
        walk_warmup_block_transform(self, block)
    }

    fn transform_resolve_block(&mut self, block: ResolveBlock) -> ResolveBlock {
        walk_resolve_block_transform(self, block)
    }

    fn transform_assert_block(&mut self, block: AssertBlock) -> AssertBlock {
        walk_assert_block_transform(self, block)
    }

    fn transform_assertion(&mut self, assertion: Assertion) -> Assertion {
        walk_assertion_transform(self, assertion)
    }

    fn transform_field_def(&mut self, def: FieldDef) -> FieldDef {
        walk_field_def_transform(self, def)
    }

    fn transform_measure_block(&mut self, block: MeasureBlock) -> MeasureBlock {
        walk_measure_block_transform(self, block)
    }

    fn transform_operator_def(&mut self, def: OperatorDef) -> OperatorDef {
        walk_operator_def_transform(self, def)
    }

    fn transform_operator_body(&mut self, body: OperatorBody) -> OperatorBody {
        walk_operator_body_transform(self, body)
    }

    fn transform_impulse_def(&mut self, def: ImpulseDef) -> ImpulseDef {
        walk_impulse_def_transform(self, def)
    }

    fn transform_apply_block(&mut self, block: ApplyBlock) -> ApplyBlock {
        walk_apply_block_transform(self, block)
    }

    fn transform_fracture_def(&mut self, def: FractureDef) -> FractureDef {
        walk_fracture_def_transform(self, def)
    }

    fn transform_chronicle_def(&mut self, def: ChronicleDef) -> ChronicleDef {
        walk_chronicle_def_transform(self, def)
    }

    fn transform_observe_block(&mut self, block: ObserveBlock) -> ObserveBlock {
        walk_observe_block_transform(self, block)
    }

    fn transform_observe_handler(&mut self, handler: ObserveHandler) -> ObserveHandler {
        walk_observe_handler_transform(self, handler)
    }

    fn transform_member_def(&mut self, def: MemberDef) -> MemberDef {
        walk_member_def_transform(self, def)
    }

    fn transform_entity_def(&mut self, def: EntityDef) -> EntityDef {
        walk_entity_def_transform(self, def)
    }

    fn transform_count_bounds(&mut self, bounds: CountBounds) -> CountBounds {
        CountBounds {
            min: self.transform_u32(bounds.min),
            max: self.transform_u32(bounds.max),
        }
    }

    fn transform_expr(&mut self, expr: Spanned<Expr>) -> Spanned<Expr> {
        walk_ast_expr_transform(self, expr)
    }

    fn transform_call_arg(&mut self, arg: CallArg) -> CallArg {
        walk_call_arg_transform(self, arg)
    }

    fn transform_type_expr(&mut self, expr: TypeExpr) -> TypeExpr {
        walk_type_expr_transform(self, expr)
    }

    fn transform_spanned_type_expr(&mut self, expr: Spanned<TypeExpr>) -> Spanned<TypeExpr> {
        Spanned {
            node: self.transform_type_expr(expr.node),
            span: expr.span,
        }
    }

    fn transform_path(&mut self, path: Path) -> Path {
        path
    }

    fn transform_spanned_path(&mut self, path: Spanned<Path>) -> Spanned<Path> {
        Spanned {
            node: self.transform_path(path.node),
            span: path.span,
        }
    }

    fn transform_string(&mut self, value: String) -> String {
        value
    }

    fn transform_spanned_string(&mut self, value: Spanned<String>) -> Spanned<String> {
        Spanned {
            node: self.transform_string(value.node),
            span: value.span,
        }
    }

    fn transform_u32(&mut self, value: u32) -> u32 {
        value
    }

    fn transform_spanned_u32(&mut self, value: Spanned<u32>) -> Spanned<u32> {
        Spanned {
            node: self.transform_u32(value.node),
            span: value.span,
        }
    }

    fn transform_f64(&mut self, value: f64) -> f64 {
        value
    }

    fn transform_spanned_f64(&mut self, value: Spanned<f64>) -> Spanned<f64> {
        Spanned {
            node: self.transform_f64(value.node),
            span: value.span,
        }
    }

    fn transform_literal(&mut self, literal: Literal) -> Literal {
        literal
    }

    fn transform_spanned_literal(&mut self, literal: Spanned<Literal>) -> Spanned<Literal> {
        Spanned {
            node: self.transform_literal(literal.node),
            span: literal.span,
        }
    }

    fn transform_math_const(&mut self, math_const: MathConst) -> MathConst {
        math_const
    }

    fn transform_binary_op(&mut self, op: BinaryOp) -> BinaryOp {
        op
    }

    fn transform_unary_op(&mut self, op: UnaryOp) -> UnaryOp {
        op
    }

    fn transform_aggregate_op(&mut self, op: AggregateOp) -> AggregateOp {
        op
    }

    fn transform_range(&mut self, range: Range) -> Range {
        range
    }

    fn transform_tensor_constraint(&mut self, constraint: TensorConstraint) -> TensorConstraint {
        constraint
    }

    fn transform_seq_constraint(&mut self, constraint: SeqConstraint) -> SeqConstraint {
        match constraint {
            SeqConstraint::Each(range) => SeqConstraint::Each(self.transform_range(range)),
            SeqConstraint::Sum(range) => SeqConstraint::Sum(self.transform_range(range)),
        }
    }

    fn transform_topology(&mut self, topology: Topology) -> Topology {
        topology
    }

    fn transform_operator_phase(&mut self, phase: OperatorPhase) -> OperatorPhase {
        phase
    }

    fn transform_strata_state_kind(&mut self, kind: StrataStateKind) -> StrataStateKind {
        match kind {
            StrataStateKind::Active => StrataStateKind::Active,
            StrataStateKind::ActiveWithStride(stride) => {
                StrataStateKind::ActiveWithStride(self.transform_u32(stride))
            }
            StrataStateKind::Gated => StrataStateKind::Gated,
        }
    }

    fn transform_assert_severity(&mut self, severity: AssertSeverity) -> AssertSeverity {
        severity
    }
}

pub fn walk_compilation_unit_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    unit: CompilationUnit,
) -> CompilationUnit {
    CompilationUnit {
        module_doc: unit.module_doc,
        items: unit
            .items
            .into_iter()
            .map(|item| transformer.transform_item(item))
            .collect(),
    }
}

pub fn walk_item_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    item: Spanned<Item>,
) -> Spanned<Item> {
    let span = item.span;
    let node = match item.node {
        Item::WorldDef(def) => Item::WorldDef(transformer.transform_world_def(def)),
        Item::ConstBlock(block) => Item::ConstBlock(transformer.transform_const_block(block)),
        Item::ConfigBlock(block) => Item::ConfigBlock(transformer.transform_config_block(block)),
        Item::TypeDef(def) => Item::TypeDef(transformer.transform_type_def(def)),
        Item::FnDef(def) => Item::FnDef(transformer.transform_fn_def(def)),
        Item::StrataDef(def) => Item::StrataDef(transformer.transform_strata_def(def)),
        Item::EraDef(def) => Item::EraDef(transformer.transform_era_def(def)),
        Item::SignalDef(def) => Item::SignalDef(transformer.transform_signal_def(def)),
        Item::FieldDef(def) => Item::FieldDef(transformer.transform_field_def(def)),
        Item::OperatorDef(def) => Item::OperatorDef(transformer.transform_operator_def(def)),
        Item::ImpulseDef(def) => Item::ImpulseDef(transformer.transform_impulse_def(def)),
        Item::FractureDef(def) => Item::FractureDef(transformer.transform_fracture_def(def)),
        Item::ChronicleDef(def) => Item::ChronicleDef(transformer.transform_chronicle_def(def)),
        Item::EntityDef(def) => Item::EntityDef(transformer.transform_entity_def(def)),
        Item::MemberDef(def) => Item::MemberDef(transformer.transform_member_def(def)),
    };

    Spanned { node, span }
}

pub fn walk_world_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: WorldDef,
) -> WorldDef {
    WorldDef {
        path: transformer.transform_spanned_path(def.path),
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        version: def
            .version
            .map(|version| transformer.transform_spanned_string(version)),
        policy: def
            .policy
            .map(|policy| transformer.transform_policy_block(policy)),
    }
}

pub fn walk_policy_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: PolicyBlock,
) -> PolicyBlock {
    PolicyBlock {
        entries: block
            .entries
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
    }
}

pub fn walk_const_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: ConstBlock,
) -> ConstBlock {
    ConstBlock {
        entries: block
            .entries
            .into_iter()
            .map(|entry| transformer.transform_const_entry(entry))
            .collect(),
    }
}

pub fn walk_const_entry_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    entry: ConstEntry,
) -> ConstEntry {
    ConstEntry {
        doc: entry.doc,
        path: transformer.transform_spanned_path(entry.path),
        value: transformer.transform_spanned_literal(entry.value),
        unit: entry
            .unit
            .map(|unit| transformer.transform_spanned_string(unit)),
    }
}

pub fn walk_config_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: ConfigBlock,
) -> ConfigBlock {
    ConfigBlock {
        entries: block
            .entries
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
    }
}

pub fn walk_config_entry_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    entry: ConfigEntry,
) -> ConfigEntry {
    ConfigEntry {
        doc: entry.doc,
        path: transformer.transform_spanned_path(entry.path),
        value: transformer.transform_spanned_literal(entry.value),
        unit: entry
            .unit
            .map(|unit| transformer.transform_spanned_string(unit)),
    }
}

pub fn walk_type_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: TypeDef,
) -> TypeDef {
    TypeDef {
        doc: def.doc,
        name: transformer.transform_spanned_string(def.name),
        fields: def
            .fields
            .into_iter()
            .map(|field| transformer.transform_type_field(field))
            .collect(),
    }
}

pub fn walk_type_field_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    field: TypeField,
) -> TypeField {
    TypeField {
        name: transformer.transform_spanned_string(field.name),
        ty: transformer.transform_spanned_type_expr(field.ty),
    }
}

pub fn walk_fn_def_transform<T: AstTransformer + ?Sized>(transformer: &mut T, def: FnDef) -> FnDef {
    FnDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        generics: def
            .generics
            .into_iter()
            .map(|generic| transformer.transform_spanned_string(generic))
            .collect(),
        params: def
            .params
            .into_iter()
            .map(|param| transformer.transform_fn_param(param))
            .collect(),
        return_type: def
            .return_type
            .map(|return_type| transformer.transform_spanned_type_expr(return_type)),
        body: transformer.transform_expr(def.body),
    }
}

pub fn walk_fn_param_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    param: FnParam,
) -> FnParam {
    FnParam {
        name: transformer.transform_spanned_string(param.name),
        ty: param
            .ty
            .map(|ty| transformer.transform_spanned_type_expr(ty)),
    }
}

pub fn walk_strata_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: StrataDef,
) -> StrataDef {
    StrataDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        symbol: def
            .symbol
            .map(|symbol| transformer.transform_spanned_string(symbol)),
        stride: def
            .stride
            .map(|stride| transformer.transform_spanned_u32(stride)),
    }
}

pub fn walk_era_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: EraDef,
) -> EraDef {
    EraDef {
        doc: def.doc,
        name: transformer.transform_spanned_string(def.name),
        is_initial: def.is_initial,
        is_terminal: def.is_terminal,
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        dt: def.dt.map(|dt| Spanned {
            node: transformer.transform_value_with_unit(dt.node),
            span: dt.span,
        }),
        config_overrides: def
            .config_overrides
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
        strata_states: def
            .strata_states
            .into_iter()
            .map(|state| transformer.transform_strata_state(state))
            .collect(),
        transitions: def
            .transitions
            .into_iter()
            .map(|transition| transformer.transform_transition(transition))
            .collect(),
    }
}

pub fn walk_value_with_unit_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    value: ValueWithUnit,
) -> ValueWithUnit {
    ValueWithUnit {
        value: transformer.transform_literal(value.value),
        unit: transformer.transform_string(value.unit),
    }
}

pub fn walk_strata_state_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    state: StrataState,
) -> StrataState {
    StrataState {
        strata: transformer.transform_spanned_path(state.strata),
        state: transformer.transform_strata_state_kind(state.state),
    }
}

pub fn walk_transition_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    transition: Transition,
) -> Transition {
    Transition {
        target: transformer.transform_spanned_path(transition.target),
        conditions: transition
            .conditions
            .into_iter()
            .map(|condition| transformer.transform_expr(condition))
            .collect(),
    }
}

pub fn walk_signal_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: SignalDef,
) -> SignalDef {
    SignalDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        ty: def.ty.map(|ty| transformer.transform_spanned_type_expr(ty)),
        strata: def
            .strata
            .map(|strata| transformer.transform_spanned_path(strata)),
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        symbol: def
            .symbol
            .map(|symbol| transformer.transform_spanned_string(symbol)),
        dt_raw: def.dt_raw,
        local_consts: def
            .local_consts
            .into_iter()
            .map(|entry| transformer.transform_const_entry(entry))
            .collect(),
        local_config: def
            .local_config
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
        warmup: def
            .warmup
            .map(|warmup| transformer.transform_warmup_block(warmup)),
        resolve: def
            .resolve
            .map(|resolve| transformer.transform_resolve_block(resolve)),
        assertions: def
            .assertions
            .map(|assertions| transformer.transform_assert_block(assertions)),
        tensor_constraints: def
            .tensor_constraints
            .into_iter()
            .map(|constraint| transformer.transform_tensor_constraint(constraint))
            .collect(),
        seq_constraints: def
            .seq_constraints
            .into_iter()
            .map(|constraint| transformer.transform_seq_constraint(constraint))
            .collect(),
    }
}

pub fn walk_warmup_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: WarmupBlock,
) -> WarmupBlock {
    WarmupBlock {
        iterations: transformer.transform_spanned_u32(block.iterations),
        convergence: block
            .convergence
            .map(|convergence| transformer.transform_spanned_f64(convergence)),
        iterate: transformer.transform_expr(block.iterate),
    }
}

pub fn walk_resolve_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: ResolveBlock,
) -> ResolveBlock {
    ResolveBlock {
        body: transformer.transform_expr(block.body),
    }
}

pub fn walk_assert_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: AssertBlock,
) -> AssertBlock {
    AssertBlock {
        assertions: block
            .assertions
            .into_iter()
            .map(|assertion| transformer.transform_assertion(assertion))
            .collect(),
    }
}

pub fn walk_assertion_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    assertion: Assertion,
) -> Assertion {
    Assertion {
        condition: transformer.transform_expr(assertion.condition),
        severity: transformer.transform_assert_severity(assertion.severity),
        message: assertion
            .message
            .map(|message| transformer.transform_spanned_string(message)),
    }
}

pub fn walk_field_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: FieldDef,
) -> FieldDef {
    FieldDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        ty: def.ty.map(|ty| transformer.transform_spanned_type_expr(ty)),
        strata: def
            .strata
            .map(|strata| transformer.transform_spanned_path(strata)),
        topology: def.topology.map(|topology| Spanned {
            node: transformer.transform_topology(topology.node),
            span: topology.span,
        }),
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        symbol: def
            .symbol
            .map(|symbol| transformer.transform_spanned_string(symbol)),
        measure: def
            .measure
            .map(|measure| transformer.transform_measure_block(measure)),
    }
}

pub fn walk_measure_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: MeasureBlock,
) -> MeasureBlock {
    MeasureBlock {
        body: transformer.transform_expr(block.body),
    }
}

pub fn walk_operator_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: OperatorDef,
) -> OperatorDef {
    OperatorDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        strata: def
            .strata
            .map(|strata| transformer.transform_spanned_path(strata)),
        phase: def.phase.map(|phase| Spanned {
            node: transformer.transform_operator_phase(phase.node),
            span: phase.span,
        }),
        body: def
            .body
            .map(|body| transformer.transform_operator_body(body)),
        assertions: def
            .assertions
            .map(|assertions| transformer.transform_assert_block(assertions)),
    }
}

pub fn walk_operator_body_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    body: OperatorBody,
) -> OperatorBody {
    match body {
        OperatorBody::Warmup(expr) => OperatorBody::Warmup(transformer.transform_expr(expr)),
        OperatorBody::Collect(expr) => OperatorBody::Collect(transformer.transform_expr(expr)),
        OperatorBody::Measure(expr) => OperatorBody::Measure(transformer.transform_expr(expr)),
    }
}

pub fn walk_impulse_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: ImpulseDef,
) -> ImpulseDef {
    ImpulseDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        payload_type: def
            .payload_type
            .map(|payload_type| transformer.transform_spanned_type_expr(payload_type)),
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        symbol: def
            .symbol
            .map(|symbol| transformer.transform_spanned_string(symbol)),
        local_config: def
            .local_config
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
        apply: def
            .apply
            .map(|apply| transformer.transform_apply_block(apply)),
    }
}

pub fn walk_apply_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: ApplyBlock,
) -> ApplyBlock {
    ApplyBlock {
        body: transformer.transform_expr(block.body),
    }
}

pub fn walk_fracture_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: FractureDef,
) -> FractureDef {
    FractureDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        strata: def
            .strata
            .map(|strata| transformer.transform_spanned_path(strata)),
        local_config: def
            .local_config
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
        conditions: def
            .conditions
            .into_iter()
            .map(|condition| transformer.transform_expr(condition))
            .collect(),
        emit: def.emit.map(|emit| transformer.transform_expr(emit)),
    }
}

pub fn walk_chronicle_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: ChronicleDef,
) -> ChronicleDef {
    ChronicleDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        observe: def
            .observe
            .map(|observe| transformer.transform_observe_block(observe)),
    }
}

pub fn walk_observe_block_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    block: ObserveBlock,
) -> ObserveBlock {
    ObserveBlock {
        handlers: block
            .handlers
            .into_iter()
            .map(|handler| transformer.transform_observe_handler(handler))
            .collect(),
    }
}

pub fn walk_observe_handler_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    handler: ObserveHandler,
) -> ObserveHandler {
    ObserveHandler {
        condition: transformer.transform_expr(handler.condition),
        event_name: transformer.transform_spanned_path(handler.event_name),
        event_fields: handler
            .event_fields
            .into_iter()
            .map(|(name, expr)| {
                (
                    transformer.transform_spanned_string(name),
                    transformer.transform_expr(expr),
                )
            })
            .collect(),
    }
}

pub fn walk_member_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: MemberDef,
) -> MemberDef {
    MemberDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        ty: def.ty.map(|ty| transformer.transform_spanned_type_expr(ty)),
        strata: def
            .strata
            .map(|strata| transformer.transform_spanned_path(strata)),
        title: def
            .title
            .map(|title| transformer.transform_spanned_string(title)),
        symbol: def
            .symbol
            .map(|symbol| transformer.transform_spanned_string(symbol)),
        local_config: def
            .local_config
            .into_iter()
            .map(|entry| transformer.transform_config_entry(entry))
            .collect(),
        initial: def
            .initial
            .map(|initial| transformer.transform_resolve_block(initial)),
        resolve: def
            .resolve
            .map(|resolve| transformer.transform_resolve_block(resolve)),
        assertions: def
            .assertions
            .map(|assertions| transformer.transform_assert_block(assertions)),
    }
}

pub fn walk_entity_def_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    def: EntityDef,
) -> EntityDef {
    EntityDef {
        doc: def.doc,
        path: transformer.transform_spanned_path(def.path),
        count_source: def
            .count_source
            .map(|count_source| transformer.transform_spanned_path(count_source)),
        count_bounds: def
            .count_bounds
            .map(|bounds| transformer.transform_count_bounds(bounds)),
    }
}

pub fn walk_call_arg_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    arg: CallArg,
) -> CallArg {
    CallArg {
        name: arg.name.map(|name| transformer.transform_string(name)),
        value: transformer.transform_expr(arg.value),
    }
}

pub fn walk_type_expr_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    expr: TypeExpr,
) -> TypeExpr {
    match expr {
        TypeExpr::Scalar { unit, range } => TypeExpr::Scalar {
            unit: transformer.transform_string(unit),
            range: range.map(|range| transformer.transform_range(range)),
        },
        TypeExpr::Vector {
            dim,
            unit,
            magnitude,
        } => TypeExpr::Vector {
            dim,
            unit: transformer.transform_string(unit),
            magnitude: magnitude.map(|range| transformer.transform_range(range)),
        },
        TypeExpr::Quat { magnitude } => TypeExpr::Quat {
            magnitude: magnitude.map(|range| transformer.transform_range(range)),
        },
        TypeExpr::Tensor {
            rows,
            cols,
            unit,
            constraints,
        } => TypeExpr::Tensor {
            rows,
            cols,
            unit: transformer.transform_string(unit),
            constraints: constraints
                .into_iter()
                .map(|constraint| transformer.transform_tensor_constraint(constraint))
                .collect(),
        },
        TypeExpr::Grid {
            width,
            height,
            element_type,
        } => TypeExpr::Grid {
            width,
            height,
            element_type: Box::new(transformer.transform_type_expr(*element_type)),
        },
        TypeExpr::Seq {
            element_type,
            constraints,
        } => TypeExpr::Seq {
            element_type: Box::new(transformer.transform_type_expr(*element_type)),
            constraints: constraints
                .into_iter()
                .map(|constraint| transformer.transform_seq_constraint(constraint))
                .collect(),
        },
        TypeExpr::Named(name) => TypeExpr::Named(transformer.transform_string(name)),
    }
}

pub fn walk_ast_expr_transform<T: AstTransformer + ?Sized>(
    transformer: &mut T,
    expr: Spanned<Expr>,
) -> Spanned<Expr> {
    let span = expr.span;
    let node = match expr.node {
        Expr::Literal(literal) => Expr::Literal(transformer.transform_literal(literal)),
        Expr::LiteralWithUnit { value, unit } => Expr::LiteralWithUnit {
            value: transformer.transform_literal(value),
            unit: transformer.transform_string(unit),
        },
        Expr::Path(path) => Expr::Path(transformer.transform_path(path)),
        Expr::Prev => Expr::Prev,
        Expr::PrevField(field) => Expr::PrevField(transformer.transform_string(field)),
        Expr::DtRaw => Expr::DtRaw,
        Expr::SimTime => Expr::SimTime,
        Expr::Payload => Expr::Payload,
        Expr::PayloadField(field) => Expr::PayloadField(transformer.transform_string(field)),
        Expr::SignalRef(path) => Expr::SignalRef(transformer.transform_path(path)),
        Expr::ConstRef(path) => Expr::ConstRef(transformer.transform_path(path)),
        Expr::ConfigRef(path) => Expr::ConfigRef(transformer.transform_path(path)),
        Expr::FieldRef(path) => Expr::FieldRef(transformer.transform_path(path)),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: transformer.transform_binary_op(op),
            left: Box::new(transformer.transform_expr(*left)),
            right: Box::new(transformer.transform_expr(*right)),
        },
        Expr::Unary { op, operand } => Expr::Unary {
            op: transformer.transform_unary_op(op),
            operand: Box::new(transformer.transform_expr(*operand)),
        },
        Expr::Call { function, args } => Expr::Call {
            function: Box::new(transformer.transform_expr(*function)),
            args: args
                .into_iter()
                .map(|arg| transformer.transform_call_arg(arg))
                .collect(),
        },
        Expr::MethodCall {
            object,
            method,
            args,
        } => Expr::MethodCall {
            object: Box::new(transformer.transform_expr(*object)),
            method: transformer.transform_string(method),
            args: args
                .into_iter()
                .map(|arg| transformer.transform_call_arg(arg))
                .collect(),
        },
        Expr::FieldAccess { object, field } => Expr::FieldAccess {
            object: Box::new(transformer.transform_expr(*object)),
            field: transformer.transform_string(field),
        },
        Expr::Let { name, value, body } => Expr::Let {
            name: transformer.transform_string(name),
            value: Box::new(transformer.transform_expr(*value)),
            body: Box::new(transformer.transform_expr(*body)),
        },
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => Expr::If {
            condition: Box::new(transformer.transform_expr(*condition)),
            then_branch: Box::new(transformer.transform_expr(*then_branch)),
            else_branch: else_branch.map(|expr| Box::new(transformer.transform_expr(*expr))),
        },
        Expr::For { var, iter, body } => Expr::For {
            var: transformer.transform_string(var),
            iter: Box::new(transformer.transform_expr(*iter)),
            body: Box::new(transformer.transform_expr(*body)),
        },
        Expr::Block(exprs) => Expr::Block(
            exprs
                .into_iter()
                .map(|expr| transformer.transform_expr(expr))
                .collect(),
        ),
        Expr::EmitSignal { target, value } => Expr::EmitSignal {
            target: transformer.transform_path(target),
            value: Box::new(transformer.transform_expr(*value)),
        },
        Expr::EmitField {
            target,
            position,
            value,
        } => Expr::EmitField {
            target: transformer.transform_path(target),
            position: Box::new(transformer.transform_expr(*position)),
            value: Box::new(transformer.transform_expr(*value)),
        },
        Expr::Struct(fields) => Expr::Struct(
            fields
                .into_iter()
                .map(|(name, expr)| {
                    (
                        transformer.transform_string(name),
                        transformer.transform_expr(expr),
                    )
                })
                .collect(),
        ),
        Expr::Vector(elems) => Expr::Vector(
            elems
                .into_iter()
                .map(|elem| transformer.transform_expr(elem))
                .collect(),
        ),
        Expr::Collected => Expr::Collected,
        Expr::MathConst(math_const) => {
            Expr::MathConst(transformer.transform_math_const(math_const))
        }
        Expr::Map { sequence, function } => Expr::Map {
            sequence: Box::new(transformer.transform_expr(*sequence)),
            function: Box::new(transformer.transform_expr(*function)),
        },
        Expr::Fold {
            sequence,
            init,
            function,
        } => Expr::Fold {
            sequence: Box::new(transformer.transform_expr(*sequence)),
            init: Box::new(transformer.transform_expr(*init)),
            function: Box::new(transformer.transform_expr(*function)),
        },
        Expr::SelfField(field) => Expr::SelfField(transformer.transform_string(field)),
        Expr::EntityRef(path) => Expr::EntityRef(transformer.transform_path(path)),
        Expr::EntityAccess { entity, instance } => Expr::EntityAccess {
            entity: transformer.transform_path(entity),
            instance: Box::new(transformer.transform_expr(*instance)),
        },
        Expr::Aggregate { op, entity, body } => Expr::Aggregate {
            op: transformer.transform_aggregate_op(op),
            entity: transformer.transform_path(entity),
            body: Box::new(transformer.transform_expr(*body)),
        },
        Expr::Other(path) => Expr::Other(transformer.transform_path(path)),
        Expr::Pairs(path) => Expr::Pairs(transformer.transform_path(path)),
        Expr::Filter { entity, predicate } => Expr::Filter {
            entity: transformer.transform_path(entity),
            predicate: Box::new(transformer.transform_expr(*predicate)),
        },
        Expr::First { entity, predicate } => Expr::First {
            entity: transformer.transform_path(entity),
            predicate: Box::new(transformer.transform_expr(*predicate)),
        },
        Expr::Nearest { entity, position } => Expr::Nearest {
            entity: transformer.transform_path(entity),
            position: Box::new(transformer.transform_expr(*position)),
        },
        Expr::Within {
            entity,
            position,
            radius,
        } => Expr::Within {
            entity: transformer.transform_path(entity),
            position: Box::new(transformer.transform_expr(*position)),
            radius: Box::new(transformer.transform_expr(*radius)),
        },
    };

    Spanned { node, span }
}
