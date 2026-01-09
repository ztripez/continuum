//! Dependency collection for lowering.
//!
//! This module extracts signal and entity references from expressions,
//! which is used to build the execution dependency graph.

use continuum_dsl::ast::Expr;
use continuum_foundation::{EntityId, SignalId};

use super::Lowerer;

impl Lowerer {
    pub(crate) fn collect_signal_refs(&self, expr: &Expr, refs: &mut Vec<SignalId>) {
        match expr {
            Expr::SignalRef(path) => {
                let id = SignalId::from(path.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
            }
            Expr::Path(path) => {
                let joined = path.join(".");
                if !self.constants.contains_key(&joined) && !self.config.contains_key(&joined) {
                    let id = SignalId::from(joined.as_str());
                    if !refs.contains(&id) {
                        refs.push(id);
                    }
                }
            }
            Expr::Binary { left, right, .. } => {
                self.collect_signal_refs(&left.node, refs);
                self.collect_signal_refs(&right.node, refs);
            }
            Expr::Unary { operand, .. } => {
                self.collect_signal_refs(&operand.node, refs);
            }
            Expr::Call { function, args } => {
                self.collect_signal_refs(&function.node, refs);
                for arg in args {
                    self.collect_signal_refs(&arg.node, refs);
                }
            }
            Expr::MethodCall { object, args, .. } => {
                self.collect_signal_refs(&object.node, refs);
                for arg in args {
                    self.collect_signal_refs(&arg.node, refs);
                }
            }
            Expr::FieldAccess { object, .. } => {
                self.collect_signal_refs(&object.node, refs);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_signal_refs(&condition.node, refs);
                self.collect_signal_refs(&then_branch.node, refs);
                if let Some(eb) = else_branch {
                    self.collect_signal_refs(&eb.node, refs);
                }
            }
            Expr::Let { value, body, .. } => {
                self.collect_signal_refs(&value.node, refs);
                self.collect_signal_refs(&body.node, refs);
            }
            Expr::Block(exprs) => {
                for e in exprs {
                    self.collect_signal_refs(&e.node, refs);
                }
            }
            Expr::For { iter, body, .. } => {
                self.collect_signal_refs(&iter.node, refs);
                self.collect_signal_refs(&body.node, refs);
            }
            Expr::Map { sequence, function } | Expr::Fold { sequence, function, .. } => {
                self.collect_signal_refs(&sequence.node, refs);
                self.collect_signal_refs(&function.node, refs);
            }
            Expr::EmitSignal { value, .. } | Expr::EmitField { value, .. } => {
                self.collect_signal_refs(&value.node, refs);
            }
            Expr::Struct(fields) => {
                for (_, v) in fields {
                    self.collect_signal_refs(&v.node, refs);
                }
            }
            // Entity expressions - recurse into their sub-expressions
            Expr::SelfField(_) | Expr::EntityRef(_) | Expr::Other(_) | Expr::Pairs(_) => {}
            Expr::EntityAccess { instance, .. } => {
                self.collect_signal_refs(&instance.node, refs);
            }
            Expr::Aggregate { body, .. } => {
                self.collect_signal_refs(&body.node, refs);
            }
            Expr::Filter { predicate, .. } => {
                self.collect_signal_refs(&predicate.node, refs);
            }
            Expr::First { predicate, .. } => {
                self.collect_signal_refs(&predicate.node, refs);
            }
            Expr::Nearest { position, .. } => {
                self.collect_signal_refs(&position.node, refs);
            }
            Expr::Within {
                position, radius, ..
            } => {
                self.collect_signal_refs(&position.node, refs);
                self.collect_signal_refs(&radius.node, refs);
            }
            _ => {}
        }
    }

    /// Collect entity references from an expression
    pub(crate) fn collect_entity_refs(&self, expr: &Expr, refs: &mut Vec<EntityId>) {
        match expr {
            Expr::EntityRef(path) => {
                let id = EntityId::from(path.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
            }
            Expr::EntityAccess { entity, instance } => {
                let id = EntityId::from(entity.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
                self.collect_entity_refs(&instance.node, refs);
            }
            Expr::Aggregate { entity, body, .. } => {
                let id = EntityId::from(entity.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
                self.collect_entity_refs(&body.node, refs);
            }
            Expr::Other(path) | Expr::Pairs(path) => {
                let id = EntityId::from(path.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
            }
            Expr::Filter { entity, predicate } | Expr::First { entity, predicate } => {
                let id = EntityId::from(entity.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
                self.collect_entity_refs(&predicate.node, refs);
            }
            Expr::Nearest { entity, position } => {
                let id = EntityId::from(entity.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
                self.collect_entity_refs(&position.node, refs);
            }
            Expr::Within {
                entity,
                position,
                radius,
            } => {
                let id = EntityId::from(entity.join(".").as_str());
                if !refs.contains(&id) {
                    refs.push(id);
                }
                self.collect_entity_refs(&position.node, refs);
                self.collect_entity_refs(&radius.node, refs);
            }
            // Recurse into compound expressions
            Expr::Binary { left, right, .. } => {
                self.collect_entity_refs(&left.node, refs);
                self.collect_entity_refs(&right.node, refs);
            }
            Expr::Unary { operand, .. } => {
                self.collect_entity_refs(&operand.node, refs);
            }
            Expr::Call { function, args } => {
                self.collect_entity_refs(&function.node, refs);
                for arg in args {
                    self.collect_entity_refs(&arg.node, refs);
                }
            }
            Expr::MethodCall { object, args, .. } => {
                self.collect_entity_refs(&object.node, refs);
                for arg in args {
                    self.collect_entity_refs(&arg.node, refs);
                }
            }
            Expr::FieldAccess { object, .. } => {
                self.collect_entity_refs(&object.node, refs);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.collect_entity_refs(&condition.node, refs);
                self.collect_entity_refs(&then_branch.node, refs);
                if let Some(eb) = else_branch {
                    self.collect_entity_refs(&eb.node, refs);
                }
            }
            Expr::Let { value, body, .. } => {
                self.collect_entity_refs(&value.node, refs);
                self.collect_entity_refs(&body.node, refs);
            }
            Expr::Block(exprs) => {
                for e in exprs {
                    self.collect_entity_refs(&e.node, refs);
                }
            }
            _ => {}
        }
    }
}
