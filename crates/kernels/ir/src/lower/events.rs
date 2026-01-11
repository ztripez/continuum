//! Event lowering (impulses and fractures).
//!
//! This module handles lowering impulse and fracture definitions from AST to IR.

use continuum_dsl::ast::{self, Expr};
use continuum_foundation::{FractureId, ImpulseId, SignalId};

use crate::{CompiledEmit, CompiledFracture, CompiledImpulse, ValueType};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_impulse(&mut self, def: &ast::ImpulseDef) -> Result<(), LowerError> {
        let id = ImpulseId::from(def.path.node.join(".").as_str());

        // Check for duplicate impulse definition
        if self.impulses.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("impulse.{}", id.0)));
        }

        let impulse = CompiledImpulse {
            id: id.clone(),
            payload_type: def
                .payload_type
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar {
                    unit: None,
                    dimension: None,
                    range: None,
                }),
            apply: def.apply.as_ref().map(|a| self.lower_expr(&a.body.node)),
        };

        self.impulses.insert(id, impulse);
        Ok(())
    }

    pub(crate) fn lower_fracture(&mut self, def: &ast::FractureDef) -> Result<(), LowerError> {
        let id = FractureId::from(def.path.node.join(".").as_str());

        // Check for duplicate fracture definition
        if self.fractures.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!(
                "fracture.{}",
                id.0
            )));
        }

        let mut reads = Vec::new();
        for cond in &def.conditions {
            self.collect_signal_refs(&cond.node, &mut reads);
        }

        // Collect emit expressions from the emit block
        let emits = self.collect_emit_expressions(def.emit.as_ref().map(|e| &e.node));
        for (_, value_expr) in &emits {
            self.collect_signal_refs(value_expr, &mut reads);
        }

        let fracture = CompiledFracture {
            id: id.clone(),
            reads,
            conditions: def
                .conditions
                .iter()
                .map(|c| self.lower_expr(&c.node))
                .collect(),
            emits: emits
                .into_iter()
                .map(|(target, value)| CompiledEmit {
                    target: SignalId::from(target.join(".").as_str()),
                    value: self.lower_expr(&value),
                })
                .collect(),
        };

        self.fractures.insert(id, fracture);
        Ok(())
    }

    /// Collect emit expressions from an expression tree.
    /// Handles single EmitSignal, Block containing multiple emits, or nested structures.
    fn collect_emit_expressions<'a>(
        &self,
        expr: Option<&'a Expr>,
    ) -> Vec<(ast::Path, &'a Expr)> {
        let mut emits = Vec::new();
        if let Some(expr) = expr {
            self.collect_emits_recursive(expr, &mut emits);
        }
        emits
    }

    fn collect_emits_recursive<'a>(&self, expr: &'a Expr, emits: &mut Vec<(ast::Path, &'a Expr)>) {
        match expr {
            Expr::EmitSignal { target, value } => {
                emits.push((target.clone(), &value.node));
            }
            Expr::Block(exprs) => {
                for spanned_expr in exprs {
                    self.collect_emits_recursive(&spanned_expr.node, emits);
                }
            }
            Expr::Let { body, .. } => {
                // Let bindings can contain emit expressions in their body
                self.collect_emits_recursive(&body.node, emits);
            }
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Conditional emits
                self.collect_emits_recursive(&then_branch.node, emits);
                if let Some(else_expr) = else_branch {
                    self.collect_emits_recursive(&else_expr.node, emits);
                }
            }
            _ => {
                // Other expressions don't contain emits
            }
        }
    }
}
