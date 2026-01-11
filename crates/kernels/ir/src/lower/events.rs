//! Event lowering (impulses and fractures).
//!
//! This module handles lowering impulse and fracture definitions from AST to IR.

use std::collections::HashSet;

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

        // Collect signal refs from the entire emit block (including let binding values)
        if let Some(emit_block) = &def.emit {
            self.collect_signal_refs(&emit_block.node, &mut reads);
        }

        // Collect emit expressions from the emit block
        let emits = self.collect_emit_expressions(def.emit.as_ref().map(|e| &e.node));

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
                .map(|(target, value, locals)| CompiledEmit {
                    target: SignalId::from(target.join(".").as_str()),
                    value: self.lower_expr_with_locals(&value, &locals),
                })
                .collect(),
        };

        self.fractures.insert(id, fracture);
        Ok(())
    }

    /// Collect emit expressions from an expression tree.
    /// Handles single EmitSignal, Block containing multiple emits, or nested structures.
    /// Returns tuples of (target_path, value_expr, locals_in_scope).
    fn collect_emit_expressions<'a>(
        &self,
        expr: Option<&'a Expr>,
    ) -> Vec<(ast::Path, &'a Expr, HashSet<String>)> {
        let mut emits = Vec::new();
        if let Some(expr) = expr {
            self.collect_emits_recursive(expr, &mut emits, HashSet::new());
        }
        emits
    }

    fn collect_emits_recursive<'a>(
        &self,
        expr: &'a Expr,
        emits: &mut Vec<(ast::Path, &'a Expr, HashSet<String>)>,
        locals: HashSet<String>,
    ) {
        match expr {
            Expr::EmitSignal { target, value } => {
                emits.push((target.clone(), &value.node, locals));
            }
            Expr::Block(exprs) => {
                for spanned_expr in exprs {
                    self.collect_emits_recursive(&spanned_expr.node, emits, locals.clone());
                }
            }
            Expr::Let { name, body, .. } => {
                // Let bindings can contain emit expressions in their body
                // Track the bound variable name for proper lowering
                let mut new_locals = locals;
                new_locals.insert(name.clone());
                self.collect_emits_recursive(&body.node, emits, new_locals);
            }
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Conditional emits
                self.collect_emits_recursive(&then_branch.node, emits, locals.clone());
                if let Some(else_expr) = else_branch {
                    self.collect_emits_recursive(&else_expr.node, emits, locals);
                }
            }
            _ => {
                // Other expressions don't contain emits
            }
        }
    }
}
