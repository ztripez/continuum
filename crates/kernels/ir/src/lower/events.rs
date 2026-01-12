//! Event lowering (impulses and fractures).
//!
//! This module handles lowering impulse and fracture definitions from AST to IR.

use std::collections::HashSet;

use continuum_dsl::ast::{self, Expr, Span, Spanned};
use continuum_foundation::{FractureId, ImpulseId, SignalId, StratumId};

use crate::{CompiledEmit, CompiledExpr, CompiledFracture, CompiledImpulse, ValueType};

use super::{LowerError, Lowerer, expr::LoweringContext};

/// A captured let binding from the emit block traversal.
#[derive(Clone)]
struct LetBinding<'a> {
    name: String,
    value: &'a Spanned<Expr>,
}

impl Lowerer {
    pub(crate) fn lower_impulse(
        &mut self,
        def: &ast::ImpulseDef,
        span: Span,
    ) -> Result<(), LowerError> {
        let id = ImpulseId::from(def.path.node.clone());

        // Check for duplicate impulse definition
        if self.impulses.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("impulse.{}", id),
                file: self.file.clone(),
                span: def.path.span.clone(),
            });
        }

        let impulse = CompiledImpulse {
            file: self.file.clone(),
            span,
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

    pub(crate) fn lower_fracture(
        &mut self,
        def: &ast::FractureDef,
        span: Span,
    ) -> Result<(), LowerError> {
        let id = FractureId::from(def.path.node.clone());
        let fracture_path = def.path.node.to_string();

        // Check for duplicate fracture definition
        if self.fractures.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("fracture.{}", id),
                file: self.file.clone(),
                span: def.path.span.clone(),
            });
        }

        // Process local config blocks - add to global config with fracture-prefixed keys
        for entry in &def.local_config {
            let local_key = entry.path.node.to_string();
            // Add with full fracture path prefix: config.fracture.path.local_key
            let full_key = format!("fracture.{}.{}", fracture_path, local_key);
            let value = self.literal_to_f64(&entry.value.node, &entry.value.span)?;
            self.config.insert(full_key, value);
        }

        // Determine stratum binding
        let stratum = if let Some(s) = &def.strata {
            let id = StratumId::from(s.node.clone());
            if !self.strata.contains_key(&id) {
                return Err(LowerError::UndefinedStratum {
                    name: id.to_string(),
                    file: self.file.clone(),
                    span: s.span.clone(),
                });
            }
            id
        } else {
            // Default to first stratum if not specified
            // This maintains backward compatibility with older fracture definitions
            if let Some((first_id, _)) = self.strata.first() {
                first_id.clone()
            } else {
                return Err(LowerError::Generic {
                    message: "Fracture defined but no strata are available to bind to".to_string(),
                    file: self.file.clone(),
                    span: def.path.span.clone(),
                });
            }
        };

        let mut reads = Vec::new();
        for cond in &def.conditions {
            self.collect_signal_refs(&cond.node, &mut reads);
        }

        // Collect signal refs from the entire emit block (including let binding values)
        if let Some(emit_block) = &def.emit {
            self.collect_signal_refs(&emit_block.node, &mut reads);
        }

        // Collect emit expressions from the emit block, preserving let bindings
        let emits = self.collect_emit_expressions(def.emit.as_ref().map(|e| &e.node));

        let fracture = CompiledFracture {
            file: self.file.clone(),
            span,
            id: id.clone(),
            stratum,
            reads,
            conditions: def
                .conditions
                .iter()
                .map(|c| self.lower_expr(&c.node))
                .collect(),
            emits: emits
                .into_iter()
                .map(|(target, value_expr)| CompiledEmit {
                    target: SignalId::from(target),
                    value: value_expr,
                })
                .collect(),
        };

        self.fractures.insert(id, fracture);
        Ok(())
    }

    /// Collect emit expressions from an expression tree.
    /// Handles single EmitSignal, Block containing multiple emits, or nested structures.
    /// Returns tuples of (target_path, lowered_value_with_let_bindings).
    fn collect_emit_expressions(&self, expr: Option<&Expr>) -> Vec<(ast::Path, CompiledExpr)> {
        let mut emits = Vec::new();
        if let Some(expr) = expr {
            self.collect_emits_recursive(expr, &mut emits, Vec::new());
        }
        emits
    }

    fn collect_emits_recursive<'a>(
        &self,
        expr: &'a Expr,
        emits: &mut Vec<(ast::Path, CompiledExpr)>,
        let_bindings: Vec<LetBinding<'a>>,
    ) {
        match expr {
            Expr::EmitSignal { target, value } => {
                // Build the local context from all accumulated bindings
                let locals: HashSet<String> = let_bindings.iter().map(|b| b.name.clone()).collect();
                let ctx = LoweringContext {
                    locals,
                    value_type: None,
                };
                // Lower the emit value with the full locals context
                let lowered_value = self.lower_expr_with_context(&value.node, &ctx);
                // Wrap it in the accumulated let bindings
                let wrapped_value = self.wrap_in_let_bindings(lowered_value, &let_bindings);
                emits.push((target.clone(), wrapped_value));
            }
            Expr::Block(exprs) => {
                // Process block items, accumulating let bindings across the block.
                // In DSL, let expressions at block level introduce bindings visible
                // to all subsequent items in the block.
                let mut accumulated_bindings = let_bindings;
                for spanned_expr in exprs {
                    accumulated_bindings = self.collect_emits_with_block_bindings(
                        &spanned_expr.node,
                        emits,
                        accumulated_bindings,
                    );
                }
            }
            Expr::Let { name, value, body } => {
                // Capture this let binding and recurse into the body
                let mut new_bindings = let_bindings;
                new_bindings.push(LetBinding {
                    name: name.clone(),
                    value,
                });
                self.collect_emits_recursive(&body.node, emits, new_bindings);
            }
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Conditional emits
                self.collect_emits_recursive(&then_branch.node, emits, let_bindings.clone());
                if let Some(else_expr) = else_branch {
                    self.collect_emits_recursive(&else_expr.node, emits, let_bindings);
                }
            }
            _ => {
                // Other expressions don't contain emits
            }
        }
    }

    /// Process a block item and return updated bindings.
    ///
    /// For Let expressions at block level, this extracts all nested let bindings
    /// and adds them to the accumulated bindings, then processes the body.
    /// Returns the bindings INCLUDING any new let bindings from this expression.
    fn collect_emits_with_block_bindings<'a>(
        &self,
        expr: &'a Expr,
        emits: &mut Vec<(ast::Path, CompiledExpr)>,
        mut bindings: Vec<LetBinding<'a>>,
    ) -> Vec<LetBinding<'a>> {
        match expr {
            Expr::Let { name, value, body } => {
                // Add this let binding and continue extracting from nested lets
                bindings.push(LetBinding {
                    name: name.clone(),
                    value,
                });
                // Recursively extract more let bindings from the body
                self.collect_emits_with_block_bindings(&body.node, emits, bindings)
            }
            _ => {
                // Not a let expression - process normally and return current bindings
                self.collect_emits_recursive(expr, emits, bindings.clone());
                bindings
            }
        }
    }

    /// Wrap an expression in a chain of let bindings.
    /// The bindings are applied in reverse order (innermost first) so that
    /// the outermost let in the source becomes the outermost let in the IR.
    ///
    /// Each binding's value is lowered with the context of all EARLIER bindings,
    /// so that let expressions can reference earlier let-bound variables.
    fn wrap_in_let_bindings(
        &self,
        body_expr: CompiledExpr,
        bindings: &[LetBinding<'_>],
    ) -> CompiledExpr {
        if bindings.is_empty() {
            return body_expr;
        }

        // Build the let chain from inside out (reverse order)
        // But first, we need to lower each binding's value with the proper context
        // that includes all PREVIOUS bindings as locals.

        // Pre-lower all binding values with proper local contexts
        let mut lowered_values = Vec::with_capacity(bindings.len());
        let mut locals = HashSet::new();
        for binding in bindings {
            // Lower this binding's value with current locals
            let ctx = LoweringContext {
                locals: locals.clone(),
                value_type: None,
            };
            let lowered = self.lower_expr_with_context(&binding.value.node, &ctx);
            lowered_values.push(lowered);
            // Add this binding's name to locals for subsequent bindings
            locals.insert(binding.name.clone());
        }

        // Now build the let chain from inside out
        let mut result = body_expr;
        for (binding, lowered_value) in bindings.iter().zip(lowered_values.iter()).rev() {
            result = CompiledExpr::Let {
                name: binding.name.clone(),
                value: Box::new(lowered_value.clone()),
                body: Box::new(result),
            };
        }
        result
    }
}
