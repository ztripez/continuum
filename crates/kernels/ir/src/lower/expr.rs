//! Expression lowering.
//!
//! This module handles the core recursive lowering of AST expressions
//! to compiled IR expressions, including function inlining.

use std::collections::HashSet;

use continuum_dsl::ast::{self, Expr, Literal};
use continuum_foundation::{EntityId, FnId, InstanceId, SignalId};

use crate::CompiledExpr;

use super::Lowerer;

impl Lowerer {
    pub(crate) fn lower_expr(&self, expr: &Expr) -> CompiledExpr {
        self.lower_expr_with_locals(expr, &HashSet::new())
    }

    pub(crate) fn lower_expr_with_locals(
        &self,
        expr: &Expr,
        locals: &HashSet<String>,
    ) -> CompiledExpr {
        match expr {
            Expr::Literal(lit) => CompiledExpr::Literal(self.literal_to_f64_unchecked(lit)),
            Expr::LiteralWithUnit { value, .. } => {
                CompiledExpr::Literal(self.literal_to_f64_unchecked(value))
            }
            Expr::Prev | Expr::PrevField(_) => CompiledExpr::Prev,
            Expr::DtRaw => CompiledExpr::DtRaw,
            Expr::Collected => CompiledExpr::Collected,
            Expr::Path(path) => {
                // Check for local variable first (single-segment paths only)
                if path.segments.len() == 1 && locals.contains(&path.segments[0]) {
                    return CompiledExpr::Local(path.segments[0].clone());
                }
                // Could be signal, const, or config reference
                let joined = path.join(".");
                if self.constants.contains_key(&joined) {
                    CompiledExpr::Const(joined)
                } else if self.config.contains_key(&joined) {
                    CompiledExpr::Config(joined)
                } else {
                    CompiledExpr::Signal(SignalId::from(joined.as_str()))
                }
            }
            Expr::SignalRef(path) => {
                // Check if the last component is a vector accessor (x, y, z, w)
                let parts = &path.segments;
                if parts.len() > 1 {
                    let last = parts.last().unwrap();
                    if matches!(last.as_str(), "x" | "y" | "z" | "w") {
                        // This is a component access like signal.foo.bar.x
                        let signal_path = parts[..parts.len() - 1].join(".");
                        return CompiledExpr::FieldAccess {
                            object: Box::new(CompiledExpr::Signal(SignalId::from(
                                signal_path.as_str(),
                            ))),
                            field: last.clone(),
                        };
                    }
                }
                CompiledExpr::Signal(SignalId::from(path.join(".").as_str()))
            }
            Expr::ConstRef(path) => CompiledExpr::Const(path.join(".")),
            Expr::ConfigRef(path) => CompiledExpr::Config(path.join(".")),
            Expr::Binary { op, left, right } => CompiledExpr::Binary {
                op: self.lower_binary_op(*op),
                left: Box::new(self.lower_expr_with_locals(&left.node, locals)),
                right: Box::new(self.lower_expr_with_locals(&right.node, locals)),
            },
            Expr::Unary { op, operand } => CompiledExpr::Unary {
                op: self.lower_unary_op(*op),
                operand: Box::new(self.lower_expr_with_locals(&operand.node, locals)),
            },
            Expr::Call { function, args } => {
                let func_name = self.expr_to_function_name(&function.node);
                let fn_id = FnId::from(func_name.as_str());

                // Check if this is a user-defined function
                if let Some(user_fn) = self.functions.get(&fn_id) {
                    // Inline the function by wrapping body in let bindings for each param
                    let lowered_args: Vec<_> = args
                        .iter()
                        .map(|a| self.lower_expr_with_locals(&a.node, locals))
                        .collect();

                    // Build nested let expressions: let param1 = arg1 in let param2 = arg2 in body
                    let mut result = user_fn.body.clone();
                    for (param, arg) in user_fn.params.iter().rev().zip(lowered_args.iter().rev()) {
                        result = CompiledExpr::Let {
                            name: param.clone(),
                            value: Box::new(arg.clone()),
                            body: Box::new(result),
                        };
                    }
                    result
                } else {
                    // Kernel function - leave as a call
                    CompiledExpr::Call {
                        function: func_name,
                        args: args
                            .iter()
                            .map(|a| self.lower_expr_with_locals(&a.node, locals))
                            .collect(),
                    }
                }
            }
            Expr::MethodCall { object, method, args } => {
                // Method calls are lowered to function calls with object as first argument
                // e.g., obj.method(a, b) -> method(obj, a, b)
                let lowered_obj = self.lower_expr_with_locals(&object.node, locals);
                let lowered_args: Vec<_> = std::iter::once(lowered_obj)
                    .chain(args.iter().map(|a| self.lower_expr_with_locals(&a.node, locals)))
                    .collect();

                CompiledExpr::Call {
                    function: method.clone(),
                    args: lowered_args,
                }
            }
            Expr::FieldAccess { object, field } => CompiledExpr::FieldAccess {
                object: Box::new(self.lower_expr_with_locals(&object.node, locals)),
                field: field.clone(),
            },
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => CompiledExpr::If {
                condition: Box::new(self.lower_expr_with_locals(&condition.node, locals)),
                then_branch: Box::new(self.lower_expr_with_locals(&then_branch.node, locals)),
                else_branch: Box::new(
                    else_branch
                        .as_ref()
                        .map(|e| self.lower_expr_with_locals(&e.node, locals))
                        .unwrap_or(CompiledExpr::Literal(0.0)),
                ),
            },
            Expr::Let { name, value, body } => {
                // Lower the value without the new local
                let lowered_value = self.lower_expr_with_locals(&value.node, locals);
                // Add the new local for the body
                let mut new_locals = locals.clone();
                new_locals.insert(name.clone());
                let lowered_body = self.lower_expr_with_locals(&body.node, &new_locals);
                CompiledExpr::Let {
                    name: name.clone(),
                    value: Box::new(lowered_value),
                    body: Box::new(lowered_body),
                }
            }
            Expr::MathConst(c) => {
                let val = match c {
                    ast::MathConst::Pi => std::f64::consts::PI,
                    ast::MathConst::Tau => std::f64::consts::TAU,
                    ast::MathConst::E => std::f64::consts::E,
                    ast::MathConst::Phi => 1.618_033_988_749_895,
                    ast::MathConst::I => 0.0, // imaginary unit not directly representable
                };
                CompiledExpr::Literal(val)
            }
            // Block, For, Map, Fold, Struct, EmitSignal, EmitField, FieldRef, Payload, PayloadField
            // These require more complex lowering or are handled specially
            Expr::Block(exprs) => {
                if exprs.is_empty() {
                    CompiledExpr::Literal(0.0)
                } else {
                    // For now, just evaluate to the last expression
                    self.lower_expr_with_locals(&exprs.last().unwrap().node, locals)
                }
            }

            // === Entity expressions ===
            Expr::SelfField(field) => CompiledExpr::SelfField(field.clone()),

            Expr::EntityRef(_path) => {
                // EntityRef by itself can't be evaluated to a value; it needs to be
                // used in aggregation context. For now, treat as placeholder.
                CompiledExpr::Literal(0.0)
            }

            Expr::EntityAccess { entity, instance } => {
                // instance expression should be a string literal for the instance ID
                let inst_id = match &instance.node {
                    Expr::Literal(Literal::String(s)) => InstanceId::from(s.as_str()),
                    _ => InstanceId::from("unknown"), // TODO: handle dynamic lookups
                };
                CompiledExpr::EntityAccess {
                    entity: EntityId::from(entity.join(".").as_str()),
                    instance: inst_id,
                    field: String::new(), // Field access happens via FieldAccess wrapping this
                }
            }

            Expr::Aggregate { op, entity, body } => CompiledExpr::Aggregate {
                op: self.lower_aggregate_op(*op),
                entity: EntityId::from(entity.join(".").as_str()),
                body: Box::new(self.lower_expr_with_locals(&body.node, locals)),
            },

            Expr::Other(path) => {
                // other() should only be used within aggregation context
                CompiledExpr::Other {
                    entity: EntityId::from(path.join(".").as_str()),
                    body: Box::new(CompiledExpr::Literal(1.0)), // placeholder
                }
            }

            Expr::Pairs(path) => CompiledExpr::Pairs {
                entity: EntityId::from(path.join(".").as_str()),
                body: Box::new(CompiledExpr::Literal(1.0)), // placeholder
            },

            Expr::Filter { entity, predicate } => CompiledExpr::Filter {
                entity: EntityId::from(entity.join(".").as_str()),
                predicate: Box::new(self.lower_expr_with_locals(&predicate.node, locals)),
                body: Box::new(CompiledExpr::Literal(1.0)), // placeholder for nested body
            },

            Expr::First { entity, predicate } => CompiledExpr::First {
                entity: EntityId::from(entity.join(".").as_str()),
                predicate: Box::new(self.lower_expr_with_locals(&predicate.node, locals)),
            },

            Expr::Nearest { entity, position } => CompiledExpr::Nearest {
                entity: EntityId::from(entity.join(".").as_str()),
                position: Box::new(self.lower_expr_with_locals(&position.node, locals)),
            },

            Expr::Within {
                entity,
                position,
                radius,
            } => CompiledExpr::Within {
                entity: EntityId::from(entity.join(".").as_str()),
                position: Box::new(self.lower_expr_with_locals(&position.node, locals)),
                radius: Box::new(self.lower_expr_with_locals(&radius.node, locals)),
                body: Box::new(CompiledExpr::Literal(1.0)), // placeholder
            },

            // Remaining expressions that need more complex handling
            _ => CompiledExpr::Literal(0.0), // placeholder for complex expressions
        }
    }

    pub(crate) fn expr_to_function_name(&self, expr: &Expr) -> String {
        match expr {
            Expr::Path(path) => path.join("."),
            Expr::FieldAccess { object, field } => {
                format!("{}.{}", self.expr_to_function_name(&object.node), field)
            }
            _ => "unknown".to_string(),
        }
    }
}
