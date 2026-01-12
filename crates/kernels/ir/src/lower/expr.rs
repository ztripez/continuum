//! Expression lowering.
//!
//! This module handles the core recursive lowering of AST expressions
//! to compiled IR expressions, including function inlining.

use std::collections::HashSet;

use continuum_dsl::ast::{self, Expr, Literal};
use continuum_foundation::{EntityId, FnId, InstanceId, SignalId};

use crate::{CompiledExpr, DtRobustOperator, IntegrationMethod, ValueType};

use super::Lowerer;

/// Context for expression lowering, carrying type information and local variables.
///
/// This struct propagates necessary context through recursive expression lowering,
/// enabling type-aware transformations such as vector expression expansion.
#[derive(Clone)]
pub struct LoweringContext<'a> {
    /// Local variable names in scope (from let bindings and function parameters)
    pub locals: HashSet<String>,
    /// The target value type for the expression being lowered.
    /// Used for type-aware expansion of vector/tensor operations.
    pub value_type: Option<&'a ValueType>,
}

impl<'a> LoweringContext<'a> {
    /// Creates a new empty context with no locals and no type information.
    pub fn new() -> Self {
        Self {
            locals: HashSet::new(),
            value_type: None,
        }
    }

    /// Creates a context with the given value type.
    pub fn with_type(value_type: &'a ValueType) -> Self {
        Self {
            locals: HashSet::new(),
            value_type: Some(value_type),
        }
    }

    /// Returns a new context with an additional local variable.
    pub fn with_local(&self, name: String) -> Self {
        let mut new_locals = self.locals.clone();
        new_locals.insert(name);
        Self {
            locals: new_locals,
            value_type: self.value_type,
        }
    }
}

impl Default for LoweringContext<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    /// Lower an expression without any context (no locals, no type info).
    pub(crate) fn lower_expr(&self, expr: &Expr) -> CompiledExpr {
        self.lower_expr_with_context(expr, &LoweringContext::new())
    }

    /// Lower an expression with type context for type-aware expansion.
    pub(crate) fn lower_expr_typed(&self, expr: &Expr, value_type: &ValueType) -> CompiledExpr {
        self.lower_expr_with_context(expr, &LoweringContext::with_type(value_type))
    }

    /// Lower an expression with only local variables (legacy compatibility).
    pub(crate) fn lower_expr_with_locals(
        &self,
        expr: &Expr,
        locals: &HashSet<String>,
    ) -> CompiledExpr {
        let ctx = LoweringContext {
            locals: locals.clone(),
            value_type: None,
        };
        self.lower_expr_with_context(expr, &ctx)
    }

    /// Core expression lowering with full context.
    pub(crate) fn lower_expr_with_context(
        &self,
        expr: &Expr,
        ctx: &LoweringContext<'_>,
    ) -> CompiledExpr {
        match expr {
            Expr::Literal(lit) => CompiledExpr::Literal(self.literal_to_f64_unchecked(lit)),
            Expr::LiteralWithUnit { value, .. } => {
                CompiledExpr::Literal(self.literal_to_f64_unchecked(value))
            }
            Expr::Prev | Expr::PrevField(_) => CompiledExpr::Prev,
            Expr::DtRaw => CompiledExpr::DtRaw,
            Expr::SimTime => CompiledExpr::SimTime,
            Expr::Collected => CompiledExpr::Collected,
            Expr::Path(path) => {
                // Check for local variable first (single-segment paths only)
                if path.segments.len() == 1 && ctx.locals.contains(&path.segments[0]) {
                    return CompiledExpr::Local(path.segments[0].clone());
                }
                // Could be signal, const, or config reference
                let joined = path.to_string();
                if self.constants.contains_key(&joined) {
                    CompiledExpr::Const(joined)
                } else if self.config.contains_key(&joined) {
                    CompiledExpr::Config(joined)
                } else {
                    CompiledExpr::Signal(SignalId::from(path.clone()))
                }
            }
            Expr::SignalRef(path) => {
                // Check if the last component is a vector accessor (x, y, z, w)
                let parts = &path.segments;
                if parts.len() > 1 {
                    let last = parts.last().unwrap();
                    if matches!(last.as_str(), "x" | "y" | "z" | "w") {
                        // This is a component access like signal.foo.bar.x
                        let mut signal_path = path.clone();
                        signal_path.segments.pop();
                        return CompiledExpr::FieldAccess {
                            object: Box::new(CompiledExpr::Signal(SignalId::from(signal_path))),
                            field: last.clone(),
                        };
                    }
                }
                CompiledExpr::Signal(SignalId::from(path.clone()))
            }
            Expr::ConstRef(path) => CompiledExpr::Const(path.to_string()),
            Expr::ConfigRef(path) => CompiledExpr::Config(path.to_string()),
            Expr::Binary { op, left, right } => CompiledExpr::Binary {
                op: self.lower_binary_op(*op),
                left: Box::new(self.lower_expr_with_context(&left.node, ctx)),
                right: Box::new(self.lower_expr_with_context(&right.node, ctx)),
            },
            Expr::Unary { op, operand } => CompiledExpr::Unary {
                op: self.lower_unary_op(*op),
                operand: Box::new(self.lower_expr_with_context(&operand.node, ctx)),
            },
            Expr::Call { function, args } => {
                let func_name = self.expr_to_function_name(&function.node);

                // Check if this is a dt-robust operator
                if let Some(operator) = self.parse_dt_robust_operator(&func_name) {
                    let lowered_args: Vec<_> = args
                        .iter()
                        .map(|a| self.lower_expr_with_context(&a.value.node, ctx))
                        .collect();
                    CompiledExpr::DtRobustCall {
                        operator,
                        args: lowered_args,
                        method: IntegrationMethod::default(),
                    }
                } else if func_name.starts_with("kernel.") {
                    // Kernel function - engine-provided primitive
                    let kernel_name = func_name.strip_prefix("kernel.").unwrap().to_string();
                    CompiledExpr::KernelCall {
                        function: kernel_name,
                        args: args
                            .iter()
                            .map(|a| self.lower_expr_with_context(&a.value.node, ctx))
                            .collect(),
                    }
                } else {
                    let fn_id = FnId::from(func_name.as_str());

                    // Check if this is a user-defined function
                    if let Some(user_fn) = self.functions.get(&fn_id) {
                        // Inline the function by wrapping body in let bindings for each param
                        let lowered_args: Vec<_> = args
                            .iter()
                            .map(|a| self.lower_expr_with_context(&a.value.node, ctx))
                            .collect();

                        // Build nested let expressions: let param1 = arg1 in let param2 = arg2 in body
                        let mut result = user_fn.body.clone();
                        for (param, arg) in
                            user_fn.params.iter().rev().zip(lowered_args.iter().rev())
                        {
                            result = CompiledExpr::Let {
                                name: param.clone(),
                                value: Box::new(arg.clone()),
                                body: Box::new(result),
                            };
                        }
                        result
                    } else {
                        // Unknown function - leave as a call (will generate warning during validation)
                        CompiledExpr::Call {
                            function: func_name,
                            args: args
                                .iter()
                                .map(|a| self.lower_expr_with_context(&a.value.node, ctx))
                                .collect(),
                        }
                    }
                }
            }
            Expr::MethodCall {
                object,
                method,
                args,
            } => {
                // Method calls are lowered to function calls with object as first argument
                // e.g., obj.method(a, b) -> method(obj, a, b)
                let lowered_obj = self.lower_expr_with_context(&object.node, ctx);
                let lowered_args: Vec<_> = std::iter::once(lowered_obj)
                    .chain(
                        args.iter()
                            .map(|a| self.lower_expr_with_context(&a.value.node, ctx)),
                    )
                    .collect();

                CompiledExpr::Call {
                    function: method.clone(),
                    args: lowered_args,
                }
            }
            Expr::FieldAccess { object, field } => CompiledExpr::FieldAccess {
                object: Box::new(self.lower_expr_with_context(&object.node, ctx)),
                field: field.clone(),
            },
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => CompiledExpr::If {
                condition: Box::new(self.lower_expr_with_context(&condition.node, ctx)),
                then_branch: Box::new(self.lower_expr_with_context(&then_branch.node, ctx)),
                else_branch: Box::new(
                    else_branch
                        .as_ref()
                        .map(|e| self.lower_expr_with_context(&e.node, ctx))
                        .unwrap_or(CompiledExpr::Literal(0.0)),
                ),
            },
            Expr::Let { name, value, body } => {
                // Lower the value without the new local
                let lowered_value = self.lower_expr_with_context(&value.node, ctx);
                // Add the new local for the body
                let body_ctx = ctx.with_local(name.clone());
                let lowered_body = self.lower_expr_with_context(&body.node, &body_ctx);
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
                    ast::MathConst::I => {
                        panic!(
                            "MathConst::I (imaginary unit) cannot be represented as a real number"
                        )
                    }
                };
                CompiledExpr::Literal(val)
            }
            // Block, For, Map, Fold, Struct, EmitSignal, EmitField, FieldRef, Payload, PayloadField
            // These require more complex lowering or are handled specially
            Expr::Block(exprs) => {
                if exprs.is_empty() {
                    panic!(
                        "Empty block expression has no value - blocks must contain at least one expression"
                    )
                } else {
                    // For now, just evaluate to the last expression
                    self.lower_expr_with_context(&exprs.last().unwrap().node, ctx)
                }
            }

            // === Entity expressions ===
            Expr::SelfField(field) => CompiledExpr::SelfField(field.clone()),

            Expr::EntityRef(path) => {
                panic!(
                    "EntityRef '{}' cannot be evaluated to a value - entity references must be used in aggregation context (e.g., sum(entity.{}, ...))",
                    path, path
                )
            }

            Expr::EntityAccess { entity, instance } => {
                // instance expression must be a literal for the instance ID
                // Supports: string ("primary"), integer (0), or float that's an integer (0.0)
                let inst_id = match &instance.node {
                    Expr::Literal(Literal::String(s)) => InstanceId::from(s.as_str()),
                    Expr::Literal(Literal::Integer(n)) => InstanceId::from(n.to_string().as_str()),
                    Expr::Literal(Literal::Float(n)) => {
                        // Convert float to integer string if it's a whole number
                        let int_val = *n as i64;
                        InstanceId::from(int_val.to_string().as_str())
                    }
                    other => panic!(
                        "EntityAccess instance must be a literal (string, integer, or float), got {:?} - dynamic instance lookups are not supported",
                        other
                    ),
                };
                CompiledExpr::EntityAccess {
                    entity: EntityId::from(entity.clone()),
                    instance: inst_id,
                    field: String::new(), // Field access happens via FieldAccess wrapping this
                }
            }

            Expr::Aggregate { op, entity, body } => CompiledExpr::Aggregate {
                op: self.lower_aggregate_op(*op),
                entity: EntityId::from(entity.clone()),
                body: Box::new(self.lower_expr_with_context(&body.node, ctx)),
            },

            Expr::Other(path) => {
                // other() should only be used within aggregation context
                CompiledExpr::Other {
                    entity: EntityId::from(path.clone()),
                    body: Box::new(CompiledExpr::Literal(1.0)), // placeholder
                }
            }

            Expr::Pairs(path) => CompiledExpr::Pairs {
                entity: EntityId::from(path.clone()),
                body: Box::new(CompiledExpr::Literal(1.0)), // placeholder
            },

            Expr::Filter { entity, predicate } => CompiledExpr::Filter {
                entity: EntityId::from(entity.clone()),
                predicate: Box::new(self.lower_expr_with_context(&predicate.node, ctx)),
                body: Box::new(CompiledExpr::Literal(1.0)), // placeholder for nested body
            },

            Expr::First { entity, predicate } => CompiledExpr::First {
                entity: EntityId::from(entity.clone()),
                predicate: Box::new(self.lower_expr_with_context(&predicate.node, ctx)),
            },

            Expr::Nearest { entity, position } => CompiledExpr::Nearest {
                entity: EntityId::from(entity.clone()),
                position: Box::new(self.lower_expr_with_context(&position.node, ctx)),
            },

            Expr::Within {
                entity,
                position,
                radius,
            } => CompiledExpr::Within {
                entity: EntityId::from(entity.clone()),
                position: Box::new(self.lower_expr_with_context(&position.node, ctx)),
                radius: Box::new(self.lower_expr_with_context(&radius.node, ctx)),
                body: Box::new(CompiledExpr::Literal(1.0)), // placeholder
            },

            // === Impulse expressions ===
            Expr::Payload => CompiledExpr::Payload,
            Expr::PayloadField(field) => CompiledExpr::PayloadField(field.clone()),
            Expr::EmitSignal { target, value } => CompiledExpr::EmitSignal {
                target: SignalId::from(target.clone()),
                value: Box::new(self.lower_expr_with_context(&value.node, ctx)),
            },

            // Remaining expressions that need more complex handling
            other => panic!(
                "Unhandled expression type in lowering: {:?} - this expression cannot be compiled",
                other
            ),
        }
    }

    pub(crate) fn expr_to_function_name(&self, expr: &Expr) -> String {
        match expr {
            Expr::Path(path) => path.to_string(),
            Expr::FieldAccess { object, field } => {
                format!("{}.{}", self.expr_to_function_name(&object.node), field)
            }
            other => panic!(
                "Cannot extract function name from expression: {:?} - expected Path or FieldAccess",
                other
            ),
        }
    }

    /// Parse a function name into a dt-robust operator if it matches.
    ///
    /// dt-robust operators are special functions that provide numerically stable
    /// time integration. They are recognized by name and converted to a dedicated
    /// IR variant for special code generation.
    pub(crate) fn parse_dt_robust_operator(&self, name: &str) -> Option<DtRobustOperator> {
        match name {
            "integrate" => Some(DtRobustOperator::Integrate),
            "decay" => Some(DtRobustOperator::Decay),
            "relax" => Some(DtRobustOperator::Relax),
            "accumulate" => Some(DtRobustOperator::Accumulate),
            "advance_phase" => Some(DtRobustOperator::AdvancePhase),
            "smooth" => Some(DtRobustOperator::Smooth),
            "damp" => Some(DtRobustOperator::Damp),
            _ => None,
        }
    }
}
