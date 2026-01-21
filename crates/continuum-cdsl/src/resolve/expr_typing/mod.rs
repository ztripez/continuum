//! Expression type resolution and inference.
//!
//! This module implements the type resolution pass for Continuum DSL expressions.
//! It transforms untyped [`Expr`] trees from the parser into [`TypedExpr`] trees
//! by inferring types, resolving units, and validating structural constraints.
//!
//! # Type Inference
//!
//! The compiler uses a bidirectional type inference system:
//! - **Synthesis**: Expressions like literals with units or kernel calls synthesize
//!   their own types based on their content or signature.
//! - **Checking**: Expressions like literals without units or struct fields infer
//!   their types from the surrounding context.
//!
//! # Phase Boundaries
//!
//! Type resolution enforces execution phase boundaries (e.g., fields can only
//! be read in the `Measure` phase).

pub mod context;
pub mod derivation;
pub mod helpers;

#[cfg(test)]
mod tests;

pub use context::TypingContext;
use crate::ast::{Expr, ExprKind, TypedExpr, UntypedKind};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{KernelType, Shape, Type, Unit};
use continuum_foundation::Phase;
use helpers::*;

/// Infers and validates the type of an expression.
///
/// This is the primary entry point for expression typing. It recursively
/// traverses the untyped expression tree and produces a [`TypedExpr`].
///
/// # Parameters
/// - `expr`: The untyped expression to resolve.
/// - `ctx`: The typing context providing access to registries and local bindings.
///
/// # Returns
/// - `Ok(TypedExpr)` containing the resolved kind, type, and source span.
/// - `Err(Vec<CompileError>)` if any typing or structural violations are found.
///
/// # Errors
///
/// Returns an error if:
/// - A symbol (local, signal, field, kernel) is undefined.
/// - Types are mismatched (e.g., in struct fields or fold bodies).
/// - A phase boundary is violated (e.g., reading a field outside the Measure phase).
/// - A capability is missing (e.g., using `prev` outside the Resolve phase).
pub fn type_expression(expr: &Expr, ctx: &TypingContext) -> Result<TypedExpr, Vec<CompileError>> {
    let span = expr.span;

    let (kind, ty) = match &expr.kind {
        // === Literals ===
        UntypedKind::Literal { value, unit } => {
            let (k, t) = type_literal(span, *value, unit.as_ref())?;
            (k, t)
        }

        UntypedKind::BoolLiteral(val) => (
            ExprKind::Literal {
                value: if *val { 1.0 } else { 0.0 },
                unit: None,
            },
            Type::Bool,
        ),

        UntypedKind::StringLiteral(val) => (ExprKind::StringLiteral(val.clone()), Type::String),

        // === References ===
        UntypedKind::Local(name) => {
            let ty = ctx.local_bindings.get(name).cloned().ok_or_else(|| {
                err_undefined(span, name, "local variable")
            })?;

            (ExprKind::Local(name.clone()), ty)
        }

        UntypedKind::Signal(path) => {
            let ty = lookup_path_type(ctx, path, span, "signal", ctx.signal_types)?;
            (ExprKind::Signal(path.clone()), ty)
        }

        UntypedKind::Field(path) => {
            // Phase boundary enforcement: Fields can only be read in Measure phase
            if let Some(phase) = ctx.phase
                && phase != Phase::Measure
            {
                return Err(vec![CompileError::new(
                    ErrorKind::PhaseBoundaryViolation,
                    span,
                    format!(
                        "field '{}' cannot be read in {:?} phase (fields are only accessible in Measure phase)",
                        path, phase
                    ),
                )]);
            }

            let ty = lookup_path_type(ctx, path, span, "field", ctx.field_types)?;
            (ExprKind::Field(path.clone()), ty)
        }

        // === Context values ===
        UntypedKind::Dt => {
            let kernel_type = KernelType {
                shape: Shape::Scalar,
                unit: Unit::seconds(),
                bounds: None,
            };
            (ExprKind::Dt, Type::Kernel(kernel_type))
        }

        // === Kernel calls ===
        UntypedKind::KernelCall { kernel, args } => {
            let (k, t) = type_as_kernel_call(ctx, kernel, args, span)?;
            (k, t)
        }

        // === Field access ===
        UntypedKind::FieldAccess { object, field } => {
            let (k, t) = type_field_access(ctx, object, field, span)?;
            (k, t)
        }

        // === Config lookup ===
        UntypedKind::Config(path) => {
            let ty = lookup_path_type(ctx, path, span, "config path", ctx.config_types)?;
            (ExprKind::Config(path.clone()), ty)
        }

        // === Const lookup ===
        UntypedKind::Const(path) => {
            let ty = lookup_path_type(ctx, path, span, "const path", ctx.const_types)?;
            (ExprKind::Const(path.clone()), ty)
        }

        // === Prev (previous tick value) ===
        UntypedKind::Prev => {
            if ctx.phase != Some(Phase::Resolve) {
                return Err(vec![CompileError::new(
                    ErrorKind::InvalidCapability,
                    span,
                    format!(
                        "'prev' may only be used in Resolve phase, found in {:?}",
                        ctx.phase
                    ),
                )]);
            }

            let ty = require_context_type(span, "prev", &ctx.node_output)?;
            (ExprKind::Prev, ty)
        }

        // === Current (just-resolved value) ===
        UntypedKind::Current => {
            let ty = require_context_type(span, "current", &ctx.node_output)?;
            (ExprKind::Current, ty)
        }

        // === Inputs (accumulated inputs) ===
        UntypedKind::Inputs => {
            let ty = require_context_type(span, "inputs", &ctx.inputs_type)?;
            (ExprKind::Inputs, ty)
        }

        // === Payload (impulse payload) ===
        UntypedKind::Payload => {
            let ty = require_context_type(span, "payload", &ctx.payload_type)?;
            (ExprKind::Payload, ty)
        }

        // === Entity context ===
        UntypedKind::Self_ => {
            let ty = require_context_type(span, "self", &ctx.self_type)?;
            (ExprKind::Self_, ty)
        }

        UntypedKind::Other => {
            let ty = require_context_type(span, "other", &ctx.other_type)?;
            (ExprKind::Other, ty)
        }

        // === Vector literal ===
        UntypedKind::Vector(elements) => {
            let (k, t) = type_vector(ctx, elements, span)?;
            (k, t)
        }

        // === Let binding ===
        UntypedKind::Let { name, value, body } => {
            let (k, t) = type_let(ctx, name, value, body, span)?;
            (k, t)
        }

        // === Struct literal ===
        UntypedKind::Struct {
            ty: ty_path,
            fields,
        } => {
            let (k, t) = type_struct(ctx, ty_path, fields, span)?;
            (k, t)
        }

        // === Aggregate operations ===
        UntypedKind::Aggregate {
            op,
            entity,
            binding,
            body,
        } => {
            let (k, t) = type_aggregate(ctx, op, entity, binding, body, span)?;
            (k, t)
        }

        UntypedKind::Fold {
            entity,
            init,
            acc,
            elem,
            body,
        } => {
            let (k, t) = type_fold(ctx, entity, init, acc, elem, body, span)?;
            (k, t)
        }

        // === Function/kernel call ===
        UntypedKind::Call { func, args } => {
            let (k, t) = type_call(ctx, func, args, span)?;
            (k, t)
        }

        // === Operator desugaring guard ===
        UntypedKind::Binary { .. } | UntypedKind::Unary { .. } | UntypedKind::If { .. } => {
            return Err(vec![CompileError::new(
                ErrorKind::Internal,
                span,
                "operator expressions must be desugared before typing".to_string(),
            )]);
        }

        UntypedKind::ParseError(_) => {
            return Err(vec![CompileError::new(
                ErrorKind::Internal,
                span,
                "expression parse error placeholder cannot be typed".to_string(),
            )]);
        }
    };

    Ok(TypedExpr::new(kind, ty, span))
}
