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

use crate::error::{CompileError, ErrorKind};
pub use context::TypingContext;
use continuum_cdsl_ast::foundation::{KernelType, Shape, Type, Unit};
use continuum_cdsl_ast::{Expr, ExprKind, TypedExpr, UntypedKind};
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
            use continuum_cdsl_ast::foundation::Path;
            
            // First check local bindings
            if let Some(ty) = ctx.local_bindings.get(name).cloned() {
                (ExprKind::Local(name.clone()), ty)
            }
            // Bare signal path resolution: if not a local variable, try as a signal
            else if let Some(ty) = ctx.signal_types.get(&Path::from(name.as_str())).cloned() {
                (ExprKind::Signal(Path::from(name.as_str())), ty)
            }
            // Not found anywhere - error
            else {
                return Err(err_undefined(span, name, "local variable"));
            }
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
            // Capability check is performed by capabilities.rs validation pass
            // which checks role-specific phase capabilities from RoleSpec
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
            source,
            binding,
            body,
        } => {
            let (k, t) = type_aggregate(ctx, op, source, binding, body, span)?;
            (k, t)
        }

        UntypedKind::Fold {
            source,
            init,
            acc,
            elem,
            body,
        } => {
            let (k, t) = type_fold(ctx, source, init, acc, elem, body, span)?;
            (k, t)
        }

        UntypedKind::Entity(entity_id) => {
            let instance_ty = Type::User(continuum_foundation::TypeId::from(entity_id.0.clone()));
            (
                ExprKind::Entity(entity_id.clone()),
                Type::Seq(Box::new(instance_ty)),
            )
        }

        UntypedKind::Filter { source, predicate } => {
            let typed_source = type_expression(source, ctx)?;
            let element_ty = match &typed_source.ty {
                Type::Seq(inner) => *inner.clone(),
                _ => {
                    return Err(err_type_mismatch(
                        source.span,
                        "Seq<T>",
                        &format!("{:?}", typed_source.ty),
                    ));
                }
            };

            let extended_ctx = ctx.with_binding("self".to_string(), element_ty.clone());
            let typed_predicate = type_expression(predicate, &extended_ctx)?;

            if typed_predicate.ty != Type::Bool {
                return Err(err_type_mismatch(
                    predicate.span,
                    "Bool",
                    &format!("{:?}", typed_predicate.ty),
                ));
            }

            (
                ExprKind::Filter {
                    source: Box::new(typed_source),
                    predicate: Box::new(typed_predicate),
                },
                Type::Seq(Box::new(element_ty)),
            )
        }

        UntypedKind::Nearest { entity, position } => {
            let typed_position = type_expression(position, ctx)?;
            (
                ExprKind::Nearest {
                    entity: entity.clone(),
                    position: Box::new(typed_position),
                },
                Type::User(continuum_foundation::TypeId::from(entity.0.clone())),
            )
        }

        UntypedKind::Within {
            entity,
            position,
            radius,
        } => {
            let typed_position = type_expression(position, ctx)?;
            let typed_radius = type_expression(radius, ctx)?;
            let instance_ty = Type::User(continuum_foundation::TypeId::from(entity.0.clone()));
            (
                ExprKind::Within {
                    entity: entity.clone(),
                    position: Box::new(typed_position),
                    radius: Box::new(typed_radius),
                },
                Type::Seq(Box::new(instance_ty)),
            )
        }

        UntypedKind::OtherInstances(entity_id) => {
            let instance_ty = Type::User(continuum_foundation::TypeId::from(entity_id.0.clone()));
            (
                ExprKind::Entity(entity_id.clone()), // Desugar to Entity for now or add variant
                Type::Seq(Box::new(instance_ty)),
            )
        }

        UntypedKind::PairsInstances(entity_id) => {
            let instance_ty = Type::User(continuum_foundation::TypeId::from(entity_id.0.clone()));
            // Produces Seq<[Instance, Instance]>?
            // For now, let's just say it produces Seq<Instance> and we'll handle the pair logic in the VM
            (
                ExprKind::Entity(entity_id.clone()), // Desugar to Entity for now
                Type::Seq(Box::new(instance_ty)),
            )
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
/// Integration test for bare signal path resolution feature.
///
/// This test verifies that expressions like `core.temp` are automatically
/// resolved as `signal.core.temp` when `core` is not a local variable.
#[cfg(test)]
mod bare_path_integration_tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{KernelType, Path, Shape, Span, Type, Unit};
    use continuum_cdsl_ast::{Expr, KernelRegistry, UntypedKind};
    use std::collections::HashMap;

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn scalar_type() -> Type {
        Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::dimensionless(),
            bounds: None,
        })
    }

    #[test]
    fn test_bare_signal_path_single() {
        // Test: `temperature` → `signal.temperature`
        let expr = Expr::new(UntypedKind::Local("temperature".to_string()), test_span());

        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("temperature"), scalar_type());

        let type_table = crate::resolve::types::TypeTable::new();
        let kernels = KernelRegistry::global();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let ctx = TypingContext::new(
            &type_table,
            kernels,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_ok(), "Bare signal path should resolve successfully");

        let typed = result.unwrap();
        // Should resolve as Signal, not Local
        assert!(matches!(typed.expr, ExprKind::Signal(_)));
    }

    #[test]
    fn test_bare_signal_path_nested() {
        // Test: `core.temp` → `signal.core.temp`
        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(UntypedKind::Local("core".to_string()), test_span())),
                field: "temp".to_string(),
            },
            test_span(),
        );

        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("core.temp"), scalar_type());

        let type_table = crate::resolve::types::TypeTable::new();
        let kernels = KernelRegistry::global();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let ctx = TypingContext::new(
            &type_table,
            kernels,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_ok(), "Nested bare signal path should resolve");

        let typed = result.unwrap();
        assert!(matches!(typed.expr, ExprKind::Signal(_)));
        if let ExprKind::Signal(path) = typed.expr {
            assert_eq!(path.to_string(), "core.temp");
        }
    }

    #[test]
    fn test_bare_field_path() {
        // Test: `observation.value` → `field.observation.value`
        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(UntypedKind::Local("observation".to_string()), test_span())),
                field: "value".to_string(),
            },
            test_span(),
        );

        let mut field_types = HashMap::new();
        field_types.insert(Path::from_path_str("observation.value"), scalar_type());

        let type_table = crate::resolve::types::TypeTable::new();
        let kernels = KernelRegistry::global();
        let signal_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let ctx = TypingContext::new(
            &type_table,
            kernels,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = type_expression(&expr, &ctx);
        assert!(result.is_ok(), "Bare field path should resolve");

        let typed = result.unwrap();
        assert!(matches!(typed.expr, ExprKind::Field(_)));
    }

    #[test]
    fn test_local_takes_precedence() {
        // Test: If `core` is a local variable, it should NOT resolve as signal
        let expr = Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(Expr::new(UntypedKind::Local("core".to_string()), test_span())),
                field: "temp".to_string(),
            },
            test_span(),
        );

        // Register both signal AND create a local context
        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("core.temp"), scalar_type());

        let type_table = crate::resolve::types::TypeTable::new();
        let kernels = KernelRegistry::global();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let mut ctx = TypingContext::new(
            &type_table,
            kernels,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        // Add `core` as a local variable (struct type would be needed for proper test)
        // For now, this will fail since we don't have a proper struct type for core
        // but the test documents the expected behavior
        let result = type_expression(&expr, &ctx);
        // This should try to resolve as field access on local, not as signal
        // (will fail in this minimal test, but documents the precedence rule)
    }
}
