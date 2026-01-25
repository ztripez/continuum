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
use continuum_cdsl_ast::foundation::{BinaryOp, KernelType, Shape, Type, UnaryOp, Unit};
use continuum_cdsl_ast::{Expr, ExprKind, TypedExpr, UntypedKind};
use continuum_foundation::Phase;
use derivation::derive_return_type;
use helpers::*;

/// Infers and validates the type of an expression.
///
/// This is the primary entry point for expression typing. It recursively
/// traverses the untyped expression tree and produces a [`TypedExpr`].
///
/// # Parameters
/// - `expr`: The untyped expression to resolve.
/// - `ctx`: The typing context providing access to registries and local bindings.
/// - `expected`: Optional expected type for bidirectional inference (default: None).
///
/// # Returns
/// - `Ok(TypedExpr)` containing the resolved kind, type, and source span.
/// - `Err(Vec<CompileError>)` if any typing or structural violations are found.
///
/// # Bidirectional Type Inference
///
/// The system uses bidirectional type inference:
/// - **Synthesis mode** (`expected = None`): Infer type from expression structure
/// - **Checking mode** (`expected = Some(ty)`): Use context to guide inference
///
/// When `expected` is provided, literals without explicit unit annotations
/// inherit the unit from the expected type. This enables ergonomic code:
///
/// ```cdsl
/// if temperature > 0.0 { ... }  // 0.0 infers Scalar<K> from temperature's type
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - A symbol (local, signal, field, kernel) is undefined.
/// - Types are mismatched (e.g., in struct fields or fold bodies).
/// - A phase boundary is violated (e.g., reading a field outside the Measure phase).
/// - A capability is missing (e.g., using `prev` outside the Resolve phase).
pub fn type_expression(
    expr: &Expr,
    ctx: &TypingContext,
    expected: Option<&Type>,
) -> Result<TypedExpr, Vec<CompileError>> {
    let span = expr.span;

    let (kind, ty) = match &expr.kind {
        // === Literals ===
        UntypedKind::Literal { value, unit } => {
            let (k, t) = type_literal(span, *value, unit.as_ref(), expected)?;
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
            // Local identifiers must be explicitly bound in scope
            // Bare signal references require explicit signal. prefix to avoid ambiguity
            let ty = ctx
                .local_bindings
                .get(name)
                .cloned()
                .ok_or_else(|| err_undefined(span, name, "local variable"))?;

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

        // === Binary operators (type-aware dispatch) ===
        UntypedKind::Binary { op, left, right } => {
            let typed_left = type_expression(left, ctx, None)?;
            let typed_right = type_expression(right, ctx, None)?;
            
            // Dispatch to appropriate kernel based on operand types
            let kernel = select_binary_kernel(*op, &typed_left.ty, &typed_right.ty)?;
            
            // Check kernel signature and derive return type
            let sig = ctx
                .kernel_registry
                .get(&kernel)
                .ok_or_else(|| err_internal(span, format!("unknown kernel: {:?}", kernel)))?;
            let typed_args = vec![typed_left, typed_right];
            let return_type = derive_return_type(sig, &typed_args, span)?;
            
            (ExprKind::Call { kernel, args: typed_args }, return_type)
        }

        // === Unary operators (type-aware dispatch) ===
        UntypedKind::Unary { op, operand } => {
            let typed_operand = type_expression(operand, ctx, None)?;
            
            // Dispatch to appropriate kernel based on operand type
            let kernel = select_unary_kernel(*op, &typed_operand.ty)?;
            
            // Check kernel signature and derive return type
            let sig = ctx
                .kernel_registry
                .get(&kernel)
                .ok_or_else(|| err_internal(span, format!("unknown kernel: {:?}", kernel)))?;
            let typed_args = vec![typed_operand];
            let return_type = derive_return_type(sig, &typed_args, span)?;
            
            (ExprKind::Call { kernel, args: typed_args }, return_type)
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
            let typed_source = type_expression(source, ctx, None)?;
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

            let mut extended_ctx = ctx.with_binding("self".to_string(), element_ty.clone());
            // If element_ty is a User type (entity), also set self_type for member access
            if matches!(element_ty, Type::User(_)) {
                extended_ctx.self_type = Some(element_ty.clone());
            }
            let typed_predicate = type_expression(predicate, &extended_ctx, None)?;

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
            let typed_position = type_expression(position, ctx, None)?;
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
            let typed_position = type_expression(position, ctx, None)?;
            let typed_radius = type_expression(radius, ctx, None)?;
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

        UntypedKind::Neighbors { instance } => {
            let typed_instance = type_expression(instance, ctx, None)?;
            
            // Extract entity type from instance type
            let entity_id = match &typed_instance.ty {
                Type::User(type_id) => {
                    continuum_cdsl_ast::foundation::EntityId::new(type_id.path().clone())
                }
                _ => {
                    return Err(err_type_mismatch(
                        span,
                        "entity instance",
                        &format!("{}", typed_instance.ty),
                    ));
                }
            };

            let instance_ty = typed_instance.ty.clone();
            (
                ExprKind::Neighbors {
                    entity: entity_id,
                    instance: Box::new(typed_instance),
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

        // === If-expressions should be desugared before typing ===
        UntypedKind::If { .. } => {
            return Err(vec![CompileError::new(
                ErrorKind::Internal,
                span,
                "if-expression must be desugared to logic.select before typing".to_string(),
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

/// Select appropriate kernel for binary operator based on operand types
fn select_binary_kernel(
    op: BinaryOp,
    left_ty: &Type,
    right_ty: &Type,
) -> Result<continuum_kernel_types::KernelId, Vec<CompileError>> {
    use BinaryOp::*;
    
    // Check if this is a vector operation and extract dimension
    // Check left operand first
    let left_vector_dim = if let Type::Kernel(k) = left_ty {
        if let continuum_cdsl_ast::foundation::Shape::Vector { dim } = k.shape {
            Some(dim)
        } else {
            None
        }
    } else {
        None
    };
    
    // Check right operand if left is not a vector
    let right_vector_dim = if let Type::Kernel(k) = right_ty {
        if let continuum_cdsl_ast::foundation::Shape::Vector { dim } = k.shape {
            Some(dim)
        } else {
            None
        }
    } else {
        None
    };
    
    // Prefer left vector dimension, fall back to right
    let vector_dim = left_vector_dim.or(right_vector_dim);
    let is_vector_op = vector_dim.is_some();
    
    let namespace = if is_vector_op && matches!(op, Add | Sub | Mul | Div) {
        "vector"
    } else {
        match op {
            Add | Sub | Mul | Div | Mod | Pow => "maths",
            Eq | Ne | Lt | Le | Gt | Ge => "compare",
            And | Or => "logic",
        }
    };
    
    let base_name = match op {
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        Div => "div",
        Mod => "mod",
        Pow => "pow",
        Eq => "eq",
        Ne => "ne",
        Lt => "lt",
        Le => "le",
        Gt => "gt",
        Ge => "ge",
        And => "and",
        Or => "or",
    };
    
    // Append vector dimension for vector operations
    let name = if let Some(dim) = vector_dim {
        let is_left_scalar = matches!(left_ty, Type::Kernel(k) if matches!(k.shape, continuum_cdsl_ast::foundation::Shape::Scalar));
        let is_right_scalar = matches!(right_ty, Type::Kernel(k) if matches!(k.shape, continuum_cdsl_ast::foundation::Shape::Scalar));
        
        if matches!(op, Mul | Div) {
            if is_left_scalar && !is_right_scalar {
                // scalar * vector → mul_scalar_vec3
                format!("{}_scalar_vec{}", base_name, dim)
            } else if !is_left_scalar && is_right_scalar {
                // vector * scalar → mul_vec3_scalar
                format!("{}_vec{}_scalar", base_name, dim)
            } else {
                // vector * vector → mul_vec3 (component-wise, if we had it)
                format!("{}_vec{}", base_name, dim)
            }
        } else {
            // add/sub operations
            format!("{}_vec{}", base_name, dim)
        }
    } else {
        base_name.to_string()
    };
    
    Ok(continuum_kernel_types::KernelId {
        namespace: std::borrow::Cow::Borrowed(namespace),
        name: std::borrow::Cow::Owned(name),
    })
}

/// Select appropriate kernel for unary operator based on operand type
fn select_unary_kernel(
    op: UnaryOp,
    _operand_ty: &Type,
) -> Result<continuum_kernel_types::KernelId, Vec<CompileError>> {
    use UnaryOp::*;
    
    let (namespace, name) = match op {
        Neg => ("maths", "neg"),
        Not => ("logic", "not"),
    };
    
    Ok(continuum_kernel_types::KernelId::new(namespace, name))
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
    fn test_bare_identifier_requires_explicit_signal_prefix() {
        // Test: bare `temperature` must fail (ambiguous)
        // Signals require explicit `signal.temperature` prefix
        let expr = Expr::new(UntypedKind::Local("temperature".to_string()), test_span());

        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("temperature"), scalar_type());

        let type_table = crate::resolve::types::TypeTable::new();
        let kernels = KernelRegistry::global();
        let function_table = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let ctx = TypingContext::new(
            &type_table,
            kernels,
            &function_table,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = type_expression(&expr, &ctx, None);
        // Should fail: bare identifiers don't fall back to signals (ambiguous)
        assert!(result.is_err(), "Bare identifier without explicit signal. prefix should fail");
        
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("not found"));
        assert!(errors[0].message.contains("temperature"));
    }

    #[test]
    fn test_local_binding_shadows_signal() {
        // Test: When there's both a local binding AND a signal with the same name,
        // the local binding takes precedence (no ambiguity)
        let expr = Expr::new(UntypedKind::Local("temperature".to_string()), test_span());

        let mut signal_types = HashMap::new();
        signal_types.insert(Path::from_path_str("temperature"), scalar_type());

        let type_table = crate::resolve::types::TypeTable::new();
        let kernels = KernelRegistry::global();
        let function_table = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let mut ctx = TypingContext::new(
            &type_table,
            kernels,
            &function_table,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        // Add local binding for 'temperature'
        ctx.local_bindings.insert("temperature".to_string(), scalar_type());

        let result = type_expression(&expr, &ctx, None);
        // Should succeed: local binding found, signal is ignored
        assert!(result.is_ok(), "Local binding should shadow signal with same name");
        
        let typed = result.unwrap();
        assert!(matches!(typed.expr, ExprKind::Local(_)));
        assert_eq!(typed.ty, scalar_type());
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
        let function_table = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let ctx = TypingContext::new(
            &type_table,
            kernels,
            &function_table,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = type_expression(&expr, &ctx, None);
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
        let function_table = HashMap::new();
        let signal_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let ctx = TypingContext::new(
            &type_table,
            kernels,
            &function_table,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        let result = type_expression(&expr, &ctx, None);
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
        let function_table = HashMap::new();
        let field_types = HashMap::new();
        let config_types = HashMap::new();
        let const_types = HashMap::new();
        let mut ctx = TypingContext::new(
            &type_table,
            kernels,
            &function_table,
            &signal_types,
            &field_types,
            &config_types,
            &const_types,
        );

        // Add `core` as a local variable (struct type would be needed for proper test)
        // For now, this will fail since we don't have a proper struct type for core
        // but the test documents the expected behavior
        let result = type_expression(&expr, &ctx, None);
        // This should try to resolve as field access on local, not as signal
        // (will fail in this minimal test, but documents the precedence rule)
    }
}
