//! Expression and literal validation.
//!
//! Main expression validator that dispatches to specialized validators
//! for kernels, structs, field access, etc. Also validates literal bounds.

use super::constraints::{validate_shape_constraint, validate_unit_constraint};
use super::kernels::validate_kernel_call;
use super::structs::{validate_field_access, validate_struct_fields};
use super::ValidationContext;
use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{Span, Type};
use continuum_cdsl_ast::{ExprKind, KernelId, TypedExpr};

pub fn validate_expr(expr: &TypedExpr, ctx: &ValidationContext<'_>) -> Vec<CompileError> {
    let mut errors = Vec::new();

    match &expr.expr {
        ExprKind::Literal { value, .. } => {
            // Validate literal value against type bounds
            errors.extend(validate_literal_bounds(*value, &expr.ty, expr.span));
        }

        ExprKind::Call { kernel, args } => {
            // Validate kernel call
            errors.extend(validate_kernel_call(kernel, args, expr.span, ctx));
        }

        ExprKind::Vector(elements) => {
            // Validate all elements recursively
            for elem in elements {
                errors.extend(validate_expr(elem, ctx));

                // Seq types cannot appear in vector literals
                if elem.ty.is_seq() {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        elem.span,
                        "Seq types cannot be stored in vectors (must be immediately consumed by aggregate/fold)".to_string(),
                    ));
                }
            }

            // Validate all elements have same type
            if !elements.is_empty() {
                let first_ty = &elements[0].ty;
                for (i, elem) in elements.iter().enumerate().skip(1) {
                    if &elem.ty != first_ty {
                        errors.push(CompileError::new(
                            ErrorKind::TypeMismatch,
                            elem.span,
                            format!(
                                "vector element {} has type {:?}, expected {:?} (from element 0)",
                                i, elem.ty, first_ty
                            ),
                        ));
                    }
                }
            }
        }

        ExprKind::Let { value, body, .. } => {
            errors.extend(validate_expr(value, ctx));
            errors.extend(validate_expr(body, ctx));

            // Seq types cannot be stored in let bindings
            if value.ty.is_seq() {
                errors.push(CompileError::new(
                    ErrorKind::TypeMismatch,
                    value.span,
                    "Seq types cannot be stored in let bindings (must be immediately consumed by aggregate/fold)".to_string(),
                ));
            }
        }

        ExprKind::Aggregate { body, .. } => {
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Fold { init, body, .. } => {
            errors.extend(validate_expr(init, ctx));
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Struct {
            ty: type_id,
            fields,
        } => {
            // Validate all field expressions recursively
            for (name, field_expr) in fields {
                errors.extend(validate_expr(field_expr, ctx));

                // Seq types cannot appear in struct fields
                if field_expr.ty.is_seq() {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        field_expr.span,
                        format!(
                            "Seq types cannot be stored in struct field '{}' (must be immediately consumed by aggregate/fold)",
                            name
                        ),
                    ));
                }
            }

            // Validate fields against user type definition
            errors.extend(validate_struct_fields(type_id, fields, expr.span, ctx));
        }

        ExprKind::FieldAccess { object, field } => {
            errors.extend(validate_expr(object, ctx));

            // Validate field exists on object type (for user types)
            if let Type::User(type_id) = &object.ty {
                errors.extend(validate_field_access(type_id, field, expr.span, ctx));
            }
        }

        ExprKind::Aggregate { source, body, .. } => {
            errors.extend(validate_expr(source, ctx));
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Fold {
            source, init, body, ..
        } => {
            errors.extend(validate_expr(source, ctx));
            errors.extend(validate_expr(init, ctx));
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Filter { source, predicate } => {
            errors.extend(validate_expr(source, ctx));
            errors.extend(validate_expr(predicate, ctx));
        }

        ExprKind::Nearest { position, .. } => {
            errors.extend(validate_expr(position, ctx));
        }

        ExprKind::Within {
            position, radius, ..
        } => {
            errors.extend(validate_expr(position, ctx));
            errors.extend(validate_expr(radius, ctx));
        }

        // References don't need validation
        ExprKind::Local(_)
        | ExprKind::Signal(_)
        | ExprKind::Field(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Self_
        | ExprKind::Other
        | ExprKind::Payload
        | ExprKind::Entity(_)
        | ExprKind::StringLiteral(_) => {}
    }

    errors
}

/// Validates a kernel call expression.
///
/// Checks that:
/// - Kernel exists
/// - Argument count is correct
/// - Argument types satisfy kernel constraints
/// - Argument units satisfy kernel constraints
///
/// # Parameters
///
/// - `kernel`: Kernel identifier (namespace + name).
/// - `args`: Arguments to the kernel call.
/// - `span`: Source location for error reporting.
/// - `ctx`: Validation context with kernel registry.
///
/// # Returns
///
/// Vector of validation errors (empty if call is valid).

fn validate_literal_bounds(value: f64, ty: &Type, span: Span) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Only kernel types have bounds
    if let Type::Kernel(kernel_ty) = ty {
        if let Some(bounds) = &kernel_ty.bounds {
            // Check minimum bound
            if let Some(min) = bounds.min {
                if value < min {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        span,
                        format!("literal value {} is below minimum bound {}", value, min),
                    ));
                }
            }

            // Check maximum bound
            if let Some(max) = bounds.max {
                if value > max {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        span,
                        format!("literal value {} exceeds maximum bound {}", value, max),
                    ));
                }
            }
        }
    }

    errors
}
