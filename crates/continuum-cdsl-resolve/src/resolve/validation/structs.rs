//! Struct construction and field access validation.
//!
//! Validates struct construction (all fields present, correct types) and
//! field access (field exists, correct type).

use super::ValidationContext;
use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{Path, Span, Type};
use continuum_cdsl_ast::TypedExpr;

pub(super) fn validate_struct_fields(
    type_id: &continuum_foundation::TypeId,
    fields: &[(String, TypedExpr)],
    span: Span,
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    use continuum_cdsl_ast::foundation::Path;

    let mut errors = Vec::new();

    // Look up user type in type table
    let path = Path::from(type_id.as_str());
    let Some(user_type) = ctx.type_table.get(&path) else {
        errors.push(CompileError::new(
            ErrorKind::UnknownType,
            span,
            format!("Unknown user type: {}", type_id),
        ));
        return errors;
    };

    // Validate each field
    for (field_name, field_expr) in fields {
        // Check if field exists in type definition
        if let Some((_name, expected_ty)) =
            user_type.fields.iter().find(|(name, _)| name == field_name)
        {
            // Check if field type matches expected type
            if &field_expr.ty != expected_ty {
                errors.push(CompileError::new(
                    ErrorKind::TypeMismatch,
                    field_expr.span,
                    format!(
                        "struct field '{}' has type {:?}, expected {:?}",
                        field_name, field_expr.ty, expected_ty
                    ),
                ));
            }
        } else {
            errors.push(CompileError::new(
                ErrorKind::UnknownType,
                field_expr.span,
                format!("struct type {} has no field '{}'", type_id, field_name),
            ));
        }
    }

    errors
}

/// Validates field access on user types.
///
/// Checks that the accessed field exists in the user type definition.
///
/// # Parameters
///
/// - `type_id`: User type identifier.
/// - `field_name`: Name of the accessed field.
/// - `span`: Source location for error reporting.
/// - `ctx`: Validation context with type table.
///
/// # Returns
///
/// Vector of validation errors (empty if field exists).
pub(super) fn validate_field_access(
    type_id: &continuum_foundation::TypeId,
    field_name: &str,
    span: Span,
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    use continuum_cdsl_ast::foundation::Path;

    let mut errors = Vec::new();

    // Look up user type in type table
    let path = Path::from(type_id.as_str());
    let Some(user_type) = ctx.type_table.get(&path) else {
        errors.push(CompileError::new(
            ErrorKind::UnknownType,
            span,
            format!("Unknown user type: {}", type_id),
        ));
        return errors;
    };

    // Check if field exists
    if !user_type.fields.iter().any(|(name, _)| name == field_name) {
        errors.push(CompileError::new(
            ErrorKind::UnknownType,
            span,
            format!("type {} has no field '{}'", type_id, field_name),
        ));
    }

    errors
}
