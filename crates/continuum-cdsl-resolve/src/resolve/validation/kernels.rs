//! Kernel call validation.
//!
//! Validates kernel calls against kernel signatures, checking argument counts,
//! types, shapes, and units.

use super::constraints::{validate_shape_constraint, validate_unit_constraint};
use super::types::validate_expr;
use super::ValidationContext;
use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::Span;
use continuum_cdsl_ast::{KernelId, TypedExpr};

pub(super) fn validate_kernel_call(
    kernel: &KernelId,
    args: &[TypedExpr],
    span: Span,
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Recursively validate all arguments
    for arg in args {
        errors.extend(validate_expr(arg, ctx));
    }

    // Look up kernel signature
    let Some(signature) = ctx.kernel_registry.get(kernel) else {
        errors.push(CompileError::new(
            ErrorKind::UnknownKernel,
            span,
            format!("Unknown kernel: {}", kernel.qualified_name()),
        ));
        return errors;
    };

    // Validate argument count
    if args.len() != signature.params.len() {
        errors.push(CompileError::new(
            ErrorKind::WrongArgCount,
            span,
            format!(
                "kernel {} expects {} arguments, got {}",
                kernel.qualified_name(),
                signature.params.len(),
                args.len()
            ),
        ));
        return errors;
    }

    // Validate each argument against its parameter constraint
    for (i, (arg, param)) in args.iter().zip(&signature.params).enumerate() {
        // Validate shape constraint
        errors.extend(validate_shape_constraint(
            &arg.ty,
            &param.shape,
            i,
            args,
            kernel,
            arg.span,
        ));

        // Validate unit constraint
        errors.extend(validate_unit_constraint(
            &arg.ty,
            &param.unit,
            i,
            args,
            kernel,
            arg.span,
        ));
    }

    errors
}
