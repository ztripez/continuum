//! Seq type escape validation.
//!
//! Validates that `Type::Seq` never escapes an aggregate boundary.
//! Seq types are intermediate results and must be consumed by aggregates.

use crate::error::{CompileError, ErrorKind};

/// Validates that `Type::Seq` never escapes an aggregate boundary.
///
/// Seq types are intermediate results of `map` operations and must be
/// consumed by an aggregate (sum, max, etc.) or fold. They cannot be
/// stored in signals, fields, constants, or configurations.
pub fn validate_seq_escape<I: continuum_cdsl_ast::Index>(
    nodes: &[continuum_cdsl_ast::Node<I>],
) -> Vec<CompileError> {
    let mut errors = Vec::new();

    for node in nodes {
        if let Some(ty) = &node.output {
            if ty.is_seq() {
                errors.push(CompileError::new(
                    ErrorKind::TypeMismatch,
                    node.span,
                    format!(
                        "node '{}' has forbidden Seq type. Seq types must be consumed by an aggregate.",
                        node.path
                    ),
                ));
            }
        }
    }

    errors
}
