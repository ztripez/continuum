//! Phase boundary validation for nodes.
//!
//! Validates that nodes respect phase restrictions (e.g., assertions only
//! in Resolve/Fracture phases, not in Measure/observer phases).

use super::types::validate_expr;
use super::ValidationContext;
use crate::error::{CompileError, ErrorKind};
use crate::resolve::types::TypeTable;
use continuum_cdsl_ast::foundation::{Phase, Type};
use continuum_cdsl_ast::{BlockBody, KernelRegistry, Stmt, TypedStmt};

///
/// This includes execution blocks, warmup logic, fracture conditions,
/// and chronicle observers.
pub fn validate_node<I: continuum_cdsl_ast::Index>(
    node: &continuum_cdsl_ast::Node<I>,
    type_table: &TypeTable,
    registry: &KernelRegistry,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();
    let ctx = ValidationContext::new(type_table, registry);

    // 1. Execution blocks
    for execution in &node.executions {
        match &execution.body {
            continuum_cdsl_ast::ExecutionBody::Expr(expr) => {
                errors.extend(validate_expr(expr, &ctx))
            }
            continuum_cdsl_ast::ExecutionBody::Statements(stmts) => {
                for stmt in stmts {
                    match stmt {
                        continuum_cdsl_ast::TypedStmt::Let { value, .. } => {
                            errors.extend(validate_expr(value, &ctx))
                        }
                        continuum_cdsl_ast::TypedStmt::SignalAssign { value, .. } => {
                            if value.ty.is_seq() {
                                errors.push(CompileError::new(
                                    ErrorKind::TypeMismatch,
                                    value.span,
                                    "Seq types cannot be assigned to signals".to_string(),
                                ));
                            }
                            errors.extend(validate_expr(value, &ctx))
                        }
                        continuum_cdsl_ast::TypedStmt::FieldAssign {
                            position, value, ..
                        } => {
                            if value.ty.is_seq() {
                                errors.push(CompileError::new(
                                    ErrorKind::TypeMismatch,
                                    value.span,
                                    "Seq types cannot be assigned to fields".to_string(),
                                ));
                            }
                            errors.extend(validate_expr(position, &ctx));
                            errors.extend(validate_expr(value, &ctx));
                        }
                        continuum_cdsl_ast::TypedStmt::Assert {
                            condition,
                            severity,
                            span,
                            ..
                        } => {
                            // Verify assertions only in causal phases (observer boundary)
                            if !matches!(execution.phase, Phase::Resolve | Phase::Fracture) {
                                errors.push(CompileError::new(
                                    ErrorKind::PhaseBoundaryViolation,
                                    *span,
                                    format!(
                                        "Assertions only valid in Resolve/Fracture phases (current: {:?})",
                                        execution.phase
                                    ),
                                ));
                            }

                            errors.extend(validate_expr(condition, &ctx));

                            // Enforce Bool type for assertion condition (fail loudly)
                            if condition.ty != Type::Bool {
                                errors.push(CompileError::new(
                                    ErrorKind::TypeMismatch,
                                    *span,
                                    format!(
                                        "assert condition must be Bool, got {:?}",
                                        condition.ty
                                    ),
                                ));
                            }

                            // Validate severity level if present
                            if let Some(sev) = severity {
                                if !matches!(sev.as_str(), "warn" | "error" | "fatal") {
                                    errors.push(CompileError::new(
                                        ErrorKind::TypeMismatch,
                                        *span,
                                        format!(
                                            "Invalid severity '{}'. Must be 'warn', 'error', or 'fatal'",
                                            sev
                                        ),
                                    ));
                                }
                            }
                        }
                        continuum_cdsl_ast::TypedStmt::Expr(expr) => {
                            errors.extend(validate_expr(expr, &ctx))
                        }
                    }
                }
            }
        }
    }

    // 2. Assertions
    for assertion in &node.assertions {
        errors.extend(validate_expr(&assertion.condition, &ctx));
    }

    // Note: Warmup, When, and Observe blocks currently contain UNTYPED Expr.
    // Full semantic validation for these is deferred until they are typed
    // or handled via untyped validation helpers.

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
