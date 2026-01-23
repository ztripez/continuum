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
                            // Verify assertions only in validation phases
                            // Assert phase is specifically for runtime validation
                            if !matches!(
                                execution.phase,
                                Phase::Resolve | Phase::Fracture | Phase::Assert
                            ) {
                                errors.push(CompileError::new(
                                    ErrorKind::PhaseBoundaryViolation,
                                    *span,
                                    format!(
                                        "Assertions only valid in Resolve/Fracture/Assert phases (current: {:?})",
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
                        continuum_cdsl_ast::TypedStmt::EmitEvent { fields, .. } => {
                            // Validate all field expressions
                            for (_, expr) in fields {
                                errors.extend(validate_expr(expr, &ctx));
                            }
                        }
                        continuum_cdsl_ast::TypedStmt::Expr(expr) => {
                            errors.extend(validate_expr(expr, &ctx))
                        }
                        continuum_cdsl_ast::TypedStmt::If {
                            condition,
                            then_branch,
                            else_branch,
                            ..
                        } => {
                            errors.extend(validate_expr(condition, &ctx));
                            // Note: branches are TypedStmt, would need recursive validation
                            // For now, just validate the condition
                            // TODO: refactor to share stmt validation logic
                            for branch_stmt in then_branch.iter().chain(else_branch.iter()) {
                                if let continuum_cdsl_ast::TypedStmt::Expr(expr)
                                | continuum_cdsl_ast::TypedStmt::Let { value: expr, .. }
                                | continuum_cdsl_ast::TypedStmt::SignalAssign {
                                    value: expr,
                                    ..
                                }
                                | continuum_cdsl_ast::TypedStmt::Assert {
                                    condition: expr,
                                    ..
                                } = branch_stmt
                                {
                                    errors.extend(validate_expr(expr, &ctx));
                                }
                            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{KernelType, Shape, Span, Unit};
    use continuum_cdsl_ast::{
        Execution, ExecutionBody, ExprKind, Node, RoleData, TypedExpr, TypedStmt,
    };
    use continuum_foundation::Path;

    fn test_span() -> Span {
        Span::new(0, 0, 10, 1)
    }

    fn assertion_scalar_type() -> Type {
        Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        })
    }

    fn assertion_bool_type() -> Type {
        Type::Bool
    }

    fn assertion_make_bool_expr(value: bool) -> TypedExpr {
        TypedExpr::new(
            ExprKind::Literal {
                value: if value { 1.0 } else { 0.0 },
                unit: None,
            },
            assertion_bool_type(),
            test_span(),
        )
    }

    fn assertion_make_scalar_expr() -> TypedExpr {
        TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            assertion_scalar_type(),
            test_span(),
        )
    }

    #[test]
    fn test_assert_in_resolve_phase_allowed() {
        // Assertions are allowed in Resolve phase
        let span = test_span();
        let path = Path::from("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: None,
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_ok(),
            "Assertions should be allowed in Resolve phase"
        );
    }

    #[test]
    fn test_assert_in_fracture_phase_allowed() {
        // Assertions are allowed in Fracture phase
        let span = test_span();
        let path = Path::from("test.operator");
        let mut node = Node::new(path, span, RoleData::Operator, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: None,
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "fracture".to_string(),
            phase: Phase::Fracture,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_ok(),
            "Assertions should be allowed in Fracture phase"
        );
    }

    #[test]
    fn test_assert_in_assert_phase_allowed() {
        // Assertions are allowed in dedicated Assert phase
        let span = test_span();
        let path = Path::from("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: None,
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "assert".to_string(),
            phase: Phase::Assert,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_ok(),
            "Assertions should be allowed in Assert phase"
        );
    }

    #[test]
    fn test_assert_in_measure_phase_rejected() {
        // Assertions are NOT allowed in Measure phase (observer boundary violation)
        let span = test_span();
        let path = Path::from("test.field");
        let mut node = Node::new(
            path,
            span,
            RoleData::Field {
                reconstruction: None,
            },
            (),
        );

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: None,
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "measure".to_string(),
            phase: Phase::Measure,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_err(),
            "Assertions should be rejected in Measure phase"
        );
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::PhaseBoundaryViolation));
        assert!(errors[0]
            .message
            .contains("Assertions only valid in Resolve/Fracture/Assert phases"));
    }

    #[test]
    fn test_assert_in_collect_phase_rejected() {
        // Assertions are NOT allowed in Collect phase (effect phase, not validation)
        let span = test_span();
        let path = Path::from("test.operator");
        let mut node = Node::new(path, span, RoleData::Operator, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: None,
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "collect".to_string(),
            phase: Phase::Collect,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_err(),
            "Assertions should be rejected in Collect phase"
        );
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::PhaseBoundaryViolation));
    }

    #[test]
    fn test_assert_non_bool_condition_rejected() {
        // Assert condition must be Bool type
        let span = test_span();
        let path = Path::from("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_scalar_expr(), // Wrong type: Scalar instead of Bool
            severity: None,
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_err(),
            "Assert with non-Bool condition should be rejected"
        );
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
        assert!(errors[0].message.contains("assert condition must be Bool"));
    }

    #[test]
    fn test_assert_valid_severity_levels() {
        // Valid severity levels: warn, error, fatal
        let span = test_span();

        for severity in ["warn", "error", "fatal"] {
            let path = Path::from("test.signal");
            let mut node = Node::new(path, span, RoleData::Signal, ());

            let assert_stmt = TypedStmt::Assert {
                condition: assertion_make_bool_expr(true),
                severity: Some(severity.to_string()),
                message: None,
                span,
            };

            node.executions.push(Execution {
                name: "resolve".to_string(),
                phase: Phase::Resolve,
                body: ExecutionBody::Statements(vec![assert_stmt]),
                reads: vec![],
                temporal_reads: vec![],
                emits: vec![],
                span,
            });

            let type_table = TypeTable::new();
            let registry = KernelRegistry::global();
            let result = validate_node(&node, &type_table, &registry);

            assert!(result.is_ok(), "Severity '{}' should be valid", severity);
        }
    }

    #[test]
    fn test_assert_invalid_severity_rejected() {
        // Invalid severity levels should be rejected
        let span = test_span();
        let path = Path::from("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: Some("critical".to_string()), // Invalid severity
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(result.is_err(), "Invalid severity should be rejected");
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::TypeMismatch));
        assert!(errors[0]
            .message
            .contains("Must be 'warn', 'error', or 'fatal'"));
    }

    #[test]
    fn test_assert_with_custom_message() {
        // Assertions can have custom messages
        let span = test_span();
        let path = Path::from("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let assert_stmt = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: Some("error".to_string()),
            message: Some("Temperature out of bounds".to_string()),
            span,
        };

        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Statements(vec![assert_stmt]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_ok(),
            "Assertions with custom messages should be valid"
        );
    }

    #[test]
    fn test_multiple_assertions_in_block() {
        // Multiple assertions in the same block should all be validated
        let span = test_span();
        let path = Path::from("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let assert1 = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: Some("warn".to_string()),
            message: None,
            span,
        };

        let assert2 = TypedStmt::Assert {
            condition: assertion_make_scalar_expr(), // Invalid: non-Bool
            severity: Some("error".to_string()),
            message: None,
            span,
        };

        let assert3 = TypedStmt::Assert {
            condition: assertion_make_bool_expr(true),
            severity: Some("invalid_level".to_string()), // Invalid severity
            message: None,
            span,
        };

        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Statements(vec![assert1, assert2, assert3]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span,
        });

        let type_table = TypeTable::new();
        let registry = KernelRegistry::global();
        let result = validate_node(&node, &type_table, &registry);

        assert!(
            result.is_err(),
            "Should report errors for invalid assertions"
        );
        let errors = result.unwrap_err();
        assert_eq!(
            errors.len(),
            2,
            "Should report both type mismatch and invalid severity"
        );
    }
}
