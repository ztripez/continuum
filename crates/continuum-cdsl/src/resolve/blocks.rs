//! Execution block compilation pass.
//!
//! Converts raw execution blocks from the parser into compiled `Execution` structures.
//! Validates that phases are appropriate for the node's role and extracts
//! signal/field dependencies for DAG construction.
//!
//! # What This Pass Does
//!
//! 1. **Phase Validation** - Verifies phase names and role compatibility.
//! 2. **Purity Enforcement** - Ensures pure phases don't have statement blocks.
//! 3. **Dependency Extraction** - Walks expression trees to find read dependencies.
//! 4. **Execution Creation** - Populates the `executions` list on the node.
//! 5. **Lifecycle Management** - Clears `execution_blocks` after successful compilation.
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Block Compilation → Validation
//!                                                         ^^^^^^^^^^^^
//!                                                         YOU ARE HERE
//! ```

use crate::ast::{
    BlockBody, Execution, ExecutionBody, ExprKind, ExpressionVisitor, Index, Node, RoleId,
    TypedExpr,
};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Path, Phase};
use std::collections::HashSet;

/// Parses a string phase name into a Phase enum value.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidCapability`] if the name is unrecognized or
/// is a legacy name like "apply" or "emit".
pub fn parse_phase_name(name: &str, span: crate::foundation::Span) -> Result<Phase, CompileError> {
    match name.to_lowercase().as_str() {
        "resolve" => Ok(Phase::Resolve),
        "collect" => Ok(Phase::Collect),
        "fracture" => Ok(Phase::Fracture),
        "measure" => Ok(Phase::Measure),
        "assert" => Ok(Phase::Assert),
        "configure" => Ok(Phase::Configure),

        // Handle legacy names with helpful error messages
        "apply" | "emit" => Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "legacy execution phase '{}' is no longer supported. Use 'collect' for signal inputs or 'measure' for observations.",
                name
            ),
        )),

        _ => Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "unknown execution phase '{}'. Valid phases are: resolve, collect, fracture, measure, assert, configure",
                name
            ),
        )),
    }
}

/// Extracts signal and field paths from an expression tree.
///
/// Recursively walks the expression tree to find all `Signal` and `Field` path references.
/// These become the `reads` for the execution block, used for DAG dependency analysis.
///
/// Returns paths in **deterministic sorted order** to satisfy the determinism invariant.
fn extract_dependencies(expr: &TypedExpr) -> Vec<Path> {
    let mut visitor = DependencyVisitor::default();
    expr.walk(&mut visitor);

    // Sort for determinism (satisfies hard invariant)
    let mut paths: Vec<_> = visitor.paths.into_iter().collect();
    paths.sort();
    paths
}

/// Visitor that collects signal and field paths from an expression tree.
#[derive(Default)]
struct DependencyVisitor {
    paths: HashSet<Path>,
}

impl ExpressionVisitor for DependencyVisitor {
    fn visit_expr(&mut self, expr: &TypedExpr) {
        match &expr.expr {
            ExprKind::Signal(path) | ExprKind::Field(path) => {
                self.paths.insert(path.clone());
            }
            _ => {}
        }
    }
}

/// Validate that a phase is allowed for a node's role.
///
/// Uses the role's spec to check if the phase is in the allowed set.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidCapability`] if the phase is not allowed for this role.
///
/// # Examples
///
/// ```rust,ignore
/// // Signal role allows Resolve phase
/// validate_phase_for_role(Phase::Resolve, RoleId::Signal, node.span)?;
///
/// // Signal role does NOT allow Collect phase
/// validate_phase_for_role(Phase::Collect, RoleId::Signal, node.span)?; // Error
/// ```
fn validate_phase_for_role(
    phase: Phase,
    role_id: RoleId,
    span: crate::foundation::Span,
) -> Result<(), CompileError> {
    let spec = role_id.spec();
    if spec.allowed_phases.contains(phase) {
        Ok(())
    } else {
        Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "{:?} role cannot have {:?} phase execution block",
                role_id, phase
            ),
        ))
    }
}

/// Compile execution blocks for a node.
///
/// Converts raw `execution_blocks` into typed `Execution` structs.
/// Populates `node.executions` and clears `node.execution_blocks`.
///
/// # Errors
///
/// Returns errors for:
/// - Unknown phase names
/// - Phases not allowed for role
/// - Statement blocks in pure phases
/// - Type resolution failures (should not happen if type resolution passed)
///
/// # Examples
///
/// ```rust,ignore
/// let mut node = parsed_node;
/// compile_execution_blocks(&mut node)?;
///
/// assert!(node.executions.len() > 0);
/// assert!(node.execution_blocks.is_empty());
/// ```
pub fn compile_execution_blocks<I: Index>(node: &mut Node<I>) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();
    let mut executions = Vec::new();

    let role_id = node.role.id();

    // Process each execution block
    for (phase_name, block_body) in &node.execution_blocks {
        // 1. Parse phase name
        let phase = match parse_phase_name(phase_name, node.span) {
            Ok(p) => p,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };

        // 2. Validate phase for role
        if let Err(e) = validate_phase_for_role(phase, role_id, node.span) {
            errors.push(e);
            continue;
        }

        // 3. Validate block body type matches phase purity
        let is_pure_phase = matches!(phase, Phase::Resolve | Phase::Measure | Phase::Assert);
        let has_statements = matches!(block_body, BlockBody::Statements(_));

        if is_pure_phase && has_statements {
            errors.push(CompileError::new(
                ErrorKind::EffectInPureContext,
                node.span,
                format!(
                    "{:?} phase is pure and cannot contain statement blocks",
                    phase
                ),
            ));
            continue;
        }

        // 4. Extract body as TypedExpr
        let body = match block_body {
            BlockBody::TypedExpression(typed_expr) => ExecutionBody::Expr(typed_expr.clone()),
            BlockBody::Expression(_) => {
                // Should never happen if expression typing ran first
                errors.push(CompileError::new(
                    ErrorKind::Internal,
                    node.span,
                    format!(
                        "execution block '{}' not typed (expression typing should run before block compilation)",
                        phase_name
                    ),
                ));
                continue;
            }
            BlockBody::Statements(_) => {
                // Statement blocks are valid in effect phases (Collect, Fracture)
                // but statement compilation is not yet implemented.
                // Fail loudly instead of silently dropping.
                errors.push(CompileError::new(
                    ErrorKind::Internal,
                    node.span,
                    format!(
                        "statement block compilation not yet implemented for '{}' phase (effect phases require statement support)",
                        phase_name
                    ),
                ));
                continue;
            }
        };

        // 5. Extract dependencies
        let reads = match &body {
            ExecutionBody::Expr(expr) => extract_dependencies(expr),
            ExecutionBody::Statements(_) => Vec::new(), // Not yet implemented
        };

        // 6. Create Execution
        let execution = Execution::new(
            phase_name.clone(),
            phase,
            body,
            reads,
            vec![], // Emits extraction for statements not yet implemented
            node.span,
        );
        executions.push(execution);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // 7. Update node
    node.executions = executions;
    node.execution_blocks.clear();

    // 8. Populate node-level reads for cycle detection (Phase 12 structure validation)
    // Union of all per-execution reads
    let mut all_reads = std::collections::HashSet::new();
    for execution in &node.executions {
        for read in &execution.reads {
            all_reads.insert(read.clone());
        }
    }
    let mut sorted_reads: Vec<_> = all_reads.into_iter().collect();
    sorted_reads.sort();
    node.reads = sorted_reads;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{KernelType, Shape, Span, Type, Unit};

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
    fn test_parse_phase_name_valid() {
        let span = test_span();
        assert_eq!(parse_phase_name("resolve", span).unwrap(), Phase::Resolve);
        assert_eq!(parse_phase_name("collect", span).unwrap(), Phase::Collect);
        assert_eq!(parse_phase_name("fracture", span).unwrap(), Phase::Fracture);
        assert_eq!(parse_phase_name("measure", span).unwrap(), Phase::Measure);
        assert_eq!(parse_phase_name("assert", span).unwrap(), Phase::Assert);
        assert_eq!(
            parse_phase_name("configure", span).unwrap(),
            Phase::Configure
        );
    }

    #[test]
    fn test_parse_phase_name_legacy_rejected() {
        let span = test_span();
        // "apply" and "emit" are legacy names that should be rejected
        assert!(parse_phase_name("apply", span).is_err());
        assert!(parse_phase_name("emit", span).is_err());
    }

    #[test]
    fn test_parse_phase_name_invalid() {
        let span = test_span();
        let result = parse_phase_name("invalid", span);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidCapability);
        assert!(err.message.contains("unknown execution phase"));
    }

    #[test]
    fn test_validate_phase_for_role_signal_resolve() {
        let span = test_span();
        assert!(validate_phase_for_role(Phase::Resolve, RoleId::Signal, span).is_ok());
    }

    #[test]
    fn test_validate_phase_for_role_signal_collect_invalid() {
        let span = test_span();
        let result = validate_phase_for_role(Phase::Collect, RoleId::Signal, span);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidCapability);
        assert!(err.message.contains("cannot have"));
    }

    #[test]
    fn test_validate_phase_for_role_operator_fracture() {
        let span = test_span();
        // Operator allows Fracture phase
        assert!(validate_phase_for_role(Phase::Fracture, RoleId::Operator, span).is_ok());
    }

    #[test]
    fn test_extract_dependencies_empty() {
        let span = test_span();
        let ty = scalar_type();
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            ty,
            span,
        );
        let deps = extract_dependencies(&expr);
        assert_eq!(deps.len(), 0);
    }

    #[test]
    fn test_extract_dependencies_signal() {
        let span = test_span();
        let ty = scalar_type();
        let path = Path::from_str("signal.temperature");
        let expr = TypedExpr::new(ExprKind::Signal(path.clone()), ty, span);
        let deps = extract_dependencies(&expr);

        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&path));
    }

    #[test]
    fn test_extract_dependencies_nested() {
        use crate::ast::KernelId;

        let span = test_span();
        let ty = scalar_type();
        let path1 = Path::from_str("signal.a");
        let path2 = Path::from_str("field.b");

        let left = TypedExpr::new(ExprKind::Signal(path1.clone()), ty.clone(), span);
        let right = TypedExpr::new(ExprKind::Field(path2.clone()), ty.clone(), span);

        // Binary ops desugar to Call(maths.add, [left, right])
        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![left, right],
            },
            ty,
            span,
        );

        let deps = extract_dependencies(&expr);

        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&path1));
        assert!(deps.contains(&path2));
    }

    #[test]
    fn test_compile_execution_blocks_with_typed_expression() {
        use crate::ast::RoleData;
        use crate::foundation::{KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");

        // Create a typed expression (simple literal)
        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });
        let typed_expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: Some(Unit::DIMENSIONLESS),
            },
            ty,
            span,
        );

        // Create node with resolve block containing typed expression
        let mut node = Node::new(path.clone(), span, RoleData::Signal, ());
        node.execution_blocks = vec![(
            "resolve".to_string(),
            BlockBody::TypedExpression(typed_expr),
        )];

        // Compile execution blocks
        let result = compile_execution_blocks(&mut node);
        assert!(result.is_ok(), "Compilation should succeed: {:?}", result);

        // Verify executions populated
        assert_eq!(node.executions.len(), 1, "Should have 1 execution");
        assert_eq!(node.executions[0].name, "resolve");
        assert_eq!(node.executions[0].phase, Phase::Resolve);
        match &node.executions[0].body {
            ExecutionBody::Expr(e) => {
                assert!(matches!(e.expr, ExprKind::Literal { .. }));
            }
            _ => panic!("Expected Expr body"),
        }

        // Verify execution_blocks cleared
        assert!(
            node.execution_blocks.is_empty(),
            "execution_blocks should be cleared"
        );

        // Verify node-level reads populated (from typed_expr, which is empty literal here)
        assert!(
            node.reads.is_empty(),
            "node.reads should be empty for literal-only execution"
        );
    }

    #[test]
    fn test_compile_execution_blocks_populates_node_reads() {
        use crate::ast::RoleData;
        use crate::foundation::{KernelType, Shape, Unit};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");
        let signal_path = Path::from_str("other.signal");

        // Create a typed expression that reads a signal
        let ty = Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        });
        let typed_expr = TypedExpr::new(ExprKind::Signal(signal_path.clone()), ty, span);

        // Create node with resolve block
        let mut node = Node::new(path.clone(), span, RoleData::Signal, ());
        node.execution_blocks = vec![(
            "resolve".to_string(),
            BlockBody::TypedExpression(typed_expr),
        )];

        // Compile execution blocks
        let result = compile_execution_blocks(&mut node);
        assert!(result.is_ok());

        // Verify node.reads contains the signal
        assert_eq!(node.reads.len(), 1);
        assert_eq!(node.reads[0], signal_path);
    }

    #[test]
    fn test_compile_execution_blocks_untyped_expression_error() {
        use crate::ast::{Expr, RoleData, UntypedKind};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.signal");

        // Create untyped expression
        let untyped_expr = Expr::new(
            UntypedKind::Literal {
                value: 42.0,
                unit: None,
            },
            span,
        );

        // Create node with resolve block containing UNTYPED expression
        let mut node = Node::new(path.clone(), span, RoleData::Signal, ());
        node.execution_blocks = vec![("resolve".to_string(), BlockBody::Expression(untyped_expr))];

        // Compile execution blocks - should fail
        let result = compile_execution_blocks(&mut node);
        assert!(result.is_err(), "Should fail with untyped expression");

        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("not typed"));
    }

    #[test]
    fn test_compile_execution_blocks_statement_block_error() {
        use crate::ast::{Expr, RoleData, Stmt, UntypedKind};

        let span = Span::new(0, 0, 10, 1);
        let path = Path::from("test.operator");

        // Create a statement block (for effect phase)
        let stmt = Stmt::Expr(Expr::new(
            UntypedKind::Literal {
                value: 42.0,
                unit: None,
            },
            span,
        ));

        // Create node with collect block containing statements
        // Operators are allowed to have collect phase
        let mut node = Node::new(path.clone(), span, RoleData::Operator, ());
        node.execution_blocks = vec![("collect".to_string(), BlockBody::Statements(vec![stmt]))];

        // Compile execution blocks - should fail with "not yet implemented"
        let result = compile_execution_blocks(&mut node);
        assert!(
            result.is_err(),
            "Should fail with statement blocks not implemented"
        );

        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::Internal));
        assert!(
            errors[0]
                .message
                .contains("statement block compilation not yet implemented"),
            "Error message: {}",
            errors[0].message
        );
    }
}
