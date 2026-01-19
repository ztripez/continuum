//! Execution block compilation
//!
//! Compiles raw execution blocks from source into typed `Execution` structs for DAG construction.
//!
//! # What This Pass Does
//!
//! ## Execution Block Compilation
//!
//! Converts raw execution blocks stored in `Node.execution_blocks` into compiled
//! `Execution` structs:
//!
//! 1. **Phase Name Parsing** - Convert "resolve"/"collect"/etc to `Phase` enum
//! 2. **Role Validation** - Verify phase is allowed for node's role
//! 3. **Dependency Extraction** - Walk TypedExpr tree to find Path references
//! 4. **Execution Creation** - Build `Execution` structs with phase, body, reads
//! 5. **Statement Validation** - Ensure statements only in effect phases
//! 6. **Cleanup** - Clear `execution_blocks` after compilation
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Name Res → Type Res → Validation → Stratum → Era → Uses → Block Compilation
//!                                                                      ^^^^^^^^^^^^^^^^^
//!                                                                       YOU ARE HERE
//! ```
//!
//! This pass runs after uses validation (Phase 12.5-C) and before execution DAG
//! construction (Phase 13). It's the final part of Phase 12.5 execution prerequisites.
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::blocks;
//! use continuum_cdsl::ast::Node;
//!
//! // After type resolution, nodes have execution_blocks populated
//! let mut nodes: Vec<Node<_>> = parsed_ast.nodes;
//!
//! // Phase 12.5-D: Compile execution blocks
//! for node in &mut nodes {
//!     match blocks::compile_execution_blocks(node) {
//!         Ok(()) => {
//!             // node.executions now populated, execution_blocks cleared
//!             assert!(node.executions.len() > 0);
//!             assert!(node.execution_blocks.is_empty());
//!         }
//!         Err(errors) => {
//!             // Invalid phase for role, statements in pure phase, etc.
//!             return Err(errors);
//!         }
//!     }
//! }
//!
//! // Ready for Phase 13: DAG construction
//! ```

use crate::ast::{BlockBody, Execution, ExprKind, Index, Node, RoleId, TypedExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Path, Phase};
use std::collections::HashSet;

/// Parse phase name string to Phase enum.
///
/// Converts execution block phase names ("resolve", "collect", etc.) to Phase enum values.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidCapability`] for unknown phase names.
///
/// # Examples
///
/// ```rust,ignore
/// let phase = parse_phase_name("resolve", node.span)?;
/// assert_eq!(phase, Phase::Resolve);
/// ```
fn parse_phase_name(name: &str, span: crate::foundation::Span) -> Result<Phase, CompileError> {
    match name {
        "resolve" => Ok(Phase::Resolve),
        "collect" => Ok(Phase::Collect),
        "fracture" => Ok(Phase::Fracture),
        "measure" => Ok(Phase::Measure),
        "assert" => Ok(Phase::Assert),
        // Legacy/alternative names (not used in Phase enum but accepted by parser)
        "apply" | "emit" => Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "execution phase '{}' is not supported (use collect, resolve, fracture, measure, or assert)",
                name
            ),
        )),
        _ => Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!("unknown execution phase '{}'", name),
        )),
    }
}

/// Extract dependencies (signal/field paths) from a typed expression.
///
/// Recursively walks the expression tree to find all `Signal` and `Field` path references.
/// These become the `reads` for the execution block, used for DAG dependency analysis.
///
/// Returns paths in **deterministic sorted order** to satisfy the determinism invariant.
///
/// # Examples
///
/// ```rust,ignore
/// // Expression: prev + field.temperature
/// let deps = extract_dependencies(&typed_expr);
/// // deps contains: current signal path (from prev), field.temperature (sorted)
/// ```
fn extract_dependencies(expr: &TypedExpr) -> Vec<Path> {
    let mut paths = HashSet::new();
    collect_paths(expr, &mut paths);
    let mut paths: Vec<_> = paths.into_iter().collect();
    paths.sort(); // Ensure deterministic ordering (AGENTS.md: "All ordering is explicit and stable")
    paths
}

/// Recursively collect all signal/field paths from expression tree.
fn collect_paths(expr: &TypedExpr, paths: &mut HashSet<Path>) {
    match &expr.expr {
        ExprKind::Signal(path) | ExprKind::Field(path) => {
            paths.insert(path.clone());
        }
        ExprKind::Prev | ExprKind::Current => {
            // These reference the current signal - handled by DAG construction
        }
        ExprKind::Let { value, body, .. } => {
            collect_paths(value, paths);
            collect_paths(body, paths);
        }
        ExprKind::Call { args, .. } => {
            // Binary/Unary/If all desugar to Call
            for arg in args {
                collect_paths(arg, paths);
            }
        }
        ExprKind::Aggregate { body, .. } => {
            collect_paths(body, paths);
        }
        ExprKind::Fold { init, body, .. } => {
            collect_paths(init, paths);
            collect_paths(body, paths);
        }
        ExprKind::Vector(elements) => {
            for elem in elements {
                collect_paths(elem, paths);
            }
        }
        ExprKind::Struct { fields, .. } => {
            for (_, expr) in fields {
                collect_paths(expr, paths);
            }
        }
        ExprKind::FieldAccess { object, .. } => {
            collect_paths(object, paths);
        }
        // Leaf nodes
        ExprKind::Literal { .. }
        | ExprKind::Dt
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Local(_)
        | ExprKind::Payload
        | ExprKind::Inputs
        | ExprKind::Self_
        | ExprKind::Other => {}
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
        let typed_expr = match block_body {
            BlockBody::TypedExpression(typed_expr) => typed_expr.clone(),
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
                // Already validated above - statements not allowed in pure phases
                continue;
            }
        };

        // 5. Extract dependencies
        let reads = extract_dependencies(&typed_expr);

        // 6. Create Execution
        let execution = Execution::new(phase, typed_expr, reads, node.span);
        executions.push(execution);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // 7. Update node
    node.executions = executions;
    node.execution_blocks.clear();

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
        assert_eq!(node.executions[0].phase, Phase::Resolve);

        // Verify execution_blocks cleared
        assert!(
            node.execution_blocks.is_empty(),
            "execution_blocks should be cleared"
        );
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
}
