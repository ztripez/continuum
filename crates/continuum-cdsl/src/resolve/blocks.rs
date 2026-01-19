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

use crate::ast::{BlockBody, ExprKind, Index, Node, RoleSpec, TypedExpr};
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
/// # Examples
///
/// ```rust,ignore
/// // Expression: prev + field.temperature
/// let deps = extract_dependencies(&typed_expr);
/// // deps contains: current signal path (from prev), field.temperature
/// ```
fn extract_dependencies(expr: &TypedExpr) -> Vec<Path> {
    let mut paths = HashSet::new();
    collect_paths(expr, &mut paths);
    paths.into_iter().collect()
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
        ExprKind::Binary { left, right, .. } => {
            collect_paths(left, paths);
            collect_paths(right, paths);
        }
        ExprKind::Unary { operand, .. } => {
            collect_paths(operand, paths);
        }
        ExprKind::Call { args, .. } => {
            for arg in args {
                collect_paths(arg, paths);
            }
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_paths(condition, paths);
            collect_paths(then_branch, paths);
            collect_paths(else_branch, paths);
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
        | ExprKind::Inputs => {}
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
/// validate_phase_for_role(Phase::Resolve, RoleData::Signal, node.span)?;
///
/// // Signal role does NOT allow Collect phase
/// validate_phase_for_role(Phase::Collect, RoleData::Signal, node.span)?; // Error
/// ```
fn validate_phase_for_role(
    phase: Phase,
    spec: &RoleSpec,
    span: crate::foundation::Span,
) -> Result<(), CompileError> {
    if spec.allowed_phases.contains(&phase) {
        Ok(())
    } else {
        Err(CompileError::new(
            ErrorKind::InvalidCapability,
            span,
            format!(
                "{} role cannot have {} phase execution block",
                spec.id, phase
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

    let role_spec = node.role.spec();

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
        if let Err(e) = validate_phase_for_role(phase, &role_spec, node.span) {
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
                    "{} phase is pure and cannot contain statement blocks",
                    phase
                ),
            ));
            continue;
        }

        // 4. Extract body as TypedExpr
        // NOTE: This assumes type resolution has already run and converted Expr → TypedExpr
        // For now, we'll need to handle the conversion or require it's already typed
        //
        // TODO: This is a placeholder - actual implementation needs to handle
        // BlockBody::Expression(Expr) → TypedExpr conversion through type resolution
        // OR we change the pipeline so type resolution populates execution_blocks with TypedExpr
        //
        // For now, skip compilation and collect error
        errors.push(CompileError::new(
            ErrorKind::Internal,
            node.span,
            "execution block compilation requires typed expressions (type resolution integration pending)".to_string(),
        ));
        continue;

        // 5. Extract dependencies
        // let reads = extract_dependencies(&body);

        // 6. Create Execution
        // executions.push(Execution::new(phase, body, reads, node.span));
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
    use crate::ast::RoleData;
    use crate::foundation::Span;

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
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
        let spec = RoleData::Signal.spec();
        assert!(validate_phase_for_role(Phase::Resolve, &spec, span).is_ok());
    }

    #[test]
    fn test_validate_phase_for_role_signal_collect_invalid() {
        let span = test_span();
        let spec = RoleData::Signal.spec();
        let result = validate_phase_for_role(Phase::Collect, &spec, span);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidCapability);
        assert!(err.message.contains("cannot have"));
        assert!(err.message.contains("Collect"));
    }

    #[test]
    fn test_validate_phase_for_role_operator_apply() {
        let span = test_span();
        let spec = RoleData::Operator.spec();
        assert!(validate_phase_for_role(Phase::Apply, &spec, span).is_ok());
    }

    #[test]
    fn test_extract_dependencies_empty() {
        use crate::foundation::Type;

        let span = test_span();
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::Scalar,
            span,
        );
        let deps = extract_dependencies(&expr);
        assert_eq!(deps.len(), 0);
    }

    #[test]
    fn test_extract_dependencies_signal() {
        use crate::foundation::Type;

        let span = test_span();
        let path = Path::from_str("signal.temperature");
        let expr = TypedExpr::new(ExprKind::Signal(path.clone()), Type::Scalar, span);
        let deps = extract_dependencies(&expr);

        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&path));
    }

    #[test]
    fn test_extract_dependencies_nested() {
        use crate::ast::BinaryOp;
        use crate::foundation::Type;

        let span = test_span();
        let path1 = Path::from_str("signal.a");
        let path2 = Path::from_str("field.b");

        let left = TypedExpr::new(ExprKind::Signal(path1.clone()), Type::Scalar, span);
        let right = TypedExpr::new(ExprKind::Field(path2.clone()), Type::Scalar, span);
        let expr = TypedExpr::binary(BinaryOp::Add, left, right, Type::Scalar, span);

        let deps = extract_dependencies(&expr);

        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&path1));
        assert!(deps.contains(&path2));
    }
}
