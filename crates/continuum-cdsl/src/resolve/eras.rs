//! Era resolution for CDSL AST.
//!
//! Validates era declarations and their references to strata and other eras.
//!
//! # What This Pass Does
//!
//! ## Era Declaration Validation
//!
//! Validates each `Era` declaration:
//! - `dt` expression must have time units
//! - `strata_policy` references must point to declared strata
//! - `transitions` target eras must exist
//! - `transitions` conditions must be Bool-typed
//!
//! ## Cycle Detection
//!
//! Detects cycles in era transition graphs to warn about unreachable eras
//! or infinite transition loops.
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Name Res → Type Res → Validation → Stratum Resolution → Era Resolution → Block Compilation
//!                                                                   ^^^^^^^^^^^^^^^
//!                                                                    YOU ARE HERE
//! ```
//!
//! This pass runs after stratum resolution (Phase 12.5-B) and before execution
//! block compilation (Phase 12.5-D). It's part of Phase 12.5 prerequisites for
//! execution DAG construction.

use crate::ast::Era;
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{EraId, KernelType, StratumId, Type};
use std::collections::{HashMap, HashSet};

/// Validate all era declarations.
///
/// Validates:
/// - dt expressions have time units
/// - strata_policy references declared strata
/// - transition targets reference declared eras
/// - transition conditions are Bool-typed
///
/// # Errors
///
/// Returns errors for:
/// - Era dt without time units
/// - References to undeclared strata
/// - References to undeclared eras
/// - Non-Bool transition conditions
///
/// # Examples
///
/// ```rust,ignore
/// let eras = vec![...];
/// let strata = vec![...];
/// let errors = resolve_eras(&mut eras, &strata);
/// ```
pub fn resolve_eras(eras: &mut [Era], strata_ids: &[StratumId]) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Build lookup sets
    let stratum_set: HashSet<&StratumId> = strata_ids.iter().collect();
    let era_ids: Vec<EraId> = eras.iter().map(|e| e.id.clone()).collect();
    let era_set: HashSet<&EraId> = era_ids.iter().collect();

    for era in eras.iter() {
        // Validate dt has time units
        if let Err(e) = validate_era_dt(era) {
            errors.push(e);
        }

        // Validate stratum references in strata_policy
        for policy in &era.strata_policy {
            if !stratum_set.contains(&policy.stratum) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    era.span,
                    format!(
                        "era '{}' references undeclared stratum '{}'",
                        era.path,
                        policy.stratum.as_str()
                    ),
                ));
            }
        }

        // Validate era transition targets and conditions
        for transition in &era.transitions {
            // Validate target era exists
            if !era_set.contains(&transition.target) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    transition.span,
                    format!(
                        "era '{}' transition references undeclared era '{}'",
                        era.path,
                        transition.target.as_str()
                    ),
                ));
            }

            // Validate condition is Bool-typed
            if transition.condition.ty != Type::Bool {
                errors.push(CompileError::new(
                    ErrorKind::TypeMismatch,
                    transition.span,
                    format!(
                        "era '{}' transition condition must be Bool, found {:?}",
                        era.path, transition.condition.ty
                    ),
                ));
            }
        }
    }

    errors
}

/// Validate that era dt expression has time units.
///
/// The dt expression must resolve to a type with time dimension.
/// Checks that the unit has a non-zero time component.
fn validate_era_dt(era: &Era) -> Result<(), CompileError> {
    match &era.dt.ty {
        Type::Kernel(KernelType { unit, .. }) => {
            // Check if unit has time dimension (time exponent != 0)
            if unit.dims().time == 0 {
                return Err(CompileError::new(
                    ErrorKind::TypeMismatch,
                    era.span,
                    format!(
                        "era '{}' dt must have time units (seconds, years, etc.), found {}",
                        era.path, unit
                    ),
                ));
            }
            Ok(())
        }
        other => Err(CompileError::new(
            ErrorKind::TypeMismatch,
            era.span,
            format!(
                "era '{}' dt must be a kernel type with time units, found {:?}",
                era.path, other
            ),
        )),
    }
}

/// Detect cycles in era transition graph.
///
/// Uses depth-first search to detect cycles. Returns warnings (not errors)
/// because cycles might be intentional (e.g., oscillating eras).
///
/// # Returns
///
/// List of warnings for detected cycles with the cycle path.
pub fn detect_era_cycles(eras: &[Era]) -> Vec<CompileError> {
    let mut warnings = Vec::new();

    // Build adjacency list
    let mut graph: HashMap<&EraId, Vec<&EraId>> = HashMap::new();
    for era in eras {
        let targets: Vec<&EraId> = era.transitions.iter().map(|t| &t.target).collect();
        graph.insert(&era.id, targets);
    }

    // DFS for cycle detection
    let mut visited = HashSet::new();
    let mut rec_stack = Vec::new();

    for era in eras {
        if !visited.contains(&era.id) {
            if let Some(cycle) = dfs_cycle(&era.id, &graph, &mut visited, &mut rec_stack) {
                warnings.push(CompileError::new(
                    ErrorKind::CyclicDependency,
                    era.span,
                    format!(
                        "era transition cycle detected: {} (cycles may cause infinite loops)",
                        cycle.join(" → ")
                    ),
                ));
            }
        }
    }

    warnings
}

/// Depth-first search for cycle detection.
///
/// Returns cycle path if found, None otherwise.
fn dfs_cycle<'a>(
    node: &'a EraId,
    graph: &HashMap<&'a EraId, Vec<&'a EraId>>,
    visited: &mut HashSet<EraId>,
    rec_stack: &mut Vec<&'a EraId>,
) -> Option<Vec<String>> {
    visited.insert(node.clone());
    rec_stack.push(node);

    if let Some(neighbors) = graph.get(node) {
        for &neighbor in neighbors {
            if !visited.contains(neighbor) {
                if let Some(cycle) = dfs_cycle(neighbor, graph, visited, rec_stack) {
                    return Some(cycle);
                }
            } else if rec_stack.contains(&neighbor) {
                // Found cycle - build cycle path
                let start_idx = rec_stack.iter().position(|&n| n == neighbor).unwrap();
                let cycle_path: Vec<String> = rec_stack[start_idx..]
                    .iter()
                    .map(|id| id.as_str().to_string())
                    .chain(std::iter::once(neighbor.as_str().to_string()))
                    .collect();
                return Some(cycle_path);
            }
        }
    }

    rec_stack.pop();
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{EraTransition, StratumPolicy};
    use crate::ast::{ExprKind, TypedExpr};
    use crate::foundation::{Path, Shape, Span, Unit};

    fn make_time_expr(span: Span) -> TypedExpr {
        TypedExpr {
            expr: ExprKind::Literal {
                value: 1000.0,
                unit: Some(Unit::seconds()),
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::seconds(),
                bounds: None,
            }),
            span,
        }
    }

    fn make_bool_expr(span: Span) -> TypedExpr {
        TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty: Type::Bool,
            span,
        }
    }

    #[test]
    fn test_valid_era() {
        let span = Span::new(0, 0, 10, 1);
        let stratum_id = StratumId::new("fast");

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            make_time_expr(span),
            span,
        );
        era.strata_policy
            .push(StratumPolicy::new(stratum_id.clone(), true));

        let errors = resolve_eras(&mut [era], &[stratum_id]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_era_dt_without_time_units() {
        let span = Span::new(0, 0, 10, 1);

        // Create dt with length units instead of time
        let dt_expr = TypedExpr {
            expr: ExprKind::Literal {
                value: 1000.0,
                unit: Some(Unit::meters()),
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::meters(),
                bounds: None,
            }),
            span,
        };

        let era = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            dt_expr,
            span,
        );

        let errors = resolve_eras(&mut [era], &[]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("time units"));
    }

    #[test]
    fn test_era_references_undeclared_stratum() {
        let span = Span::new(0, 0, 10, 1);

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            make_time_expr(span),
            span,
        );
        era.strata_policy
            .push(StratumPolicy::new(StratumId::new("nonexistent"), true));

        let errors = resolve_eras(&mut [era], &[StratumId::new("fast")]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("undeclared stratum"));
    }

    #[test]
    fn test_era_transition_to_undeclared_era() {
        let span = Span::new(0, 0, 10, 1);

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            make_time_expr(span),
            span,
        );
        era.transitions.push(EraTransition::new(
            EraId::new("nonexistent"),
            make_bool_expr(span),
            span,
        ));

        let errors = resolve_eras(&mut [era], &[]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("undeclared era"));
    }

    #[test]
    fn test_era_transition_non_bool_condition() {
        let span = Span::new(0, 0, 10, 1);

        // Create condition with wrong type (Scalar instead of Bool)
        let condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::dimensionless(),
                bounds: None,
            }),
            span,
        };

        let mut era1 = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            make_time_expr(span),
            span,
        );
        let era2 = Era::new(
            EraId::new("stable"),
            Path::from_str("stable"),
            make_time_expr(span),
            span,
        );

        era1.transitions
            .push(EraTransition::new(EraId::new("stable"), condition, span));

        let errors = resolve_eras(&mut [era1, era2], &[]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must be Bool"));
    }

    #[test]
    fn test_detect_era_cycle() {
        let span = Span::new(0, 0, 10, 1);

        // Create cycle: formation → stable → formation
        let mut era1 = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            make_time_expr(span),
            span,
        );
        let mut era2 = Era::new(
            EraId::new("stable"),
            Path::from_str("stable"),
            make_time_expr(span),
            span,
        );

        era1.transitions.push(EraTransition::new(
            EraId::new("stable"),
            make_bool_expr(span),
            span,
        ));
        era2.transitions.push(EraTransition::new(
            EraId::new("formation"),
            make_bool_expr(span),
            span,
        ));

        let warnings = detect_era_cycles(&[era1, era2]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("cycle detected"));
    }

    #[test]
    fn test_no_cycle_in_linear_eras() {
        let span = Span::new(0, 0, 10, 1);

        // Linear: formation → stable (no cycle)
        let mut era1 = Era::new(
            EraId::new("formation"),
            Path::from_str("formation"),
            make_time_expr(span),
            span,
        );
        let era2 = Era::new(
            EraId::new("stable"),
            Path::from_str("stable"),
            make_time_expr(span),
            span,
        );

        era1.transitions.push(EraTransition::new(
            EraId::new("stable"),
            make_bool_expr(span),
            span,
        ));

        let warnings = detect_era_cycles(&[era1, era2]);
        assert!(warnings.is_empty());
    }
}
