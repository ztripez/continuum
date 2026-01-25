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
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::{strata, eras};
//! use continuum_cdsl::ast::{Era, Stratum};
//!
//! // After parsing and type resolution
//! let mut eras: Vec<Era> = parsed_ast.eras;
//! let strata: Vec<Stratum> = parsed_ast.strata;
//!
//! // Phase 12.5-A: Resolve strata
//! let stratum_ids: Vec<StratumId> = strata.iter().map(|s| s.id.clone()).collect();
//! let stratum_errors = strata::resolve_strata(&mut nodes, &stratum_ids);
//!
//! // Phase 12.5-B: Resolve eras
//! let era_errors = eras::resolve_eras(&mut eras, &stratum_ids);
//!
//! // Phase 12.5-C: Detect era cycles (warnings)
//! let cycle_warnings = eras::detect_era_cycles(&eras);
//!
//! // Collect all errors and warnings
//! let mut all_errors = Vec::new();
//! all_errors.extend(stratum_errors);
//! all_errors.extend(era_errors);
//! all_errors.extend(cycle_warnings);  // Optional: treat as warnings or errors
//!
//! if !all_errors.is_empty() {
//!     return Err(all_errors);
//! }
//!
//! // Ready for Phase 12.5-D: Execution block compilation
//! ```

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{EraId, KernelType, StratumId, Type};
use continuum_cdsl_ast::Era;
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
            if unit.dims().time.is_zero() {
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
    use continuum_cdsl_ast::foundation::{Path, Shape, Span, Unit};
    use continuum_cdsl_ast::{EraTransition, StratumPolicy};
    use continuum_cdsl_ast::{ExprKind, TypedExpr};

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
            Path::from_path_str("formation"),
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
            Path::from_path_str("formation"),
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
            Path::from_path_str("formation"),
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
            Path::from_path_str("formation"),
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
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let era2 = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
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
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let mut era2 = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
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
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let era2 = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
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

    // ===== Edge Case Tests =====

    #[test]
    fn test_era_with_empty_strata_policy() {
        let span = Span::new(0, 0, 10, 1);

        // Era with no strata policies (valid - era just controls dt)
        let era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );

        let errors = resolve_eras(&mut [era], &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_era_with_empty_transitions() {
        let span = Span::new(0, 0, 10, 1);

        // Terminal era with no transitions (valid)
        let era = Era::new(
            EraId::new("terminal"),
            Path::from_path_str("terminal"),
            make_time_expr(span),
            span,
        );

        let errors = resolve_eras(&mut [era], &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_era_with_multiple_strata_policies() {
        let span = Span::new(0, 0, 10, 1);
        let stratum1 = StratumId::new("fast");
        let stratum2 = StratumId::new("slow");
        let stratum3 = StratumId::new("rare");

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        era.strata_policy
            .push(StratumPolicy::new(stratum1.clone(), true));
        era.strata_policy
            .push(StratumPolicy::new(stratum2.clone(), true));
        era.strata_policy
            .push(StratumPolicy::new(stratum3.clone(), false)); // gated

        let errors = resolve_eras(&mut [era], &[stratum1, stratum2, stratum3]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_era_with_multiple_transitions() {
        let span = Span::new(0, 0, 10, 1);

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );

        // Multiple transitions (first matching wins)
        era.transitions.push(EraTransition::new(
            EraId::new("stable"),
            make_bool_expr(span),
            span,
        ));
        era.transitions.push(EraTransition::new(
            EraId::new("decay"),
            make_bool_expr(span),
            span,
        ));
        era.transitions.push(EraTransition::new(
            EraId::new("collapse"),
            make_bool_expr(span),
            span,
        ));

        let era_stable = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
            make_time_expr(span),
            span,
        );
        let era_decay = Era::new(
            EraId::new("decay"),
            Path::from_path_str("decay"),
            make_time_expr(span),
            span,
        );
        let era_collapse = Era::new(
            EraId::new("collapse"),
            Path::from_path_str("collapse"),
            make_time_expr(span),
            span,
        );

        let errors = resolve_eras(&mut [era, era_stable, era_decay, era_collapse], &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_era_with_cadence_override() {
        let span = Span::new(0, 0, 10, 1);
        let stratum_id = StratumId::new("fast");

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );

        // Stratum policy with cadence override
        let mut policy = StratumPolicy::new(stratum_id.clone(), true);
        policy.cadence_override = Some(10);
        era.strata_policy.push(policy);

        let errors = resolve_eras(&mut [era], &[stratum_id]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_era_with_gated_stratum() {
        let span = Span::new(0, 0, 10, 1);
        let stratum_id = StratumId::new("slow");

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );

        // Gated stratum (active: false)
        era.strata_policy
            .push(StratumPolicy::new(stratum_id.clone(), false));

        let errors = resolve_eras(&mut [era], &[stratum_id]);
        assert!(errors.is_empty());
    }

    // ===== Complex Cycle Tests =====

    #[test]
    fn test_self_transition_cycle() {
        let span = Span::new(0, 0, 10, 1);

        // Era transitions to itself
        let mut era = Era::new(
            EraId::new("oscillating"),
            Path::from_path_str("oscillating"),
            make_time_expr(span),
            span,
        );

        era.transitions.push(EraTransition::new(
            EraId::new("oscillating"),
            make_bool_expr(span),
            span,
        ));

        let warnings = detect_era_cycles(&[era]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("cycle detected"));
        assert!(warnings[0].message.contains("oscillating"));
    }

    #[test]
    fn test_three_way_cycle() {
        let span = Span::new(0, 0, 10, 1);

        // Cycle: formation → stable → decay → formation
        let mut era1 = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let mut era2 = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
            make_time_expr(span),
            span,
        );
        let mut era3 = Era::new(
            EraId::new("decay"),
            Path::from_path_str("decay"),
            make_time_expr(span),
            span,
        );

        era1.transitions.push(EraTransition::new(
            EraId::new("stable"),
            make_bool_expr(span),
            span,
        ));
        era2.transitions.push(EraTransition::new(
            EraId::new("decay"),
            make_bool_expr(span),
            span,
        ));
        era3.transitions.push(EraTransition::new(
            EraId::new("formation"),
            make_bool_expr(span),
            span,
        ));

        let warnings = detect_era_cycles(&[era1, era2, era3]);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("cycle detected"));
    }

    #[test]
    fn test_multiple_separate_cycles() {
        let span = Span::new(0, 0, 10, 1);

        // Cycle 1: A ↔ B
        let mut era_a = Era::new(
            EraId::new("a"),
            Path::from_path_str("a"),
            make_time_expr(span),
            span,
        );
        let mut era_b = Era::new(
            EraId::new("b"),
            Path::from_path_str("b"),
            make_time_expr(span),
            span,
        );

        era_a.transitions.push(EraTransition::new(
            EraId::new("b"),
            make_bool_expr(span),
            span,
        ));
        era_b.transitions.push(EraTransition::new(
            EraId::new("a"),
            make_bool_expr(span),
            span,
        ));

        // Cycle 2: C ↔ D
        let mut era_c = Era::new(
            EraId::new("c"),
            Path::from_path_str("c"),
            make_time_expr(span),
            span,
        );
        let mut era_d = Era::new(
            EraId::new("d"),
            Path::from_path_str("d"),
            make_time_expr(span),
            span,
        );

        era_c.transitions.push(EraTransition::new(
            EraId::new("d"),
            make_bool_expr(span),
            span,
        ));
        era_d.transitions.push(EraTransition::new(
            EraId::new("c"),
            make_bool_expr(span),
            span,
        ));

        let warnings = detect_era_cycles(&[era_a, era_b, era_c, era_d]);
        // Should detect both cycles
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn test_unreachable_era() {
        let span = Span::new(0, 0, 10, 1);

        // formation → stable, but decay is unreachable
        let mut era_formation = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let era_stable = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
            make_time_expr(span),
            span,
        );
        let era_decay = Era::new(
            EraId::new("decay"),
            Path::from_path_str("decay"),
            make_time_expr(span),
            span,
        );

        era_formation.transitions.push(EraTransition::new(
            EraId::new("stable"),
            make_bool_expr(span),
            span,
        ));

        // Note: This test validates that unreachable eras don't cause errors
        // Unreachability detection is a future enhancement (not implemented yet)
        let errors = resolve_eras(&mut [era_formation, era_stable, era_decay], &[]);
        assert!(errors.is_empty());
    }

    // ===== Transition Condition Type Tests =====

    #[test]
    fn test_transition_condition_with_int_type() {
        let span = Span::new(0, 0, 10, 1);

        let mut era1 = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let era2 = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
            make_time_expr(span),
            span,
        );

        // Condition with Int type (invalid - must be Bool)
        let int_condition = TypedExpr {
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

        era1.transitions.push(EraTransition::new(
            EraId::new("stable"),
            int_condition,
            span,
        ));

        let errors = resolve_eras(&mut [era1, era2], &[]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must be Bool"));
    }

    #[test]
    fn test_transition_condition_with_vector_type() {
        let span = Span::new(0, 0, 10, 1);

        let mut era1 = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );
        let era2 = Era::new(
            EraId::new("stable"),
            Path::from_path_str("stable"),
            make_time_expr(span),
            span,
        );

        // Condition with Vec3 type (invalid - must be Bool)
        let vec_condition = TypedExpr {
            expr: ExprKind::Vector(vec![]),
            ty: Type::Kernel(KernelType {
                shape: Shape::vec3(),
                unit: Unit::dimensionless(),
                bounds: None,
            }),
            span,
        };

        era1.transitions.push(EraTransition::new(
            EraId::new("stable"),
            vec_condition,
            span,
        ));

        let errors = resolve_eras(&mut [era1, era2], &[]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must be Bool"));
    }

    // ===== Multiple Error Tests =====

    #[test]
    fn test_era_with_multiple_errors() {
        let span = Span::new(0, 0, 10, 1);

        // Era with dt without time units
        let bad_dt_expr = TypedExpr {
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

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            bad_dt_expr,
            span,
        );

        // Reference undeclared stratum
        era.strata_policy
            .push(StratumPolicy::new(StratumId::new("nonexistent"), true));

        // Transition to undeclared era with non-Bool condition
        let int_condition = TypedExpr {
            expr: ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            ty: Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::dimensionless(),
                bounds: None,
            }),
            span,
        };

        era.transitions.push(EraTransition::new(
            EraId::new("missing_era"),
            int_condition,
            span,
        ));

        let errors = resolve_eras(&mut [era], &[]);
        // Should have 3 errors: bad dt, undeclared stratum, undeclared era + non-Bool condition
        assert!(errors.len() >= 3);

        let error_messages: Vec<String> = errors.iter().map(|e| e.message.clone()).collect();
        assert!(error_messages.iter().any(|m| m.contains("time units")));
        assert!(error_messages.iter().any(|m| m.contains("nonexistent")));
        assert!(error_messages
            .iter()
            .any(|m| m.contains("missing_era") || m.contains("must be Bool")));
    }

    #[test]
    fn test_all_strata_referenced_correctly() {
        let span = Span::new(0, 0, 10, 1);
        let stratum1 = StratumId::new("fast");
        let stratum2 = StratumId::new("slow");
        let stratum3 = StratumId::new("rare");

        let mut era = Era::new(
            EraId::new("formation"),
            Path::from_path_str("formation"),
            make_time_expr(span),
            span,
        );

        // Reference all strata
        era.strata_policy
            .push(StratumPolicy::new(stratum1.clone(), true));
        era.strata_policy
            .push(StratumPolicy::new(stratum2.clone(), true));
        era.strata_policy
            .push(StratumPolicy::new(stratum3.clone(), false));

        let errors = resolve_eras(&mut [era], &[stratum1, stratum2, stratum3]);
        assert!(errors.is_empty());
    }
}
