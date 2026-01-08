//! IR validation
//!
//! Validates the compiled IR and emits warnings for potential issues.

use std::collections::HashSet;

use tracing::warn;

// Import functions crate to ensure kernels are registered
use continuum_functions as _;

use crate::{CompiledExpr, CompiledWorld, ValueType};

/// A compilation warning
#[derive(Debug, Clone)]
pub struct CompileWarning {
    /// Warning code for filtering/identification
    pub code: WarningCode,
    /// Human-readable message
    pub message: String,
    /// The entity this warning relates to (signal path, operator path, etc.)
    pub entity: String,
}

/// Warning codes for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningCode {
    /// Signal has range constraint but no assertions to validate it
    MissingRangeAssertion,
    /// Reference to undefined symbol (likely typo)
    UndefinedSymbol,
    /// Unknown function call
    UnknownFunction,
}

/// Validate a compiled world and return any warnings
pub fn validate(world: &CompiledWorld) -> Vec<CompileWarning> {
    let mut warnings = Vec::new();

    check_range_assertions(world, &mut warnings);
    check_undefined_symbols(world, &mut warnings);

    // Log warnings
    for warning in &warnings {
        warn!(
            code = ?warning.code,
            entity = %warning.entity,
            "{}",
            warning.message
        );
    }

    warnings
}

/// Check that signals with range types have assertions
fn check_range_assertions(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    for (signal_id, signal) in &world.signals {
        // Check if the signal has a range constraint
        let has_range = matches!(&signal.value_type, ValueType::Scalar { range: Some(_) });

        if has_range && signal.assertions.is_empty() {
            warnings.push(CompileWarning {
                code: WarningCode::MissingRangeAssertion,
                message: format!(
                    "signal '{}' has a range constraint but no assertions to validate it at runtime",
                    signal_id.0
                ),
                entity: signal_id.0.clone(),
            });
        }
    }
}

/// Known built-in functions - sourced from the kernel registry
fn is_known_function(name: &str) -> bool {
    continuum_kernel_registry::is_known(name)
}

/// Check for undefined symbols in expressions
fn check_undefined_symbols(world: &CompiledWorld, warnings: &mut Vec<CompileWarning>) {
    // Collect all defined symbols
    let mut defined_signals: HashSet<&str> = HashSet::new();
    for signal_id in world.signals.keys() {
        defined_signals.insert(&signal_id.0);
    }

    let defined_constants: HashSet<&str> = world.constants.keys().map(|s| s.as_str()).collect();
    let defined_config: HashSet<&str> = world.config.keys().map(|s| s.as_str()).collect();

    // Check signals
    for (signal_id, signal) in &world.signals {
        if let Some(resolve) = &signal.resolve {
            check_expr_symbols(
                resolve,
                &format!("signal.{}", signal_id.0),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
        for assertion in &signal.assertions {
            check_expr_symbols(
                &assertion.condition,
                &format!("signal.{} assert", signal_id.0),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check fields
    for (field_id, field) in &world.fields {
        if let Some(measure) = &field.measure {
            check_expr_symbols(
                measure,
                &format!("field.{}", field_id.0),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check fractures
    for (fracture_id, fracture) in &world.fractures {
        for condition in &fracture.conditions {
            check_expr_symbols(
                condition,
                &format!("fracture.{}", fracture_id.0),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
        for emit in &fracture.emits {
            check_expr_symbols(
                &emit.value,
                &format!("fracture.{}", fracture_id.0),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }

    // Check era transitions
    for (era_id, era) in &world.eras {
        for transition in &era.transitions {
            check_expr_symbols(
                &transition.condition,
                &format!("era.{} transition", era_id.0),
                &defined_signals,
                &defined_constants,
                &defined_config,
                warnings,
            );
        }
    }
}

/// Check a single expression for undefined symbols
fn check_expr_symbols(
    expr: &CompiledExpr,
    context: &str,
    defined_signals: &HashSet<&str>,
    defined_constants: &HashSet<&str>,
    defined_config: &HashSet<&str>,
    warnings: &mut Vec<CompileWarning>,
) {
    match expr {
        CompiledExpr::Signal(signal_id) => {
            if !defined_signals.contains(signal_id.0.as_str()) {
                warnings.push(CompileWarning {
                    code: WarningCode::UndefinedSymbol,
                    message: format!(
                        "undefined signal '{}' in {} (possible typo?)",
                        signal_id.0, context
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Const(name) => {
            if !defined_constants.contains(name.as_str()) {
                warnings.push(CompileWarning {
                    code: WarningCode::UndefinedSymbol,
                    message: format!(
                        "undefined constant '{}' in {} (possible typo?)",
                        name, context
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Config(name) => {
            if !defined_config.contains(name.as_str()) {
                warnings.push(CompileWarning {
                    code: WarningCode::UndefinedSymbol,
                    message: format!(
                        "undefined config '{}' in {} (possible typo?)",
                        name, context
                    ),
                    entity: context.to_string(),
                });
            }
        }
        CompiledExpr::Call { function, args } => {
            if !is_known_function(function) {
                warnings.push(CompileWarning {
                    code: WarningCode::UnknownFunction,
                    message: format!(
                        "unknown function '{}' in {} (possible typo?)",
                        function, context
                    ),
                    entity: context.to_string(),
                });
            }
            for arg in args {
                check_expr_symbols(arg, context, defined_signals, defined_constants, defined_config, warnings);
            }
        }
        CompiledExpr::Binary { left, right, .. } => {
            check_expr_symbols(left, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(right, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::Unary { operand, .. } => {
            check_expr_symbols(operand, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            check_expr_symbols(condition, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(then_branch, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(else_branch, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::Let { value, body, .. } => {
            check_expr_symbols(value, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(body, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::FieldAccess { object, .. } => {
            check_expr_symbols(object, context, defined_signals, defined_constants, defined_config, warnings);
        }
        // Literals, Prev, DtRaw, SumInputs, Local don't need checking
        // Local variables are validated at parse/lower time
        CompiledExpr::Literal(_) | CompiledExpr::Prev | CompiledExpr::DtRaw | CompiledExpr::SumInputs | CompiledExpr::Local(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower;
    use continuum_dsl::parse;

    fn parse_and_lower(src: &str) -> CompiledWorld {
        let (unit, errors) = parse(src);
        assert!(errors.is_empty(), "parse errors: {:?}", errors);
        lower(&unit.unwrap()).unwrap()
    }

    #[test]
    fn test_signal_with_range_no_assertion_warns() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K, 100..10000>
                : strata(terra)
                resolve { prev }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, WarningCode::MissingRangeAssertion);
        assert!(warnings[0].entity.contains("terra.temp"));
    }

    #[test]
    fn test_signal_with_range_and_assertion_no_warning() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K, 100..10000>
                : strata(terra)
                resolve { prev }
                assert {
                    prev > 100
                }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert!(warnings.is_empty());
    }

    #[test]
    fn test_signal_without_range_no_warning() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { prev }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        assert!(warnings.is_empty());
    }

    #[test]
    fn test_undefined_symbol_warns() {
        // "sum_inputs" is a typo - should be "sum(inputs)"
        // The parser will treat it as a path/signal reference
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { prev + sum_inputs }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should warn about undefined symbol "sum_inputs"
        assert!(!warnings.is_empty());
        let undefined_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UndefinedSymbol)
            .collect();
        assert_eq!(undefined_warnings.len(), 1);
        assert!(undefined_warnings[0].message.contains("sum_inputs"));
    }

    #[test]
    fn test_unknown_function_warns() {
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { unknownfunc(prev) }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should warn about unknown function
        let func_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UnknownFunction)
            .collect();
        assert_eq!(func_warnings.len(), 1);
        assert!(func_warnings[0].message.contains("unknownfunc"));
    }

    #[test]
    fn test_valid_symbols_no_warning() {
        let src = r#"
            const {
                physics.gravity: 9.81
            }

            config {
                thermal.decay_halflife: 1.0
            }

            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { decay(prev, config.thermal.decay_halflife) + sum(inputs) }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // No undefined symbol or unknown function warnings
        let symbol_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| matches!(w.code, WarningCode::UndefinedSymbol | WarningCode::UnknownFunction))
            .collect();
        assert!(symbol_warnings.is_empty(), "unexpected warnings: {:?}", symbol_warnings);
    }
}
