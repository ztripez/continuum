//! IR Validation and Warning Generation
//!
//! This module validates compiled IR and generates warnings for potential
//! issues that don't prevent compilation but may indicate problems.
//!
//! # Overview
//!
//! Validation runs after lowering and checks for:
//!
//! - **Missing assertions**: Signals with range constraints but no runtime validation
//! - **Undefined symbols**: References to signals, constants, or config that don't exist
//! - **Unknown functions**: Calls to functions not registered in the kernel registry
//!
//! # Warning vs Error
//!
//! This module produces warnings, not errors. Warnings indicate potential issues
//! but allow compilation to proceed. For example:
//!
//! - A signal with range `0..100` but no assertion may produce values outside that range
//! - A reference to `signal.temp` when only `signal.temperature` exists is likely a typo
//!
//! # Usage
//!
//! ```ignore
//! let world = lower(&compilation_unit)?;
//! let warnings = validate(&world);
//!
//! for warning in &warnings {
//!     eprintln!("[{:?}] {}: {}", warning.code, warning.entity, warning.message);
//! }
//! ```
//!
//! # Warning Codes
//!
//! Warnings are categorized by [`WarningCode`] for filtering and tooling:
//!
//! - `MissingRangeAssertion`: Compile with explicit assertions to fix
//! - `UndefinedSymbol`: Check for typos in symbol names
//! - `UnknownFunction`: Check function name or register new kernel

use std::collections::HashSet;

use tracing::warn;

// Import functions crate to ensure kernels are registered
use continuum_functions as _;

use crate::{CompiledExpr, CompiledWorld, ValueType};

/// A compilation warning indicating a potential issue in the IR.
///
/// Warnings do not prevent compilation but may indicate bugs or
/// configuration issues that should be addressed.
#[derive(Debug, Clone)]
pub struct CompileWarning {
    /// Warning code for filtering/identification
    pub code: WarningCode,
    /// Human-readable message
    pub message: String,
    /// The entity this warning relates to (signal path, operator path, etc.)
    pub entity: String,
}

/// Warning codes for categorization and filtering.
///
/// Each warning has a code that can be used by tooling to:
/// - Filter warnings by category
/// - Configure which warnings to treat as errors
/// - Generate targeted fix suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningCode {
    /// A signal declares a range constraint but has no assertions to validate it.
    ///
    /// This means the range is documentation-only and won't be checked at runtime.
    /// Add an `assert` block to enforce the range.
    MissingRangeAssertion,

    /// A reference to a signal, constant, or config that doesn't exist.
    ///
    /// This usually indicates a typo in the symbol name. Check spelling and
    /// ensure the referenced definition exists.
    UndefinedSymbol,

    /// A call to a function not registered in the kernel registry.
    ///
    /// This may indicate a typo in the function name or a missing kernel
    /// registration.
    UnknownFunction,
}

/// Validates a compiled world and returns any warnings.
///
/// This is the main entry point for IR validation. It runs all validation
/// checks and collects warnings into a single list.
///
/// # Checks Performed
///
/// 1. **Range assertions**: Signals with range types should have assertions
/// 2. **Undefined symbols**: All referenced symbols should exist
/// 3. **Unknown functions**: All called functions should be registered
///
/// # Logging
///
/// Warnings are also logged via `tracing::warn!` for immediate visibility.
///
/// # Example
///
/// ```ignore
/// let world = lower(&unit)?;
/// let warnings = validate(&world);
///
/// if !warnings.is_empty() {
///     println!("Compilation produced {} warning(s)", warnings.len());
/// }
/// ```
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

/// Checks that signals with range type constraints have runtime assertions.
///
/// A signal declared as `Scalar<K, 100..10000>` should have assertions to
/// validate the range at runtime. Without assertions, the range is purely
/// documentary and violations won't be detected.
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

/// Checks if a function name is registered in the kernel registry.
///
/// This delegates to `continuum_kernel_registry::is_known()` which tracks
/// all registered kernel functions.
fn is_known_function(name: &str) -> bool {
    continuum_kernel_registry::is_known(name)
}

/// Checks for undefined symbols in all expressions.
///
/// Scans all resolve expressions, measure expressions, assertion conditions,
/// transition conditions, and fracture expressions for references to
/// undefined signals, constants, config values, or unknown functions.
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

/// Recursively checks an expression for undefined symbols.
///
/// Walks the expression tree and reports warnings for:
/// - Signal references that don't exist in `defined_signals`
/// - Constant references that don't exist in `defined_constants`
/// - Config references that don't exist in `defined_config`
/// - Function calls to unknown kernel functions
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
        // Entity expressions - recurse into sub-expressions
        CompiledExpr::SelfField(_) => {}
        CompiledExpr::EntityAccess { .. } => {
            // Entity access validation happens at runtime
        }
        CompiledExpr::Aggregate { body, .. } => {
            check_expr_symbols(body, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::Other { body, .. } | CompiledExpr::Pairs { body, .. } => {
            check_expr_symbols(body, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::Filter { predicate, body, .. } => {
            check_expr_symbols(predicate, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(body, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::First { predicate, .. } => {
            check_expr_symbols(predicate, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::Nearest { position, .. } => {
            check_expr_symbols(position, context, defined_signals, defined_constants, defined_config, warnings);
        }
        CompiledExpr::Within { position, radius, body, .. } => {
            check_expr_symbols(position, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(radius, context, defined_signals, defined_constants, defined_config, warnings);
            check_expr_symbols(body, context, defined_signals, defined_constants, defined_config, warnings);
        }
        // Literals, Prev, DtRaw, Collected, Local don't need checking
        // Local variables are validated at parse/lower time
        CompiledExpr::Literal(_) | CompiledExpr::Prev | CompiledExpr::DtRaw | CompiledExpr::Collected | CompiledExpr::Local(_) => {}
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
        // "colected" is a typo - should be "collected"
        // The parser will treat it as a path/signal reference
        let src = r#"
            strata.terra {}
            era.main { : initial }

            signal.terra.temp {
                : Scalar<K>
                : strata(terra)
                resolve { prev + colected }
            }
        "#;

        let world = parse_and_lower(src);
        let warnings = validate(&world);

        // Should warn about undefined symbol "colected"
        assert!(!warnings.is_empty());
        let undefined_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.code == WarningCode::UndefinedSymbol)
            .collect();
        assert_eq!(undefined_warnings.len(), 1);
        assert!(undefined_warnings[0].message.contains("colected"));
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
                resolve { decay(prev, config.thermal.decay_halflife) + collected }
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
