//! Semantic validation for the Continuum DSL.
//!
//! This module performs semantic validation on parsed ASTs before lowering
//! to IR. It catches errors that are syntactically valid but semantically
//! incorrect, such as:
//!
//! - Using `dt_raw` without the required `: uses(dt_raw)` declaration
//! - Calling functions that don't exist (neither user-defined nor kernel)
//! - Using `dt_raw` in contexts where it's not allowed (operators, impulses)
//!
//! # Validation vs Lowering Errors
//!
//! Validation errors are caught here, before lowering begins. This provides
//! better error messages with source spans. Lowering errors (`LowerError`
//! from `continuum_ir`) catch issues that require resolved symbol tables.
//!
//! # Usage
//!
//! ```ignore
//! use continuum_dsl::{parse, validate};
//!
//! let (unit, parse_errors) = parse(source);
//! if let Some(unit) = unit {
//!     let validation_errors = validate(&unit);
//!     if !validation_errors.is_empty() {
//!         for err in &validation_errors {
//!             eprintln!("{}", err);
//!         }
//!     }
//! }
//! ```

use crate::ast::{AstVisitor, CompilationUnit, Expr, Item, Spanned, uses_dt_raw};
use continuum_kernel_registry::{Arity, get_in_namespace};
use std::collections::HashMap;

/// A semantic validation error with source location.
///
/// Validation errors indicate that the parsed AST violates semantic rules
/// that can be checked without full symbol resolution. The span points to
/// the source location where the error was detected.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Human-readable description of the validation error.
    pub message: String,
    /// Byte range in the source where the error occurred.
    pub span: std::ops::Range<usize>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {:?}", self.message, self.span)
    }
}

impl std::error::Error for ValidationError {}

/// Validate a compilation unit for semantic correctness.
///
/// Performs checks that can be done without full symbol resolution:
/// - `dt_raw` usage requires explicit declaration
/// - Function calls must reference known functions and have correct arity
/// - `dt_raw` is not allowed in operators, impulses, or fractures
///
/// Returns a vector of validation errors. An empty vector means validation passed.
pub fn validate(unit: &CompilationUnit) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Collect all user-defined function names and their arities
    let mut user_functions = HashMap::new();
    let mut user_function_names = HashMap::new();
    for item in &unit.items {
        if let Item::FnDef(f) = &item.node {
            let arity = f.params.len();
            user_functions.insert(f.path.node.to_string(), arity);
            // Also store just the function name for method call matching
            if let Some(name) = f.path.node.segments.last() {
                user_function_names.insert(name.clone(), arity);
            }
        }
    }

    for item in &unit.items {
        // Check for unknown function calls and arity mismatches in all expressions
        collect_unknown_functions(
            &item.node,
            &user_functions,
            &user_function_names,
            &mut errors,
        );

        match &item.node {
            Item::SignalDef(signal) => {
                // Check if dt_raw is used in resolve block without : dt_raw declaration
                if let Some(resolve) = &signal.resolve {
                    if uses_dt_raw(&resolve.body.node) && !signal.dt_raw {
                        errors.push(ValidationError {
                            message: format!(
                                "signal '{}' uses dt_raw but does not declare : dt_raw",
                                signal.path.node
                            ),
                            span: resolve.body.span.clone(),
                        });
                    }
                }
            }
            Item::OperatorDef(operator) => {
                // Check if dt_raw is used in operator body (not allowed)
                if let Some(body) = &operator.body {
                    let expr = match body {
                        crate::ast::OperatorBody::Warmup(e) => e,
                        crate::ast::OperatorBody::Collect(e) => e,
                        crate::ast::OperatorBody::Measure(e) => e,
                    };
                    if uses_dt_raw(&expr.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "operator '{}' uses dt_raw which is not allowed in kernels",
                                operator.path.node
                            ),
                            span: expr.span.clone(),
                        });
                    }
                }
            }
            Item::ImpulseDef(impulse) => {
                // Check if dt_raw is used in impulse apply block (not allowed)
                if let Some(apply) = &impulse.apply {
                    if uses_dt_raw(&apply.body.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "impulse '{}' uses dt_raw which is not allowed in impulses",
                                impulse.path.node
                            ),
                            span: apply.body.span.clone(),
                        });
                    }
                }
            }
            Item::FractureDef(fracture) => {
                // Check if dt_raw is used in fracture conditions or emit
                for condition in &fracture.conditions {
                    if uses_dt_raw(&condition.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "fracture '{}' uses dt_raw in condition which is not allowed",
                                fracture.path.node
                            ),
                            span: condition.span.clone(),
                        });
                    }
                }
                if let Some(emit) = &fracture.emit {
                    if uses_dt_raw(&emit.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "fracture '{}' uses dt_raw in emit which is not allowed",
                                fracture.path.node
                            ),
                            span: emit.span.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    errors
}

/// Collect unknown function calls and arity errors from an item.
fn collect_unknown_functions(
    item: &Item,
    user_functions: &HashMap<String, usize>,
    user_function_names: &HashMap<String, usize>,
    errors: &mut Vec<ValidationError>,
) {
    match item {
        Item::SignalDef(signal) => {
            if let Some(resolve) = &signal.resolve {
                check_expr_for_unknown_functions(
                    &resolve.body,
                    user_functions,
                    user_function_names,
                    errors,
                );
            }
            if let Some(assertions) = &signal.assertions {
                for assertion in &assertions.assertions {
                    check_expr_for_unknown_functions(
                        &assertion.condition,
                        user_functions,
                        user_function_names,
                        errors,
                    );
                }
            }
        }
        Item::FieldDef(field) => {
            if let Some(measure) = &field.measure {
                check_expr_for_unknown_functions(
                    &measure.body,
                    user_functions,
                    user_function_names,
                    errors,
                );
            }
        }
        Item::OperatorDef(operator) => {
            if let Some(body) = &operator.body {
                let expr = match body {
                    crate::ast::OperatorBody::Warmup(e) => e,
                    crate::ast::OperatorBody::Collect(e) => e,
                    crate::ast::OperatorBody::Measure(e) => e,
                };
                check_expr_for_unknown_functions(expr, user_functions, user_function_names, errors);
            }
            if let Some(assertions) = &operator.assertions {
                for assertion in &assertions.assertions {
                    check_expr_for_unknown_functions(
                        &assertion.condition,
                        user_functions,
                        user_function_names,
                        errors,
                    );
                }
            }
        }
        Item::ImpulseDef(impulse) => {
            if let Some(apply) = &impulse.apply {
                check_expr_for_unknown_functions(
                    &apply.body,
                    user_functions,
                    user_function_names,
                    errors,
                );
            }
        }
        Item::FractureDef(fracture) => {
            for condition in &fracture.conditions {
                check_expr_for_unknown_functions(
                    condition,
                    user_functions,
                    user_function_names,
                    errors,
                );
            }
            if let Some(emit) = &fracture.emit {
                check_expr_for_unknown_functions(emit, user_functions, user_function_names, errors);
            }
        }
        Item::FnDef(f) => {
            check_expr_for_unknown_functions(&f.body, user_functions, user_function_names, errors);
        }
        Item::EntityDef(_) => {
            // Entities are pure index spaces - no expressions to validate
        }
        _ => {}
    }
}

/// Check if a function name is valid and return its expected arity.
fn get_expected_arity(name: &str, user_functions: &HashMap<String, usize>) -> Option<Arity> {
    if let Some((namespace, function)) = name.split_once('.') {
        if let Some(k) = get_in_namespace(namespace, function) {
            return Some(k.arity);
        }
    }

    // Check user-defined functions
    if let Some(&arity) = user_functions.get(name) {
        return Some(Arity::Fixed(arity));
    }
    None
}

/// Check an expression for unknown function calls and arity mismatches.
fn check_expr_for_unknown_functions(
    expr: &Spanned<Expr>,
    user_functions: &HashMap<String, usize>,
    user_function_names: &HashMap<String, usize>,
    errors: &mut Vec<ValidationError>,
) {
    let mut visitor = UnknownFunctionVisitor {
        user_functions,
        user_function_names,
        errors,
    };
    visitor.visit_expr(expr);
}

struct UnknownFunctionVisitor<'a> {
    user_functions: &'a HashMap<String, usize>,
    user_function_names: &'a HashMap<String, usize>,
    errors: &'a mut Vec<ValidationError>,
}

impl AstVisitor for UnknownFunctionVisitor<'_> {
    fn visit_expr(&mut self, expr: &Spanned<Expr>) {
        match &expr.node {
            Expr::Call { function, args } => {
                if let Expr::Path(path) = &function.node {
                    let name = path.to_string();
                    if let Some(expected) = get_expected_arity(&name, self.user_functions) {
                        match expected {
                            Arity::Fixed(n) if n != args.len() => {
                                self.errors.push(ValidationError {
                                    message: format!(
                                        "function '{}' expects {} arguments, got {}",
                                        name,
                                        n,
                                        args.len()
                                    ),
                                    span: expr.span.clone(),
                                });
                            }
                            _ => {}
                        }
                    } else {
                        self.errors.push(ValidationError {
                            message: format!("unknown function '{}'", name),
                            span: function.span.clone(),
                        });
                    }
                }
            }
            Expr::MethodCall { method, args, .. } => {
                // For method calls, the object is the first argument
                let total_args = args.len() + 1;
                if let Some(expected) = get_expected_arity(method, self.user_function_names) {
                    match expected {
                        Arity::Fixed(n) if n != total_args => {
                            self.errors.push(ValidationError {
                                message: format!(
                                    "method '{}' expects {} arguments, got {}",
                                    method, n, total_args
                                ),
                                span: expr.span.clone(),
                            });
                        }
                        _ => {}
                    }
                } else {
                    self.errors.push(ValidationError {
                        message: format!("unknown method '{}'", method),
                        span: expr.span.clone(),
                    });
                }
            }
            _ => {}
        }

        self.walk_expr(expr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    // Force linking of continuum-functions so built-in kernels are registered
    #[allow(unused_extern_crates)]
    extern crate continuum_functions;

    #[test]
    fn test_dt_raw_without_declaration() {
        let source = r#"
            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev + dt_raw
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert_eq!(errors.len(), 1);
        assert!(
            errors[0]
                .message
                .contains("uses dt_raw but does not declare : dt_raw")
        );
    }

    #[test]
    fn test_dt_raw_with_declaration() {
        let source = r#"
            signal.core.temp {
                : Scalar<K>
                : strata(thermal)
                : dt_raw

                resolve {
                    prev + dt_raw
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(errors.is_empty(), "validation errors: {:?}", errors);
    }

    #[test]
    fn test_no_dt_raw_usage() {
        // Using prev without dt_raw is fine
        let source = r#"
            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev * 0.99
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(errors.is_empty(), "validation errors: {:?}", errors);
    }

    #[test]
    fn test_unknown_method_call() {
        // Method calls should be validated against known functions
        let source = r#"
            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev.unknown_method()
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert_eq!(errors.len(), 1);
        assert!(
            errors[0]
                .message
                .contains("unknown method 'unknown_method'")
        );
    }

    #[test]
    fn test_known_method_call() {
        // User-defined functions should not error when called as methods
        let source = r#"
            fn.math.double(val) {
                val * 2.0
            }

            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev.double()
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(errors.is_empty(), "validation errors: {:?}", errors);
    }

    #[test]
    fn test_arity_mismatch_builtin() {
        // sin expects 1 argument
        let source = r#"
            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    sin(1.0, 2.0)
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(!errors.is_empty(), "expected validation errors");
        assert!(
            errors[0].message.contains("expects 1 arguments, got 2"),
            "got message: {}",
            errors[0].message
        );
    }

    #[test]
    fn test_arity_mismatch_user_fn() {
        let source = r#"
            fn.math.double(x) { x * 2.0 }

            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    math.double()
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("expects 1 arguments, got 0"));
    }

    #[test]
    fn test_arity_mismatch_method() {
        // clamp expects 3 arguments (object + 2 args)
        let source = r#"
            signal.core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev.clamp(0.0)
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(!errors.is_empty());
        assert!(
            errors[0]
                .message
                .contains("method 'clamp' expects 3 arguments, got 2")
        );
    }
}
