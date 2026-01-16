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

                    // Check if kernel functions requiring uses() are called without declaration
                    check_requires_uses(
                        &resolve.body,
                        &signal.uses,
                        &signal.path.node.to_string(),
                        &mut errors,
                    );
                }
            }
            Item::MemberDef(member) => {
                // Check if dt_raw is used in resolve block without : dt_raw declaration
                if let Some(resolve) = &member.resolve {
                    if uses_dt_raw(&resolve.body.node) && !member.dt_raw {
                        errors.push(ValidationError {
                            message: format!(
                                "member '{}' uses dt_raw but does not declare : dt_raw",
                                member.path.node
                            ),
                            span: resolve.body.span.clone(),
                        });
                    }

                    // Check if kernel functions requiring uses() are called without declaration
                    check_requires_uses(
                        &resolve.body,
                        &member.uses,
                        &member.path.node.to_string(),
                        &mut errors,
                    );
                }
                // Also check initial block
                if let Some(initial) = &member.initial {
                    if uses_dt_raw(&initial.body.node) && !member.dt_raw {
                        errors.push(ValidationError {
                            message: format!(
                                "member '{}' uses dt_raw in initial block but does not declare : dt_raw",
                                member.path.node
                            ),
                            span: initial.body.span.clone(),
                        });
                    }

                    check_requires_uses(
                        &initial.body,
                        &member.uses,
                        &member.path.node.to_string(),
                        &mut errors,
                    );
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

                    // Check if kernel functions requiring uses() are called without declaration
                    check_requires_uses(
                        expr,
                        &operator.uses,
                        &operator.path.node.to_string(),
                        &mut errors,
                    );
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

                    // Check if kernel functions requiring uses() are called without declaration
                    check_requires_uses(
                        &apply.body,
                        &impulse.uses,
                        &impulse.path.node.to_string(),
                        &mut errors,
                    );
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

                    // Check if kernel functions requiring uses() are called without declaration
                    check_requires_uses(
                        condition,
                        &fracture.uses,
                        &fracture.path.node.to_string(),
                        &mut errors,
                    );
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

                    // Check if kernel functions requiring uses() are called without declaration
                    check_requires_uses(
                        emit,
                        &fracture.uses,
                        &fracture.path.node.to_string(),
                        &mut errors,
                    );
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

/// Check if expression calls kernel functions that require uses() declarations
fn check_requires_uses(
    expr: &Spanned<Expr>,
    uses_declarations: &[String],
    context_name: &str,
    errors: &mut Vec<ValidationError>,
) {
    match &expr.node {
        Expr::Call { function, args, .. } => {
            // Check if this is a namespaced function call (e.g., maths.clamp)
            if let Expr::Path(path) = &function.node {
                if path.segments.len() >= 2 {
                    let namespace = &path.segments[0];
                    let func_name = path.segments.last().unwrap();

                    // Look up kernel descriptor
                    if let Some(descriptor) =
                        continuum_kernel_registry::get_in_namespace(namespace, func_name)
                    {
                        if let Some(requires) = descriptor.requires_uses {
                            // Build full uses key: namespace.key
                            let full_key = format!("{}.{}", namespace, requires.key);

                            // Check if signal/member has this uses declaration
                            if !uses_declarations.contains(&full_key) {
                                errors.push(ValidationError {
                                    message: format!(
                                        "{} uses {}.{} which requires : uses({}). {}",
                                        context_name, namespace, func_name, full_key, requires.hint
                                    ),
                                    span: expr.span.clone(),
                                });
                            }
                        }
                    }
                }
            }

            // Recursively check arguments
            for arg in args {
                check_requires_uses(&arg.value, uses_declarations, context_name, errors);
            }
        }
        Expr::Binary { left, right, .. } => {
            check_requires_uses(left, uses_declarations, context_name, errors);
            check_requires_uses(right, uses_declarations, context_name, errors);
        }
        Expr::Unary { operand, .. } => {
            check_requires_uses(operand, uses_declarations, context_name, errors);
        }
        Expr::FieldAccess { object, .. } => {
            check_requires_uses(object, uses_declarations, context_name, errors);
        }
        Expr::Let { value, body, .. } => {
            check_requires_uses(value, uses_declarations, context_name, errors);
            check_requires_uses(body, uses_declarations, context_name, errors);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            check_requires_uses(condition, uses_declarations, context_name, errors);
            check_requires_uses(then_branch, uses_declarations, context_name, errors);
            if let Some(else_b) = else_branch {
                check_requires_uses(else_b, uses_declarations, context_name, errors);
            }
        }
        Expr::For { iter, body, .. } => {
            check_requires_uses(iter, uses_declarations, context_name, errors);
            check_requires_uses(body, uses_declarations, context_name, errors);
        }
        Expr::MethodCall { object, args, .. } => {
            check_requires_uses(object, uses_declarations, context_name, errors);
            for arg in args {
                check_requires_uses(&arg.value, uses_declarations, context_name, errors);
            }
        }
        Expr::Aggregate { body, .. } => {
            check_requires_uses(body, uses_declarations, context_name, errors);
        }
        _ => {}
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
            signal core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev + dt.raw
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
            signal core.temp {
                : Scalar<K>
                : strata(thermal)
                : uses(dt.raw)

                resolve {
                    prev + dt.raw
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
            signal core.temp {
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
            signal core.temp {
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
            fn math.double(val) {
                val * 2.0
            }

            signal core.temp {
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
            signal core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    maths.sin(1.0, 2.0)
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
            fn math.double(x) { x * 2.0 }

            signal core.temp {
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
        // my_clamp expects 3 arguments (object + 2 args)
        let source = r#"
            fn maths.my_clamp(val, min, max) { val }

            signal core.temp {
                : Scalar<K>
                : strata(thermal)

                resolve {
                    prev.my_clamp(0.0)
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
                .contains("method 'my_clamp' expects 3 arguments, got 2")
        );
    }

    #[test]
    fn test_clamp_without_uses_declaration() {
        let source = r#"
            signal test.value {
                : Scalar<K>
                resolve {
                    maths.clamp(prev, 0.0, 100.0)
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert_eq!(errors.len(), 1, "Expected 1 validation error");
        assert!(
            errors[0].message.contains("uses maths.clamp"),
            "Error should mention maths.clamp usage"
        );
        assert!(
            errors[0].message.contains("uses(maths.clamping)"),
            "Error should mention required declaration"
        );
    }

    #[test]
    fn test_clamp_with_uses_declaration() {
        let source = r#"
            signal test.value {
                : Scalar<K>
                : uses(maths.clamping)
                resolve {
                    maths.clamp(prev, 0.0, 100.0)
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(
            errors.is_empty(),
            "Should have no validation errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_saturate_without_uses_declaration() {
        let source = r#"
            signal test.value {
                : Scalar<1>
                resolve {
                    maths.saturate(prev + 0.1)
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert_eq!(errors.len(), 1, "Expected 1 validation error");
        assert!(
            errors[0].message.contains("uses maths.saturate"),
            "Error should mention maths.saturate usage"
        );
    }

    #[test]
    fn test_saturate_with_uses_declaration() {
        let source = r#"
            signal test.value {
                : Scalar<1>
                : uses(maths.clamping)
                resolve {
                    maths.saturate(prev + 0.1)
                }
            }
        "#;
        let (result, parse_errors) = parse(source);
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);
        let unit = result.unwrap();
        let errors = validate(&unit);
        assert!(
            errors.is_empty(),
            "Should have no validation errors: {:?}",
            errors
        );
    }
}

#[test]
fn test_fracture_clamp_without_uses() {
    let source = r#"
        fracture test.fracture {
            : strata(test)
            when { maths.clamp(signal.stress, 0.0, 100.0) > 50.0 }
            emit {
                signal.output <- 1.0
            }
        }
    "#;
    let (unit, parse_errors) = crate::parse(source);
    assert!(parse_errors.is_empty());
    let errors = validate(&unit.unwrap());
    assert!(
        errors.len() == 1,
        "Expected 1 error, got {}: {:?}",
        errors.len(),
        errors
    );
    assert!(errors[0].message.contains("uses maths.clamp"));
    assert!(
        errors[0]
            .message
            .contains("requires : uses(maths.clamping)")
    );
}

#[test]
fn test_fracture_clamp_with_uses() {
    let source = r#"
        fracture test.fracture {
            : strata(test)
            : uses(maths.clamping)
            when { maths.clamp(signal.stress, 0.0, 100.0) > 50.0 }
            emit {
                signal.output <- 1.0
            }
        }
    "#;
    let (unit, parse_errors) = crate::parse(source);
    assert!(parse_errors.is_empty());
    let errors = validate(&unit.unwrap());
    assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
}

#[test]
fn test_operator_clamp_without_uses() {
    let source = r#"
        operator test.operator {
            : strata(test)
            : phase(collect)
            collect {
                maths.clamp(signal.value, 0.0, 1.0)
            }
        }
    "#;
    let (unit, parse_errors) = crate::parse(source);
    assert!(parse_errors.is_empty());
    let errors = validate(&unit.unwrap());
    assert!(
        errors.len() == 1,
        "Expected 1 error, got {}: {:?}",
        errors.len(),
        errors
    );
    assert!(errors[0].message.contains("uses maths.clamp"));
    assert!(
        errors[0]
            .message
            .contains("requires : uses(maths.clamping)")
    );
}

#[test]
fn test_operator_clamp_with_uses() {
    let source = r#"
        operator test.operator {
            : strata(test)
            : phase(collect)
            : uses(maths.clamping)
            collect {
                maths.clamp(signal.value, 0.0, 1.0)
            }
        }
    "#;
    let (unit, parse_errors) = crate::parse(source);
    assert!(parse_errors.is_empty());
    let errors = validate(&unit.unwrap());
    assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
}

#[test]
fn test_impulse_clamp_without_uses() {
    let source = r#"
        impulse test.impulse {
            : Scalar<1>
            apply {
                let x = maths.clamp(payload, 0.0, 1.0) in
                signal.output <- x
            }
        }
    "#;
    let (unit, parse_errors) = crate::parse(source);
    assert!(parse_errors.is_empty());
    let errors = validate(&unit.unwrap());
    assert!(
        errors.len() == 1,
        "Expected 1 error, got {}: {:?}",
        errors.len(),
        errors
    );
    assert!(errors[0].message.contains("uses maths.clamp"));
    assert!(
        errors[0]
            .message
            .contains("requires : uses(maths.clamping)")
    );
}

#[test]
fn test_impulse_clamp_with_uses() {
    let source = r#"
        impulse test.impulse {
            : Scalar<1>
            : uses(maths.clamping)
            apply {
                let x = maths.clamp(payload, 0.0, 1.0) in
                signal.output <- x
            }
        }
    "#;
    let (unit, parse_errors) = crate::parse(source);
    assert!(parse_errors.is_empty());
    let errors = validate(&unit.unwrap());
    assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
}
