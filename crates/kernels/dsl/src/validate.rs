//! Semantic Validation for Continuum DSL.
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

use crate::ast::{CompilationUnit, Expr, Item, Spanned};

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
/// - Function calls must reference known functions
/// - `dt_raw` is not allowed in operators, impulses, or fractures
///
/// Returns a vector of validation errors. An empty vector means validation passed.
pub fn validate(unit: &CompilationUnit) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Collect all user-defined function names (full path and last segment for method calls)
    let mut user_functions = std::collections::HashSet::new();
    let mut user_function_names = std::collections::HashSet::new();
    for item in &unit.items {
        if let Item::FnDef(f) = &item.node {
            user_functions.insert(f.path.node.to_string());
            // Also store just the function name for method call matching
            if let Some(name) = f.path.node.segments.last() {
                user_function_names.insert(name.clone());
            }
        }
    }

    for item in &unit.items {
        // Check for unknown function calls in all expressions
        collect_unknown_functions(&item.node, &user_functions, &user_function_names, &mut errors);

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
            Item::OperatorDef(op) => {
                // Check operator body for dt_raw usage
                if let Some(body) = &op.body {
                    let expr = match body {
                        crate::ast::OperatorBody::Warmup(e) => e,
                        crate::ast::OperatorBody::Collect(e) => e,
                        crate::ast::OperatorBody::Measure(e) => e,
                    };
                    if uses_dt_raw(&expr.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "operator '{}' uses dt_raw which is not allowed in operators",
                                op.path.node
                            ),
                            span: expr.span.clone(),
                        });
                    }
                }
            }
            Item::ImpulseDef(impulse) => {
                // Check impulse apply block for dt_raw usage
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
                // Check fracture conditions for dt_raw usage
                for condition in &fracture.conditions {
                    if uses_dt_raw(&condition.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "fracture '{}' uses dt_raw which is not allowed in fractures",
                                fracture.path.node
                            ),
                            span: condition.span.clone(),
                        });
                    }
                }
            }
            Item::FieldDef(field) => {
                // Check field measure block for dt_raw usage
                if let Some(measure) = &field.measure {
                    if uses_dt_raw(&measure.body.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "field '{}' uses dt_raw which is not allowed in measure blocks",
                                field.path.node
                            ),
                            span: measure.body.span.clone(),
                        });
                    }
                }
            }
            Item::EntityDef(entity) => {
                // Check entity resolve block for dt_raw usage
                if let Some(resolve) = &entity.resolve {
                    if uses_dt_raw(&resolve.body.node) {
                        errors.push(ValidationError {
                            message: format!(
                                "entity '{}' uses dt_raw which is not allowed in entity resolve blocks",
                                entity.path.node
                            ),
                            span: resolve.body.span.clone(),
                        });
                    }
                }
                // Check entity field measure blocks
                for field in &entity.fields {
                    if let Some(measure) = &field.measure {
                        if uses_dt_raw(&measure.body.node) {
                            errors.push(ValidationError {
                                message: format!(
                                    "entity field '{}.{}' uses dt_raw which is not allowed in measure blocks",
                                    entity.path.node, field.name.node
                                ),
                                span: measure.body.span.clone(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    errors
}

/// Check all expressions in an item for unknown function calls
fn collect_unknown_functions(
    item: &Item,
    user_functions: &std::collections::HashSet<String>,
    user_function_names: &std::collections::HashSet<String>,
    errors: &mut Vec<ValidationError>,
) {
    match item {
        Item::SignalDef(signal) => {
            if let Some(resolve) = &signal.resolve {
                check_expr_for_unknown_functions(&resolve.body, user_functions, user_function_names, errors);
            }
            if let Some(warmup) = &signal.warmup {
                check_expr_for_unknown_functions(&warmup.iterate, user_functions, user_function_names, errors);
            }
        }
        Item::OperatorDef(op) => {
            if let Some(body) = &op.body {
                let expr = match body {
                    crate::ast::OperatorBody::Warmup(e) => e,
                    crate::ast::OperatorBody::Collect(e) => e,
                    crate::ast::OperatorBody::Measure(e) => e,
                };
                check_expr_for_unknown_functions(expr, user_functions, user_function_names, errors);
            }
        }
        Item::FieldDef(field) => {
            if let Some(measure) = &field.measure {
                check_expr_for_unknown_functions(&measure.body, user_functions, user_function_names, errors);
            }
        }
        Item::ImpulseDef(impulse) => {
            if let Some(apply) = &impulse.apply {
                check_expr_for_unknown_functions(&apply.body, user_functions, user_function_names, errors);
            }
        }
        Item::FractureDef(fracture) => {
            for condition in &fracture.conditions {
                check_expr_for_unknown_functions(condition, user_functions, user_function_names, errors);
            }
            for emit in &fracture.emit {
                check_expr_for_unknown_functions(&emit.value, user_functions, user_function_names, errors);
            }
        }
        Item::FnDef(f) => {
            check_expr_for_unknown_functions(&f.body, user_functions, user_function_names, errors);
        }
        Item::EntityDef(entity) => {
            if let Some(resolve) = &entity.resolve {
                check_expr_for_unknown_functions(&resolve.body, user_functions, user_function_names, errors);
            }
            for field in &entity.fields {
                if let Some(measure) = &field.measure {
                    check_expr_for_unknown_functions(&measure.body, user_functions, user_function_names, errors);
                }
            }
        }
        _ => {}
    }
}

/// Check if a function name is valid (either a kernel function or user-defined)
fn is_known_function(name: &str, user_functions: &std::collections::HashSet<String>) -> bool {
    // Check kernel registry
    if continuum_kernel_registry::is_known(name) {
        return true;
    }
    // Check user-defined functions (full path)
    if user_functions.contains(name) {
        return true;
    }
    false
}

/// Check if a method name is valid (kernel function or user-defined function name)
fn is_known_method(name: &str, user_function_names: &std::collections::HashSet<String>) -> bool {
    // Check kernel registry
    if continuum_kernel_registry::is_known(name) {
        return true;
    }
    // Check user-defined function names (just the name part, not full path)
    if user_function_names.contains(name) {
        return true;
    }
    false
}

/// Recursively check an expression for unknown function calls
fn check_expr_for_unknown_functions(
    expr: &Spanned<Expr>,
    user_functions: &std::collections::HashSet<String>,
    user_function_names: &std::collections::HashSet<String>,
    errors: &mut Vec<ValidationError>,
) {
    match &expr.node {
        Expr::Call { function, args } => {
            // Check if function is a path (direct function call)
            if let Expr::Path(path) = &function.node {
                let name = path.to_string();
                if !is_known_function(&name, user_functions) {
                    errors.push(ValidationError {
                        message: format!("unknown function '{}'", name),
                        span: function.span.clone(),
                    });
                }
            }
            // Recurse into function expression and arguments
            check_expr_for_unknown_functions(function, user_functions, user_function_names, errors);
            for arg in args {
                check_expr_for_unknown_functions(arg, user_functions, user_function_names, errors);
            }
        }
        Expr::MethodCall { object, method, args } => {
            // Method calls are lowered to Call with object as first arg
            // Validate method name against known functions and user function names
            if !is_known_method(method, user_function_names) {
                errors.push(ValidationError {
                    message: format!("unknown method '{}'", method),
                    span: expr.span.clone(),
                });
            }
            check_expr_for_unknown_functions(object, user_functions, user_function_names, errors);
            for arg in args {
                check_expr_for_unknown_functions(arg, user_functions, user_function_names, errors);
            }
        }
        Expr::Binary { left, right, .. } => {
            check_expr_for_unknown_functions(left, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(right, user_functions, user_function_names, errors);
        }
        Expr::Unary { operand, .. } => {
            check_expr_for_unknown_functions(operand, user_functions, user_function_names, errors);
        }
        Expr::FieldAccess { object, .. } => {
            check_expr_for_unknown_functions(object, user_functions, user_function_names, errors);
        }
        Expr::Let { value, body, .. } => {
            check_expr_for_unknown_functions(value, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(body, user_functions, user_function_names, errors);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            check_expr_for_unknown_functions(condition, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(then_branch, user_functions, user_function_names, errors);
            if let Some(e) = else_branch {
                check_expr_for_unknown_functions(e, user_functions, user_function_names, errors);
            }
        }
        Expr::For { iter, body, .. } => {
            check_expr_for_unknown_functions(iter, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(body, user_functions, user_function_names, errors);
        }
        Expr::Block(exprs) => {
            for e in exprs {
                check_expr_for_unknown_functions(e, user_functions, user_function_names, errors);
            }
        }
        Expr::EmitSignal { value, .. } => {
            check_expr_for_unknown_functions(value, user_functions, user_function_names, errors);
        }
        Expr::EmitField { position, value, .. } => {
            check_expr_for_unknown_functions(position, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(value, user_functions, user_function_names, errors);
        }
        Expr::Struct(fields) => {
            for (_, e) in fields {
                check_expr_for_unknown_functions(e, user_functions, user_function_names, errors);
            }
        }
        Expr::Map { sequence, function } => {
            check_expr_for_unknown_functions(sequence, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(function, user_functions, user_function_names, errors);
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            check_expr_for_unknown_functions(sequence, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(init, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(function, user_functions, user_function_names, errors);
        }
        // Entity expressions
        Expr::SelfField(_) | Expr::EntityRef(_) | Expr::Other(_) | Expr::Pairs(_) => {}
        Expr::EntityAccess { instance, .. } => {
            check_expr_for_unknown_functions(instance, user_functions, user_function_names, errors);
        }
        Expr::Aggregate { body, .. } => {
            check_expr_for_unknown_functions(body, user_functions, user_function_names, errors);
        }
        Expr::Filter { predicate, .. } => {
            check_expr_for_unknown_functions(predicate, user_functions, user_function_names, errors);
        }
        Expr::First { predicate, .. } => {
            check_expr_for_unknown_functions(predicate, user_functions, user_function_names, errors);
        }
        Expr::Nearest { position, .. } => {
            check_expr_for_unknown_functions(position, user_functions, user_function_names, errors);
        }
        Expr::Within {
            position, radius, ..
        } => {
            check_expr_for_unknown_functions(position, user_functions, user_function_names, errors);
            check_expr_for_unknown_functions(radius, user_functions, user_function_names, errors);
        }
        // These don't contain function calls
        Expr::Literal(_)
        | Expr::LiteralWithUnit { .. }
        | Expr::Path(_)
        | Expr::Prev
        | Expr::PrevField(_)
        | Expr::DtRaw
        | Expr::Payload
        | Expr::PayloadField(_)
        | Expr::SignalRef(_)
        | Expr::ConstRef(_)
        | Expr::ConfigRef(_)
        | Expr::FieldRef(_)
        | Expr::Collected
        | Expr::MathConst(_) => {}
    }
}

/// Check if an expression uses dt_raw
fn uses_dt_raw(expr: &Expr) -> bool {
    match expr {
        Expr::DtRaw => true,
        Expr::Binary { left, right, .. } => {
            uses_dt_raw(&left.node) || uses_dt_raw(&right.node)
        }
        Expr::Unary { operand, .. } => uses_dt_raw(&operand.node),
        Expr::Call { function, args } => {
            uses_dt_raw(&function.node) || args.iter().any(|a| uses_dt_raw(&a.node))
        }
        Expr::MethodCall { object, args, .. } => {
            uses_dt_raw(&object.node) || args.iter().any(|a| uses_dt_raw(&a.node))
        }
        Expr::FieldAccess { object, .. } => uses_dt_raw(&object.node),
        Expr::Let { value, body, .. } => {
            uses_dt_raw(&value.node) || uses_dt_raw(&body.node)
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            uses_dt_raw(&condition.node)
                || uses_dt_raw(&then_branch.node)
                || else_branch
                    .as_ref()
                    .map(|e| uses_dt_raw(&e.node))
                    .unwrap_or(false)
        }
        Expr::For { iter, body, .. } => {
            uses_dt_raw(&iter.node) || uses_dt_raw(&body.node)
        }
        Expr::Block(exprs) => exprs.iter().any(|e| uses_dt_raw(&e.node)),
        Expr::EmitSignal { value, .. } => uses_dt_raw(&value.node),
        Expr::EmitField { position, value, .. } => {
            uses_dt_raw(&position.node) || uses_dt_raw(&value.node)
        }
        Expr::Struct(fields) => fields.iter().any(|(_, e)| uses_dt_raw(&e.node)),
        Expr::Map { sequence, function } => {
            uses_dt_raw(&sequence.node) || uses_dt_raw(&function.node)
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            uses_dt_raw(&sequence.node)
                || uses_dt_raw(&init.node)
                || uses_dt_raw(&function.node)
        }
        // Entity expressions
        Expr::SelfField(_) | Expr::EntityRef(_) | Expr::Other(_) | Expr::Pairs(_) => false,
        Expr::EntityAccess { instance, .. } => uses_dt_raw(&instance.node),
        Expr::Aggregate { body, .. } => uses_dt_raw(&body.node),
        Expr::Filter { predicate, .. } => uses_dt_raw(&predicate.node),
        Expr::First { predicate, .. } => uses_dt_raw(&predicate.node),
        Expr::Nearest { position, .. } => uses_dt_raw(&position.node),
        Expr::Within {
            position, radius, ..
        } => uses_dt_raw(&position.node) || uses_dt_raw(&radius.node),
        // These don't contain dt_raw
        Expr::Literal(_)
        | Expr::LiteralWithUnit { .. }
        | Expr::Path(_)
        | Expr::Prev
        | Expr::PrevField(_)
        | Expr::Payload
        | Expr::PayloadField(_)
        | Expr::SignalRef(_)
        | Expr::ConstRef(_)
        | Expr::ConfigRef(_)
        | Expr::FieldRef(_)
        | Expr::Collected
        | Expr::MathConst(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

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
        assert!(errors[0].message.contains("uses dt_raw but does not declare : dt_raw"));
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
        assert!(errors[0].message.contains("unknown method 'unknown_method'"));
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
}
