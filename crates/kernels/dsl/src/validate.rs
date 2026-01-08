//! Validation pass for Continuum DSL
//!
//! Performs semantic checks after parsing, including:
//! - dt_raw usage requires : dt_raw declaration on signal
//! - function calls reference known kernel functions

use crate::ast::{CompilationUnit, Expr, Item, Spanned};

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
    pub span: std::ops::Range<usize>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {:?}", self.message, self.span)
    }
}

impl std::error::Error for ValidationError {}

/// Validate a compilation unit
pub fn validate(unit: &CompilationUnit) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Collect all user-defined function names
    let mut user_functions = std::collections::HashSet::new();
    for item in &unit.items {
        if let Item::FnDef(f) = &item.node {
            user_functions.insert(f.path.node.to_string());
        }
    }

    for item in &unit.items {
        // Check for unknown function calls in all expressions
        collect_unknown_functions(&item.node, &user_functions, &mut errors);

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
            _ => {}
        }
    }

    errors
}

/// Check all expressions in an item for unknown function calls
fn collect_unknown_functions(
    item: &Item,
    user_functions: &std::collections::HashSet<String>,
    errors: &mut Vec<ValidationError>,
) {
    match item {
        Item::SignalDef(signal) => {
            if let Some(resolve) = &signal.resolve {
                check_expr_for_unknown_functions(&resolve.body, user_functions, errors);
            }
            if let Some(warmup) = &signal.warmup {
                check_expr_for_unknown_functions(&warmup.iterate, user_functions, errors);
            }
        }
        Item::OperatorDef(op) => {
            if let Some(body) = &op.body {
                let expr = match body {
                    crate::ast::OperatorBody::Warmup(e) => e,
                    crate::ast::OperatorBody::Collect(e) => e,
                    crate::ast::OperatorBody::Measure(e) => e,
                };
                check_expr_for_unknown_functions(expr, user_functions, errors);
            }
        }
        Item::FieldDef(field) => {
            if let Some(measure) = &field.measure {
                check_expr_for_unknown_functions(&measure.body, user_functions, errors);
            }
        }
        Item::ImpulseDef(impulse) => {
            if let Some(apply) = &impulse.apply {
                check_expr_for_unknown_functions(&apply.body, user_functions, errors);
            }
        }
        Item::FractureDef(fracture) => {
            for condition in &fracture.conditions {
                check_expr_for_unknown_functions(condition, user_functions, errors);
            }
            for emit in &fracture.emit {
                check_expr_for_unknown_functions(&emit.value, user_functions, errors);
            }
        }
        Item::FnDef(f) => {
            check_expr_for_unknown_functions(&f.body, user_functions, errors);
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
    // Check user-defined functions
    if user_functions.contains(name) {
        return true;
    }
    false
}

/// Recursively check an expression for unknown function calls
fn check_expr_for_unknown_functions(
    expr: &Spanned<Expr>,
    user_functions: &std::collections::HashSet<String>,
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
            check_expr_for_unknown_functions(function, user_functions, errors);
            for arg in args {
                check_expr_for_unknown_functions(arg, user_functions, errors);
            }
        }
        Expr::Binary { left, right, .. } => {
            check_expr_for_unknown_functions(left, user_functions, errors);
            check_expr_for_unknown_functions(right, user_functions, errors);
        }
        Expr::Unary { operand, .. } => {
            check_expr_for_unknown_functions(operand, user_functions, errors);
        }
        Expr::FieldAccess { object, .. } => {
            check_expr_for_unknown_functions(object, user_functions, errors);
        }
        Expr::Let { value, body, .. } => {
            check_expr_for_unknown_functions(value, user_functions, errors);
            check_expr_for_unknown_functions(body, user_functions, errors);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            check_expr_for_unknown_functions(condition, user_functions, errors);
            check_expr_for_unknown_functions(then_branch, user_functions, errors);
            if let Some(e) = else_branch {
                check_expr_for_unknown_functions(e, user_functions, errors);
            }
        }
        Expr::For { iter, body, .. } => {
            check_expr_for_unknown_functions(iter, user_functions, errors);
            check_expr_for_unknown_functions(body, user_functions, errors);
        }
        Expr::Block(exprs) => {
            for e in exprs {
                check_expr_for_unknown_functions(e, user_functions, errors);
            }
        }
        Expr::EmitSignal { value, .. } => {
            check_expr_for_unknown_functions(value, user_functions, errors);
        }
        Expr::EmitField { position, value, .. } => {
            check_expr_for_unknown_functions(position, user_functions, errors);
            check_expr_for_unknown_functions(value, user_functions, errors);
        }
        Expr::Struct(fields) => {
            for (_, e) in fields {
                check_expr_for_unknown_functions(e, user_functions, errors);
            }
        }
        Expr::Map { sequence, function } => {
            check_expr_for_unknown_functions(sequence, user_functions, errors);
            check_expr_for_unknown_functions(function, user_functions, errors);
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            check_expr_for_unknown_functions(sequence, user_functions, errors);
            check_expr_for_unknown_functions(init, user_functions, errors);
            check_expr_for_unknown_functions(function, user_functions, errors);
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
        | Expr::SumInputs
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
        | Expr::SumInputs
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
}
