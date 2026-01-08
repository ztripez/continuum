//! Validation pass for Continuum DSL
//!
//! Performs semantic checks after parsing, including:
//! - dt_raw usage requires : dt_raw declaration on signal

use crate::ast::{CompilationUnit, Expr, Item};

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

    for item in &unit.items {
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
