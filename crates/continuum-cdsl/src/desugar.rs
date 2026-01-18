//! Desugaring pass - converts syntax sugar to kernel calls
//!
//! This module implements the desugaring transformation that converts:
//! - Binary operators (`+`, `-`, `*`, `/`, etc.) → `maths.*` kernel calls
//! - Unary operators (`-`, `!`) → `maths.neg` / `logic.not` kernel calls
//! - Comparison operators (`<`, `>`, `==`, etc.) → `compare.*` kernel calls
//! - Logical operators (`&&`, `||`) → `logic.and` / `logic.or` kernel calls
//! - If-expressions (`if c { t } else { e }`) → `logic.select(c, t, e)` kernel call
//!
//! # Design
//!
//! Desugaring happens **before type resolution**. It transforms `Expr` (untyped AST)
//! into simpler `Expr` with only `Call` nodes for operations.
//!
//! This separation allows:
//! - Type resolution to work with a smaller set of expression variants
//! - Kernel registry to handle all operations uniformly
//! - New operators to be added via kernel signatures, not AST changes
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{Expr, BinaryOp};
//! use continuum_cdsl::desugar::desugar_expr;
//!
//! // Binary operator
//! let expr = Expr::binary(BinaryOp::Add, a, b, span);
//! let desugared = desugar_expr(expr);
//! // Now: Call { kernel: maths.add, args: [a, b] }
//!
//! // If-expression
//! let expr = Expr::if_then_else(cond, then_val, else_val, span);
//! let desugared = desugar_expr(expr);
//! // Now: Call { kernel: logic.select, args: [cond, then_val, else_val] }
//! ```
//!
//! # Operator Mappings
//!
//! ## Arithmetic
//!
//! | Syntax | Desugars To |
//! |--------|-------------|
//! | `a + b` | `maths.add(a, b)` |
//! | `a - b` | `maths.sub(a, b)` |
//! | `a * b` | `maths.mul(a, b)` |
//! | `a / b` | `maths.div(a, b)` |
//! | `a % b` | `maths.mod(a, b)` |
//! | `a ^ b` | `maths.pow(a, b)` |
//! | `-a` | `maths.neg(a)` |
//!
//! ## Comparison
//!
//! | Syntax | Desugars To |
//! |--------|-------------|
//! | `a == b` | `compare.eq(a, b)` |
//! | `a != b` | `compare.ne(a, b)` |
//! | `a < b` | `compare.lt(a, b)` |
//! | `a <= b` | `compare.le(a, b)` |
//! | `a > b` | `compare.gt(a, b)` |
//! | `a >= b` | `compare.ge(a, b)` |
//!
//! ## Logic
//!
//! | Syntax | Desugars To |
//! |--------|-------------|
//! | `a && b` | `logic.and(a, b)` |
//! | `a \|\| b` | `logic.or(a, b)` |
//! | `!a` | `logic.not(a)` |
//!
//! ## Control Flow
//!
//! | Syntax | Desugars To |
//! |--------|-------------|
//! | `if c { t } else { e }` | `logic.select(c, t, e)` |

use crate::ast::{BinaryOp, Expr, UnaryOp, UntypedKind as ExprKind};
use crate::foundation::{Path, Span};

/// Desugar an expression, converting operators to kernel calls
///
/// Recursively transforms:
/// - `Binary { op, left, right }` → `Call { kernel: op.kernel(), args: [left, right] }`
/// - `Unary { op, operand }` → `Call { kernel: op.kernel(), args: [operand] }`
/// - `If { condition, then_branch, else_branch }` → `Call { kernel: logic.select, args: [cond, then, else] }`
///
/// All other expression variants are recursively processed but not transformed.
///
/// # Examples
///
/// ```rust,ignore
/// let expr = Expr::binary(BinaryOp::Add, a, b, span);
/// let desugared = desugar_expr(expr);
/// // desugared.kind == ExprKind::Call { kernel: maths.add, args: [a, b] }
/// ```
/// Reconstruct leaf expression variants without transformation
fn passthrough(kind: ExprKind, span: Span) -> Expr {
    Expr { kind, span }
}

/// Desugar an expression, converting operators to kernel calls
///
/// Recursively transforms:
/// - `Binary { op, left, right }` → `Call { kernel: op.kernel(), args: [left, right] }`
/// - `Unary { op, operand }` → `Call { kernel: op.kernel(), args: [operand] }`
/// - `If { condition, then_branch, else_branch }` → `Call { kernel: logic.select, args: [cond, then, else] }`
///
/// All other expression variants are recursively processed but not transformed.
pub fn desugar_expr(expr: Expr) -> Expr {
    let span = expr.span;

    match expr.kind {
        // === Operators → Kernel Calls ===
        ExprKind::Binary { op, left, right } => {
            let left = Box::new(desugar_expr(*left));
            let right = Box::new(desugar_expr(*right));
            let kernel_id = op.kernel();
            Expr {
                kind: ExprKind::Call {
                    func: Path::from_str(&kernel_id.qualified_name()),
                    args: vec![*left, *right],
                },
                span,
            }
        }

        ExprKind::Unary { op, operand } => {
            let operand = Box::new(desugar_expr(*operand));
            let kernel_id = op.kernel();
            Expr {
                kind: ExprKind::Call {
                    func: Path::from_str(&kernel_id.qualified_name()),
                    args: vec![*operand],
                },
                span,
            }
        }

        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition = Box::new(desugar_expr(*condition));
            let then_branch = Box::new(desugar_expr(*then_branch));
            let else_branch = Box::new(desugar_expr(*else_branch));
            Expr {
                kind: ExprKind::Call {
                    func: Path::from_str("logic.select"),
                    args: vec![*condition, *then_branch, *else_branch],
                },
                span,
            }
        }

        // === Recursive Cases (no transformation, just recurse) ===
        ExprKind::Let { name, value, body } => Expr {
            kind: ExprKind::Let {
                name,
                value: Box::new(desugar_expr(*value)),
                body: Box::new(desugar_expr(*body)),
            },
            span,
        },

        ExprKind::Vector(elements) => Expr {
            kind: ExprKind::Vector(elements.into_iter().map(desugar_expr).collect()),
            span,
        },

        ExprKind::Call { func, args } => Expr {
            kind: ExprKind::Call {
                func,
                args: args.into_iter().map(desugar_expr).collect(),
            },
            span,
        },

        ExprKind::Aggregate {
            op,
            entity,
            binding,
            body,
        } => Expr {
            kind: ExprKind::Aggregate {
                op,
                entity,
                binding,
                body: Box::new(desugar_expr(*body)),
            },
            span,
        },

        ExprKind::Fold {
            entity,
            init,
            acc,
            elem,
            body,
        } => Expr {
            kind: ExprKind::Fold {
                entity,
                init: Box::new(desugar_expr(*init)),
                acc,
                elem,
                body: Box::new(desugar_expr(*body)),
            },
            span,
        },

        ExprKind::Struct { ty, fields } => Expr {
            kind: ExprKind::Struct {
                ty,
                fields: fields
                    .into_iter()
                    .map(|(name, expr)| (name, desugar_expr(expr)))
                    .collect(),
            },
            span,
        },

        ExprKind::FieldAccess { object, field } => Expr {
            kind: ExprKind::FieldAccess {
                object: Box::new(desugar_expr(*object)),
                field,
            },
            span,
        },

        // === Leaf Cases (no recursion needed) ===
        ExprKind::Literal { value, unit } => passthrough(ExprKind::Literal { value, unit }, span),
        ExprKind::BoolLiteral(value) => passthrough(ExprKind::BoolLiteral(value), span),
        ExprKind::Local(name) => passthrough(ExprKind::Local(name), span),
        ExprKind::Signal(path) => passthrough(ExprKind::Signal(path), span),
        ExprKind::Field(path) => passthrough(ExprKind::Field(path), span),
        ExprKind::Config(path) => passthrough(ExprKind::Config(path), span),
        ExprKind::Const(path) => passthrough(ExprKind::Const(path), span),
        ExprKind::Prev => passthrough(ExprKind::Prev, span),
        ExprKind::Current => passthrough(ExprKind::Current, span),
        ExprKind::Inputs => passthrough(ExprKind::Inputs, span),
        ExprKind::Dt => passthrough(ExprKind::Dt, span),
        ExprKind::Self_ => passthrough(ExprKind::Self_, span),
        ExprKind::Other => passthrough(ExprKind::Other, span),
        ExprKind::Payload => passthrough(ExprKind::Payload, span),
        ExprKind::ParseError(msg) => passthrough(ExprKind::ParseError(msg), span),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Span;

    fn make_span() -> Span {
        Span::new(0, 0, 10, 1)
    }

    fn make_literal(value: f64) -> Expr {
        Expr {
            kind: ExprKind::Literal { value, unit: None },
            span: make_span(),
        }
    }

    #[test]
    fn test_desugar_binary_add() {
        let left = make_literal(1.0);
        let right = make_literal(2.0);
        let expr = Expr::binary(BinaryOp::Add, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("maths.add"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_binary_mul() {
        let left = make_literal(3.0);
        let right = make_literal(4.0);
        let expr = Expr::binary(BinaryOp::Mul, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("maths.mul"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_comparison() {
        let left = make_literal(5.0);
        let right = make_literal(10.0);
        let expr = Expr::binary(BinaryOp::Lt, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("compare.lt"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_logical_and() {
        let left = make_literal(1.0);
        let right = make_literal(0.0);
        let expr = Expr::binary(BinaryOp::And, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("logic.and"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_unary_neg() {
        let operand = make_literal(5.0);
        let expr = Expr::unary(UnaryOp::Neg, operand, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("maths.neg"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_unary_not() {
        let operand = make_literal(1.0);
        let expr = Expr::unary(UnaryOp::Not, operand, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("logic.not"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_if_expression() {
        let condition = make_literal(1.0);
        let then_branch = make_literal(10.0);
        let else_branch = make_literal(20.0);
        let expr = Expr {
            kind: ExprKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            span: make_span(),
        };

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("logic.select"));
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_nested_operators() {
        // (a + b) * c
        let a = make_literal(1.0);
        let b = make_literal(2.0);
        let c = make_literal(3.0);

        let add = Expr::binary(BinaryOp::Add, a, b, make_span());
        let mul = Expr::binary(BinaryOp::Mul, add, c, make_span());

        let desugared = desugar_expr(mul);

        // Outer should be mul
        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_str("maths.mul"));
                assert_eq!(args.len(), 2);

                // First arg should be add
                match &args[0].kind {
                    ExprKind::Call { func, .. } => {
                        assert_eq!(*func, Path::from_str("maths.add"));
                    }
                    _ => panic!("Expected nested Call"),
                }
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_preserves_let_bindings() {
        let value = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let body = make_literal(3.0);

        let expr = Expr {
            kind: ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            span: make_span(),
        };

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Let { name, value, .. } => {
                assert_eq!(name, "x");
                // Value should be desugared to Call
                match value.kind {
                    ExprKind::Call { func, .. } => {
                        assert_eq!(func, Path::from_str("maths.add"));
                    }
                    _ => panic!("Expected Call in let value"),
                }
            }
            _ => panic!("Expected Let, got {:?}", desugared.kind),
        }
    }
}
