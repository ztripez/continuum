//! Parser for Continuum DSL using chumsky combinators.
//!
//! This module implements a recursive descent parser that transforms a stream
//! of tokens from the lexer into an untyped AST ([`Expr`]).
//!
//! # Design Principles
//!
//! ## Token-Based Parsing
//!
//! The parser operates on [`Token`](crate::lexer::Token) streams, not source text.
//! This provides:
//! - Clean separation between lexing and parsing
//! - Structured error recovery
//! - Consistent span tracking
//!
//! ## Unit Expression Parsing
//!
//! Units are **NOT** lexed as single tokens. The lexer produces:
//! - `Lt` (`<`) - angle bracket start
//! - Identifier/operators (`m`, `/`, `^`, etc.)
//! - `Gt` (`>`) - angle bracket end
//!
//! The parser reconstructs unit expressions from these token sequences:
//! ```text
//! <m/s>  →  [Lt, Ident("m"), Slash, Ident("s"), Gt]  →  UnitExpr::Divide(Base("m"), Base("s"))
//! ```
//!
//! ## Operator Precedence
//!
//! Binary operators are parsed with precedence climbing:
//! 1. Logical OR (`||`) - lowest
//! 2. Logical AND (`&&`)
//! 3. Comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`)
//! 4. Addition/Subtraction (`+`, `-`)
//! 5. Multiplication/Division/Modulo (`*`, `/`, `%`)
//! 6. Power (`^`) - highest precedence (right-associative: `2 ^ 3 ^ 4` → `2 ^ (3 ^ 4)`)
//!
//! ## Error Recovery
//!
//! Error recovery is not implemented yet.
//! Most parse failures stop with an error rather than producing
//! [`ExprKind::ParseError`] placeholders.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::parser::parse_expr;
//! use continuum_cdsl::lexer::Token;
//!
//! let source = "10.0<m> + velocity.x";
//! let tokens: Vec<_> = Token::lexer(source).filter_map(|r| r.ok()).collect();
//! let expr = parse_expr(&tokens)?;
//! ```

use chumsky::error::EmptyErr;
use chumsky::prelude::*;

use crate::ast::{BinaryOp, Expr, TypeExpr, UnaryOp, UnitExpr, UntypedKind as ExprKind};
use crate::foundation::{Path, Span};
use crate::lexer::Token;

/// Parse an expression from a token stream.
///
/// # Parameters
///
/// - `tokens`: Slice of tokens to parse
///
/// # Returns
///
/// - `Ok(Expr)`: Successfully parsed expression
///
/// # Errors
///
/// Returns `Err(Vec<EmptyErr>)` when parsing fails. [`EmptyErr`] is a placeholder
/// error type from chumsky that carries no diagnostic information—no source spans,
/// no expected token lists, no error messages. The `Vec` length indicates the
/// number of parse failures encountered, but provides no details about what went
/// wrong or where.
///
/// To check for errors, use `.into_result().is_err()` on the returned [`ParseResult`].
///
/// TODO: Replace [`EmptyErr`] with [`Rich<Token>`](chumsky::error::Rich) for
/// meaningful diagnostics with source locations and expected token information.
///
/// # Examples
///
/// ```rust,ignore
/// let tokens = vec![
///     Token::Integer(10),
///     Token::Plus,
///     Token::Integer(20),
/// ];
/// let expr = parse_expr(&tokens)?;
/// ```
pub fn parse_expr(tokens: &[Token]) -> ParseResult<Expr, EmptyErr> {
    expr_parser().parse(tokens)
}

/// Main expression parser (recursive).
///
/// Parses the full expression grammar including:
/// - Literals (numeric, boolean, vector)
/// - Operators (binary, unary)
/// - Let bindings
/// - If expressions
/// - Function calls
/// - Field access
/// - Struct construction
fn expr_parser<'src>() -> impl Parser<'src, &'src [Token], Expr> + Clone {
    recursive(|expr| {
        // === Atoms ===

        // Boolean literals
        let bool_literal = select! {
            Token::True => ExprKind::BoolLiteral(true),
            Token::False => ExprKind::BoolLiteral(false),
        }
        .map_with(|kind, e| Expr::new(kind, token_span(e.span())));

        // Numeric literals (integers and floats)
        let numeric_literal = select! {
            Token::Integer(n) => n as f64,
            Token::Float(f) => f,
        }
        .then(
            // Optional unit: < unit_expr >
            just(Token::Lt)
                .ignore_then(unit_expr_parser())
                .then_ignore(just(Token::Gt))
                .or_not(),
        )
        .map_with(|(value, unit), e| {
            Expr::new(ExprKind::Literal { value, unit }, token_span(e.span()))
        });

        // Context values (prev, current, inputs, dt, self, other, payload)
        let context_value = select! {
            Token::Prev => ExprKind::Prev,
            Token::Current => ExprKind::Current,
            Token::Inputs => ExprKind::Inputs,
            Token::Dt => ExprKind::Dt,
            Token::Self_ => ExprKind::Self_,
            Token::Other => ExprKind::Other,
            Token::Payload => ExprKind::Payload,
        }
        .map_with(|kind, e| Expr::new(kind, token_span(e.span())));

        // Identifiers - can be called
        let identifier = select! {
            Token::Ident(name) => name,
        }
        .map_with(|name, e| Expr::new(ExprKind::Local(name), token_span(e.span())))
        .foldl(
            expr.clone()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .repeated(),
            |func_expr, args| {
                let span = func_expr.span;
                match &func_expr.kind {
                    ExprKind::Local(name) => {
                        let path = Path::from_str(name);
                        Expr::new(ExprKind::Call { func: path, args }, span)
                    }
                    ExprKind::Call { .. } => {
                        // Calling a call result - not supported yet
                        Expr::new(
                            ExprKind::ParseError("nested function calls not supported".to_string()),
                            span,
                        )
                    }
                    _ => unreachable!("identifier foldl should only produce Local or Call"),
                }
            },
        );

        // Parenthesized expressions
        let parens = expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Vector literals: [expr, expr, ...]
        let vector_literal = expr
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with(|elements, e| Expr::new(ExprKind::Vector(elements), token_span(e.span())));

        // Atom: any of the above
        let atom = choice((
            bool_literal,
            numeric_literal,
            context_value,
            vector_literal,
            parens,
            identifier,
        ));

        // === Postfix (field access only - calls handled above) ===

        // Field access: atom.field
        let postfix = atom.foldl(
            just(Token::Dot)
                .ignore_then(select! { Token::Ident(name) => name })
                .repeated(),
            |object, field| {
                let span = object.span;
                Expr::new(
                    ExprKind::FieldAccess {
                        object: Box::new(object),
                        field,
                    },
                    span,
                )
            },
        );

        // === Unary operators ===

        let unary = choice((
            just(Token::Minus).to(UnaryOp::Neg),
            just(Token::Bang).to(UnaryOp::Not),
        ))
        .repeated()
        .foldr(postfix, |op, operand| {
            let span = operand.span;
            Expr::new(
                ExprKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
            )
        });

        // === Binary operators with precedence ===

        // Power (highest precedence, right-associative)
        // For `a ^ b ^ c`, parse as `a ^ (b ^ c)`
        let power_op = just(Token::Caret).to(BinaryOp::Pow);
        let power = unary
            .clone()
            .then(power_op.then(unary.clone()).repeated().collect::<Vec<_>>())
            .map(|(first, rest)| {
                if rest.is_empty() {
                    return first;
                }

                // Helper to build power binary node (eliminates duplication)
                let make_pow = |left: Expr, right: Expr| {
                    let span = left.span;
                    Expr::new(
                        ExprKind::Binary {
                            op: BinaryOp::Pow,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                };

                // Build right-to-left using iterator fold
                // For `2 ^ 3 ^ 4`, rest = [(Pow, 3), (Pow, 4)]
                // Reverse to [(Pow, 4), (Pow, 3)], start with 4, fold left with 3
                let mut iter = rest.into_iter().rev();
                let (_, rightmost) = iter.next().unwrap();

                let right_tree = iter.fold(rightmost, |acc, (_, operand)| make_pow(operand, acc));

                make_pow(first, right_tree)
            });

        // Multiplication, division, modulo
        let mul_op = choice((
            just(Token::Star).to(BinaryOp::Mul),
            just(Token::Slash).to(BinaryOp::Div),
            just(Token::Percent).to(BinaryOp::Mod),
        ));
        let mul =
            power
                .clone()
                .foldl_with(mul_op.then(power).repeated(), |left, (op, right), _e| {
                    let span = left.span;
                    Expr::new(
                        ExprKind::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                });

        // Addition, subtraction
        let add_op = choice((
            just(Token::Plus).to(BinaryOp::Add),
            just(Token::Minus).to(BinaryOp::Sub),
        ));
        let add = mul
            .clone()
            .foldl_with(add_op.then(mul).repeated(), |left, (op, right), _e| {
                let span = left.span;
                Expr::new(
                    ExprKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            });

        // Comparison operators
        let cmp_op = choice((
            just(Token::Lt).to(BinaryOp::Lt),
            just(Token::LtEq).to(BinaryOp::Le),
            just(Token::Gt).to(BinaryOp::Gt),
            just(Token::GtEq).to(BinaryOp::Ge),
            just(Token::EqEq).to(BinaryOp::Eq),
            just(Token::BangEq).to(BinaryOp::Ne),
        ));
        let comparison =
            add.clone()
                .foldl_with(cmp_op.then(add).repeated(), |left, (op, right), _e| {
                    let span = left.span;
                    Expr::new(
                        ExprKind::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                });

        // Logical AND
        let and = comparison.clone().foldl_with(
            just(Token::AndAnd)
                .to(BinaryOp::And)
                .then(comparison)
                .repeated(),
            |left, (op, right), _e| {
                let span = left.span;
                Expr::new(
                    ExprKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        // Logical OR (lowest precedence)
        let or = and.clone().foldl_with(
            just(Token::OrOr).to(BinaryOp::Or).then(and).repeated(),
            |left, (op, right), _e| {
                let span = left.span;
                Expr::new(
                    ExprKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        // TODO: Let bindings
        // TODO: If expressions
        // TODO: Struct construction
        // TODO: Aggregates (sum, map, fold)

        // Return the lowest precedence parser (logical OR)
        or
    })
}

/// Parse a unit expression from angle bracket sequence.
///
/// Units are lexed as token sequences: `<`, content, `>`
///
/// # Grammar
///
/// ```text
/// unit_expr := '<' unit_term '>'
/// unit_term := base_unit
///            | unit_term '*' unit_term
///            | unit_term '/' unit_term
///            | unit_term '^' integer
///            | '(' unit_term ')'
/// base_unit := identifier
/// ```
///
/// # Examples
///
/// ```text
/// <m>       → Base("m")
/// <m/s>     → Divide(Base("m"), Base("s"))
/// <kg*m^2>  → Multiply(Base("kg"), Power(Base("m"), 2))
/// <>        → Dimensionless
/// ```
///
/// # Parameters
///
/// - `tokens`: Token stream positioned at `<`
///
/// # Returns
///
/// Parsed unit expression or error.
fn unit_expr_parser<'src>() -> impl Parser<'src, &'src [Token], UnitExpr> + Clone {
    recursive(|unit_term| {
        // Base unit: identifier
        let base = select! {
            Token::Ident(name) => UnitExpr::Base(name),
        };

        // Parenthesized unit
        let parens = unit_term
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Primary: base or parens
        let primary = choice((base, parens));

        // Power: primary ^ integer
        let power = primary
            .then(
                just(Token::Caret)
                    .ignore_then(select! {
                        Token::Integer(n) => n as i8,
                    })
                    .or_not(),
            )
            .map(|(base_unit, exp)| {
                if let Some(exp) = exp {
                    UnitExpr::Power(Box::new(base_unit), exp)
                } else {
                    base_unit
                }
            });

        // Multiply/Divide (left-associative)
        let mul_div = power.clone().foldl(
            choice((
                just(Token::Star).to(true),   // true = multiply
                just(Token::Slash).to(false), // false = divide
            ))
            .then(power)
            .repeated(),
            |left, (is_mul, right)| {
                if is_mul {
                    UnitExpr::Multiply(Box::new(left), Box::new(right))
                } else {
                    UnitExpr::Divide(Box::new(left), Box::new(right))
                }
            },
        );

        mul_div
    })
}

/// Parse a type expression.
///
/// # Grammar
///
/// ```text
/// type_expr := 'Scalar' [ '<' unit_expr '>' ]
///            | 'Vector' '<' integer ',' unit_expr '>'
///            | 'Matrix' '<' integer ',' integer ',' unit_expr '>'
///            | 'Bool'
///            | identifier
/// ```
///
/// # Examples
///
/// ```text
/// Scalar<m>           → Scalar { unit: Some(Base("m")) }
/// Vector<3, m/s>      → Vector { dim: 3, unit: Some(Divide(...)) }
/// Matrix<3, 3, kg>    → Matrix { rows: 3, cols: 3, unit: Some(Base("kg")) }
/// Bool                → Bool
/// OrbitalElements     → User(Path(...))
/// ```
fn type_expr_parser<'src>() -> impl Parser<'src, &'src [Token], TypeExpr> + Clone {
    // Bool type
    let bool_type = just(Token::Ident("Bool".to_string())).to(TypeExpr::Bool);

    // Scalar<unit>
    let scalar_type = just(Token::Ident("Scalar".to_string()))
        .then(
            just(Token::Lt)
                .ignore_then(unit_expr_parser())
                .then_ignore(just(Token::Gt))
                .or_not(),
        )
        .map(|(_, unit)| TypeExpr::Scalar { unit });

    // Vector<dim, unit>
    let vector_type = just(Token::Ident("Vector".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|(dim, unit)| TypeExpr::Vector {
            dim,
            unit: Some(unit),
        });

    // Matrix<rows, cols, unit>
    let matrix_type = just(Token::Ident("Matrix".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|((rows, cols), unit)| TypeExpr::Matrix {
            rows,
            cols,
            unit: Some(unit),
        });

    // User types (identifiers)
    let user_type = select! {
        Token::Ident(name) => TypeExpr::User(Path::from_str(&name)),
    };

    choice((bool_type, scalar_type, vector_type, matrix_type, user_type))
}

/// Helper to convert chumsky span to Continuum Span.
///
/// # Parameters
///
/// - `span`: Chumsky simple span (start, end offsets)
///
/// # Returns
///
/// Continuum [`Span`] with file ID and line information.
///
/// # Implementation Note
///
/// Currently returns a placeholder span. Full implementation needs:
/// - File ID mapping
/// - Line number calculation from byte offsets
/// - Source map integration
fn token_span(span: SimpleSpan) -> Span {
    // TODO: Implement proper span conversion
    //
    // Needs:
    // - File ID from context
    // - Line number lookup from byte offset
    // - SourceMap integration

    Span::new(0, span.start as u32, span.end as u32, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Token;
    use logos::Logos;

    /// Helper to lex and parse a source string.
    ///
    /// # Parameters
    ///
    /// - `source`: CDSL source code
    ///
    /// # Returns
    ///
    /// Parsed expression or panic on error.
    fn lex_and_parse(source: &str) -> Expr {
        let tokens: Vec<_> = Token::lexer(source).filter_map(|r| r.ok()).collect();
        parse_expr(&tokens).unwrap()
    }

    #[test]
    fn test_parse_literal() {
        let expr = lex_and_parse("42.0");
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 42.0);
                assert_eq!(unit, None);
            }
            _ => panic!("expected literal, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_bool_literal() {
        let expr = lex_and_parse("true");
        assert!(matches!(expr.kind, ExprKind::BoolLiteral(true)));

        let expr = lex_and_parse("false");
        assert!(matches!(expr.kind, ExprKind::BoolLiteral(false)));
    }

    #[test]
    fn test_parse_binary_add() {
        let expr = lex_and_parse("10 + 20");
        match expr.kind {
            ExprKind::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert!(matches!(left.kind, ExprKind::Literal { value: 10.0, .. }));
                assert!(matches!(right.kind, ExprKind::Literal { value: 20.0, .. }));
            }
            _ => panic!("expected binary, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_unit_meters() {
        // Units are token sequences: Lt, Ident("m"), Gt
        let expr = lex_and_parse("100.0<m>");
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 100.0);
                assert_eq!(unit, Some(UnitExpr::Base("m".to_string())));
            }
            _ => panic!("expected literal with unit, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_unit_velocity() {
        // <m/s> → Lt, Ident("m"), Slash, Ident("s"), Gt
        let expr = lex_and_parse("10.0<m/s>");
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 10.0);
                match unit {
                    Some(UnitExpr::Divide(num, denom)) => {
                        assert_eq!(*num, UnitExpr::Base("m".to_string()));
                        assert_eq!(*denom, UnitExpr::Base("s".to_string()));
                    }
                    _ => panic!("expected divide unit, got {:?}", unit),
                }
            }
            _ => panic!("expected literal with unit, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_unary_neg() {
        let expr = lex_and_parse("-42.0");
        match expr.kind {
            ExprKind::Unary { op, operand } => {
                assert_eq!(op, UnaryOp::Neg);
                assert!(matches!(
                    operand.kind,
                    ExprKind::Literal { value: 42.0, .. }
                ));
            }
            _ => panic!("expected unary, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_precedence_mul_over_add() {
        let expr = lex_and_parse("1 + 2 * 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 1.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("expected add with mul on right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_right_associative() {
        let expr = lex_and_parse("2 ^ 3 ^ 4");
        // Should parse as 2 ^ (3 ^ 4) for right-associativity
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("expected pow chain, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_single() {
        // Single power should just work
        let expr = lex_and_parse("2 ^ 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(right.kind, ExprKind::Literal { value: 3.0, .. }));
            }
            _ => panic!("expected single pow, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_vs_mul_precedence() {
        // Power should bind tighter than multiplication
        // 2 * 3 ^ 4 should parse as 2 * (3 ^ 4)
        let expr = lex_and_parse("2 * 3 ^ 4");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("expected mul with pow on right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_vs_mul_precedence_left() {
        // Power on left: 2 ^ 3 * 4 should parse as (2 ^ 3) * 4
        let expr = lex_and_parse("2 ^ 3 * 4");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 4.0, .. }));
            }
            _ => panic!("expected mul with pow on left, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_parentheses_override() {
        // Parentheses should override right-associativity
        // (2 ^ 3) ^ 4 should parse as left-associative Binary(Binary(2, 3), 4)
        let expr = lex_and_parse("(2 ^ 3) ^ 4");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 4.0, .. }));
            }
            _ => panic!("expected pow with pow on left, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_with_unary_right() {
        // Power with unary on right: 2 ^ -3
        let expr = lex_and_parse("2 ^ -3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("expected pow with unary right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_with_unary_left() {
        // Power with unary on left: -2 ^ 3
        // Unary binds to its operand first, so this is (-2) ^ 3
        let expr = lex_and_parse("-2 ^ 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 3.0, .. }));
            }
            _ => panic!("expected pow with unary left, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_longer_chain() {
        // Longer chain: 2 ^ 3 ^ 4 ^ 5 should be 2 ^ (3 ^ (4 ^ 5))
        let expr = lex_and_parse("2 ^ 3 ^ 4 ^ 5");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                // right should be 3 ^ (4 ^ 5)
                match right.kind {
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        left: inner_left,
                        right: inner_right,
                    } => {
                        assert!(matches!(
                            inner_left.kind,
                            ExprKind::Literal { value: 3.0, .. }
                        ));
                        // inner_right should be 4 ^ 5 with full leaves verified
                        match inner_right.kind {
                            ExprKind::Binary {
                                op: BinaryOp::Pow,
                                left: leaf_left,
                                right: leaf_right,
                            } => {
                                assert!(matches!(
                                    leaf_left.kind,
                                    ExprKind::Literal { value: 4.0, .. }
                                ));
                                assert!(matches!(
                                    leaf_right.kind,
                                    ExprKind::Literal { value: 5.0, .. }
                                ));
                            }
                            _ => panic!("expected innermost pow, got {:?}", inner_right.kind),
                        }
                    }
                    _ => panic!("expected nested pow, got {:?}", right.kind),
                }
            }
            _ => panic!("expected pow chain, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_parenthesized_right() {
        // Explicit parentheses on right: 2 ^ (3 ^ 4)
        let expr = lex_and_parse("2 ^ (3 ^ 4)");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                match right.kind {
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        left: rleft,
                        right: rright,
                    } => {
                        assert!(matches!(rleft.kind, ExprKind::Literal { value: 3.0, .. }));
                        assert!(matches!(rright.kind, ExprKind::Literal { value: 4.0, .. }));
                    }
                    _ => panic!("expected parenthesized pow on right, got {:?}", right.kind),
                }
            }
            _ => panic!("expected pow with parenthesized right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_vs_comparison() {
        // Power should bind tighter than comparison: 1 < 2 ^ 3 is 1 < (2 ^ 3)
        let expr = lex_and_parse("1 < 2 ^ 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Lt,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 1.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("expected comparison with pow on right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_missing_rhs() {
        // Missing right operand should fail
        let tokens: Vec<_> = Token::lexer("2 ^").filter_map(|r| r.ok()).collect();
        assert!(parse_expr(&tokens).into_result().is_err());
    }

    #[test]
    fn test_power_missing_lhs() {
        // Missing left operand should fail
        let tokens: Vec<_> = Token::lexer("^ 2").filter_map(|r| r.ok()).collect();
        assert!(parse_expr(&tokens).into_result().is_err());
    }

    #[test]
    fn test_power_double_operator() {
        // Double operator should fail
        let tokens: Vec<_> = Token::lexer("2 ^ ^ 3").filter_map(|r| r.ok()).collect();
        assert!(parse_expr(&tokens).into_result().is_err());
    }

    #[test]
    fn test_postfix_call_then_field() {
        let expr = lex_and_parse("f(1).x");
        match expr.kind {
            ExprKind::FieldAccess { object, field } => {
                assert_eq!(field, "x");
                assert!(matches!(object.kind, ExprKind::Call { .. }));
            }
            _ => panic!("expected field access on call, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_vector_empty() {
        let expr = lex_and_parse("[]");
        match expr.kind {
            ExprKind::Vector(elements) => {
                assert_eq!(elements.len(), 0);
            }
            _ => panic!("expected empty vector, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_vector_trailing_comma() {
        let expr = lex_and_parse("[1.0, 2.0, 3.0,]");
        match expr.kind {
            ExprKind::Vector(elements) => {
                assert_eq!(elements.len(), 3);
            }
            _ => panic!("expected vector, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_unary_chaining() {
        let expr = lex_and_parse("!-42.0");
        match expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                assert!(matches!(
                    operand.kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("expected unary chain, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parenthesis_override() {
        let expr = lex_and_parse("(1 + 2) * 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 3.0, .. }));
            }
            _ => panic!("expected mul with add on left, got {:?}", expr.kind),
        }
    }
}
