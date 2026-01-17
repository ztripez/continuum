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
//! 6. Power (`**`) - highest precedence (right-associative)
//!
//! ## Error Recovery
//!
//! Parser uses chumsky's error recovery to:
//! - Report multiple errors in a single pass
//! - Insert [`ExprKind::ParseError`] placeholders for missing expressions
//! - Continue parsing after syntax errors
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

use chumsky::prelude::*;

use crate::ast::{Expr, TypeExpr, UnitExpr, UntypedKind as ExprKind};
use crate::foundation::Span;
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
/// - `Err(Vec<ParseError>)`: Parse errors encountered
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
pub fn parse_expr(tokens: &[Token]) -> ParseResult<Expr, Rich<Token>> {
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
fn expr_parser<'src>()
-> impl Parser<'src, &'src [Token], Expr, extra::Err<Rich<'src, Token>>> + Clone {
    recursive(|_expr| {
        // TODO: Implement recursive expression parser
        //
        // Structure:
        // 1. Atoms (literals, identifiers, context values, parenthesized expressions)
        // 2. Postfix (field access, function calls)
        // 3. Unary (negation, logical not)
        // 4. Binary with precedence (power, mul/div/mod, add/sub, comparison, logical)
        // 5. Let bindings
        // 6. If expressions

        // Stub: parse any token as error placeholder
        any().map(|_| {
            Expr::new(
                ExprKind::ParseError("not yet implemented".to_string()),
                Span::new(0, 0, 0, 0),
            )
        })
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
    // TODO: Implement unit expression parser
    //
    // This is critical since units are NOT single tokens.
    // Must parse:
    // 1. Lt (`<`)
    // 2. Unit term (recursive: base, multiply, divide, power)
    // 3. Gt (`>`)
    //
    // Special case: `<>` (empty brackets) → Dimensionless

    // Stub: parse any identifier as base unit
    any().map(|_| UnitExpr::Dimensionless)
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
    // TODO: Implement type expression parser
    //
    // Handle:
    // 1. Scalar<unit>
    // 2. Vector<dim, unit>
    // 3. Matrix<rows, cols, unit>
    // 4. Bool
    // 5. User types (identifiers)

    // Stub: return Bool for any input
    any().map(|_| TypeExpr::Bool)
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
    #[ignore = "parser not yet implemented"]
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
    #[ignore = "parser not yet implemented"]
    fn test_parse_bool_literal() {
        let expr = lex_and_parse("true");
        assert!(matches!(expr.kind, ExprKind::BoolLiteral(true)));

        let expr = lex_and_parse("false");
        assert!(matches!(expr.kind, ExprKind::BoolLiteral(false)));
    }

    #[test]
    #[ignore = "parser not yet implemented"]
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
    #[ignore = "parser not yet implemented"]
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
    #[ignore = "parser not yet implemented"]
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
    #[ignore = "parser not yet implemented"]
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
}
