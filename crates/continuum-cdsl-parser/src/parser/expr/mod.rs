//! Expression parser using Pratt parsing (precedence climbing).
//!
//! This module implements a Pratt parser for CDSL expressions with proper
//! operator precedence and associativity.
//!
//! ## Precedence Levels (lowest to highest)
//!
//! 1. `||` (Or) - left associative
//! 2. `&&` (And) - left associative
//! 3. `==`, `!=`, `<`, `<=`, `>`, `>=` (Comparison) - left associative
//! 4. `+`, `-` (Addition) - left associative
//! 5. `*`, `/`, `%` (Multiplication) - left associative
//! 6. `^` (Power) - right associative
//! 7. Unary `-`, `!` - prefix
//! 8. Postfix: `.field`, `(args)` - left associative
//!
//! ## Module Organization
//!
//! - `pratt` - Pratt parser core (precedence climbing, binary/unary ops)
//! - `atoms` - Atomic expressions (literals, identifiers, parenthesized)
//! - `special` - Special forms (if/let/filter)
//! - `spatial` - Spatial queries (nearest/within/pairs)
//! - `aggregate` - Aggregate operations (agg.*)

mod aggregate;
mod atoms;
mod pratt;
mod spatial;
mod special;

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::Expr;

/// Parse an expression.
pub fn parse_expr(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    pratt::parse_pratt(stream, 0)
}

/// Parse an atomic expression (literal, identifier, parenthesized expr).
///
/// This parser does not consume binary operators, making it suitable for
/// contexts where operators would be ambiguous (e.g., type bounds where `>`
/// closes the type parameters rather than starting a comparison).
pub(super) fn parse_atom(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    atoms::parse_atom(stream)
}

/// Parse a primary expression (unary operators + atoms, no binary ops).
///
/// Handles:
/// - Unary operators: `-x`, `!flag`
/// - Atoms: literals, identifiers, parenthesized expressions
///
/// Does NOT handle:
/// - Binary operators: `+`, `-`, `*`, `/`, `>`, etc.
/// - Special forms: `if`, `let`
/// - Postfix: `.field`, `(args)`
///
/// Suitable for type bounds where `-273.15` should parse but `>` should not
/// be consumed as a comparison operator.
pub(super) fn parse_primary(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    use continuum_cdsl_ast::foundation::UnaryOp;
    use continuum_cdsl_ast::UntypedKind;
    use continuum_cdsl_lexer::Token;

    match stream.peek() {
        Some(Token::Minus) | Some(Token::Not) => {
            let start = stream.current_pos();
            let span = stream.current_span();
            let op = match stream.advance() {
                Some(Token::Minus) => UnaryOp::Neg,
                Some(Token::Not) => UnaryOp::Not,
                other => {
                    return Err(ParseError::unexpected_token(other, "unary operator", span));
                }
            };

            let operand = parse_primary(stream)?; // Recursive for nested unary
            let span = stream.span_from(start);

            Ok(Expr::new(
                UntypedKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
            ))
        }
        _ => atoms::parse_atom(stream),
    }
}
