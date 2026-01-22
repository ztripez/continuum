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
mod special;
mod spatial;

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::Expr;

/// Parse an expression.
pub fn parse_expr(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    pratt::parse_pratt(stream, 0)
}
