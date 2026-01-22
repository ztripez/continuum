//! Expression parser using Pratt parsing (precedence climbing).

use super::{ParseError, TokenStream};
use crate::ast::Expr;

/// Parse an expression.
pub fn parse_expr(_stream: &mut TokenStream) -> Result<Expr, ParseError> {
    // TODO: Implement expression parser
    todo!("Expression parser not yet implemented")
}
