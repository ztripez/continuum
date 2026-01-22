//! Declaration parsers (keyword-dispatched).

use super::{ParseError, TokenStream};
use crate::ast::Declaration;

/// Parse all declarations from a token stream.
pub fn parse_declarations(_stream: &mut TokenStream) -> Result<Vec<Declaration>, Vec<ParseError>> {
    // TODO: Implement declaration parser
    todo!("Declaration parser not yet implemented")
}
