///! Common parsing helper functions to reduce duplication.
///!
///! This module contains reusable parsing patterns that appear multiple times
///! throughout the parser codebase.
use crate::parser::{error::ParseError, stream::TokenStream, token_utils, types};
use continuum_cdsl_ast::{Attribute, TypeExpr};
use continuum_cdsl_lexer::Token;

/// Attempts to parse a type expression preceded by a colon.
///
/// This performs lookahead to disambiguate between:
/// - `: TypeKeyword<...>` (type annotation)
/// - `: attribute_name(...)` (attribute)
///
/// # Grammar Ambiguity
///
/// Both type expressions and attributes start with `:`. This function performs
/// semantic lookahead by checking if the token after `:` is a type keyword.
///
/// This is an architectural tradeoff:
/// - Parser does semantic check (violates phase boundary)
/// - Alternative: Parse ambiguously, resolve in semantic analysis (harder)
/// - Grammar ambiguity cannot be eliminated without changing DSL syntax
///
/// # Returns
///
/// - `Ok(Some(TypeExpr))` if a type annotation is found
/// - `Ok(None)` if no colon or not a type keyword
/// - `Err(ParseError)` if type parsing fails
///
/// # Example
///
/// ```cdsl
/// signal x {
///     : Scalar<K>    // <- Parsed by this function
///     : title("...")  // <- Not a type, returns None
/// }
/// ```
pub fn try_parse_type_expr(stream: &mut TokenStream) -> Result<Option<TypeExpr>, ParseError> {
    if matches!(stream.peek(), Some(Token::Colon)) {
        // Peek ahead to see if this is a type keyword
        let is_type = if let Some(Token::Ident(name)) = stream.peek_nth(1) {
            token_utils::is_type_keyword(name)
        } else {
            false
        };

        if is_type {
            stream.advance(); // consume ':'
            Some(types::parse_type_expr(stream)).transpose()
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Parses zero or more attributes starting with `:`.
///
/// Attributes are prefixed with `:` and continue until a non-colon token is encountered.
///
/// # Returns
///
/// Vector of parsed attributes (empty if none found).
///
/// # Example
///
/// ```cdsl
/// signal x {
///     : title("Temperature")
///     : symbol("T")
///     resolve { ... }
/// }
/// ```
///
/// This function is typically called in two contexts:
/// 1. Before `{` — pre-body attributes
/// 2. After `{` — body-level attributes
pub fn parse_attributes(stream: &mut TokenStream) -> Result<Vec<Attribute>, ParseError> {
    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(super::decl::parse_attribute(stream)?);
    }
    Ok(attributes)
}
