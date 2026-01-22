//! Primitive node declarations: signal, field, operator, impulse, fracture, chronicle.

use super::{ParseError, TokenStream, parse_node_declaration};
use continuum_cdsl_ast::{Declaration, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse signal declaration.
pub(super) fn parse_signal(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    parse_node_declaration(stream, Token::Signal, RoleData::Signal)
}

/// Parse field declaration.
pub(super) fn parse_field(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    parse_node_declaration(
        stream,
        Token::Field,
        RoleData::Field {
            reconstruction: None,
        },
    )
}

/// Parse operator declaration.
pub(super) fn parse_operator(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    parse_node_declaration(stream, Token::Operator, RoleData::Operator)
}

/// Parse impulse declaration.
pub(super) fn parse_impulse(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    parse_node_declaration(stream, Token::Impulse, RoleData::Impulse { payload: None })
}

/// Parse fracture declaration.
pub(super) fn parse_fracture(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    parse_node_declaration(stream, Token::Fracture, RoleData::Fracture)
}

/// Parse chronicle declaration.
pub(super) fn parse_chronicle(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    parse_node_declaration(stream, Token::Chronicle, RoleData::Chronicle)
}
