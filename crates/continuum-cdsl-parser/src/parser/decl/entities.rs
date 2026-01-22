//! Entity and member declarations.

use super::{ParseError, TokenStream, parse_attributes};
use continuum_cdsl_ast::foundation::EntityId;
use continuum_cdsl_ast::{Declaration, Entity, Node, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse entity declaration.
pub(super) fn parse_entity(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Entity)?;

    let path = super::super::types::parse_path(stream)?;
    let attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;
    stream.expect(Token::RBrace)?;

    let mut entity = Entity::new(
        EntityId::new(path.to_string()),
        path,
        stream.span_from(start),
    );
    entity.attributes = attributes;
    Ok(Declaration::Entity(entity))
}

/// Parse member declaration.
pub(super) fn parse_member(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Member)?;

    let entity_path = super::super::types::parse_path(stream)?;
    stream.expect(Token::Dot)?;
    let member_path = super::super::types::parse_path(stream)?;

    // Determine role from next keyword
    let role = match stream.peek() {
        Some(Token::Signal) => {
            stream.advance();
            RoleData::Signal
        }
        Some(Token::Field) => {
            stream.advance();
            RoleData::Field {
                reconstruction: None,
            }
        }
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "member role (signal/field)",
                stream.current_span(),
            ));
        }
    };

    let attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(
        member_path,
        stream.span_from(start),
        role,
        EntityId::new(entity_path.to_string()),
    );
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Member(node))
}
