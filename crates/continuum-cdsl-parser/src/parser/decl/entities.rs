//! Entity and member declarations.

use super::{parse_attributes, ParseError, TokenStream};
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

    // Parse the full path (e.g., "terra.plate.velocity")
    // parse_path is greedy and will consume the entire dotted path
    let full_path = super::super::types::parse_path(stream)?;

    // Split the path into entity path and member path
    // The member path is the last segment, everything before is the entity path
    let segments = full_path.segments();
    if segments.is_empty() {
        return Err(ParseError::unexpected_token(
            None,
            "member path",
            stream.current_span(),
        ));
    }

    // Split: entity = all but last segment, member = last segment
    let (entity_segments, member_segments): (Vec<String>, Vec<String>) = if segments.len() == 1 {
        // Just "velocity" - entity is implicit/empty, member is "velocity"
        (vec![], segments.to_vec())
    } else {
        // "terra.plate.velocity" -> entity="terra.plate", member="velocity"
        let split_at = segments.len() - 1;
        (segments[..split_at].to_vec(), segments[split_at..].to_vec())
    };

    let entity_path = continuum_foundation::Path::new(entity_segments);
    let member_path = continuum_foundation::Path::new(member_segments);

    // Role is NOT explicitly declared in member syntax
    // Default to Signal; it will be inferred from execution blocks later
    let role = RoleData::Signal;

    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    // Parse attributes inside the body (before execution blocks)
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(super::parse_attribute(stream)?);
    }

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
