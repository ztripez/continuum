//! Entity and member declarations.

use super::{parse_attributes, ParseError, TokenStream};
use continuum_cdsl_ast::foundation::EntityId;
use continuum_cdsl_ast::{Declaration, Entity, Node, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse entity declaration with nested members.
pub(super) fn parse_entity(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Entity)?;

    let entity_path = super::super::types::parse_path(stream)?;
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(super::parse_attribute(stream)?);
    }

    let mut members = Vec::new();
    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let member = parse_entity_member(stream, &entity_path)?;
        members.push(member);
    }

    stream.expect(Token::RBrace)?;

    let mut entity = Entity::new(
        EntityId::new(entity_path.to_string()),
        entity_path,
        stream.span_from(start),
    );
    entity.attributes = attributes;
    entity.members = members;
    Ok(Declaration::Entity(entity))
}

/// Parse nested member (signal/field/fracture/impulse) inside entity block.
fn parse_entity_member(
    stream: &mut TokenStream,
    entity_path: &continuum_foundation::Path,
) -> Result<Node<EntityId>, ParseError> {
    let start = stream.current_pos();

    let (keyword, role) = match stream.peek() {
        Some(Token::Signal) => (Token::Signal, RoleData::Signal),
        Some(Token::Field) => (
            Token::Field,
            RoleData::Field {
                reconstruction: None,
            },
        ),
        Some(Token::Fracture) => (Token::Fracture, RoleData::Fracture),
        Some(Token::Impulse) => (Token::Impulse, RoleData::Impulse { payload: None }),
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "signal, field, fracture, or impulse",
                stream.current_span(),
            ))
        }
    };

    stream.expect(keyword)?;

    let member_name_token = stream.peek();
    let member_name = match member_name_token {
        Some(Token::Ident(name)) => {
            let n = name.clone();
            stream.advance();
            n
        }
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "member name (identifier)",
                stream.current_span(),
            ))
        }
    };

    let member_path = continuum_foundation::Path::new(vec![member_name]);
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    let type_expr = if matches!(stream.peek(), Some(Token::Colon)) {
        let is_type = if let Some(Token::Ident(name)) = stream.peek_nth(1) {
            super::super::token_utils::is_type_keyword(name)
        } else {
            false
        };

        if is_type {
            stream.advance();
            Some(super::super::types::parse_type_expr(stream)?)
        } else {
            None
        }
    } else {
        None
    };

    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(super::parse_attribute(stream)?);
    }

    let when = if matches!(stream.peek(), Some(Token::When)) {
        Some(super::super::blocks::parse_when_block(stream)?)
    } else {
        None
    };

    let warmup = if matches!(stream.peek(), Some(Token::WarmUp)) {
        Some(super::super::blocks::parse_warmup_block(stream)?)
    } else {
        None
    };

    let observe = if matches!(stream.peek(), Some(Token::Observe)) {
        Some(super::super::blocks::parse_observe_block(stream)?)
    } else {
        None
    };

    let execution_blocks = super::super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let entity_id = EntityId::new(entity_path.to_string());
    let mut node = Node::new(member_path, stream.span_from(start), role, entity_id);
    node.attributes = attributes;
    node.type_expr = type_expr;
    node.when = when;
    node.warmup = warmup;
    node.observe = observe;
    node.execution_blocks = execution_blocks;

    Ok(node)
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
