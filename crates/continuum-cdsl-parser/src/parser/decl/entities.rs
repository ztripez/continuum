//! Entity and member declarations.

use super::{parse_attributes, ParseError, TokenStream};
use continuum_cdsl_ast::foundation::EntityId;
use continuum_cdsl_ast::{Declaration, Entity, Node, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse entity declaration with nested members.
///
/// Syntax: `entity path { ... }`
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

    let member_path = continuum_foundation::Path::new(vec![member_name.to_string()]);
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    let type_expr = super::super::helpers::try_parse_type_expr(stream)?;
    attributes.extend(super::super::helpers::parse_attributes(stream)?);

    // Parse config/const blocks inside members
    let mut nested_blocks = super::parse_nested_config_const(stream)?;

    let special = super::super::helpers::parse_special_blocks(stream)?;

    // Parse any additional config/const blocks after special blocks
    nested_blocks.extend(super::parse_nested_config_const(stream)?);

    let execution_blocks = super::super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let entity_id = EntityId::new(entity_path.to_string());
    let mut node = Node::new(member_path, stream.span_from(start), role, entity_id);
    node.attributes = attributes;
    node.type_expr = type_expr;
    node.when = special.when;
    node.warmup = special.warmup;
    node.observe = special.observe;
    node.nested_blocks = nested_blocks;
    node.execution_blocks = execution_blocks;

    Ok(node)
}
