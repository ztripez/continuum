//! Entity and member declarations.

use super::{parse_attributes, ParseError, TokenStream};
use continuum_cdsl_ast::foundation::{EntityId, Path};
use continuum_cdsl_ast::{Declaration, Entity, Node, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse entity declaration with nested members and child entities.
///
/// Syntax: `entity path { [attributes] [members] [child entities] }`
///
/// Entity bodies accept:
/// - Attributes (`:name` or `:name(args)`)
/// - Member primitives: `signal`, `field`, `fracture`, `impulse`, `operator`
/// - Child entities: `entity` (recursive)
pub(super) fn parse_entity(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Entity)?;

    let entity_path = super::super::types::parse_path(stream)?;
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(super::parse_attribute(stream)?);
    }

    let mut entity = Entity::new(
        EntityId::new(entity_path.to_string()),
        entity_path.clone(),
        // Span is provisional — updated after closing brace
        stream.span_from(start),
    );
    entity.attributes = attributes;

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        if matches!(stream.peek(), Some(Token::Entity)) {
            // Recursive child entity
            let child_decl = parse_entity(stream)?;
            if let Declaration::Entity(child) = child_decl {
                entity.children.insert(child.path.clone(), child);
            }
        } else {
            // Member primitive (signal, field, fracture, impulse, operator)
            let member = parse_entity_member(stream, &entity_path)?;
            entity.members.push(member);
        }
    }

    stream.expect(Token::RBrace)?;
    entity.span = stream.span_from(start);

    Ok(Declaration::Entity(entity))
}

/// Parse nested member (signal/field/fracture/impulse/operator) inside entity block.
fn parse_entity_member(stream: &mut TokenStream, entity_path: &Path) -> Result<Node, ParseError> {
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
        Some(Token::Operator) => (Token::Operator, RoleData::Operator),
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "signal, field, fracture, impulse, or operator",
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

    let member_path = Path::new(vec![member_name.to_string()]);
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
    let mut node = Node::new(member_path, stream.span_from(start), role, Some(entity_id));
    node.attributes = attributes;
    node.type_expr = type_expr;
    node.when = special.when;
    node.warmup = special.warmup;
    node.observe = special.observe;
    node.nested_blocks = nested_blocks;
    node.execution_blocks = execution_blocks;

    Ok(node)
}

/// Parse standalone `member` declaration at top level.
///
/// Syntax: `member path { ... }`
///
/// The `member` keyword declares a signal belonging to an entity without
/// nesting inside the entity block. The entity is inferred from the path:
/// the first path segment is the entity name.
///
/// The role defaults to `Signal` for standalone members.
///
/// Example:
/// ```cdsl
/// member plate.velocity {
///     : Vec3<m/s>
///     resolve { ... }
/// }
/// ```
pub(super) fn parse_member(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Member)?;

    let path = super::super::types::parse_path(stream)?;

    // Extract entity from path: first segment is the entity
    let segments = path.segments();
    if segments.len() < 2 {
        return Err(ParseError::invalid_syntax(
            format!(
                "member path must have at least two segments (entity.name), got: {}",
                path
            ),
            stream.span_from(start),
        ));
    }

    let entity_name = &segments[0];
    let entity_id = EntityId::new(entity_name.to_string());

    // Member path is the remaining segments after the entity
    let member_path = Path::new(segments[1..].to_vec());

    let mut attributes = super::parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    let type_expr = super::super::helpers::try_parse_type_expr(stream)?;
    attributes.extend(super::super::helpers::parse_attributes(stream)?);

    let mut nested_blocks = super::parse_nested_config_const(stream)?;

    let special = super::super::helpers::parse_special_blocks(stream)?;

    nested_blocks.extend(super::parse_nested_config_const(stream)?);

    let execution_blocks = super::super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(
        member_path,
        stream.span_from(start),
        RoleData::Signal,
        Some(entity_id),
    );
    node.attributes = attributes;
    node.type_expr = type_expr;
    node.when = special.when;
    node.warmup = special.warmup;
    node.observe = special.observe;
    node.nested_blocks = nested_blocks;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Member(node))
}
