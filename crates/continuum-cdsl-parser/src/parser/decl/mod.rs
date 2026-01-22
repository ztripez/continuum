//! Declaration parsers (keyword-dispatched).
//!
//! This module implements parsers for all top-level CDSL declarations.
//! Organized by semantic grouping:
//! - `primitives` - signal, field, operator, impulse, fracture, chronicle
//! - `entities` - entity, member
//! - `time` - stratum, era
//! - `definitions` - type, const, config
//! - `world` - world, warmup, policy

mod definitions;
mod entities;
mod primitives;
mod time;
mod world;

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{Attribute, Declaration, Node, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse all declarations from a token stream.
pub fn parse_declarations(stream: &mut TokenStream) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let mut declarations = Vec::new();
    let mut errors = Vec::new();

    while !stream.at_end() {
        match parse_declaration(stream) {
            Ok(decl) => declarations.push(decl),
            Err(e) => {
                errors.push(e);
                stream.synchronize(); // Skip to next declaration
            }
        }
    }

    if errors.is_empty() {
        Ok(declarations)
    } else {
        Err(errors)
    }
}

/// Parse a single declaration (keyword-dispatched).
fn parse_declaration(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    match stream.peek() {
        Some(Token::World) => world::parse_world(stream),
        Some(Token::Signal) => primitives::parse_signal(stream),
        Some(Token::Field) => primitives::parse_field(stream),
        Some(Token::Operator) => primitives::parse_operator(stream),
        Some(Token::Impulse) => primitives::parse_impulse(stream),
        Some(Token::Fracture) => primitives::parse_fracture(stream),
        Some(Token::Chronicle) => primitives::parse_chronicle(stream),
        Some(Token::Entity) => entities::parse_entity(stream),
        Some(Token::Member) => entities::parse_member(stream),
        Some(Token::Strata) => time::parse_stratum(stream),
        Some(Token::Era) => time::parse_era(stream),
        Some(Token::Type) => definitions::parse_type_decl(stream),
        Some(Token::Const) => definitions::parse_const_block(stream),
        Some(Token::Config) => definitions::parse_config_block(stream),
        other => Err(ParseError::unexpected_token(
            other,
            "at declaration",
            stream.current_span(),
        )),
    }
}

// ============================================================================
// Common Helpers
// ============================================================================

/// Parse attribute: `:name` or `:name(args)`.
pub(super) fn parse_attribute(stream: &mut TokenStream) -> Result<Attribute, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Colon)?;

    let name = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => s.clone(),
            Some(token) => super::token_utils::keyword_to_string(&token).ok_or_else(|| {
                ParseError::unexpected_token(Some(&token), "attribute name", span)
            })?,
            None => {
                return Err(ParseError::unexpected_token(None, "attribute name", span));
            }
        }
    };

    let args = if matches!(stream.peek(), Some(Token::LParen)) {
        stream.advance();
        let mut args = Vec::new();

        if !matches!(stream.peek(), Some(Token::RParen)) {
            loop {
                args.push(super::expr::parse_expr(stream)?);
                if matches!(stream.peek(), Some(Token::RParen)) {
                    break;
                }
                stream.expect(Token::Comma)?;
            }
        }

        stream.expect(Token::RParen)?;
        args
    } else {
        Vec::new()
    };

    Ok(Attribute {
        name,
        args,
        span: stream.span_from(start),
    })
}

/// Parse multiple attributes (repeated pattern across declarations).
pub(super) fn parse_attributes(stream: &mut TokenStream) -> Result<Vec<Attribute>, ParseError> {
    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }
    Ok(attributes)
}

/// Parse a node declaration with common structure.
///
/// This helper eliminates duplication across signal, field, operator,
/// impulse, fracture, and chronicle declarations which all follow the pattern:
/// ```text
/// <keyword> <path> { [attributes] [execution_blocks] }
/// ```
pub(super) fn parse_node_declaration(
    stream: &mut TokenStream,
    keyword: Token,
    role: RoleData,
) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(keyword)?;

    let path = super::types::parse_path(stream)?;
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    // Check for type expression as first attribute: `: Scalar<unit>`
    let type_expr = if matches!(stream.peek(), Some(Token::Colon)) {
        // Peek ahead to see if this is a type keyword
        let is_type = if let Some(Token::Ident(name)) = stream.peek_nth(1) {
            super::token_utils::is_type_keyword(name)
        } else {
            false
        };

        if is_type {
            stream.advance(); // consume ':'
            Some(super::types::parse_type_expr(stream)?)
        } else {
            None
        }
    } else {
        None
    };

    // Parse remaining attributes inside the body (before special/execution blocks)
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    // Parse special blocks (when, warmup, observe)
    let when = if matches!(stream.peek(), Some(Token::When)) {
        Some(super::blocks::parse_when_block(stream)?)
    } else {
        None
    };

    let warmup = if matches!(stream.peek(), Some(Token::WarmUp)) {
        Some(super::blocks::parse_warmup_block(stream)?)
    } else {
        None
    };

    let observe = if matches!(stream.peek(), Some(Token::Observe)) {
        Some(super::blocks::parse_observe_block(stream)?)
    } else {
        None
    };

    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(path, stream.span_from(start), role, ());
    node.attributes = attributes;
    node.type_expr = type_expr;
    node.when = when;
    node.warmup = warmup;
    node.observe = observe;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}
