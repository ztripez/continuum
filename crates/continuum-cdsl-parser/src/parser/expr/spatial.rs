//! Spatial query expressions - nearest/within/pairs.

use super::super::{ParseError, TokenStream};
use continuum_cdsl_ast::foundation::EntityId;
use continuum_cdsl_ast::{Expr, UntypedKind};
use continuum_cdsl_lexer::Token;

/// Parse nearest(entity, position).
pub(super) fn parse_nearest(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Nearest)?;
    stream.expect(Token::LParen)?;

    let entity_path = super::super::types::parse_path(stream)?;
    stream.expect(Token::Comma)?;
    let position = super::parse_expr(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Nearest {
            entity: EntityId::new(entity_path.to_string()),
            position: Box::new(position),
        },
        stream.span_from(start),
    ))
}

/// Parse within(entity, position, radius).
pub(super) fn parse_within(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Within)?;
    stream.expect(Token::LParen)?;

    let entity_path = super::super::types::parse_path(stream)?;
    stream.expect(Token::Comma)?;
    let position = super::parse_expr(stream)?;
    stream.expect(Token::Comma)?;
    let radius = super::parse_expr(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Within {
            entity: EntityId::new(entity_path.to_string()),
            position: Box::new(position),
            radius: Box::new(radius),
        },
        stream.span_from(start),
    ))
}

/// Parse pairs(entity).
pub(super) fn parse_pairs(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Pairs)?;
    stream.expect(Token::LParen)?;

    let entity_path = super::super::types::parse_path(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::PairsInstances(EntityId::new(entity_path.to_string())),
        stream.span_from(start),
    ))
}
