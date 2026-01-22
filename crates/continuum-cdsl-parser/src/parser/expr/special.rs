//! Special expression forms - if/let/filter.

use super::super::{ParseError, TokenStream};
use continuum_cdsl_ast::{Expr, UntypedKind};
use continuum_cdsl_lexer::Token;

/// Parse if-then-else expression.
pub(super) fn parse_if(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::If)?;

    let condition = super::parse_expr(stream)?;

    stream.expect(Token::LBrace)?;
    let then_branch = super::parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    stream.expect(Token::Else)?;

    stream.expect(Token::LBrace)?;
    let else_branch = super::parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    let span = stream.span_from(start);
    Ok(Expr::new(
        UntypedKind::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        span,
    ))
}

/// Parse let-in expression.
pub(super) fn parse_let(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Let)?;

    let name = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => s.clone(),
            other => {
                return Err(ParseError::unexpected_token(other, "in let binding", span));
            }
        }
    };

    stream.expect(Token::Eq)?;
    let value = super::parse_expr(stream)?;

    stream.expect(Token::In)?;
    let body = super::parse_expr(stream)?;

    let span = stream.span_from(start);
    Ok(Expr::new(
        UntypedKind::Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
        },
        span,
    ))
}

/// Parse filter(source, predicate).
pub(super) fn parse_filter(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Filter)?;
    stream.expect(Token::LParen)?;

    let source = super::parse_expr(stream)?;
    stream.expect(Token::Comma)?;
    let predicate = super::parse_expr(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Filter {
            source: Box::new(source),
            predicate: Box::new(predicate),
        },
        stream.span_from(start),
    ))
}
