//! Type, const, and config declarations.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{ConfigEntry, ConstEntry, Declaration, TypeDecl, TypeField};
use continuum_cdsl_lexer::Token;

/// Parse type declaration.
pub(super) fn parse_type_decl(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Type)?;

    let name = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => s.clone(),
            other => {
                return Err(ParseError::unexpected_token(other, "type name", span));
            }
        }
    };

    stream.expect(Token::LBrace)?;

    let mut fields = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let field_start = stream.current_pos();
        let field_name = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(s)) => s.clone(),
                other => {
                    return Err(ParseError::unexpected_token(other, "field name", span));
                }
            }
        };

        stream.expect(Token::Colon)?;

        let type_expr = super::super::types::parse_type_expr(stream)?;

        fields.push(TypeField {
            name: field_name,
            type_expr,
            span: stream.span_from(field_start),
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(Declaration::Type(TypeDecl {
        name,
        fields,
        span: stream.span_from(start),
        doc: None,
    }))
}

/// Parse const block.
pub(super) fn parse_const_block(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    stream.expect(Token::Const)?;
    stream.expect(Token::LBrace)?;

    let mut entries = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let entry_start = stream.current_pos();
        let path = super::super::types::parse_path(stream)?;
        stream.expect(Token::Colon)?;
        let type_expr = super::super::types::parse_type_expr(stream)?;
        stream.expect(Token::Eq)?;
        let value = super::super::expr::parse_expr(stream)?;

        entries.push(ConstEntry {
            path,
            value,
            type_expr,
            span: stream.span_from(entry_start),
            doc: None,
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(Declaration::Const(entries))
}

/// Parse config block.
pub(super) fn parse_config_block(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    stream.expect(Token::Config)?;
    stream.expect(Token::LBrace)?;

    let mut entries = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let entry_start = stream.current_pos();
        let path = super::super::types::parse_path(stream)?;
        stream.expect(Token::Colon)?;
        let type_expr = super::super::types::parse_type_expr(stream)?;

        let default = if matches!(stream.peek(), Some(Token::Eq)) {
            stream.advance();
            Some(super::super::expr::parse_expr(stream)?)
        } else {
            None
        };

        entries.push(ConfigEntry {
            path,
            default,
            type_expr,
            span: stream.span_from(entry_start),
            doc: None,
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(Declaration::Config(entries))
}
