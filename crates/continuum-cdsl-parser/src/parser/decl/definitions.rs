//! Type, const, and config declarations.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::foundation::{Path, Span};
use continuum_cdsl_ast::{
    ConfigEntry, ConstEntry, Declaration, Expr, TypeDecl, TypeExpr, TypeField,
};
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

/// Parse const/config entry syntax (shared logic).
///
/// Supports:
/// - `path: TYPE = value` (explicit type)
/// - `path: value` (inferred type)
/// - `path: TYPE` (explicit type, no value - only valid for config)
fn parse_const_or_config_entry(
    stream: &mut TokenStream,
    allow_missing_value: bool,
) -> Result<(Path, TypeExpr, Option<Expr>, Span), ParseError> {
    let entry_start = stream.current_pos();
    let path = super::super::types::parse_path(stream)?;
    stream.expect(Token::Colon)?;

    // Check if next token is a type keyword
    let has_explicit_type = matches!(
        stream.peek(),
        Some(Token::Ident(name)) if matches!(
            name.as_str(),
            "Bool" | "Scalar" | "Vec2" | "Vec3" | "Vec4" | "Quat" | "Mat2" | "Mat3" | "Mat4" | "Tensor"
        )
    );

    let type_expr = if has_explicit_type {
        let t = super::super::types::parse_type_expr(stream)?;
        // After explicit type, expect '=' if value follows
        if matches!(stream.peek(), Some(Token::Eq)) {
            stream.advance();
        }
        t
    } else {
        TypeExpr::Infer
    };

    // Parse value if present
    let value = if matches!(stream.peek(), Some(Token::RBrace)) {
        // End of block - no value
        if !allow_missing_value {
            return Err(ParseError::unexpected_token(
                stream.peek(),
                "value expression",
                stream.current_span(),
            ));
        }
        None
    } else {
        // Value present
        Some(super::super::expr::parse_expr(stream)?)
    };

    Ok((path, type_expr, value, stream.span_from(entry_start)))
}

/// Parse const block.
pub(super) fn parse_const_block(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    stream.expect(Token::Const)?;
    stream.expect(Token::LBrace)?;

    let mut entries = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let (path, type_expr, value, span) = parse_const_or_config_entry(stream, false)?;

        entries.push(ConstEntry {
            path,
            value: value.expect("const value must be present"),
            type_expr,
            span,
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
        let (path, type_expr, default, span) = parse_const_or_config_entry(stream, true)?;

        entries.push(ConfigEntry {
            path,
            default,
            type_expr,
            span,
            doc: None,
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(Declaration::Config(entries))
}
