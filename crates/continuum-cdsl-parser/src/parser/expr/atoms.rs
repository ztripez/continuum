//! Atomic expressions - literals, identifiers, parenthesized expressions.

use super::super::{ParseError, TokenStream};
use super::{aggregate, spatial, special};
use continuum_cdsl_ast::foundation::EntityId;
use continuum_cdsl_ast::{Expr, UntypedKind};
use continuum_cdsl_lexer::Token;

/// Parse atomic expressions (literals, identifiers, special keywords).
pub(super) fn parse_atom(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();

    match stream.peek() {
        Some(Token::True) => {
            stream.advance();
            Ok(Expr::new(
                UntypedKind::BoolLiteral(true),
                stream.span_from(start),
            ))
        }
        Some(Token::False) => {
            stream.advance();
            Ok(Expr::new(
                UntypedKind::BoolLiteral(false),
                stream.span_from(start),
            ))
        }
        Some(Token::Integer(_)) | Some(Token::Float(_)) => parse_numeric_literal(stream),
        Some(Token::String(_)) => {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::String(s)) => Ok(Expr::new(
                    UntypedKind::StringLiteral(s.to_string()),
                    stream.span_from(start),
                )),
                other => Err(ParseError::unexpected_token(other, "string literal", span)),
            }
        }
        Some(Token::LBracket) => parse_vector_literal(stream),
        Some(Token::LParen) => parse_parenthesized(stream),
        Some(Token::Prev) => {
            stream.advance();
            Ok(Expr::new(UntypedKind::Prev, stream.span_from(start)))
        }
        Some(Token::Current) => {
            stream.advance();
            Ok(Expr::new(UntypedKind::Current, stream.span_from(start)))
        }
        Some(Token::Inputs) => {
            stream.advance();
            Ok(Expr::new(UntypedKind::Inputs, stream.span_from(start)))
        }
        Some(Token::Self_) => {
            stream.advance();
            Ok(Expr::new(UntypedKind::Self_, stream.span_from(start)))
        }
        Some(Token::Other) => parse_other_or_field_access(stream),
        Some(Token::Payload) => {
            stream.advance();
            Ok(Expr::new(UntypedKind::Payload, stream.span_from(start)))
        }
        Some(Token::Filter) => special::parse_filter(stream),
        Some(Token::Nearest) => spatial::parse_nearest(stream),
        Some(Token::Within) => spatial::parse_within(stream),
        Some(Token::Pairs) => spatial::parse_pairs(stream),
        Some(Token::Agg) => aggregate::parse_aggregate(stream),
        Some(Token::Entity) => parse_entity_reference(stream),
        Some(Token::Config) => parse_config_reference(stream),
        Some(Token::Const) => parse_const_reference(stream),
        Some(Token::Ident(_)) | Some(Token::Signal) | Some(Token::Field) => {
            parse_identifier(stream)
        }
        other => Err(ParseError::unexpected_token(
            other,
            "in expression",
            stream.current_span(),
        )),
    }
}

/// Parse numeric literal.
fn parse_numeric_literal(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    let span = stream.current_span();

    let value = match stream.advance() {
        Some(Token::Integer(n)) => *n as f64,
        Some(Token::Float(f)) => *f,
        other => {
            return Err(ParseError::unexpected_token(other, "numeric literal", span));
        }
    };

    // Check for optional unit: <unit>
    let unit = if matches!(stream.peek(), Some(Token::Lt)) {
        stream.advance();
        let unit_expr = super::super::types::parse_unit_expr(stream)?;
        stream.expect(Token::Gt)?;
        Some(unit_expr)
    } else {
        None
    };

    Ok(Expr::new(
        UntypedKind::Literal { value, unit },
        stream.span_from(start),
    ))
}

/// Parse vector literal: [expr, expr, ...]
fn parse_vector_literal(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::LBracket)?;

    let mut elements = Vec::new();
    while !matches!(stream.peek(), Some(Token::RBracket)) {
        elements.push(super::parse_expr(stream)?);

        if !matches!(stream.peek(), Some(Token::RBracket)) {
            stream.expect(Token::Comma)?;
        }
    }

    stream.expect(Token::RBracket)?;

    Ok(Expr::new(
        UntypedKind::Vector(elements),
        stream.span_from(start),
    ))
}

/// Parse parenthesized expression.
fn parse_parenthesized(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    stream.expect(Token::LParen)?;
    let expr = super::parse_expr(stream)?;
    stream.expect(Token::RParen)?;
    Ok(expr)
}

/// Parse identifier or path.
fn parse_identifier(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    let span = stream.current_span();

    let name = match stream.advance() {
        Some(Token::Ident(s)) => s.clone(),
        Some(token) => super::super::token_utils::keyword_to_string(&token)
            .ok_or_else(|| ParseError::unexpected_token(Some(&token), "identifier", span))?,
        None => {
            return Err(ParseError::unexpected_token(None, "identifier", span));
        }
    };

    Ok(Expr::new(
        UntypedKind::Local(name.to_string()),
        stream.span_from(start),
    ))
}

/// Parse `other` keyword or `other(entity)`.
fn parse_other_or_field_access(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Other)?;

    // Check if it's other(entity) or just `other`
    if matches!(stream.peek(), Some(Token::LParen)) {
        stream.advance();
        let path = super::super::types::parse_path(stream)?;
        stream.expect(Token::RParen)?;
        Ok(Expr::new(
            UntypedKind::OtherInstances(EntityId::new(path.to_string())),
            stream.span_from(start),
        ))
    } else {
        Ok(Expr::new(UntypedKind::Other, stream.span_from(start)))
    }
}

/// Parse entity.path reference.
///
/// Parses `entity.path.segments` but stops before segments that are followed
/// by `(` to allow method calls like `.at(0)` to be handled by postfix parsing.
fn parse_entity_reference(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Entity)?;
    stream.expect(Token::Dot)?;

    // Parse path segments, but stop if the next segment would be followed by `(`
    // This allows `.at(0)` method calls to be handled by postfix parsing
    let mut segments = Vec::new();

    loop {
        let segment = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(s)) => s.clone(),
                Some(token) => {
                    super::super::token_utils::keyword_to_string(&token).ok_or_else(|| {
                        ParseError::unexpected_token(Some(&token), "in entity path", span)
                    })?
                }
                None => {
                    return Err(ParseError::unexpected_token(None, "in entity path", span));
                }
            }
        };
        segments.push(segment.to_string());

        // Check if next token is `.` followed by identifier followed by `(`
        // If so, stop here and let postfix parsing handle the method call
        if matches!(stream.peek(), Some(Token::Dot)) {
            // Look ahead: is next segment followed by `(`?
            if let (Some(Token::Ident(_)), Some(Token::LParen)) =
                (stream.peek_nth(1), stream.peek_nth(2))
            {
                // Stop here - let postfix parsing handle `.method(...)`
                break;
            }
            stream.advance(); // consume '.'
        } else {
            break;
        }
    }

    let path = continuum_cdsl_ast::foundation::Path { segments };

    Ok(Expr::new(
        UntypedKind::Entity(EntityId::new(path.to_string())),
        stream.span_from(start),
    ))
}

/// Parse a `config.path` reference expression.
///
/// Parses the `config` keyword followed by a dot-separated path to a configuration value.
/// Config references live in a separate namespace from simulation declarations.
///
/// # Syntax
///
/// ```text
/// config.path.to.value
/// ```
///
/// # Examples
///
/// ```cdsl
/// config.world.gravity
/// config.rendering.resolution
/// ```
///
/// # Returns
///
/// An untyped expression of kind `Config(path)` containing the parsed path.
fn parse_config_reference(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Config)?;
    stream.expect(Token::Dot)?;

    let path = super::super::types::parse_path(stream)?;

    Ok(Expr::new(
        UntypedKind::Config(path),
        stream.span_from(start),
    ))
}

/// Parse a `const.path` reference expression.
///
/// Parses the `const` keyword followed by a dot-separated path to a constant value.
/// Const references live in a separate namespace from simulation declarations.
///
/// # Syntax
///
/// ```text
/// const.path.to.value
/// ```
///
/// # Examples
///
/// ```cdsl
/// const.physics.gravitational
/// const.math.pi
/// ```
///
/// # Returns
///
/// An untyped expression of kind `Const(path)` containing the parsed path.
fn parse_const_reference(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Const)?;
    stream.expect(Token::Dot)?;

    let path = super::super::types::parse_path(stream)?;

    Ok(Expr::new(UntypedKind::Const(path), stream.span_from(start)))
}
