//! Type and unit expression parsers.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::foundation::Path;
use continuum_cdsl_ast::{TypeExpr, UnitExpr};
use continuum_cdsl_lexer::Token;

/// Parse a path (dot-separated identifiers).
pub fn parse_path(stream: &mut TokenStream) -> Result<Path, ParseError> {
    let mut segments = Vec::new();

    loop {
        let segment = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(s)) => s.clone(),
                Some(Token::Config) => "config".to_string(),
                Some(Token::Const) => "const".to_string(),
                Some(Token::Signal) => "signal".to_string(),
                Some(Token::Field) => "field".to_string(),
                Some(Token::Entity) => "entity".to_string(),
                Some(Token::Strata) => "strata".to_string(),
                Some(Token::Type) => "type".to_string(),
                other => {
                    return Err(ParseError::unexpected_token(other, "in path", span));
                }
            }
        };
        segments.push(segment);

        if !matches!(stream.peek(), Some(Token::Dot)) {
            break;
        }
        stream.advance(); // consume '.'
    }

    Ok(Path { segments })
}

/// Parse a type expression.
pub fn parse_type_expr(stream: &mut TokenStream) -> Result<TypeExpr, ParseError> {
    match stream.peek() {
        Some(Token::Ident(name)) if name == "Bool" => {
            stream.advance();
            Ok(TypeExpr::Bool)
        }
        Some(Token::Ident(name)) if name == "Scalar" => {
            stream.advance();
            parse_scalar_type(stream)
        }
        Some(Token::Ident(name)) if name == "Vector" => {
            stream.advance();
            parse_vector_type(stream)
        }
        Some(Token::Ident(_)) => {
            // User-defined type
            let path = parse_path(stream)?;
            Ok(TypeExpr::User(path))
        }
        other => Err(ParseError::unexpected_token(
            other,
            "in type expression",
            stream.current_span(),
        )),
    }
}

/// Parse Scalar type with optional unit and bounds.
fn parse_scalar_type(stream: &mut TokenStream) -> Result<TypeExpr, ParseError> {
    if !matches!(stream.peek(), Some(Token::Lt)) {
        // Scalar without unit
        return Ok(TypeExpr::Scalar {
            unit: None,
            bounds: None,
        });
    }

    stream.expect(Token::Lt)?;

    let unit = parse_unit_expr(stream)?;

    // Check for optional bounds: , min..max
    let bounds = if matches!(stream.peek(), Some(Token::Comma)) {
        stream.advance();
        let min = super::expr::parse_expr(stream)?;
        stream.expect(Token::DotDot)?;
        let max = super::expr::parse_expr(stream)?;
        Some((min, max))
    } else {
        None
    };

    stream.expect(Token::Gt)?;

    Ok(TypeExpr::Scalar {
        unit: Some(unit),
        bounds,
    })
}

/// Parse Vector type.
fn parse_vector_type(stream: &mut TokenStream) -> Result<TypeExpr, ParseError> {
    stream.expect(Token::Lt)?;

    let dim = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Integer(n)) => *n as u8,
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "vector dimension",
                    span,
                ));
            }
        }
    };

    stream.expect(Token::Comma)?;

    let unit = parse_unit_expr(stream)?;

    stream.expect(Token::Gt)?;

    Ok(TypeExpr::Vector {
        dim,
        unit: Some(unit),
    })
}

/// Parse a unit expression.
///
/// Grammar:
/// ```text
/// unit_expr := unit_term (('*' | '/') unit_term)*
/// unit_term := base_unit ('^' integer)?
/// base_unit := identifier | '(' unit_expr ')'
/// ```
pub fn parse_unit_expr(stream: &mut TokenStream) -> Result<UnitExpr, ParseError> {
    parse_unit_product(stream)
}

/// Parse unit product/quotient.
fn parse_unit_product(stream: &mut TokenStream) -> Result<UnitExpr, ParseError> {
    let mut left = parse_unit_power(stream)?;

    while matches!(stream.peek(), Some(Token::Star) | Some(Token::Slash)) {
        let is_mul = matches!(stream.peek(), Some(Token::Star));
        stream.advance();

        let right = parse_unit_power(stream)?;

        left = if is_mul {
            UnitExpr::Multiply(Box::new(left), Box::new(right))
        } else {
            UnitExpr::Divide(Box::new(left), Box::new(right))
        };
    }

    Ok(left)
}

/// Parse unit with optional power.
fn parse_unit_power(stream: &mut TokenStream) -> Result<UnitExpr, ParseError> {
    let base = parse_unit_base(stream)?;

    if matches!(stream.peek(), Some(Token::Caret)) {
        stream.advance();
        let exp = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Integer(n)) => *n as i8,
                other => {
                    return Err(ParseError::unexpected_token(other, "unit exponent", span));
                }
            }
        };
        Ok(UnitExpr::Power(Box::new(base), exp))
    } else {
        Ok(base)
    }
}

/// Parse base unit or parenthesized unit expression.
fn parse_unit_base(stream: &mut TokenStream) -> Result<UnitExpr, ParseError> {
    match stream.peek() {
        Some(Token::LParen) => {
            stream.advance();
            let expr = parse_unit_expr(stream)?;
            stream.expect(Token::RParen)?;
            Ok(expr)
        }
        Some(Token::Ident(name)) => {
            let unit_name = name.clone();
            stream.advance();
            Ok(UnitExpr::Base(unit_name))
        }
        other => Err(ParseError::unexpected_token(
            other,
            "in unit expression",
            stream.current_span(),
        )),
    }
}
