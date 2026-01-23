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
                Some(token) => super::token_utils::keyword_to_string(&token)
                    .ok_or_else(|| ParseError::unexpected_token(Some(&token), "in path", span))?,
                None => {
                    return Err(ParseError::unexpected_token(None, "in path", span));
                }
            }
        };
        segments.push(segment.to_string());

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
        Some(Token::Ident(name)) if &**name == "Bool" => {
            stream.advance();
            Ok(TypeExpr::Bool)
        }
        Some(Token::Ident(name)) if &**name == "Scalar" => {
            stream.advance();
            parse_scalar_type(stream)
        }
        Some(Token::Ident(name)) if &**name == "Vector" => {
            stream.advance();
            parse_vector_type(stream)
        }
        // Vec2, Vec3, Vec4 shorthand - fixed dimension vectors
        Some(Token::Ident(name)) if &**name == "Vec2" => {
            stream.advance();
            parse_vecn_type(stream, 2)
        }
        Some(Token::Ident(name)) if &**name == "Vec3" => {
            stream.advance();
            parse_vecn_type(stream, 3)
        }
        Some(Token::Ident(name)) if &**name == "Vec4" => {
            stream.advance();
            parse_vecn_type(stream, 4)
        }
        // Quat - quaternion type (parsed as Vec4 for now - no dedicated Quaternion type in AST)
        Some(Token::Ident(name)) if &**name == "Quat" => {
            stream.advance();
            // Quaternions are dimensionless 4-vectors
            Ok(TypeExpr::Vector { dim: 4, unit: None })
        }
        // Matrix types - Mat2/Mat3/Mat4 are square matrices
        Some(Token::Ident(name)) if &**name == "Mat2" => {
            stream.advance();
            parse_matrix_type(stream, 2, 2)
        }
        Some(Token::Ident(name)) if &**name == "Mat3" => {
            stream.advance();
            parse_matrix_type(stream, 3, 3)
        }
        Some(Token::Ident(name)) if &**name == "Mat4" => {
            stream.advance();
            parse_matrix_type(stream, 4, 4)
        }
        // Tensor type - not in current AST, parse as User type for now
        Some(Token::Ident(name)) if &**name == "Tensor" => {
            stream.advance();
            // Skip any type parameters for now
            if matches!(stream.peek(), Some(Token::Lt)) {
                skip_angle_brackets(stream)?;
            }
            Ok(TypeExpr::User(Path {
                segments: vec!["Tensor".to_string()],
            }))
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
///
/// # Syntax
///
/// ```text
/// Scalar                    # Dimensionless scalar, no bounds
/// Scalar<m/s²>              # Scalar with unit
/// Scalar<>                  # Dimensionless (explicit)
/// Scalar<1>                 # Dimensionless shorthand
/// Scalar<K, 0.0..1000.0>    # Dimensionless with bounds
/// Scalar<m/s, -100..100>    # Unit with bounds
/// ```
///
/// # Dimensionless Shortcuts
///
/// Three forms represent dimensionless scalars:
/// - `Scalar` (no `<>` brackets)
/// - `Scalar<>` (empty brackets)
/// - `Scalar<1>` (shorthand: `<1>` means dimensionless)
///
/// # Bounds Parsing
///
/// When bounds are present (`, min..max`), this function uses [`parse_primary()`]
/// instead of [`parse_expr()`] to avoid consuming the closing `>` as a comparison
/// operator. This allows negative bounds like `-273.15` via unary operators while
/// preventing binary operator parsing.
///
/// # Parameters
///
/// - `stream`: Token stream positioned **after** the `Scalar` keyword
///
/// # Returns
///
/// `TypeExpr::Scalar` with:
/// - `unit`: `None` (bare `Scalar`), `Some(Dimensionless)`, or `Some(UnitExpr)`
/// - `bounds`: `None` or `Some((min_expr, max_expr))`
///
/// # Errors
///
/// Returns `ParseError` if:
/// - Missing `>` after unit/bounds
/// - Unit expression syntax is invalid
/// - Bounds expressions are invalid (must be primary expressions)
/// - Missing `..` between min and max bounds
///
/// # Examples
///
/// ```text
/// Scalar<K, 0.0..373.15>     # Temperature with bounds
/// Scalar<1, -1.0..1.0>       # Normalized dimensionless
/// Scalar<m/s²>               # Acceleration
/// ```
fn parse_scalar_type(stream: &mut TokenStream) -> Result<TypeExpr, ParseError> {
    if !matches!(stream.peek(), Some(Token::Lt)) {
        // Scalar without unit
        return Ok(TypeExpr::Scalar {
            unit: None,
            bounds: None,
        });
    }

    stream.expect(Token::Lt)?;

    // Check for dimensionless: <> or <1> or <1, bounds>
    // But NOT <1/unit> which is a unit expression starting with 1
    let unit = if matches!(stream.peek(), Some(Token::Gt | Token::Comma)) {
        // Empty unit <> or <, ...> = dimensionless
        UnitExpr::Dimensionless
    } else if matches!(stream.peek(), Some(Token::Integer(n)) if *n == 1)
        && !matches!(
            stream.peek_nth(1),
            Some(Token::Slash) | Some(Token::Star) | Some(Token::Caret)
        )
    {
        // Shorthand: <1> = dimensionless (but not <1/unit> or <1*unit> or <1^n>)
        stream.advance();
        UnitExpr::Dimensionless
    } else {
        parse_unit_expr(stream)?
    };

    // Check for optional bounds: , min..max
    // Use parse_primary instead of parse_expr to avoid consuming > as comparison operator
    // parse_primary handles unary operators (-273.15) but not binary ops
    let bounds = if matches!(stream.peek(), Some(Token::Comma)) {
        stream.advance();
        let min = super::expr::parse_primary(stream)?;
        stream.expect(Token::DotDot)?;
        let max = super::expr::parse_primary(stream)?;
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

/// Parse Vec2/Vec3/Vec4 type: `Vec3<unit>` or `Vec3` (dimensionless)
fn parse_vecn_type(stream: &mut TokenStream, dim: u8) -> Result<TypeExpr, ParseError> {
    let unit = if matches!(stream.peek(), Some(Token::Lt)) {
        stream.advance(); // consume '<'
        let unit = parse_unit_expr(stream)?;
        stream.expect(Token::Gt)?;
        Some(unit)
    } else {
        None
    };

    Ok(TypeExpr::Vector { dim, unit })
}

/// Parse Mat2/Mat3/Mat4 type: `Mat3<unit>` or `Mat3` (dimensionless)
fn parse_matrix_type(stream: &mut TokenStream, rows: u8, cols: u8) -> Result<TypeExpr, ParseError> {
    let unit = if matches!(stream.peek(), Some(Token::Lt)) {
        stream.advance(); // consume '<'
        let unit = parse_unit_expr(stream)?;
        stream.expect(Token::Gt)?;
        Some(unit)
    } else {
        None
    };

    Ok(TypeExpr::Matrix { rows, cols, unit })
}

/// Skip angle-bracketed content: `<...>`
fn skip_angle_brackets(stream: &mut TokenStream) -> Result<(), ParseError> {
    stream.expect(Token::Lt)?;
    let mut depth = 1;
    while depth > 0 && !stream.at_end() {
        match stream.peek() {
            Some(Token::Lt) => {
                depth += 1;
                stream.advance();
            }
            Some(Token::Gt) => {
                depth -= 1;
                stream.advance();
            }
            _ => {
                stream.advance();
            }
        }
    }
    Ok(())
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

    // Check for caret notation (e.g., m^2)
    if matches!(stream.peek(), Some(Token::Caret)) {
        stream.advance();

        // Handle both positive and negative exponents: s^2 or s^-1
        let (is_negative, span) = if matches!(stream.peek(), Some(Token::Minus)) {
            let span = stream.current_span();
            stream.advance();
            (true, span)
        } else {
            (false, stream.current_span())
        };

        let exp = {
            match stream.advance() {
                Some(Token::Integer(n)) => {
                    let exp = *n as i8;
                    if is_negative {
                        -exp
                    } else {
                        exp
                    }
                }
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
            Ok(UnitExpr::Base(unit_name.to_string()))
        }
        // Handle `1` as dimensionless in unit expressions (e.g., 1/Myr)
        Some(Token::Integer(n)) if *n == 1 => {
            stream.advance();
            Ok(UnitExpr::Dimensionless)
        }
        other => Err(ParseError::unexpected_token(
            other,
            "in unit expression",
            stream.current_span(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_lexer::Token;
    use logos::Logos;

    #[test]
    fn test_scalar_with_bounds_tokens() {
        let source = "Scalar<K, 0.0..1000.0>";
        let mut lexer = Token::lexer(source);
        let mut tokens_with_spans = Vec::new();

        while let Some(result) = lexer.next() {
            if let Ok(token) = result {
                let span = lexer.span();
                tokens_with_spans.push((token, span));
            }
        }

        eprintln!("Tokens for '{}':", source);
        for (i, (tok, span)) in tokens_with_spans.iter().enumerate() {
            eprintln!("  {}: {:?} at {:?}", i, tok, span);
        }

        let mut stream = TokenStream::new(&tokens_with_spans, 0);
        // Skip 'Scalar' token
        assert!(matches!(stream.peek(), Some(Token::Ident(_))));
        stream.advance();

        let result = parse_scalar_type(&mut stream);
        eprintln!("Result: {:?}", result);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
    }
}
