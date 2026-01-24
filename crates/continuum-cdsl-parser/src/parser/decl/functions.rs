//! Function declaration parser.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{Declaration, FunctionDecl};
use continuum_cdsl_lexer::Token;

/// Parse function declaration.
///
/// # Syntax
/// ```cdsl
/// fn.path(arg1, arg2, ...) { expr }
/// ```
///
/// # Example
/// ```cdsl
/// fn.atmosphere.saturation_vapor_pressure(temperature_k) {
///     let t_celsius = temperature_k - 273.15 in
///     610.94 * maths.exp((17.625 * t_celsius) / (t_celsius + 243.04))
/// }
/// ```
pub(super) fn parse_function_decl(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Fn)?;

    // Parse function path
    let path = super::super::types::parse_path(stream)?;

    // Parse parameter list
    stream.expect(Token::LParen)?;
    let mut params = Vec::new();

    if !matches!(stream.peek(), Some(Token::RParen)) {
        loop {
            let param = super::super::helpers::expect_ident(stream, "parameter name")?;
            params.push(param);

            if matches!(stream.peek(), Some(Token::RParen)) {
                break;
            }
            stream.expect(Token::Comma)?;
        }
    }

    stream.expect(Token::RParen)?;

    // Parse body - single expression in braces
    stream.expect(Token::LBrace)?;
    let body = super::super::expr::parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    Ok(Declaration::Function(FunctionDecl {
        path,
        params,
        body,
        span: stream.span_from(start),
        doc: None,
    }))
}
