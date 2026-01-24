//! Function declaration parser.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{Declaration, FunctionDecl};
use continuum_cdsl_lexer::Token;

/// Parse function declaration.
///
/// # Syntax
/// ```cdsl
/// fn.path(arg1, arg2, ...) {
///     : uses(maths.clamping)  // optional attributes
///     expr
/// }
/// ```
///
/// # Example
/// ```cdsl
/// fn.atmosphere.saturation_vapor_pressure(temperature_k) {
///     let t_celsius = temperature_k - 273.15 in
///     610.94 * maths.exp((17.625 * t_celsius) / (t_celsius + 243.04))
/// }
///
/// fn.hydrology.water_presence(water_mass_kg, reference_mass_kg) {
///     : uses(maths.clamping)
///     maths.clamp(water_mass_kg / reference_mass_kg, 0.0, 1.0)
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

    // Parse body - optional attributes followed by single expression in braces
    stream.expect(Token::LBrace)?;

    // Parse optional attributes (e.g., `: uses(maths.clamping)`)
    let attrs = super::super::helpers::parse_attributes(stream)?;

    // Parse body expression
    let body = super::super::expr::parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    Ok(Declaration::Function(FunctionDecl {
        path,
        params,
        body,
        attrs,
        span: stream.span_from(start),
        doc: None,
    }))
}
