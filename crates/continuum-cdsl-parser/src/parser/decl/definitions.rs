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

    let name = super::super::helpers::expect_ident(stream, "type name")?;

    stream.expect(Token::LBrace)?;

    let mut fields = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let field_start = stream.current_pos();
        let field_name = super::super::helpers::expect_ident(stream, "field name")?;

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
/// Parse a const or config entry with flexible syntax.
///
/// This function handles three distinct parsing modes depending on the input syntax:
///
/// 1. **Explicit type with value**: `path: TYPE = value`
///    - Type is parsed explicitly
///    - Value is required
///    - Common for clarity or when type inference would be ambiguous
///
/// 2. **Inferred type**: `path: value`
///    - Type is inferred from the value expression during resolution
///    - Uses `TypeExpr::Infer` which is later resolved by `infer_type_from_expr()`
///    - Value is required
///
/// 3. **Type-only (config)**: `path: TYPE`
///    - Type is parsed explicitly
///    - No value provided (no default)
///    - Only valid when `allow_missing_value` is true (config blocks)
///    - Used for required configuration parameters
///
/// # Parameters
///
/// - `stream`: Token stream positioned at the start of an entry
/// - `allow_missing_value`: Whether to allow entries without values
///   - `true` for config blocks (supports type-only declarations)
///   - `false` for const blocks (value always required)
///
/// # Returns
///
/// Returns a tuple: `(path, type_expr, value, span)` where:
/// - `path`: Canonical path to the const/config value
/// - `type_expr`: Either explicit type or `TypeExpr::Infer`
/// - `value`: Optional value expression (`None` only if `allow_missing_value` is true)
/// - `span`: Source span covering the entire entry
///
/// # Errors
///
/// Returns `ParseError` if:
/// - Path syntax is invalid
/// - Missing `:` after path
/// - Type syntax is malformed (when explicit type is used)
/// - Value is missing when `allow_missing_value` is false
/// - Value expression syntax is invalid
///
/// # Examples
///
/// ```text
/// // Explicit type with value
/// gravity: Scalar<m/s²> = 9.81
///
/// // Inferred type (resolved to Scalar<1> from literal)
/// max_iterations: 100
///
/// // Inferred type with units (resolved to Scalar<m/s²>)
/// terminal_velocity: 53.0 m/s
///
/// // Type-only (config only, no default)
/// population_growth_rate: Scalar<1/yr>
/// ```
///
/// # Type Inference
///
/// When `TypeExpr::Infer` is used, the resolver performs type inference via
/// `infer_type_from_expr()` which applies these rules:
/// - `BoolLiteral` → `Type::Bool`
/// - `Literal { value, unit }` → `Type::kernel(Scalar, unit, None)`
/// - `Vector([e1, ..., en])` → `Type::kernel(Vector { dim: n }, dimensionless, None)`
/// - Complex expressions → Error (explicit type required)
///
/// # See Also
///
/// - `parse_const_block()`: Uses this with `allow_missing_value = false`
/// - `parse_config_block()`: Uses this with `allow_missing_value = true`
/// - `continuum-cdsl-resolve::infer_type_from_expr()`: Performs type inference
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
        Some(Token::Ident(name)) if super::super::token_utils::is_type_keyword(name)
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
