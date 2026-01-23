//! Declaration parsers (keyword-dispatched).
//!
//! This module implements parsers for all top-level CDSL declarations.
//! Organized by semantic grouping:
//! - `primitives` - signal, field, operator, impulse, fracture, chronicle
//! - `entities` - entity, member
//! - `time` - stratum, era
//! - `definitions` - type, const, config
//! - `world` - world, warmup, policy

mod definitions;
mod entities;
mod functions;
mod primitives;
mod time;
mod world;

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{Attribute, Declaration, Node, RoleData};
use continuum_cdsl_lexer::Token;

/// Parse all declarations from a token stream.
pub fn parse_declarations(stream: &mut TokenStream) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let mut declarations = Vec::new();
    let mut errors = Vec::new();

    while !stream.at_end() {
        // Skip doc comments at declaration level (not yet attached to declarations)
        while matches!(stream.peek(), Some(Token::DocComment(_))) {
            stream.advance();
        }

        if stream.at_end() {
            break;
        }

        match parse_declaration(stream) {
            Ok(decl) => declarations.push(decl),
            Err(e) => {
                errors.push(e);
                stream.synchronize(); // Skip to next declaration
            }
        }
    }

    if errors.is_empty() {
        Ok(declarations)
    } else {
        Err(errors)
    }
}

/// Parse a single declaration (keyword-dispatched).
fn parse_declaration(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    match stream.peek() {
        Some(Token::World) => world::parse_world(stream),
        Some(Token::Signal) => primitives::parse_signal(stream),
        Some(Token::Field) => primitives::parse_field(stream),
        Some(Token::Operator) => primitives::parse_operator(stream),
        Some(Token::Impulse) => primitives::parse_impulse(stream),
        Some(Token::Fracture) => primitives::parse_fracture(stream),
        Some(Token::Chronicle) => primitives::parse_chronicle(stream),
        Some(Token::Analyzer) => parse_analyzer_stub(stream),
        Some(Token::Entity) => entities::parse_entity(stream),
        Some(Token::Strata) => time::parse_stratum(stream),
        Some(Token::Era) => time::parse_era(stream),
        Some(Token::Type) => definitions::parse_type_decl(stream),
        Some(Token::Const) => definitions::parse_const_block(stream),
        Some(Token::Config) => definitions::parse_config_block(stream),
        Some(Token::Fn) => functions::parse_function_decl(stream),
        other => Err(ParseError::unexpected_token(
            other,
            "at declaration",
            stream.current_span(),
        )),
    }
}

/// Stub parser for `analyzer` declarations (not yet implemented).
///
/// This allows terra CDSL files with analyzer blocks to parse without error.
/// The analyzer declaration is consumed and skipped entirely.
///
/// Returns an error that will be handled by parse_declarations (synchronize and continue).
fn parse_analyzer_stub(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start_span = stream.current_span();
    stream.advance(); // consume 'analyzer' keyword

    // Skip analyzer name (path)
    while matches!(stream.peek(), Some(Token::Ident(_))) {
        stream.advance();
        if matches!(stream.peek(), Some(Token::Dot)) {
            stream.advance();
        } else {
            break;
        }
    }

    // Skip entire block
    if matches!(stream.peek(), Some(Token::LBrace)) {
        stream.advance();
        let mut depth = 1;
        while depth > 0 && !stream.at_end() {
            match stream.peek() {
                Some(Token::LBrace) => {
                    depth += 1;
                    stream.advance();
                }
                Some(Token::RBrace) => {
                    depth -= 1;
                    stream.advance();
                }
                _ => {
                    stream.advance();
                }
            }
        }
    }

    // Return error to skip this declaration
    // parse_declarations will call synchronize() and continue
    Err(ParseError::invalid_syntax(
        "analyzer declarations not yet implemented (skipped)".to_string(),
        start_span,
    ))
}

// ============================================================================
// Common Helpers
// ============================================================================

/// Skip config/const blocks that appear inside node declarations.
///
/// Terra CDSL uses inline config blocks inside signals to define per-node configuration:
/// ```cdsl
/// signal foo.bar {
///     config {
///         default_value: 42.0
///     }
///     resolve { ... }
/// }
/// ```
///
/// These are currently skipped as the Node AST doesn't support inline config.
/// The config values are instead accessed via `config.foo.bar.default_value` in expressions.
fn skip_nested_config_const(stream: &mut TokenStream) -> Result<(), ParseError> {
    while matches!(stream.peek(), Some(Token::Config) | Some(Token::Const)) {
        stream.advance(); // consume config/const keyword

        if matches!(stream.peek(), Some(Token::LBrace)) {
            stream.advance(); // consume '{'
            let mut depth = 1;
            while depth > 0 && !stream.at_end() {
                match stream.peek() {
                    Some(Token::LBrace) => {
                        depth += 1;
                        stream.advance();
                    }
                    Some(Token::RBrace) => {
                        depth -= 1;
                        stream.advance();
                    }
                    _ => {
                        stream.advance();
                    }
                }
            }
        }
    }
    Ok(())
}

/// Parse attribute: `:name` or `:name(args)`.
pub(super) fn parse_attribute(stream: &mut TokenStream) -> Result<Attribute, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Colon)?;

    let name = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => s.clone(),
            Some(token) => super::token_utils::keyword_to_string(&token).ok_or_else(|| {
                ParseError::unexpected_token(Some(&token), "attribute name", span)
            })?,
            None => {
                return Err(ParseError::unexpected_token(None, "attribute name", span));
            }
        }
    };

    let args = if matches!(stream.peek(), Some(Token::LParen)) {
        stream.advance();
        let mut args = Vec::new();

        if !matches!(stream.peek(), Some(Token::RParen)) {
            loop {
                args.push(super::expr::parse_expr(stream)?);
                if matches!(stream.peek(), Some(Token::RParen)) {
                    break;
                }
                stream.expect(Token::Comma)?;
            }
        }

        stream.expect(Token::RParen)?;
        args
    } else {
        Vec::new()
    };

    Ok(Attribute {
        name: name.to_string(),
        args,
        span: stream.span_from(start),
    })
}

/// Parse multiple attributes (repeated pattern across declarations).
pub(super) fn parse_attributes(stream: &mut TokenStream) -> Result<Vec<Attribute>, ParseError> {
    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }
    Ok(attributes)
}

/// Parse a node declaration (signal, field, operator, impulse, fracture, chronicle).
///
/// All node declarations follow the pattern:
/// ```text
/// <keyword> <path> { [: TypeExpr] [attributes] [special_blocks] [execution_blocks] }
/// ```
///
/// # Type Expression Detection
///
/// Type expressions can appear as the first element inside the body:
/// ```text
/// signal foo.bar {
///     : Scalar<m>      # Type expression (detected via is_type_keyword)
///     : strata(fast)   # Regular attribute
///     resolve { ... }
/// }
/// ```
///
/// This function uses **semantic lookahead** (`peek_nth(1)`) to distinguish type
/// expressions from named attributes. After seeing `:`, it checks if the next token
/// is a type keyword (`Bool`, `Scalar`, `Vec2`, etc.) via [`is_type_keyword()`].
///
/// **Rationale for semantic lookahead in parser**:
/// - Grammar is inherently ambiguous: both `:Scalar` and `:strata` start with `:`
/// - Cannot be resolved syntactically without unlimited lookahead (`<` may be far ahead)
/// - Resolver cannot disambiguate because both forms are valid AST attributes
/// - Parser-level disambiguation required to populate correct `Node.type_expr` field
///
/// # Parameters
///
/// - `stream`: Token stream positioned at the keyword token
/// - `keyword`: The expected declaration keyword (Signal, Field, Operator, etc.)
/// - `role`: The role data to associate with the parsed node
///
/// # Returns
///
/// `Declaration::Node` containing:
/// - Parsed path
/// - Optional type expression (if `: TypeKeyword<...>` present)
/// - Attributes (`:name` or `:name(args)`)
/// - Special blocks (`when`, `warmup`, `observe`)
/// - Execution blocks (`resolve`, `collect`, `emit`, `assert`, `measure`)
///
/// # Errors
///
/// Returns `ParseError` if:
/// - Keyword token doesn't match `keyword` parameter
/// - Path syntax is invalid
/// - Type expression syntax is malformed
/// - Attribute syntax is invalid
/// - Special block syntax is invalid
/// - Execution block syntax is invalid
/// - Missing closing `}`
///
/// # Examples
///
/// ```text
/// // Type expression detected via lookahead
/// signal plate.velocity {
///     : Vec3<m/s>      # is_type_keyword("Vec3") → true, parse as type
///     : strata(fast)   # is_type_keyword("strata") → false, parse as attribute
///     resolve { ... }
/// }
///
/// // Fracture with special blocks
/// fracture overstress {
///     when { tension > 100.0 }
///     collect { ... }
/// }
/// ```
///
/// # See Also
///
/// - [`is_type_keyword()`]: Determines if identifier is a type keyword (semantic check)
/// - [`parse_type_expr()`]: Parses type expression syntax
/// - [`parse_attribute()`]: Parses individual attributes
/// - [`parse_when_block()`], [`parse_warmup_block()`], [`parse_observe_block()`]: Special blocks
pub(super) fn parse_node_declaration(
    stream: &mut TokenStream,
    keyword: Token,
    role: RoleData,
) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(keyword)?;

    let path = super::types::parse_path(stream)?;
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    // Check for type expression as first attribute: `: Scalar<unit>`
    //
    // RATIONALE FOR SEMANTIC LOOKAHEAD:
    // The DSL allows both `: TypeExpr` and `: attr_name(...)` after `{`.
    // Syntactically indistinguishable without infinite lookahead (need to see `<` or `(`).
    // We use is_type_keyword() to perform semantic disambiguation at parse time.
    let type_expr = super::helpers::try_parse_type_expr(stream)?;

    // Parse remaining attributes inside the body (before special/execution blocks)
    attributes.extend(super::helpers::parse_attributes(stream)?);

    // Skip config/const blocks inside primitives (not yet implemented in AST)
    // These can appear before or after special blocks
    skip_nested_config_const(stream)?;

    // Parse special blocks (when, warmup, observe)
    let special = super::helpers::parse_special_blocks(stream)?;

    // Skip any additional config/const blocks after special blocks
    skip_nested_config_const(stream)?;

    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(path, stream.span_from(start), role, ());
    node.attributes = attributes;
    node.type_expr = type_expr;
    node.when = special.when;
    node.warmup = special.warmup;
    node.observe = special.observe;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}
