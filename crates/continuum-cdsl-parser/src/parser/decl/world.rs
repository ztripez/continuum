//! World declaration and policy parsing.

use super::{ParseError, TokenStream, parse_attribute, parse_attributes};
use continuum_cdsl_ast::foundation::{DeterminismPolicy, FaultPolicy, WorldPolicy};
use continuum_cdsl_ast::{Declaration, RawWarmupPolicy};
use continuum_cdsl_lexer::Token;

/// Parse world declaration.
pub(super) fn parse_world(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::World)?;

    let path = super::super::types::parse_path(stream)?;

    // Parse optional header attributes (before body)
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    // Parse body content
    let mut warmup = None;
    let mut policy = WorldPolicy::default();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        match stream.peek() {
            Some(Token::Colon) => {
                attributes.push(parse_attribute(stream)?);
            }
            Some(Token::WarmUp) => {
                warmup = Some(parse_warmup_policy(stream)?);
            }
            Some(Token::Policy) => {
                policy = parse_policy_block(stream)?;
            }
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "in world body",
                    stream.current_span(),
                ));
            }
        }
    }

    stream.expect(Token::RBrace)?;

    // Extract title and version from attributes
    let mut title = None;
    let mut version = None;
    for attr in &attributes {
        match attr.name.as_str() {
            "title" if attr.args.len() == 1 => {
                if let continuum_cdsl_ast::UntypedKind::StringLiteral(s) = &attr.args[0].kind {
                    title = Some(s.clone());
                }
            }
            "version" if attr.args.len() == 1 => {
                if let continuum_cdsl_ast::UntypedKind::StringLiteral(s) = &attr.args[0].kind {
                    version = Some(s.clone());
                }
            }
            _ => {}
        }
    }

    Ok(Declaration::World(continuum_cdsl_ast::WorldDecl {
        path,
        title,
        version,
        warmup,
        attributes,
        span: stream.span_from(start),
        doc: None,
        debug: false,
        policy,
    }))
}

/// Parse warmup policy block.
fn parse_warmup_policy(stream: &mut TokenStream) -> Result<RawWarmupPolicy, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::WarmUp)?;
    stream.expect(Token::LBrace)?;

    let mut attributes = Vec::new();
    while !matches!(stream.peek(), Some(Token::RBrace)) {
        if matches!(stream.peek(), Some(Token::Colon)) {
            attributes.push(parse_attribute(stream)?);
        } else {
            return Err(ParseError::unexpected_token(
                stream.peek(),
                "in warmup block",
                stream.current_span(),
            ));
        }
    }

    stream.expect(Token::RBrace)?;

    Ok(RawWarmupPolicy {
        attributes,
        span: stream.span_from(start),
    })
}

/// Parse policy block.
fn parse_policy_block(stream: &mut TokenStream) -> Result<WorldPolicy, ParseError> {
    stream.advance(); // consume "policy" identifier
    stream.expect(Token::LBrace)?;

    let mut determinism = DeterminismPolicy::Strict;
    let mut faults = FaultPolicy::Fatal;

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let field_name = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(name)) => name.clone(),
                Some(Token::Determinism) => "determinism".to_string(),
                Some(Token::Faults) => "faults".to_string(),
                other => {
                    return Err(ParseError::unexpected_token(
                        other,
                        "policy field name",
                        span,
                    ));
                }
            }
        };

        stream.expect(Token::Colon)?;

        let value = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(val)) => val.clone(),
                other => {
                    return Err(ParseError::unexpected_token(other, "policy value", span));
                }
            }
        };

        stream.expect(Token::Semicolon)?;

        match field_name.as_str() {
            "determinism" => {
                determinism = match value.as_str() {
                    "strict" => DeterminismPolicy::Strict,
                    "relaxed" => DeterminismPolicy::Relaxed,
                    _ => {
                        return Err(ParseError::invalid_syntax(
                            format!("unknown determinism policy: {}", value),
                            stream.current_span(),
                        ));
                    }
                }
            }
            "faults" => {
                faults = match value.as_str() {
                    "fatal" => FaultPolicy::Fatal,
                    "warn" => FaultPolicy::Warn,
                    "ignore" => FaultPolicy::Ignore,
                    _ => {
                        return Err(ParseError::invalid_syntax(
                            format!("unknown fault policy: {}", value),
                            stream.current_span(),
                        ));
                    }
                }
            }
            _ => {
                return Err(ParseError::invalid_syntax(
                    format!("unknown policy field: {}", field_name),
                    stream.current_span(),
                ));
            }
        }
    }

    stream.expect(Token::RBrace)?;

    Ok(WorldPolicy {
        determinism,
        faults,
    })
}
