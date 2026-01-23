//! Time-related declarations: stratum and era.

use super::{parse_attributes, ParseError, TokenStream};
use continuum_cdsl_ast::foundation::StratumId;
use continuum_cdsl_ast::{Declaration, EraDecl, Stratum, StratumPolicyEntry, TransitionDecl};
use continuum_cdsl_lexer::Token;

/// Parse stratum declaration.
pub(super) fn parse_stratum(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Strata)?;

    let path = super::super::types::parse_path(stream)?;
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    // Parse attributes inside the body
    while !matches!(stream.peek(), Some(Token::RBrace)) {
        if matches!(stream.peek(), Some(Token::Colon)) {
            attributes.push(super::parse_attribute(stream)?);
        } else {
            return Err(ParseError::unexpected_token(
                stream.peek(),
                "in stratum body",
                stream.current_span(),
            ));
        }
    }

    stream.expect(Token::RBrace)?;

    let mut stratum = Stratum::new(
        StratumId::new(path.to_string()),
        path,
        stream.span_from(start),
    );
    stratum.attributes = attributes;
    Ok(Declaration::Stratum(stratum))
}

/// Parse era declaration.
pub(super) fn parse_era(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Era)?;

    let path = super::super::types::parse_path(stream)?;
    let mut attributes = parse_attributes(stream)?;

    stream.expect(Token::LBrace)?;

    let mut strata_policy = Vec::new();
    let mut transitions = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        match stream.peek() {
            Some(Token::Colon) => {
                attributes.push(super::parse_attribute(stream)?);
            }
            Some(Token::Strata) => {
                strata_policy = parse_strata_policy_block(stream)?;
            }
            Some(Token::Transition) => {
                transitions.push(parse_transition(stream)?);
            }
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "in era body",
                    stream.current_span(),
                ));
            }
        }
    }

    stream.expect(Token::RBrace)?;

    // Extract dt from attributes
    let mut dt = None;
    for attr in &attributes {
        if attr.name == "dt" && attr.args.len() == 1 {
            dt = Some(attr.args[0].clone());
            break;
        }
    }

    Ok(Declaration::Era(EraDecl {
        path,
        dt,
        strata_policy,
        transitions,
        attributes,
        span: stream.span_from(start),
        doc: None,
    }))
}

/// Parse strata policy block.
fn parse_strata_policy_block(
    stream: &mut TokenStream,
) -> Result<Vec<StratumPolicyEntry>, ParseError> {
    stream.advance(); // consume "strata" identifier
    stream.expect(Token::LBrace)?;

    let mut entries = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let entry_start = stream.current_pos();
        let stratum = super::super::types::parse_path(stream)?;
        stream.expect(Token::Colon)?;

        let state_name = super::super::helpers::expect_ident(stream, "stratum state")?;

        entries.push(StratumPolicyEntry {
            stratum,
            state_name,
            stride: None,
            span: stream.span_from(entry_start),
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(entries)
}

/// Parse transition declaration.
fn parse_transition(stream: &mut TokenStream) -> Result<TransitionDecl, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Transition)?;

    let target = super::super::types::parse_path(stream)?;

    stream.expect(Token::When)?;
    stream.expect(Token::LBrace)?;

    // Parse semicolon-separated conditions (same as WhenBlock)
    let conditions = super::super::helpers::parse_semicolon_separated_exprs(stream)?;

    stream.expect(Token::RBrace)?;

    Ok(TransitionDecl {
        target,
        conditions,
        span: stream.span_from(start),
    })
}
