//! Declaration parsers (keyword-dispatched).
//!
//! This module implements parsers for all top-level CDSL declarations:
//! - world, signal, field, operator, impulse, fracture, chronicle
//! - entity, member, strata, era
//! - type, const, config

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::foundation::{
    DeterminismPolicy, EntityId, FaultPolicy, Path, StratumId, WorldPolicy,
};
use continuum_cdsl_ast::{
    Attribute, ConfigEntry, ConstEntry, Declaration, EraDecl, RawWarmupPolicy, StratumPolicyEntry,
    TransitionDecl, TypeDecl, TypeField,
};
use continuum_cdsl_ast::{Entity, Node, RoleData, Stratum};
use continuum_cdsl_lexer::Token;

/// Parse all declarations from a token stream.
pub fn parse_declarations(stream: &mut TokenStream) -> Result<Vec<Declaration>, Vec<ParseError>> {
    let mut declarations = Vec::new();
    let mut errors = Vec::new();

    while !stream.at_end() {
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
        Some(Token::World) => parse_world(stream),
        Some(Token::Signal) => parse_signal(stream),
        Some(Token::Field) => parse_field(stream),
        Some(Token::Operator) => parse_operator(stream),
        Some(Token::Impulse) => parse_impulse(stream),
        Some(Token::Fracture) => parse_fracture(stream),
        Some(Token::Chronicle) => parse_chronicle(stream),
        Some(Token::Entity) => parse_entity(stream),
        Some(Token::Member) => parse_member(stream),
        Some(Token::Strata) => parse_stratum(stream),
        Some(Token::Era) => parse_era(stream),
        Some(Token::Type) => parse_type_decl(stream),
        Some(Token::Const) => parse_const_block(stream),
        Some(Token::Config) => parse_config_block(stream),
        other => Err(ParseError::unexpected_token(
            other,
            "at declaration",
            stream.current_span(),
        )),
    }
}

/// Parse world declaration.
fn parse_world(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::World)?;

    let path = super::types::parse_path(stream)?;

    // Parse optional header attributes (before body)
    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

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
            Some(Token::Ident(name)) if name == "policy" => {
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

/// Parse attribute: `:name` or `:name(args)`.
pub fn parse_attribute(stream: &mut TokenStream) -> Result<Attribute, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Colon)?;

    let name = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => s.clone(),
            Some(Token::Dt) => "dt".to_string(),
            other => {
                return Err(ParseError::unexpected_token(other, "attribute name", span));
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
        name,
        args,
        span: stream.span_from(start),
    })
}

/// Parse signal declaration.
fn parse_signal(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Signal)?;

    let path = super::types::parse_path(stream)?;

    // Parse optional header attributes
    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    // Parse optional type annotation
    let type_expr = if matches!(stream.peek(), Some(Token::Colon))
        && !matches!(stream.peek_nth(1), Some(Token::Ident(_)))
    {
        // This is a type annotation, not an attribute
        // Actually, attributes are `:name` so this won't conflict
        // Type annotations come after attributes in the form `: TypeExpr`
        None // TODO: Handle type annotations properly
    } else {
        None
    };

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(path, stream.span_from(start), RoleData::Signal, ());
    node.type_expr = type_expr;
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}

/// Parse field declaration.
fn parse_field(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Field)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(
        path,
        stream.span_from(start),
        RoleData::Field {
            reconstruction: None,
        },
        (),
    );
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}

/// Parse operator declaration.
fn parse_operator(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Operator)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(path, stream.span_from(start), RoleData::Operator, ());
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}

/// Parse impulse declaration.
fn parse_impulse(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Impulse)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(
        path,
        stream.span_from(start),
        RoleData::Impulse { payload: None },
        (),
    );
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}

/// Parse fracture declaration.
fn parse_fracture(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Fracture)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(path, stream.span_from(start), RoleData::Fracture, ());
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}

/// Parse chronicle declaration.
fn parse_chronicle(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Chronicle)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(path, stream.span_from(start), RoleData::Chronicle, ());
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Node(node))
}

/// Parse entity declaration.
fn parse_entity(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Entity)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    stream.expect(Token::RBrace)?;

    let mut entity = Entity::new(
        EntityId::new(path.to_string()),
        path,
        stream.span_from(start),
    );
    entity.attributes = attributes;
    Ok(Declaration::Entity(entity))
}

/// Parse member declaration.
fn parse_member(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Member)?;

    let entity_path = super::types::parse_path(stream)?;
    stream.expect(Token::Dot)?;
    let member_path = super::types::parse_path(stream)?;

    // Determine role from next keyword
    let role = match stream.peek() {
        Some(Token::Signal) => {
            stream.advance();
            RoleData::Signal
        }
        Some(Token::Field) => {
            stream.advance();
            RoleData::Field {
                reconstruction: None,
            }
        }
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "member role (signal/field)",
                stream.current_span(),
            ));
        }
    };

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
    let execution_blocks = super::blocks::parse_execution_blocks(stream)?;
    stream.expect(Token::RBrace)?;

    let mut node = Node::new(
        member_path,
        stream.span_from(start),
        role,
        EntityId::new(entity_path.to_string()),
    );
    node.attributes = attributes;
    node.execution_blocks = execution_blocks;

    Ok(Declaration::Member(node))
}

/// Parse stratum declaration.
fn parse_stratum(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Strata)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;
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
fn parse_era(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Era)?;

    let path = super::types::parse_path(stream)?;

    let mut attributes = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attributes.push(parse_attribute(stream)?);
    }

    stream.expect(Token::LBrace)?;

    let mut strata_policy = Vec::new();
    let mut transitions = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        match stream.peek() {
            Some(Token::Colon) => {
                attributes.push(parse_attribute(stream)?);
            }
            Some(Token::Ident(name)) if name == "strata" => {
                strata_policy = parse_strata_policy_block(stream)?;
            }
            Some(Token::Ident(name)) if name == "transition" => {
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
        let stratum = super::types::parse_path(stream)?;
        stream.expect(Token::Colon)?;

        let state_name = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(name)) => name.clone(),
                other => {
                    return Err(ParseError::unexpected_token(other, "stratum state", span));
                }
            }
        };

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
    stream.advance(); // consume "transition" identifier

    let target = super::types::parse_path(stream)?;

    stream.expect(Token::When)?;
    stream.expect(Token::LBrace)?;

    // Parse semicolon-separated conditions (same as WhenBlock)
    let mut conditions = Vec::new();
    loop {
        conditions.push(super::expr::parse_expr(stream)?);

        if !matches!(stream.peek(), Some(Token::Semicolon)) {
            break;
        }
        stream.advance(); // consume semicolon
    }

    stream.expect(Token::RBrace)?;

    Ok(TransitionDecl {
        target,
        conditions,
        span: stream.span_from(start),
    })
}

/// Parse type declaration.
fn parse_type_decl(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
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

        let type_expr = super::types::parse_type_expr(stream)?;

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

/// Parse const block.
fn parse_const_block(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    stream.expect(Token::Const)?;
    stream.expect(Token::LBrace)?;

    let mut entries = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let entry_start = stream.current_pos();
        let path = super::types::parse_path(stream)?;
        stream.expect(Token::Colon)?;
        let type_expr = super::types::parse_type_expr(stream)?;
        stream.expect(Token::Eq)?;
        let value = super::expr::parse_expr(stream)?;

        entries.push(ConstEntry {
            path,
            value,
            type_expr,
            span: stream.span_from(entry_start),
            doc: None,
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(Declaration::Const(entries))
}

/// Parse config block.
fn parse_config_block(stream: &mut TokenStream) -> Result<Declaration, ParseError> {
    stream.expect(Token::Config)?;
    stream.expect(Token::LBrace)?;

    let mut entries = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let entry_start = stream.current_pos();
        let path = super::types::parse_path(stream)?;
        stream.expect(Token::Colon)?;
        let type_expr = super::types::parse_type_expr(stream)?;

        let default = if matches!(stream.peek(), Some(Token::Eq)) {
            stream.advance();
            Some(super::expr::parse_expr(stream)?)
        } else {
            None
        };

        entries.push(ConfigEntry {
            path,
            default,
            type_expr,
            span: stream.span_from(entry_start),
            doc: None,
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(Declaration::Config(entries))
}
