//! Expression parser using Pratt parsing (precedence climbing).
//!
//! This module implements a Pratt parser for CDSL expressions with proper
//! operator precedence and associativity.
//!
//! ## Precedence Levels (lowest to highest)
//!
//! 1. `||` (Or) - left associative
//! 2. `&&` (And) - left associative
//! 3. `==`, `!=`, `<`, `<=`, `>`, `>=` (Comparison) - left associative
//! 4. `+`, `-` (Addition) - left associative
//! 5. `*`, `/`, `%` (Multiplication) - left associative
//! 6. `^` (Power) - right associative
//! 7. Unary `-`, `!` - prefix
//! 8. Postfix: `.field`, `(args)` - left associative
//!
//! ## Special Forms
//!
//! - `if condition { then } else { else }` - eager select
//! - `let name = value in body` - local binding
//! - Aggregates: `agg.sum(source, body)`
//! - Spatial: `nearest(entity, pos)`, `within(entity, pos, radius)`

use super::{ParseError, TokenStream};
use crate::ast::{Expr, UntypedKind};
use crate::foundation::{AggregateOp, BinaryOp, EntityId, Path, UnaryOp};
use crate::lexer::Token;

/// Parse an expression.
pub fn parse_expr(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    parse_pratt(stream, 0)
}

/// Operator precedence and associativity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Assoc {
    Left,
    Right,
}

/// Get precedence and associativity for a binary operator token.
///
/// Returns (precedence, associativity) where higher precedence = tighter binding.
fn binary_precedence(token: &Token) -> Option<(u8, Assoc)> {
    match token {
        Token::Or => Some((10, Assoc::Left)),
        Token::And => Some((20, Assoc::Left)),
        Token::EqEq | Token::BangEq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => {
            Some((30, Assoc::Left))
        }
        Token::Plus | Token::Minus => Some((40, Assoc::Left)),
        Token::Star | Token::Slash | Token::Percent => Some((50, Assoc::Left)),
        Token::Caret => Some((60, Assoc::Right)),
        _ => None,
    }
}

/// Convert token to BinaryOp.
fn token_to_binary_op(token: &Token) -> Option<BinaryOp> {
    match token {
        Token::Plus => Some(BinaryOp::Add),
        Token::Minus => Some(BinaryOp::Sub),
        Token::Star => Some(BinaryOp::Mul),
        Token::Slash => Some(BinaryOp::Div),
        Token::Percent => Some(BinaryOp::Mod),
        Token::Caret => Some(BinaryOp::Pow),
        Token::EqEq => Some(BinaryOp::Eq),
        Token::BangEq => Some(BinaryOp::Ne),
        Token::Lt => Some(BinaryOp::Lt),
        Token::LtEq => Some(BinaryOp::Le),
        Token::Gt => Some(BinaryOp::Gt),
        Token::GtEq => Some(BinaryOp::Ge),
        Token::And => Some(BinaryOp::And),
        Token::Or => Some(BinaryOp::Or),
        _ => None,
    }
}

/// Pratt parser - handles binary operators with precedence climbing.
fn parse_pratt(stream: &mut TokenStream, min_prec: u8) -> Result<Expr, ParseError> {
    let mut left = parse_prefix(stream)?;

    while let Some(token) = stream.peek() {
        if let Some((prec, assoc)) = binary_precedence(token) {
            if prec < min_prec {
                break;
            }

            let op = token_to_binary_op(token).unwrap();
            let span_start = stream.current_pos();
            stream.advance();

            let next_prec = if assoc == Assoc::Left { prec + 1 } else { prec };
            let right = parse_pratt(stream, next_prec)?;

            let span = stream.span_from(span_start);
            left = Expr::new(
                UntypedKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        } else {
            break;
        }
    }

    Ok(left)
}

/// Parse prefix expressions (unary operators, special forms, atoms).
fn parse_prefix(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    match stream.peek() {
        Some(Token::Minus) | Some(Token::Not) => parse_unary(stream),
        Some(Token::If) => parse_if(stream),
        Some(Token::Let) => parse_let(stream),
        _ => parse_postfix(stream),
    }
}

/// Parse unary operators.
fn parse_unary(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    let op = match stream.advance() {
        Some(Token::Minus) => UnaryOp::Neg,
        Some(Token::Not) => UnaryOp::Not,
        _ => unreachable!(),
    };

    let operand = parse_prefix(stream)?;
    let span = stream.span_from(start);

    Ok(Expr::new(
        UntypedKind::Unary {
            op,
            operand: Box::new(operand),
        },
        span,
    ))
}

/// Parse if-then-else expression.
fn parse_if(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::If)?;

    let condition = parse_expr(stream)?;

    stream.expect(Token::LBrace)?;
    let then_branch = parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    stream.expect(Token::Else)?;

    stream.expect(Token::LBrace)?;
    let else_branch = parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    let span = stream.span_from(start);
    Ok(Expr::new(
        UntypedKind::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        span,
    ))
}

/// Parse let-in expression.
fn parse_let(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Let)?;

    let name = match stream.advance() {
        Some(Token::Ident(s)) => s.clone(),
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "in let binding",
                stream.current_span(),
            ));
        }
    };

    stream.expect(Token::Eq)?;
    let value = parse_expr(stream)?;

    stream.expect(Token::In)?;
    let body = parse_expr(stream)?;

    let span = stream.span_from(start);
    Ok(Expr::new(
        UntypedKind::Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
        },
        span,
    ))
}

/// Parse postfix expressions (field access, function calls).
fn parse_postfix(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let mut expr = parse_atom(stream)?;

    loop {
        match stream.peek() {
            Some(Token::Dot) => {
                stream.advance();
                let field = match stream.advance() {
                    Some(Token::Ident(s)) => s.clone(),
                    Some(Token::Config) => "config".to_string(),
                    Some(Token::Const) => "const".to_string(),
                    Some(Token::Signal) => "signal".to_string(),
                    Some(Token::Field) => "field".to_string(),
                    Some(Token::Entity) => "entity".to_string(),
                    Some(Token::Dt) => "dt".to_string(),
                    Some(Token::Strata) => "strata".to_string(),
                    Some(Token::Type) => "type".to_string(),
                    other => {
                        return Err(ParseError::unexpected_token(
                            other,
                            "after '.'",
                            stream.current_span(),
                        ));
                    }
                };
                let span = expr.span;
                expr = Expr::new(
                    UntypedKind::FieldAccess {
                        object: Box::new(expr),
                        field,
                    },
                    span,
                );
            }
            Some(Token::LParen) => {
                let args = parse_call_args(stream)?;
                let span = expr.span;

                // Convert to Call if expr is a path
                if let Some(path) = expr.as_path() {
                    expr = Expr::new(UntypedKind::Call { func: path, args }, span);
                } else {
                    return Err(ParseError::invalid_syntax(
                        "dynamic dispatch not supported",
                        span,
                    ));
                }
            }
            _ => break,
        }
    }

    Ok(expr)
}

/// Parse function call arguments.
fn parse_call_args(stream: &mut TokenStream) -> Result<Vec<Expr>, ParseError> {
    stream.expect(Token::LParen)?;

    let mut args = Vec::new();
    while !matches!(stream.peek(), Some(Token::RParen)) {
        args.push(parse_expr(stream)?);

        if !matches!(stream.peek(), Some(Token::RParen)) {
            stream.expect(Token::Comma)?;
        }
    }

    stream.expect(Token::RParen)?;
    Ok(args)
}

/// Parse atomic expressions (literals, identifiers, special keywords).
fn parse_atom(stream: &mut TokenStream) -> Result<Expr, ParseError> {
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
            if let Some(Token::String(s)) = stream.advance() {
                Ok(Expr::new(
                    UntypedKind::StringLiteral(s.clone()),
                    stream.span_from(start),
                ))
            } else {
                unreachable!()
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
        Some(Token::Dt) => {
            stream.advance();
            Ok(Expr::new(UntypedKind::Dt, stream.span_from(start)))
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
        Some(Token::Filter) => parse_filter(stream),
        Some(Token::Nearest) => parse_nearest(stream),
        Some(Token::Within) => parse_within(stream),
        Some(Token::Pairs) => parse_pairs(stream),
        Some(Token::Agg) => parse_aggregate(stream),
        Some(Token::Entity) => parse_entity_reference(stream),
        Some(Token::Ident(_))
        | Some(Token::Config)
        | Some(Token::Const)
        | Some(Token::Signal)
        | Some(Token::Field) => parse_identifier(stream),
        other => Err(ParseError::unexpected_token(
            other,
            "in expression",
            stream.current_span(),
        )),
    }
}

/// Parse numeric literal with optional unit.
fn parse_numeric_literal(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();

    let value = match stream.advance() {
        Some(Token::Integer(n)) => *n as f64,
        Some(Token::Float(f)) => *f,
        _ => unreachable!(),
    };

    // Check for optional unit: <unit>
    let unit = if matches!(stream.peek(), Some(Token::Lt)) {
        stream.advance();
        let unit_expr = super::types::parse_unit_expr(stream)?;
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
        elements.push(parse_expr(stream)?);

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
    let expr = parse_expr(stream)?;
    stream.expect(Token::RParen)?;
    Ok(expr)
}

/// Parse identifier or path.
fn parse_identifier(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();

    let name = match stream.advance() {
        Some(Token::Ident(s)) => s.clone(),
        Some(Token::Config) => "config".to_string(),
        Some(Token::Const) => "const".to_string(),
        Some(Token::Signal) => "signal".to_string(),
        Some(Token::Field) => "field".to_string(),
        Some(Token::Dt) => "dt".to_string(),
        _ => unreachable!(),
    };

    Ok(Expr::new(UntypedKind::Local(name), stream.span_from(start)))
}

/// Parse `other` keyword or `other(entity)`.
fn parse_other_or_field_access(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Other)?;

    // Check if it's other(entity) or just `other`
    if matches!(stream.peek(), Some(Token::LParen)) {
        stream.advance();
        let path = super::types::parse_path(stream)?;
        stream.expect(Token::RParen)?;
        Ok(Expr::new(
            UntypedKind::OtherInstances(EntityId::new(path.to_string())),
            stream.span_from(start),
        ))
    } else {
        Ok(Expr::new(UntypedKind::Other, stream.span_from(start)))
    }
}

/// Parse filter(source, predicate).
fn parse_filter(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Filter)?;
    stream.expect(Token::LParen)?;

    let source = parse_expr(stream)?;
    stream.expect(Token::Comma)?;
    let predicate = parse_expr(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Filter {
            source: Box::new(source),
            predicate: Box::new(predicate),
        },
        stream.span_from(start),
    ))
}

/// Parse nearest(entity, position).
fn parse_nearest(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Nearest)?;
    stream.expect(Token::LParen)?;

    let entity_path = super::types::parse_path(stream)?;
    stream.expect(Token::Comma)?;
    let position = parse_expr(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Nearest {
            entity: EntityId::new(entity_path.to_string()),
            position: Box::new(position),
        },
        stream.span_from(start),
    ))
}

/// Parse within(entity, position, radius).
fn parse_within(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Within)?;
    stream.expect(Token::LParen)?;

    let entity_path = super::types::parse_path(stream)?;
    stream.expect(Token::Comma)?;
    let position = parse_expr(stream)?;
    stream.expect(Token::Comma)?;
    let radius = parse_expr(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Within {
            entity: EntityId::new(entity_path.to_string()),
            position: Box::new(position),
            radius: Box::new(radius),
        },
        stream.span_from(start),
    ))
}

/// Parse pairs(entity).
fn parse_pairs(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Pairs)?;
    stream.expect(Token::LParen)?;

    let entity_path = super::types::parse_path(stream)?;

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::PairsInstances(EntityId::new(entity_path.to_string())),
        stream.span_from(start),
    ))
}

/// Parse aggregate: agg.op(source) or agg.op(source, body).
fn parse_aggregate(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Agg)?;
    stream.expect(Token::Dot)?;

    let op = match stream.advance() {
        Some(Token::Ident(s)) => match s.as_str() {
            "sum" => AggregateOp::Sum,
            "product" => AggregateOp::Product,
            "min" => AggregateOp::Min,
            "max" => AggregateOp::Max,
            "mean" => AggregateOp::Mean,
            "count" => AggregateOp::Count,
            "any" => AggregateOp::Any,
            "all" => AggregateOp::All,
            "none" => AggregateOp::None,
            "map" => AggregateOp::Map,
            "first" => AggregateOp::First,
            _ => {
                return Err(ParseError::invalid_syntax(
                    format!("unknown aggregate operation: {}", s),
                    stream.current_span(),
                ));
            }
        },
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "aggregate operation",
                stream.current_span(),
            ));
        }
    };

    stream.expect(Token::LParen)?;
    let source = parse_expr(stream)?;

    // count() takes only source, others take source + body
    let body = if op == AggregateOp::Count {
        if matches!(stream.peek(), Some(Token::RParen)) {
            // count(source) - use default body
            Expr::new(UntypedKind::BoolLiteral(true), stream.current_span())
        } else {
            stream.expect(Token::Comma)?;
            parse_expr(stream)?
        }
    } else {
        stream.expect(Token::Comma)?;
        parse_expr(stream)?
    };

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Aggregate {
            op,
            source: Box::new(source),
            binding: "self".to_string(),
            body: Box::new(body),
        },
        stream.span_from(start),
    ))
}

/// Parse entity.path reference.
fn parse_entity_reference(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Entity)?;
    stream.expect(Token::Dot)?;

    let path = super::types::parse_path(stream)?;

    Ok(Expr::new(
        UntypedKind::Entity(EntityId::new(path.to_string())),
        stream.span_from(start),
    ))
}
