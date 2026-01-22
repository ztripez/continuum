//! Pratt parser core - precedence climbing for binary and unary operators.

use super::super::{ParseError, TokenStream};
use super::{atoms, special};
use continuum_cdsl_ast::foundation::{BinaryOp, UnaryOp};
use continuum_cdsl_ast::{Expr, UntypedKind};
use continuum_cdsl_lexer::Token;

/// Operator associativity.
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
pub(super) fn parse_pratt(stream: &mut TokenStream, min_prec: u8) -> Result<Expr, ParseError> {
    let mut left = parse_prefix(stream)?;

    while let Some(token) = stream.peek() {
        if let Some((prec, assoc)) = binary_precedence(token) {
            if prec < min_prec {
                break;
            }

            // SAFETY: If binary_precedence returned Some, token_to_binary_op must also return Some
            // because they match on the same token set. If this fails, it's a logic error in the
            // parser implementation, not a user input error.
            let op = token_to_binary_op(token).expect(
                "internal parser error: binary_precedence and token_to_binary_op out of sync",
            );
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
        Some(Token::If) => special::parse_if(stream),
        Some(Token::Let) => special::parse_let(stream),
        _ => parse_postfix(stream),
    }
}

/// Parse unary operators.
fn parse_unary(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    let span = stream.current_span();
    let op = match stream.advance() {
        Some(Token::Minus) => UnaryOp::Neg,
        Some(Token::Not) => UnaryOp::Not,
        other => {
            return Err(ParseError::unexpected_token(other, "unary operator", span));
        }
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

/// Parse postfix expressions (field access, function calls).
fn parse_postfix(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let mut expr = atoms::parse_atom(stream)?;

    loop {
        match stream.peek() {
            Some(Token::Dot) => {
                stream.advance();
                let field = {
                    let span = stream.current_span();
                    match stream.advance() {
                        Some(Token::Ident(s)) => s.clone(),
                        Some(token) => super::super::token_utils::keyword_to_string(&token)
                            .ok_or_else(|| {
                                ParseError::unexpected_token(Some(&token), "after '.'", span)
                            })?,
                        None => {
                            return Err(ParseError::unexpected_token(None, "after '.'", span));
                        }
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
        args.push(super::parse_expr(stream)?);

        if !matches!(stream.peek(), Some(Token::RParen)) {
            stream.expect(Token::Comma)?;
        }
    }

    stream.expect(Token::RParen)?;
    Ok(args)
}
