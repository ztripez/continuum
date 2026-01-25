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

/// Get binary operator metadata (precedence, associativity, and operator enum).
///
/// Returns (precedence, associativity, op) where higher precedence = tighter binding.
/// This is the single source of truth for binary operator parsing.
fn binary_op_info(token: &Token) -> Option<(u8, Assoc, BinaryOp)> {
    match token {
        Token::Or => Some((10, Assoc::Left, BinaryOp::Or)),
        Token::And => Some((20, Assoc::Left, BinaryOp::And)),
        Token::EqEq => Some((30, Assoc::Left, BinaryOp::Eq)),
        Token::BangEq => Some((30, Assoc::Left, BinaryOp::Ne)),
        Token::Lt => Some((30, Assoc::Left, BinaryOp::Lt)),
        Token::LtEq => Some((30, Assoc::Left, BinaryOp::Le)),
        Token::Gt => Some((30, Assoc::Left, BinaryOp::Gt)),
        Token::GtEq => Some((30, Assoc::Left, BinaryOp::Ge)),
        Token::Plus => Some((40, Assoc::Left, BinaryOp::Add)),
        Token::Minus => Some((40, Assoc::Left, BinaryOp::Sub)),
        Token::Star => Some((50, Assoc::Left, BinaryOp::Mul)),
        Token::Slash => Some((50, Assoc::Left, BinaryOp::Div)),
        Token::Percent => Some((50, Assoc::Left, BinaryOp::Mod)),
        Token::Caret => Some((60, Assoc::Right, BinaryOp::Pow)),
        _ => None,
    }
}

/// Pratt parser - handles binary operators with precedence climbing.
pub(super) fn parse_pratt(stream: &mut TokenStream, min_prec: u8) -> Result<Expr, ParseError> {
    let mut left = parse_prefix(stream)?;

    while let Some(token) = stream.peek() {
        if let Some((prec, assoc, op)) = binary_op_info(token) {
            if prec < min_prec {
                break;
            }

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

/// Parse postfix expressions (field access, function calls, method calls).
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

                // Check for method call syntax: `.method(...)`
                if matches!(stream.peek(), Some(Token::LParen)) {
                    let args = parse_call_args(stream)?;
                    let span = expr.span;

                    // Check if this looks like a namespaced function call: path.function(args)
                    // If the base expression is a simple path (identifier or dotted path),
                    // preserve the full path and let the resolver determine if it's a
                    // function call or method call.
                    if let Some(namespace_path) = expr.as_path() {
                        // Preserve as namespace.function call for:
                        // - Kernel namespaces (dt, maths, etc.)
                        // - Single-segment identifiers that could be user function namespaces
                        // The resolver will distinguish between functions and method calls
                        let full_path = namespace_path.append(field.as_ref());
                        expr = Expr::new(
                            UntypedKind::Call {
                                func: full_path,
                                args,
                            },
                            span,
                        );
                        continue;
                    }

                    // Non-path base expression (e.g., computed value): method call
                    // obj.method(args) becomes method(obj, args)
                    let mut method_args = args;
                    method_args.insert(0, expr);

                    expr = Expr::new(
                        UntypedKind::Call {
                            func: continuum_cdsl_ast::foundation::Path::from(field.as_ref()),
                            args: method_args,
                        },
                        span,
                    );
                } else {
                    // Regular field access
                    let span = expr.span;
                    expr = Expr::new(
                        UntypedKind::FieldAccess {
                            object: Box::new(expr),
                            field: field.to_string(),
                        },
                        span,
                    );
                }
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
            Some(Token::LBrace) => {
                // Struct literal: type_name { field1: value1, field2: value2 }
                // Disambiguate from block expressions by checking for field_name: pattern
                // Block expressions start with { <expr> or { let or { if
                // Struct literals start with { ident :

                if !expr.as_path().is_some() {
                    // Not a path, can't be struct literal
                    break;
                }

                // Lookahead: check if pattern matches struct literal
                // We need: { ident :
                let is_struct = stream
                    .peek_nth(1)
                    .is_some_and(|t| matches!(t, Token::Ident(_)))
                    && stream
                        .peek_nth(2)
                        .is_some_and(|t| matches!(t, Token::Colon));

                if !is_struct {
                    // This is a block expression, not a struct literal
                    break;
                }

                // Parse as struct literal
                let ty = expr.as_path().unwrap(); // Safe: checked above
                let fields = parse_struct_fields(stream)?;
                let span = expr.span;

                expr = Expr::new(UntypedKind::Struct { ty, fields }, span);
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
        if matches!(stream.peek(), Some(Token::RParen)) {
            break;
        }
        stream.expect(Token::Comma)?;
    }

    stream.expect(Token::RParen)?;
    Ok(args)
}

/// Parse struct literal fields: `{ field1: value1, field2: value2 }`
fn parse_struct_fields(stream: &mut TokenStream) -> Result<Vec<(String, Expr)>, ParseError> {
    stream.expect(Token::LBrace)?;

    let mut fields = Vec::new();
    while !matches!(stream.peek(), Some(Token::RBrace)) {
        // Parse field name
        let field_name = {
            let span = stream.current_span();
            match stream.advance() {
                Some(Token::Ident(s)) => s.to_string(),
                other => {
                    return Err(ParseError::unexpected_token(
                        other,
                        "field name in struct literal",
                        span,
                    ))
                }
            }
        };

        stream.expect(Token::Colon)?;
        let value = super::parse_expr(stream)?;

        fields.push((field_name, value));

        // Check for comma or closing brace
        if !matches!(stream.peek(), Some(Token::RBrace)) {
            stream.expect(Token::Comma)?;
        }
    }

    stream.expect(Token::RBrace)?;
    Ok(fields)
}
