//! Block body, warmup, when, and observe parsers.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{BlockBody, ObserveBlock, ObserveWhen, Stmt, WarmupBlock, WhenBlock};
use continuum_cdsl_lexer::Token;

/// Parse execution blocks inside a primitive declaration body.
/// Returns Vec<(phase_name, body)> where phase_name is lowercase ("resolve", "collect", etc.).
pub fn parse_execution_blocks(
    stream: &mut TokenStream,
) -> Result<Vec<(String, BlockBody)>, ParseError> {
    let mut blocks = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        match stream.peek() {
            Some(Token::Resolve) => {
                blocks.push(parse_execution_block(stream, Token::Resolve, "resolve")?);
            }
            Some(Token::Collect) => {
                blocks.push(parse_execution_block(stream, Token::Collect, "collect")?);
            }
            Some(Token::Emit) => {
                blocks.push(parse_execution_block(stream, Token::Emit, "emit")?);
            }
            Some(Token::Assert) => {
                blocks.push(parse_execution_block(stream, Token::Assert, "assert")?);
            }
            Some(Token::Measure) => {
                blocks.push(parse_execution_block(stream, Token::Measure, "measure")?);
            }
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "execution block keyword (resolve, collect, emit, assert, measure)",
                    stream.current_span(),
                ));
            }
        }
    }

    Ok(blocks)
}

/// Parse a standard execution block (resolve/collect/emit/assert).
fn parse_execution_block(
    stream: &mut TokenStream,
    keyword: Token,
    name: &str,
) -> Result<(String, BlockBody), ParseError> {
    stream.expect(keyword)?;

    stream.expect(Token::LBrace)?;

    let body = parse_block_body(stream)?;

    stream.expect(Token::RBrace)?;

    Ok((name.to_string(), body))
}

/// Parse warmup block: `warmup { @attr1 @attr2 iterate { expr } }`
pub fn parse_warmup_block(stream: &mut TokenStream) -> Result<WarmupBlock, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::WarmUp)?;

    stream.expect(Token::LBrace)?;

    // Parse attributes
    let mut attrs = Vec::new();
    while matches!(stream.peek(), Some(Token::Colon)) {
        attrs.push(super::decl::parse_attribute(stream)?);
    }

    // Parse iterate block
    stream.expect(Token::Iterate)?;
    stream.expect(Token::LBrace)?;
    let iterate = super::expr::parse_expr(stream)?;
    stream.expect(Token::RBrace)?;

    stream.expect(Token::RBrace)?;

    Ok(WarmupBlock {
        attrs,
        iterate,
        span: stream.span_from(start),
    })
}

/// Parse when block: `when { condition1; condition2; ... }`
pub fn parse_when_block(stream: &mut TokenStream) -> Result<WhenBlock, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::When)?;

    stream.expect(Token::LBrace)?;

    // Parse semicolon-separated condition expressions
    let mut conditions = Vec::new();
    loop {
        conditions.push(super::expr::parse_expr(stream)?);

        if !matches!(stream.peek(), Some(Token::Semicolon)) {
            break;
        }
        stream.advance(); // consume semicolon
    }

    stream.expect(Token::RBrace)?;

    Ok(WhenBlock {
        conditions,
        span: stream.span_from(start),
    })
}

/// Parse observe block: `observe { when expr { stmts } when expr { stmts } ... }`
pub fn parse_observe_block(stream: &mut TokenStream) -> Result<ObserveBlock, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Observe)?;

    stream.expect(Token::LBrace)?;

    // Parse multiple when clauses
    let mut when_clauses = Vec::new();
    while matches!(stream.peek(), Some(Token::When)) {
        let when_start = stream.current_pos();
        stream.advance(); // consume 'when'

        // Parse condition
        let condition = super::expr::parse_expr(stream)?;

        // Parse emit block
        stream.expect(Token::LBrace)?;
        let emit_block = parse_statements(stream)?;
        stream.expect(Token::RBrace)?;

        when_clauses.push(ObserveWhen {
            condition,
            emit_block,
            span: stream.span_from(when_start),
        });
    }

    stream.expect(Token::RBrace)?;

    Ok(ObserveBlock {
        when_clauses,
        span: stream.span_from(start),
    })
}

/// Parse block body (expression or statements).
fn parse_block_body(stream: &mut TokenStream) -> Result<BlockBody, ParseError> {
    // Try to parse as statement list first
    if is_statement_start(stream.peek()) {
        let statements = parse_statements(stream)?;
        Ok(BlockBody::Statements(statements))
    } else {
        // Parse as single expression
        let expr = super::expr::parse_expr(stream)?;
        Ok(BlockBody::Expression(expr))
    }
}

/// Check if the next token can start a statement.
fn is_statement_start(token: Option<&Token>) -> bool {
    matches!(
        token,
        Some(Token::Let) | Some(Token::Ident(_)) | Some(Token::Signal) | Some(Token::Field)
    )
}

/// Parse a list of statements.
fn parse_statements(stream: &mut TokenStream) -> Result<Vec<Stmt>, ParseError> {
    let mut statements = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        statements.push(parse_statement(stream)?);
    }

    Ok(statements)
}

/// Parse a single statement.
fn parse_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    match stream.peek() {
        Some(Token::Let) => parse_let_statement(stream),
        Some(Token::Ident(_)) | Some(Token::Signal) | Some(Token::Field) => {
            // Could be assignment or expression
            // Lookahead for assignment operator
            if is_assignment(stream) {
                parse_assignment_statement(stream)
            } else {
                // Expression statement
                let expr = super::expr::parse_expr(stream)?;
                Ok(Stmt::Expr(expr))
            }
        }
        other => Err(ParseError::unexpected_token(
            other,
            "in statement",
            stream.current_span(),
        )),
    }
}

/// Check if this looks like an assignment.
fn is_assignment(stream: &TokenStream) -> bool {
    // Lookahead for '<-' token
    // This is a simplified check - in a full implementation you'd need
    // to parse the path and check for '<-'
    // For now, assume any identifier followed by more tokens could be assignment
    let mut pos = 0;
    loop {
        match stream.peek_nth(pos) {
            Some(Token::Ident(_)) | Some(Token::Dot) | Some(Token::Signal) | Some(Token::Field) => {
                pos += 1;
            }
            Some(Token::LeftArrow) => return true,
            _ => return false,
        }
    }
}

/// Parse let statement.
fn parse_let_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Let)?;

    let name = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => s.clone(),
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "variable name in let",
                    span,
                ));
            }
        }
    };

    stream.expect(Token::Eq)?;

    let value = super::expr::parse_expr(stream)?;

    Ok(Stmt::Let {
        name,
        value,
        span: stream.span_from(start),
    })
}

/// Parse assignment statement.
fn parse_assignment_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();

    let target = super::types::parse_path(stream)?;

    stream.expect(Token::LeftArrow)?;

    // Check if it's a field assignment (has position)
    // Field assignment: field.path <- position, value
    // Signal assignment: signal.path <- value
    // We need to lookahead to distinguish

    // For now, simple heuristic: if there's a comma before the end, it's field assignment
    let has_comma = {
        let mut pos = 0;
        let mut depth = 0;
        loop {
            match stream.peek_nth(pos) {
                Some(Token::LParen) | Some(Token::LBracket) | Some(Token::LBrace) => depth += 1,
                Some(Token::RParen) | Some(Token::RBracket) | Some(Token::RBrace) => {
                    if depth == 0 {
                        break false;
                    }
                    depth -= 1;
                }
                Some(Token::Comma) if depth == 0 => break true,
                None => break false,
                _ => {}
            }
            pos += 1;
        }
    };

    if has_comma {
        // Field assignment
        let position = super::expr::parse_expr(stream)?;
        stream.expect(Token::Comma)?;
        let value = super::expr::parse_expr(stream)?;

        Ok(Stmt::FieldAssign {
            target,
            position,
            value,
            span: stream.span_from(start),
        })
    } else {
        // Signal assignment
        let value = super::expr::parse_expr(stream)?;

        Ok(Stmt::SignalAssign {
            target,
            value,
            span: stream.span_from(start),
        })
    }
}
