//! Block body, warmup, when, and observe parsers.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{BlockBody, ObserveBlock, ObserveWhen, Stmt, WarmupBlock, WhenBlock};
use continuum_cdsl_lexer::Token;

/// Parse execution blocks inside a primitive declaration body.
///
/// Parses zero or more phase-tagged execution blocks within a declaration
/// (e.g., operator, signal, field). Each block is tagged with a phase name
/// that determines when it executes in the simulation lifecycle.
///
/// # Arguments
///
/// * `stream` - Token stream positioned after the opening `{` of the declaration body,
///   before the first execution block keyword or closing `}`.
///
/// # Returns
///
/// * `Ok(blocks)` - Vector of `(phase_name, body)` tuples where:
///   - `phase_name`: lowercase phase identifier ("resolve", "collect", "measure", etc.)
///   - `body`: parsed block body (expression or statement list)
/// * `Err(error)` - Parse error if block syntax is invalid or phase keyword is unrecognized
///
/// # Examples
///
/// ```cdsl
/// operator example {
///   resolve { x + y }           // Single expression body
///   collect { signal.x <- 10 }  // Statement list body
/// }
/// ```
///
/// # Notes
///
/// - Phase names are normalized to lowercase for consistent matching
/// - Stops parsing when it encounters `}` (end of declaration body)
/// - Returns empty vector if no execution blocks present
pub fn parse_execution_blocks(
    stream: &mut TokenStream,
) -> Result<Vec<(String, BlockBody)>, ParseError> {
    let mut blocks = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        let token = stream.peek();

        if let Some(tok) = token {
            if let Some(block_name) = super::token_utils::execution_block_name(tok) {
                // Clone token and name before advancing stream
                let keyword_token = tok.clone();

                if matches!(keyword_token, Token::WarmUp) {
                    stream.expect(Token::WarmUp)?;
                    stream.expect(Token::LBrace)?;

                    while matches!(stream.peek(), Some(Token::Colon)) {
                        stream.advance();
                        while !matches!(
                            stream.peek(),
                            Some(Token::Iterate) | Some(Token::Colon) | Some(Token::RBrace)
                        ) {
                            stream.advance();
                        }
                    }

                    let body = if matches!(stream.peek(), Some(Token::Iterate)) {
                        stream.expect(Token::Iterate)?;
                        stream.expect(Token::LBrace)?;
                        let iterate_body = parse_block_body(stream)?;
                        stream.expect(Token::RBrace)?;
                        iterate_body
                    } else {
                        BlockBody::Statements(Vec::new())
                    };

                    stream.expect(Token::RBrace)?;
                    blocks.push((block_name.to_string(), body));
                } else {
                    let is_assert = matches!(keyword_token, Token::Assert);
                    blocks.push(parse_execution_block(
                        stream,
                        keyword_token,
                        block_name,
                        is_assert,
                    )?);
                }
            } else {
                return Err(ParseError::unexpected_token(
                    token,
                    "execution block keyword (resolve, collect, emit, assert, measure, warmup)",
                    stream.current_span(),
                ));
            }
        } else {
            return Err(ParseError::unexpected_token(
                None,
                "execution block keyword (resolve, collect, emit, assert, measure, warmup)",
                stream.current_span(),
            ));
        }
    }

    Ok(blocks)
}

/// Parse a standard execution block (resolve/collect/emit/assert).
fn parse_execution_block(
    stream: &mut TokenStream,
    keyword: Token,
    name: &str,
    is_assert: bool,
) -> Result<(String, BlockBody), ParseError> {
    stream.expect(keyword)?;

    stream.expect(Token::LBrace)?;

    let body = if is_assert {
        // Parse assert block with metadata support
        parse_assert_block_body(stream)?
    } else {
        parse_block_body(stream)?
    };

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
    if is_statement_start(stream) {
        let statements = parse_statements(stream)?;
        Ok(BlockBody::Statements(statements))
    } else {
        // Parse as single expression
        let expr = super::expr::parse_expr(stream)?;
        Ok(BlockBody::Expression(expr))
    }
}

/// Parse assert block body as a list of assertion statements.
///
/// Each assertion can have optional severity and message metadata:
/// ```cdsl
/// assert {
///     x > 0;                                      // Basic assertion
///     y < 100 : fatal;                            // With severity
///     z != 0 : "z cannot be zero";               // With message
///     w > 0 : fatal, "w must be positive";       // Both
/// }
/// ```
fn parse_assert_block_body(stream: &mut TokenStream) -> Result<BlockBody, ParseError> {
    let mut assertions = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        assertions.push(parse_assertion_statement(stream)?);

        // Consume optional semicolon
        if matches!(stream.peek(), Some(Token::Semicolon)) {
            stream.advance();
        }
    }

    Ok(BlockBody::Statements(assertions))
}

/// Parse a single assertion statement with optional metadata.
///
/// Syntax:
/// - `condition`
/// - `condition : severity`
/// - `condition : "message"`
/// - `condition : severity, "message"`
fn parse_assertion_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();

    // Parse condition expression
    let condition = super::expr::parse_expr(stream)?;

    // Check for optional metadata after ':'
    let (severity, message) = if matches!(stream.peek(), Some(Token::Colon)) {
        stream.advance(); // consume ':'
        parse_assertion_metadata(stream)?
    } else {
        (None, None)
    };

    Ok(Stmt::Assert {
        condition,
        severity,
        message,
        span: stream.span_from(start),
    })
}

/// Parse assertion metadata: severity and/or message.
///
/// Valid patterns:
/// - `fatal` - severity only
/// - `"message"` - message only
/// - `fatal, "message"` - both (severity first)
fn parse_assertion_metadata(
    stream: &mut TokenStream,
) -> Result<(Option<String>, Option<String>), ParseError> {
    // Parse first item (severity or message)
    let (mut severity, mut message) = match stream.peek() {
        Some(Token::Ident(name)) => (Some(name.clone()), None),
        Some(Token::String(msg)) => (None, Some(msg.clone())),
        other => {
            return Err(ParseError::unexpected_token(
                other,
                "severity identifier or message string",
                stream.current_span(),
            ))
        }
    };
    stream.advance();

    // If comma, parse second item (must be message)
    if matches!(stream.peek(), Some(Token::Comma)) {
        stream.advance();
        match stream.peek() {
            Some(Token::String(msg)) => {
                message = Some(msg.clone());
                stream.advance();
            }
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "string literal for assertion message",
                    stream.current_span(),
                ))
            }
        }
    }

    Ok((severity, message))
}

/// Check if the next token can start a statement.
/// Check if the next token can start a statement.
///
/// Disambiguates between single expressions and statement blocks to ensure
/// correct parse tree structure. This is critical for purity validation during
/// compilation.
///
/// # Rationale
///
/// The parser must distinguish:
/// - `{ x + y }` → `BlockBody::Expression` (pure, allowed in resolve phase)
/// - `{ signal <- x }` → `BlockBody::Statements` (effect, restricted to collect/fracture)
///
/// Previously, ANY identifier was treated as a statement start, causing
/// single-expression blocks like `{ maths.abs(x) }` to incorrectly parse as
/// statement blocks. This triggered false "effect in pure context" errors
/// because the resolve phase purity checker rejects statement blocks even
/// if the statement is just a pure expression.
///
/// # Special Cases
///
/// - `let x = val` → statement (returns true)
/// - `let x = val in body` → expression (returns false, handled by [`is_let_expression`])
/// - `signal.path <- value` → statement (returns true if followed by `<-`)
/// - `maths.abs(x)` → expression (returns false, no assignment operator)
///
/// # Parameters
///
/// - `stream`: Token stream positioned at the potential statement start
///
/// # Returns
///
/// - `true`: Next tokens form a statement (let-statement or assignment)
/// - `false`: Next tokens form an expression (let-expression or function call)
fn is_statement_start(stream: &TokenStream) -> bool {
    match stream.peek() {
        Some(Token::Let) => {
            // Disambiguate: let-statement vs let-expression
            !is_let_expression(stream)
        }
        Some(Token::Ident(_)) | Some(Token::Signal) | Some(Token::Field) => {
            // Only treat as statement if it's an assignment
            // Single expressions like `maths.abs(x)` should be parsed as BlockBody::Expression
            is_assignment(stream)
        }
        _ => false,
    }
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

/// Check if the current position is the start of an assignment statement.
///
/// Uses lookahead to detect the assignment pattern:
/// `<path> <- <expr>` where `<path>` is a dot-separated identifier chain.
///
/// # Algorithm
///
/// Scans forward through identifier and dot tokens until hitting:
/// - `<-` (LeftArrow) → assignment detected, return true
/// - Any other token → not an assignment, return false
///
/// # Examples
///
/// ```cdsl
/// signal.value <- 10       // true (assignment)
/// core.temp <- x + y       // true (assignment)
/// maths.abs(x)            // false (function call, no <-)
/// x                       // false (bare identifier)
/// ```
///
/// # Parameters
///
/// - `stream`: Token stream positioned at potential assignment start
///
/// # Returns
///
/// - `true`: Token sequence matches assignment pattern `<path> <- ...`
/// - `false`: Not an assignment (expression or other construct)
fn is_assignment(stream: &TokenStream) -> bool {
    // Scan forward through path tokens (ident, dot, signal, field)
    // looking for the assignment operator '<-'
    let mut pos = 0;
    loop {
        match stream.peek_nth(pos) {
            Some(Token::Ident(_)) | Some(Token::Dot) | Some(Token::Signal) | Some(Token::Field) => {
                pos += 1;
            }
            Some(Token::LeftArrow) => return true, // Found assignment operator
            _ => return false,                     // Not an assignment
        }
    }
}

/// Check if this looks like a let-expression (has 'in' keyword) vs let-statement.
///
/// Uses bounded lookahead to find the pattern: `let <ident> = <expr> in`
///
/// # Returns
/// - `true`: let-expression (continue with expression parsing)
/// - `false`: let-statement (use statement parsing)
///
/// # Algorithm
/// 1. Verify `let <ident> =` prefix
/// 2. Scan forward tracking delimiter depth to skip past value expression
/// 3. Return true if we find `in` at depth 0 before closing brace
/// 4. Use MAX_LOOKAHEAD bound to prevent infinite loops
fn is_let_expression(stream: &TokenStream) -> bool {
    // Pattern: let <ident> = ...anything... in

    if !matches!(stream.peek(), Some(Token::Let)) {
        return false;
    }

    if !matches!(stream.peek_nth(1), Some(Token::Ident(_))) {
        return false; // Invalid let syntax, let statement parser handle error
    }

    if !matches!(stream.peek_nth(2), Some(Token::Eq)) {
        return false; // Invalid let syntax
    }

    // Now we need to skip past the value expression to find 'in'
    // Use depth-based scanning to handle nested parens/brackets/braces
    let mut pos = 3; // Start after '='
    let mut depth = 0;
    const MAX_LOOKAHEAD: usize = 100; // Prevent infinite loops

    while pos < MAX_LOOKAHEAD {
        match stream.peek_nth(pos) {
            // Increase depth on opening delimiters
            Some(Token::LParen) | Some(Token::LBracket) | Some(Token::LBrace) => {
                depth += 1;
            }
            // Decrease depth on closing delimiters
            Some(Token::RParen) | Some(Token::RBracket) | Some(Token::RBrace) => {
                if depth == 0 {
                    // Found closing brace of block body, no 'in' found
                    return false;
                }
                depth -= 1;
            }
            // Found 'in' at top level (depth 0) → let-expression
            Some(Token::In) if depth == 0 => {
                return true;
            }
            // End of tokens → not a let-expression
            None => return false,
            // Continue scanning
            _ => {}
        }
        pos += 1;
    }

    // Exceeded lookahead limit → assume statement (safer default)
    false
}

/// Parse let statement.
fn parse_let_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Let)?;

    let name = super::helpers::expect_ident(stream, "variable name in let")?;

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
