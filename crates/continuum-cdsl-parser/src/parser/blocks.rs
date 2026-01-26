//! Block body, warmup, when, and observe parsers.

use super::{ParseError, TokenStream};
use continuum_cdsl_ast::{BlockBody, ObserveBlock, ObserveWhen, Stmt, WarmupBlock, WhenBlock};
use continuum_cdsl_lexer::Token;
use std::rc::Rc;

/// Parse execution blocks inside a primitive declaration body.
///
/// Parses zero or more phase-tagged execution blocks within a declaration
/// (e.g., operator, signal, field). Each block is tagged with a phase name
/// that determines when it executes in the simulation lifecycle.
///
/// **Phase Validation Architecture**: This function performs **syntax parsing only**.
/// It recognizes phase keywords and converts them to strings, but does NOT validate:
/// - Whether a phase name is semantically valid (e.g., "emit" is legacy)
/// - Whether a phase is allowed for the node's role (e.g., signals can't have "collect")
/// - Whether the phase makes sense in context
///
/// All semantic validation happens in the **resolver** via:
/// - `continuum_cdsl_resolve::resolve::blocks::parse_phase_name()` - validates phase names
/// - `continuum_cdsl_resolve::resolve::blocks::validate_phase_for_role()` - validates role compatibility
///
/// This separation ensures the parser focuses on syntax while the resolver handles
/// all semantic rules, maintaining clean layer boundaries.
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
    let conditions = super::helpers::parse_semicolon_separated_exprs(stream)?;

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
    let (severity, mut message) = match stream.peek() {
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

    Ok((
        severity.map(|s| s.to_string()),
        message.map(|s| s.to_string()),
    ))
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
/// - `true`: Next tokens form a statement (let-statement, assignment, or if-statement)
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
        Some(Token::If) => {
            // If statements in statement blocks (apply/collect) have statement bodies
            // Look for patterns that indicate statement bodies inside the if
            is_if_with_statements(stream)
        }
        Some(Token::Emit) => true, // emit is always a statement
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
        Some(Token::Emit) => parse_emit_event_statement(stream),
        Some(Token::If) => {
            // If statement with statement bodies
            parse_if_statement(stream)
        }
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
/// Uses bounded lookahead to find the pattern: `let <ident> = <expr> in <body>`
///
/// # Returns
/// - `true`: let-expression (continue with expression parsing)
/// - `false`: let-statement (use statement parsing)
///
/// # Algorithm
/// 1. Verify `let <ident> =` prefix
/// 2. Scan forward tracking delimiter depth to skip past value expression
/// 3. If we find `in` at depth 0, check what follows
/// 4. If body looks like a statement (path followed by `<-`), return false
/// 5. Use MAX_LOOKAHEAD bound to prevent infinite loops
fn is_let_expression(stream: &TokenStream) -> bool {
    // Pattern: let <ident> = ...anything... in <expr>

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
            // Found 'in' at top level (depth 0) → check what follows
            Some(Token::In) if depth == 0 => {
                // Look at what comes after 'in' to see if it looks like a statement
                // If the body starts with ident.ident...ident <- then it's not a let-expression
                // (it's a let followed by a statement, which should be parsed as let-statement)
                return !is_assignment_after_in(stream, pos + 1);
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

/// Check if an `if` at the current position contains statement-style bodies.
///
/// Scans past the condition to find the `{`, then looks at the first token inside
/// the body to determine if it's a statement (let without in, assignment, or emit).
fn is_if_with_statements(stream: &TokenStream) -> bool {
    // Pattern: if <condition> { <body> }
    // We need to find the { after the condition, then check what's inside

    if !matches!(stream.peek(), Some(Token::If)) {
        return false;
    }

    // Skip past 'if' and find the opening brace of the body
    let mut pos = 1; // Start after 'if'
    let mut depth = 0;
    const MAX_LOOKAHEAD: usize = 100;

    // Scan past condition to find `{`
    while pos < MAX_LOOKAHEAD {
        match stream.peek_nth(pos) {
            Some(Token::LParen) | Some(Token::LBracket) => {
                depth += 1;
            }
            Some(Token::RParen) | Some(Token::RBracket) => {
                if depth > 0 {
                    depth -= 1;
                }
            }
            Some(Token::LBrace) if depth == 0 => {
                // Found the body's opening brace
                // Check what's inside (pos + 1 is first token of body)
                let body_start = pos + 1;
                return is_statement_token_at(stream, body_start);
            }
            None => return false,
            _ => {}
        }
        pos += 1;
    }

    false
}

/// Check if the token at a given position looks like the start of a statement.
fn is_statement_token_at(stream: &TokenStream, pos: usize) -> bool {
    match stream.peek_nth(pos) {
        Some(Token::Let) => {
            // Check if it's let-statement (no 'in' follows the value)
            // Simplified: if we see `let ident =` followed eventually by `<-` before `in`, it's a statement
            // For now, assume let in a body is a statement unless we see strong evidence otherwise
            true
        }
        Some(Token::Emit) => true,
        Some(Token::Ident(_)) | Some(Token::Signal) | Some(Token::Field) => {
            // Check if followed by path and `<-`
            is_assignment_at(stream, pos)
        }
        _ => false,
    }
}

/// Check if position starts an assignment: `path <- ...`
fn is_assignment_at(stream: &TokenStream, start_pos: usize) -> bool {
    let mut pos = start_pos;
    const MAX_LOOKAHEAD: usize = 20;

    while pos < start_pos + MAX_LOOKAHEAD {
        match stream.peek_nth(pos) {
            Some(Token::Ident(_)) | Some(Token::Dot) | Some(Token::Signal) | Some(Token::Field) => {
                pos += 1;
            }
            Some(Token::LeftArrow) => return true,
            _ => return false,
        }
    }
    false
}

/// Check if tokens after 'in' look like an assignment statement.
///
/// Returns true if we see a path followed by `<-`.
fn is_assignment_after_in(stream: &TokenStream, start_pos: usize) -> bool {
    let mut pos = start_pos;
    const MAX_LOOKAHEAD: usize = 20;

    // Scan through identifier.identifier... pattern looking for <-
    while pos < start_pos + MAX_LOOKAHEAD {
        match stream.peek_nth(pos) {
            Some(Token::Ident(_)) | Some(Token::Signal) | Some(Token::Field) => {
                pos += 1;
            }
            Some(Token::Dot) => {
                pos += 1;
            }
            Some(Token::LeftArrow) => {
                // Found assignment after path
                return true;
            }
            _ => {
                // Not a simple path, so not an assignment pattern
                return false;
            }
        }
    }

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

/// Parse if statement with statement bodies.
///
/// Syntax: `if condition { statements } [else { statements }]`
///
/// Unlike if-expressions, if-statements have statement bodies and may omit the else branch.
fn parse_if_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::If)?;

    let condition = super::expr::parse_expr(stream)?;

    stream.expect(Token::LBrace)?;
    let then_stmts = parse_statements(stream)?;
    stream.expect(Token::RBrace)?;

    let else_stmts = if matches!(stream.peek(), Some(Token::Else)) {
        stream.advance(); // consume 'else'

        if matches!(stream.peek(), Some(Token::If)) {
            // else if - parse as single if statement
            vec![parse_if_statement(stream)?]
        } else {
            stream.expect(Token::LBrace)?;
            let stmts = parse_statements(stream)?;
            stream.expect(Token::RBrace)?;
            stmts
        }
    } else {
        Vec::new()
    };

    Ok(Stmt::If {
        condition,
        then_branch: then_stmts,
        else_branch: else_stmts,
        span: stream.span_from(start),
    })
}

/// Parse assignment statement.
///
/// Handles three patterns:
/// 1. Signal assignment: `signal.path <- value`
/// 2. Member signal assignment: `entity.X[i].member <- value`
/// 3. Field assignment: `field.path <- position, value`
fn parse_assignment_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();

    // Parse the target path
    let target = super::types::parse_path(stream)?;

    // Check for index (member signal pattern)
    if matches!(stream.peek(), Some(Token::LBracket)) {
        // Member signal assignment: entity.X[i].member <- value
        stream.advance(); // consume '['
        let instance = super::expr::parse_expr(stream)?;
        stream.expect(Token::RBracket)?;

        // Expect dot and member name
        stream.expect(Token::Dot)?;
        let member = super::helpers::expect_ident(stream, "member name after index")?;

        // Expect arrow
        stream.expect(Token::LeftArrow)?;

        // Parse value
        let value = super::expr::parse_expr(stream)?;

        return Ok(Stmt::MemberSignalAssign {
            entity: target,
            instance,
            member,
            value,
            span: stream.span_from(start),
        });
    }

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

/// Parse emit event statement: `emit event.<path> { field: expr, ... }`
///
/// # Syntax
///
/// ```cdsl
/// emit event.rapid_cooling {
///     gradient: temp_gradient,
///     core_temp: core.temp,
///     severity: "warning"
/// }
/// ```
fn parse_emit_event_statement(stream: &mut TokenStream) -> Result<Stmt, ParseError> {
    let start = stream.current_pos();

    stream.expect(Token::Emit)?;

    // Parse event path (e.g., "event.rapid_cooling")
    let path = super::types::parse_path(stream)?;

    // Parse field list in braces
    stream.expect(Token::LBrace)?;

    let mut fields = Vec::new();

    while !matches!(stream.peek(), Some(Token::RBrace)) {
        // Parse field name (identifier)
        let field_name = match stream.peek() {
            Some(Token::Ident(name)) => {
                let name = name.to_string();
                stream.advance();
                name
            }
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "field name in event emission",
                    stream.current_span(),
                ))
            }
        };

        stream.expect(Token::Colon)?;

        // Parse field value expression
        let value = super::expr::parse_expr(stream)?;

        fields.push((field_name, value));

        // Consume optional comma or semicolon
        if matches!(stream.peek(), Some(Token::Comma) | Some(Token::Semicolon)) {
            stream.advance();
        }
    }

    stream.expect(Token::RBrace)?;

    Ok(Stmt::EmitEvent {
        path,
        fields,
        span: stream.span_from(start),
    })
}
