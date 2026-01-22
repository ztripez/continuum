//! Token stream wrapper for hand-written parser.

use continuum_cdsl_ast::foundation::Span;
use continuum_cdsl_lexer::Token;
use std::ops::Range;

/// Token stream with lookahead and position tracking.
///
/// Provides methods for consuming tokens, lookahead, and span tracking
/// for the hand-written recursive descent parser.
///
/// Each token is paired with its byte span from the source, enabling
/// accurate error message locations.
pub struct TokenStream<'src> {
    tokens: &'src [(Token, Range<usize>)],
    pos: usize,
    file_id: u16,
}

impl<'src> TokenStream<'src> {
    /// Create a new token stream from tokens with their byte spans.
    pub fn new(tokens: &'src [(Token, Range<usize>)], file_id: u16) -> Self {
        Self {
            tokens,
            pos: 0,
            file_id,
        }
    }

    /// Peek at the current token without consuming it.
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|(tok, _)| tok)
    }

    /// Peek at the nth token ahead without consuming.
    pub fn peek_nth(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n).map(|(tok, _)| tok)
    }

    /// Advance to the next token and return the current one.
    pub fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos).map(|(tok, _)| tok);
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Check if the current token matches the expected token.
    pub fn check(&self, expected: &Token) -> bool {
        matches!(self.peek(), Some(t) if std::mem::discriminant(t) == std::mem::discriminant(expected))
    }

    /// Expect a specific token and advance if it matches.
    ///
    /// Returns an error if the token doesn't match.
    pub fn expect(&mut self, expected: Token) -> Result<Span, super::ParseError> {
        if self.check(&expected) {
            let start = self.pos;
            self.advance();
            Ok(self.span_from(start))
        } else {
            Err(super::ParseError::expected_token(
                expected,
                self.peek().cloned(),
                self.current_span(),
            ))
        }
    }

    /// Check if we've reached the end of the token stream.
    pub fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Get the current position in the token stream.
    pub fn current_pos(&self) -> usize {
        self.pos
    }

    /// Create a span from a starting position to the current position.
    ///
    /// Uses actual byte offsets from the source file for accurate error locations.
    ///
    /// # Panics
    ///
    /// Panics if `start` position is out of bounds for the token stream.
    pub fn span_from(&self, start: usize) -> Span {
        assert!(
            start < self.tokens.len(),
            "span_from: start position {} out of bounds (stream length: {})",
            start,
            self.tokens.len()
        );

        let start_byte = self
            .tokens
            .get(start)
            .map(|(_, span)| span.start)
            .expect("BUG: start position validated but token not found");

        let end_byte = if self.pos > 0 && self.pos <= self.tokens.len() {
            // Use the end of the previous token (last consumed token)
            self.tokens
                .get(self.pos - 1)
                .map(|(_, span)| span.end)
                .expect("BUG: pos-1 in valid range but token not found")
        } else {
            // At EOF or start, use start position
            start_byte
        };

        Span::new(self.file_id, start_byte as u32, end_byte as u32, 0)
    }

    /// Get a span for the current token.
    ///
    /// # Panics
    ///
    /// Panics if called on an empty token stream. The parser should always
    /// validate non-empty input before creating a TokenStream.
    pub fn current_span(&self) -> Span {
        if let Some((_, span)) = self.tokens.get(self.pos) {
            Span::new(self.file_id, span.start as u32, span.end as u32, 0)
        } else {
            // At EOF - use the end of the last token
            if let Some((_, span)) = self.tokens.last() {
                Span::new(self.file_id, span.end as u32, span.end as u32, 0)
            } else {
                // Empty token stream is a bug - should be validated before parsing
                panic!(
                    "BUG: current_span() called on empty token stream (file_id: {})",
                    self.file_id
                );
            }
        }
    }

    /// Synchronize to the next declaration keyword for error recovery.
    ///
    /// Skips tokens until we find a declaration keyword or EOF.
    pub fn synchronize(&mut self) {
        while !self.at_end() {
            match self.peek() {
                Some(Token::World)
                | Some(Token::Signal)
                | Some(Token::Field)
                | Some(Token::Operator)
                | Some(Token::Impulse)
                | Some(Token::Fracture)
                | Some(Token::Chronicle)
                | Some(Token::Entity)
                | Some(Token::Member)
                | Some(Token::Strata)
                | Some(Token::Era)
                | Some(Token::Type)
                | Some(Token::Const)
                | Some(Token::Config) => break,
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Get the file_id for this token stream.
    pub fn file_id(&self) -> u16 {
        self.file_id
    }
}
