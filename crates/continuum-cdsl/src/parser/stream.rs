//! Token stream wrapper for hand-written parser.

use crate::foundation::Span;
use crate::lexer::Token;

/// Token stream with lookahead and position tracking.
///
/// Provides methods for consuming tokens, lookahead, and span tracking
/// for the hand-written recursive descent parser.
pub struct TokenStream<'src> {
    tokens: &'src [Token],
    pos: usize,
    file_id: u16,
}

impl<'src> TokenStream<'src> {
    /// Create a new token stream.
    pub fn new(tokens: &'src [Token], file_id: u16) -> Self {
        Self {
            tokens,
            pos: 0,
            file_id,
        }
    }

    /// Peek at the current token without consuming it.
    pub fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// Peek at the nth token ahead without consuming.
    pub fn peek_nth(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n)
    }

    /// Advance to the next token and return the current one.
    pub fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
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
    pub fn span_from(&self, start: usize) -> Span {
        // For now, create a simple span
        // In a full implementation, we'd track byte offsets from tokens
        Span::new(self.file_id, start as u32, self.pos as u32, 0)
    }

    /// Get a span for the current token.
    pub fn current_span(&self) -> Span {
        Span::new(self.file_id, self.pos as u32, self.pos as u32, 0)
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
