//! Parse error types and error recovery.

use crate::foundation::Span;
use crate::lexer::Token;
use std::fmt;

/// Parse error with source location and context.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// Kind of parse error
    pub kind: ParseErrorKind,
    /// Source location where error occurred
    pub span: Span,
    /// Human-readable error message
    pub message: String,
}

/// Category of parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// Unexpected token (found X, expected Y)
    UnexpectedToken,
    /// Unexpected end of input
    UnexpectedEof,
    /// Invalid syntax
    InvalidSyntax,
    /// Other parse error
    Other,
}

impl ParseError {
    /// Create an "expected token" error.
    pub fn expected_token(expected: Token, found: Option<Token>, span: Span) -> Self {
        let message = match found {
            Some(token) => format!("expected {:?}, found {:?}", expected, token),
            None => format!("expected {:?}, found end of input", expected),
        };
        Self {
            kind: if found.is_none() {
                ParseErrorKind::UnexpectedEof
            } else {
                ParseErrorKind::UnexpectedToken
            },
            span,
            message,
        }
    }

    /// Create an "unexpected token" error.
    pub fn unexpected_token(found: Option<&Token>, context: &str, span: Span) -> Self {
        let message = match found {
            Some(token) => format!("unexpected {:?} {}", token, context),
            None => format!("unexpected end of input {}", context),
        };
        Self {
            kind: if found.is_none() {
                ParseErrorKind::UnexpectedEof
            } else {
                ParseErrorKind::UnexpectedToken
            },
            span,
            message,
        }
    }

    /// Create an "invalid syntax" error.
    pub fn invalid_syntax(message: impl Into<String>, span: Span) -> Self {
        Self {
            kind: ParseErrorKind::InvalidSyntax,
            span,
            message: message.into(),
        }
    }

    /// Create a generic parse error.
    pub fn other(message: impl Into<String>, span: Span) -> Self {
        Self {
            kind: ParseErrorKind::Other,
            span,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {:?}", self.message, self.span)
    }
}

impl std::error::Error for ParseError {}
