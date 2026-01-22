//! Parse error types and error recovery.

use continuum_cdsl_ast::foundation::Span;
use continuum_cdsl_lexer::Token;
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
///
/// Each variant represents a specific class of parsing failure to enable
/// targeted error recovery and clear diagnostic messages.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// Unexpected token encountered where a specific token was expected.
    ///
    /// Use when the parser expected a particular token (e.g., `{`, `:`, `=`)
    /// but found a different one. The parser typically skips to a recovery
    /// point (e.g., next statement or declaration boundary).
    ///
    /// Example: Expected `{` to start block, found identifier instead.
    UnexpectedToken,

    /// Unexpected end of input while parsing was incomplete.
    ///
    /// Use when the parser reaches EOF but expected more tokens to complete
    /// the current construct (e.g., unclosed block, incomplete expression).
    /// This indicates the source file was truncated or malformed.
    ///
    /// Example: Reached EOF while parsing function body, missing `}`.
    UnexpectedEof,

    /// Syntactically invalid construct that violates language grammar.
    ///
    /// Use when tokens are present but violate syntax rules (e.g., invalid
    /// operator combination, malformed declaration). Unlike `UnexpectedToken`,
    /// this indicates a structural grammar violation, not just wrong token.
    ///
    /// Example: `signal x: :` (double colon in type position).
    InvalidSyntax,

    /// Other parse error not covered by specific categories.
    ///
    /// Use for errors that don't fit the above patterns or for temporary
    /// error reporting during parser development. Prefer specific variants
    /// when possible for better diagnostics.
    Other,
}

impl ParseError {
    /// Create an "expected token" error.
    pub fn expected_token(expected: Token, found: Option<Token>, span: Span) -> Self {
        let message = match &found {
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
