//! Placeholder error types for AST
//!
//! The real CompileError is in continuum-cdsl-resolve, but we need
//! a type here for ValidationError alias.

use crate::foundation::Span;

/// Placeholder compile error
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompileError {
    pub kind: ErrorKind,
    pub span: Span,
    pub message: String,
}

impl CompileError {
    pub fn new(kind: ErrorKind, span: Span, message: String) -> Self {
        Self {
            kind,
            span,
            message,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ErrorKind {
    TypeMismatch = 4,
    // Add others as needed
}
