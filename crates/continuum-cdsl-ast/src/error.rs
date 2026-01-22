//! Placeholder error module for AST crate.
//!
//! The AST crate doesn't define its own errors - compile errors
//! are defined in continuum-cdsl-resolve. This module exists
//! only to satisfy module structure.

/// Placeholder error type.
/// Actual CompileError is in continuum-cdsl-resolve.
#[derive(Debug, Clone)]
pub struct PlaceholderError;

impl std::fmt::Display for PlaceholderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PlaceholderError - should not be constructed")
    }
}

impl std::error::Error for PlaceholderError {}
