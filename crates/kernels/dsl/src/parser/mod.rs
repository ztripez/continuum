//! Parser for Continuum DSL.
//!
//! This module implements a recursive descent parser using the Chumsky
//! parser combinator library. The parser converts DSL source text directly
//! into an Abstract Syntax Tree ([`crate::ast::CompilationUnit`]).
//!
//! # Architecture
//!
//! The parser is organized into three submodules:
//!
//! - [`primitives`] - Low-level parsers for tokens (identifiers, paths, literals, units)
//! - [`items`] - Top-level declaration parsers (signals, fields, operators, etc.)
//! - [`expr`] - Expression parsers with operator precedence handling
//!
//! # Error Recovery
//!
//! Chumsky provides automatic error recovery, allowing the parser to continue
//! after encountering syntax errors. This enables reporting multiple errors
//! in a single parse pass and provides better IDE integration.
//!
//! # Example
//!
//! ```ignore
//! use continuum_dsl::parser::parse;
//!
//! let source = r#"
//!     const.physics {
//!         gravity: 9.81
//!     }
//!
//!     signal.position {
//!         : Scalar<m>
//!         resolve { prev + 1.0 }
//!     }
//! "#;
//!
//! let (ast, errors) = parse(source);
//! if errors.is_empty() {
//!     let unit = ast.unwrap();
//!     println!("Parsed {} items", unit.items.len());
//! }
//! ```
//!
//! # Span Tracking
//!
//! All AST nodes are wrapped in [`crate::ast::Spanned`] to preserve source
//! locations for error reporting and IDE features.

mod expr;
mod items;
mod primitives;

use chumsky::prelude::*;

use crate::ast::{CompilationUnit, Spanned};
use primitives::ws;

/// A rich parse error with span and context information.
///
/// Chumsky's [`Rich`] error type provides detailed information including:
/// - The span where the error occurred
/// - Expected tokens at that location
/// - The actual token found
/// - Context labels for nested parse states
pub type ParseError<'src> = Rich<'src, char>;

/// Parses DSL source code into a compilation unit.
///
/// This is the main entry point for parsing Continuum DSL source code.
/// The parser supports error recovery, so it may return both a partial
/// AST and a list of errors.
///
/// # Arguments
///
/// * `source` - The DSL source code as a string slice
///
/// # Returns
///
/// A tuple containing:
/// - `Option<CompilationUnit>` - The parsed AST, or `None` if parsing failed completely
/// - `Vec<ParseError>` - All parse errors encountered (may be empty for valid input)
///
/// # Example
///
/// ```ignore
/// let (ast, errors) = parse("signal.temp { : Scalar<K> resolve { 0.0 } }");
/// assert!(errors.is_empty());
/// assert_eq!(ast.unwrap().items.len(), 1);
/// ```
pub fn parse(source: &str) -> (Option<CompilationUnit>, Vec<ParseError<'_>>) {
    compilation_unit().parse(source).into_output_errors()
}

fn compilation_unit<'src>(
) -> impl Parser<'src, &'src str, CompilationUnit, extra::Err<ParseError<'src>>> {
    ws().ignore_then(
        items::item()
            .map_with(|i, e| Spanned::new(i, e.span().into()))
            .padded_by(ws())
            .repeated()
            .collect()
            .map(|items| CompilationUnit { items }),
    )
}

#[cfg(test)]
mod tests;
