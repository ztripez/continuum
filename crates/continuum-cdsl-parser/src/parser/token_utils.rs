//! Token utility functions for keyword-to-string conversion.
//!
//! This module provides the canonical mappings from keyword tokens to their
//! string representations. This is the single source of truth for keyword
//! conversions, used throughout the parser when keywords can appear as
//! identifiers (in paths, field names, etc.) or when dispatching on keyword
//! types without hardcoded match statements.

use continuum_cdsl_lexer::Token;

/// Convert a keyword token to its string representation.
///
/// Many CDSL contexts allow keywords to be used as identifiers (e.g., paths
/// like `config.field`, attribute names like `:signal`). This function provides
/// the canonical mapping from keyword tokens to their string form.
///
/// # Parameters
/// - `token`: The token to convert
///
/// # Returns
/// - `Some(String)` if the token is a keyword that can be used as an identifier
/// - `None` if the token is not a convertible keyword
///
/// # Examples
/// ```
/// use continuum_cdsl_lexer::Token;
/// use continuum_cdsl_parser::parser::token_utils::keyword_to_string;
///
/// assert_eq!(keyword_to_string(&Token::Config), Some("config".to_string()));
/// assert_eq!(keyword_to_string(&Token::Signal), Some("signal".to_string()));
/// assert_eq!(keyword_to_string(&Token::Plus), None);
/// ```
pub fn keyword_to_string(token: &Token) -> Option<String> {
    match token {
        Token::Config => Some("config".to_string()),
        Token::Const => Some("const".to_string()),
        Token::Signal => Some("signal".to_string()),
        Token::Field => Some("field".to_string()),
        Token::Entity => Some("entity".to_string()),
        Token::Dt => Some("dt".to_string()),
        Token::Strata => Some("strata".to_string()),
        Token::Type => Some("type".to_string()),
        _ => None,
    }
}

/// Check if a token is a keyword that can be used as an identifier.
///
/// # Examples
/// ```
/// use continuum_cdsl_lexer::Token;
/// use continuum_cdsl_parser::parser::token_utils::is_keyword_identifier;
///
/// assert!(is_keyword_identifier(&Token::Config));
/// assert!(is_keyword_identifier(&Token::Signal));
/// assert!(!is_keyword_identifier(&Token::Plus));
/// assert!(!is_keyword_identifier(&Token::LParen));
/// ```
pub fn is_keyword_identifier(token: &Token) -> bool {
    matches!(
        token,
        Token::Config
            | Token::Const
            | Token::Signal
            | Token::Field
            | Token::Entity
            | Token::Dt
            | Token::Strata
            | Token::Type
    )
}

/// Get the execution block name for a phase keyword token.
///
/// Execution blocks (resolve, collect, emit, assert, measure) are used in
/// operator/signal/field declarations. This function provides the canonical
/// mapping from phase keyword tokens to their lowercase block names.
///
/// # Parameters
/// - `token`: The token to convert
///
/// # Returns
/// - `Some(&str)` if the token is an execution block keyword
/// - `None` if the token is not an execution block keyword
///
/// # Examples
/// ```
/// use continuum_cdsl_lexer::Token;
/// use continuum_cdsl_parser::parser::token_utils::execution_block_name;
///
/// assert_eq!(execution_block_name(&Token::Resolve), Some("resolve"));
/// assert_eq!(execution_block_name(&Token::Collect), Some("collect"));
/// assert_eq!(execution_block_name(&Token::Emit), Some("emit"));
/// assert_eq!(execution_block_name(&Token::Plus), None);
/// ```
pub fn execution_block_name(token: &Token) -> Option<&'static str> {
    match token {
        Token::Resolve => Some("resolve"),
        Token::Collect => Some("collect"),
        Token::Emit => Some("emit"),
        Token::Assert => Some("assert"),
        Token::Measure => Some("measure"),
        _ => None,
    }
}

/// Check if a token is an execution block keyword.
///
/// # Examples
/// ```
/// use continuum_cdsl_lexer::Token;
/// use continuum_cdsl_parser::parser::token_utils::is_execution_block_keyword;
///
/// assert!(is_execution_block_keyword(&Token::Resolve));
/// assert!(is_execution_block_keyword(&Token::Measure));
/// assert!(!is_execution_block_keyword(&Token::Signal));
/// assert!(!is_execution_block_keyword(&Token::Plus));
/// ```
pub fn is_execution_block_keyword(token: &Token) -> bool {
    matches!(
        token,
        Token::Resolve | Token::Collect | Token::Emit | Token::Assert | Token::Measure
    )
}
