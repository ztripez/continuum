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
        Token::Strata => Some("strata".to_string()),
        Token::Type => Some("type".to_string()),
        Token::Initial => Some("initial".to_string()),
        Token::Terminal => Some("terminal".to_string()),
        _ => None,
    }
}

/// Check if a token is a keyword that can be used as an identifier.
///
/// This is derived from `keyword_to_string()` and maintains One Truth.
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
    keyword_to_string(token).is_some()
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
        Token::Initial => Some("initial"),
        Token::Assert => Some("assert"),
        Token::Measure => Some("measure"),
        Token::WarmUp => Some("warmup"),
        _ => None,
    }
}

/// Check if a token is an execution block keyword.
///
/// This is derived from `execution_block_name()` and maintains One Truth.
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
    execution_block_name(token).is_some()
}

/// Check if a token/identifier is a type keyword.
///
/// Type keywords are: Scalar, Vec2, Vec3, Vec4, Quat, Mat2, Mat3, Mat4, Tensor, Bool
///
/// These can appear after `:` in attribute position to declare a node's type.
///
/// # Examples
/// ```
/// use continuum_cdsl_lexer::Token;
/// use continuum_cdsl_parser::parser::token_utils::is_type_keyword;
///
/// assert!(is_type_keyword("Scalar"));
/// assert!(is_type_keyword("Vec3"));
/// assert!(!is_type_keyword("signal"));
/// assert!(!is_type_keyword("resolve"));
/// ```
pub fn is_type_keyword(name: &str) -> bool {
    matches!(
        name,
        "Bool" | "Scalar" | "Vec2" | "Vec3" | "Vec4" | "Quat" | "Mat2" | "Mat3" | "Mat4" | "Tensor"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_to_string_all_keywords() {
        assert_eq!(
            keyword_to_string(&Token::Config),
            Some("config".to_string())
        );
        assert_eq!(keyword_to_string(&Token::Const), Some("const".to_string()));
        assert_eq!(
            keyword_to_string(&Token::Signal),
            Some("signal".to_string())
        );
        assert_eq!(keyword_to_string(&Token::Field), Some("field".to_string()));
        assert_eq!(
            keyword_to_string(&Token::Entity),
            Some("entity".to_string())
        );
        assert_eq!(
            keyword_to_string(&Token::Strata),
            Some("strata".to_string())
        );
        assert_eq!(keyword_to_string(&Token::Type), Some("type".to_string()));
    }

    #[test]
    fn test_keyword_to_string_non_keywords() {
        assert_eq!(keyword_to_string(&Token::Plus), None);
        assert_eq!(keyword_to_string(&Token::Minus), None);
        assert_eq!(keyword_to_string(&Token::LParen), None);
        assert_eq!(keyword_to_string(&Token::Resolve), None); // Phase keyword, not identifier keyword
    }

    #[test]
    fn test_execution_block_name_all_phases() {
        assert_eq!(execution_block_name(&Token::Resolve), Some("resolve"));
        assert_eq!(execution_block_name(&Token::Collect), Some("collect"));
        assert_eq!(execution_block_name(&Token::Emit), Some("emit"));
        assert_eq!(execution_block_name(&Token::Assert), Some("assert"));
        assert_eq!(execution_block_name(&Token::Measure), Some("measure"));
    }

    #[test]
    fn test_execution_block_name_non_phases() {
        assert_eq!(execution_block_name(&Token::Signal), None);
        assert_eq!(execution_block_name(&Token::Plus), None);
        assert_eq!(execution_block_name(&Token::Config), None); // Identifier keyword, not phase
        assert_eq!(execution_block_name(&Token::LBrace), None);
    }

    #[test]
    fn test_is_keyword_identifier_matches_keyword_to_string() {
        // Ensure is_keyword_identifier and keyword_to_string agree
        let keyword_tokens = [
            Token::Config,
            Token::Const,
            Token::Signal,
            Token::Field,
            Token::Entity,
            Token::Strata,
            Token::Type,
        ];

        for token in &keyword_tokens {
            assert!(
                is_keyword_identifier(token),
                "{:?} should be keyword identifier",
                token
            );
            assert!(
                keyword_to_string(token).is_some(),
                "{:?} should convert to string",
                token
            );
        }
    }

    #[test]
    fn test_is_keyword_identifier_rejects_non_keywords() {
        let non_keywords = [Token::Plus, Token::Minus, Token::LParen, Token::Resolve];

        for token in &non_keywords {
            assert!(
                !is_keyword_identifier(token),
                "{:?} should not be keyword identifier",
                token
            );
        }
    }

    #[test]
    fn test_is_execution_block_matches_execution_block_name() {
        // Ensure consistency between predicates and converters
        let phase_tokens = [
            Token::Resolve,
            Token::Collect,
            Token::Emit,
            Token::Assert,
            Token::Measure,
        ];

        for token in &phase_tokens {
            assert!(
                is_execution_block_keyword(token),
                "{:?} should be execution block",
                token
            );
            assert!(
                execution_block_name(token).is_some(),
                "{:?} should have block name",
                token
            );
        }
    }

    #[test]
    fn test_is_execution_block_rejects_non_phases() {
        let non_phases = [Token::Signal, Token::Config, Token::Plus, Token::LBrace];

        for token in &non_phases {
            assert!(
                !is_execution_block_keyword(token),
                "{:?} should not be execution block",
                token
            );
        }
    }

    #[test]
    fn test_keyword_and_phase_sets_are_disjoint() {
        // Ensure a token is never both a keyword identifier AND an execution block
        let all_tokens = [
            Token::Config,
            Token::Const,
            Token::Signal,
            Token::Field,
            Token::Entity,
            Token::Strata,
            Token::Type,
            Token::Resolve,
            Token::Collect,
            Token::Emit,
            Token::Assert,
            Token::Measure,
        ];

        for token in &all_tokens {
            let is_keyword = is_keyword_identifier(token);
            let is_phase = is_execution_block_keyword(token);
            assert!(
                !(is_keyword && is_phase),
                "{:?} should not be both keyword identifier and execution block",
                token
            );
        }
    }

    #[test]
    fn test_is_type_keyword_recognizes_all_types() {
        let type_keywords = [
            "Bool", "Scalar", "Vec2", "Vec3", "Vec4", "Quat", "Mat2", "Mat3", "Mat4", "Tensor",
        ];

        for keyword in &type_keywords {
            assert!(
                is_type_keyword(keyword),
                "{} should be recognized as type keyword",
                keyword
            );
        }
    }

    #[test]
    fn test_is_type_keyword_rejects_non_types() {
        let non_types = [
            "signal", "resolve", "Vector", // Vector is not a keyword (it's Vec2/Vec3/etc.)
            "scalar", // lowercase
            "SCALAR", // uppercase
            "collect", "field", "operator", "config", "const", "strata",
        ];

        for name in &non_types {
            assert!(
                !is_type_keyword(name),
                "{} should NOT be recognized as type keyword",
                name
            );
        }
    }

    #[test]
    fn test_is_type_keyword_case_sensitive() {
        // Correct case
        assert!(is_type_keyword("Scalar"));
        assert!(is_type_keyword("Bool"));
        assert!(is_type_keyword("Vec3"));

        // Wrong case
        assert!(!is_type_keyword("scalar"));
        assert!(!is_type_keyword("bool"));
        assert!(!is_type_keyword("vec3"));
        assert!(!is_type_keyword("SCALAR"));
        assert!(!is_type_keyword("VEC3"));
    }
}
