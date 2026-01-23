//! Token utility functions for keyword-to-string conversion.
//!
//! This module provides the canonical mappings from keyword tokens to their
//! string representations. This is the single source of truth for keyword
//! conversions, used throughout the parser when keywords can appear as
//! identifiers (in paths, field names, etc.) or when dispatching on keyword
//! types without hardcoded match statements.

use continuum_cdsl_lexer::Token;
use std::rc::Rc;

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
pub fn keyword_to_string(token: &Token) -> Option<Rc<str>> {
    match token {
        // Declaration keywords that can appear in paths
        Token::Config => Some(Rc::from("config")),
        Token::Const => Some(Rc::from("const")),
        Token::Signal => Some(Rc::from("signal")),
        Token::Field => Some(Rc::from("field")),
        Token::Entity => Some(Rc::from("entity")),
        Token::Strata => Some(Rc::from("strata")),
        Token::Type => Some(Rc::from("type")),
        Token::Initial => Some(Rc::from("initial")),
        Token::Terminal => Some(Rc::from("terminal")),
        // Primitive keywords that can appear as path segments
        Token::Operator => Some(Rc::from("operator")),
        Token::Impulse => Some(Rc::from("impulse")),
        Token::Fracture => Some(Rc::from("fracture")),
        Token::Chronicle => Some(Rc::from("chronicle")),
        Token::Analyzer => Some(Rc::from("analyzer")),
        // Context keywords used in expressions/paths
        Token::Other => Some(Rc::from("other")),
        Token::Self_ => Some(Rc::from("self")),
        Token::Pairs => Some(Rc::from("pairs")),
        _ => None,
    }
}

/// Get the execution block name for a phase keyword token.
///
/// Execution blocks (resolve, collect, emit, assert, measure) are used in
/// operator/signal/field declarations. This function provides the canonical
/// mapping from phase keyword tokens to their lowercase block names.
///
/// **IMPORTANT**: This function performs **syntactic recognition only**.
/// It does NOT validate whether a phase name is semantically valid or allowed
/// for a particular role. All semantic validation (legacy name rejection,
/// role compatibility checks) happens in the resolver via `parse_phase_name()`
/// and `validate_phase_for_role()`.
///
/// For example, this function returns `Some("emit")` for `Token::Emit`,
/// but the resolver will reject "emit" as a legacy phase name. The parser
/// just converts tokens to strings; the resolver validates them.
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
/// assert_eq!(execution_block_name(&Token::Emit), Some("emit")); // Note: resolver rejects this as legacy
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
        Token::Apply => Some("apply"),
        _ => None,
    }
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
        assert_eq!(keyword_to_string(&Token::Config), Some(Rc::from("config")));
        assert_eq!(keyword_to_string(&Token::Const), Some(Rc::from("const")));
        assert_eq!(keyword_to_string(&Token::Signal), Some(Rc::from("signal")));
        assert_eq!(keyword_to_string(&Token::Field), Some(Rc::from("field")));
        assert_eq!(keyword_to_string(&Token::Entity), Some(Rc::from("entity")));
        assert_eq!(keyword_to_string(&Token::Strata), Some(Rc::from("strata")));
        assert_eq!(keyword_to_string(&Token::Type), Some(Rc::from("type")));
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
