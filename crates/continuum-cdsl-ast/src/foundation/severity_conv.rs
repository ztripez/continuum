//! Assertion severity string-to-enum conversion
//!
//! This module provides the canonical implementation for converting assertion
//! severity strings (as written in CDSL) to the `AssertionSeverity` enum.
//!
//! **One Truth**: This is the single source of truth for severity parsing.
//! All conversion logic must be centralized here.

use crate::foundation::AssertionSeverity;

/// Parse assertion severity string to enum value.
///
/// Accepts severity strings as written in CDSL assertion blocks:
/// - `"fatal"` → `AssertionSeverity::Fatal`
/// - `"error"` → `AssertionSeverity::Error`
/// - `"warn"` or `"warning"` → `AssertionSeverity::Warn`
///
/// Comparison is case-insensitive.
///
/// # Arguments
///
/// * `s` - Severity string from DSL (e.g., "fatal", "Error", "WARN")
///
/// # Returns
///
/// * `Ok(severity)` - Successfully parsed severity level
/// * `Err(message)` - Complete formatted error message for invalid severity
///
/// # Examples
///
/// ```
/// use continuum_cdsl_ast::foundation::{parse_severity, AssertionSeverity};
///
/// assert_eq!(parse_severity("fatal").unwrap(), AssertionSeverity::Fatal);
/// assert_eq!(parse_severity("Error").unwrap(), AssertionSeverity::Error);
/// assert_eq!(parse_severity("WARN").unwrap(), AssertionSeverity::Warn);
/// assert_eq!(parse_severity("warning").unwrap(), AssertionSeverity::Warn);
/// assert!(parse_severity("invalid").unwrap_err().contains("invalid"));
/// assert!(parse_severity("invalid").unwrap_err().contains("fatal, error, warn, warning"));
/// ```
pub fn parse_severity(s: &str) -> Result<AssertionSeverity, String> {
    match s.to_lowercase().as_str() {
        "fatal" => Ok(AssertionSeverity::Fatal),
        "error" => Ok(AssertionSeverity::Error),
        "warn" | "warning" => Ok(AssertionSeverity::Warn),
        unknown => Err(format!(
            "unknown assertion severity '{}', valid values: {}",
            unknown,
            valid_severity_strings().join(", ")
        )),
    }
}

/// Returns all valid severity string values.
///
/// This includes both canonical names and aliases:
/// - `"fatal"` (canonical)
/// - `"error"` (canonical)
/// - `"warn"` (canonical)
/// - `"warning"` (alias for warn)
///
/// Use this for error messages and validation.
///
/// # Examples
///
/// ```ignore
/// use continuum_cdsl_ast::foundation::valid_severity_strings;
///
/// let valid = valid_severity_strings();
/// assert!(valid.contains(&"fatal"));
/// assert!(valid.contains(&"warning"));
/// ```
pub fn valid_severity_strings() -> &'static [&'static str] {
    &["fatal", "error", "warn", "warning"]
}

/// Returns the default severity when none is specified.
///
/// Default is `AssertionSeverity::Error` per the enum's `#[default]` attribute.
///
/// # Examples
///
/// ```ignore
/// use continuum_cdsl_ast::foundation::{default_severity, AssertionSeverity};
///
/// assert_eq!(default_severity(), AssertionSeverity::Error);
/// ```
pub fn default_severity() -> AssertionSeverity {
    AssertionSeverity::Error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_severity_fatal() {
        assert_eq!(parse_severity("fatal").unwrap(), AssertionSeverity::Fatal);
        assert_eq!(parse_severity("FATAL").unwrap(), AssertionSeverity::Fatal);
        assert_eq!(parse_severity("Fatal").unwrap(), AssertionSeverity::Fatal);
    }

    #[test]
    fn test_parse_severity_error() {
        assert_eq!(parse_severity("error").unwrap(), AssertionSeverity::Error);
        assert_eq!(parse_severity("ERROR").unwrap(), AssertionSeverity::Error);
    }

    #[test]
    fn test_parse_severity_warn() {
        assert_eq!(parse_severity("warn").unwrap(), AssertionSeverity::Warn);
        assert_eq!(parse_severity("WARN").unwrap(), AssertionSeverity::Warn);
    }

    #[test]
    fn test_parse_severity_warning_alias() {
        assert_eq!(parse_severity("warning").unwrap(), AssertionSeverity::Warn);
        assert_eq!(parse_severity("WARNING").unwrap(), AssertionSeverity::Warn);
    }

    #[test]
    fn test_parse_severity_invalid() {
        let result = parse_severity("invalid");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("unknown assertion severity"));
        assert!(err.contains("invalid"));
        assert!(err.contains("fatal"));
        assert!(err.contains("error"));
        assert!(err.contains("warn"));
    }

    #[test]
    fn test_valid_severity_strings() {
        let valid = valid_severity_strings();
        assert_eq!(valid.len(), 4);
        assert!(valid.contains(&"fatal"));
        assert!(valid.contains(&"error"));
        assert!(valid.contains(&"warn"));
        assert!(valid.contains(&"warning"));
    }

    #[test]
    fn test_default_severity() {
        assert_eq!(default_severity(), AssertionSeverity::Error);
    }

    #[test]
    fn test_default_matches_enum_default() {
        assert_eq!(default_severity(), AssertionSeverity::default());
    }
}
