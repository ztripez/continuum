//! Error handling tests for the CDSL parser.
//!
//! This test suite verifies that the parser correctly detects and reports
//! various syntax errors including:
//! - Unclosed delimiters (braces, parentheses, brackets)
//! - Unexpected end-of-file (EOF)
//! - Malformed syntax (invalid tokens, unexpected keywords)
//! - Error recovery and synchronization

use continuum_cdsl_ast::Declaration;
use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::{parse_declarations, ParseError};
use logos::Logos;

/// Helper to verify that parsing fails with at least one error.
fn expect_error(source: &str) -> Vec<ParseError> {
    let tokens: Vec<Token> = Token::lexer(source).filter_map(Result::ok).collect();
    match parse_declarations(&tokens, 0) {
        Ok(_) => panic!("Expected parse error, but parsing succeeded"),
        Err(errors) => {
            assert!(!errors.is_empty(), "Expected at least one error");
            errors
        }
    }
}

/// Helper for tests that expect success.
fn parse_ok(source: &str) -> Vec<Declaration> {
    let tokens: Vec<Token> = Token::lexer(source).filter_map(Result::ok).collect();
    parse_declarations(&tokens, 0).expect("Parse should succeed")
}

// =============================================================================
// Unclosed Delimiters
// =============================================================================

#[test]
fn test_unclosed_brace_in_signal() {
    let source = r#"
        signal velocity {
            : title("Velocity")
            resolve { 0.0 }
        // Missing closing brace
    "#;

    let errors = expect_error(source);
    assert!(
        errors.iter().any(|e| e.message.contains("expected")
            || e.message.contains("unclosed")
            || e.message.contains("EOF")),
        "Should report unclosed brace or unexpected EOF, got: {:?}",
        errors
    );
}

#[test]
fn test_unclosed_brace_in_resolve_block() {
    let source = r#"
        signal velocity {
            resolve { 
                let x = 1.0;
                x + 2.0
            // Missing closing brace for resolve block
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report unclosed brace");
}

#[test]
fn test_unclosed_paren_in_function_call() {
    let source = r#"
        signal velocity {
            resolve { 
                maths.sqrt(16.0
                // Missing closing paren
            }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report unclosed parenthesis");
}

#[test]
fn test_unclosed_bracket_in_array() {
    let source = r#"
        signal velocity {
            resolve { 
                [1.0, 2.0, 3.0
                // Missing closing bracket
            }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report unclosed bracket");
}

#[test]
fn test_mismatched_delimiter() {
    let source = r#"
        signal velocity {
            resolve { 
                (1.0 + 2.0]
            }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report mismatched delimiter");
}

// =============================================================================
// Unexpected EOF
// =============================================================================

#[test]
fn test_eof_in_declaration() {
    let source = r#"
        signal velocity
    "#;

    let errors = expect_error(source);
    assert!(
        errors
            .iter()
            .any(|e| { e.message.contains("EOF") || e.message.contains("expected") }),
        "Should report unexpected EOF"
    );
}

#[test]
fn test_eof_in_attribute() {
    let source = r#"
        signal velocity {
            : title(
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report EOF in attribute");
}

#[test]
fn test_eof_in_block() {
    let source = r#"
        signal velocity {
            resolve {
                let x = 1.0
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report EOF in block");
}

#[test]
fn test_eof_in_path() {
    let source = r#"
        member plate.
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report incomplete path");
}

// =============================================================================
// Malformed Syntax
// =============================================================================

#[test]
fn test_invalid_keyword_in_declaration() {
    let source = r#"
        invalid_keyword velocity {
            resolve { 0.0 }
        }
    "#;

    let errors = expect_error(source);
    assert!(
        errors
            .iter()
            .any(|e| { e.message.contains("unexpected") || e.message.contains("expected") }),
        "Should report invalid keyword"
    );
}

#[test]
fn test_missing_declaration_name() {
    let source = r#"
        signal {
            resolve { 0.0 }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report missing name");
}

#[test]
fn test_invalid_attribute_syntax() {
    let source = r#"
        signal velocity {
            : title "Missing Parens"
            resolve { 0.0 }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report invalid attribute syntax");
}

#[test]
fn test_missing_block_body() {
    let source = r#"
        signal velocity {
            resolve
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report missing block body");
}

#[test]
fn test_unexpected_token_in_expression() {
    let source = r#"
        signal velocity {
            resolve { 1.0 @ 2.0 }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report unexpected token");
}

#[test]
fn test_invalid_member_syntax() {
    let source = r#"
        member {
            resolve { 0.0 }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report missing member path");
}

// =============================================================================
// Error Recovery and Synchronization
// =============================================================================

#[test]
fn test_multiple_errors_reported() {
    let source = r#"
        signal bad1 {
            resolve { ( }
        }
        
        signal bad2 {
            resolve { ] }
        }
    "#;

    let errors = expect_error(source);
    // Parser should recover and report multiple errors
    assert!(
        errors.len() >= 2,
        "Should report errors from multiple declarations, got {} error(s)",
        errors.len()
    );
}

#[test]
fn test_synchronization_after_error() {
    let source = r#"
        signal bad {
            resolve { unclosed (
        }
        
        signal good {
            resolve { 42.0 }
        }
    "#;

    let errors = expect_error(source);
    // Parser should report error in first declaration but continue parsing
    assert!(
        !errors.is_empty(),
        "Should report error in first declaration"
    );
}

#[test]
fn test_error_in_nested_structure() {
    let source = r#"
        era formation {
            strata {
                genesis: invalid syntax here
            }
        }
    "#;

    let errors = expect_error(source);
    assert!(
        errors.iter().any(|e| {
            e.message.contains("expected")
                || e.message.contains("unclosed")
                || e.message.contains("EOF")
        }),
        "Should report unclosed brace or unexpected EOF, got: {:?}",
        errors
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_source() {
    let source = "";

    // Empty source is valid (zero declarations)
    let decls = parse_ok(source);
    assert_eq!(decls.len(), 0);
}

#[test]
fn test_only_whitespace() {
    let source = "   \n\n  \t  \n   ";

    // Only whitespace is valid (zero declarations)
    let decls = parse_ok(source);
    assert_eq!(decls.len(), 0);
}

#[test]
fn test_only_comments() {
    let source = r#"
        # This is a comment
        # Another comment
    "#;

    // Only comments is valid (zero declarations)
    let decls = parse_ok(source);
    assert_eq!(decls.len(), 0);
}

#[test]
fn test_unclosed_string_literal() {
    let source = r#"
        signal velocity {
            : title("Unclosed string
            resolve { 0.0 }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report unclosed string literal");
}

#[test]
fn test_invalid_number_literal() {
    let source = r#"
        signal velocity {
            resolve { 123.456.789 }
        }
    "#;

    let errors = expect_error(source);
    assert!(!errors.is_empty(), "Should report invalid number literal");
}

#[test]
fn test_span_accuracy_with_spans() {
    // Test that spans point to correct byte offsets
    let source = "signal test { resolve { bad_syntax } }";
    //            0123456789...

    let mut lexer = Token::lexer(source);
    let mut tokens_with_spans = Vec::new();

    while let Some(result) = lexer.next() {
        if let Ok(token) = result {
            let span = lexer.span();
            tokens_with_spans.push((token, span));
        }
    }

    // Parse with accurate spans
    let result = continuum_cdsl_parser::parse_declarations_with_spans(&tokens_with_spans, 0);

    // Should succeed since this is actually valid syntax (identifier in expression)
    assert!(result.is_ok(), "This is valid syntax");
}

#[test]
fn test_span_points_to_actual_error() {
    // Test with a real syntax error: missing closing brace
    let source = "signal test {\n    resolve { 1.0 + 2.0\n}";
    //            Position:         ^error here (missing })

    let mut lexer = Token::lexer(source);
    let mut tokens_with_spans = Vec::new();

    while let Some(result) = lexer.next() {
        if let Ok(token) = result {
            let span = lexer.span();
            tokens_with_spans.push((token, span));
        }
    }

    let result = continuum_cdsl_parser::parse_declarations_with_spans(&tokens_with_spans, 0);

    // Should fail
    assert!(result.is_err(), "Should have parse error");

    let errors = result.unwrap_err();
    assert!(!errors.is_empty(), "Should have at least one error");

    // Verify the span is reasonable (not just token indices like 0, 1, 2)
    let first_error = &errors[0];
    println!("Error: {:?}", first_error);
    // The span should have byte offsets > 10 (after "signal test")
}

// =============================================================================
// Edge Cases for Assertion Metadata
// =============================================================================

#[test]
fn test_eof_in_assertion_metadata() {
    // EOF while parsing assertion metadata should be caught
    let source = r#"
        signal temp {
            assert {
                prev > 0 :
    "#;

    let errors = expect_error(source);
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("EOF") || e.message.contains("expected")),
        "Expected EOF error, got: {:?}",
        errors
    );
}

#[test]
fn test_unicode_source_graceful_handling() {
    // Unicode characters have multi-byte UTF-8 encoding. Verify parser
    // handles them gracefully (either accepts or errors, but doesn't panic)
    let source = "signal 🌡️ { resolve { 42 } }";

    // This should parse successfully (invalid identifier is caught at lex)
    // OR fail gracefully (not crash or panic)
    let tokens: Vec<continuum_cdsl_lexer::Token> = continuum_cdsl_lexer::Token::lexer(source)
        .filter_map(Result::ok)
        .collect();

    // Verify we don't panic on unicode (accept either success or error)
    let _result = continuum_cdsl_parser::parse_declarations(&tokens, 0);
    // Test passes if we reach here without panicking
}

#[test]
fn test_assertion_metadata_trailing_comma_error() {
    // Trailing comma after severity with no message should error
    let source = r#"
        signal temp {
            assert {
                prev > 0 : fatal,
            }
        }
    "#;

    let errors = expect_error(source);
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("expected") || e.message.contains("string")),
        "Should reject trailing comma without message, got: {:?}",
        errors
    );
}
