// Integration tests that parse real .cdsl files
//
// These tests verify that the lexer and parser work correctly with
// realistic multi-line CDSL files, catching integration issues that
// inline string tests might miss.

use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::parse_declarations;
use logos::Logos;

/// Helper: lex source and collect tokens, panicking on lexer errors
fn lex(source: &str) -> Result<Vec<Token>, String> {
    let results: Vec<Result<Token, ()>> = Token::lexer(source).collect();

    let mut tokens = Vec::new();
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(token) => tokens.push(token.clone()),
            Err(_) => return Err(format!("Lexer error at token {}", i)),
        }
    }

    Ok(tokens)
}

#[test]
fn test_parse_assertion_metadata() {
    let source = include_str!("integration/assertion_metadata.cdsl");

    // Lex the source
    let tokens = lex(source).expect("Lexing should succeed");

    // Parse declarations
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    // Verify we got expected declarations
    assert!(
        !declarations.is_empty(),
        "Should parse at least one declaration"
    );

    // Verify signal declarations exist
    let signals: Vec<_> = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Signal)
            }
            _ => false,
        })
        .collect();
    assert_eq!(signals.len(), 1, "Should have 1 signal declaration");

    // Verify field declarations
    let fields: Vec<_> = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Field { .. })
            }
            _ => false,
        })
        .collect();
    assert_eq!(fields.len(), 1, "Should have 1 field declaration");

    // Verify operator declarations
    let operators: Vec<_> = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Operator)
            }
            _ => false,
        })
        .collect();
    assert_eq!(operators.len(), 1, "Should have 1 operator declaration");
}

#[test]
fn test_parse_span_tracking() {
    let source = include_str!("integration/span_tracking.cdsl");

    // Lex the source
    let tokens = lex(source).expect("Lexing should succeed");

    // Parse declarations
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    // Verify parsing succeeded
    assert!(!declarations.is_empty(), "Should parse declarations");

    // Verify we have signals with multi-line expressions
    let signals: Vec<_> = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Signal)
            }
            _ => false,
        })
        .collect();
    assert_eq!(
        signals.len(),
        1,
        "Should have 1 signal with multi-line expressions"
    );

    // Verify operators
    let operators: Vec<_> = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Operator)
            }
            _ => false,
        })
        .collect();
    assert_eq!(operators.len(), 1, "Should have 1 operator");

    // Verify fields
    let fields: Vec<_> = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Field { .. })
            }
            _ => false,
        })
        .collect();
    assert_eq!(fields.len(), 2, "Should have 2 fields");
}

#[test]
fn test_parse_realistic_declarations() {
    let source = include_str!("integration/realistic_declarations.cdsl");

    // Lex the source
    let tokens = lex(source).expect("Lexing should succeed");

    // Parse declarations
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    // Verify we got many declarations
    assert!(
        declarations.len() >= 10,
        "Should have at least 10 declarations, got {}",
        declarations.len()
    );

    // Count declaration types
    let type_count = declarations
        .iter()
        .filter(|d| matches!(d, continuum_cdsl_ast::Declaration::Type(_)))
        .count();
    assert_eq!(type_count, 2, "Should have 2 type declarations");

    let config_count = declarations
        .iter()
        .filter(|d| matches!(d, continuum_cdsl_ast::Declaration::Config(_)))
        .count();
    assert_eq!(config_count, 1, "Should have 1 config declaration");

    let const_count = declarations
        .iter()
        .filter(|d| matches!(d, continuum_cdsl_ast::Declaration::Const(_)))
        .count();
    assert_eq!(const_count, 1, "Should have 1 const declaration");

    let entity_count = declarations
        .iter()
        .filter(|d| matches!(d, continuum_cdsl_ast::Declaration::Entity(_)))
        .count();
    assert_eq!(entity_count, 1, "Should have 1 entity declaration");

    let signal_count = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Signal)
            }
            _ => false,
        })
        .count();
    assert_eq!(signal_count, 1, "Should have 1 signal declaration");

    let field_count = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Field { .. })
            }
            _ => false,
        })
        .count();
    assert_eq!(field_count, 1, "Should have 1 field declaration");

    let operator_count = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Operator)
            }
            _ => false,
        })
        .count();
    assert_eq!(operator_count, 1, "Should have 1 operator declaration");

    let impulse_count = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Impulse { .. })
            }
            _ => false,
        })
        .count();
    assert_eq!(impulse_count, 1, "Should have 1 impulse declaration");

    let fracture_count = declarations
        .iter()
        .filter(|d| match d {
            continuum_cdsl_ast::Declaration::Node(node) => {
                matches!(node.role, continuum_cdsl_ast::RoleData::Fracture)
            }
            _ => false,
        })
        .count();
    assert_eq!(fracture_count, 1, "Should have 1 fracture declaration");
}

#[test]
fn test_unicode_in_comments_and_strings() {
    // Test that unicode characters in comments and strings are handled correctly
    let source = include_str!("integration/assertion_metadata.cdsl");

    let tokens = lex(source).expect("Lexing should succeed with unicode");
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed with unicode");

    assert!(
        !declarations.is_empty(),
        "Should parse declarations with unicode"
    );
}

#[test]
fn test_complex_nested_expressions() {
    // Verify deeply nested multi-line expressions parse correctly
    let source = include_str!("integration/span_tracking.cdsl");

    let tokens = lex(source).expect("Lexing should succeed");
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    // Just verify it parses - complex expression structure is validated
    // by the resolver, not the parser
    assert!(
        !declarations.is_empty(),
        "Should parse complex nested expressions"
    );
}
