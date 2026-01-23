// Integration tests that parse real .cdsl files
//
// These tests verify that the lexer and parser work correctly with
// realistic multi-line CDSL files, catching integration issues that
// inline string tests might miss.
//
// Note: These tests use simplified syntax that is currently supported.
// More complex features (type annotations, dotted paths, etc.) will be
// added as the parser evolves.

use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::parse_declarations;
use logos::Logos;

/// Helper: lex source and collect tokens, returning error on lexer failure
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
fn test_parse_simple_signal() {
    let source = include_str!("integration/simple_signal.cdsl");

    // Lex the source
    let tokens = lex(source).expect("Lexing should succeed");

    // Parse declarations
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    // Verify we got a signal declaration
    assert_eq!(declarations.len(), 1, "Should have 1 declaration");

    match &declarations[0] {
        continuum_cdsl_ast::Declaration::Node(node) => {
            assert!(matches!(node.role, continuum_cdsl_ast::RoleData::Signal));
        }
        _ => panic!("Expected Node(Signal) declaration"),
    }
}

#[test]
fn test_parse_simple_field() {
    let source = include_str!("integration/simple_field.cdsl");

    let tokens = lex(source).expect("Lexing should succeed");
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    assert_eq!(declarations.len(), 1, "Should have 1 declaration");

    match &declarations[0] {
        continuum_cdsl_ast::Declaration::Node(node) => {
            assert!(matches!(
                node.role,
                continuum_cdsl_ast::RoleData::Field { .. }
            ));
        }
        _ => panic!("Expected Node(Field) declaration"),
    }
}

#[test]
fn test_parse_simple_operator() {
    let source = include_str!("integration/simple_operator.cdsl");

    let tokens = lex(source).expect("Lexing should succeed");
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    assert_eq!(declarations.len(), 1, "Should have 1 declaration");

    match &declarations[0] {
        continuum_cdsl_ast::Declaration::Node(node) => {
            assert!(matches!(node.role, continuum_cdsl_ast::RoleData::Operator));
        }
        _ => panic!("Expected Node(Operator) declaration"),
    }
}

#[test]
fn test_multiline_expressions() {
    let source = include_str!("integration/multiline_expr.cdsl");

    let tokens = lex(source).expect("Lexing should succeed");
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    assert_eq!(declarations.len(), 1, "Should have 1 declaration");

    // Verify it's a signal with multi-line expressions
    match &declarations[0] {
        continuum_cdsl_ast::Declaration::Node(node) => {
            assert!(matches!(node.role, continuum_cdsl_ast::RoleData::Signal));
            // Note: Block structure validated by parser, we just verify it parses
        }
        _ => panic!("Expected Node(Signal) declaration"),
    }
}

#[test]
fn test_integration_lexer_parser_roundtrip() {
    // Test that lexer → parser works correctly for multiple declarations in one file
    let source = r#"
        signal temp {
            resolve { 300.0 }
        }
        
        field heat {
            measure { temp }
        }
        
        operator physics {
            resolve { gravity }
        }
    "#;

    let tokens = lex(source).expect("Lexing should succeed");
    let declarations = parse_declarations(&tokens, 0).expect("Parsing should succeed");

    assert_eq!(declarations.len(), 3, "Should have 3 declarations");

    // Verify each declaration type
    let signal_count = declarations.iter().filter(|d| matches!(
        d,
        continuum_cdsl_ast::Declaration::Node(node) if matches!(node.role, continuum_cdsl_ast::RoleData::Signal)
    )).count();
    assert_eq!(signal_count, 1);

    let field_count = declarations.iter().filter(|d| matches!(
        d,
        continuum_cdsl_ast::Declaration::Node(node) if matches!(node.role, continuum_cdsl_ast::RoleData::Field { .. })
    )).count();
    assert_eq!(field_count, 1);

    let operator_count = declarations.iter().filter(|d| matches!(
        d,
        continuum_cdsl_ast::Declaration::Node(node) if matches!(node.role, continuum_cdsl_ast::RoleData::Operator)
    )).count();
    assert_eq!(operator_count, 1);
}
