//! Expression precedence and associativity tests.
//!
//! These tests verify the Pratt parser correctly handles operator precedence
//! and associativity across all 7 precedence levels and 10 binary operators.
//!
//! ## CDSL Operator Syntax
//!
//! CDSL uses keywords for logical operators:
//! - `or` (not `||`)
//! - `and` (not `&&`)
//! - `not` (not `!`)
//!
//! All other operators use symbols: `+`, `-`, `*`, `/`, `%`, `^`, `==`, `!=`, `<`, `<=`, `>`, `>=`

use continuum_cdsl_ast::{Expr, UntypedKind};
use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::parse_expr;
use logos::Logos;

/// Helper to parse an expression from source.
fn parse(source: &str) -> Expr {
    let tokens: Vec<Token> = Token::lexer(source).filter_map(Result::ok).collect();
    parse_expr(&tokens, 0).expect("Parse failed")
}

/// Helper to check if an expression is a binary operation.
fn is_binary(expr: &Expr, expected_op: &str) -> bool {
    match &expr.kind {
        UntypedKind::Binary { op, .. } => format!("{:?}", op).contains(expected_op),
        _ => false,
    }
}

/// Helper to get left and right operands of a binary expression.
fn get_operands(expr: &Expr) -> Option<(&Expr, &Expr)> {
    match &expr.kind {
        UntypedKind::Binary { left, right, .. } => Some((left.as_ref(), right.as_ref())),
        _ => None,
    }
}

// =============================================================================
// Precedence Level 1: || (Or) - Lowest Precedence
// =============================================================================

#[test]
fn test_or_vs_and() {
    // a or b and c should parse as: a or (b and c)
    let expr = parse("a or b and c");
    assert!(is_binary(&expr, "Or"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "And"));
}

#[test]
fn test_or_left_associative() {
    // a or b or c should parse as: (a or b) or c
    let expr = parse("a or b or c");
    assert!(is_binary(&expr, "Or"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Or"));
}

// =============================================================================
// Precedence Level 2: && (And)
// =============================================================================

#[test]
fn test_and_vs_comparison() {
    // a and b == c should parse as: a and (b == c)
    let expr = parse("a and b == c");
    assert!(is_binary(&expr, "And"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "Eq"));
}

#[test]
fn test_and_left_associative() {
    // a and b and c should parse as: (a and b) and c
    let expr = parse("a and b and c");
    assert!(is_binary(&expr, "And"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "And"));
}

// =============================================================================
// Precedence Level 3: Comparison (==, !=, <, <=, >, >=)
// =============================================================================

#[test]
fn test_comparison_vs_addition() {
    // a + b == c should parse as: (a + b) == c
    let expr = parse("a + b == c");
    assert!(is_binary(&expr, "Eq"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Add"));
    assert!(matches!(right.kind, UntypedKind::Local(_)));
}

#[test]
fn test_all_comparison_ops() {
    // Test all 6 comparison operators parse correctly
    for op in ["==", "!=", "<", "<=", ">", ">="] {
        let source = format!("a {} b", op);
        let expr = parse(&source);
        assert!(matches!(expr.kind, UntypedKind::Binary { .. }));
    }
}

#[test]
fn test_comparison_left_associative() {
    // a < b < c should parse as: (a < b) < c
    let expr = parse("a < b < c");
    assert!(is_binary(&expr, "Lt"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Lt"));
}

// =============================================================================
// Precedence Level 4: Addition/Subtraction (+, -)
// =============================================================================

#[test]
fn test_addition_vs_multiplication() {
    // a + b * c should parse as: a + (b * c)
    let expr = parse("a + b * c");
    assert!(is_binary(&expr, "Add"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "Mul"));
}

#[test]
fn test_subtraction_vs_multiplication() {
    // a - b / c should parse as: a - (b / c)
    let expr = parse("a - b / c");
    assert!(is_binary(&expr, "Sub"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "Div"));
}

#[test]
fn test_addition_left_associative() {
    // a + b - c should parse as: (a + b) - c
    let expr = parse("a + b - c");
    assert!(is_binary(&expr, "Sub"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Add"));
}

// =============================================================================
// Precedence Level 5: Multiplication/Division/Modulo (*, /, %)
// =============================================================================

#[test]
fn test_multiplication_vs_power() {
    // a * b ^ c should parse as: a * (b ^ c)
    let expr = parse("a * b ^ c");
    assert!(is_binary(&expr, "Mul"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "Pow"));
}

#[test]
fn test_division_vs_power() {
    // a / b ^ c should parse as: a / (b ^ c)
    let expr = parse("a / b ^ c");
    assert!(is_binary(&expr, "Div"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "Pow"));
}

#[test]
fn test_modulo() {
    // a % b should parse correctly
    let expr = parse("a % b");
    assert!(is_binary(&expr, "Mod"));
}

#[test]
fn test_multiplication_left_associative() {
    // a * b / c should parse as: (a * b) / c
    let expr = parse("a * b / c");
    assert!(is_binary(&expr, "Div"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Mul"));
}

// =============================================================================
// Precedence Level 6: Power (^) - RIGHT Associative
// =============================================================================

#[test]
fn test_power_vs_unary() {
    // -a ^ b should parse as: (-a) ^ b [unary binds tighter]
    let expr = parse("-a ^ b");
    assert!(is_binary(&expr, "Pow"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Unary { .. }));
}

#[test]
fn test_power_right_associative() {
    // a ^ b ^ c should parse as: a ^ (b ^ c) [RIGHT associative]
    let expr = parse("a ^ b ^ c");
    assert!(is_binary(&expr, "Pow"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Local(_)));
    assert!(is_binary(right, "Pow"));
}

// =============================================================================
// Precedence Level 7: Unary (-, !)
// =============================================================================

#[test]
fn test_unary_minus() {
    let expr = parse("-a");
    assert!(matches!(expr.kind, UntypedKind::Unary { .. }));
}

#[test]
fn test_unary_not() {
    let expr = parse("not a");
    assert!(matches!(expr.kind, UntypedKind::Unary { .. }));
}

#[test]
fn test_unary_vs_postfix() {
    // -a.field should parse as: -(a.field)
    let expr = parse("-a.field");
    assert!(matches!(expr.kind, UntypedKind::Unary { .. }));
    match &expr.kind {
        UntypedKind::Unary { operand, .. } => {
            assert!(matches!(operand.kind, UntypedKind::FieldAccess { .. }));
        }
        _ => panic!("Expected unary"),
    }
}

#[test]
fn test_double_unary() {
    // --a should parse as: -(-a)
    let expr = parse("--a");
    assert!(matches!(expr.kind, UntypedKind::Unary { .. }));
    match &expr.kind {
        UntypedKind::Unary { operand, .. } => {
            assert!(matches!(operand.kind, UntypedKind::Unary { .. }));
        }
        _ => panic!("Expected nested unary"),
    }
}

// =============================================================================
// Precedence Level 8: Postfix (.field, (args))
// =============================================================================

#[test]
fn test_field_access() {
    let expr = parse("a.field");
    assert!(matches!(expr.kind, UntypedKind::FieldAccess { .. }));
}

#[test]
fn test_chained_field_access() {
    // a.b.c should parse as: (a.b).c
    let expr = parse("a.b.c");
    assert!(matches!(expr.kind, UntypedKind::FieldAccess { .. }));
    match &expr.kind {
        UntypedKind::FieldAccess { object, .. } => {
            assert!(matches!(object.kind, UntypedKind::FieldAccess { .. }));
        }
        _ => panic!("Expected field access"),
    }
}

#[test]
fn test_function_call() {
    let expr = parse("func(a, b)");
    assert!(matches!(expr.kind, UntypedKind::Call { .. }));
}

#[test]
fn test_field_then_call() {
    // a.method(b) should parse as: (a.method)(b)
    let expr = parse("a.method(b)");
    assert!(matches!(expr.kind, UntypedKind::Call { .. }));
}

// =============================================================================
// Complex Precedence Combinations
// =============================================================================

#[test]
fn test_complex_expression_1() {
    // a + b * c ^ d should parse as: a + (b * (c ^ d))
    let expr = parse("a + b * c ^ d");
    assert!(is_binary(&expr, "Add"));
    let (_left, right) = get_operands(&expr).unwrap();
    assert!(is_binary(right, "Mul"));
    let (_left2, right2) = get_operands(right).unwrap();
    assert!(is_binary(right2, "Pow"));
}

#[test]
fn test_complex_expression_2() {
    // a or b and c == d + e * f
    // Should parse as: a or (b and (c == (d + (e * f))))
    let expr = parse("a or b and c == d + e * f");
    assert!(is_binary(&expr, "Or"));
    let (_left, right) = get_operands(&expr).unwrap();
    assert!(is_binary(right, "And"));
    let (_left2, right2) = get_operands(right).unwrap();
    assert!(is_binary(right2, "Eq"));
    let (_left3, right3) = get_operands(right2).unwrap();
    assert!(is_binary(right3, "Add"));
    let (_left4, right4) = get_operands(right3).unwrap();
    assert!(is_binary(right4, "Mul"));
}

#[test]
fn test_complex_expression_3() {
    // -a.field + b ^ c * d
    // Should parse as: (-(a.field)) + ((b ^ c) * d)
    let expr = parse("-a.field + b ^ c * d");
    assert!(is_binary(&expr, "Add"));
    let (left, right) = get_operands(&expr).unwrap();
    assert!(matches!(left.kind, UntypedKind::Unary { .. }));
    assert!(is_binary(right, "Mul"));
}

// =============================================================================
// Parentheses Override Precedence
// =============================================================================

#[test]
fn test_parentheses_override_1() {
    // (a + b) * c should parse as: (a + b) * c
    let expr = parse("(a + b) * c");
    assert!(is_binary(&expr, "Mul"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Add"));
}

#[test]
fn test_parentheses_override_2() {
    // a ^ (b ^ c) should parse as: a ^ (b ^ c) [explicit right assoc]
    let expr = parse("a ^ (b ^ c)");
    assert!(is_binary(&expr, "Pow"));
    let (_left, right) = get_operands(&expr).unwrap();
    assert!(is_binary(right, "Pow"));
}

#[test]
fn test_nested_parentheses() {
    // ((a + b) * c) + d
    let expr = parse("((a + b) * c) + d");
    assert!(is_binary(&expr, "Add"));
    let (left, _right) = get_operands(&expr).unwrap();
    assert!(is_binary(left, "Mul"));
}
