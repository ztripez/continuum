//! Type and unit expression parsing tests.
//!
//! Tests type/unit expressions parse successfully in signal declarations.
//!
//! ## Features Tested
//!
//! **Scalar bounds**: `Scalar<K, 0.0..1000.0>`, `Scalar<C, -273.15..1000.0>`
//! - Bounds are parsed using `parse_primary` which handles unary operators
//!   but not binary operators (to avoid consuming `>` as comparison)
//!
//! **Negative unit exponents**: `s^-1`, `kg*m^2*s^-2`
//! - Parser handles `^-N` by consuming minus token after caret

use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::parse_declarations;
use logos::Logos;

/// Helper to test that a type expression parses successfully in a type declaration.
fn assert_type_parses(type_str: &str) {
    let source = format!("type Test {{ value: {} }}", type_str);
    let tokens: Vec<Token> = Token::lexer(&source).filter_map(Result::ok).collect();

    let result = parse_declarations(&tokens, 0);
    assert!(
        result.is_ok(),
        "Failed to parse '{}': {:?}",
        type_str,
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "Expected exactly one declaration");
}

// === Basic Type Expressions ===

#[test]
fn test_type_bool() {
    assert_type_parses("Bool");
}

#[test]
fn test_type_scalar_no_unit() {
    assert_type_parses("Scalar");
}

#[test]
fn test_type_scalar_with_unit() {
    assert_type_parses("Scalar<m>");
}

#[test]
fn test_type_scalar_kg() {
    assert_type_parses("Scalar<kg>");
}

#[test]
fn test_type_scalar_kelvin() {
    assert_type_parses("Scalar<K>");
}

// === Dimensionless Scalar Tests ===

#[test]
fn test_type_scalar_dimensionless_empty() {
    assert_type_parses("Scalar<>");
}

#[test]
fn test_type_scalar_dimensionless_one() {
    assert_type_parses("Scalar<1>");
}

#[test]
fn test_type_scalar_dimensionless_with_bounds() {
    assert_type_parses("Scalar<1, 0.0..1.0>");
}

#[test]
fn test_type_scalar_dimensionless_empty_with_bounds() {
    assert_type_parses("Scalar<, 0.0..100.0>");
}

#[test]
fn test_type_scalar_with_bounds() {
    assert_type_parses("Scalar<K, 0.0..1000.0>");
}

#[test]
fn test_type_scalar_negative_bounds() {
    assert_type_parses("Scalar<C, -273.15..1000.0>");
}

#[test]
fn test_type_vector_2d() {
    assert_type_parses("Vector<2, m>");
}

#[test]
fn test_type_vector_3d() {
    assert_type_parses("Vector<3, m/s>");
}

#[test]
fn test_type_user_simple() {
    assert_type_parses("Position");
}

#[test]
fn test_type_user_path() {
    assert_type_parses("physics.Motion");
}

#[test]
fn test_type_user_nested() {
    assert_type_parses("game.plate.Motion");
}

// === Unit Expression Tests ===

#[test]
fn test_unit_multiply() {
    assert_type_parses("Scalar<kg*m>");
}

#[test]
fn test_unit_divide() {
    assert_type_parses("Scalar<m/s>");
}

#[test]
fn test_unit_power_positive() {
    assert_type_parses("Scalar<m^2>");
}

#[test]
fn test_unit_power_negative() {
    assert_type_parses("Scalar<s^-1>");
}

#[test]
fn test_unit_mixed_operations() {
    assert_type_parses("Scalar<kg*m^2*s^-2>");
}

#[test]
fn test_unit_power_three() {
    assert_type_parses("Scalar<m^3>");
}

// === Complex Unit Expressions ===

#[test]
fn test_unit_velocity() {
    assert_type_parses("Scalar<m/s>");
}

#[test]
fn test_unit_acceleration() {
    assert_type_parses("Scalar<m/s^2>");
}

#[test]
fn test_unit_force_newton() {
    assert_type_parses("Scalar<kg*m/s^2>");
}

#[test]
fn test_unit_energy_joule() {
    assert_type_parses("Scalar<kg*m^2/s^2>");
}

#[test]
fn test_unit_pressure_pascal() {
    assert_type_parses("Scalar<kg/m/s^2>");
}

#[test]
fn test_unit_density() {
    assert_type_parses("Scalar<kg/m^3>");
}

#[test]
fn test_unit_volume() {
    assert_type_parses("Scalar<m^3>");
}

#[test]
fn test_unit_area() {
    assert_type_parses("Scalar<m^2>");
}

#[test]
fn test_unit_parenthesized() {
    assert_type_parses("Scalar<(kg*m)/s^2>");
}

#[test]
fn test_unit_multiple_multiply() {
    assert_type_parses("Scalar<kg*m*s>");
}

#[test]
fn test_unit_multiple_divide() {
    assert_type_parses("Scalar<m/s/s>");
}

#[test]
fn test_unit_compound_power() {
    assert_type_parses("Scalar<kg^2*m/s^3>");
}

// === Vector with Complex Units ===

#[test]
fn test_vector_velocity() {
    assert_type_parses("Vector<3, m/s>");
}

#[test]
fn test_vector_acceleration() {
    assert_type_parses("Vector<3, m/s^2>");
}

#[test]
fn test_vector_force() {
    assert_type_parses("Vector<3, kg*m/s^2>");
}

#[test]
fn test_vector_2d_velocity() {
    assert_type_parses("Vector<2, m/s>");
}

#[test]
fn test_vector_4d() {
    assert_type_parses("Vector<4, m>");
}

// === Custom Units ===

#[test]
fn test_unit_custom_radians() {
    assert_type_parses("Scalar<radians>");
}

#[test]
fn test_unit_custom_complex() {
    assert_type_parses("Scalar<newton*meter>");
}

#[test]
fn test_unit_angular_velocity() {
    assert_type_parses("Scalar<radians/s>");
}

#[test]
fn test_unit_torque() {
    assert_type_parses("Scalar<kg*m^2/s^2>");
}
