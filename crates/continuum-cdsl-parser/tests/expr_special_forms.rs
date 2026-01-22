//! Tests for special expression forms: if, let, filter, aggregates, spatial queries.
//!
//! These tests verify that the parser correctly handles:
//! - if-then-else expressions
//! - let-in bindings
//! - filter operations
//! - aggregate operations (agg.sum, agg.mean, etc.)
//! - spatial queries (nearest, within)

use continuum_cdsl_lexer::Token;
use continuum_cdsl_parser::parse_declarations;
use logos::Logos;

/// Helper to test that an expression parses successfully in a signal resolve block.
fn assert_expr_parses(expr: &str) {
    let source = format!("signal test {{ resolve {{ {} }} }}", expr);
    let tokens: Vec<Token> = Token::lexer(&source).filter_map(Result::ok).collect();

    let result = parse_declarations(&tokens, 0);
    assert!(
        result.is_ok(),
        "Failed to parse '{}': {:?}",
        expr,
        result.err()
    );
}

// === If-Then-Else Tests ===

#[test]
fn test_if_simple() {
    assert_expr_parses("if active { 1.0 } else { 0.0 }");
}

#[test]
fn test_if_with_comparison() {
    assert_expr_parses("if x > 0.0 { x } else { 0.0 }");
}

#[test]
fn test_if_with_and() {
    assert_expr_parses("if x > 0.0 and y > 0.0 { 1.0 } else { 0.0 }");
}

#[test]
fn test_if_with_or() {
    assert_expr_parses("if x > 10.0 or y > 10.0 { 1.0 } else { 0.0 }");
}

#[test]
fn test_if_nested() {
    assert_expr_parses("if x > 0.0 { if y > 0.0 { 1.0 } else { 2.0 } } else { 3.0 }");
}

#[test]
fn test_if_with_computation() {
    assert_expr_parses("if active { mass * velocity } else { 0.0 }");
}

// === Let-In Tests ===
// NOTE: let-in expressions conflict with let statements in block body disambiguation
// The parser's is_statement_start() treats Let as a statement, preventing let-in expressions
// in resolve blocks. These tests are ignored until parser disambiguation is fixed.

#[test]
#[ignore] // Parser disambiguation issue: Let treated as statement, not expression
fn test_let_simple() {
    assert_expr_parses("let x = 10.0 in x");
}

#[test]
#[ignore] // Parser disambiguation issue
fn test_let_with_expression() {
    assert_expr_parses("let x = 10.0 in x * 2.0");
}

#[test]
#[ignore] // Parser disambiguation issue
fn test_let_with_computation() {
    assert_expr_parses("let area = width * height in area");
}

#[test]
#[ignore] // Parser disambiguation issue
fn test_let_nested() {
    assert_expr_parses("let x = 10.0 in let y = x * 2.0 in y");
}

#[test]
#[ignore] // Parser disambiguation issue
fn test_let_in_if() {
    assert_expr_parses("let threshold = 5.0 in if x > threshold { x } else { 0.0 }");
}

#[test]
#[ignore] // Parser disambiguation issue
fn test_if_with_let_in_branch() {
    assert_expr_parses("if active { let v = velocity in v * mass } else { 0.0 }");
}

// === Aggregate Tests ===

#[test]
fn test_agg_sum() {
    assert_expr_parses("agg.sum(particles, self.mass)");
}

#[test]
fn test_agg_mean() {
    assert_expr_parses("agg.mean(particles, self.velocity)");
}

#[test]
fn test_agg_min() {
    assert_expr_parses("agg.min(cells, self.temperature)");
}

#[test]
fn test_agg_max() {
    assert_expr_parses("agg.max(cells, self.pressure)");
}

#[test]
fn test_agg_count() {
    assert_expr_parses("agg.count(agents)");
}

#[test]
fn test_agg_count_with_predicate() {
    assert_expr_parses("agg.count(agents, self.active)");
}

#[test]
fn test_agg_all() {
    assert_expr_parses("agg.all(validators, self.ready)");
}

#[test]
fn test_agg_any() {
    assert_expr_parses("agg.any(nodes, self.failed)");
}

#[test]
fn test_agg_none() {
    assert_expr_parses("agg.none(items, self.invalid)");
}

#[test]
fn test_agg_map() {
    assert_expr_parses("agg.map(points, self.position)");
}

#[test]
#[ignore] // "first" is a keyword token, parser expects Ident
fn test_agg_first() {
    assert_expr_parses("agg.first(queue, self.priority)");
}

#[test]
fn test_agg_product() {
    assert_expr_parses("agg.product(factors, self.value)");
}

// === Filter Tests ===

#[test]
fn test_filter_simple() {
    assert_expr_parses("filter(particles, self.active)");
}

#[test]
fn test_filter_with_comparison() {
    assert_expr_parses("filter(agents, self.health > 50.0)");
}

#[test]
fn test_filter_with_and() {
    assert_expr_parses("filter(cells, self.active and self.temperature > 300.0)");
}

#[test]
fn test_filter_with_computation() {
    assert_expr_parses("filter(items, self.mass * self.volume > 100.0)");
}

// === Spatial Query Tests ===

#[test]
fn test_nearest() {
    assert_expr_parses("nearest(particles, self.position)");
}

#[test]
fn test_nearest_with_local() {
    assert_expr_parses("nearest(agents, target_position)");
}

#[test]
fn test_within() {
    assert_expr_parses("within(particles, self.position, 10.0)");
}

#[test]
fn test_within_with_unit() {
    assert_expr_parses("within(cells, center, 5.0<m>)");
}

#[test]
fn test_within_with_computation() {
    assert_expr_parses("within(nodes, origin, search_radius * 2.0)");
}

// === Combined Operations ===

#[test]
fn test_agg_on_filter() {
    assert_expr_parses("agg.sum(filter(plates, self.active), self.mass)");
}

#[test]
fn test_agg_on_nearest() {
    assert_expr_parses("agg.mean(nearest(particles, pos), self.velocity)");
}

#[test]
fn test_agg_on_within() {
    assert_expr_parses("agg.count(within(cells, center, 10.0))");
}

#[test]
fn test_filter_on_nearest() {
    assert_expr_parses("filter(nearest(nodes, origin), self.distance < 100.0)");
}

#[test]
#[ignore] // Contains let-in (parser disambiguation issue)
fn test_let_with_agg() {
    assert_expr_parses("let total = agg.sum(items, self.value) in total / 10.0");
}

#[test]
fn test_if_with_agg() {
    assert_expr_parses("if agg.count(active_agents) > 5 { 1.0 } else { 0.0 }");
}

#[test]
#[ignore] // Contains let-in (parser disambiguation issue)
fn test_complex_nested() {
    assert_expr_parses(
        "let nearby = filter(within(particles, self.position, 10.0), self.active) in agg.sum(nearby, self.mass)"
    );
}

// === Edge Cases ===

#[test]
#[ignore] // Contains let-in (parser disambiguation issue)
fn test_multiple_let_bindings() {
    assert_expr_parses("let x = 1.0 in let y = 2.0 in let z = 3.0 in x + y + z");
}

#[test]
fn test_agg_with_complex_body() {
    assert_expr_parses("agg.sum(plates, self.mass * self.area * 2.0)");
}

#[test]
#[ignore] // Contains let-in (parser disambiguation issue)
fn test_nested_if_in_let() {
    assert_expr_parses("let x = if active { 10.0 } else { 5.0 } in x * 2.0");
}

#[test]
fn test_filter_with_or() {
    assert_expr_parses("filter(items, self.active or self.pending)");
}

#[test]
fn test_agg_chain() {
    assert_expr_parses("agg.mean(items, self.value) + agg.max(items, self.value)");
}
