//! Compile-fail tests for kernel_fn macro
//!
//! These tests verify that the macro correctly rejects invalid usage patterns.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
