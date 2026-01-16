//! Tests for the lowering phase.

use continuum_dsl::parse;
use continuum_foundation::{FnId, SignalId, StratumId};

use crate::{BinaryOp, CompiledExpr, LowerError, ValueType, lower};

#[test]
fn test_lower_empty() {
    use continuum_dsl::ast::CompilationUnit;
    let unit = CompilationUnit::default();
    let world = lower(&unit).unwrap();
    assert!(world.signals().is_empty());
    assert!(world.strata().is_empty());
}
