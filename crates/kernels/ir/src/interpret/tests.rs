//! Tests for the interpret module.

use indexmap::IndexMap;

use continuum_runtime::SignalId;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::Value;

use crate::{
    BinaryOpIr, CompiledEmit, CompiledEra, CompiledExpr, CompiledFracture, CompiledTransition,
    CompiledWorld,
};

use super::build_transition_fn;

#[test]
fn test_build_transition_fn() {
    let constants = IndexMap::new();
    let config = IndexMap::new();
    let mut signals = SignalStorage::default();

    // Initialize signal at 100
    signals.init(SignalId::from("temp"), Value::Scalar(100.0));

    // Create an era with a transition when temp < 50
    let era = CompiledEra {
        file: None,
        span: 0..0,
        id: continuum_foundation::EraId::from("test"),
        is_initial: true,
        is_terminal: false,
        title: None,
        dt_seconds: 1.0,
        strata_states: IndexMap::new(),
        transitions: vec![CompiledTransition {
            target_era: continuum_foundation::EraId::from("next_era"),
            condition: CompiledExpr::Binary {
                op: BinaryOpIr::Lt,
                left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from(
                    "temp",
                ))),
                right: Box::new(CompiledExpr::Literal(50.0, None)),
            },
        }],
    };

    let transition_fn = build_transition_fn(&era, &constants, &config).unwrap();

    // Signal at 100, should not transition (100 < 50 is false)
    assert!(transition_fn(&signals, 0.0).is_none());

    // Update signal to 30
    signals.set_current(SignalId::from("temp"), Value::Scalar(30.0));

    // Signal at 30, should transition (30 < 50 is true)
    let result = transition_fn(&signals, 0.0);
    assert!(result.is_some());
    assert_eq!(result.unwrap().to_string(), "next_era");
}

#[test]
fn test_build_fracture() {
    use continuum_runtime::executor::FractureContext;
    use continuum_runtime::types::Dt;

    use super::build_fracture;

    let world = CompiledWorld {
        constants: IndexMap::new(),
        config: IndexMap::new(),
        nodes: IndexMap::new(),
    };

    // Create a fracture that triggers when temp > 100 and emits to energy
    let fracture = CompiledFracture {
        file: None,
        span: 0..0,
        id: continuum_foundation::FractureId::from("test_fracture"),
        stratum: continuum_foundation::StratumId::from("default"),
        reads: vec![continuum_foundation::SignalId::from("temp")],
        conditions: vec![CompiledExpr::Binary {
            op: BinaryOpIr::Gt,
            left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from(
                "temp",
            ))),
            right: Box::new(CompiledExpr::Literal(100.0, None)),
        }],
        emits: vec![CompiledEmit {
            target: continuum_foundation::SignalId::from("energy"),
            value: CompiledExpr::Literal(50.0, None),
        }],
    };

    let fracture_fn = build_fracture(&fracture, &world);

    let mut signals = SignalStorage::default();
    signals.init(SignalId::from("temp"), Value::Scalar(50.0));

    let ctx = FractureContext {
        signals: &signals,
        dt: Dt(1.0),
        sim_time: 0.0,
    };

    // Temp is 50, condition (temp > 100) is false, should not trigger
    assert!(fracture_fn(&ctx).is_none());

    // Update temp to 150
    signals.set_current(SignalId::from("temp"), Value::Scalar(150.0));
    let ctx = FractureContext {
        signals: &signals,
        dt: Dt(1.0),
        sim_time: 0.0,
    };

    // Temp is 150, condition (temp > 100) is true, should trigger
    let result = fracture_fn(&ctx);
    assert!(result.is_some());
    let emits = result.unwrap();
    assert_eq!(emits.len(), 1);
    assert_eq!(emits[0].0, SignalId::from("energy"));
    assert_eq!(emits[0].1, 50.0);
}
