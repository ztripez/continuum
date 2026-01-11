//! Integration tests for end-to-end Continuum execution.
//!
//! These tests verify the full pipeline:
//! Load world → Compile → Execute → Verify
//!
//! Related to: https://github.com/ztripez/continuum/issues/28

use continuum_tests::TestHarness;

/// Test that a simple world with one signal executes correctly.
///
/// Verifies: Parse → Lower → Compile → Execute → Signal values
#[test]
fn test_simple_world_executes() {
    let source = r#"
        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.counter {
            : Scalar<unit>
            : strata(terra)
            resolve { prev + 1.0 }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Initial value should be 0
    assert_eq!(harness.get_scalar("terra.counter"), Some(0.0));

    // Execute 10 ticks
    harness.run_ticks(10);

    // Counter should have incremented 10 times
    assert_eq!(harness.get_scalar("terra.counter"), Some(10.0));
}

/// Test that signal dependency chains propagate across ticks.
///
/// Chain: A (base) → B (reads A) → C (reads B)
///
/// Per docs/execution/phases.md: signal references read values from the
/// PREVIOUS tick, making resolution order-independent. Values propagate
/// across multiple ticks through the dependency chain.
#[test]
fn test_signal_dependency_chain() {
    let source = r#"
        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        # Base signal: constant 10
        signal.terra.base {
            : Scalar<unit>
            : strata(terra)
            resolve { 10.0 }
        }

        # Derived: doubles base (reads base from previous tick)
        signal.terra.doubled {
            : Scalar<unit>
            : strata(terra)
            resolve { signal.terra.base * 2.0 }
        }

        # Final: adds 5 to doubled (reads doubled from previous tick)
        signal.terra.final {
            : Scalar<unit>
            : strata(terra)
            resolve { signal.terra.doubled + 5.0 }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Initial: all signals start at 0.0
    assert_eq!(harness.get_scalar("terra.base"), Some(0.0));
    assert_eq!(harness.get_scalar("terra.doubled"), Some(0.0));
    assert_eq!(harness.get_scalar("terra.final"), Some(0.0));

    // Tick 1: base resolves to 10, others read previous tick's values (0)
    harness.tick();
    assert_eq!(harness.get_scalar("terra.base"), Some(10.0));
    assert_eq!(harness.get_scalar("terra.doubled"), Some(0.0));  // 0.0 * 2.0
    assert_eq!(harness.get_scalar("terra.final"), Some(5.0));   // 0.0 + 5.0

    // Tick 2: doubled now sees base=10 from previous tick
    harness.tick();
    assert_eq!(harness.get_scalar("terra.base"), Some(10.0));
    assert_eq!(harness.get_scalar("terra.doubled"), Some(20.0)); // 10.0 * 2.0
    assert_eq!(harness.get_scalar("terra.final"), Some(5.0));    // 0.0 + 5.0

    // Tick 3: final now sees doubled=20 from previous tick
    harness.tick();
    assert_eq!(harness.get_scalar("terra.base"), Some(10.0));
    assert_eq!(harness.get_scalar("terra.doubled"), Some(20.0)); // 10.0 * 2.0
    assert_eq!(harness.get_scalar("terra.final"), Some(25.0));   // 20.0 + 5.0
}

/// Test that execution is deterministic.
///
/// Same world, same initial conditions → identical results.
#[test]
fn test_execution_is_deterministic() {
    let source = r#"
        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.growth {
            : Scalar<unit>
            : strata(terra)
            resolve { prev * 1.5 + 1.0 }
        }
    "#;

    // Run twice with same source
    let mut harness1 = TestHarness::from_source(source);
    let mut harness2 = TestHarness::from_source(source);

    // Initialize both to same starting value
    // (default is 0.0 for both)

    // Execute same number of ticks
    for _ in 0..20 {
        harness1.tick();
        harness2.tick();
    }

    // Results must be identical
    let v1 = harness1.get_scalar("terra.growth").unwrap();
    let v2 = harness2.get_scalar("terra.growth").unwrap();

    assert!(
        (v1 - v2).abs() < f64::EPSILON,
        "Determinism violated: {} != {}",
        v1,
        v2
    );
}

/// Test era transitions work correctly.
///
/// World with two eras: era_a → era_b
/// Transition when counter reaches 5.
#[test]
fn test_era_transitions_work() {
    let source = r#"
        strata.terra {}

        era.era_a {
            : initial

            strata {
                terra: active
            }

            transition {
                to: era.era_b
                when {
                    signal.terra.counter >= 5.0
                }
            }
        }

        era.era_b {
            strata {
                terra: active
            }
        }

        signal.terra.counter {
            : Scalar<unit>
            : strata(terra)
            resolve { prev + 1.0 }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Start in era_a
    assert_eq!(harness.current_era().0, "era_a");

    // Execute ticks until transition
    for i in 0..10 {
        harness.tick();
        let era = harness.current_era().0.clone();
        let counter = harness.get_scalar("terra.counter").unwrap();

        // After tick 4 (counter=5), should transition to era_b
        if i < 4 {
            assert_eq!(era, "era_a", "tick {}: should still be in era_a", i);
        } else {
            assert_eq!(era, "era_b", "tick {}: should be in era_b", i);
        }

        // Counter keeps incrementing regardless of era
        assert_eq!(counter, (i + 1) as f64);
    }
}

/// Test multiple signals without dependencies execute in parallel.
///
/// Two independent signals should both resolve each tick.
#[test]
fn test_parallel_independent_signals() {
    let source = r#"
        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.alpha {
            : Scalar<unit>
            : strata(terra)
            resolve { prev + 1.0 }
        }

        signal.terra.beta {
            : Scalar<unit>
            : strata(terra)
            resolve { prev + 100.0 }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    harness.run_ticks(5);

    // Both should have executed independently
    assert_eq!(harness.get_scalar("terra.alpha"), Some(5.0));
    assert_eq!(harness.get_scalar("terra.beta"), Some(500.0));
}

/// Test that constants are accessible in resolve expressions.
#[test]
fn test_constants_in_resolve() {
    let source = r#"
        const {
            physics.gravity: 9.81
            physics.time_scale: 2.0
        }

        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.velocity {
            : Scalar<m/s>
            : strata(terra)
            resolve { prev + const.physics.gravity * const.physics.time_scale }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    harness.tick();

    // velocity = 0 + 9.81 * 2.0 = 19.62
    let velocity = harness.get_scalar("terra.velocity").unwrap();
    assert!((velocity - 19.62).abs() < 0.001);
}

/// Test that config values are accessible in resolve expressions.
#[test]
fn test_config_in_resolve() {
    let source = r#"
        config {
            simulation.decay_rate: 0.5
            terra.initial_energy: 100.0
        }

        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.energy {
            : Scalar<J>
            : strata(terra)
            resolve { prev * config.simulation.decay_rate }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Initial: 100
    assert_eq!(harness.get_scalar("terra.energy"), Some(100.0));

    harness.tick();
    // After tick 1: 100 * 0.5 = 50
    let energy = harness.get_scalar("terra.energy").unwrap();
    assert!((energy - 50.0).abs() < 0.001);

    harness.tick();
    // After tick 2: 50 * 0.5 = 25
    let energy = harness.get_scalar("terra.energy").unwrap();
    assert!((energy - 25.0).abs() < 0.001);
}

/// Test kernel functions work in resolve expressions.
#[test]
fn test_kernel_functions() {
    let source = r#"
        config {
            terra.initial_temp: 1000.0
        }

        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.temp {
            : Scalar<K>
            : strata(terra)
            resolve { decay(prev, 100.0) }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Initial: 1000
    assert_eq!(harness.get_scalar("terra.temp"), Some(1000.0));

    // Execute several ticks
    harness.run_ticks(10);

    // After decay, should be lower than initial
    let temp = harness.get_scalar("terra.temp").unwrap();
    assert!(temp < 1000.0, "Temperature should have decayed");
    assert!(temp > 0.0, "Temperature should still be positive");
}

/// Test complex expression with multiple operations.
///
/// Per docs/execution/phases.md: signal references read values from the
/// PREVIOUS tick. So complex reads base from the previous tick.
#[test]
fn test_complex_expression() {
    let source = r#"
        const {
            scale: 2.0
        }

        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.base {
            : Scalar<unit>
            : strata(terra)
            resolve { 10.0 }
        }

        signal.terra.complex {
            : Scalar<unit>
            : strata(terra)
            resolve {
                (signal.terra.base * const.scale + 5.0) / 2.0 - 1.0
            }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Tick 1: base=10, complex reads base=0 from previous tick
    // complex = (0 * 2 + 5) / 2 - 1 = 5 / 2 - 1 = 2.5 - 1 = 1.5
    harness.tick();
    let complex = harness.get_scalar("terra.complex").unwrap();
    assert!((complex - 1.5).abs() < 0.001);

    // Tick 2: base=10, complex reads base=10 from previous tick
    // complex = (10 * 2 + 5) / 2 - 1 = 25 / 2 - 1 = 12.5 - 1 = 11.5
    harness.tick();
    let complex = harness.get_scalar("terra.complex").unwrap();
    assert!((complex - 11.5).abs() < 0.001);
}

/// Test if-then-else conditional expressions.
///
/// Per docs/execution/phases.md: signal references read values from the
/// PREVIOUS tick. So threshold reads counter from the previous tick,
/// meaning the condition triggers one tick after counter exceeds 5.
#[test]
fn test_conditional_expression() {
    let source = r#"
        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.counter {
            : Scalar<unit>
            : strata(terra)
            resolve { prev + 1.0 }
        }

        signal.terra.threshold {
            : Scalar<unit>
            : strata(terra)
            resolve {
                if signal.terra.counter > 5.0 { 100.0 } else { 0.0 }
            }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    // Ticks 1-6: threshold reads counter from previous tick
    // Tick 1: counter=1, threshold sees counter=0 from prev tick -> 0
    // Tick 2: counter=2, threshold sees counter=1 -> 0
    // Tick 3: counter=3, threshold sees counter=2 -> 0
    // Tick 4: counter=4, threshold sees counter=3 -> 0
    // Tick 5: counter=5, threshold sees counter=4 -> 0
    // Tick 6: counter=6, threshold sees counter=5 -> 0 (5 is NOT > 5)
    for i in 0..6 {
        harness.tick();
        assert_eq!(
            harness.get_scalar("terra.threshold"),
            Some(0.0),
            "tick {}: threshold should be 0 (counter from prev tick <= 5)",
            i + 1
        );
    }

    // Tick 7: counter=7, threshold sees counter=6 from prev tick -> 6 > 5 = true
    harness.tick();
    assert_eq!(harness.get_scalar("terra.threshold"), Some(100.0));
}

/// Test multiple strata execute independently.
#[test]
fn test_multiple_strata() {
    let source = r#"
        strata.terra {}
        strata.climate {}

        era.main {
            : initial
            strata {
                terra: active
                climate: active
            }
        }

        signal.terra.ground_temp {
            : Scalar<K>
            : strata(terra)
            resolve { prev + 1.0 }
        }

        signal.climate.air_temp {
            : Scalar<K>
            : strata(climate)
            resolve { prev + 10.0 }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    harness.run_ticks(5);

    // Each stratum should have executed independently
    assert_eq!(harness.get_scalar("terra.ground_temp"), Some(5.0));
    assert_eq!(harness.get_scalar("climate.air_temp"), Some(50.0));
}

/// Test tick context (tick number advances correctly).
#[test]
fn test_tick_context() {
    let source = r#"
        strata.terra {}

        era.main {
            : initial
            strata {
                terra: active
            }
        }

        signal.terra.dummy {
            : Scalar<unit>
            : strata(terra)
            resolve { 0.0 }
        }
    "#;

    let mut harness = TestHarness::from_source(source);

    assert_eq!(harness.current_tick(), 0);

    for expected in 1..=10 {
        harness.tick();
        assert_eq!(harness.current_tick(), expected);
    }
}
