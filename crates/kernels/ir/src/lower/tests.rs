//! Tests for the lowering phase.

use continuum_dsl::parse;
use continuum_foundation::{FnId, SignalId, StratumId};

use crate::{lower, BinaryOpIr, CompiledExpr, LowerError, ValueType};

#[test]
fn test_lower_empty() {
    use continuum_dsl::ast::CompilationUnit;
    let unit = CompilationUnit::default();
    let world = lower(&unit).unwrap();
    assert!(world.signals.is_empty());
    assert!(world.strata.is_empty());
}

#[test]
fn test_lower_const() {
    let src = r#"
        const {
            physics.gravity: 9.81
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    assert_eq!(world.constants.get("physics.gravity"), Some(&9.81));
}

#[test]
fn test_lower_strata() {
    let src = r#"
        strata.terra {
            : title("Terra")
            : stride(10)
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let terra = world.strata.get(&StratumId::from("terra")).unwrap();
    assert_eq!(terra.title, Some("Terra".to_string()));
    assert_eq!(terra.default_stride, 10);
}

#[test]
fn test_negative_range_in_type() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.elevation {
            : Scalar<m, -11000..9000>
            : strata(test)
            resolve { prev }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.elevation")).unwrap();

    // Verify the negative range is captured
    match &signal.value_type {
        ValueType::Scalar { range: Some(r) } => {
            assert_eq!(r.min, -11000.0);
            assert_eq!(r.max, 9000.0);
        }
        _ => panic!("expected Scalar with range, got {:?}", signal.value_type),
    }
}

#[test]
fn test_cross_strata_signal_dependency() {
    let src = r#"
        strata.alpha {}
        strata.beta {}
        era.main { : initial }

        signal.alpha.source {
            : Scalar<K>
            : strata(alpha)
            resolve { 100.0 }
        }

        signal.beta.consumer {
            : Scalar<K>
            : strata(beta)
            resolve { signal.alpha.source * 2.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    // Consumer signal should have alpha.source in its reads
    let consumer = world.signals.get(&SignalId::from("beta.consumer")).unwrap();
    assert!(
        consumer.reads.contains(&SignalId::from("alpha.source")),
        "reads: {:?}",
        consumer.reads
    );
}

#[test]
fn test_lower_let_expression() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.sum {
            : strata(test)
            resolve {
                let a = 10.0 in
                let b = 20.0 in
                a + b
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.sum")).unwrap();
    assert!(signal.resolve.is_some());

    // Check the structure of the lowered expression
    let resolve = signal.resolve.as_ref().unwrap();
    match resolve {
        CompiledExpr::Let { name, value, body } => {
            assert_eq!(name, "a");
            assert!(matches!(value.as_ref(), CompiledExpr::Literal(10.0)));
            match body.as_ref() {
                CompiledExpr::Let { name, value, body } => {
                    assert_eq!(name, "b");
                    assert!(matches!(value.as_ref(), CompiledExpr::Literal(20.0)));
                    match body.as_ref() {
                        CompiledExpr::Binary { op, left, right } => {
                            assert!(matches!(op, BinaryOpIr::Add));
                            assert!(matches!(left.as_ref(), CompiledExpr::Local(n) if n == "a"));
                            assert!(matches!(right.as_ref(), CompiledExpr::Local(n) if n == "b"));
                        }
                        _ => panic!("expected Binary, got {:?}", body),
                    }
                }
                _ => panic!("expected inner Let, got {:?}", body),
            }
        }
        _ => panic!("expected Let, got {:?}", resolve),
    }
}

#[test]
fn test_lower_fn_def() {
    let src = r#"
        fn.math.add(a, b) {
            a + b
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let fn_id = FnId::from("math.add");
    let func = world.functions.get(&fn_id).expect("function not found");
    assert_eq!(func.params, vec!["a", "b"]);

    // Check the body is a + b
    match &func.body {
        CompiledExpr::Binary { op, left, right } => {
            assert!(matches!(op, BinaryOpIr::Add));
            assert!(matches!(left.as_ref(), CompiledExpr::Local(n) if n == "a"));
            assert!(matches!(right.as_ref(), CompiledExpr::Local(n) if n == "b"));
        }
        _ => panic!("expected Binary, got {:?}", func.body),
    }
}

#[test]
fn test_lower_fn_with_const_config() {
    let src = r#"
        const {
            physics.gravity: 9.81
        }
        config {
            factor: 2.0
        }
        fn.physics.scaled_gravity() {
            const.physics.gravity * config.factor
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let fn_id = FnId::from("physics.scaled_gravity");
    let func = world.functions.get(&fn_id).expect("function not found");
    assert!(func.params.is_empty());

    // Check the body references const and config
    match &func.body {
        CompiledExpr::Binary { op, left, right } => {
            assert!(matches!(op, BinaryOpIr::Mul));
            assert!(matches!(left.as_ref(), CompiledExpr::Const(s) if s == "physics.gravity"));
            assert!(matches!(right.as_ref(), CompiledExpr::Config(s) if s == "factor"));
        }
        _ => panic!("expected Binary, got {:?}", func.body),
    }
}

#[test]
fn test_fn_inlining_in_signal() {
    let src = r#"
        fn.math.add(a, b) {
            a + b
        }
        strata.test {}
        era.main { : initial }
        signal.test.result {
            : strata(test)
            resolve {
                math.add(10.0, 20.0)
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.result")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();

    // The function call should be inlined as:
    // let a = 10.0 in let b = 20.0 in a + b
    match resolve {
        CompiledExpr::Let { name, value, body } => {
            assert_eq!(name, "a");
            assert!(matches!(value.as_ref(), CompiledExpr::Literal(10.0)));
            match body.as_ref() {
                CompiledExpr::Let { name, value, body } => {
                    assert_eq!(name, "b");
                    assert!(matches!(value.as_ref(), CompiledExpr::Literal(20.0)));
                    match body.as_ref() {
                        CompiledExpr::Binary { op, left, right } => {
                            assert!(matches!(op, BinaryOpIr::Add));
                            assert!(matches!(left.as_ref(), CompiledExpr::Local(n) if n == "a"));
                            assert!(matches!(right.as_ref(), CompiledExpr::Local(n) if n == "b"));
                        }
                        _ => panic!("expected Binary, got {:?}", body),
                    }
                }
                _ => panic!("expected inner Let, got {:?}", body),
            }
        }
        _ => panic!("expected Let, got {:?}", resolve),
    }
}

#[test]
fn test_fn_calling_kernel() {
    // Kernel functions should NOT be inlined, they should remain as Call
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.result {
            : strata(test)
            resolve {
                kernel.abs(-5.0)
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.result")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();

    // Kernel call should remain as a Call, not inlined
    match resolve {
        CompiledExpr::Call { function, args } => {
            assert_eq!(function, "kernel.abs");
            assert_eq!(args.len(), 1);
        }
        _ => panic!("expected Call, got {:?}", resolve),
    }
}

#[test]
fn test_signal_local_config() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.core.temp {
            : strata(test)
            config {
                initial_temp: 5500.0
                decay_rate: 0.01
            }
            resolve {
                config.core.temp.initial_temp
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    // Local config should be added to global config with signal path prefix
    assert_eq!(
        world.config.get("core.temp.initial_temp"),
        Some(&5500.0),
        "config: {:?}",
        world.config
    );
    assert_eq!(
        world.config.get("core.temp.decay_rate"),
        Some(&0.01),
        "config: {:?}",
        world.config
    );

    // Check the resolve expression references the config correctly
    let signal = world.signals.get(&SignalId::from("core.temp")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();
    match resolve {
        CompiledExpr::Config(key) => {
            assert_eq!(key, "core.temp.initial_temp");
        }
        _ => panic!("expected Config, got {:?}", resolve),
    }
}

#[test]
fn test_signal_local_const() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.physics.gravity {
            : strata(test)
            const {
                G: 6.674e-11
                mass: 5.972e24
            }
            resolve {
                const.physics.gravity.G * const.physics.gravity.mass
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    // Local const should be added to global constants with signal path prefix
    assert_eq!(
        world.constants.get("physics.gravity.G"),
        Some(&6.674e-11),
        "constants: {:?}",
        world.constants
    );
    assert_eq!(
        world.constants.get("physics.gravity.mass"),
        Some(&5.972e24),
        "constants: {:?}",
        world.constants
    );
}

#[test]
fn test_signal_local_config_with_global_config() {
    // Test that local config and global config can coexist
    let src = r#"
        config {
            global.factor: 2.0
        }
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            config {
                local_scale: 10.0
            }
            resolve {
                config.global.factor * config.test.value.local_scale
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    // Both global and local config should be present
    assert_eq!(world.config.get("global.factor"), Some(&2.0));
    assert_eq!(world.config.get("test.value.local_scale"), Some(&10.0));
}

#[test]
fn test_dt_raw_without_declaration_fails() {
    // Using dt_raw without `: uses(dt_raw)` should fail
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve {
                prev + dt_raw * 0.5
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::UndeclaredDtRawUsage(signal) => {
            assert_eq!(signal, "test.value");
        }
        e => panic!("expected UndeclaredDtRawUsage, got: {:?}", e),
    }
}

#[test]
fn test_dt_raw_with_declaration_succeeds() {
    // Using dt_raw WITH `: uses(dt_raw)` should work
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            : uses(dt_raw)
            resolve {
                prev + dt_raw * 0.5
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.value")).unwrap();
    assert!(signal.uses_dt_raw);
}

#[test]
fn test_dt_raw_in_nested_expr_without_declaration_fails() {
    // dt_raw buried in nested expression should still be detected
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve {
                let factor = dt_raw * 2.0 in
                prev + factor
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LowerError::UndeclaredDtRawUsage(_)));
}

// ============================================================================
// Error Variant Tests
// ============================================================================

// NOTE: LowerError::UndefinedStratum and LowerError::DuplicateDefinition
// are defined but never raised by the current implementation.
// See: https://github.com/ztripez/continuum/issues/45 (dead code)

#[test]
fn test_lower_error_display() {
    // Test Display impl for all error variants
    let err = LowerError::UndefinedStratum("terra".to_string());
    assert!(err.to_string().contains("undefined stratum"));
    assert!(err.to_string().contains("terra"));

    let err = LowerError::UndefinedSignal("test.value".to_string());
    assert!(err.to_string().contains("undefined signal"));
    assert!(err.to_string().contains("test.value"));

    let err = LowerError::DuplicateDefinition("signal.test".to_string());
    assert!(err.to_string().contains("duplicate definition"));

    let err = LowerError::MissingRequiredField("stratum".to_string());
    assert!(err.to_string().contains("missing required field"));

    let err = LowerError::InvalidExpression("bad expr".to_string());
    assert!(err.to_string().contains("invalid expression"));

    let err = LowerError::UndeclaredDtRawUsage("test.value".to_string());
    assert!(err.to_string().contains("dt_raw"));
    assert!(err.to_string().contains("test.value"));
}

// ============================================================================
// Operator Lowering Tests
// ============================================================================

#[test]
fn test_lower_operator_collect_phase() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve { 0.0 }
        }
        operator.test.collector {
            : strata(test)
            collect {
                let x = signal.test.value in
                x
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    assert!(!world.operators.is_empty(), "operators should not be empty");
}

#[test]
fn test_lower_operator_measure_phase() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve { 0.0 }
        }
        operator.test.measurer {
            : strata(test)
            measure {
                signal.test.value * 2.0
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    assert!(!world.operators.is_empty());
}

// ============================================================================
// Expression Lowering Tests
// ============================================================================

#[test]
fn test_lower_unary_negation() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.neg {
            : strata(test)
            resolve { -5.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.neg")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();
    match resolve {
        CompiledExpr::Unary { .. } | CompiledExpr::Literal(_) => {}
        _ => panic!("expected Unary or Literal, got {:?}", resolve),
    }
}

#[test]
fn test_lower_comparison_ops() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.compare {
            : strata(test)
            resolve {
                if 5.0 > 3.0 { 1.0 } else { 0.0 }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.compare")).unwrap();
    assert!(signal.resolve.is_some());
}

#[test]
fn test_lower_nested_if() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.nested {
            : strata(test)
            resolve {
                if 1.0 > 0.0 {
                    if 2.0 > 1.0 { 100.0 } else { 50.0 }
                } else {
                    0.0
                }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.nested")).unwrap();
    assert!(signal.resolve.is_some());
}

#[test]
fn test_lower_all_binary_ops() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.ops {
            : strata(test)
            resolve {
                let a = 10.0 in
                let b = 3.0 in
                let add = a + b in
                let sub = a - b in
                let mul = a * b in
                let div = a / b in
                div
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.ops")).unwrap();
    assert!(signal.resolve.is_some());
}

#[test]
fn test_lower_prev_reference() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.accumulator {
            : strata(test)
            resolve { prev + 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.accumulator")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();
    match resolve {
        CompiledExpr::Binary { left, .. } => {
            assert!(matches!(left.as_ref(), CompiledExpr::Prev));
        }
        _ => panic!("expected Binary with Prev, got {:?}", resolve),
    }
}

// ============================================================================
// Complex Signal Dependencies
// ============================================================================

#[test]
fn test_multi_signal_dependency_chain() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.a {
            : strata(test)
            resolve { 1.0 }
        }
        signal.test.b {
            : strata(test)
            resolve { signal.test.a * 2.0 }
        }
        signal.test.c {
            : strata(test)
            resolve { signal.test.a + signal.test.b }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let c = world.signals.get(&SignalId::from("test.c")).unwrap();
    assert!(c.reads.contains(&SignalId::from("test.a")));
    assert!(c.reads.contains(&SignalId::from("test.b")));
}

#[test]
fn test_signal_self_reference_via_prev() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.counter {
            : strata(test)
            resolve { prev + 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.counter")).unwrap();
    // prev doesn't create a read dependency on other signals
    assert!(signal.reads.is_empty() || !signal.reads.contains(&SignalId::from("test.counter")));
}

// ============================================================================
// Entity Lowering Tests
// ============================================================================

use continuum_foundation::EntityId;

#[test]
fn test_lower_entity_basic() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert_eq!(entity.stratum, StratumId::from("stellar"));
}

#[test]
fn test_lower_entity_with_schema() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)
            schema {
                mass: Scalar<kg>
                position: Vec3<m>
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert_eq!(entity.schema.len(), 2);
    assert_eq!(entity.schema[0].name, "mass");
    assert_eq!(entity.schema[1].name, "position");
}

#[test]
fn test_lower_entity_with_count_source() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)
            : count(config.stellar.moon_count)
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert_eq!(entity.count_source, Some("stellar.moon_count".to_string()));
}

#[test]
fn test_lower_entity_with_count_bounds() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)
            : count(1..20)
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert_eq!(entity.count_bounds, Some((1, 20)));
}

#[test]
fn test_lower_entity_with_resolve() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)
            resolve {
                self.position + self.velocity
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert!(entity.resolve.is_some());
}

#[test]
fn test_lower_entity_with_field() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)
            field.energy {
                : Scalar<J>
                measure { self.mass * 1000.0 }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert_eq!(entity.fields.len(), 1);
    assert_eq!(entity.fields[0].name, "energy");
    assert!(entity.fields[0].measure.is_some());
}

#[test]
fn test_lower_entity_signal_reads() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        signal.stellar.gravity {
            : strata(stellar)
            resolve { 9.81 }
        }
        entity.stellar.moon {
            : strata(stellar)
            resolve {
                signal.stellar.gravity * self.mass
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert!(
        entity.reads.contains(&SignalId::from("stellar.gravity")),
        "entity reads: {:?}",
        entity.reads
    );
}

#[test]
fn test_lower_entity_default_stratum() {
    // Entity without explicit stratum should default to "default"
    let src = r#"
        era.main { : initial }
        entity.test.item {}
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("test.item")).unwrap();
    assert_eq!(entity.stratum, StratumId::from("default"));
}

#[test]
fn test_lower_entity_multiple_fields() {
    let src = r#"
        strata.stellar {}
        era.main { : initial }
        entity.stellar.moon {
            : strata(stellar)

            field.position {
                measure { self.position }
            }

            field.velocity {
                measure { self.velocity }
            }

            field.energy {
                measure { self.mass * 100.0 }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let entity = world.entities.get(&EntityId::from("stellar.moon")).unwrap();
    assert_eq!(entity.fields.len(), 3);
    assert_eq!(entity.fields[0].name, "position");
    assert_eq!(entity.fields[1].name, "velocity");
    assert_eq!(entity.fields[2].name, "energy");
}
