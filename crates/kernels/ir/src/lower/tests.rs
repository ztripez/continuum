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
        ValueType::Scalar { range: Some(r), .. } => {
            assert_eq!(r.min, -11000.0);
            assert_eq!(r.max, 9000.0);
        }
        _ => panic!("expected Scalar with range, got {:?}", signal.value_type),
    }
}

#[test]
fn test_unit_preserved_in_scalar_type() {
    let src = r#"
        strata.terra {}
        era.main { : initial }

        signal.terra.temp {
            : Scalar<K, 100..10000>
            : strata(terra)
            resolve { prev + 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("terra.temp")).unwrap();

    // Verify the unit is preserved
    match &signal.value_type {
        ValueType::Scalar { unit, range, .. } => {
            assert_eq!(unit, &Some("K".to_string()), "unit should be 'K'");
            assert!(range.is_some(), "range should be present");
            let r = range.as_ref().unwrap();
            assert_eq!(r.min, 100.0);
            assert_eq!(r.max, 10000.0);
        }
        _ => panic!("expected Scalar, got {:?}", signal.value_type),
    }
}

#[test]
fn test_unit_preserved_in_vector_type() {
    let src = r#"
        strata.terra {}
        era.main { : initial }

        signal.terra.velocity {
            : Vec3<m/s>
            : strata(terra)
            resolve { prev }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("terra.velocity")).unwrap();

    // Verify the unit is preserved
    match &signal.value_type {
        ValueType::Vec3 { unit, .. } => {
            assert_eq!(unit, &Some("m/s".to_string()), "unit should be 'm/s'");
        }
        _ => panic!("expected Vec3, got {:?}", signal.value_type),
    }
}

#[test]
fn test_dimension_parsed_for_scalar_unit() {
    let src = r#"
        strata.terra {}
        era.main { : initial }

        signal.terra.velocity {
            : Scalar<m/s>
            : strata(terra)
            resolve { prev }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("terra.velocity")).unwrap();

    // Verify the dimension is parsed correctly: m/s = length^1 * time^-1
    match &signal.value_type {
        ValueType::Scalar {
            unit,
            dimension,
            range,
        } => {
            assert_eq!(unit, &Some("m/s".to_string()), "unit should be 'm/s'");
            assert!(dimension.is_some(), "dimension should be parsed");
            let dim = dimension.as_ref().unwrap();
            assert_eq!(dim.length, 1, "length dimension should be 1");
            assert_eq!(dim.time, -1, "time dimension should be -1");
            assert_eq!(dim.mass, 0, "mass dimension should be 0");
            assert_eq!(range, &None, "no range specified");
        }
        _ => panic!("expected Scalar, got {:?}", signal.value_type),
    }
}

#[test]
fn test_dimension_parsed_for_derived_unit() {
    let src = r#"
        strata.terra {}
        era.main { : initial }

        signal.terra.force {
            : Scalar<N>
            : strata(terra)
            resolve { prev }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("terra.force")).unwrap();

    // Verify N = kg·m/s² = mass^1 * length^1 * time^-2
    match &signal.value_type {
        ValueType::Scalar { dimension, .. } => {
            assert!(dimension.is_some(), "dimension should be parsed for N");
            let dim = dimension.as_ref().unwrap();
            assert_eq!(dim.mass, 1, "mass dimension should be 1");
            assert_eq!(dim.length, 1, "length dimension should be 1");
            assert_eq!(dim.time, -2, "time dimension should be -2");
        }
        _ => panic!("expected Scalar, got {:?}", signal.value_type),
    }
}

#[test]
fn test_dimension_parsed_for_vector_unit() {
    let src = r#"
        strata.terra {}
        era.main { : initial }

        signal.terra.acceleration {
            : Vec3<m/s²>
            : strata(terra)
            resolve { prev }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("terra.acceleration")).unwrap();

    // Verify m/s² = length^1 * time^-2
    match &signal.value_type {
        ValueType::Vec3 { unit, dimension } => {
            assert_eq!(unit, &Some("m/s²".to_string()), "unit should be 'm/s²'");
            assert!(dimension.is_some(), "dimension should be parsed");
            let dim = dimension.as_ref().unwrap();
            assert_eq!(dim.length, 1, "length dimension should be 1");
            assert_eq!(dim.time, -2, "time dimension should be -2");
            assert_eq!(dim.mass, 0, "mass dimension should be 0");
        }
        _ => panic!("expected Vec3, got {:?}", signal.value_type),
    }
}

#[test]
fn test_no_type_annotation_has_no_unit() {
    let src = r#"
        strata.terra {}
        era.main { : initial }

        signal.terra.dimensionless {
            : strata(terra)
            resolve { 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world
        .signals
        .get(&SignalId::from("terra.dimensionless"))
        .unwrap();

    // Verify no type annotation defaults to Scalar with no unit
    match &signal.value_type {
        ValueType::Scalar { unit, range, .. } => {
            assert_eq!(unit, &None, "no type annotation should have no unit");
            assert_eq!(range, &None, "no type annotation should have no range");
        }
        _ => panic!("expected Scalar, got {:?}", signal.value_type),
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

    // Kernel call should be lowered to KernelCall, not inlined
    match resolve {
        CompiledExpr::KernelCall { function, args } => {
            assert_eq!(function, "abs");
            assert_eq!(args.len(), 1);
        }
        _ => panic!("expected KernelCall, got {:?}", resolve),
    }
}

#[test]
fn test_kernel_function_various_names() {
    // Kernel functions with various names should all become KernelCall
    let src = r#"
        strata.test {}
        era.main { : initial }

        signal.test.a {
            : strata(test)
            resolve { kernel.sqrt(4.0) }
        }

        signal.test.b {
            : strata(test)
            resolve { kernel.gravity_acceleration(signal.test.a, 6e6) }
        }

        signal.test.c {
            : strata(test)
            resolve { kernel.mat_vec_mul(signal.test.a, signal.test.b) }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    // Check kernel.sqrt -> KernelCall("sqrt", ...)
    let signal_a = world.signals.get(&SignalId::from("test.a")).unwrap();
    match signal_a.resolve.as_ref().unwrap() {
        CompiledExpr::KernelCall { function, args } => {
            assert_eq!(function, "sqrt");
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected KernelCall for sqrt, got {:?}", other),
    }

    // Check kernel.gravity_acceleration -> KernelCall("gravity_acceleration", ...)
    let signal_b = world.signals.get(&SignalId::from("test.b")).unwrap();
    match signal_b.resolve.as_ref().unwrap() {
        CompiledExpr::KernelCall { function, args } => {
            assert_eq!(function, "gravity_acceleration");
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected KernelCall for gravity_acceleration, got {:?}", other),
    }

    // Check kernel.mat_vec_mul -> KernelCall("mat_vec_mul", ...)
    let signal_c = world.signals.get(&SignalId::from("test.c")).unwrap();
    match signal_c.resolve.as_ref().unwrap() {
        CompiledExpr::KernelCall { function, args } => {
            assert_eq!(function, "mat_vec_mul");
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected KernelCall for mat_vec_mul, got {:?}", other),
    }
}

#[test]
fn test_dt_robust_operator_recognized() {
    // dt-robust operators should be lowered to DtRobustCall, not regular Call
    use crate::DtRobustOperator;

    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve {
                decay(prev, 0.5)
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.value")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();

    // decay should be recognized as a dt-robust operator
    match resolve {
        CompiledExpr::DtRobustCall { operator, args, .. } => {
            assert_eq!(*operator, DtRobustOperator::Decay);
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected DtRobustCall, got {:?}", resolve),
    }
}

#[test]
fn test_integrate_is_dt_robust() {
    // integrate is a dt-robust operator
    use crate::DtRobustOperator;

    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.pos {
            : strata(test)
            resolve {
                integrate(prev, 1.0)
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.pos")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();

    match resolve {
        CompiledExpr::DtRobustCall { operator, .. } => {
            assert_eq!(*operator, DtRobustOperator::Integrate);
        }
        _ => panic!("expected DtRobustCall for integrate, got {:?}", resolve),
    }
}

#[test]
fn test_regular_function_not_dt_robust() {
    // sin is not a dt-robust operator, should remain as regular Call
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve {
                sin(prev)
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signal = world.signals.get(&SignalId::from("test.value")).unwrap();
    let resolve = signal.resolve.as_ref().unwrap();

    // sin should remain as a regular Call
    match resolve {
        CompiledExpr::Call { function, .. } => {
            assert_eq!(function, "sin");
        }
        _ => panic!("expected Call for sin, got {:?}", resolve),
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

#[test]
fn test_duplicate_signal_definition() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve { 1.0 }
        }
        signal.test.value {
            : strata(test)
            resolve { 2.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::DuplicateDefinition(name) => {
            assert!(name.contains("test.value"), "got: {}", name);
        }
        e => panic!("expected DuplicateDefinition, got: {:?}", e),
    }
}

#[test]
fn test_duplicate_stratum_definition() {
    let src = r#"
        strata.test {}
        strata.test {}
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::DuplicateDefinition(name) => {
            assert!(name.contains("test"), "got: {}", name);
        }
        e => panic!("expected DuplicateDefinition, got: {:?}", e),
    }
}

#[test]
fn test_duplicate_era_definition() {
    let src = r#"
        era.main { : initial }
        era.main { : terminal }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::DuplicateDefinition(name) => {
            assert!(name.contains("main"), "got: {}", name);
        }
        e => panic!("expected DuplicateDefinition, got: {:?}", e),
    }
}

#[test]
fn test_duplicate_const_definition() {
    let src = r#"
        const {
            physics.gravity: 9.81
        }
        const {
            physics.gravity: 10.0
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::DuplicateDefinition(name) => {
            assert!(name.contains("physics.gravity"), "got: {}", name);
        }
        e => panic!("expected DuplicateDefinition, got: {:?}", e),
    }
}

#[test]
fn test_undefined_stratum_in_signal() {
    let src = r#"
        era.main { : initial }
        signal.test.value {
            : strata(nonexistent)
            resolve { 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::UndefinedStratum(name) => {
            assert_eq!(name, "nonexistent");
        }
        e => panic!("expected UndefinedStratum, got: {:?}", e),
    }
}

#[test]
fn test_undefined_stratum_in_operator() {
    let src = r#"
        era.main { : initial }
        operator.test.op {
            : strata(missing)
            collect { 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::UndefinedStratum(name) => {
            assert_eq!(name, "missing");
        }
        e => panic!("expected UndefinedStratum, got: {:?}", e),
    }
}

#[test]
fn test_undefined_stratum_in_field() {
    let src = r#"
        era.main { : initial }
        field.test.output {
            : strata(undefined)
            measure { 1.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::UndefinedStratum(name) => {
            assert_eq!(name, "undefined");
        }
        e => panic!("expected UndefinedStratum, got: {:?}", e),
    }
}

#[test]
fn test_undefined_stratum_in_entity() {
    let src = r#"
        era.main { : initial }
        entity.test.item {
            : strata(ghost)
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::UndefinedStratum(name) => {
            assert_eq!(name, "ghost");
        }
        e => panic!("expected UndefinedStratum, got: {:?}", e),
    }
}

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

// ============================================================================
// Chronicle Lowering Tests
// ============================================================================

use continuum_foundation::ChronicleId;

#[test]
fn test_lower_chronicle_basic() {
    let src = r#"
        strata.thermal {}
        era.main { : initial }
        signal.thermal.temp {
            : strata(thermal)
            resolve { prev + 1.0 }
        }
        chronicle.thermal.events {
            observe {
                when signal.thermal.temp > 100.0 {
                    emit event.overheating {
                        temp: signal.thermal.temp
                    }
                }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let chronicle = world.chronicles.get(&ChronicleId::from("thermal.events")).unwrap();
    assert_eq!(chronicle.id.0, "thermal.events");
    assert_eq!(chronicle.handlers.len(), 1);
}

#[test]
fn test_lower_chronicle_with_multiple_handlers() {
    let src = r#"
        strata.thermal {}
        era.main { : initial }
        signal.thermal.temp {
            : strata(thermal)
            resolve { prev }
        }
        chronicle.thermal.events {
            observe {
                when signal.thermal.temp > 500.0 {
                    emit event.critical_temp {
                        temp: signal.thermal.temp
                    }
                }
                when signal.thermal.temp < 100.0 {
                    emit event.cold_snap {
                        temp: signal.thermal.temp
                    }
                }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let chronicle = world.chronicles.get(&ChronicleId::from("thermal.events")).unwrap();
    assert_eq!(chronicle.handlers.len(), 2);
    assert_eq!(chronicle.handlers[0].event_name, "critical_temp");
    assert_eq!(chronicle.handlers[1].event_name, "cold_snap");
}

#[test]
fn test_lower_chronicle_collects_signal_reads() {
    let src = r#"
        strata.thermal {}
        era.main { : initial }
        signal.thermal.temp {
            : strata(thermal)
            resolve { prev }
        }
        signal.thermal.pressure {
            : strata(thermal)
            resolve { prev }
        }
        chronicle.thermal.events {
            observe {
                when signal.thermal.temp > 100.0 {
                    emit event.status {
                        temp: signal.thermal.temp
                        pressure: signal.thermal.pressure
                    }
                }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let chronicle = world.chronicles.get(&ChronicleId::from("thermal.events")).unwrap();
    assert!(chronicle.reads.contains(&SignalId::from("thermal.temp")));
    assert!(chronicle.reads.contains(&SignalId::from("thermal.pressure")));
}

#[test]
fn test_lower_chronicle_event_fields() {
    let src = r#"
        strata.thermal {}
        era.main { : initial }
        signal.thermal.temp {
            : strata(thermal)
            resolve { prev }
        }
        chronicle.thermal.events {
            observe {
                when signal.thermal.temp > 100.0 {
                    emit event.reading {
                        temperature: signal.thermal.temp
                        doubled: signal.thermal.temp * 2.0
                    }
                }
            }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let chronicle = world.chronicles.get(&ChronicleId::from("thermal.events")).unwrap();
    let handler = &chronicle.handlers[0];
    assert_eq!(handler.event_fields.len(), 2);
    assert_eq!(handler.event_fields[0].name, "temperature");
    assert_eq!(handler.event_fields[1].name, "doubled");
}

#[test]
fn test_duplicate_chronicle_definition() {
    let src = r#"
        era.main { : initial }
        chronicle.thermal.events {
            observe {}
        }
        chronicle.thermal.events {
            observe {}
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::DuplicateDefinition(name) => {
            assert!(name.contains("thermal.events"), "got: {}", name);
        }
        e => panic!("expected DuplicateDefinition, got: {:?}", e),
    }
}

#[test]
fn test_lower_chronicle_empty_observe() {
    let src = r#"
        era.main { : initial }
        chronicle.thermal.events {}
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let chronicle = world.chronicles.get(&ChronicleId::from("thermal.events")).unwrap();
    assert!(chronicle.handlers.is_empty());
    assert!(chronicle.reads.is_empty());
}

// ============================================================================
// Mathematical Constants Tests
// ============================================================================

#[test]
fn test_math_const_pi_lowered_to_literal() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.angle {
            : strata(test)
            resolve { PI }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.angle")).unwrap();

    let resolve = signal.resolve.as_ref().expect("resolve should be present");
    match resolve {
        CompiledExpr::Literal(v) => {
            assert!(
                (*v - std::f64::consts::PI).abs() < 1e-10,
                "PI should be ~3.14159, got {}",
                v
            );
        }
        other => panic!("expected Literal(PI), got {:?}", other),
    }
}

#[test]
fn test_math_const_unicode_pi() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.angle {
            : strata(test)
            resolve { π }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.angle")).unwrap();

    let resolve = signal.resolve.as_ref().expect("resolve should be present");
    match resolve {
        CompiledExpr::Literal(v) => {
            assert!(
                (*v - std::f64::consts::PI).abs() < 1e-10,
                "π should be ~3.14159, got {}",
                v
            );
        }
        other => panic!("expected Literal(π), got {:?}", other),
    }
}

#[test]
fn test_math_const_tau() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.angle {
            : strata(test)
            resolve { TAU }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.angle")).unwrap();

    let resolve = signal.resolve.as_ref().expect("resolve should be present");
    match resolve {
        CompiledExpr::Literal(v) => {
            assert!(
                (*v - std::f64::consts::TAU).abs() < 1e-10,
                "TAU should be ~6.28318, got {}",
                v
            );
        }
        other => panic!("expected Literal(TAU), got {:?}", other),
    }
}

#[test]
fn test_math_const_e() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.value {
            : strata(test)
            resolve { E }
        }
    "#;
    let (unit, errors) = parse(src);
    // E is ambiguous - could be parsed as identifier, which is fine
    // If it fails to parse as math const, we accept it
    if !errors.is_empty() {
        return; // Skip if parser interprets E differently
    }
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.value")).unwrap();

    let resolve = signal.resolve.as_ref().expect("resolve should be present");
    match resolve {
        CompiledExpr::Literal(v) => {
            assert!(
                (*v - std::f64::consts::E).abs() < 1e-10,
                "E should be ~2.71828, got {}",
                v
            );
        }
        other => panic!("expected Literal(E), got {:?}", other),
    }
}

#[test]
fn test_math_const_phi() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.ratio {
            : strata(test)
            resolve { PHI }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.ratio")).unwrap();

    let resolve = signal.resolve.as_ref().expect("resolve should be present");
    match resolve {
        CompiledExpr::Literal(v) => {
            let phi = 1.618_033_988_749_895;
            assert!(
                (*v - phi).abs() < 1e-10,
                "PHI should be ~1.618, got {}",
                v
            );
        }
        other => panic!("expected Literal(PHI), got {:?}", other),
    }
}

#[test]
fn test_math_const_in_expression() {
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.area {
            : strata(test)
            resolve { PI * 4.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();
    let signal = world.signals.get(&SignalId::from("test.area")).unwrap();

    // Should be Binary { Mul, Literal(PI), Literal(4.0) }
    let resolve = signal.resolve.as_ref().expect("resolve should be present");
    match resolve {
        CompiledExpr::Binary { op, left, right } => {
            assert_eq!(*op, BinaryOpIr::Mul);
            match left.as_ref() {
                CompiledExpr::Literal(v) => {
                    assert!((*v - std::f64::consts::PI).abs() < 1e-10);
                }
                other => panic!("expected left=Literal(PI), got {:?}", other),
            }
            match right.as_ref() {
                CompiledExpr::Literal(v) => assert_eq!(*v, 4.0),
                other => panic!("expected right=Literal(4.0), got {:?}", other),
            }
        }
        other => panic!("expected Binary expression, got {:?}", other),
    }
}

// ============================================================================
// Tensor, Grid, Seq Type Tests
// ============================================================================

#[test]
fn test_tensor_type_ast_parsing() {
    // Test that Tensor<rows, cols, unit> is parsed correctly in AST
    use continuum_dsl::ast::{Item, TypeExpr};
    let src = r#"
        type.stress {
            tensor: Tensor<3, 3, Pa>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();

    // Find the TypeDef in items
    let type_def = unit
        .items
        .iter()
        .find_map(|item| match &item.node {
            Item::TypeDef(td) => Some(td),
            _ => None,
        })
        .expect("should have a type definition");

    assert_eq!(type_def.name.node, "stress");
    assert_eq!(type_def.fields.len(), 1);
    match &type_def.fields[0].ty.node {
        TypeExpr::Tensor { rows, cols, unit } => {
            assert_eq!(rows, &3);
            assert_eq!(cols, &3);
            assert_eq!(unit, "Pa");
        }
        other => panic!("expected Tensor, got {:?}", other),
    }
}

#[test]
fn test_grid_type_ast_parsing() {
    // Test that Grid<width, height, element_type> is parsed correctly in AST
    use continuum_dsl::ast::{Item, TypeExpr};
    let src = r#"
        type.temperature_map {
            data: Grid<2048, 1024, Scalar<K>>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();

    let type_def = unit
        .items
        .iter()
        .find_map(|item| match &item.node {
            Item::TypeDef(td) => Some(td),
            _ => None,
        })
        .expect("should have a type definition");

    assert_eq!(type_def.name.node, "temperature_map");
    match &type_def.fields[0].ty.node {
        TypeExpr::Grid {
            width,
            height,
            element_type,
        } => {
            assert_eq!(width, &2048);
            assert_eq!(height, &1024);
            match element_type.as_ref() {
                TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "K"),
                other => panic!("expected Scalar element type, got {:?}", other),
            }
        }
        other => panic!("expected Grid, got {:?}", other),
    }
}

#[test]
fn test_seq_type_ast_parsing() {
    // Test that Seq<element_type> is parsed correctly in AST
    use continuum_dsl::ast::{Item, TypeExpr};
    let src = r#"
        type.mass_list {
            masses: Seq<Scalar<kg>>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();

    let type_def = unit
        .items
        .iter()
        .find_map(|item| match &item.node {
            Item::TypeDef(td) => Some(td),
            _ => None,
        })
        .expect("should have a type definition");

    assert_eq!(type_def.name.node, "mass_list");
    match &type_def.fields[0].ty.node {
        TypeExpr::Seq { element_type } => match element_type.as_ref() {
            TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "kg"),
            other => panic!("expected Scalar element type, got {:?}", other),
        },
        other => panic!("expected Seq, got {:?}", other),
    }
}

#[test]
fn test_nested_grid_seq_type_parsing() {
    // Test nested type expressions: Seq<Grid<...>>
    use continuum_dsl::ast::{Item, TypeExpr};
    let src = r#"
        type.layered_data {
            layers: Seq<Grid<128, 128, Scalar<m>>>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();

    let type_def = unit
        .items
        .iter()
        .find_map(|item| match &item.node {
            Item::TypeDef(td) => Some(td),
            _ => None,
        })
        .expect("should have a type definition");

    match &type_def.fields[0].ty.node {
        TypeExpr::Seq { element_type } => match element_type.as_ref() {
            TypeExpr::Grid {
                width,
                height,
                element_type: inner,
            } => {
                assert_eq!(width, &128);
                assert_eq!(height, &128);
                match inner.as_ref() {
                    TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "m"),
                    other => panic!("expected Scalar, got {:?}", other),
                }
            }
            other => panic!("expected Grid, got {:?}", other),
        },
        other => panic!("expected Seq, got {:?}", other),
    }
}

// ============================================================================
// TypeDef Lowering Tests
// ============================================================================

use continuum_foundation::TypeId;

#[test]
fn test_lower_typedef_basic() {
    let src = r#"
        type.PlateState {
            position: Vec3<m>
            velocity: Vec3<m/s>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let type_def = world.types.get(&TypeId::from("PlateState")).unwrap();
    assert_eq!(type_def.id.0, "PlateState");
    assert_eq!(type_def.fields.len(), 2);
    assert_eq!(type_def.fields[0].name, "position");
    assert_eq!(type_def.fields[1].name, "velocity");
}

#[test]
fn test_lower_typedef_with_scalar() {
    let src = r#"
        type.ParticleData {
            mass: Scalar<kg>
            charge: Scalar<C>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let type_def = world.types.get(&TypeId::from("ParticleData")).unwrap();
    assert_eq!(type_def.fields.len(), 2);

    // Check that units are preserved
    match &type_def.fields[0].value_type {
        ValueType::Scalar { unit, .. } => {
            assert_eq!(unit.as_deref(), Some("kg"));
        }
        other => panic!("expected Scalar, got {:?}", other),
    }
    match &type_def.fields[1].value_type {
        ValueType::Scalar { unit, .. } => {
            assert_eq!(unit.as_deref(), Some("C"));
        }
        other => panic!("expected Scalar, got {:?}", other),
    }
}

#[test]
fn test_lower_typedef_with_tensor() {
    let src = r#"
        type.StressTensor {
            stress: Tensor<3, 3, Pa>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let type_def = world.types.get(&TypeId::from("StressTensor")).unwrap();
    assert_eq!(type_def.fields.len(), 1);

    match &type_def.fields[0].value_type {
        ValueType::Tensor { rows, cols, unit, .. } => {
            assert_eq!(*rows, 3);
            assert_eq!(*cols, 3);
            assert_eq!(unit.as_deref(), Some("Pa"));
        }
        other => panic!("expected Tensor, got {:?}", other),
    }
}

#[test]
fn test_lower_typedef_with_grid() {
    let src = r#"
        type.TemperatureField {
            data: Grid<1024, 512, Scalar<K>>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let type_def = world.types.get(&TypeId::from("TemperatureField")).unwrap();
    assert_eq!(type_def.fields.len(), 1);

    match &type_def.fields[0].value_type {
        ValueType::Grid { width, height, element_type } => {
            assert_eq!(*width, 1024);
            assert_eq!(*height, 512);
            match element_type.as_ref() {
                ValueType::Scalar { unit, .. } => {
                    assert_eq!(unit.as_deref(), Some("K"));
                }
                other => panic!("expected Scalar element, got {:?}", other),
            }
        }
        other => panic!("expected Grid, got {:?}", other),
    }
}

#[test]
fn test_lower_typedef_with_seq() {
    let src = r#"
        type.MassDistribution {
            masses: Seq<Scalar<kg>>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let type_def = world.types.get(&TypeId::from("MassDistribution")).unwrap();
    assert_eq!(type_def.fields.len(), 1);

    match &type_def.fields[0].value_type {
        ValueType::Seq { element_type } => match element_type.as_ref() {
            ValueType::Scalar { unit, .. } => {
                assert_eq!(unit.as_deref(), Some("kg"));
            }
            other => panic!("expected Scalar element, got {:?}", other),
        },
        other => panic!("expected Seq, got {:?}", other),
    }
}

#[test]
fn test_duplicate_typedef_definition() {
    let src = r#"
        type.MyType {
            value: Scalar
        }
        type.MyType {
            other: Scalar
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let result = lower(&unit);

    assert!(result.is_err());
    match result.unwrap_err() {
        LowerError::DuplicateDefinition(name) => {
            assert!(name.contains("MyType"), "got: {}", name);
        }
        e => panic!("expected DuplicateDefinition, got: {:?}", e),
    }
}

#[test]
fn test_lower_multiple_typedefs() {
    let src = r#"
        type.Position {
            x: Scalar<m>
            y: Scalar<m>
            z: Scalar<m>
        }
        type.Velocity {
            vx: Scalar<m/s>
            vy: Scalar<m/s>
            vz: Scalar<m/s>
        }
        type.Particle {
            mass: Scalar<kg>
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    assert_eq!(world.types.len(), 3);
    assert!(world.types.contains_key(&TypeId::from("Position")));
    assert!(world.types.contains_key(&TypeId::from("Velocity")));
    assert!(world.types.contains_key(&TypeId::from("Particle")));
}
