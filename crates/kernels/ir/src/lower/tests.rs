//! Tests for the lowering phase.

use continuum_dsl::parse;
use continuum_foundation::{FnId, SignalId, StratumId};

use crate::{BinaryOpIr, CompiledExpr, LowerError, ValueType, lower};

#[test]
fn test_lower_empty() {
    use continuum_dsl::ast::CompilationUnit;
    let unit = CompilationUnit::default();
    let world = lower(&unit).unwrap();
    assert!(world.signals().is_empty());
    assert!(world.strata().is_empty());
}

#[test]
fn test_vec3_signal_reference_expanded() {
    // For Vec3 signals, referencing another Vec3 signal should expand to component access
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.velocity {
            : Vec3<m/s>
            : strata(test)
            resolve { prev }
        }
        signal.test.position {
            : Vec3<m>
            : strata(test)
            resolve { prev + signal.test.velocity }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signals = world.signals();
    let signal = signals.get(&SignalId::from("test.position")).unwrap();
    let components = signal.resolve_components.as_ref().unwrap();

    // Check that signal reference is expanded to component access
    for (i, comp) in components.iter().enumerate() {
        match comp {
            CompiledExpr::Binary { right, .. } => {
                // Right side should be signal.test.velocity.x/y/z
                match right.as_ref() {
                    CompiledExpr::FieldAccess { object, field } => {
                        match object.as_ref() {
                            CompiledExpr::Signal(id) => {
                                assert_eq!(id.to_string(), "test.velocity");
                            }
                            other => panic!("object should be Signal, got {:?}", other),
                        }
                        let expected = ["x", "y", "z"][i];
                        assert_eq!(field, expected);
                    }
                    other => panic!("right should be FieldAccess(Signal), got {:?}", other),
                }
            }
            other => panic!("component should be Binary, got {:?}", other),
        }
    }
}

#[test]
fn test_vec3_binary_ops_expanded_componentwise() {
    // Binary operations on Vec3 should be expanded per-component
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.velocity {
            : Vec3<m/s>
            : strata(test)
            resolve { prev * 2.0 }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signals = world.signals();
    let signal = signals.get(&SignalId::from("test.velocity")).unwrap();
    let components = signal.resolve_components.as_ref().unwrap();

    for (i, comp) in components.iter().enumerate() {
        match comp {
            CompiledExpr::Binary { op, left, right } => {
                assert!(matches!(op, BinaryOpIr::Mul));

                // Left should be prev.x/y/z
                match left.as_ref() {
                    CompiledExpr::FieldAccess { object, field } => {
                        assert!(matches!(object.as_ref(), CompiledExpr::Prev));
                        let expected = ["x", "y", "z"][i];
                        assert_eq!(field, expected);
                    }
                    other => panic!("left should be FieldAccess, got {:?}", other),
                }

                // Right should be literal 2.0 (scalars are broadcast)
                match right.as_ref() {
                    CompiledExpr::Literal(v, _) => assert_eq!(*v, 2.0),
                    other => panic!("right should be Literal(2.0), got {:?}", other),
                }
            }
            other => panic!("component should be Binary, got {:?}", other),
        }
    }
}

#[test]
fn test_vec3_constructor_expanded() {
    // vec3(x, y, z) in resolve should extract the correct component
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.velocity {
            : Vec3<m/s>
            : strata(test)
            resolve { vec3(1.0, 2.0, 3.0) }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signals = world.signals();
    let signal = signals.get(&SignalId::from("test.velocity")).unwrap();
    let components = signal.resolve_components.as_ref().unwrap();

    // Component 0 (x) should extract arg 0 = 1.0
    match &components[0] {
        CompiledExpr::Literal(v, _) => assert_eq!(*v, 1.0),
        other => panic!("x component should be Literal(1.0), got {:?}", other),
    }

    // Component 1 (y) should extract arg 1 = 2.0
    match &components[1] {
        CompiledExpr::Literal(v, _) => assert_eq!(*v, 2.0),
        other => panic!("y component should be Literal(2.0), got {:?}", other),
    }

    // Component 2 (z) should extract arg 2 = 3.0
    match &components[2] {
        CompiledExpr::Literal(v, _) => assert_eq!(*v, 3.0),
        other => panic!("z component should be Literal(3.0), got {:?}", other),
    }
}

#[test]
fn test_vec3_explicit_component_access_preserved() {
    // Explicit component access like prev.x should be preserved, not double-expanded
    let src = r#"
        strata.test {}
        era.main { : initial }
        signal.test.velocity {
            : Vec3<m/s>
            : strata(test)
            resolve { vec3(prev.x, prev.y * 2.0, prev.z) }
        }
    "#;
    let (unit, errors) = parse(src);
    assert!(errors.is_empty(), "parse errors: {:?}", errors);
    let unit = unit.unwrap();
    let world = lower(&unit).unwrap();

    let signals = world.signals();
    let signal = signals.get(&SignalId::from("test.velocity")).unwrap();
    let components = signal.resolve_components.as_ref().unwrap();

    // Component 0 should be prev.x (preserved)
    match &components[0] {
        CompiledExpr::FieldAccess { object, field } => {
            assert!(matches!(object.as_ref(), CompiledExpr::Prev));
            assert_eq!(field, "x");
        }
        other => panic!(
            "x component should be FieldAccess(Prev, x), got {:?}",
            other
        ),
    }

    // Component 1 should be prev.y * 2.0
    match &components[1] {
        CompiledExpr::Binary { op, left, .. } => {
            assert!(matches!(op, BinaryOpIr::Mul));
            match left.as_ref() {
                CompiledExpr::FieldAccess { object, field } => {
                    assert!(matches!(object.as_ref(), CompiledExpr::Prev));
                    assert_eq!(field, "y");
                }
                other => panic!("left should be FieldAccess(Prev, y), got {:?}", other),
            }
        }
        other => panic!("y component should be Binary, got {:?}", other),
    }

    // Component 2 should be prev.z (preserved)
    match &components[2] {
        CompiledExpr::FieldAccess { object, field } => {
            assert!(matches!(object.as_ref(), CompiledExpr::Prev));
            assert_eq!(field, "z");
        }
        other => panic!(
            "z component should be FieldAccess(Prev, z), got {:?}",
            other
        ),
    }
}
