use super::*;
use crate::ast::{BinaryOp, Expr, Item, Literal};

#[test]
fn test_parse_const_block() {
    let source = r#"
        const {
            physics.stefan_boltzmann: 5.67e-8 <W>
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::ConstBlock(block) => {
            assert_eq!(block.entries.len(), 1);
            assert_eq!(
                block.entries[0].path.node.join("."),
                "physics.stefan_boltzmann"
            );
        }
        _ => panic!("expected ConstBlock"),
    }
}

#[test]
fn test_parse_strata_def() {
    let source = r#"
        strata.terra.thermal {
            : title("Thermal")
            : symbol("Q")
            : stride(5)
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::StrataDef(def) => {
            assert_eq!(def.path.node.join("."), "terra.thermal");
            assert_eq!(def.title.as_ref().unwrap().node, "Thermal");
            assert_eq!(def.symbol.as_ref().unwrap().node, "Q");
            assert_eq!(def.stride.as_ref().unwrap().node, 5);
        }
        _ => panic!("expected StrataDef"),
    }
}

#[test]
fn test_parse_signal_def() {
    let source = r#"
        signal.terra.core.temp {
            : Scalar<K, 100..10000>
            : strata(terra.thermal)

            resolve {
                prev
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert_eq!(def.path.node.join("."), "terra.core.temp");
            assert!(def.ty.is_some());
            assert!(def.strata.is_some());
            assert!(def.resolve.is_some());
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_expression() {
    let source = "signal.terra.temp { resolve { prev + 1.0 } }";
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
}

#[test]
fn test_parse_era_def() {
    let source = r#"
        era.hadean {
            : initial
            : title("Hadean")
            : dt(1 <Myr>)

            strata {
                terra.thermal: active
                terra.tectonics: gated
            }

            transition {
                to: era.archean
                when {
                    signal.time.planet_age > 500
                }
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::EraDef(def) => {
            assert_eq!(def.name.node, "hadean");
            assert!(def.is_initial);
            assert!(!def.is_terminal);
            assert!(def.title.is_some());
            assert!(def.dt.is_some());
            assert_eq!(def.strata_states.len(), 2);
            assert_eq!(def.transitions.len(), 1);
        }
        _ => panic!("expected EraDef"),
    }
}

#[test]
fn test_parse_field_def() {
    let source = r#"
        field.terra.surface.temperature_map {
            : Scalar<K>
            : strata(terra.atmosphere)
            : topology(sphere_surface)
            : title("Surface Temperature")

            measure {
                signal.terra.atmosphere.temp_profile
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::FieldDef(def) => {
            assert_eq!(def.path.node.join("."), "terra.surface.temperature_map");
            assert!(def.ty.is_some());
            assert!(def.strata.is_some());
            assert!(def.topology.is_some());
            assert!(def.measure.is_some());
        }
        _ => panic!("expected FieldDef"),
    }
}

#[test]
fn test_parse_fracture_def() {
    let source = r#"
        fracture.terra.climate.runaway_greenhouse {
            when {
                signal.terra.atmosphere.co2 > 1000
                signal.terra.surface.avg_temp > 350
            }

            emit {
                signal.terra.atmosphere.feedback <- 1.5
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::FractureDef(def) => {
            assert_eq!(def.path.node.join("."), "terra.climate.runaway_greenhouse");
            assert_eq!(def.conditions.len(), 2);
            assert_eq!(def.emit.len(), 1);
        }
        _ => panic!("expected FractureDef"),
    }
}

#[test]
fn test_parse_complex_expression() {
    let source = r#"
        signal.terra.thermal.loss {
            : Scalar<W>
            : strata(terra.thermal)

            resolve {
                const.physics.stefan_boltzmann * prev * prev * prev * prev
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert!(def.resolve.is_some());
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_type_def() {
    let source = r#"
        type.ThermalState {
            temperature: Scalar<K>
            flux: Scalar<W>
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::TypeDef(def) => {
            assert_eq!(def.name.node, "ThermalState");
            assert_eq!(def.fields.len(), 2);
        }
        _ => panic!("expected TypeDef"),
    }
}

#[test]
fn test_parse_operator_def() {
    let source = r#"
        operator.terra.thermal.budget {
            : strata(terra.thermal)
            : phase(collect)

            collect {
                signal.terra.geophysics.mantle.heat_j
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::OperatorDef(def) => {
            assert_eq!(def.path.node.join("."), "terra.thermal.budget");
            assert!(def.strata.is_some());
            assert!(def.phase.is_some());
            assert!(def.body.is_some());
        }
        _ => panic!("expected OperatorDef"),
    }
}

#[test]
fn test_parse_impulse_def() {
    let source = r#"
        impulse.terra.impact.asteroid {
            : ImpactEvent

            apply {
                payload
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::ImpulseDef(def) => {
            assert_eq!(def.path.node.join("."), "terra.impact.asteroid");
            assert!(def.payload_type.is_some());
            assert!(def.apply.is_some());
        }
        _ => panic!("expected ImpulseDef"),
    }
}

#[test]
fn test_parse_unit_qualified_literals() {
    let source = r#"
        era.hadean {
            : initial
            : dt(1 <Myr>)
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::EraDef(def) => {
            assert_eq!(def.name.node, "hadean");
            assert!(def.dt.is_some());
            let dt = def.dt.as_ref().unwrap();
            assert_eq!(dt.node.unit, "Myr");
        }
        _ => panic!("expected EraDef"),
    }
}

#[test]
fn test_parse_comparison_with_unit() {
    let source = r#"
        era.hadean {
            : initial
            : dt(1 <Myr>)

            transition {
                to: era.archean
                when {
                    signal.terra.core.temp < 5000 <K>
                }
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::EraDef(def) => {
            assert_eq!(def.transitions.len(), 1);
            let transition = &def.transitions[0];
            assert_eq!(transition.conditions.len(), 1);
            match &transition.conditions[0].node {
                crate::ast::Expr::Binary { op, right, .. } => {
                    assert_eq!(op, &BinaryOp::Lt);
                    match &right.node {
                        crate::ast::Expr::LiteralWithUnit { value, unit } => {
                            match value {
                                Literal::Float(f) => assert_eq!(*f, 5000.0),
                                _ => panic!("expected float literal"),
                            }
                            assert_eq!(unit, "K");
                        }
                        _ => panic!("expected LiteralWithUnit, got {:?}", right.node),
                    }
                }
                _ => panic!("expected Binary expression"),
            }
        }
        _ => panic!("expected EraDef"),
    }
}

#[test]
fn test_parse_function_call_simple() {
    let source = r#"
        signal.core.temp {
            : Scalar<K>
            : strata(thermal)

            resolve {
                decay(prev, 0.5)
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert!(def.resolve.is_some());
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Call { function, args } => {
                    match &function.node {
                        Expr::Path(p) => assert_eq!(p.join("."), "decay"),
                        _ => panic!("expected Path, got {:?}", function.node),
                    }
                    assert_eq!(args.len(), 2);
                }
                _ => panic!("expected Call, got {:?}", resolve.body.node),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_function_call_nested() {
    let source = r#"
        signal.core.temp {
            : Scalar<K>
            resolve {
                max(min(prev, 1000), 100)
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Call { function, args } => {
                    match &function.node {
                        Expr::Path(p) => assert_eq!(p.join("."), "max"),
                        _ => panic!("expected Path"),
                    }
                    assert_eq!(args.len(), 2);
                    // First arg should be min(prev, 1000)
                    match &args[0].node {
                        Expr::Call { function, args: inner_args } => {
                            match &function.node {
                                Expr::Path(p) => assert_eq!(p.join("."), "min"),
                                _ => panic!("expected Path"),
                            }
                            assert_eq!(inner_args.len(), 2);
                        }
                        _ => panic!("expected Call"),
                    }
                }
                _ => panic!("expected Call"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_function_call_in_expression() {
    let source = r#"
        signal.core.temp {
            : Scalar<K>
            resolve {
                prev * exp(-config.thermal.decay_rate)
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Binary { op, left, right } => {
                    assert_eq!(*op, BinaryOp::Mul);
                    match &left.node {
                        Expr::Prev => {}
                        _ => panic!("expected Prev, got {:?}", left.node),
                    }
                    match &right.node {
                        Expr::Call { function, args } => {
                            match &function.node {
                                Expr::Path(p) => assert_eq!(p.join("."), "exp"),
                                _ => panic!("expected Path"),
                            }
                            assert_eq!(args.len(), 1);
                        }
                        _ => panic!("expected Call, got {:?}", right.node),
                    }
                }
                _ => panic!("expected Binary, got {:?}", resolve.body.node),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_namespaced_function_call() {
    let source = r#"
        signal.core.temp {
            : Scalar<K>
            resolve {
                math.clamp(prev, 100, 10000)
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Call { function, args } => {
                    match &function.node {
                        Expr::Path(p) => assert_eq!(p.join("."), "math.clamp"),
                        _ => panic!("expected Path"),
                    }
                    assert_eq!(args.len(), 3);
                }
                _ => panic!("expected Call"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_let_expression() {
    let source = r#"
        signal.core.temp {
            : Scalar<K>
            resolve {
                let a = 1.0
                let b = 2.0
                a + b
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Let { name, value, body } => {
                    assert_eq!(name, "a");
                    match &value.node {
                        Expr::Literal(Literal::Float(f)) => assert_eq!(*f, 1.0),
                        _ => panic!("expected float literal for value"),
                    }
                    // Body should be another let
                    match &body.node {
                        Expr::Let { name, value, body } => {
                            assert_eq!(name, "b");
                            match &value.node {
                                Expr::Literal(Literal::Float(f)) => assert_eq!(*f, 2.0),
                                _ => panic!("expected float literal for value"),
                            }
                            // Inner body should be a + b
                            match &body.node {
                                Expr::Binary { op, left, right } => {
                                    assert_eq!(*op, BinaryOp::Add);
                                    match &left.node {
                                        Expr::Path(p) => assert_eq!(p.join("."), "a"),
                                        _ => panic!("expected path 'a'"),
                                    }
                                    match &right.node {
                                        Expr::Path(p) => assert_eq!(p.join("."), "b"),
                                        _ => panic!("expected path 'b'"),
                                    }
                                }
                                _ => panic!("expected Binary add, got {:?}", body.node),
                            }
                        }
                        _ => panic!("expected inner Let, got {:?}", body.node),
                    }
                }
                _ => panic!("expected Let, got {:?}", resolve.body.node),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}
