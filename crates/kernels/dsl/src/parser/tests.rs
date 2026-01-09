use super::*;
use crate::ast::{BinaryOp, Expr, Item, Literal, TypeExpr, UnaryOp};

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
                let a = 1.0 in
                let b = 2.0 in
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

#[test]
fn test_parse_fn_def() {
    let source = r#"
        fn.physics.stefan_boltzmann_loss(temp: Scalar<K>) -> Scalar<K> {
            temp * 4.0
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::FnDef(def) => {
            assert_eq!(def.path.node.join("."), "physics.stefan_boltzmann_loss");
            assert_eq!(def.params.len(), 1);
            assert_eq!(def.params[0].name.node, "temp");
            assert!(def.params[0].ty.is_some());
            assert!(def.return_type.is_some());
            // Body should be temp * 4.0
            match &def.body.node {
                Expr::Binary { op, .. } => {
                    assert_eq!(*op, BinaryOp::Mul);
                }
                _ => panic!("expected Binary mul, got {:?}", def.body.node),
            }
        }
        _ => panic!("expected FnDef"),
    }
}

#[test]
fn test_parse_fn_def_no_return_type() {
    let source = r#"
        fn.math.add(a, b) {
            a + b
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::FnDef(def) => {
            assert_eq!(def.path.node.join("."), "math.add");
            assert_eq!(def.params.len(), 2);
            assert_eq!(def.params[0].name.node, "a");
            assert!(def.params[0].ty.is_none());
            assert_eq!(def.params[1].name.node, "b");
            assert!(def.params[1].ty.is_none());
            assert!(def.return_type.is_none());
        }
        _ => panic!("expected FnDef"),
    }
}

#[test]
fn test_parse_fn_def_with_const_config() {
    let source = r#"
        fn.isostasy.factor() {
            1.0 - config.isostasy.crustal_density / config.isostasy.mantle_density
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::FnDef(def) => {
            assert_eq!(def.path.node.join("."), "isostasy.factor");
            assert_eq!(def.params.len(), 0);
        }
        _ => panic!("expected FnDef"),
    }
}

#[test]
fn test_parse_unicode_unit_superscripts() {
    // Test various Unicode superscript units
    let source = r#"
        signal.test.density {
            : Scalar<kg/m³>
            resolve { 2700.0 }
        }
        signal.test.flux {
            : Scalar<W/m²>
            resolve { 100.0 }
        }
        signal.test.accel {
            : Scalar<m/s²>
            resolve { 9.81 }
        }
        signal.test.stefan {
            : Scalar<W/m²/K⁴>
            resolve { 5.67e-8 }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 4);

    // Check each signal has the correct unit
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "kg/m³"),
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }

    match &unit.items[1].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "W/m²"),
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }

    match &unit.items[2].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "m/s²"),
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }

    match &unit.items[3].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, .. } => assert_eq!(unit, "W/m²/K⁴"),
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_unicode_unit_with_range() {
    // Test Unicode units combined with ranges
    let source = r#"
        signal.test.density {
            : Scalar<kg/m³, 1000..10000>
            resolve { 2700.0 }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();

    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, range } => {
                    assert_eq!(unit, "kg/m³");
                    let r = range.as_ref().unwrap();
                    assert_eq!(r.min, 1000.0);
                    assert_eq!(r.max, 10000.0);
                }
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_unit_with_multiplication() {
    // Test units with multiplication like Pa*s (Pascal-seconds for viscosity)
    let source = r#"
        signal.test.viscosity {
            : Scalar<Pa*s, 0..1e24>
            resolve { 1e21 }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();

    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, range } => {
                    assert_eq!(unit, "Pa*s");
                    let r = range.as_ref().unwrap();
                    assert_eq!(r.min, 0.0);
                    assert_eq!(r.max, 1e24);
                }
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_unit_with_multiple_slashes() {
    // Test compound units with multiple slashes like kg/m²/yr
    let source = r#"
        signal.test.weathering {
            : Scalar<kg/m²/yr, 0..1>
            resolve { 0.01 }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();

    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, range } => {
                    assert_eq!(unit, "kg/m²/yr");
                    let r = range.as_ref().unwrap();
                    assert_eq!(r.min, 0.0);
                    assert_eq!(r.max, 1.0);
                }
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_geophysics_viscosity_signal() {
    // Full signal test matching geophysics.cdsl signal.mantle.viscosity
    let source = r#"
signal.mantle.viscosity {
    : Scalar<Pa*s, 0..1e24>
    : strata(tectonics)
    : title("Mantle Viscosity")
    : symbol("eta")

    resolve {
        prev + collected
    }

    assert {
        prev >= 0.0 : warn, "Viscosity cannot be negative"
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            match &ty.node {
                TypeExpr::Scalar { unit, range } => {
                    assert_eq!(unit, "Pa*s");
                    let r = range.as_ref().unwrap();
                    assert_eq!(r.min, 0.0);
                    assert_eq!(r.max, 1e24);
                }
                _ => panic!("expected Scalar"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_complete_geophysics_file() {
    // Test a simplified version of the geophysics structure
    let source = r#"
fn.isostasy.buoyancy_factor(crustal_density, mantle_density) {
    1.0 - crustal_density / mantle_density
}

const {
    physics.gravitational: 6.67430e-11
    earth.mass: 5.972e24
}

config {
    planet.mass: 5.972e24
    planet.radius: 6.371e6
}

signal.mantle.viscosity {
    : Scalar<Pa*s, 0..1e24>
    : strata(tectonics)

    resolve {
        prev + collected
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 4);
}

#[test]
fn test_parse_config_with_unit() {
    // Test config entry with unit (this is supported in const but maybe not config)
    let source = r#"
config {
    thermal.decay_halflife: 4.5e17 <s>
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
}

#[test]
fn test_parse_vec2_and_mod_calls() {
    // Test vec2 constructor and mod function from geophysics
    let source = r#"
signal.rotation.state {
    : Vec2<rad>
    : strata(rotation)
    : uses(dt_raw)

    resolve {
        let phase = prev.x + prev.y * dt_raw in
        let omega = prev.y + collected in
        vec2(mod(phase, 6.283185307), omega)
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
}

#[test]
fn test_parse_if_else_expression() {
    // Test if-else expression parsing
    let source = r#"
signal.test.conditional {
    : Scalar<1>
    resolve {
        if prev > 0.0 {
            prev * 2.0
        } else {
            0.0
        }
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
                Expr::If { condition, then_branch, else_branch } => {
                    // Verify condition is prev > 0.0
                    match &condition.node {
                        Expr::Binary { op, .. } => {
                            assert_eq!(*op, BinaryOp::Gt);
                        }
                        _ => panic!("expected Binary comparison"),
                    }
                    // Verify then branch
                    match &then_branch.node {
                        Expr::Binary { op, .. } => {
                            assert_eq!(*op, BinaryOp::Mul);
                        }
                        _ => panic!("expected Binary mul in then branch"),
                    }
                    // Verify else branch exists
                    assert!(else_branch.is_some());
                }
                _ => panic!("expected If expression, got {:?}", resolve.body.node),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_nested_if_else() {
    // Test nested if-else from geophysics
    let source = r#"
signal.test.nested_if {
    : Scalar<1>
    resolve {
        let raw_shear = collected in
        if raw_shear > 0.0 {
            raw_shear / 100.0
        } else {
            0.0
        }
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
}

#[test]
fn test_comparison_chaining_disallowed() {
    // Comparison operators should NOT chain: a < b < c is disallowed
    // This prevents confusing behavior where a < b < c would parse as (a < b) < c
    let source = r#"
signal.test.chained {
    : Scalar<1>
    resolve {
        if 1 < 2 < 3 {
            1.0
        } else {
            0.0
        }
    }
}
    "#;
    let (result, errors) = parse(source);
    // Should produce parse errors because chaining is not allowed
    assert!(
        !errors.is_empty() || result.is_none(),
        "comparison chaining should not be allowed, but parsing succeeded"
    );
}

#[test]
fn test_single_comparison_allowed() {
    // Single comparisons should still work
    let source = r#"
signal.test.single_compare {
    : Scalar<1>
    resolve {
        if prev > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    assert!(result.is_some());
}

#[test]
fn test_parse_logical_not_operator() {
    // Test logical not (!) operator
    let source = r#"
signal.test.not_op {
    : Scalar<1>
    resolve {
        if !condition {
            1.0
        } else {
            0.0
        }
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
                Expr::If { condition, .. } => {
                    // Verify condition is a unary Not expression
                    match &condition.node {
                        Expr::Unary { op, .. } => {
                            assert_eq!(*op, UnaryOp::Not);
                        }
                        _ => panic!("expected Unary, got {:?}", condition.node),
                    }
                }
                _ => panic!("expected If expression"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_double_negation() {
    // Test double negation (--)
    let source = r#"
signal.test.double_neg {
    : Scalar<1>
    resolve {
        --prev
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
                Expr::Unary { op: outer_op, operand } => {
                    assert_eq!(*outer_op, UnaryOp::Neg);
                    match &operand.node {
                        Expr::Unary { op: inner_op, .. } => {
                            assert_eq!(*inner_op, UnaryOp::Neg);
                        }
                        _ => panic!("expected inner Unary"),
                    }
                }
                _ => panic!("expected outer Unary"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

// ============================================================================
// Entity Parsing Tests
// ============================================================================

#[test]
fn test_parse_entity_basic() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.path.node.join("."), "stellar.moon");
            assert!(def.strata.is_some());
            assert_eq!(def.strata.as_ref().unwrap().node.join("."), "stellar");
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_with_schema() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)
    schema {
        mass: Scalar<kg>
        position: Vec3<m>
        velocity: Vec3<m/s>
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.schema.len(), 3);
            assert_eq!(def.schema[0].name.node, "mass");
            assert_eq!(def.schema[1].name.node, "position");
            assert_eq!(def.schema[2].name.node, "velocity");
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_with_count_source() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)
    : count(config.stellar.moon_count)
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert!(def.count_source.is_some());
            assert_eq!(
                def.count_source.as_ref().unwrap().node.join("."),
                "stellar.moon_count"
            );
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_with_count_bounds() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)
    : count(1..20)
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert!(def.count_bounds.is_some());
            let bounds = def.count_bounds.as_ref().unwrap();
            assert_eq!(bounds.min, 1);
            assert_eq!(bounds.max, 20);
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_with_resolve() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)
    resolve {
        self.position + self.velocity * dt
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert!(def.resolve.is_some());
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_with_field() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)
    field.orbital_energy {
        : Scalar<J>
        : topology(point_cloud)
        measure {
            0.5 * self.mass * kernel.dot(self.velocity, self.velocity)
        }
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.fields.len(), 1);
            assert_eq!(def.fields[0].name.node, "orbital_energy");
            assert!(def.fields[0].ty.is_some());
            assert!(def.fields[0].topology.is_some());
            assert!(def.fields[0].measure.is_some());
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_full() {
    // Full entity definition with all features
    let source = r#"
entity.stellar.planet {
    : strata(stellar)
    : count(config.stellar.planet_count)

    schema {
        mass: Scalar<kg, 1e20..1e30>
        radius: Scalar<m>
        position: Vec3<m>
    }

    config {
        orbital_period: 365.25 <days>
    }

    resolve {
        self.position + signal.stellar.gravity_field * dt
    }

    assert {
        self.mass > 0.0, "mass_positive"
    }

    field.surface_gravity {
        : Scalar<m/s2>
        measure {
            const.physics.G * self.mass / (self.radius * self.radius)
        }
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.path.node.join("."), "stellar.planet");
            assert!(def.strata.is_some());
            assert!(def.count_source.is_some());
            assert_eq!(def.schema.len(), 3);
            assert!(!def.config_defaults.is_empty());
            assert!(def.resolve.is_some());
            assert!(def.assertions.is_some());
            assert_eq!(def.fields.len(), 1);
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_multiple_fields() {
    let source = r#"
entity.stellar.moon {
    : strata(stellar)

    field.position {
        : Vec3<m>
        measure { self.position }
    }

    field.velocity {
        : Vec3<m/s>
        measure { self.velocity }
    }

    field.distance {
        : Scalar<m>
        measure { kernel.length(self.position) }
    }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.fields.len(), 3);
            assert_eq!(def.fields[0].name.node, "position");
            assert_eq!(def.fields[1].name.node, "velocity");
            assert_eq!(def.fields[2].name.node, "distance");
        }
        _ => panic!("expected EntityDef"),
    }
}

// === Logical operator keyword tests ===

#[test]
fn test_parse_and_keyword() {
    // Tests that 'and' keyword is accepted as alternative to '&&'
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                let a = 1 in
                let b = 2 in
                a > 0 and b > 0
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            // Navigate through let expressions to the binary expression
            fn find_binary(expr: &Expr) -> Option<&BinaryOp> {
                match expr {
                    Expr::Let { body, .. } => find_binary(&body.node),
                    Expr::Binary { op, .. } => Some(op),
                    _ => None,
                }
            }
            let op = find_binary(&resolve.body.node).expect("expected Binary inside Let");
            assert_eq!(op, &BinaryOp::And);
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_or_keyword() {
    // Tests that 'or' keyword is accepted as alternative to '||'
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                let x = 0 in
                x < -1 or x > 1
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            fn find_binary(expr: &Expr) -> Option<&BinaryOp> {
                match expr {
                    Expr::Let { body, .. } => find_binary(&body.node),
                    Expr::Binary { op, .. } => Some(op),
                    _ => None,
                }
            }
            let op = find_binary(&resolve.body.node).expect("expected Binary inside Let");
            assert_eq!(op, &BinaryOp::Or);
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_not_keyword() {
    // Tests that 'not' keyword is accepted as alternative to '!'
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                let flag = 1 in
                not flag
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            fn find_unary(expr: &Expr) -> Option<&UnaryOp> {
                match expr {
                    Expr::Let { body, .. } => find_unary(&body.node),
                    Expr::Unary { op, .. } => Some(op),
                    _ => None,
                }
            }
            let op = find_unary(&resolve.body.node).expect("expected Unary inside Let");
            assert_eq!(op, &UnaryOp::Not);
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_mixed_logical_operators() {
    // Tests mixing symbol and keyword forms in the same expression
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                let a = 1 in
                let b = 2 in
                let c = 3 in
                a > 0 && b > 0 or c > 0
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            // The top-level should be 'or' since it has lower precedence than 'and'
            fn find_top_binary(expr: &Expr) -> Option<&BinaryOp> {
                match expr {
                    Expr::Let { body, .. } => find_top_binary(&body.node),
                    Expr::Binary { op, .. } => Some(op),
                    _ => None,
                }
            }
            let op = find_top_binary(&resolve.body.node).expect("expected Binary");
            assert_eq!(op, &BinaryOp::Or);
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_not_does_not_match_notation() {
    // Ensures 'not' doesn't accidentally match the start of 'notation' or similar words
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                notation
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
                Expr::Path(path) => {
                    assert_eq!(path.join("."), "notation");
                }
                other => panic!("expected Path, got {:?}", other),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}
