use super::*;
use crate::ast::{
    BinaryOp, Expr, Item, Literal, PrimitiveParamValue, PrimitiveTypeExpr, TypeExpr, UnaryOp,
};
use continuum_foundation::PrimitiveParamKind;

fn expect_primitive<'a>(expr: &'a TypeExpr, expected: &str) -> &'a PrimitiveTypeExpr {
    match expr {
        TypeExpr::Primitive(primitive) => {
            assert_eq!(primitive.id.name(), expected);
            primitive
        }
        TypeExpr::Named(name) => panic!("Expected primitive type, got named '{name}'"),
    }
}

fn find_param<'a>(
    primitive: &'a PrimitiveTypeExpr,
    kind: PrimitiveParamKind,
) -> Option<&'a PrimitiveParamValue> {
    primitive.params.iter().find(|param| param.kind() == kind)
}

fn expect_unit<'a>(expr: &'a TypeExpr, expected_type: &str) -> &'a str {
    let primitive = expect_primitive(expr, expected_type);
    match find_param(primitive, PrimitiveParamKind::Unit) {
        Some(PrimitiveParamValue::Unit(unit)) => unit,
        _ => panic!("expected {expected_type} unit"),
    }
}

#[test]
fn test_parse_comparison_with_unit_repro() {
    let source = r#"
        fracture.test {
            when {
                signal.temp > 350.0 <K>
            }
            emit {
                signal.vapor <- 50.0
            }
        }
    "#;
    let (result, errors) = parse(source);
    for err in &errors {
        println!("Error at {}: {}", err.span().start, err.reason());
    }
    assert!(
        errors.is_empty(),
        "Reproduction of comparison with unit failed: {:?}",
        errors
    );
    assert!(result.is_some());
}

#[test]
fn test_parse_member_followed_by_fracture_repro() {
    let source = r#"
        member.test.signal {
            : Scalar
            : strata(test)
            resolve { prev }
        }

        fracture.atmosphere.runaway_greenhouse {
            when {
                signal.atmosphere.surface_temp.x > 350.0
            }
            emit {
                signal.atmosphere.water_vapor <- 50.0
            }
        }
    "#;
    let (result, errors) = parse(source);
    for err in &errors {
        println!("Error at {}: {}", err.span().start, err.reason());
    }
    assert!(
        errors.is_empty(),
        "Failed to parse member followed by fracture: {:?}",
        errors
    );
    assert!(result.is_some());
}

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
fn test_parse_signal_with_tensor_constraints() {
    let source = r#"
        signal.terra.stress_tensor {
            : Tensor<3,3,Pa>
            : symmetric
            : positive_definite
            : strata(terra.tectonics)

            resolve {
                prev
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            // Verify the type is Tensor
            let ty = def.ty.as_ref().expect("should have type");
            expect_primitive(&ty.node, "Tensor");
            // Constraints are now stored on SignalDef, not TypeExpr
            assert_eq!(def.tensor_constraints.len(), 2);
            assert_eq!(
                def.tensor_constraints[0],
                crate::ast::TensorConstraint::Symmetric
            );
            assert_eq!(
                def.tensor_constraints[1],
                crate::ast::TensorConstraint::PositiveDefinite
            );
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_signal_with_seq_constraints() {
    let source = r#"
        signal.terra.particle_masses {
            : Seq<Scalar<kg>>
            : each(1e20..1e28)
            : sum(1e25..1e30)
            : strata(terra.physics)

            resolve {
                prev
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            // Verify the type is Seq
            let ty = def.ty.as_ref().expect("should have type");
            expect_primitive(&ty.node, "Seq");
            // Constraints are now stored on SignalDef, not TypeExpr
            assert_eq!(def.seq_constraints.len(), 2);
            match &def.seq_constraints[0] {
                crate::ast::SeqConstraint::Each(range) => {
                    assert_eq!(range.min, 1e20);
                    assert_eq!(range.max, 1e28);
                }
                _ => panic!("expected Each constraint"),
            }
            match &def.seq_constraints[1] {
                crate::ast::SeqConstraint::Sum(range) => {
                    assert_eq!(range.min, 1e25);
                    assert_eq!(range.max, 1e30);
                }
                _ => panic!("expected Sum constraint"),
            }
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
            assert!(def.emit.is_some(), "emit should be present");
        }
        _ => panic!("expected FractureDef"),
    }
}

#[test]
fn test_parse_fracture_with_strata_and_config() {
    let source = r#"
        fracture.thermal.mechanical_coupling {
            : strata(thermal)

            config {
                reference_heat_j: 8.0e30
                coupling_strength: 0.1
                base_flow_strength: 5.0
            }

            when {
                abs(signal.mantle.heat_content - config.fracture.thermal.mechanical_coupling.reference_heat_j) > 1e29
            }

            emit {
                signal.mantle.flow_strength <- 1.0
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::FractureDef(def) => {
            assert_eq!(def.path.node.join("."), "thermal.mechanical_coupling");
            assert!(def.strata.is_some(), "strata should be present");
            assert_eq!(def.strata.as_ref().unwrap().node.join("."), "thermal");
            assert_eq!(def.local_config.len(), 3);
            assert_eq!(def.conditions.len(), 1);
            assert!(def.emit.is_some(), "emit should be present");
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
fn test_parse_vector_with_magnitude_range() {
    let source = r#"
        type.OrbitalState {
            position: Vec3<m, magnitude: 1e10..1e12>
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::TypeDef(def) => {
            assert_eq!(def.fields.len(), 1);
            let primitive = expect_primitive(&def.fields[0].ty.node, "Vec3");
            let unit = match find_param(primitive, PrimitiveParamKind::Unit) {
                Some(PrimitiveParamValue::Unit(unit)) => unit,
                _ => panic!("expected Vec3 unit"),
            };
            assert_eq!(unit, "m");
            let magnitude = match find_param(primitive, PrimitiveParamKind::Magnitude) {
                Some(PrimitiveParamValue::Magnitude(range)) => range,
                _ => panic!("expected Vec3 magnitude"),
            };
            assert_eq!(magnitude.min, 1e10);
            assert_eq!(magnitude.max, 1e12);
        }
        _ => panic!("expected TypeDef"),
    }
}

#[test]
fn test_parse_vec4_unit_quaternion() {
    // Vec4<1, magnitude: 1> is a unit quaternion (magnitude exactly 1)
    let source = r#"
        type.Orientation {
            rotation: Vec4<1, magnitude: 1>
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::TypeDef(def) => {
            let primitive = expect_primitive(&def.fields[0].ty.node, "Vec4");
            let unit = match find_param(primitive, PrimitiveParamKind::Unit) {
                Some(PrimitiveParamValue::Unit(unit)) => unit,
                _ => panic!("expected Vec4 unit"),
            };
            assert_eq!(unit, "1");
            let magnitude = match find_param(primitive, PrimitiveParamKind::Magnitude) {
                Some(PrimitiveParamValue::Magnitude(range)) => range,
                _ => panic!("expected Vec4 magnitude"),
            };
            // Single value 1 is converted to range 1..1
            assert_eq!(magnitude.min, 1.0);
            assert_eq!(magnitude.max, 1.0);
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
                    match &args[0].value.node {
                        Expr::Call {
                            function,
                            args: inner_args,
                        } => {
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
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "kg/m³");
        }
        _ => panic!("expected SignalDef"),
    }

    match &unit.items[1].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "W/m²");
        }
        _ => panic!("expected SignalDef"),
    }

    match &unit.items[2].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "m/s²");
        }
        _ => panic!("expected SignalDef"),
    }

    match &unit.items[3].node {
        Item::SignalDef(def) => {
            let ty = def.ty.as_ref().unwrap();
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "W/m²/K⁴");
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
            let primitive = expect_primitive(&ty.node, "Scalar");
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "kg/m³");
            let range = match find_param(primitive, PrimitiveParamKind::Range) {
                Some(PrimitiveParamValue::Range(range)) => range,
                _ => panic!("expected Scalar range"),
            };
            assert_eq!(range.min, 1000.0);
            assert_eq!(range.max, 10000.0);
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
            let primitive = expect_primitive(&ty.node, "Scalar");
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "Pa*s");
            let range = match find_param(primitive, PrimitiveParamKind::Range) {
                Some(PrimitiveParamValue::Range(range)) => range,
                _ => panic!("expected Scalar range"),
            };
            assert_eq!(range.min, 0.0);
            assert_eq!(range.max, 1e24);
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
            let primitive = expect_primitive(&ty.node, "Scalar");
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "kg/m²/yr");
            let range = match find_param(primitive, PrimitiveParamKind::Range) {
                Some(PrimitiveParamValue::Range(range)) => range,
                _ => panic!("expected Scalar range"),
            };
            assert_eq!(range.min, 0.0);
            assert_eq!(range.max, 1.0);
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
            let primitive = expect_primitive(&ty.node, "Scalar");
            let unit = expect_unit(&ty.node, "Scalar");
            assert_eq!(unit, "Pa*s");
            let range = match find_param(primitive, PrimitiveParamKind::Range) {
                Some(PrimitiveParamValue::Range(range)) => range,
                _ => panic!("expected Scalar range"),
            };
            assert_eq!(range.min, 0.0);
            assert_eq!(range.max, 1e24);
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

    resolve {
        let phase = prev.x + prev.y * dt.raw in
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
                Expr::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
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
fn test_parse_else_if_chain() {
    // Test else-if chain parsing
    let source = r#"
signal.test.else_if {
    : Scalar<1>
    resolve {
        if prev > 100.0 {
            3.0
        } else if prev > 50.0 {
            2.0
        } else if prev > 0.0 {
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
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert!(def.resolve.is_some());
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::If {
                    condition,
                    then_branch: _,
                    else_branch,
                } => {
                    // First condition: prev > 100.0
                    match &condition.node {
                        Expr::Binary { op, .. } => {
                            assert_eq!(*op, BinaryOp::Gt);
                        }
                        _ => panic!("expected Binary comparison"),
                    }
                    // Else branch should be another If (the else-if)
                    let else_if = else_branch.as_ref().expect("should have else branch");
                    match &else_if.node {
                        Expr::If {
                            condition: cond2,
                            else_branch: else2,
                            ..
                        } => {
                            // Second condition: prev > 50.0
                            match &cond2.node {
                                Expr::Binary { op, .. } => {
                                    assert_eq!(*op, BinaryOp::Gt);
                                }
                                _ => panic!("expected Binary comparison in else-if"),
                            }
                            // Should have another else-if or else
                            let else_if2 = else2.as_ref().expect("should have second else branch");
                            match &else_if2.node {
                                Expr::If {
                                    else_branch: else3, ..
                                } => {
                                    // Final else should exist
                                    assert!(else3.is_some(), "should have final else");
                                }
                                _ => panic!("expected nested If in second else-if"),
                            }
                        }
                        _ => panic!("expected If in else branch (else-if)"),
                    }
                }
                _ => panic!("expected If expression, got {:?}", resolve.body.node),
            }
        }
        _ => panic!("expected SignalDef"),
    }
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
                Expr::Unary {
                    op: outer_op,
                    operand,
                } => {
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
// Entities are pure index spaces - they only define what exists, not state.
// Per-entity state is defined via member signals.

#[test]
fn test_parse_entity_empty() {
    let source = r#"
entity.stellar.moon {}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.path.node.join("."), "stellar.moon");
            assert!(def.count_source.is_none());
            assert!(def.count_bounds.is_none());
        }
        _ => panic!("expected EntityDef"),
    }
}

#[test]
fn test_parse_entity_with_count_source() {
    let source = r#"
entity.stellar.moon {
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
fn test_parse_entity_with_both_count_options() {
    // Can have both count source and bounds for validation
    let source = r#"
entity.stellar.planet {
    : count(config.stellar.planet_count)
    : count(1..10)
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::EntityDef(def) => {
            assert_eq!(def.path.node.join("."), "stellar.planet");
            assert!(def.count_source.is_some());
            assert!(def.count_bounds.is_some());
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

// ============================================================================
// Named Parameter Tests (#68)
// ============================================================================

#[test]
fn test_parse_named_argument_basic() {
    // Test single named argument: func(a, method: rk4)
    let source = r#"
        signal.test {
            : Scalar<1>
            resolve {
                integrate(prev, rate, method: rk4)
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
                Expr::Call { args, .. } => {
                    assert_eq!(args.len(), 3);
                    // First two args are positional
                    assert!(args[0].name.is_none());
                    assert!(args[1].name.is_none());
                    // Third arg is named
                    assert_eq!(args[2].name.as_ref().unwrap(), "method");
                    match &args[2].value.node {
                        Expr::Path(p) => assert_eq!(p.join("."), "rk4"),
                        _ => panic!("expected Path for named arg value"),
                    }
                }
                other => panic!("expected Call, got {:?}", other),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_multiple_named_arguments() {
    // Test multiple named arguments
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                relax(current, target, tau: 0.5, method: exp)
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
                Expr::Call { args, .. } => {
                    assert_eq!(args.len(), 4);
                    // First two args are positional
                    assert!(args[0].name.is_none());
                    assert!(args[1].name.is_none());
                    // Third and fourth args are named
                    assert_eq!(args[2].name.as_ref().unwrap(), "tau");
                    assert_eq!(args[3].name.as_ref().unwrap(), "method");
                }
                other => panic!("expected Call, got {:?}", other),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_only_named_arguments() {
    // Test function call with only named arguments
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                configure(x: 1.0, y: 2.0, z: 3.0)
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
                Expr::Call { args, .. } => {
                    assert_eq!(args.len(), 3);
                    assert_eq!(args[0].name.as_ref().unwrap(), "x");
                    assert_eq!(args[1].name.as_ref().unwrap(), "y");
                    assert_eq!(args[2].name.as_ref().unwrap(), "z");
                }
                other => panic!("expected Call, got {:?}", other),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_named_argument_with_expression_value() {
    // Test named argument with complex expression as value
    let source = r#"
        signal.test {
            : Scalar
            resolve {
                decay(prev, rate: config.physics.tau * 2.0)
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
                Expr::Call { args, .. } => {
                    assert_eq!(args.len(), 2);
                    assert!(args[0].name.is_none()); // prev is positional
                    assert_eq!(args[1].name.as_ref().unwrap(), "rate");
                    // Value should be a binary multiplication
                    match &args[1].value.node {
                        Expr::Binary { op, .. } => assert_eq!(*op, BinaryOp::Mul),
                        other => panic!("expected Binary, got {:?}", other),
                    }
                }
                other => panic!("expected Call, got {:?}", other),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

// ============================================================================
// Doc Comment Tests (#99)
// ============================================================================

#[test]
fn test_parse_doc_comment_signal() {
    let source = r#"
/// This signal tracks temperature in Kelvin.
/// It is resolved by adding collected thermal energy.
signal.test.temperature {
    : Scalar<K>
    resolve { prev + collected }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert!(def.doc.is_some());
            let doc = def.doc.as_ref().unwrap();
            assert!(doc.contains("temperature in Kelvin"));
            assert!(doc.contains("thermal energy"));
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_doc_comment_function() {
    let source = r#"
/// Linear interpolation between two values.
/// Returns a + (b - a) * t
fn.math.lerp(a, b, t) {
    a + (b - a) * t
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::FnDef(def) => {
            assert!(def.doc.is_some());
            let doc = def.doc.as_ref().unwrap();
            assert!(doc.contains("Linear interpolation"));
        }
        _ => panic!("expected FnDef"),
    }
}

#[test]
fn test_parse_module_doc() {
    let source = r#"
//! Module for thermal physics simulation.
//! This file defines temperature signals.

signal.test.temp {
    : Scalar<K>
    resolve { prev }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert!(unit.module_doc.is_some());
    let doc = unit.module_doc.as_ref().unwrap();
    assert!(doc.contains("thermal physics"));
    assert!(doc.contains("temperature signals"));
}

#[test]
fn test_parse_no_doc_comment() {
    // Items without doc comments should have doc: None
    let source = r#"
signal.test.nodoc {
    : Scalar<1>
    resolve { prev }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert!(def.doc.is_none());
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_regular_comment_not_doc() {
    // Regular comments (// without third /) should not be captured as doc
    let source = r#"
// This is a regular comment, not a doc comment
signal.test.regular {
    : Scalar<1>
    resolve { prev }
}
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            assert!(def.doc.is_none(), "regular comment should not become doc");
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_poc_file_parses() {
    // Navigate from crate root to workspace root
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // kernels
        .unwrap()
        .parent() // crates
        .unwrap()
        .parent() // workspace root
        .unwrap();
    let poc_path = workspace_root.join("examples/poc/poc.cdsl");
    let content = std::fs::read_to_string(&poc_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", poc_path.display(), e));
    let (ast, errors) = crate::parse(&content);

    if !errors.is_empty() {
        for err in &errors {
            let span = err.span();
            let line_num = content[..span.start].matches('\n').count() + 1;
            eprintln!("Line {}: {}", line_num, err);
        }
    }

    assert!(errors.is_empty(), "poc.cdsl should parse without errors");
    assert!(ast.is_some(), "poc.cdsl should produce an AST");
    let ast = ast.unwrap();
    assert!(ast.items.len() > 0, "poc.cdsl should have items");
    // Verify module doc was captured
    assert!(ast.module_doc.is_some(), "poc.cdsl should have module doc");
}

#[test]
fn test_terra_file_parses() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let terra_path = workspace_root.join("examples/terra/terra.cdsl");
    let content = std::fs::read_to_string(&terra_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", terra_path.display(), e));
    let (ast, errors) = crate::parse(&content);

    if !errors.is_empty() {
        for err in &errors {
            let span = err.span();
            let line_num = content[..span.start].matches('\n').count() + 1;
            eprintln!("Line {}: {}", line_num, err);
        }
    }

    assert!(errors.is_empty(), "terra.cdsl should parse without errors");
    assert!(ast.is_some(), "terra.cdsl should produce an AST");
    let ast = ast.unwrap();
    assert!(ast.items.len() > 0, "terra.cdsl should have items");
    // Verify module doc was captured
    assert!(
        ast.module_doc.is_some(),
        "terra.cdsl should have module doc"
    );
}

#[test]
fn test_doc_comments_in_config_block_simple() {
    // First test: just a single entry with doc comment
    let source = "config { /// Doc\natmosphere.temp: 288.0 }";
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::ConfigBlock(block) => {
            assert_eq!(block.entries.len(), 1);
            assert_eq!(block.entries[0].doc.as_ref().unwrap(), "Doc");
        }
        _ => panic!("expected ConfigBlock"),
    }
}

#[test]
fn test_config_entry_without_doc() {
    // Test entry without doc comment
    let source = "config { atmosphere.temp: 288.0 }";
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::ConfigBlock(block) => {
            assert_eq!(block.entries.len(), 1);
            assert!(block.entries[0].doc.is_none());
        }
        _ => panic!("expected ConfigBlock"),
    }
}

#[test]
fn test_config_mixed_doc_entries() {
    // Test: entry with doc followed by entry without doc
    let source = "config { /// Doc\na.x: 1.0\nb.y: 2.0 }";
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::ConfigBlock(block) => {
            assert_eq!(block.entries.len(), 2);
            assert_eq!(block.entries[0].doc.as_ref().unwrap(), "Doc");
            assert!(block.entries[1].doc.is_none());
        }
        _ => panic!("expected ConfigBlock"),
    }
}

#[test]
fn test_doc_comments_in_config_block() {
    let source = r#"
        config {
            /// First doc comment
            atmosphere.initial_temp: 288.0
            /// Second doc
            /// with multiple lines
            atmosphere.pressure: 101325.0
            atmosphere.no_doc: 1.0
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::ConfigBlock(block) => {
            assert_eq!(block.entries.len(), 3);
            // First entry has doc
            assert_eq!(block.entries[0].doc.as_ref().unwrap(), "First doc comment");
            // Second entry has multi-line doc
            assert_eq!(
                block.entries[1].doc.as_ref().unwrap(),
                "Second doc\nwith multiple lines"
            );
            // Third entry has no doc
            assert!(block.entries[2].doc.is_none());
        }
        _ => panic!("expected ConfigBlock"),
    }
}

#[test]
fn test_doc_comments_in_const_block() {
    let source = r#"
        const {
            /// Stefan-Boltzmann constant
            physics.sigma: 5.67e-8
            physics.no_doc: 3.14
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::ConstBlock(block) => {
            assert_eq!(block.entries.len(), 2);
            // First entry has doc
            assert_eq!(
                block.entries[0].doc.as_ref().unwrap(),
                "Stefan-Boltzmann constant"
            );
            // Second entry has no doc
            assert!(block.entries[1].doc.is_none());
        }
        _ => panic!("expected ConstBlock"),
    }
}

#[test]
fn test_atmosphere_file_parses() {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let atmosphere_path = workspace_root.join("examples/terra/atmosphere/atmosphere.cdsl");
    let content = std::fs::read_to_string(&atmosphere_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", atmosphere_path.display(), e));
    let (ast, errors) = crate::parse(&content);

    if !errors.is_empty() {
        for err in &errors {
            let span = err.span();
            let line_num = content[..span.start].matches('\n').count() + 1;
            eprintln!("Line {}: {}", line_num, err);
        }
    }

    assert!(
        errors.is_empty(),
        "atmosphere.cdsl should parse without errors"
    );
    assert!(ast.is_some(), "atmosphere.cdsl should produce an AST");
    let ast = ast.unwrap();
    assert!(ast.items.len() > 0, "atmosphere.cdsl should have items");

    // Verify config entry doc comments are captured
    for item in &ast.items {
        if let Item::ConfigBlock(block) = &item.node {
            // The first entry should have a doc comment "Radiative balance"
            if let Some(first_entry) = block.entries.first() {
                if first_entry.path.node.join(".") == "atmosphere.initial_surface_temp" {
                    assert!(
                        first_entry.doc.is_some(),
                        "atmosphere.initial_surface_temp should have doc comment"
                    );
                    assert!(
                        first_entry
                            .doc
                            .as_ref()
                            .unwrap()
                            .contains("Radiative balance"),
                        "doc should contain 'Radiative balance'"
                    );
                }
            }
        }
    }
}

#[test]
fn test_parse_world_def() {
    let source = r#"
        world.terra {
            : title("Earth Planetary Simulation")
            : version("1.0.0")

            policy {
                determinism: "strict"
                faults: "fatal"
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::WorldDef(def) => {
            assert_eq!(def.path.node.join("."), "terra");
            assert_eq!(
                def.title.as_ref().unwrap().node,
                "Earth Planetary Simulation"
            );
            assert_eq!(def.version.as_ref().unwrap().node, "1.0.0");

            let policy = def.policy.as_ref().expect("expected policy block");
            assert_eq!(policy.entries.len(), 2);

            assert_eq!(policy.entries[0].path.node.join("."), "determinism");
            match &policy.entries[0].value.node {
                Literal::String(s) => assert_eq!(s, "strict"),
                _ => panic!("expected string literal"),
            }
        }
        _ => panic!("expected WorldDef"),
    }
}

#[test]
fn test_parse_math_constant_with_digit() {
    let source = r#"
        signal.test.const {
            : Scalar
            resolve {
                SQRT2 * FRAC_1_PI
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    assert_eq!(unit.items.len(), 1);
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Binary { op, left, right } => {
                    assert_eq!(*op, BinaryOp::Mul);
                    // SQRT2
                    match &left.node {
                        Expr::Path(path) => {
                            assert_eq!(path.to_string(), "SQRT2");
                        }
                        _ => panic!("expected Path for SQRT2, got {:?}", left.node),
                    }
                    // FRAC_1_PI
                    match &right.node {
                        Expr::Path(path) => {
                            assert_eq!(path.to_string(), "FRAC_1_PI");
                        }
                        _ => panic!("expected Path for FRAC_1_PI, got {:?}", right.node),
                    }
                }
                _ => panic!("expected Binary"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_fracture_emit_semicolons() {
    let source = r#"
        fracture.test {
            when { true }
            emit {
                signal.a <- 1.0;
                signal.b <- 2.0
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::FractureDef(def) => {
            let emit = def.emit.as_ref().unwrap();
            match &emit.node {
                Expr::Block(exprs) => {
                    assert_eq!(exprs.len(), 2);
                }
                _ => panic!("expected Block"),
            }
        }
        _ => panic!("expected FractureDef"),
    }
}

#[test]
fn test_parse_sim_time_expression() {
    let source = r#"
        signal.test.clock {
            : Scalar<1>
            resolve { sim.time + 1.0 }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::SignalDef(def) => {
            let resolve = def.resolve.as_ref().unwrap();
            match &resolve.body.node {
                Expr::Binary { op, left, .. } => {
                    assert_eq!(*op, BinaryOp::Add);
                    // sim.time is parsed as Path(["sim", "time"])
                    match &left.node {
                        Expr::Path(path) => {
                            assert_eq!(path.segments.len(), 2);
                            assert_eq!(path.segments[0], "sim");
                            assert_eq!(path.segments[1], "time");
                        }
                        _ => panic!("expected Path for sim.time, got {:?}", left.node),
                    }
                }
                _ => panic!("expected binary expression"),
            }
        }
        _ => panic!("expected SignalDef"),
    }
}

#[test]
fn test_parse_impulse_with_metadata() {
    let source = r#"
        impulse.test.quake {
            : title("Earthquake")
            : symbol("Q")
            config {
                strength: 1.0
            }
            apply {
                payload * config.test.quake.strength
            }
        }
    "#;
    let (result, errors) = parse(source);
    assert!(errors.is_empty(), "errors: {:?}", errors);
    let unit = result.unwrap();
    match &unit.items[0].node {
        Item::ImpulseDef(def) => {
            assert_eq!(def.path.node.join("."), "test.quake");
            assert_eq!(def.title.as_ref().unwrap().node, "Earthquake");
            assert_eq!(def.symbol.as_ref().unwrap().node, "Q");
            assert_eq!(def.local_config.len(), 1);
            assert!(def.apply.is_some());
        }
        _ => panic!("expected ImpulseDef"),
    }
}
