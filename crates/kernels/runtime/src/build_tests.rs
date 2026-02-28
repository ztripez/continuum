//! Tests for runtime construction from compiled worlds.
//!
//! Validates build_runtime panic conditions, scenario config override validation,
//! and correct handling of era/stratum/signal initialization.

use crate::build::{build_runtime, Scenario};
use crate::types::*;
use continuum_cdsl::ast::{CompiledWorld, EraTransition, TypeExpr, TypedExpr, World, WorldDecl};
use continuum_cdsl::foundation::{Path, Shape, Span, Type, Unit};
use continuum_foundation::WorldPolicy;
use continuum_kernel_types::KernelId;

fn empty_world() -> World {
    World::new(WorldDecl {
        path: Path::from_path_str("demo"),
        title: None,
        version: None,
        warmup: None,
        attributes: Vec::new(),
        span: Span::new(0, 0, 0, 0),
        doc: None,
        debug: false,
        policy: WorldPolicy::default(),
    })
}

#[test]
#[should_panic(expected = "world missing initial era")]
fn test_runtime_panics_without_initial_era() {
    let world = empty_world();
    let compiled = CompiledWorld::new(world, Default::default());
    let _runtime = build_runtime(compiled, None);
}

#[test]
#[should_panic(expected = "initial era 'missing' not found in era configs")]
fn test_runtime_panics_when_initial_missing_in_configs() {
    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("missing"));
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());
    let _runtime = build_runtime(compiled, None);
}

#[test]
#[should_panic(expected = "non-literal dt expression")]
fn test_runtime_panics_on_non_literal_dt() {
    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    let left = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    let right = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 2.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Call {
            kernel: KernelId::new("maths", "add"),
            args: vec![left, right],
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );

    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());
    let _runtime = build_runtime(compiled, None);
}

#[test]
fn test_runtime_compiles_transitions() {
    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );

    // A Bool literal condition — always true
    let condition = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::BoolLiteral(true),
        Type::Bool,
        span,
    );

    let mut era_main = continuum_cdsl::ast::Era::new(
        EraId::new("main"),
        Path::from_path_str("main"),
        dt.clone(),
        span,
    );
    era_main
        .transitions
        .push(EraTransition::new(EraId::new("next"), condition, span));
    world.eras.insert(Path::from_path_str("main"), era_main);

    // Add the target era so it's a valid transition
    let era_next =
        continuum_cdsl::ast::Era::new(EraId::new("next"), Path::from_path_str("next"), dt, span);
    world.eras.insert(Path::from_path_str("next"), era_next);

    let compiled = CompiledWorld::new(world, Default::default());
    let runtime = build_runtime(compiled, None);

    // The runtime should start in "main" era
    assert_eq!(runtime.era(), &EraId::new("main"));
}

// ========== Scenario Config Override Validation Tests ==========

#[test]
#[should_panic(expected = "non-existent config path")]
fn test_scenario_rejects_non_existent_config_path() {
    use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    // Add a valid config entry
    let config_entry = ConfigEntry {
        path: Path::from_path_str("thermal.decay_halflife"),
        default: Some(Expr {
            kind: UntypedKind::Literal {
                value: 1.42e17,
                unit: None,
            },
            span,
        }),
        type_expr: TypeExpr::Scalar {
            unit: None,
            bounds: None,
        },
        span,
        doc: None,
    };
    world
        .declarations
        .push(Declaration::Config(vec![config_entry]));

    // Add era
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());

    // Try to override a NON-EXISTENT path (typo: "inital" instead of "initial")
    let scenario = Scenario::with_config_overrides(
        [(
            Path::from_path_str("thermal.inital_temp"),
            Value::Scalar(6000.0),
        )]
        .into_iter()
        .collect(),
    );

    let _runtime = build_runtime(compiled, Some(scenario));
}

#[test]
#[should_panic(expected = "override immutable const path")]
fn test_scenario_rejects_const_override() {
    use continuum_cdsl::ast::{ConstEntry, Declaration, Expr, UntypedKind};

    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    // Add a const entry
    let const_entry = ConstEntry {
        path: Path::from_path_str("physics.stefan_boltzmann"),
        value: Expr {
            kind: UntypedKind::Literal {
                value: 5.67e-8,
                unit: None,
            },
            span,
        },
        type_expr: TypeExpr::Scalar {
            unit: None,
            bounds: None,
        },
        span,
        doc: None,
    };
    world
        .declarations
        .push(Declaration::Const(vec![const_entry]));

    // Add era
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());

    // Try to override a CONST value (should fail)
    let scenario = Scenario::with_config_overrides(
        [(
            Path::from_path_str("physics.stefan_boltzmann"),
            Value::Scalar(6.0e-8),
        )]
        .into_iter()
        .collect(),
    );

    let _runtime = build_runtime(compiled, Some(scenario));
}

#[test]
#[should_panic(expected = "incompatible type")]
fn test_scenario_rejects_wrong_value_type() {
    use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    // Add a config entry expecting Scalar
    let config_entry = ConfigEntry {
        path: Path::from_path_str("thermal.initial_temp"),
        default: Some(Expr {
            kind: UntypedKind::Literal {
                value: 5500.0,
                unit: None,
            },
            span,
        }),
        type_expr: TypeExpr::Scalar {
            unit: None,
            bounds: None,
        },
        span,
        doc: None,
    };
    world
        .declarations
        .push(Declaration::Config(vec![config_entry]));

    // Add era
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());

    // Try to override with WRONG TYPE (Vec3 instead of Scalar)
    let scenario = Scenario::with_config_overrides(
        [(
            Path::from_path_str("thermal.initial_temp"),
            Value::Vec3([1.0, 2.0, 3.0]),
        )]
        .into_iter()
        .collect(),
    );

    let _runtime = build_runtime(compiled, Some(scenario));
}

#[test]
fn test_scenario_valid_override_succeeds() {
    use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    // Add a config entry
    let config_entry = ConfigEntry {
        path: Path::from_path_str("thermal.initial_temp"),
        default: Some(Expr {
            kind: UntypedKind::Literal {
                value: 5500.0,
                unit: None,
            },
            span,
        }),
        type_expr: TypeExpr::Scalar {
            unit: None,
            bounds: None,
        },
        span,
        doc: None,
    };
    world
        .declarations
        .push(Declaration::Config(vec![config_entry]));

    // Add era
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());

    // Valid override with correct type
    let scenario = Scenario::with_config_overrides(
        [(
            Path::from_path_str("thermal.initial_temp"),
            Value::Scalar(6000.0),
        )]
        .into_iter()
        .collect(),
    );

    // Should succeed without panic
    let _runtime = build_runtime(compiled, Some(scenario));
}

#[test]
#[should_panic(expected = "incompatible type")]
fn test_scenario_rejects_vector_dimension_mismatch() {
    use continuum_cdsl::ast::{ConfigEntry, Declaration, Expr, UntypedKind};

    let span = Span::new(0, 0, 0, 0);
    let mut world = empty_world();
    world.initial_era = Some(EraId::new("main"));

    // Add a config entry expecting Vector<3>
    let config_entry = ConfigEntry {
        path: Path::from_path_str("physics.position"),
        default: Some(Expr {
            kind: UntypedKind::Vector(vec![
                Expr {
                    kind: UntypedKind::Literal {
                        value: 0.0,
                        unit: None,
                    },
                    span,
                },
                Expr {
                    kind: UntypedKind::Literal {
                        value: 0.0,
                        unit: None,
                    },
                    span,
                },
                Expr {
                    kind: UntypedKind::Literal {
                        value: 0.0,
                        unit: None,
                    },
                    span,
                },
            ]),
            span,
        }),
        type_expr: TypeExpr::Vector { dim: 3, unit: None },
        span,
        doc: None,
    };
    world
        .declarations
        .push(Declaration::Config(vec![config_entry]));

    // Add era
    let dt = TypedExpr::new(
        continuum_cdsl::ast::ExprKind::Literal {
            value: 1.0,
            unit: Some(Unit::seconds()),
        },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        span,
    );
    world.eras.insert(
        Path::from_path_str("main"),
        continuum_cdsl::ast::Era::new(EraId::new("main"), Path::from_path_str("main"), dt, span),
    );

    let compiled = CompiledWorld::new(world, Default::default());

    // Try to override with WRONG VECTOR DIMENSION (Vec2 instead of Vec3)
    let scenario = Scenario::with_config_overrides(
        [(
            Path::from_path_str("physics.position"),
            Value::Vec2([1.0, 2.0]),
        )]
        .into_iter()
        .collect(),
    );

    let _runtime = build_runtime(compiled, Some(scenario));
}
