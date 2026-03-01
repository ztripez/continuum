use indexmap::IndexMap;
use std::collections::HashSet;

use crate::dag::DagSet;
use crate::dag::{DagBuilder, DagNode, EraDags, NodeId, NodeKind};
use crate::executor::runtime::{EraConfig, Runtime};
use crate::executor::EmittedEvent;
use crate::types::AssertionSeverity;
use crate::types::{
    Dt, EraId, FieldId, Phase, SignalId, StratumId, StratumState, Value, WarmupConfig, WorldPolicy,
};
use crate::FaultPolicy;

fn create_minimal_runtime(era_id: EraId) -> Runtime {
    let mut eras = IndexMap::new();
    eras.insert(
        era_id.clone(),
        EraConfig {
            dt: Dt(1.0),
            strata: IndexMap::new(),
            transition: None,
        },
    );
    Runtime::new(
        era_id,
        eras,
        DagSet::default(),
        Vec::new(),
        WorldPolicy::default(),
    )
}

#[test]
fn test_runtime_creation() {
    let era_id: EraId = "default".into();
    let mut eras = IndexMap::new();
    eras.insert(
        era_id.clone(),
        EraConfig {
            dt: Dt(1.0),
            strata: IndexMap::new(),
            transition: None,
        },
    );
    let runtime = Runtime::new(
        era_id,
        eras,
        DagSet::default(),
        Vec::new(),
        WorldPolicy::default(),
    );
    assert_eq!(runtime.tick(), 0);
    assert_eq!(runtime.sim_time(), 0.0);
}

#[test]
fn test_warmup_fixed_iterations() {
    let era_id: EraId = "test".into();
    let signal_id: SignalId = "temp".into();

    let mut runtime = create_minimal_runtime(era_id);
    runtime.init_signal(signal_id.clone(), Value::Scalar(1000.0));

    // Warmup: halve the value each iteration
    runtime.register_warmup(
        signal_id.clone(),
        Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap();
            Value::Scalar(prev * 0.5)
        }),
        WarmupConfig {
            max_iterations: 5,
            convergence_epsilon: None,
        },
    );

    let result = runtime.execute_warmup().unwrap();

    assert_eq!(result.iterations, 5);
    assert!(!result.converged);

    // 1000 * 0.5^5 = 31.25
    let final_value = runtime.get_signal(&signal_id).unwrap().as_scalar().unwrap();
    assert!((final_value - 31.25).abs() < 0.001);
}

#[test]
fn test_warmup_convergence() {
    let era_id: EraId = "test".into();
    let signal_id: SignalId = "equilibrium".into();

    let mut runtime = create_minimal_runtime(era_id);
    runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

    // Warmup: converge toward 50 by halving the distance each iteration
    runtime.register_warmup(
        signal_id.clone(),
        Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap();
            let target = 50.0;
            Value::Scalar(prev + (target - prev) * 0.5)
        }),
        WarmupConfig {
            max_iterations: 100,
            convergence_epsilon: Some(0.01),
        },
    );

    let result = runtime.execute_warmup().unwrap();

    assert!(result.converged);
    assert!(result.iterations < 100); // Should converge before max

    let final_value = runtime.get_signal(&signal_id).unwrap().as_scalar().unwrap();
    assert!((final_value - 50.0).abs() < 0.1);
}

#[test]
fn test_warmup_divergence_nan() {
    let era_id: EraId = "test".into();
    let signal_id: SignalId = "bad".into();

    let mut runtime = create_minimal_runtime(era_id);
    runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

    // Warmup: produce NaN
    runtime.register_warmup(
        signal_id.clone(),
        Box::new(|_ctx| Value::Scalar(f64::NAN)),
        WarmupConfig {
            max_iterations: 10,
            convergence_epsilon: None,
        },
    );

    let result = runtime.execute_warmup();
    assert!(matches!(
        result,
        Err(crate::error::Error::WarmupDivergence { .. })
    ));
}

#[test]
fn test_warmup_no_functions() {
    let era_id: EraId = "test".into();
    let mut runtime = create_minimal_runtime(era_id);

    let result = runtime.execute_warmup().unwrap();

    assert_eq!(result.iterations, 0);
    assert!(result.converged);
    assert!(runtime.is_warmup_complete());
}

#[test]
fn test_warmup_multiple_signals() {
    let era_id: EraId = "test".into();
    let signal_a: SignalId = "a".into();
    let signal_b: SignalId = "b".into();

    let mut runtime = create_minimal_runtime(era_id);
    runtime.init_signal(signal_a.clone(), Value::Scalar(100.0));
    runtime.init_signal(signal_b.clone(), Value::Scalar(0.0));

    // Signal A: decays toward 50
    runtime.register_warmup(
        signal_a.clone(),
        Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap();
            Value::Scalar(prev + (50.0 - prev) * 0.5)
        }),
        WarmupConfig {
            max_iterations: 50,
            convergence_epsilon: Some(0.1),
        },
    );

    // Signal B: grows toward 50
    runtime.register_warmup(
        signal_b.clone(),
        Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap();
            Value::Scalar(prev + (50.0 - prev) * 0.5)
        }),
        WarmupConfig {
            max_iterations: 50,
            convergence_epsilon: Some(0.1),
        },
    );

    let result = runtime.execute_warmup().unwrap();

    assert!(result.converged);

    let a_val = runtime.get_signal(&signal_a).unwrap().as_scalar().unwrap();
    let b_val = runtime.get_signal(&signal_b).unwrap().as_scalar().unwrap();

    assert!((a_val - 50.0).abs() < 1.0);
    assert!((b_val - 50.0).abs() < 1.0);
}

#[test]
fn test_measure_phase_field_emission() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "temperature".into();
    let field_id: FieldId = "temp_field".into();

    // Build DAG for Resolve phase
    let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    resolve_builder.add_node(DagNode {
        id: NodeId("temp_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let resolve_dag = resolve_builder.build().unwrap();

    // Build DAG for Measure phase
    let mut measure_builder = DagBuilder::new(Phase::Measure, stratum_id.clone());
    measure_builder.add_node(DagNode {
        id: NodeId("temp_measure".to_string()),
        reads: [signal_id.clone()].into_iter().collect(),
        writes: None,
        kind: NodeKind::OperatorMeasure { operator_idx: 0 },
    });
    let measure_dag = measure_builder.build().unwrap();

    let mut era_dags = EraDags::default();
    era_dags.insert(resolve_dag);
    era_dags.insert(measure_dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    // Register resolver: temperature increments by 10 each tick
    runtime.register_resolver(Box::new(|ctx| {
        let prev = ctx.prev.as_scalar().unwrap_or(0.0);
        Value::Scalar(prev + 10.0)
    }));

    // Register measure operator: emit temperature to field
    let signal_id_clone = signal_id.clone();
    let field_id_clone = field_id.clone();
    runtime.register_measure_op(Box::new(move |ctx| {
        let temp = ctx
            .signals
            .get_global_or_prev(&signal_id_clone.to_string())
            .unwrap()
            .as_scalar()
            .unwrap();
        ctx.fields.emit_scalar(field_id_clone.clone(), temp);
    }));

    runtime.init_signal(signal_id, Value::Scalar(100.0));

    // Execute tick
    runtime.execute_tick().unwrap();

    // Check field buffer has the emitted value
    let drained = runtime.drain_fields();
    let samples = drained.get(&field_id).unwrap();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].value.as_scalar(), Some(110.0));

    // Execute another tick
    runtime.execute_tick().unwrap();

    let drained = runtime.drain_fields();
    let samples = drained.get(&field_id).unwrap();
    assert_eq!(samples[0].value.as_scalar(), Some(120.0));
}

#[test]
fn test_impulse_injection() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "energy".into();

    // Build DAG for Resolve phase
    let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    resolve_builder.add_node(DagNode {
        id: NodeId("energy_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let resolve_dag = resolve_builder.build().unwrap();

    let mut era_dags = EraDags::default();
    era_dags.insert(resolve_dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    // Register resolver: prev + collected
    runtime.register_resolver(Box::new(|ctx| {
        let prev = ctx.prev.as_scalar().unwrap_or(0.0);
        Value::Scalar(prev + ctx.inputs)
    }));

    // Register impulse handler: adds payload value to signal
    let signal_id_clone = signal_id.clone();
    let handler_idx = runtime.register_impulse(Box::new(move |ctx, payload| {
        let value = payload.as_scalar().unwrap();
        ctx.channels.accumulate(&signal_id_clone, value);
    }));

    runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

    // Tick 1: no impulse, signal stays at 0
    runtime.execute_tick().unwrap();
    assert_eq!(runtime.get_signal(&signal_id), Some(Value::Scalar(0.0)));

    // Inject impulse for next tick
    runtime.inject_impulse(handler_idx, Value::Scalar(100.0));

    // Tick 2: impulse adds 100
    runtime.execute_tick().unwrap();
    assert_eq!(runtime.get_signal(&signal_id), Some(Value::Scalar(100.0)));

    // Tick 3: no impulse, signal stays at 100
    runtime.execute_tick().unwrap();
    assert_eq!(runtime.get_signal(&signal_id), Some(Value::Scalar(100.0)));

    // Inject multiple impulses
    runtime.inject_impulse(handler_idx, Value::Scalar(25.0));
    runtime.inject_impulse(handler_idx, Value::Scalar(25.0));

    // Tick 4: both impulses add 50 total
    runtime.execute_tick().unwrap();
    assert_eq!(runtime.get_signal(&signal_id), Some(Value::Scalar(150.0)));
}

#[test]
fn test_assertion_during_resolve() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "bounded".into();

    let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    builder.add_node(DagNode {
        id: NodeId("bounded_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let dag = builder.build().unwrap();

    let mut era_dags = EraDags::default();
    era_dags.insert(dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let policy = WorldPolicy {
        faults: FaultPolicy::Fatal,
        ..Default::default()
    };
    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), policy);

    // Register resolver that produces negative values after tick 2
    let tick_counter = std::sync::atomic::AtomicU64::new(0);
    runtime.register_resolver(Box::new(move |_ctx| {
        let tick = tick_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if tick >= 2 {
            Value::Scalar(-10.0)
        } else {
            Value::Scalar(10.0)
        }
    }));

    // Register assertion: value must be positive
    runtime.register_assertion(
        signal_id.clone(),
        Box::new(|ctx| ctx.current.as_scalar().unwrap_or(0.0) > 0.0),
        AssertionSeverity::Error,
        Some("value must be positive".to_string()),
    );

    runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

    // First two ticks should succeed
    runtime.execute_tick().unwrap();
    runtime.execute_tick().unwrap();

    // Third tick should fail assertion
    let result = runtime.execute_tick();
    assert!(matches!(
        result,
        Err(crate::error::Error::AssertionFailed { .. })
    ));
}

#[test]
fn test_era_transition() {
    let era_a: EraId = "era_a".into();
    let era_b: EraId = "era_b".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "counter".into();

    // Build DAG for Resolve phase
    let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    builder.add_node(DagNode {
        id: NodeId("counter_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let dag = builder.build().unwrap();

    let mut era_dags_a = EraDags::default();
    era_dags_a.insert(dag);

    // Build same DAG for era B
    let mut builder_b = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    builder_b.add_node(DagNode {
        id: NodeId("counter_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let dag_b = builder_b.build().unwrap();
    let mut era_dags_b = EraDags::default();
    era_dags_b.insert(dag_b);

    let mut dags = DagSet::default();
    dags.insert_era(era_a.clone(), era_dags_a);
    dags.insert_era(era_b.clone(), era_dags_b);

    // Era A transitions to Era B when counter >= 5
    let mut strata_a = IndexMap::new();
    strata_a.insert(stratum_id.clone(), StratumState::Active);

    let era_b_clone = era_b.clone();
    let signal_id_clone = signal_id.clone();
    let era_a_config = EraConfig {
        dt: Dt(1.0),
        strata: strata_a.clone(),
        transition: Some(Box::new(move |signals, _entities, _sim_time| {
            if let Some(value) = signals.get_global_or_prev(&signal_id_clone.to_string())
                && value.as_scalar().unwrap_or(0.0) >= 5.0 {
                    return Some(era_b_clone.clone());
                }
            None
        })),
    };

    let era_b_config = EraConfig {
        dt: Dt(10.0), // Different dt to verify transition
        strata: strata_a,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_a.clone(), era_a_config);
    eras.insert(era_b.clone(), era_b_config);

    let mut runtime = Runtime::new(
        era_a.clone(),
        eras,
        dags,
        Vec::new(),
        WorldPolicy::default(),
    );

    // Register resolver: increment by 1 each tick
    runtime.register_resolver(Box::new(|ctx| {
        let prev = ctx.prev.as_scalar().unwrap_or(0.0);
        Value::Scalar(prev + 1.0)
    }));

    runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

    // Start in era A
    assert_eq!(runtime.era(), &era_a);

    // Ticks 0-4: counter goes from 0 to 5, still in era A until tick ends
    for i in 0..5 {
        let ctx = runtime.execute_tick().unwrap();
        assert_eq!(ctx.dt.0, 1.0, "tick {} should have dt=1.0", i);
    }

    // After tick 4, counter is 5, transition should happen
    assert_eq!(runtime.era(), &era_b);

    // Next tick should be in era B with dt=10
    let ctx = runtime.execute_tick().unwrap();
    assert_eq!(ctx.dt.0, 10.0, "should now use era B's dt");
    assert_eq!(runtime.era(), &era_b);
}

#[test]
fn test_stratum_gating() {
    let era_id: EraId = "test".into();
    let active_stratum: StratumId = "active".into();
    let gated_stratum: StratumId = "gated".into();
    let active_signal: SignalId = "active_counter".into();
    let gated_signal: SignalId = "gated_counter".into();

    // Build DAGs for both strata
    let mut active_builder = DagBuilder::new(Phase::Resolve, active_stratum.clone());
    active_builder.add_node(DagNode {
        id: NodeId("active_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(active_signal.clone()),
        kind: NodeKind::SignalResolve {
            signal: active_signal.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });

    let mut gated_builder = DagBuilder::new(Phase::Resolve, gated_stratum.clone());
    gated_builder.add_node(DagNode {
        id: NodeId("gated_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(gated_signal.clone()),
        kind: NodeKind::SignalResolve {
            signal: gated_signal.clone(),
            resolver_idx: 1,
            entity: None,
        },
    });

    let mut era_dags = EraDags::default();
    era_dags.insert(active_builder.build().unwrap());
    era_dags.insert(gated_builder.build().unwrap());

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    // Configure: active stratum is Active, gated stratum is Gated
    let mut strata = IndexMap::new();
    strata.insert(active_stratum, StratumState::Active);
    strata.insert(gated_stratum, StratumState::Gated);

    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    // Register resolvers
    runtime.register_resolver(Box::new(|ctx| {
        let prev = ctx.prev.as_scalar().unwrap_or(0.0);
        Value::Scalar(prev + 1.0)
    }));
    runtime.register_resolver(Box::new(|ctx| {
        let prev = ctx.prev.as_scalar().unwrap_or(0.0);
        Value::Scalar(prev + 10.0) // Gated signal would increment by 10
    }));

    runtime.init_signal(active_signal.clone(), Value::Scalar(0.0));
    runtime.init_signal(gated_signal.clone(), Value::Scalar(0.0));

    // Execute tick - only active stratum should run
    runtime.execute_tick().unwrap();

    // Active signal should have incremented
    assert_eq!(
        runtime.get_signal(&active_signal),
        Some(Value::Scalar(1.0))
    );

    // Gated signal should NOT have changed (gated stratum skipped)
    assert_eq!(runtime.get_signal(&gated_signal), Some(Value::Scalar(0.0)));
}

#[test]
fn test_parallel_level_signals() {
    // Two independent signals in the same level should both execute
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_a: SignalId = "a".into();
    let signal_b: SignalId = "b".into();

    let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    // Two nodes with no dependencies - same level
    builder.add_node(DagNode {
        id: NodeId("a_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_a.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_a.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    builder.add_node(DagNode {
        id: NodeId("b_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_b.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_b.clone(),
            resolver_idx: 1,
            entity: None,
        },
    });

    let dag = builder.build().unwrap();
    // Verify they're in the same level
    assert_eq!(dag.levels.len(), 1, "both should be in same level");
    assert_eq!(dag.levels[0].nodes.len(), 2, "two nodes in level");

    let mut era_dags = EraDags::default();
    era_dags.insert(dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    runtime.register_resolver(Box::new(|ctx| {
        Value::Scalar(ctx.prev.as_scalar().unwrap_or(0.0) + 1.0)
    }));
    runtime.register_resolver(Box::new(|ctx| {
        Value::Scalar(ctx.prev.as_scalar().unwrap_or(0.0) + 100.0)
    }));

    runtime.init_signal(signal_a.clone(), Value::Scalar(0.0));
    runtime.init_signal(signal_b.clone(), Value::Scalar(0.0));

    runtime.execute_tick().unwrap();

    // Both signals should have been resolved
    assert_eq!(runtime.get_signal(&signal_a), Some(Value::Scalar(1.0)));
    assert_eq!(runtime.get_signal(&signal_b), Some(Value::Scalar(100.0)));
}

#[test]
fn test_dependency_chain_levels() {
    // Signal chain: A -> B -> C (three levels)
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_a: SignalId = "a".into();
    let signal_b: SignalId = "b".into();
    let signal_c: SignalId = "c".into();

    let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    builder.add_node(DagNode {
        id: NodeId("a_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_a.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_a.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    builder.add_node(DagNode {
        id: NodeId("b_resolve".to_string()),
        reads: [signal_a.clone()].into_iter().collect(),
        writes: Some(signal_b.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_b.clone(),
            resolver_idx: 1,
            entity: None,
        },
    });
    builder.add_node(DagNode {
        id: NodeId("c_resolve".to_string()),
        reads: [signal_b.clone()].into_iter().collect(),
        writes: Some(signal_c.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_c.clone(),
            resolver_idx: 2,
            entity: None,
        },
    });

    let dag = builder.build().unwrap();
    // Should have 3 levels
    assert_eq!(dag.levels.len(), 3, "chain should produce 3 levels");

    let mut era_dags = EraDags::default();
    era_dags.insert(dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    // A: returns 10
    runtime.register_resolver(Box::new(|_| Value::Scalar(10.0)));
    // B: reads A and doubles it
    runtime.register_resolver(Box::new(|ctx| {
        let a = ctx.signals.get_global_or_prev("a").unwrap().as_scalar().unwrap();
        Value::Scalar(a * 2.0)
    }));
    // C: reads B and doubles it
    runtime.register_resolver(Box::new(|ctx| {
        let b = ctx.signals.get_global_or_prev("b").unwrap().as_scalar().unwrap();
        Value::Scalar(b * 2.0)
    }));

    runtime.init_signal(signal_a.clone(), Value::Scalar(0.0));
    runtime.init_signal(signal_b.clone(), Value::Scalar(0.0));
    runtime.init_signal(signal_c.clone(), Value::Scalar(0.0));

    runtime.execute_tick().unwrap();

    // Chain: A=10, B=20, C=40
    assert_eq!(runtime.get_signal(&signal_a), Some(Value::Scalar(10.0)));
    assert_eq!(runtime.get_signal(&signal_b), Some(Value::Scalar(20.0)));
    assert_eq!(runtime.get_signal(&signal_c), Some(Value::Scalar(40.0)));
}

#[test]
fn test_chronicle_event_emission() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "temperature".into();

    // Build DAG for Resolve phase
    let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    resolve_builder.add_node(DagNode {
        id: NodeId("temp_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let resolve_dag = resolve_builder.build().unwrap();

    // Build DAG for Measure phase with chronicle
    let mut measure_builder = DagBuilder::new(Phase::Measure, stratum_id.clone());
    measure_builder.add_node(DagNode {
        id: NodeId("temp_chronicle".to_string()),
        reads: [signal_id.clone()].into_iter().collect(),
        writes: None,
        kind: NodeKind::ChronicleObserve { chronicle_idx: 0 },
    });
    let measure_dag = measure_builder.build().unwrap();

    let mut era_dags = EraDags::default();
    era_dags.insert(resolve_dag);
    era_dags.insert(measure_dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    // Register resolver: temperature increments by 10 each tick
    runtime.register_resolver(Box::new(|ctx| {
        let prev = ctx.prev.as_scalar().unwrap_or(0.0);
        Value::Scalar(prev + 10.0)
    }));

    // Register chronicle: emit event when temperature > 100
    let signal_id_clone = signal_id.clone();
    runtime.register_chronicle(Box::new(move |ctx| {
        let temp = ctx
            .signals
            .get_global_or_prev(&signal_id_clone.to_string())
            .unwrap()
            .as_scalar()
            .unwrap();
        if temp > 100.0 {
            vec![EmittedEvent {
                chronicle_id: "test.chronicle".to_string(),
                name: "high_temperature".to_string(),
                fields: vec![("temp".to_string(), Value::Scalar(temp))],
            }]
        } else {
            vec![]
        }
    }));

    runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

    // Tick 1: temp = 110, should emit event
    runtime.execute_tick().unwrap();
    assert_eq!(runtime.get_signal(&signal_id), Some(Value::Scalar(110.0)));

    // Check event buffer
    assert!(!runtime.event_buffer().is_empty());
    assert_eq!(runtime.event_buffer().len(), 1);

    let events = runtime.drain_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "high_temperature");
    assert_eq!(events[0].fields.len(), 1);
    assert_eq!(events[0].fields[0].0, "temp");
    assert_eq!(events[0].fields[0].1.as_scalar(), Some(110.0));

    // After drain, buffer should be empty
    assert!(runtime.event_buffer().is_empty());

    // Tick 2: temp = 120, should emit another event
    runtime.execute_tick().unwrap();
    assert_eq!(runtime.event_buffer().len(), 1);
}

#[test]
fn test_chronicle_no_emission_when_condition_false() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "pressure".into();

    // Build DAG for Resolve phase
    let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    resolve_builder.add_node(DagNode {
        id: NodeId("pressure_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 0,
            entity: None,
        },
    });
    let resolve_dag = resolve_builder.build().unwrap();

    // Build DAG for Measure phase with chronicle
    let mut measure_builder = DagBuilder::new(Phase::Measure, stratum_id.clone());
    measure_builder.add_node(DagNode {
        id: NodeId("pressure_chronicle".to_string()),
        reads: [signal_id.clone()].into_iter().collect(),
        writes: None,
        kind: NodeKind::ChronicleObserve { chronicle_idx: 0 },
    });
    let measure_dag = measure_builder.build().unwrap();

    let mut era_dags = EraDags::default();
    era_dags.insert(resolve_dag);
    era_dags.insert(measure_dag);

    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    let era_config = EraConfig {
        dt: Dt(1.0),
        strata,
        transition: None,
    };

    let mut eras = IndexMap::new();
    eras.insert(era_id.clone(), era_config);

    let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

    // Register resolver: pressure stays constant at 50
    runtime.register_resolver(Box::new(|_ctx| Value::Scalar(50.0)));

    // Register chronicle: emit event only when pressure > 100 (never true)
    let signal_id_clone = signal_id.clone();
    runtime.register_chronicle(Box::new(move |ctx| {
        let pressure = ctx
            .signals
            .get_global_or_prev(&signal_id_clone.to_string())
            .unwrap()
            .as_scalar()
            .unwrap();
        if pressure > 100.0 {
            vec![EmittedEvent {
                chronicle_id: "test.chronicle".to_string(),
                name: "high_pressure".to_string(),
                fields: vec![],
            }]
        } else {
            vec![]
        }
    }));

    runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

    // Execute ticks - no events should be emitted
    runtime.execute_tick().unwrap();
    assert!(runtime.event_buffer().is_empty());

    runtime.execute_tick().unwrap();
    assert!(runtime.event_buffer().is_empty());

    runtime.execute_tick().unwrap();
    assert!(runtime.event_buffer().is_empty());
}
