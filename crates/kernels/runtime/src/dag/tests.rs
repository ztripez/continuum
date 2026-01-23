//! Tests for DAG construction, leveling, and barrier semantics.

use std::collections::HashSet;

use crate::reductions::ReductionOp;
use crate::types::{EntityId, Phase, SignalId, StratumId};
use crate::vectorized::MemberSignalId;

use super::builder::{AggregateBarrier, BarrierDagBuilder, DagBuilder, StratumEligibility};
use super::collection::{DagSet, EraDags};
use super::topology::topological_levels;
use super::types::{DagNode, ExecutableDag, Level, NodeId, NodeKind};
use super::verification::{verify_barrier_semantics, BarrierViolation};
use crate::types::EraId;

fn make_node(id: &str, reads: &[&str], writes: Option<&str>) -> DagNode {
    DagNode {
        id: NodeId(id.to_string()),
        reads: reads.iter().map(|s| SignalId::from(*s)).collect(),
        writes: writes.map(|s| SignalId::from(s)),
        kind: NodeKind::SignalResolve {
            signal: SignalId::from(writes.unwrap_or(id)),
            resolver_idx: 0,
        },
    }
}

#[test]
fn test_topological_levels_simple() {
    // A -> B -> C
    let nodes = vec![
        make_node("a", &[], Some("sig.a")),
        make_node("b", &["sig.a"], Some("sig.b")),
        make_node("c", &["sig.b"], Some("sig.c")),
    ];

    let levels = topological_levels(&nodes).unwrap();
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0].nodes.len(), 1);
    assert_eq!(levels[0].nodes[0].id.0, "a");
    assert_eq!(levels[1].nodes[0].id.0, "b");
    assert_eq!(levels[2].nodes[0].id.0, "c");
}

#[test]
fn test_topological_levels_parallel() {
    // A, B (parallel) -> C
    let nodes = vec![
        make_node("a", &[], Some("sig.a")),
        make_node("b", &[], Some("sig.b")),
        make_node("c", &["sig.a", "sig.b"], Some("sig.c")),
    ];

    let levels = topological_levels(&nodes).unwrap();
    assert_eq!(levels.len(), 2);
    assert_eq!(levels[0].nodes.len(), 2); // a and b parallel
    assert_eq!(levels[1].nodes.len(), 1); // c after both
}

#[test]
fn test_cycle_detection() {
    // A -> B -> A (cycle)
    let nodes = vec![
        make_node("a", &["sig.b"], Some("sig.a")),
        make_node("b", &["sig.a"], Some("sig.b")),
    ];

    let result = topological_levels(&nodes);
    assert!(result.is_err());
}

#[test]
fn test_complex_cycle_detection() {
    // A -> B -> C -> D -> B (cycle in longer chain)
    let nodes = vec![
        make_node("a", &[], Some("sig.a")),
        make_node("b", &["sig.a", "sig.d"], Some("sig.b")),
        make_node("c", &["sig.b"], Some("sig.c")),
        make_node("d", &["sig.c"], Some("sig.d")),
    ];

    let result = topological_levels(&nodes);
    assert!(result.is_err());
    let err = result.unwrap_err();
    // b, c, d should be in the cycle (a has no incoming deps from the cycle)
    assert!(err.involved_nodes.len() >= 3);
}

#[test]
fn test_diamond_no_false_positive() {
    // Diamond pattern: A -> B -> D, A -> C -> D (NOT a cycle)
    let nodes = vec![
        make_node("a", &[], Some("sig.a")),
        make_node("b", &["sig.a"], Some("sig.b")),
        make_node("c", &["sig.a"], Some("sig.c")),
        make_node("d", &["sig.b", "sig.c"], Some("sig.d")),
    ];

    let levels = topological_levels(&nodes).expect("diamond should not be a cycle");
    assert_eq!(levels.len(), 3);
    assert_eq!(levels[0].nodes.len(), 1); // a
    assert_eq!(levels[1].nodes.len(), 2); // b, c parallel
    assert_eq!(levels[2].nodes.len(), 1); // d
}

#[test]
fn test_deterministic_ordering() {
    // Same nodes added in different order should produce same levels
    let nodes1 = vec![
        make_node("z", &[], Some("sig.z")),
        make_node("a", &[], Some("sig.a")),
        make_node("m", &[], Some("sig.m")),
    ];
    let nodes2 = vec![
        make_node("a", &[], Some("sig.a")),
        make_node("m", &[], Some("sig.m")),
        make_node("z", &[], Some("sig.z")),
    ];

    let levels1 = topological_levels(&nodes1).unwrap();
    let levels2 = topological_levels(&nodes2).unwrap();

    assert_eq!(levels1.len(), levels2.len());
    let ids1: Vec<_> = levels1[0].nodes.iter().map(|n| &n.id.0).collect();
    let ids2: Vec<_> = levels2[0].nodes.iter().map(|n| &n.id.0).collect();
    assert_eq!(ids1, ids2); // Should be sorted: a, m, z
}

#[test]
fn test_empty_dag() {
    let nodes: Vec<DagNode> = vec![];
    let levels = topological_levels(&nodes).unwrap();
    assert!(levels.is_empty());
}

#[test]
fn test_single_node() {
    let nodes = vec![make_node("only", &[], Some("sig.only"))];
    let levels = topological_levels(&nodes).unwrap();
    assert_eq!(levels.len(), 1);
    assert_eq!(levels[0].nodes.len(), 1);
}

#[test]
fn test_dag_builder_creates_correct_structure() {
    let mut builder = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
    builder.add_node(make_node("a", &[], Some("sig.a")));
    builder.add_node(make_node("b", &["sig.a"], Some("sig.b")));

    let dag = builder.build().unwrap();
    assert_eq!(dag.phase, Phase::Resolve);
    assert_eq!(dag.stratum.0, "terra");
    assert_eq!(dag.levels.len(), 2);
    assert_eq!(dag.node_count(), 2);
    assert!(!dag.is_empty());
}

#[test]
fn test_era_dags_phase_iteration() {
    let mut era_dags = EraDags::default();

    // Add DAGs for different phases
    let mut collect_builder = DagBuilder::new(Phase::Collect, StratumId::from("terra"));
    collect_builder.add_node(make_node("collect_op", &[], None));
    era_dags.insert(collect_builder.build().unwrap());

    let mut resolve_builder = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
    resolve_builder.add_node(make_node("resolver", &[], Some("sig.x")));
    era_dags.insert(resolve_builder.build().unwrap());

    // Iterate by phase
    let collect_dags: Vec<_> = era_dags.for_phase(Phase::Collect).collect();
    let resolve_dags: Vec<_> = era_dags.for_phase(Phase::Resolve).collect();

    assert_eq!(collect_dags.len(), 1);
    assert_eq!(resolve_dags.len(), 1);
    assert_eq!(collect_dags[0].phase, Phase::Collect);
    assert_eq!(resolve_dags[0].phase, Phase::Resolve);
}

#[test]
fn test_dag_set_multiple_eras() {
    let mut dag_set = DagSet::default();

    // Era 1
    let mut era1 = EraDags::default();
    let builder1 = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
    era1.insert(builder1.build().unwrap());
    dag_set.insert_era(EraId::from("era1"), era1);

    // Era 2
    let mut era2 = EraDags::default();
    let builder2 = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
    era2.insert(builder2.build().unwrap());
    dag_set.insert_era(EraId::from("era2"), era2);

    assert_eq!(dag_set.era_count(), 2);
    assert!(!dag_set.is_empty());
    assert!(dag_set.get_era(&EraId::from("era1")).is_some());
    assert!(dag_set.get_era(&EraId::from("era2")).is_some());
    assert!(dag_set.get_era(&EraId::from("era3")).is_none());
}

#[test]
fn test_stratum_separation() {
    let mut era_dags = EraDags::default();

    // Same phase, different strata
    let builder_terra = DagBuilder::new(Phase::Resolve, StratumId::from("terra"));
    let builder_climate = DagBuilder::new(Phase::Resolve, StratumId::from("climate"));

    era_dags.insert(builder_terra.build().unwrap());
    era_dags.insert(builder_climate.build().unwrap());

    // Should be able to get each separately
    assert!(era_dags
        .get(Phase::Resolve, &StratumId::from("terra"))
        .is_some());
    assert!(era_dags
        .get(Phase::Resolve, &StratumId::from("climate"))
        .is_some());
    assert!(era_dags
        .get(Phase::Resolve, &StratumId::from("nonexistent"))
        .is_none());
}

#[test]
fn test_cycle_error_contains_all_cycle_nodes() {
    // Self-cycle: A depends on itself
    // This is actually impossible in our model since writes != reads for same node
    // But we can test a simple 2-node cycle
    let nodes = vec![
        make_node("x", &["sig.y"], Some("sig.x")),
        make_node("y", &["sig.x"], Some("sig.y")),
    ];

    let err = topological_levels(&nodes).unwrap_err();
    assert_eq!(err.involved_nodes.len(), 2);
    let ids: Vec<_> = err.involved_nodes.iter().map(|n| &n.0).collect();
    assert!(ids.contains(&&"x".to_string()));
    assert!(ids.contains(&&"y".to_string()));
}

// ========================================================================
// Barrier-Aware DAG Builder Tests
// ========================================================================

#[test]
fn test_barrier_dag_builder_member_signal_before_aggregate() {
    let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("physics"));

    // Add member signal resolution for person.age
    let member_signal = MemberSignalId {
        entity_id: EntityId::from("person"),
        signal_name: "age".to_string(),
    };
    builder.add_member_signal_resolve(member_signal.clone(), 0);

    // Add aggregate barrier that computes mean age
    builder.add_aggregate_barrier(AggregateBarrier {
        id: NodeId("mean_age_barrier".to_string()),
        member_signal: member_signal.clone(),
        reduction_op: ReductionOp::Mean,
        output_signal: SignalId::from("population.mean_age"),
        aggregate_idx: 0,
    });

    let dag = builder.build().expect("should build without cycles");

    // Verify: member signal is in level 0, aggregate is in level 1
    assert_eq!(dag.levels.len(), 2);
    assert_eq!(dag.levels[0].nodes.len(), 1);
    assert_eq!(dag.levels[1].nodes.len(), 1);

    // Check level 0 is the member signal
    assert!(matches!(
        dag.levels[0].nodes[0].kind,
        NodeKind::MemberSignalResolve { .. }
    ));

    // Check level 1 is the aggregate
    assert!(matches!(
        dag.levels[1].nodes[0].kind,
        NodeKind::PopulationAggregate { .. }
    ));
}

#[test]
fn test_barrier_dag_builder_signal_after_aggregate() {
    let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("physics"));

    // Add member signal resolution
    let member_signal = MemberSignalId {
        entity_id: EntityId::from("particle"),
        signal_name: "energy".to_string(),
    };
    builder.add_member_signal_resolve(member_signal.clone(), 0);

    // Add aggregate barrier
    builder.add_aggregate_barrier(AggregateBarrier {
        id: NodeId("total_energy_barrier".to_string()),
        member_signal: member_signal.clone(),
        reduction_op: ReductionOp::Sum,
        output_signal: SignalId::from("system.total_energy"),
        aggregate_idx: 0,
    });

    // Add a global signal that reads the aggregate
    builder.add_signal_resolve(
        SignalId::from("system.energy_ratio"),
        1,
        &[SignalId::from("system.total_energy")],
    );

    let dag = builder.build().expect("should build without cycles");

    // Verify: 3 levels - member signal, aggregate, dependent signal
    assert_eq!(dag.levels.len(), 3);
    assert!(matches!(
        dag.levels[0].nodes[0].kind,
        NodeKind::MemberSignalResolve { .. }
    ));
    assert!(matches!(
        dag.levels[1].nodes[0].kind,
        NodeKind::PopulationAggregate { .. }
    ));
    assert!(matches!(
        dag.levels[2].nodes[0].kind,
        NodeKind::SignalResolve { .. }
    ));
}

#[test]
fn test_barrier_dag_builder_multiple_aggregates() {
    let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("physics"));

    let member_signal = MemberSignalId {
        entity_id: EntityId::from("cell"),
        signal_name: "temperature".to_string(),
    };
    builder.add_member_signal_resolve(member_signal.clone(), 0);

    // Multiple aggregates over the same member signal
    builder.add_aggregate_barrier(AggregateBarrier {
        id: NodeId("min_temp".to_string()),
        member_signal: member_signal.clone(),
        reduction_op: ReductionOp::Min,
        output_signal: SignalId::from("grid.min_temp"),
        aggregate_idx: 0,
    });

    builder.add_aggregate_barrier(AggregateBarrier {
        id: NodeId("max_temp".to_string()),
        member_signal: member_signal.clone(),
        reduction_op: ReductionOp::Max,
        output_signal: SignalId::from("grid.max_temp"),
        aggregate_idx: 1,
    });

    let dag = builder.build().expect("should build");

    // Both aggregates should be at level 1 (they can run in parallel)
    assert_eq!(dag.levels.len(), 2);
    assert_eq!(dag.levels[0].nodes.len(), 1); // member signal
    assert_eq!(dag.levels[1].nodes.len(), 2); // both aggregates
}

#[test]
fn test_barrier_stats() {
    let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("test"));

    let ms1 = MemberSignalId {
        entity_id: EntityId::from("a"),
        signal_name: "x".to_string(),
    };
    let ms2 = MemberSignalId {
        entity_id: EntityId::from("b"),
        signal_name: "y".to_string(),
    };

    builder.add_member_signal_resolve(ms1.clone(), 0);
    builder.add_member_signal_resolve(ms2.clone(), 1);

    builder.add_aggregate_barrier(AggregateBarrier {
        id: NodeId("agg1".to_string()),
        member_signal: ms1,
        reduction_op: ReductionOp::Sum,
        output_signal: SignalId::from("out1"),
        aggregate_idx: 0,
    });

    let stats = builder.barrier_stats();
    assert_eq!(stats.member_signal_count, 2);
    assert_eq!(stats.aggregate_barrier_count, 1);
    assert_eq!(stats.total_node_count, 3);
}

// ========================================================================
// Barrier Verification Tests
// ========================================================================

#[test]
fn test_verify_barrier_semantics_valid() {
    let mut builder = BarrierDagBuilder::new(Phase::Resolve, StratumId::from("test"));

    let member_signal = MemberSignalId {
        entity_id: EntityId::from("entity"),
        signal_name: "value".to_string(),
    };
    builder.add_member_signal_resolve(member_signal.clone(), 0);

    builder.add_aggregate_barrier(AggregateBarrier {
        id: NodeId("agg".to_string()),
        member_signal,
        reduction_op: ReductionOp::Sum,
        output_signal: SignalId::from("sum"),
        aggregate_idx: 0,
    });

    builder.add_signal_resolve(SignalId::from("derived"), 0, &[SignalId::from("sum")]);

    let dag = builder.build().unwrap();
    assert!(verify_barrier_semantics(&dag).is_ok());
}

#[test]
fn test_verify_barrier_semantics_rejects_aggregate_before_member() {
    // Manually construct a DAG where aggregate is at the SAME level
    // as member signal resolve (violation: aggregate should be AFTER member)
    // The member node must appear first in the level so it's registered
    // before the aggregate is processed.
    let member_signal = MemberSignalId {
        entity_id: EntityId::from("entity"),
        signal_name: "value".to_string(),
    };
    let output_signal = SignalId::from("sum");

    // Member signal resolve - processed first within level 0
    let member_node = DagNode {
        id: NodeId("member.entity.value".to_string()),
        reads: HashSet::new(),
        writes: None,
        kind: NodeKind::MemberSignalResolve {
            member_signal: member_signal.clone(),
            kernel_idx: 0,
        },
    };

    // Aggregate at same level - WRONG: should be at a later level
    let aggregate_node = DagNode {
        id: NodeId("agg.sum".to_string()),
        reads: HashSet::new(),
        writes: Some(output_signal.clone()),
        kind: NodeKind::PopulationAggregate {
            entity_id: EntityId::from("entity"),
            member_signal: member_signal.clone(),
            reduction_op: ReductionOp::Sum,
            output_signal: output_signal.clone(),
            aggregate_idx: 0,
        },
    };

    let dag = ExecutableDag {
        phase: Phase::Resolve,
        stratum: StratumId::from("test"),
        levels: vec![Level {
            // Member first so it's registered before aggregate is checked
            nodes: vec![member_node, aggregate_node],
        }],
    };

    let result = verify_barrier_semantics(&dag);
    assert!(result.is_err());

    match result.unwrap_err() {
        BarrierViolation::AggregateBeforeMemberSignal {
            aggregate_signal,
            member_signal: err_member,
            aggregate_level,
            member_level,
        } => {
            assert_eq!(aggregate_signal, output_signal);
            assert_eq!(err_member, member_signal);
            // Both at level 0 - aggregate is not AFTER member
            assert_eq!(aggregate_level, 0);
            assert_eq!(member_level, 0);
        }
        other => panic!("Expected AggregateBeforeMemberSignal, got {:?}", other),
    }
}

#[test]
fn test_verify_barrier_semantics_rejects_signal_before_aggregate() {
    // Construct a DAG where a signal reading from aggregate
    // is at the same level as the aggregate (wrong order)
    let member_signal = MemberSignalId {
        entity_id: EntityId::from("entity"),
        signal_name: "value".to_string(),
    };
    let aggregate_signal = SignalId::from("sum");
    let derived_signal = SignalId::from("derived");

    // Level 0: Member signal resolve
    let member_node = DagNode {
        id: NodeId("member.entity.value".to_string()),
        reads: HashSet::new(),
        writes: None,
        kind: NodeKind::MemberSignalResolve {
            member_signal: member_signal.clone(),
            kernel_idx: 0,
        },
    };

    // Level 1: Both aggregate AND signal reading aggregate (WRONG - signal should be after)
    let aggregate_node = DagNode {
        id: NodeId("agg.sum".to_string()),
        reads: HashSet::new(),
        writes: Some(aggregate_signal.clone()),
        kind: NodeKind::PopulationAggregate {
            entity_id: EntityId::from("entity"),
            member_signal: member_signal.clone(),
            reduction_op: ReductionOp::Sum,
            output_signal: aggregate_signal.clone(),
            aggregate_idx: 0,
        },
    };

    let derived_node = DagNode {
        id: NodeId("sig.derived".to_string()),
        reads: {
            let mut reads = HashSet::new();
            reads.insert(aggregate_signal.clone());
            reads
        },
        writes: Some(derived_signal.clone()),
        kind: NodeKind::SignalResolve {
            signal: derived_signal.clone(),
            resolver_idx: 0,
        },
    };

    let dag = ExecutableDag {
        phase: Phase::Resolve,
        stratum: StratumId::from("test"),
        levels: vec![
            Level {
                nodes: vec![member_node],
            },
            Level {
                nodes: vec![aggregate_node, derived_node],
            },
        ],
    };

    let result = verify_barrier_semantics(&dag);
    assert!(result.is_err());

    match result.unwrap_err() {
        BarrierViolation::SignalBeforeAggregate {
            signal,
            aggregate_signal: err_agg,
            signal_level,
            aggregate_level,
        } => {
            assert_eq!(signal, derived_signal);
            assert_eq!(err_agg, aggregate_signal);
            assert_eq!(signal_level, 1);
            assert_eq!(aggregate_level, 1);
        }
        other => panic!("Expected SignalBeforeAggregate, got {:?}", other),
    }
}

// ========================================================================
// Stratum Eligibility Tests
// ========================================================================

#[test]
fn test_stratum_eligibility_active() {
    let eligibility = StratumEligibility::from_state(crate::types::StratumState::Active, 0);
    assert_eq!(eligibility, StratumEligibility::Eligible);
    assert!(eligibility.should_execute());
}

#[test]
fn test_stratum_eligibility_gated() {
    let eligibility = StratumEligibility::from_state(crate::types::StratumState::Gated, 0);
    assert_eq!(eligibility, StratumEligibility::Gated);
    assert!(!eligibility.should_execute());
}

#[test]
fn test_stratum_eligibility_stride_aligned() {
    // Stride 4, tick 8 is aligned (8 % 4 == 0)
    let eligibility =
        StratumEligibility::from_state(crate::types::StratumState::ActiveWithStride(4), 8);
    assert_eq!(eligibility, StratumEligibility::Eligible);
    assert!(eligibility.should_execute());
}

#[test]
fn test_stratum_eligibility_stride_not_aligned() {
    // Stride 4, tick 7 is not aligned (7 % 4 != 0)
    let eligibility =
        StratumEligibility::from_state(crate::types::StratumState::ActiveWithStride(4), 7);
    assert_eq!(eligibility, StratumEligibility::StrideSkipped);
    assert!(!eligibility.should_execute());
}

#[test]
fn test_stratum_eligibility_stride_zero_tick() {
    // Tick 0 should always be aligned (0 % n == 0 for any n)
    let eligibility =
        StratumEligibility::from_state(crate::types::StratumState::ActiveWithStride(10), 0);
    assert_eq!(eligibility, StratumEligibility::Eligible);
    assert!(eligibility.should_execute());
}
