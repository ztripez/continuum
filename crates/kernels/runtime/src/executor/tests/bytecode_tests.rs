use indexmap::IndexMap;
use std::collections::HashSet;

use crate::bytecode::Compiler;
use crate::dag::{DagBuilder, DagNode, DagSet, EraDags, NodeId, NodeKind};
use crate::executor::runtime::{EraConfig, Runtime};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, Value};
use crate::WorldPolicy;
use continuum_cdsl::ast::{Execution, ExecutionBody, ExprKind, Stmt, TypedExpr};
use continuum_cdsl::foundation::{Shape, Span, Type, Unit};
use continuum_foundation::Path;
use continuum_functions as _;
use continuum_kernel_types::KernelId;

fn make_span() -> Span {
    Span::new(0, 0, 0, 0)
}

fn make_scalar_literal(value: f64) -> TypedExpr {
    TypedExpr::new(
        ExprKind::Literal { value, unit: None },
        Type::kernel(Shape::Scalar, Unit::seconds(), None),
        make_span(),
    )
}

#[test]
fn test_bytecode_integration_resolve_simple() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "counter".into();

    // 1. Compile a resolve block: prev + 1.0
    let mut compiler = Compiler::new();
    let execution = Execution {
        name: "resolve".to_string(),
        phase: Phase::Resolve,
        body: ExecutionBody::Expr(TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Prev,
                        Type::kernel(Shape::Scalar, Unit::seconds(), None),
                        make_span(),
                    ),
                    make_scalar_literal(1.0),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            make_span(),
        )),
        reads: vec![],
        temporal_reads: vec![Path::from("counter")],
        emits: vec![],
        span: make_span(),
    };

    let compiled = compiler.compile_execution(&execution).unwrap();
    let blocks = vec![compiled];

    // 2. Setup DAG
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
    let mut era_dags = EraDags::default();
    era_dags.insert(dag);
    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    // 3. Setup Runtime
    let mut eras = IndexMap::new();
    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    eras.insert(
        era_id.clone(),
        EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        },
    );

    let mut runtime = Runtime::new(era_id, eras, dags, blocks, WorldPolicy::default());
    runtime.init_signal(signal_id.clone(), Value::Scalar(10.0));

    // 4. Execute tick
    runtime.execute_tick().unwrap();

    // 5. Verify result: 10.0 + 1.0 = 11.0
    let val = runtime.get_signal(&signal_id).unwrap();
    assert_eq!(val.as_scalar(), Some(11.0));
}

#[test]
fn test_bytecode_integration_collect_emit() {
    let era_id: EraId = "test".into();
    let stratum_id: StratumId = "default".into();
    let signal_id: SignalId = "accumulator".into();

    // 1. Compile a collect block: emit(accumulator, 5.0)
    let mut compiler = Compiler::new();
    let execution = Execution {
        name: "collect".to_string(),
        phase: Phase::Collect,
        body: ExecutionBody::Statements(vec![Stmt::SignalAssign {
            target: Path::from("accumulator"),
            value: make_scalar_literal(5.0),
            span: make_span(),
        }]),
        reads: vec![],
        temporal_reads: vec![],
        emits: vec![Path::from("accumulator")],
        span: make_span(),
    };

    let compiled = compiler.compile_execution(&execution).unwrap();

    // 2. Compile a resolve block: prev + inputs
    let resolve_execution = Execution {
        name: "resolve".to_string(),
        phase: Phase::Resolve,
        body: ExecutionBody::Expr(TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Prev,
                        Type::kernel(Shape::Scalar, Unit::seconds(), None),
                        make_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Inputs,
                        Type::kernel(Shape::Scalar, Unit::seconds(), None),
                        make_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            make_span(),
        )),
        reads: vec![],
        temporal_reads: vec![Path::from("accumulator")],
        emits: vec![],
        span: make_span(),
    };
    let compiled_resolve = compiler.compile_execution(&resolve_execution).unwrap();

    let blocks = vec![compiled, compiled_resolve];

    // 3. Setup DAG
    let mut collect_builder = DagBuilder::new(Phase::Collect, stratum_id.clone());
    collect_builder.add_node(DagNode {
        id: NodeId("acc_collect".to_string()),
        reads: [signal_id.clone()].into_iter().collect(),
        writes: None,
        kind: NodeKind::OperatorCollect { operator_idx: 0 },
    });
    let collect_dag = collect_builder.build().unwrap();

    let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
    resolve_builder.add_node(DagNode {
        id: NodeId("acc_resolve".to_string()),
        reads: HashSet::new(),
        writes: Some(signal_id.clone()),
        kind: NodeKind::SignalResolve {
            signal: signal_id.clone(),
            resolver_idx: 1,
            entity: None,
        },
    });
    let resolve_dag = resolve_builder.build().unwrap();

    let mut era_dags = EraDags::default();
    era_dags.insert(collect_dag);
    era_dags.insert(resolve_dag);
    let mut dags = DagSet::default();
    dags.insert_era(era_id.clone(), era_dags);

    // 4. Setup Runtime
    let mut eras = IndexMap::new();
    let mut strata = IndexMap::new();
    strata.insert(stratum_id, StratumState::Active);
    eras.insert(
        era_id.clone(),
        EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        },
    );

    let mut runtime = Runtime::new(era_id, eras, dags, blocks, WorldPolicy::default());
    runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

    // 5. Execute tick
    runtime.execute_tick().unwrap();

    // 6. Verify result: 100.0 + 5.0 = 105.0
    let val = runtime.get_signal(&signal_id).unwrap();
    assert_eq!(val.as_scalar(), Some(105.0));
}
