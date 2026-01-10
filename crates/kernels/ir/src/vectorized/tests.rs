//! Tests for L2 vectorized kernel execution.

use std::sync::Arc;

use super::*;
use crate::ssa::lower_to_ssa;
use crate::CompiledExpr;

/// Helper to create and execute an L2 kernel.
fn execute_l2(expr: &CompiledExpr, prev_values: &[f64], dt: f64) -> Vec<f64> {
    let ssa = Arc::new(lower_to_ssa(expr));
    let executor = L2VectorizedExecutor::new(ssa);
    let signals = SignalStorage::default();
    executor
        .execute_scalar(prev_values, Dt(dt), &signals)
        .expect("L2 execution failed")
}

#[test]
fn test_l2_literal() {
    let expr = CompiledExpr::Literal(42.0);
    let prev = vec![0.0, 1.0, 2.0, 3.0];
    let result = execute_l2(&expr, &prev, 1.0);

    // Literal should return the same value for all entities
    assert_eq!(result, vec![42.0, 42.0, 42.0, 42.0]);
}

#[test]
fn test_l2_prev() {
    let expr = CompiledExpr::Prev;
    let prev = vec![10.0, 20.0, 30.0, 40.0];
    let result = execute_l2(&expr, &prev, 1.0);

    // Prev should return the previous values unchanged
    assert_eq!(result, prev);
}

#[test]
fn test_l2_dt() {
    let expr = CompiledExpr::DtRaw;
    let prev = vec![0.0, 0.0, 0.0, 0.0];
    let result = execute_l2(&expr, &prev, 0.016);

    // dt should be uniform across all entities
    for v in &result {
        assert!((v - 0.016).abs() < 1e-10);
    }
}

#[test]
fn test_l2_binary_add() {
    // prev + 1.0
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Add,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Literal(1.0)),
    };
    let prev = vec![0.0, 10.0, 20.0, 30.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![1.0, 11.0, 21.0, 31.0]);
}

#[test]
fn test_l2_binary_mul() {
    // prev * 2.0
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Mul,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Literal(2.0)),
    };
    let prev = vec![1.0, 2.0, 3.0, 4.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_l2_nested_expression() {
    // (prev + 1.0) * 2.0
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Mul,
        left: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        }),
        right: Box::new(CompiledExpr::Literal(2.0)),
    };
    let prev = vec![0.0, 1.0, 2.0, 3.0];
    let result = execute_l2(&expr, &prev, 1.0);

    // (0+1)*2=2, (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_l2_unary_neg() {
    let expr = CompiledExpr::Unary {
        op: UnaryOpIr::Neg,
        operand: Box::new(CompiledExpr::Prev),
    };
    let prev = vec![1.0, -2.0, 3.0, -4.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_l2_kernel_sqrt() {
    let expr = CompiledExpr::KernelCall {
        function: "sqrt".to_string(),
        args: vec![CompiledExpr::Prev],
    };
    let prev = vec![1.0, 4.0, 9.0, 16.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_l2_kernel_sin() {
    let expr = CompiledExpr::KernelCall {
        function: "sin".to_string(),
        args: vec![CompiledExpr::Literal(0.0)],
    };
    let prev = vec![0.0; 4];
    let result = execute_l2(&expr, &prev, 1.0);

    for v in &result {
        assert!(v.abs() < 1e-10);
    }
}

#[test]
fn test_l2_kernel_clamp() {
    let expr = CompiledExpr::KernelCall {
        function: "clamp".to_string(),
        args: vec![
            CompiledExpr::Prev,
            CompiledExpr::Literal(0.0),
            CompiledExpr::Literal(5.0),
        ],
    };
    let prev = vec![-2.0, 1.0, 3.0, 10.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![0.0, 1.0, 3.0, 5.0]);
}

#[test]
fn test_l2_kernel_lerp() {
    let expr = CompiledExpr::KernelCall {
        function: "lerp".to_string(),
        args: vec![
            CompiledExpr::Literal(0.0),
            CompiledExpr::Literal(10.0),
            CompiledExpr::Prev, // t
        ],
    };
    let prev = vec![0.0, 0.25, 0.5, 1.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 2.5).abs() < 1e-10);
    assert!((result[2] - 5.0).abs() < 1e-10);
    assert!((result[3] - 10.0).abs() < 1e-10);
}

#[test]
fn test_l2_integrate_euler() {
    // prev + rate * dt where rate = 1.0
    let expr = CompiledExpr::DtRobustCall {
        operator: DtRobustOperator::Integrate,
        args: vec![CompiledExpr::Prev, CompiledExpr::Literal(1.0)],
        method: IntegrationMethod::Euler,
    };
    let prev = vec![0.0, 10.0, 20.0, 30.0];
    let dt = 0.1;
    let result = execute_l2(&expr, &prev, dt);

    // With rate=1.0 and dt=0.1, each value increases by 0.1
    assert!((result[0] - 0.1).abs() < 1e-10);
    assert!((result[1] - 10.1).abs() < 1e-10);
    assert!((result[2] - 20.1).abs() < 1e-10);
    assert!((result[3] - 30.1).abs() < 1e-10);
}

#[test]
fn test_l2_comparison_ops() {
    // prev > 5.0
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Gt,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Literal(5.0)),
    };
    let prev = vec![3.0, 5.0, 7.0, 10.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_l2_let_binding() {
    // let x = 2.0 in x * prev
    let expr = CompiledExpr::Let {
        name: "x".to_string(),
        value: Box::new(CompiledExpr::Literal(2.0)),
        body: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Mul,
            left: Box::new(CompiledExpr::Local("x".to_string())),
            right: Box::new(CompiledExpr::Prev),
        }),
    };
    let prev = vec![1.0, 2.0, 3.0, 4.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_l2_large_population() {
    // Test with a larger population for SIMD benefits
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Add,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Literal(1.0)),
    };

    let population = 10_000;
    let prev: Vec<f64> = (0..population).map(|i| i as f64).collect();
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result.len(), population);
    for (i, &v) in result.iter().enumerate() {
        assert_eq!(v, i as f64 + 1.0);
    }
}

#[test]
fn test_l2_uniform_optimization() {
    // Expression that should remain uniform throughout
    // 2.0 * 3.0 (no prev or per-entity data)
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Mul,
        left: Box::new(CompiledExpr::Literal(2.0)),
        right: Box::new(CompiledExpr::Literal(3.0)),
    };
    let prev = vec![0.0; 100];
    let result = execute_l2(&expr, &prev, 1.0);

    // All values should be 6.0
    for v in &result {
        assert!((v - 6.0).abs() < 1e-10);
    }
}

#[test]
fn test_l2_scalar_kernel_implementation() {
    // Test the ScalarL2Kernel type
    // Use the L2VectorizedExecutor directly since ScalarL2Kernel requires full PopulationStorage
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Add,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Literal(1.0)),
    };
    let ssa = Arc::new(lower_to_ssa(&expr));
    let executor = L2VectorizedExecutor::new(ssa);

    let prev = vec![0.0, 10.0, 20.0];
    let signals = SignalStorage::default();
    let result = executor.execute_scalar(&prev, Dt(1.0), &signals).unwrap();

    assert_eq!(result, vec![1.0, 11.0, 21.0]);
}

#[test]
fn test_l2_kernel_struct_properties() {
    // Test kernel properties without full execution through PopulationStorage
    use continuum_foundation::EntityId;

    let expr = CompiledExpr::Prev;
    let ssa = Arc::new(lower_to_ssa(&expr));

    let kernel = ScalarL2Kernel::new(
        MemberSignalId::new(EntityId("test".to_string()), "signal".to_string()),
        "test.signal".to_string(),
        ssa,
        100,
    );

    assert_eq!(kernel.strategy(), LoweringStrategy::VectorKernel);
    assert_eq!(kernel.population_hint(), 100);
    assert_eq!(kernel.member_signal_id().entity_id.0, "test");
    assert_eq!(kernel.member_signal_id().signal_name, "signal");
}

#[test]
fn test_l2_decay_operator() {
    // decay(prev, half_life=1.0)
    let expr = CompiledExpr::DtRobustCall {
        operator: DtRobustOperator::Decay,
        args: vec![
            CompiledExpr::Prev,
            CompiledExpr::Literal(1.0), // half_life
        ],
        method: IntegrationMethod::Euler, // method not used for decay
    };

    let prev = vec![100.0, 100.0, 100.0, 100.0];
    let dt = 1.0; // One second
    let result = execute_l2(&expr, &prev, dt);

    // After 1 second with half_life=1.0, value should be halved
    let expected = 100.0 * (-std::f64::consts::LN_2).exp(); // ~50
    for v in &result {
        assert!((v - expected).abs() < 0.01);
    }
}

#[test]
fn test_l2_smooth_operator() {
    // smooth(prev, target=100.0, tau=1.0)
    let expr = CompiledExpr::DtRobustCall {
        operator: DtRobustOperator::Smooth,
        args: vec![
            CompiledExpr::Prev,
            CompiledExpr::Literal(100.0), // target
            CompiledExpr::Literal(1.0),   // tau
        ],
        method: IntegrationMethod::Euler,
    };

    let prev = vec![0.0, 50.0, 90.0, 100.0];
    let dt = 0.1;
    let result = execute_l2(&expr, &prev, dt);

    // Values should move toward 100
    assert!(result[0] > 0.0); // 0 moves up
    assert!(result[1] > 50.0); // 50 moves up
    assert!(result[2] > 90.0); // 90 moves up
    assert!((result[3] - 100.0).abs() < 0.01); // 100 stays at 100
}
