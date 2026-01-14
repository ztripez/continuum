//! Tests for L2 vectorized kernel execution.

use std::sync::Arc;

use super::*;
use crate::CompiledExpr;
use crate::ssa::lower_to_ssa;

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
    let expr = CompiledExpr::Literal(42.0, None);
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
        right: Box::new(CompiledExpr::Literal(1.0, None)),
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
        right: Box::new(CompiledExpr::Literal(2.0, None)),
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
            right: Box::new(CompiledExpr::Literal(1.0, None)),
        }),
        right: Box::new(CompiledExpr::Literal(2.0, None)),
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
        namespace: "maths".to_string(),
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
        namespace: "maths".to_string(),
        function: "sin".to_string(),
        args: vec![CompiledExpr::Literal(0.0, None)],
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
        namespace: "maths".to_string(),
        function: "clamp".to_string(),
        args: vec![
            CompiledExpr::Prev,
            CompiledExpr::Literal(0.0, None),
            CompiledExpr::Literal(5.0, None),
        ],
    };
    let prev = vec![-2.0, 1.0, 3.0, 10.0];
    let result = execute_l2(&expr, &prev, 1.0);

    assert_eq!(result, vec![0.0, 1.0, 3.0, 5.0]);
}

#[test]
fn test_l2_kernel_lerp() {
    let expr = CompiledExpr::KernelCall {
        namespace: "maths".to_string(),
        function: "lerp".to_string(),
        args: vec![
            CompiledExpr::Literal(0.0, None),
            CompiledExpr::Literal(10.0, None),
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
    let expr = CompiledExpr::KernelCall {
        namespace: "dt".to_string(),
        function: "integrate".to_string(),
        args: vec![CompiledExpr::Prev, CompiledExpr::Literal(1.0, None)],
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
        right: Box::new(CompiledExpr::Literal(5.0, None)),
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
        value: Box::new(CompiledExpr::Literal(2.0, None)),
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
        right: Box::new(CompiledExpr::Literal(1.0, None)),
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
        left: Box::new(CompiledExpr::Literal(2.0, None)),
        right: Box::new(CompiledExpr::Literal(3.0, None)),
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
        right: Box::new(CompiledExpr::Literal(1.0, None)),
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
        MemberSignalId::new(EntityId::from("test"), "signal".to_string()),
        "test.signal".to_string(),
        ssa,
        100,
    );

    assert_eq!(kernel.strategy(), LoweringStrategy::VectorKernel);
    assert_eq!(kernel.population_hint(), 100);
    assert_eq!(kernel.member_signal_id().entity_id.to_string(), "test");
    assert_eq!(kernel.member_signal_id().signal_name, "signal");
}

#[test]
fn test_l2_decay_operator() {
    // decay(prev, half_life=1.0)
    let expr = CompiledExpr::KernelCall {
        namespace: "dt".to_string(),
        function: "decay".to_string(),
        args: vec![
            CompiledExpr::Prev,
            CompiledExpr::Literal(1.0, None), // half_life
        ],
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
    let expr = CompiledExpr::KernelCall {
        namespace: "dt".to_string(),
        function: "smooth".to_string(),
        args: vec![
            CompiledExpr::Prev,
            CompiledExpr::Literal(100.0, None), // target
            CompiledExpr::Literal(1.0, None),   // tau
        ],
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

// ============================================================================
// Snapshot/Next-State Semantics Tests
// ============================================================================

use continuum_runtime::ValueType;

#[test]
fn test_l2_self_field_snapshot_semantics() {
    // Test that SelfField reads from the snapshot (previous tick buffer)
    // Expression: self.velocity (reads another member signal)
    let expr = CompiledExpr::SelfField("velocity".to_string());
    let ssa = Arc::new(lower_to_ssa(&expr));
    let executor = L2VectorizedExecutor::new(ssa);

    // Create a member signal buffer with velocity values
    let mut member_buf = MemberSignalBuffer::new();
    member_buf.register_signal("velocity".to_string(), ValueType::scalar());
    member_buf.init_instances(4);

    // Set previous tick values (the snapshot)
    // We need to advance tick to make current -> previous
    {
        let current = member_buf.scalar_slice_mut("velocity").unwrap();
        current[0] = 10.0;
        current[1] = 20.0;
        current[2] = 30.0;
        current[3] = 40.0;
    }
    member_buf.advance_tick(); // Now these values are in the previous buffer

    // Set new current values (should NOT be read by snapshot semantics)
    {
        let current = member_buf.scalar_slice_mut("velocity").unwrap();
        current[0] = 100.0;
        current[1] = 200.0;
        current[2] = 300.0;
        current[3] = 400.0;
    }

    let signals = SignalStorage::default();
    let prev_values = vec![0.0; 4]; // prev for the target signal (not velocity)

    // Execute with member buffer - should read from snapshot (previous tick)
    let result = executor
        .execute_with_members(&prev_values, Dt(1.0), &signals, Some(&member_buf))
        .expect("L2 execution failed");

    // Should read the PREVIOUS tick values (10, 20, 30, 40), not current (100, 200, 300, 400)
    assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_l2_self_field_without_member_buffer() {
    // Test that SelfField returns zeros when no member buffer is provided
    let expr = CompiledExpr::SelfField("velocity".to_string());
    let ssa = Arc::new(lower_to_ssa(&expr));
    let executor = L2VectorizedExecutor::new(ssa);

    let signals = SignalStorage::default();
    let prev_values = vec![1.0, 2.0, 3.0, 4.0];

    // Execute without member buffer
    let result = executor
        .execute_scalar(&prev_values, Dt(1.0), &signals)
        .expect("L2 execution failed");

    // Should return zeros when no member buffer is provided
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_l2_self_field_missing_field() {
    // Test that SelfField returns zeros for non-existent field
    let expr = CompiledExpr::SelfField("nonexistent".to_string());
    let ssa = Arc::new(lower_to_ssa(&expr));
    let executor = L2VectorizedExecutor::new(ssa);

    // Create member buffer without the requested field
    let mut member_buf = MemberSignalBuffer::new();
    member_buf.register_signal("velocity".to_string(), ValueType::scalar());
    member_buf.init_instances(4);

    let signals = SignalStorage::default();
    let prev_values = vec![0.0; 4];

    let result = executor
        .execute_with_members(&prev_values, Dt(1.0), &signals, Some(&member_buf))
        .expect("L2 execution failed");

    // Should return zeros for missing field
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_l2_self_field_expression_with_snapshot() {
    // Test snapshot semantics in a larger expression:
    // prev + self.velocity * 0.5 (prev is our signal, self.velocity is snapshot)
    let expr = CompiledExpr::Binary {
        op: BinaryOpIr::Add,
        left: Box::new(CompiledExpr::Prev),
        right: Box::new(CompiledExpr::Binary {
            op: BinaryOpIr::Mul,
            left: Box::new(CompiledExpr::SelfField("velocity".to_string())),
            right: Box::new(CompiledExpr::Literal(0.5, None)),
        }),
    };
    let ssa = Arc::new(lower_to_ssa(&expr));
    let executor = L2VectorizedExecutor::new(ssa);

    // Create member buffer with velocity snapshot values
    let mut member_buf = MemberSignalBuffer::new();
    member_buf.register_signal("velocity".to_string(), ValueType::scalar());
    member_buf.init_instances(4);

    {
        let current = member_buf.scalar_slice_mut("velocity").unwrap();
        current[0] = 2.0;
        current[1] = 4.0;
        current[2] = 6.0;
        current[3] = 8.0;
    }
    member_buf.advance_tick(); // Snapshot now has 2, 4, 6, 8

    let signals = SignalStorage::default();
    let prev_values = vec![10.0, 20.0, 30.0, 40.0];

    let result = executor
        .execute_with_members(&prev_values, Dt(1.0), &signals, Some(&member_buf))
        .expect("L2 execution failed");

    // prev + velocity * 0.5 = [10 + 2*0.5, 20 + 4*0.5, 30 + 6*0.5, 40 + 8*0.5]
    //                       = [11, 22, 33, 44]
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}
