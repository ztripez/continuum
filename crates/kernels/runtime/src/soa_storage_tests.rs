//! Tests for SoA (Struct-of-Arrays) storage.

use crate::soa_storage::{
    MemberSignalBuffer, MemberSignalRegistry, PopulationStorage, TypedBuffer, ValueType,
};
use crate::types::Value;

#[test]
fn test_aligned_buffer_basic() {
    let mut buf: TypedBuffer<f64> = TypedBuffer::new();
    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);

    assert_eq!(buf.len(), 3);
    assert_eq!(buf.get(0), Some(&1.0));
    assert_eq!(buf.get(1), Some(&2.0));
    assert_eq!(buf.get(2), Some(&3.0));
    assert_eq!(buf.get(3), None);
}

#[test]
fn test_aligned_buffer_set() {
    let mut buf: TypedBuffer<f64> = TypedBuffer::new();
    buf.push(1.0);
    buf.push(2.0);

    buf.set(1, 42.0);
    assert_eq!(buf.get(1), Some(&42.0));
}

#[test]
fn test_aligned_buffer_slice() {
    let mut buf: TypedBuffer<f64> = TypedBuffer::new();
    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);

    let slice = buf.as_slice();
    assert_eq!(slice, &[1.0, 2.0, 3.0]);
}

#[test]
fn test_aligned_buffer_vec3() {
    let mut buf: TypedBuffer<[f64; 3]> = TypedBuffer::new();
    buf.push([1.0, 2.0, 3.0]);
    buf.push([4.0, 5.0, 6.0]);

    assert_eq!(buf.get(0), Some(&[1.0, 2.0, 3.0]));
    assert_eq!(buf.get(1), Some(&[4.0, 5.0, 6.0]));
}

#[test]
fn test_member_signal_registry() {
    let mut registry = MemberSignalRegistry::new();

    let meta1 = registry.register("mass".to_string(), ValueType::scalar());
    assert_eq!(meta1.value_type, ValueType::scalar());
    assert_eq!(meta1.buffer_index, 0);

    let meta2 = registry.register("position".to_string(), ValueType::vec3());
    assert_eq!(meta2.value_type, ValueType::vec3());
    assert_eq!(meta2.buffer_index, 0); // First Vec3

    let meta3 = registry.register("velocity".to_string(), ValueType::vec3());
    assert_eq!(meta3.value_type, ValueType::vec3());
    assert_eq!(meta3.buffer_index, 1); // Second Vec3

    assert_eq!(registry.type_count(ValueType::scalar()), 1);
    assert_eq!(registry.type_count(ValueType::vec3()), 2);
}

#[test]
fn test_member_signal_buffer_basic() {
    let mut buf = MemberSignalBuffer::new();

    buf.register_signal("mass".to_string(), ValueType::scalar());
    buf.register_signal("position".to_string(), ValueType::vec3());
    buf.init_instances(3);

    // Set values
    buf.set_current("mass", 0, Value::Scalar(100.0)).unwrap();
    buf.set_current("mass", 1, Value::Scalar(200.0)).unwrap();
    buf.set_current("mass", 2, Value::Scalar(300.0)).unwrap();

    buf.set_current("position", 0, Value::Vec3([1.0, 0.0, 0.0]))
        .unwrap();
    buf.set_current("position", 1, Value::Vec3([0.0, 1.0, 0.0]))
        .unwrap();
    buf.set_current("position", 2, Value::Vec3([0.0, 0.0, 1.0]))
        .unwrap();

    // Read back
    assert_eq!(buf.get_current("mass", 0), Some(Value::Scalar(100.0)));
    assert_eq!(buf.get_current("mass", 2), Some(Value::Scalar(300.0)));
    assert_eq!(
        buf.get_current("position", 1),
        Some(Value::Vec3([0.0, 1.0, 0.0]))
    );
}

#[test]
fn test_member_signal_buffer_slices() {
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("mass".to_string(), ValueType::scalar());
    buf.init_instances(4);

    buf.set_current("mass", 0, Value::Scalar(1.0)).unwrap();
    buf.set_current("mass", 1, Value::Scalar(2.0)).unwrap();
    buf.set_current("mass", 2, Value::Scalar(3.0)).unwrap();
    buf.set_current("mass", 3, Value::Scalar(4.0)).unwrap();

    let slice = buf.scalar_slice("mass").unwrap();
    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_member_signal_buffer_tick_advance() {
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("mass".to_string(), ValueType::scalar());
    buf.init_instances(2);

    buf.set_current("mass", 0, Value::Scalar(100.0)).unwrap();
    buf.set_current("mass", 1, Value::Scalar(200.0)).unwrap();

    buf.advance_tick();

    // Previous now has old values
    assert_eq!(buf.get_previous("mass", 0), Some(Value::Scalar(100.0)));
    assert_eq!(buf.get_previous("mass", 1), Some(Value::Scalar(200.0)));

    // Set new current values
    buf.set_current("mass", 0, Value::Scalar(150.0)).unwrap();

    assert_eq!(buf.get_current("mass", 0), Some(Value::Scalar(150.0)));
    assert_eq!(buf.get_previous("mass", 0), Some(Value::Scalar(100.0)));
}

#[test]
fn test_population_storage() {
    let mut pop = PopulationStorage::new("stellar.moon".into());

    pop.register_signal("mass".to_string(), ValueType::scalar());
    pop.register_signal("position".to_string(), ValueType::vec3());

    pop.register_instance("moon_1".to_string());
    pop.register_instance("moon_2".to_string());
    pop.finalize();

    assert_eq!(pop.instance_count(), 2);
    assert_eq!(pop.instance_index("moon_1"), Some(0));
    assert_eq!(pop.instance_index("moon_2"), Some(1));

    pop.set_current("moon_1", "mass", Value::Scalar(100.0));
    pop.set_current("moon_2", "mass", Value::Scalar(200.0));

    assert_eq!(
        pop.get_current("moon_1", "mass"),
        Some(Value::Scalar(100.0))
    );
    assert_eq!(
        pop.get_current("moon_2", "mass"),
        Some(Value::Scalar(200.0))
    );
}

#[test]
fn test_population_storage_tick_advance() {
    let mut pop = PopulationStorage::new("stellar.moon".into());
    pop.register_signal("mass".to_string(), ValueType::scalar());
    pop.register_instance("moon_1".to_string());
    pop.finalize();

    pop.set_current("moon_1", "mass", Value::Scalar(100.0));
    pop.advance_tick();

    assert_eq!(
        pop.get_previous("moon_1", "mass"),
        Some(Value::Scalar(100.0))
    );

    pop.set_current("moon_1", "mass", Value::Scalar(150.0));
    assert_eq!(
        pop.get_current("moon_1", "mass"),
        Some(Value::Scalar(150.0))
    );
    assert_eq!(
        pop.get_previous("moon_1", "mass"),
        Some(Value::Scalar(100.0))
    );
}

// ========================================================================
// Double-buffered tick semantics tests
// ========================================================================

#[test]
fn test_double_buffer_tick_isolation() {
    // Verifies that writes to current buffer don't affect prev buffer
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("energy".to_string(), ValueType::scalar());
    buf.init_instances(3);

    // Write initial values to current tick
    buf.set_current("energy", 0, Value::Scalar(100.0)).unwrap();
    buf.set_current("energy", 1, Value::Scalar(200.0)).unwrap();
    buf.set_current("energy", 2, Value::Scalar(300.0)).unwrap();

    // Advance tick: current becomes previous
    buf.advance_tick();

    // Verify prev has the old values
    assert_eq!(buf.get_previous("energy", 0), Some(Value::Scalar(100.0)));
    assert_eq!(buf.get_previous("energy", 1), Some(Value::Scalar(200.0)));
    assert_eq!(buf.get_previous("energy", 2), Some(Value::Scalar(300.0)));

    // Write completely new values to current - SHOULD NOT affect prev
    buf.set_current("energy", 0, Value::Scalar(999.0)).unwrap();
    buf.set_current("energy", 1, Value::Scalar(888.0)).unwrap();
    buf.set_current("energy", 2, Value::Scalar(777.0)).unwrap();

    // Verify current has new values
    assert_eq!(buf.get_current("energy", 0), Some(Value::Scalar(999.0)));
    assert_eq!(buf.get_current("energy", 1), Some(Value::Scalar(888.0)));
    assert_eq!(buf.get_current("energy", 2), Some(Value::Scalar(777.0)));

    // CRITICAL: prev must still have original values (tick isolation)
    assert_eq!(buf.get_previous("energy", 0), Some(Value::Scalar(100.0)));
    assert_eq!(buf.get_previous("energy", 1), Some(Value::Scalar(200.0)));
    assert_eq!(buf.get_previous("energy", 2), Some(Value::Scalar(300.0)));
}

#[test]
fn test_resolver_reads_prev_writes_current() {
    // Simulates a resolver operation:
    // - Read from prev_tick (snapshot of previous state)
    // - Compute new value based on prev
    // - Write to current_tick
    // - Verify reads from prev remain stable throughout
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("population".to_string(), ValueType::scalar());
    buf.init_instances(4);

    // Set initial population values
    buf.set_current("population", 0, Value::Scalar(1000.0))
        .unwrap();
    buf.set_current("population", 1, Value::Scalar(2000.0))
        .unwrap();
    buf.set_current("population", 2, Value::Scalar(3000.0))
        .unwrap();
    buf.set_current("population", 3, Value::Scalar(4000.0))
        .unwrap();

    // Advance tick to make these the "previous" values
    buf.advance_tick();

    // Simulate resolver: read prev, compute new, write current
    // Growth rate: 10% per tick
    let growth_rate = 0.1;

    for i in 0..4 {
        let prev = buf
            .get_previous("population", i)
            .unwrap()
            .as_scalar()
            .unwrap();
        let new_value = prev * (1.0 + growth_rate);
        buf.set_current("population", i, Value::Scalar(new_value))
            .unwrap();
    }

    // Verify final state
    assert_eq!(
        buf.get_current("population", 0),
        Some(Value::Scalar(1100.0))
    );
    assert_eq!(
        buf.get_current("population", 1),
        Some(Value::Scalar(2200.0))
    );
    assert_eq!(
        buf.get_previous("population", 0),
        Some(Value::Scalar(1000.0))
    );
    assert_eq!(
        buf.get_previous("population", 1),
        Some(Value::Scalar(2000.0))
    );
}

#[test]
fn test_double_buffer_slice_isolation() {
    // Tests that slice access also maintains tick isolation
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("velocity".to_string(), ValueType::scalar());
    buf.init_instances(4);

    // Set initial velocities
    {
        let slice = buf.scalar_slice_mut("velocity").unwrap();
        slice[0] = 10.0;
        slice[1] = 20.0;
        slice[2] = 30.0;
        slice[3] = 40.0;
    }

    buf.advance_tick();

    // Verify prev slice has old values
    {
        let prev_slice = buf.prev_scalar_slice("velocity").unwrap();
        assert_eq!(prev_slice, &[10.0, 20.0, 30.0, 40.0]);
    }

    // Modify current slice
    {
        let current_slice = buf.scalar_slice_mut("velocity").unwrap();
        current_slice[0] = 100.0;
        current_slice[1] = 200.0;
        current_slice[2] = 300.0;
        current_slice[3] = 400.0;
    }

    // CRITICAL: prev slice must still have original values
    {
        let prev_slice = buf.prev_scalar_slice("velocity").unwrap();
        assert_eq!(prev_slice, &[10.0, 20.0, 30.0, 40.0]);
    }

    // Verify current slice has new values
    {
        let current_slice = buf.scalar_slice("velocity").unwrap();
        assert_eq!(current_slice, &[100.0, 200.0, 300.0, 400.0]);
    }
}

#[test]
fn test_multiple_tick_advances() {
    // Tests that multiple tick advances maintain correct state progression
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("counter".to_string(), ValueType::scalar());
    buf.init_instances(1);

    // Tick 0: counter = 1
    buf.set_current("counter", 0, Value::Scalar(1.0)).unwrap();
    buf.advance_tick();
    assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(1.0)));

    buf.set_current("counter", 0, Value::Scalar(2.0)).unwrap();
    buf.advance_tick();
    assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(2.0)));

    buf.set_current("counter", 0, Value::Scalar(3.0)).unwrap();
    buf.advance_tick();
    assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(3.0)));

    buf.set_current("counter", 0, Value::Scalar(4.0)).unwrap();

    // Verify current state
    assert_eq!(buf.get_current("counter", 0), Some(Value::Scalar(4.0)));
    assert_eq!(buf.get_previous("counter", 0), Some(Value::Scalar(3.0)));
}

#[test]
fn test_vec3_double_buffer_isolation() {
    // Tests tick isolation for Vec3 type (not just scalars)
    let mut buf = MemberSignalBuffer::new();
    buf.register_signal("position".to_string(), ValueType::vec3());
    buf.init_instances(2);

    buf.set_current("position", 0, Value::Vec3([1.0, 2.0, 3.0]))
        .unwrap();
    buf.set_current("position", 1, Value::Vec3([4.0, 5.0, 6.0]))
        .unwrap();

    buf.advance_tick();

    assert_eq!(
        buf.get_previous("position", 0),
        Some(Value::Vec3([1.0, 2.0, 3.0]))
    );
    assert_eq!(
        buf.get_previous("position", 1),
        Some(Value::Vec3([4.0, 5.0, 6.0]))
    );

    // Set new values
    buf.set_current("position", 0, Value::Vec3([10.0, 20.0, 30.0]))
        .unwrap();
    buf.set_current("position", 1, Value::Vec3([40.0, 50.0, 60.0]))
        .unwrap();

    // CRITICAL: prev must be unchanged
    assert_eq!(
        buf.get_previous("position", 0),
        Some(Value::Vec3([1.0, 2.0, 3.0]))
    );
    assert_eq!(
        buf.get_previous("position", 1),
        Some(Value::Vec3([4.0, 5.0, 6.0]))
    );

    // Current has new values
    assert_eq!(
        buf.get_current("position", 0),
        Some(Value::Vec3([10.0, 20.0, 30.0]))
    );
}
