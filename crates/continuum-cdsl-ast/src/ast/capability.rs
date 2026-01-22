//! Capability traits for execution contexts
//!
//! These traits define orthogonal capabilities that compose to form execution
//! contexts. Each phase provides different capabilities based on what data is
//! available and what operations are allowed.
//!
//! # Architecture
//!
//! Capabilities are **orthogonal** - they compose independently based on phase
//! and role. Contexts are built by implementing the subset of capabilities
//! needed for that specific phase and role combination.
//!
//! **Capabilities are NOT a hierarchy.** They are independent traits that can
//! be mixed and matched.
//!
//! # Value Types
//!
//! All capability traits use `continuum_foundation::Value` for runtime values
//! instead of primitive types like `f64`. This preserves type information
//! through the execution pipeline and supports:
//! - Scalars, booleans, integers
//! - Vectors (Vec2, Vec3, Vec4) and quaternions
//! - Matrices (Mat2, Mat3, Mat4)
//! - Tensors and structured maps
//!
//! This design allows the DSL to support rich typed data while maintaining
//! compile-time type safety through the Type system.
//!
//! # Phase Capabilities
//!
//! Different phases provide different subsets of capabilities:
//!
//! | Phase      | Scoping | Signals | Prev | Current | Inputs | Dt | Payload | Emit |
//! |------------|---------|---------|------|---------|--------|----|---------|------|
//! | Configure  | ✓       | -       | -    | -       | -      | -  | -       | -    |
//! | Collect    | ✓       | ✓       | -    | -       | -      | ✓  | ✓       | ✓    |
//! | Resolve    | ✓       | ✓       | ✓    | -       | ✓      | ✓  | -       | -    |
//! | Fracture   | ✓       | ✓       | -    | -       | -      | ✓  | -       | -    |
//! | Measure    | ✓       | ✓       | -    | ✓       | -      | ✓  | -       | -    |
//! | Assert     | ✓       | ✓       | ✓    | ✓       | -      | ✓  | -       | -    |
//!
//! **Note:** This table shows *maximum* capabilities per phase. Each role gets a
//! subset based on `RoleSpec.phase_capabilities`.
//!
//! # Index Capability
//!
//! `HasIndex` is orthogonal to phases. It's available when `I = EntityId` via
//! `Indexed<C>` wrapper, adding access to entity-specific fields.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_foundation::Value;
//!
//! // Context for Resolve phase with full capabilities
//! struct ResolveContext {
//!     scoping: ScopingData,
//!     signals: SignalData,
//!     prev: Value,
//!     inputs: Value,
//!     dt: Value,
//! }
//!
//! impl HasScoping for ResolveContext { /* ... */ }
//! impl HasSignals for ResolveContext { /* ... */ }
//! impl HasPrev for ResolveContext { /* ... */ }
//! impl HasInputs for ResolveContext { /* ... */ }
//! impl HasDt for ResolveContext { /* ... */ }
//!
//! // Generic function that works with any context providing dt
//! fn time_integrate<C: HasDt>(ctx: &C, rate: Value) -> Value {
//!     if let (Value::Scalar(dt), Value::Scalar(r)) = (ctx.dt(), &rate) {
//!         Value::Scalar(r * dt)
//!     } else {
//!         panic!("dt and rate must be scalar")
//!     }
//! }
//! ```

use crate::foundation::Path;
use continuum_foundation::Value;

/// Access to config values and constants
///
/// Provides access to configuration values and constants defined in the world.
/// Available in all execution phases.
///
/// # Panics
///
/// All lookup methods **panic** if the path is invalid. This is intentional:
/// - Invalid paths indicate compiler bugs (name resolution should have caught them)
/// - Panics are loud, making bugs visible immediately
/// - At runtime, all paths should be pre-validated by the compiler
///
/// **Note:** Future phases will use resolved IDs (ConfigId, ConstId) instead of
/// Path, making invalid lookups unrepresentable at compile time.
pub trait HasScoping {
    /// Look up a configuration value by path
    ///
    /// Config values are runtime-configurable parameters that can be set
    /// per scenario or run.
    ///
    /// # Parameters
    ///
    /// * `path` - Hierarchical path to the config symbol
    ///
    /// # Returns
    ///
    /// Reference to the config value
    ///
    /// # Panics
    ///
    /// Panics if `path` does not refer to a valid config symbol. Invalid paths
    /// indicate a compiler bug - name resolution should have validated all paths.
    fn config(&self, path: &Path) -> &Value;

    /// Look up a constant value by path
    ///
    /// Constants are compile-time values that cannot change during execution.
    ///
    /// # Parameters
    ///
    /// * `path` - Hierarchical path to the constant symbol
    ///
    /// # Returns
    ///
    /// Reference to the constant value
    ///
    /// # Panics
    ///
    /// Panics if `path` does not refer to a valid constant symbol. Invalid paths
    /// indicate a compiler bug - name resolution should have validated all paths.
    fn constant(&self, path: &Path) -> &Value;
}

/// Access to resolved signal values
///
/// Provides read access to signal values. The specific values returned depend
/// on the phase:
/// - **Collect**: Returns previous tick values (current tick not yet resolved)
/// - **Resolve**: Signal being resolved reads `prev`; other signals return previous tick
/// - **Fracture/Measure/Assert**: Returns current tick values (just resolved)
///
/// # Panics
///
/// Lookup methods **panic** if the path is invalid. Invalid paths indicate
/// compiler bugs (name resolution should have validated all signal references).
///
/// **Note:** Future phases will use SignalId instead of Path, making invalid
/// lookups unrepresentable at compile time.
pub trait HasSignals {
    /// Read a signal value by path
    ///
    /// Returns the signal value appropriate for the current phase.
    ///
    /// # Parameters
    ///
    /// * `path` - Hierarchical path to the signal
    ///
    /// # Returns
    ///
    /// Reference to the signal value. The specific value depends on the current
    /// phase (previous tick in Collect, current tick in Measure/Assert).
    ///
    /// # Panics
    ///
    /// Panics if `path` does not refer to a valid signal. Invalid paths indicate
    /// a compiler bug - name resolution should have validated all signal references.
    fn signal(&self, path: &Path) -> &Value;
}

/// Access to previous tick value
///
/// Available in Resolve and Assert phases. Provides read access to the value
/// this node produced in the previous tick.
pub trait HasPrev {
    /// Get the previous tick value
    ///
    /// This is the value this node produced last tick.
    ///
    /// # Returns
    ///
    /// Reference to the previous tick's value
    fn prev(&self) -> &Value;
}

/// Access to current tick value
///
/// Available in Measure and Assert phases. Provides read access to the value
/// this node just produced this tick (after resolution).
pub trait HasCurrent {
    /// Get the current tick value
    ///
    /// This is the value this node produced this tick.
    ///
    /// # Returns
    ///
    /// Reference to the current tick's value
    fn current(&self) -> &Value;
}

/// Access to accumulated inputs
///
/// Available in Resolve phase for signals. Provides read access to the sum
/// of all values collected during the Collect phase.
pub trait HasInputs {
    /// Get the accumulated input value
    ///
    /// This is the sum of all `emit()` calls targeting this signal during
    /// the Collect phase.
    ///
    /// # Returns
    ///
    /// Reference to the accumulated input value
    fn inputs(&self) -> &Value;
}

/// Access to delta time
///
/// Available in all tick execution phases (Collect, Resolve, Fracture, Measure, Assert).
/// Provides the time step for this tick.
pub trait HasDt {
    /// Get the time step for this tick
    ///
    /// Returns dt in world time units as a scalar Value.
    ///
    /// # Returns
    ///
    /// Reference to the time step value (typically Value::Scalar)
    fn dt(&self) -> &Value;
}

/// Access to impulse payload
///
/// Available in Collect phase for impulses. Provides read access to the
/// payload data sent with the impulse.
pub trait HasPayload {
    /// Get the impulse payload value
    ///
    /// Returns the payload sent with this impulse event.
    ///
    /// # Returns
    ///
    /// Reference to the impulse payload value
    fn payload(&self) -> &Value;
}

/// Ability to emit values to signals
///
/// Available in Collect phase and impulse Apply blocks. Allows accumulating
/// values into signals for resolution.
///
/// # Panics
///
/// `emit` **panics** if the target path is invalid. Invalid paths indicate
/// compiler bugs (name resolution should have validated all emit targets).
///
/// **Note:** Future phases will use SignalId instead of Path.
pub trait CanEmit {
    /// Emit a value to a target signal
    ///
    /// Accumulates the value into the target signal's inputs. The signal will
    /// resolve these inputs during the Resolve phase.
    ///
    /// # Parameters
    ///
    /// * `target` - Hierarchical path to the signal receiving the value
    /// * `value` - Value to emit (owned, will be accumulated)
    ///
    /// # Returns
    ///
    /// Nothing. Emission is a write-only operation that accumulates the value
    /// into the target signal's pending state.
    ///
    /// # Panics
    ///
    /// Panics if `target` does not refer to a valid signal. Invalid targets
    /// indicate a compiler bug - name resolution should have validated all
    /// emit targets.
    fn emit(&mut self, target: &Path, value: Value);
}

/// Access to entity-specific fields (per-entity index)
///
/// This capability is orthogonal to phases. It's available when the node has
/// `I = EntityId` via the `Indexed<C>` wrapper.
///
/// Provides access to other members of the same entity (e.g., `self.velocity`
/// when inside a `plate` entity context).
///
/// # Panics
///
/// `self_field` **panics** if the field name is invalid. Invalid field names
/// indicate compiler bugs (name resolution should have validated all member
/// access).
///
/// **Note:** Future phases will use MemberId instead of string names.
pub trait HasIndex {
    /// Access a field on the current entity
    ///
    /// Returns the value of another member on the same entity instance.
    /// For example, inside a `plate.force` operator, `self_field("velocity")`
    /// would return `plate.velocity` for the current plate instance.
    ///
    /// # Parameters
    ///
    /// * `name` - Field name (member name within the current entity)
    ///
    /// # Returns
    ///
    /// Reference to the field value on the current entity instance
    ///
    /// # Panics
    ///
    /// Panics if `name` does not refer to a valid member of the current entity.
    /// Invalid names indicate a compiler bug - name resolution should have
    /// validated all member access.
    fn self_field(&self, name: &str) -> &Value;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock context for testing capability composition
    struct MockResolveContext {
        dt: Value,
        prev: Value,
        inputs: Value,
        config: Value,
        constant: Value,
        signal: Value,
    }

    impl HasDt for MockResolveContext {
        fn dt(&self) -> &Value {
            &self.dt
        }
    }

    impl HasPrev for MockResolveContext {
        fn prev(&self) -> &Value {
            &self.prev
        }
    }

    impl HasInputs for MockResolveContext {
        fn inputs(&self) -> &Value {
            &self.inputs
        }
    }

    impl HasScoping for MockResolveContext {
        fn config(&self, path: &Path) -> &Value {
            if path == &Path::from_path_str("test") {
                &self.config
            } else {
                panic!("MockResolveContext: unknown config path: {}", path)
            }
        }

        fn constant(&self, path: &Path) -> &Value {
            if path == &Path::from_path_str("test") {
                &self.constant
            } else {
                panic!("MockResolveContext: unknown constant path: {}", path)
            }
        }
    }

    impl HasSignals for MockResolveContext {
        fn signal(&self, path: &Path) -> &Value {
            if path == &Path::from_path_str("test") {
                &self.signal
            } else {
                panic!("MockResolveContext: unknown signal path: {}", path)
            }
        }
    }

    #[test]
    fn test_has_scoping_trait() {
        struct TestContext {
            config_val: Value,
            const_val: Value,
        }
        impl HasScoping for TestContext {
            fn config(&self, _path: &Path) -> &Value {
                &self.config_val
            }
            fn constant(&self, _path: &Path) -> &Value {
                &self.const_val
            }
        }

        let ctx = TestContext {
            config_val: Value::Scalar(42.0),
            const_val: Value::Scalar(100.0),
        };
        let path = Path::from_path_str("test.value");
        assert_eq!(ctx.config(&path), &Value::Scalar(42.0));
        assert_eq!(ctx.constant(&path), &Value::Scalar(100.0));
    }

    #[test]
    fn test_has_signals_trait() {
        struct TestContext {
            temperature: Value,
        }
        impl HasSignals for TestContext {
            fn signal(&self, path: &Path) -> &Value {
                // In a real implementation, unknown paths would panic
                // This test implementation only supports one known path
                if path == &Path::from_path_str("world.temperature") {
                    &self.temperature
                } else {
                    panic!("Unknown signal path: {}", path)
                }
            }
        }

        let ctx = TestContext {
            temperature: Value::Scalar(300.0),
        };
        assert_eq!(
            ctx.signal(&Path::from_path_str("world.temperature")),
            &Value::Scalar(300.0)
        );
        // Accessing unknown signal would panic (as documented)
    }

    #[test]
    fn test_has_prev_trait() {
        struct TestContext {
            value: Value,
        }
        impl HasPrev for TestContext {
            fn prev(&self) -> &Value {
                &self.value
            }
        }

        let ctx = TestContext {
            value: Value::Scalar(10.5),
        };
        assert_eq!(ctx.prev(), &Value::Scalar(10.5));
    }

    #[test]
    fn test_has_current_trait() {
        struct TestContext {
            value: Value,
        }
        impl HasCurrent for TestContext {
            fn current(&self) -> &Value {
                &self.value
            }
        }

        let ctx = TestContext {
            value: Value::Scalar(20.5),
        };
        assert_eq!(ctx.current(), &Value::Scalar(20.5));
    }

    #[test]
    fn test_has_inputs_trait() {
        struct TestContext {
            accumulated: Value,
        }
        impl HasInputs for TestContext {
            fn inputs(&self) -> &Value {
                &self.accumulated
            }
        }

        let ctx = TestContext {
            accumulated: Value::Scalar(15.0),
        };
        assert_eq!(ctx.inputs(), &Value::Scalar(15.0));
    }

    #[test]
    fn test_has_dt_trait() {
        struct TestContext {
            time_step: Value,
        }
        impl HasDt for TestContext {
            fn dt(&self) -> &Value {
                &self.time_step
            }
        }

        let ctx = TestContext {
            time_step: Value::Scalar(0.016),
        };
        assert_eq!(ctx.dt(), &Value::Scalar(0.016));
    }

    #[test]
    fn test_has_payload_trait() {
        struct TestContext {
            data: Value,
        }
        impl HasPayload for TestContext {
            fn payload(&self) -> &Value {
                &self.data
            }
        }

        let ctx = TestContext {
            data: Value::Scalar(99.9),
        };
        assert_eq!(ctx.payload(), &Value::Scalar(99.9));
    }

    #[test]
    fn test_can_emit_trait() {
        struct TestContext {
            emissions: Vec<(Path, Value)>,
        }
        impl CanEmit for TestContext {
            fn emit(&mut self, target: &Path, value: Value) {
                self.emissions.push((target.clone(), value));
            }
        }

        let mut ctx = TestContext {
            emissions: Vec::new(),
        };
        let target1 = Path::from_path_str("signal.a");
        let target2 = Path::from_path_str("signal.b");

        ctx.emit(&target1, Value::Scalar(10.0));
        ctx.emit(&target2, Value::Scalar(20.0));

        assert_eq!(ctx.emissions.len(), 2);
        assert_eq!(ctx.emissions[0].0, target1);
        assert_eq!(ctx.emissions[0].1, Value::Scalar(10.0));
        assert_eq!(ctx.emissions[1].0, target2);
        assert_eq!(ctx.emissions[1].1, Value::Scalar(20.0));
    }

    #[test]
    fn test_has_index_trait() {
        struct TestContext {
            velocity: Value,
            mass: Value,
        }
        impl HasIndex for TestContext {
            fn self_field(&self, name: &str) -> &Value {
                match name {
                    "velocity" => &self.velocity,
                    "mass" => &self.mass,
                    _ => panic!("Unknown field: {}", name),
                }
            }
        }

        let ctx = TestContext {
            velocity: Value::Scalar(5.0),
            mass: Value::Scalar(1000.0),
        };
        assert_eq!(ctx.self_field("velocity"), &Value::Scalar(5.0));
        assert_eq!(ctx.self_field("mass"), &Value::Scalar(1000.0));
        // Accessing unknown field would panic (as documented)
    }

    #[test]
    fn test_capability_composition() {
        let ctx = MockResolveContext {
            dt: Value::Scalar(0.016),
            prev: Value::Scalar(5.0),
            inputs: Value::Scalar(2.0),
            config: Value::Scalar(1.0),
            constant: Value::Scalar(2.0),
            signal: Value::Scalar(3.0),
        };

        // Context implements multiple orthogonal capabilities
        assert_eq!(ctx.dt(), &Value::Scalar(0.016));
        assert_eq!(ctx.prev(), &Value::Scalar(5.0));
        assert_eq!(ctx.inputs(), &Value::Scalar(2.0));
        assert_eq!(
            ctx.config(&Path::from_path_str("test")),
            &Value::Scalar(1.0)
        );
        assert_eq!(
            ctx.constant(&Path::from_path_str("test")),
            &Value::Scalar(2.0)
        );
        assert_eq!(
            ctx.signal(&Path::from_path_str("test")),
            &Value::Scalar(3.0)
        );
    }

    #[test]
    fn test_generic_functions_with_capabilities() {
        // Generic function using HasDt
        fn time_integrate<C: HasDt>(ctx: &C, rate: f64) -> f64 {
            if let Value::Scalar(dt) = ctx.dt() {
                rate * dt
            } else {
                panic!("dt must be scalar")
            }
        }

        // Generic function using HasPrev and HasInputs
        fn compute_delta<C: HasPrev + HasInputs>(ctx: &C) -> f64 {
            match (ctx.inputs(), ctx.prev()) {
                (Value::Scalar(i), Value::Scalar(p)) => i - p,
                _ => panic!("inputs and prev must be scalar"),
            }
        }

        let ctx = MockResolveContext {
            dt: Value::Scalar(0.1),
            prev: Value::Scalar(10.0),
            inputs: Value::Scalar(12.0),
            config: Value::Scalar(1.0),
            constant: Value::Scalar(2.0),
            signal: Value::Scalar(3.0),
        };

        assert_eq!(time_integrate(&ctx, 5.0), 0.5);
        assert_eq!(compute_delta(&ctx), 2.0);
    }

    #[test]
    fn test_trait_bounds_compile() {
        // Verify trait bounds work correctly for different capability combinations
        fn needs_resolve_capabilities<C: HasScoping + HasSignals + HasPrev + HasInputs + HasDt>(
            _ctx: &C,
        ) {
            // Function that requires full Resolve phase capabilities
        }

        let ctx = MockResolveContext {
            dt: Value::Scalar(0.016),
            prev: Value::Scalar(5.0),
            inputs: Value::Scalar(2.0),
            config: Value::Scalar(1.0),
            constant: Value::Scalar(2.0),
            signal: Value::Scalar(3.0),
        };

        // Should compile with matching capabilities
        needs_resolve_capabilities(&ctx);

        // Capability enforcement is compile-time:
        // A function requiring Collect capabilities (HasPayload + CanEmit)
        // would NOT accept MockResolveContext (it only has Resolve capabilities)
    }
}
