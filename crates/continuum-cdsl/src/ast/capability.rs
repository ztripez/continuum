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
//! // Context for Resolve phase with full capabilities
//! struct ResolveContext {
//!     scoping: ScopingData,
//!     signals: SignalData,
//!     prev: Value,
//!     inputs: f64,
//!     dt: f64,
//! }
//!
//! impl HasScoping for ResolveContext { /* ... */ }
//! impl HasSignals for ResolveContext { /* ... */ }
//! impl HasPrev for ResolveContext { /* ... */ }
//! impl HasInputs for ResolveContext { /* ... */ }
//! impl HasDt for ResolveContext { /* ... */ }
//!
//! // Generic function that works with any context providing dt
//! fn time_integrate<C: HasDt>(ctx: &C, value: f64) -> f64 {
//!     value * ctx.dt()
//! }
//! ```

use crate::foundation::Path;

/// Access to config values and constants
///
/// Provides access to configuration values and constants defined in the world.
/// Available in all execution phases.
pub trait HasScoping {
    /// Look up a configuration value by path
    ///
    /// Config values are runtime-configurable parameters that can be set
    /// per scenario or run.
    fn config(&self, path: &Path) -> f64;

    /// Look up a constant value by path
    ///
    /// Constants are compile-time values that cannot change during execution.
    fn constant(&self, path: &Path) -> f64;
}

/// Access to resolved signal values
///
/// Provides read access to signal values. The specific values returned depend
/// on the phase:
/// - **Collect**: Returns previous tick values (current tick not yet resolved)
/// - **Resolve**: Signal being resolved reads `prev`; other signals return previous tick
/// - **Fracture/Measure/Assert**: Returns current tick values (just resolved)
pub trait HasSignals {
    /// Read a signal value by path
    ///
    /// Returns the signal value appropriate for the current phase.
    fn signal(&self, path: &Path) -> f64;
}

/// Access to previous tick value
///
/// Available in Resolve and Assert phases. Provides read access to the value
/// this node produced in the previous tick.
pub trait HasPrev {
    /// Get the previous tick value
    ///
    /// This is the value this node produced last tick.
    fn prev(&self) -> f64;
}

/// Access to current tick value
///
/// Available in Measure and Assert phases. Provides read access to the value
/// this node just produced this tick (after resolution).
pub trait HasCurrent {
    /// Get the current tick value
    ///
    /// This is the value this node produced this tick.
    fn current(&self) -> f64;
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
    fn inputs(&self) -> f64;
}

/// Access to delta time
///
/// Available in all tick execution phases (Collect, Resolve, Fracture, Measure, Assert).
/// Provides the time step for this tick.
pub trait HasDt {
    /// Get the time step for this tick
    ///
    /// Returns dt in world time units.
    fn dt(&self) -> f64;
}

/// Access to impulse payload
///
/// Available in Collect phase for impulses. Provides read access to the
/// payload data sent with the impulse.
pub trait HasPayload {
    /// Get the impulse payload value
    ///
    /// Returns the payload sent with this impulse event.
    fn payload(&self) -> f64;
}

/// Ability to emit values to signals
///
/// Available in Collect phase and impulse Apply blocks. Allows accumulating
/// values into signals for resolution.
pub trait CanEmit {
    /// Emit a value to a target signal
    ///
    /// Accumulates the value into the target signal's inputs. The signal will
    /// resolve these inputs during the Resolve phase.
    fn emit(&mut self, target: &Path, value: f64);
}

/// Access to entity-specific fields (per-entity index)
///
/// This capability is orthogonal to phases. It's available when the node has
/// `I = EntityId` via the `Indexed<C>` wrapper.
///
/// Provides access to other members of the same entity (e.g., `self.velocity`
/// when inside a `plate` entity context).
pub trait HasIndex {
    /// Access a field on the current entity
    ///
    /// Returns the value of another member on the same entity instance.
    /// For example, inside a `plate.force` operator, `self_field("velocity")`
    /// would return `plate.velocity` for the current plate instance.
    fn self_field(&self, name: &str) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock context for testing capability composition
    struct MockResolveContext {
        dt: f64,
        prev: f64,
        inputs: f64,
    }

    impl HasDt for MockResolveContext {
        fn dt(&self) -> f64 {
            self.dt
        }
    }

    impl HasPrev for MockResolveContext {
        fn prev(&self) -> f64 {
            self.prev
        }
    }

    impl HasInputs for MockResolveContext {
        fn inputs(&self) -> f64 {
            self.inputs
        }
    }

    impl HasScoping for MockResolveContext {
        fn config(&self, _path: &Path) -> f64 {
            1.0 // Mock value
        }

        fn constant(&self, _path: &Path) -> f64 {
            2.0 // Mock value
        }
    }

    impl HasSignals for MockResolveContext {
        fn signal(&self, _path: &Path) -> f64 {
            3.0 // Mock value
        }
    }

    #[test]
    fn test_has_scoping_trait() {
        struct TestContext;
        impl HasScoping for TestContext {
            fn config(&self, _path: &Path) -> f64 {
                42.0
            }
            fn constant(&self, _path: &Path) -> f64 {
                100.0
            }
        }

        let ctx = TestContext;
        let path = Path::from_str("test.value");
        assert_eq!(ctx.config(&path), 42.0);
        assert_eq!(ctx.constant(&path), 100.0);
    }

    #[test]
    fn test_has_signals_trait() {
        struct TestContext;
        impl HasSignals for TestContext {
            fn signal(&self, path: &Path) -> f64 {
                if path == &Path::from_str("world.temperature") {
                    300.0
                } else {
                    0.0
                }
            }
        }

        let ctx = TestContext;
        assert_eq!(ctx.signal(&Path::from_str("world.temperature")), 300.0);
        assert_eq!(ctx.signal(&Path::from_str("other.signal")), 0.0);
    }

    #[test]
    fn test_has_prev_trait() {
        struct TestContext {
            value: f64,
        }
        impl HasPrev for TestContext {
            fn prev(&self) -> f64 {
                self.value
            }
        }

        let ctx = TestContext { value: 10.5 };
        assert_eq!(ctx.prev(), 10.5);
    }

    #[test]
    fn test_has_current_trait() {
        struct TestContext {
            value: f64,
        }
        impl HasCurrent for TestContext {
            fn current(&self) -> f64 {
                self.value
            }
        }

        let ctx = TestContext { value: 20.5 };
        assert_eq!(ctx.current(), 20.5);
    }

    #[test]
    fn test_has_inputs_trait() {
        struct TestContext {
            accumulated: f64,
        }
        impl HasInputs for TestContext {
            fn inputs(&self) -> f64 {
                self.accumulated
            }
        }

        let ctx = TestContext { accumulated: 15.0 };
        assert_eq!(ctx.inputs(), 15.0);
    }

    #[test]
    fn test_has_dt_trait() {
        struct TestContext {
            time_step: f64,
        }
        impl HasDt for TestContext {
            fn dt(&self) -> f64 {
                self.time_step
            }
        }

        let ctx = TestContext { time_step: 0.016 };
        assert_eq!(ctx.dt(), 0.016);
    }

    #[test]
    fn test_has_payload_trait() {
        struct TestContext {
            data: f64,
        }
        impl HasPayload for TestContext {
            fn payload(&self) -> f64 {
                self.data
            }
        }

        let ctx = TestContext { data: 99.9 };
        assert_eq!(ctx.payload(), 99.9);
    }

    #[test]
    fn test_can_emit_trait() {
        struct TestContext {
            emissions: Vec<(Path, f64)>,
        }
        impl CanEmit for TestContext {
            fn emit(&mut self, target: &Path, value: f64) {
                self.emissions.push((target.clone(), value));
            }
        }

        let mut ctx = TestContext {
            emissions: Vec::new(),
        };
        let target1 = Path::from_str("signal.a");
        let target2 = Path::from_str("signal.b");

        ctx.emit(&target1, 10.0);
        ctx.emit(&target2, 20.0);

        assert_eq!(ctx.emissions.len(), 2);
        assert_eq!(ctx.emissions[0].0, target1);
        assert_eq!(ctx.emissions[0].1, 10.0);
        assert_eq!(ctx.emissions[1].0, target2);
        assert_eq!(ctx.emissions[1].1, 20.0);
    }

    #[test]
    fn test_has_index_trait() {
        struct TestContext;
        impl HasIndex for TestContext {
            fn self_field(&self, name: &str) -> f64 {
                match name {
                    "velocity" => 5.0,
                    "mass" => 1000.0,
                    _ => 0.0,
                }
            }
        }

        let ctx = TestContext;
        assert_eq!(ctx.self_field("velocity"), 5.0);
        assert_eq!(ctx.self_field("mass"), 1000.0);
        assert_eq!(ctx.self_field("unknown"), 0.0);
    }

    #[test]
    fn test_capability_composition() {
        let ctx = MockResolveContext {
            dt: 0.016,
            prev: 5.0,
            inputs: 2.0,
        };

        // Context implements multiple orthogonal capabilities
        assert_eq!(ctx.dt(), 0.016);
        assert_eq!(ctx.prev(), 5.0);
        assert_eq!(ctx.inputs(), 2.0);
        assert_eq!(ctx.config(&Path::from_str("test")), 1.0);
        assert_eq!(ctx.constant(&Path::from_str("test")), 2.0);
        assert_eq!(ctx.signal(&Path::from_str("test")), 3.0);
    }

    #[test]
    fn test_generic_functions_with_capabilities() {
        // Generic function using HasDt
        fn time_integrate<C: HasDt>(ctx: &C, rate: f64) -> f64 {
            rate * ctx.dt()
        }

        // Generic function using HasPrev and HasInputs
        fn compute_delta<C: HasPrev + HasInputs>(ctx: &C) -> f64 {
            ctx.inputs() - ctx.prev()
        }

        let ctx = MockResolveContext {
            dt: 0.1,
            prev: 10.0,
            inputs: 12.0,
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

        #[allow(dead_code)]
        fn needs_collect_capabilities<C: HasScoping + HasSignals + HasDt + HasPayload + CanEmit>(
            _ctx: &mut C,
        ) {
            // Function that requires Collect phase capabilities
            // Not called in test, but demonstrates trait bounds work correctly
        }

        let ctx = MockResolveContext {
            dt: 0.016,
            prev: 5.0,
            inputs: 2.0,
        };

        // Should compile with matching capabilities
        needs_resolve_capabilities(&ctx);

        // Note: needs_collect_capabilities would not compile with MockResolveContext
        // because it doesn't implement HasPayload or CanEmit (different phase)
    }
}
