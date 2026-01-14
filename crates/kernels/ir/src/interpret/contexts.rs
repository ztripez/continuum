//! Execution contexts for VM evaluation.
//!
//! Each context implements `ExecutionContext` for the VM, providing access
//! to the values needed during expression evaluation. Different phases have
//! different available values (e.g., resolvers have `prev` and `inputs`,
//! but measure contexts don't).

use indexmap::IndexMap;

use continuum_foundation::SignalId;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::Value;
use continuum_vm::ExecutionContext;

fn split_kernel_name(name: &str) -> (&str, &str) {
    name.split_once('.')
        .unwrap_or_else(|| panic!("Kernel call '{}' is missing a namespace", name))
}

/// Shared data available to all execution contexts.
///
/// This struct holds the common data needed across all phases:
/// signals, constants, and config values. Phase-specific contexts
/// wrap this with additional phase-specific data.
pub(crate) struct SharedContextData<'a> {
    pub(crate) constants: &'a IndexMap<String, (f64, Option<crate::units::Unit>)>,
    pub(crate) config: &'a IndexMap<String, (f64, Option<crate::units::Unit>)>,
    pub(crate) signals: &'a SignalStorage,
}

impl SharedContextData<'_> {
    /// Get signal value by name
    pub(crate) fn signal(&self, name: &str) -> Value {
        let runtime_id = SignalId::from(name);
        match self.signals.get(&runtime_id) {
            Some(v) => v.clone(),
            None => panic!("Signal '{}' not found in storage", name),
        }
    }

    /// Get signal component by name and component (x, y, z, w)
    pub(crate) fn signal_component(&self, name: &str, component: &str) -> Value {
        let runtime_id = SignalId::from(name);
        match self.signals.get(&runtime_id) {
            Some(v) => {
                if let Some(c) = v.component(component) {
                    Value::Scalar(c)
                } else {
                    Value::Scalar(0.0) // Or panic?
                }
            }
            None => panic!("Signal '{}' not found in storage", name),
        }
    }

    /// Get constant value by name
    pub(crate) fn constant(&self, name: &str) -> Value {
        self.constants
            .get(name)
            .map(|(v, _)| Value::Scalar(*v))
            .unwrap_or_else(|| panic!("Constant '{}' not defined", name))
    }

    /// Get config value by name
    pub(crate) fn config(&self, name: &str) -> Value {
        self.config
            .get(name)
            .map(|(v, _)| Value::Scalar(*v))
            .unwrap_or_else(|| panic!("Config value '{}' not defined", name))
    }
}

/// Execution context for signal resolution.
///
/// Provides access to previous value, accumulated inputs, time step,
/// constants, config, and other signal values.
pub(crate) struct ResolverContext<'a> {
    pub(crate) prev: &'a Value,
    pub(crate) inputs: f64,
    pub(crate) dt: f64,
    pub(crate) sim_time: f64,
    pub(crate) shared: SharedContextData<'a>,
}

impl ExecutionContext for ResolverContext<'_> {
    fn prev(&self) -> Value {
        self.prev.clone()
    }

    fn dt_scalar(&self) -> f64 {
        self.dt
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(self.inputs)
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn signal_component(&self, name: &str, component: &str) -> Value {
        self.shared.signal_component(name, component)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = split_kernel_name(name);
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, self.dt)
            .unwrap_or_else(|| {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            })
    }
}

/// Execution context for assertion evaluation.
///
/// In assertions, `prev` returns the current (post-resolve) value being
/// validated, not the previous tick's value.
pub(crate) struct AssertionContext<'a> {
    pub(crate) current: &'a Value,
    #[allow(dead_code)] // May be used for future 'prev' semantics in assertions
    pub(crate) prev: &'a Value,
    pub(crate) dt: f64,
    pub(crate) sim_time: f64,
    pub(crate) shared: SharedContextData<'a>,
}

impl ExecutionContext for AssertionContext<'_> {
    fn prev(&self) -> Value {
        // In assertions, 'prev' refers to the current (post-resolve) value being asserted
        self.current.clone()
    }

    fn dt_scalar(&self) -> f64 {
        self.dt
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(0.0) // Not used in assertions
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn signal_component(&self, name: &str, component: &str) -> Value {
        self.shared.signal_component(name, component)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = split_kernel_name(name);
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, 0.0).unwrap_or_else(
            || {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            },
        )
    }
}

/// Execution context for era transition evaluation.
///
/// Transitions only have access to signals, constants, and config.
/// They cannot access `prev`, `dt`, or `inputs`.
pub(crate) struct TransitionContext<'a> {
    pub(crate) sim_time: f64,
    pub(crate) shared: SharedContextData<'a>,
}

impl ExecutionContext for TransitionContext<'_> {
    fn prev(&self) -> Value {
        Value::Scalar(0.0) // Not used in transitions
    }

    fn dt_scalar(&self) -> f64 {
        0.0 // Not used in transitions
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(0.0) // Not used in transitions
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn signal_component(&self, name: &str, component: &str) -> Value {
        self.shared.signal_component(name, component)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = split_kernel_name(name);
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, 0.0).unwrap_or_else(
            || {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            },
        )
    }
}

/// Execution context for field measurement.
///
/// Measure contexts have access to signals (read-only), constants, config,
/// and the current time step. They cannot access `prev` or `inputs`.
pub(crate) struct MeasureContext<'a> {
    pub(crate) dt: f64,
    pub(crate) sim_time: f64,
    pub(crate) shared: SharedContextData<'a>,
}

impl ExecutionContext for MeasureContext<'_> {
    fn prev(&self) -> Value {
        Value::Scalar(0.0) // Not used in measure
    }

    fn dt_scalar(&self) -> f64 {
        self.dt
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(0.0) // Not used in measure
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn signal_component(&self, name: &str, component: &str) -> Value {
        self.shared.signal_component(name, component)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = split_kernel_name(name);
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, self.dt)
            .unwrap_or_else(|| {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            })
    }
}

/// Execution context for fracture condition and emission evaluation.
///
/// Fractures have access to signals (read-only), constants, config, and
/// the current time step. They cannot access `prev` or `inputs`.
pub(crate) struct FractureExecContext<'a> {
    pub(crate) dt: f64,
    pub(crate) sim_time: f64,
    pub(crate) shared: SharedContextData<'a>,
}

impl ExecutionContext for FractureExecContext<'_> {
    fn prev(&self) -> Value {
        Value::Scalar(0.0) // Not used in fractures
    }

    fn dt_scalar(&self) -> f64 {
        self.dt
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(0.0) // Not used in fractures
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn signal_component(&self, name: &str, component: &str) -> Value {
        self.shared.signal_component(name, component)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = split_kernel_name(name);
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, self.dt)
            .unwrap_or_else(|| {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            })
    }
}

/// Execution context for warmup iterations.
///
/// Provides access to current warmup value as 'prev' and other signals.
pub(crate) struct WarmupContext<'a> {
    pub(crate) current: &'a Value,
    pub(crate) sim_time: f64,
    pub(crate) shared: SharedContextData<'a>,
}

impl ExecutionContext for WarmupContext<'_> {
    fn prev(&self) -> Value {
        self.current.clone()
    }

    fn dt_scalar(&self) -> f64 {
        0.0 // dt is not available during warmup
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(0.0) // inputs are not available during warmup
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn signal_component(&self, name: &str, component: &str) -> Value {
        self.shared.signal_component(name, component)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = split_kernel_name(name);
        // Kernels that depend on dt might behave unexpectedly here
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, 0.0).unwrap_or_else(
            || {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            },
        )
    }
}
