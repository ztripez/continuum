//! IR Interpretation and Runtime Function Building
//!
//! This module builds runtime closures from compiled IR expressions. These
//! closures capture the necessary context (constants, config, bytecode) and
//! can be invoked during simulation execution.
//!
//! # Overview
//!
//! The interpreter bridges the gap between compiled IR and runtime execution:
//!
//! 1. **Pre-compile to bytecode**: Expressions are compiled once at startup
//! 2. **Capture context**: Constants and config are cloned into closures
//! 3. **Build execution closures**: Returns boxed functions for runtime use
//!
//! # Closure Types
//!
//! Several closure types are built for different purposes:
//!
//! - [`ResolverFn`]: Computes new signal values from previous values and inputs
//! - [`MeasureFn`]: Computes field values for observation
//! - [`FractureFn`]: Evaluates fracture conditions and computes emissions
//! - [`TransitionFn`]: Evaluates era transition conditions
//! - [`AssertionFn`]: Validates signal invariants after resolution
//!
//! # Execution Contexts
//!
//! Each closure type has a corresponding context struct that implements
//! [`ExecutionContext`] for the VM. These contexts provide access to:
//!
//! - Previous signal value (`prev`)
//! - Current time step (`dt`)
//! - Signal storage
//! - Constants and config
//! - Kernel function dispatch
//!
//! # Thread Safety
//!
//! Built closures are `Send + Sync` and can be used across threads. They
//! hold owned copies of constants and config, avoiding shared state.

use std::collections::HashMap;

use indexmap::IndexMap;

use continuum_foundation::{EraId, FieldId, SignalId, StratumId};
use continuum_runtime::executor::{
    AssertionFn, AssertionSeverity, EraConfig, FractureFn, MeasureFn, ResolverFn, TransitionFn,
};
// Import functions crate to ensure kernels are registered
use continuum_functions as _;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::{Dt, StratumState, Value};
use continuum_vm::{execute, BytecodeChunk, ExecutionContext};

use crate::{
    codegen, AssertionSeverity as IrAssertionSeverity, CompiledEra, CompiledExpr, CompiledFracture,
    CompiledWorld, StratumStateIr, ValueType,
};

/// Builds runtime era configurations from a compiled world.
///
/// Creates an [`EraConfig`] for each era in the world, including:
/// - Time step (`dt`) in seconds
/// - Stratum activation states
/// - Transition function (if transitions are defined)
///
/// # Returns
///
/// A map from era IDs to their runtime configurations, suitable for use
/// by the simulation executor.
pub fn build_era_configs(world: &CompiledWorld) -> HashMap<EraId, EraConfig> {
    let mut configs = HashMap::new();

    for (era_id, era) in &world.eras {
        let mut strata = HashMap::new();
        for (stratum_id, state) in &era.strata_states {
            let runtime_state = match state {
                StratumStateIr::Active => StratumState::Active,
                StratumStateIr::ActiveWithStride(s) => StratumState::ActiveWithStride(*s),
                StratumStateIr::Gated => StratumState::Gated,
            };
            strata.insert(StratumId(stratum_id.0.clone()), runtime_state);
        }

        // Build transition function if there are transitions
        let transition = build_transition_fn(era, &world.constants, &world.config);

        configs.insert(
            EraId(era_id.0.clone()),
            EraConfig {
                dt: Dt(era.dt_seconds),
                strata,
                transition,
            },
        );
    }

    configs
}

/// Builds a transition function for evaluating era change conditions.
///
/// The returned function evaluates all transition conditions in order and
/// returns the target era ID for the first condition that evaluates to
/// non-zero.
///
/// # Returns
///
/// - `Some(TransitionFn)` if the era has transitions
/// - `None` if the era has no transitions (terminal or stuck)
fn build_transition_fn(
    era: &CompiledEra,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
) -> Option<TransitionFn> {
    if era.transitions.is_empty() {
        return None;
    }

    // Pre-compile all transition conditions to bytecode
    let transitions: Vec<_> = era
        .transitions
        .iter()
        .map(|t| (t.target_era.clone(), codegen::compile(&t.condition)))
        .collect();
    let constants = constants.clone();
    let config = config.clone();

    Some(Box::new(move |signals: &SignalStorage| {
        // Evaluate each transition condition in order
        // First matching condition wins
        for (target_era, bytecode) in &transitions {
            let ctx = TransitionContext {
                constants: &constants,
                config: &config,
                signals,
            };
            let result = execute(bytecode, &ctx);
            if result != 0.0 {
                return Some(EraId(target_era.0.clone()));
            }
        }
        None
    }))
}

/// Gets the initial scalar value for a signal from config.
///
/// Searches for a config key matching the pattern `initial_<signal_name>`
/// for the given signal path. Returns 0.0 if no matching config is found.
///
/// # Example
///
/// For signal `terra.temp`, this looks for config keys like:
/// - `terra.initial_temp`
/// - `initial_temp`
pub fn get_initial_value(world: &CompiledWorld, signal_id: &SignalId) -> f64 {
    let signal_name = &signal_id.0;
    let parts: Vec<&str> = signal_name.split('.').collect();

    // Try various config key patterns
    if parts.len() >= 2 {
        let last = parts.last().unwrap();

        // Try config.<domain>.initial_<signal>
        for (key, value) in &world.config {
            if key.ends_with(&format!("initial_{}", last)) {
                return *value;
            }
        }
    }

    // Default to 0
    0.0
}

/// Gets the initial value for a signal with proper type wrapping.
///
/// Returns a [`Value`] matching the signal's declared type (Scalar, Vec2,
/// Vec3, or Vec4), initialized from config or defaulting to zero.
pub fn get_initial_signal_value(world: &CompiledWorld, signal_id: &SignalId) -> Value {
    let initial_value = get_initial_value(world, signal_id);

    if let Some(signal) = world.signals.get(signal_id) {
        match signal.value_type {
            ValueType::Scalar { .. } => Value::Scalar(initial_value),
            ValueType::Vec2 => Value::Vec2([initial_value; 2]),
            ValueType::Vec3 => Value::Vec3([initial_value; 3]),
            ValueType::Vec4 => Value::Vec4([initial_value; 4]),
        }
    } else {
        Value::Scalar(initial_value)
    }
}

/// Builds a resolver function from a compiled expression.
///
/// The resolver computes the next value for a signal based on its previous
/// value, accumulated inputs, current time step, and other signal values.
///
/// # Closure Capture
///
/// The returned closure captures:
/// - Pre-compiled bytecode for the expression
/// - Cloned constants and config maps
///
/// # Arguments
///
/// - `expr`: The compiled resolve expression
/// - `world`: The compiled world (for constants and config)
/// - `uses_dt_raw`: Whether this signal uses raw dt (currently unused, both
///   paths use the same dt value)
///
/// # Example
///
/// ```ignore
/// let resolver = build_resolver(&signal.resolve.unwrap(), &world, signal.uses_dt_raw);
/// let new_value = resolver(&resolver_context);
/// ```
pub fn build_resolver(expr: &CompiledExpr, world: &CompiledWorld, uses_dt_raw: bool) -> ResolverFn {
    // Pre-compile to bytecode
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();
    let _ = uses_dt_raw; // Currently both paths use the same dt

    Box::new(move |ctx| {
        let exec_ctx = ResolverContext {
            prev: ctx.prev,
            inputs: ctx.inputs,
            dt: ctx.dt.seconds(),
            constants: &constants,
            config: &config,
            signals: ctx.signals,
        };
        let result = execute(&bytecode, &exec_ctx);
        Value::Scalar(result)
    })
}

/// Builds a measure function for computing field values.
///
/// Field measure functions evaluate expressions against current signal values
/// and emit results to the field buffer. They run during the Measure phase
/// and have no effect on causal simulation.
///
/// # Closure Capture
///
/// The returned closure captures:
/// - The field ID for emission
/// - Pre-compiled bytecode for the measure expression
/// - Cloned constants and config maps
///
/// # Arguments
///
/// - `field_id`: The ID of the field to emit to
/// - `expr`: The compiled measure expression
/// - `world`: The compiled world (for constants and config)
pub fn build_field_measure(
    field_id: &FieldId,
    expr: &CompiledExpr,
    world: &CompiledWorld,
) -> MeasureFn {
    let field_id = FieldId(field_id.0.clone());
    // Pre-compile to bytecode
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let exec_ctx = MeasureContext {
            dt: ctx.dt.seconds(),
            constants: &constants,
            config: &config,
            signals: ctx.signals,
        };
        let result = execute(&bytecode, &exec_ctx);
        ctx.fields.emit_scalar(field_id.clone(), result);
    })
}

/// Builds a fracture detection function.
///
/// Fractures monitor simulation state and trigger emissions when conditions
/// are met. The returned function evaluates all conditions and, if all pass,
/// returns the computed emission values.
///
/// # Condition Evaluation
///
/// All conditions must evaluate to non-zero for the fracture to trigger.
/// Conditions are evaluated in order; early exit occurs on the first zero.
///
/// # Returns
///
/// The built function returns:
/// - `Some(Vec<(SignalId, f64)>)` if all conditions pass, with emission values
/// - `None` if any condition fails
///
/// # Closure Capture
///
/// The returned closure captures:
/// - Pre-compiled bytecode for all condition expressions
/// - Pre-compiled bytecode for all emit value expressions
/// - Target signal IDs for emissions
/// - Cloned constants and config maps
pub fn build_fracture(fracture: &CompiledFracture, world: &CompiledWorld) -> FractureFn {
    // Pre-compile all conditions and emit expressions
    let conditions: Vec<BytecodeChunk> = fracture
        .conditions
        .iter()
        .map(|c| codegen::compile(c))
        .collect();
    let emits: Vec<(SignalId, BytecodeChunk)> = fracture
        .emits
        .iter()
        .map(|e| (e.target.clone(), codegen::compile(&e.value)))
        .collect();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let exec_ctx = FractureExecContext {
            dt: ctx.dt.seconds(),
            constants: &constants,
            config: &config,
            signals: ctx.signals,
        };

        // Check all conditions - all must be non-zero
        for condition in &conditions {
            let result = execute(condition, &exec_ctx);
            if result == 0.0 {
                return None;
            }
        }

        // All conditions passed - evaluate emit expressions
        let outputs: Vec<(continuum_runtime::SignalId, f64)> = emits
            .iter()
            .map(|(target, bytecode)| {
                let value = execute(bytecode, &exec_ctx);
                (continuum_runtime::SignalId(target.0.clone()), value)
            })
            .collect();

        Some(outputs)
    })
}

/// Builds an assertion function for validating signal invariants.
///
/// Assertion functions check conditions after signal resolution and return
/// a boolean indicating whether the assertion passed.
///
/// # Returns
///
/// The built function returns:
/// - `true` if the assertion condition evaluates to non-zero
/// - `false` if the condition evaluates to zero (assertion failed)
///
/// # Note
///
/// The assertion function does not handle the failure response (warn, error,
/// fatal) - that is determined by the assertion's severity and handled by
/// the caller.
pub fn build_assertion(expr: &CompiledExpr, world: &CompiledWorld) -> AssertionFn {
    // Pre-compile to bytecode
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let exec_ctx = AssertionContext {
            current: ctx.current,
            prev: ctx.prev,
            dt: ctx.dt.seconds(),
            constants: &constants,
            config: &config,
            signals: ctx.signals,
        };
        let result = execute(&bytecode, &exec_ctx);
        result != 0.0
    })
}

/// Converts IR assertion severity to runtime assertion severity.
///
/// This is a direct mapping between the IR and runtime representations
/// of assertion severity levels.
pub fn convert_assertion_severity(severity: IrAssertionSeverity) -> AssertionSeverity {
    match severity {
        IrAssertionSeverity::Warn => AssertionSeverity::Warn,
        IrAssertionSeverity::Error => AssertionSeverity::Error,
        IrAssertionSeverity::Fatal => AssertionSeverity::Fatal,
    }
}

// === Execution Contexts ===
//
// Each context implements `ExecutionContext` for the VM, providing access
// to the values needed during expression evaluation. Different phases have
// different available values (e.g., resolvers have `prev` and `inputs`,
// but measure contexts don't).

/// Execution context for signal resolution.
///
/// Provides access to previous value, accumulated inputs, time step,
/// constants, config, and other signal values.
struct ResolverContext<'a> {
    prev: &'a Value,
    inputs: f64,
    dt: f64,
    constants: &'a IndexMap<String, f64>,
    config: &'a IndexMap<String, f64>,
    signals: &'a SignalStorage,
}

impl ExecutionContext for ResolverContext<'_> {
    fn prev(&self) -> f64 {
        self.prev.as_scalar().unwrap_or(0.0)
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn inputs(&self) -> f64 {
        self.inputs
    }

    fn signal(&self, name: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.as_scalar())
            .unwrap_or(0.0)
    }

    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.component(component))
            .unwrap_or(0.0)
    }

    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).copied().unwrap_or(0.0)
    }

    fn config(&self, name: &str) -> f64 {
        self.config.get(name).copied().unwrap_or(0.0)
    }

    fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
        continuum_kernel_registry::eval(name, args, self.dt).unwrap_or_else(|| {
            tracing::warn!("unknown function '{}'", name);
            0.0
        })
    }
}

/// Execution context for assertion evaluation.
///
/// In assertions, `prev` returns the current (post-resolve) value being
/// validated, not the previous tick's value.
struct AssertionContext<'a> {
    current: &'a Value,
    #[allow(dead_code)] // May be used for future 'prev' semantics in assertions
    prev: &'a Value,
    dt: f64,
    constants: &'a IndexMap<String, f64>,
    config: &'a IndexMap<String, f64>,
    signals: &'a SignalStorage,
}

impl ExecutionContext for AssertionContext<'_> {
    fn prev(&self) -> f64 {
        // In assertions, 'prev' refers to the current (post-resolve) value being asserted
        self.current.as_scalar().unwrap_or(0.0)
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn inputs(&self) -> f64 {
        0.0 // Not used in assertions
    }

    fn signal(&self, name: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.as_scalar())
            .unwrap_or(0.0)
    }

    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.component(component))
            .unwrap_or(0.0)
    }

    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).copied().unwrap_or(0.0)
    }

    fn config(&self, name: &str) -> f64 {
        self.config.get(name).copied().unwrap_or(0.0)
    }

    fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
        continuum_kernel_registry::eval(name, args, 0.0).unwrap_or_else(|| {
            tracing::warn!("unknown function '{}'", name);
            0.0
        })
    }
}

/// Execution context for era transition evaluation.
///
/// Transitions only have access to signals, constants, and config.
/// They cannot access `prev`, `dt`, or `inputs`.
struct TransitionContext<'a> {
    constants: &'a IndexMap<String, f64>,
    config: &'a IndexMap<String, f64>,
    signals: &'a SignalStorage,
}

impl ExecutionContext for TransitionContext<'_> {
    fn prev(&self) -> f64 {
        0.0 // Not used in transitions
    }

    fn dt(&self) -> f64 {
        0.0 // Not used in transitions
    }

    fn inputs(&self) -> f64 {
        0.0 // Not used in transitions
    }

    fn signal(&self, name: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.as_scalar())
            .unwrap_or(0.0)
    }

    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.component(component))
            .unwrap_or(0.0)
    }

    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).copied().unwrap_or(0.0)
    }

    fn config(&self, name: &str) -> f64 {
        self.config.get(name).copied().unwrap_or(0.0)
    }

    fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
        continuum_kernel_registry::eval(name, args, 0.0).unwrap_or_else(|| {
            tracing::warn!("unknown function '{}'", name);
            0.0
        })
    }
}

/// Execution context for field measurement.
///
/// Measure contexts have access to signals (read-only), constants, config,
/// and the current time step. They cannot access `prev` or `inputs`.
struct MeasureContext<'a> {
    dt: f64,
    constants: &'a IndexMap<String, f64>,
    config: &'a IndexMap<String, f64>,
    signals: &'a SignalStorage,
}

impl ExecutionContext for MeasureContext<'_> {
    fn prev(&self) -> f64 {
        0.0 // Not used in measure
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn inputs(&self) -> f64 {
        0.0 // Not used in measure
    }

    fn signal(&self, name: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.as_scalar())
            .unwrap_or(0.0)
    }

    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.component(component))
            .unwrap_or(0.0)
    }

    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).copied().unwrap_or(0.0)
    }

    fn config(&self, name: &str) -> f64 {
        self.config.get(name).copied().unwrap_or(0.0)
    }

    fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
        continuum_kernel_registry::eval(name, args, self.dt).unwrap_or_else(|| {
            tracing::warn!("unknown function '{}'", name);
            0.0
        })
    }
}

/// Execution context for fracture condition and emission evaluation.
///
/// Fractures have access to signals (read-only), constants, config, and
/// the current time step. They cannot access `prev` or `inputs`.
struct FractureExecContext<'a> {
    dt: f64,
    constants: &'a IndexMap<String, f64>,
    config: &'a IndexMap<String, f64>,
    signals: &'a SignalStorage,
}

impl ExecutionContext for FractureExecContext<'_> {
    fn prev(&self) -> f64 {
        0.0 // Not used in fractures
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn inputs(&self) -> f64 {
        0.0 // Not used in fractures
    }

    fn signal(&self, name: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.as_scalar())
            .unwrap_or(0.0)
    }

    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        self.signals
            .get(&runtime_id)
            .and_then(|v| v.component(component))
            .unwrap_or(0.0)
    }

    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).copied().unwrap_or(0.0)
    }

    fn config(&self, name: &str) -> f64 {
        self.config.get(name).copied().unwrap_or(0.0)
    }

    fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
        continuum_kernel_registry::eval(name, args, self.dt).unwrap_or_else(|| {
            tracing::warn!("unknown function '{}'", name);
            0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryOpIr, CompiledEmit, CompiledTransition};

    #[test]
    fn test_build_transition_fn() {
        use continuum_runtime::storage::SignalStorage;

        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut signals = SignalStorage::default();

        // Initialize signal at 100
        signals.init(SignalId::from("temp"), Value::Scalar(100.0));

        // Create an era with a transition when temp < 50
        let era = CompiledEra {
            id: continuum_foundation::EraId::from("test"),
            is_initial: true,
            is_terminal: false,
            title: None,
            dt_seconds: 1.0,
            strata_states: IndexMap::new(),
            transitions: vec![CompiledTransition {
                target_era: continuum_foundation::EraId::from("next_era"),
                condition: CompiledExpr::Binary {
                    op: BinaryOpIr::Lt,
                    left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from(
                        "temp",
                    ))),
                    right: Box::new(CompiledExpr::Literal(50.0)),
                },
            }],
        };

        let transition_fn = build_transition_fn(&era, &constants, &config).unwrap();

        // Signal at 100, should not transition (100 < 50 is false)
        assert!(transition_fn(&signals).is_none());

        // Update signal to 30
        signals.set_current(SignalId::from("temp"), Value::Scalar(30.0));

        // Signal at 30, should transition (30 < 50 is true)
        let result = transition_fn(&signals);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "next_era");
    }

    #[test]
    fn test_build_fracture() {
        use crate::CompiledFracture;
        use continuum_runtime::executor::FractureContext;
        use continuum_runtime::storage::SignalStorage;
        use continuum_runtime::types::Dt;

        let world = CompiledWorld {
            constants: IndexMap::new(),
            config: IndexMap::new(),
            functions: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
            entities: IndexMap::new(),
        };

        // Create a fracture that triggers when temp > 100 and emits to energy
        let fracture = CompiledFracture {
            id: continuum_foundation::FractureId::from("test_fracture"),
            reads: vec![continuum_foundation::SignalId::from("temp")],
            conditions: vec![CompiledExpr::Binary {
                op: BinaryOpIr::Gt,
                left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from(
                    "temp",
                ))),
                right: Box::new(CompiledExpr::Literal(100.0)),
            }],
            emits: vec![CompiledEmit {
                target: continuum_foundation::SignalId::from("energy"),
                value: CompiledExpr::Literal(50.0),
            }],
        };

        let fracture_fn = build_fracture(&fracture, &world);

        let mut signals = SignalStorage::default();
        signals.init(SignalId::from("temp"), Value::Scalar(50.0));

        let ctx = FractureContext {
            signals: &signals,
            dt: Dt(1.0),
        };

        // Temp is 50, condition (temp > 100) is false, should not trigger
        assert!(fracture_fn(&ctx).is_none());

        // Update temp to 150
        signals.set_current(SignalId::from("temp"), Value::Scalar(150.0));
        let ctx = FractureContext {
            signals: &signals,
            dt: Dt(1.0),
        };

        // Temp is 150, condition (temp > 100) is true, should trigger
        let result = fracture_fn(&ctx);
        assert!(result.is_some());
        let emits = result.unwrap();
        assert_eq!(emits.len(), 1);
        assert_eq!(emits[0].0, SignalId::from("energy"));
        assert_eq!(emits[0].1, 50.0);
    }
}
