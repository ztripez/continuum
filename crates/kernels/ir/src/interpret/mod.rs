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
//! `ExecutionContext` for the VM. These contexts provide access to:
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

mod contexts;

#[cfg(test)]
mod tests;

use indexmap::IndexMap;

use continuum_foundation::{EraId, FieldId, SignalId, StratumId};
use continuum_runtime::executor::{
    AssertionFn, AssertionSeverity, EraConfig, FractureFn, MeasureFn, ResolverFn, TransitionFn,
};
// Import functions crate to ensure kernels are registered
use continuum_functions as _;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::{Dt, Value};
use continuum_vm::{execute, BytecodeChunk};

use crate::{
    codegen, AssertionSeverity as IrAssertionSeverity, CompiledEra, CompiledExpr, CompiledFracture,
    CompiledWorld, ValueType,
};

use contexts::{
    AssertionContext, FractureExecContext, MeasureContext, ResolverContext, SharedContextData,
    TransitionContext,
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
pub fn build_era_configs(world: &CompiledWorld) -> IndexMap<EraId, EraConfig> {
    let mut configs = IndexMap::new();

    for (era_id, era) in &world.eras {
        // Clone strata states directly - both IR and runtime use the same type from foundation
        let strata: IndexMap<_, _> = era
            .strata_states
            .iter()
            .map(|(stratum_id, state)| (StratumId(stratum_id.0.clone()), *state))
            .collect();

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
                shared: SharedContextData {
                    constants: &constants,
                    config: &config,
                    signals,
                },
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
            ValueType::Vec2 { .. } => Value::Vec2([initial_value; 2]),
            ValueType::Vec3 { .. } => Value::Vec3([initial_value; 3]),
            ValueType::Vec4 { .. } => Value::Vec4([initial_value; 4]),
            // Tensor, Grid, and Seq types are not yet fully supported in the interpreter.
            // For now, treat them as scalar placeholders.
            ValueType::Tensor { .. } | ValueType::Grid { .. } | ValueType::Seq { .. } => {
                Value::Scalar(initial_value)
            }
        }
    } else {
        Value::Scalar(initial_value)
    }
}

/// Builds a resolver function for a compiled signal.
///
/// This function handles both scalar signals (with a single `resolve` expression)
/// and vector signals (with per-component `resolve_components` expressions).
///
/// # Vector Signal Handling
///
/// For vector signals (Vec2, Vec3, Vec4), the resolver executes each component's
/// bytecode independently and assembles the results into the appropriate Value type.
/// This allows vector operations to be expressed as scalar operations at the
/// bytecode level while maintaining type safety at the signal level.
///
/// # Returns
///
/// - `Some(ResolverFn)` if the signal has a resolve expression
/// - `None` if the signal has no resolve logic (externally driven)
pub fn build_signal_resolver(
    signal: &crate::CompiledSignal,
    world: &CompiledWorld,
) -> Option<ResolverFn> {
    // Check for component-wise resolution (vector signals)
    if let Some(ref components) = signal.resolve_components {
        return Some(build_vector_resolver(
            components,
            &signal.value_type,
            world,
        ));
    }

    // Fall back to scalar resolution
    signal
        .resolve
        .as_ref()
        .map(|expr| build_resolver(expr, world, signal.uses_dt_raw))
}

/// Builds a resolver for vector signals with per-component expressions.
///
/// Each component expression is compiled to bytecode and executed independently.
/// The results are assembled into a Vec2, Vec3, or Vec4 based on the signal's type.
fn build_vector_resolver(
    components: &[CompiledExpr],
    value_type: &ValueType,
    world: &CompiledWorld,
) -> ResolverFn {
    // Pre-compile all component expressions to bytecode
    let bytecodes: Vec<BytecodeChunk> = components.iter().map(|e| codegen::compile(e)).collect();
    let constants = world.constants.clone();
    let config = world.config.clone();
    let value_type = value_type.clone();

    Box::new(move |ctx| {
        let exec_ctx = ResolverContext {
            prev: ctx.prev,
            inputs: ctx.inputs,
            dt: ctx.dt.seconds(),
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
        };

        // Execute each component's bytecode
        let results: Vec<f64> = bytecodes.iter().map(|bc| execute(bc, &exec_ctx)).collect();

        // Assemble into the appropriate Value type
        match value_type {
            ValueType::Vec2 { .. } => {
                debug_assert_eq!(results.len(), 2, "Vec2 requires exactly 2 components");
                Value::Vec2([results[0], results[1]])
            }
            ValueType::Vec3 { .. } => {
                debug_assert_eq!(results.len(), 3, "Vec3 requires exactly 3 components");
                Value::Vec3([results[0], results[1], results[2]])
            }
            ValueType::Vec4 { .. } => {
                debug_assert_eq!(results.len(), 4, "Vec4 requires exactly 4 components");
                Value::Vec4([results[0], results[1], results[2], results[3]])
            }
            _ => panic!(
                "build_vector_resolver called with non-vector type: {:?}",
                value_type
            ),
        }
    })
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
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
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
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
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
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
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
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
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
