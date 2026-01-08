//! IR Interpretation
//!
//! Compiles expressions to bytecode and executes them at runtime.

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

/// Build era configurations from compiled world
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

/// Build a transition function for an era
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

/// Get initial value for a signal from config
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

/// Get initial value for a signal based on its type
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

/// Build a resolver function from a compiled expression
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

/// Build a measure function for a field
///
/// Field measure expressions evaluate against current signal values and emit to the field buffer.
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

/// Build a fracture detection function
///
/// Fractures check conditions and emit to signals when triggered.
/// Returns `Some(emits)` if all conditions pass, `None` otherwise.
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

/// Build an assertion function from a compiled expression
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

/// Convert IR assertion severity to runtime severity
pub fn convert_assertion_severity(severity: IrAssertionSeverity) -> AssertionSeverity {
    match severity {
        IrAssertionSeverity::Warn => AssertionSeverity::Warn,
        IrAssertionSeverity::Error => AssertionSeverity::Error,
        IrAssertionSeverity::Fatal => AssertionSeverity::Fatal,
    }
}

// === Execution Contexts ===

/// Context for resolver execution
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

/// Context for assertion execution
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

/// Context for transition execution
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

/// Context for measure execution
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

/// Context for fracture execution
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
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
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
