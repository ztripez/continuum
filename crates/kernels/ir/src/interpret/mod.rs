//! IR Interpretation and Runtime Function Building
//!
//! This module builds runtime closures from compiled IR expressions. These
//! closures capture the necessary context (constants, config, bytecode) and
//! can be invoked during simulation execution.

mod contexts;
mod member_interp;

#[cfg(test)]
mod tests;

// Re-export member interpreter types
pub use member_interp::{
    InterpValue, MemberInterpContext, MemberResolverFn, Vec3MemberResolverFn,
    build_member_resolver, build_vec3_member_resolver, interpret_expr,
};

use indexmap::IndexMap;
use std::collections::HashMap;

use continuum_foundation::{EraId, FieldId, SignalId};
use continuum_runtime::executor::{
    AggregateResolverFn, AssertionFn, AssertionSeverity, EraConfig, FractureFn, MeasureFn,
    ResolverFn, TransitionFn,
};
// Import functions crate to ensure kernels are registered
use continuum_functions as _;
use continuum_runtime::soa_storage::MemberSignalBuffer;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::{Dt, Value};
use continuum_vm::{BytecodeChunk, execute};

use crate::{
    AssertionSeverity as IrAssertionSeverity, CompiledEra, CompiledExpr, CompiledFracture,
    CompiledWorld, ValueType, codegen,
};

use contexts::{
    AssertionContext, FractureExecContext, MeasureContext, ResolverContext, SharedContextData,
    TransitionContext,
};

/// Checks if an expression contains any entity-related constructs.
fn contains_entity_expression(expr: &CompiledExpr) -> bool {
    match expr {
        CompiledExpr::SelfField(_)
        | CompiledExpr::EntityAccess { .. }
        | CompiledExpr::Aggregate { .. }
        | CompiledExpr::Other { .. }
        | CompiledExpr::Pairs { .. }
        | CompiledExpr::Filter { .. }
        | CompiledExpr::First { .. }
        | CompiledExpr::Nearest { .. }
        | CompiledExpr::Within { .. } => true,

        CompiledExpr::Payload | CompiledExpr::PayloadField(_) | CompiledExpr::EmitSignal { .. } => {
            true
        }

        CompiledExpr::Binary { left, right, .. } => {
            contains_entity_expression(left) || contains_entity_expression(right)
        }
        CompiledExpr::Unary { operand, .. } => contains_entity_expression(operand),
        CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
            args.iter().any(contains_entity_expression)
        }
        CompiledExpr::DtRobustCall { args, .. } => args.iter().any(contains_entity_expression),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            contains_entity_expression(condition)
                || contains_entity_expression(then_branch)
                || contains_entity_expression(else_branch)
        }
        CompiledExpr::Let { value, body, .. } => {
            contains_entity_expression(value) || contains_entity_expression(body)
        }
        CompiledExpr::FieldAccess { object, .. } => contains_entity_expression(object),

        CompiledExpr::Literal(..)
        | CompiledExpr::Prev
        | CompiledExpr::DtRaw
        | CompiledExpr::SimTime
        | CompiledExpr::Collected
        | CompiledExpr::Signal(_)
        | CompiledExpr::Const(_)
        | CompiledExpr::Config(_)
        | CompiledExpr::Local(_) => false,
    }
}

/// Evaluates an initial expression for a member signal.
pub fn eval_initial_expr(
    expr: &CompiledExpr,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
) -> Value {
    let empty_signals = SignalStorage::default();
    let empty_members = MemberSignalBuffer::new();

    let mut ctx = MemberInterpContext {
        prev: InterpValue::Scalar(0.0),
        index: 0,
        dt: 0.0,
        sim_time: 0.0,
        signals: &empty_signals,
        members: &empty_members,
        constants,
        config,
        locals: HashMap::new(),
        entity_prefix: String::new(),
        read_current: false,
    };

    let result = interpret_expr(expr, &mut ctx);

    match result {
        InterpValue::Scalar(v) => Value::Scalar(v),
        InterpValue::Vec3(v) => Value::Vec3(v),
    }
}

/// Builds runtime era configurations from a compiled world.
pub fn build_era_configs(world: &CompiledWorld) -> IndexMap<EraId, EraConfig> {
    let mut configs = IndexMap::new();

    let eras = world.eras();
    for (era_id, era) in &eras {
        let strata: IndexMap<_, _> = era
            .strata_states
            .iter()
            .map(|(stratum_id, state)| (stratum_id.clone(), *state))
            .collect();

        let transition = build_transition_fn(era, &world.constants, &world.config);

        configs.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(era.dt_seconds),
                strata,
                transition,
            },
        );
    }

    configs
}

fn build_transition_fn(
    era: &CompiledEra,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
) -> Option<TransitionFn> {
    if era.transitions.is_empty() {
        return None;
    }

    let transitions: Vec<_> = era
        .transitions
        .iter()
        .map(|t| (t.target_era.clone(), codegen::compile(&t.condition)))
        .collect();
    let constants = constants.clone();
    let config = config.clone();

    Some(Box::new(move |signals: &SignalStorage, sim_time: f64| {
        for (target_era, bytecode) in &transitions {
            let ctx = TransitionContext {
                sim_time,
                shared: SharedContextData {
                    constants: &constants,
                    config: &config,
                    signals,
                },
            };

            let result = execute(bytecode, &ctx);
            if result != 0.0 {
                return Some(target_era.clone());
            }
        }
        None
    }))
}

/// Builds a measure function for computing field values.
pub fn build_field_measure(
    field_id: &FieldId,
    expr: &CompiledExpr,
    world: &CompiledWorld,
) -> Option<MeasureFn> {
    if contains_entity_expression(expr) {
        return None;
    }

    let field_id = field_id.clone();
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();

    Some(Box::new(move |ctx| {
        let exec_ctx = MeasureContext {
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
        };
        let result = execute(&bytecode, &exec_ctx);
        ctx.fields.emit_scalar(field_id.clone(), result);
    }))
}

/// Builds a fracture detection function.
pub fn build_fracture(fracture: &CompiledFracture, world: &CompiledWorld) -> FractureFn {
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
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
        };

        for condition in &conditions {
            let result = execute(condition, &exec_ctx);
            if result == 0.0 {
                return None;
            }
        }

        let outputs: Vec<(SignalId, f64)> = emits
            .iter()
            .map(|(target, bytecode)| {
                let value = execute(bytecode, &exec_ctx);
                (target.clone(), value)
            })
            .collect();

        Some(outputs)
    })
}

/// Builds an assertion function for validating signal invariants.
pub fn build_assertion(expr: &CompiledExpr, world: &CompiledWorld) -> AssertionFn {
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let exec_ctx = AssertionContext {
            current: ctx.current,
            prev: ctx.prev,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
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
pub fn convert_assertion_severity(severity: IrAssertionSeverity) -> AssertionSeverity {
    match severity {
        IrAssertionSeverity::Warn => AssertionSeverity::Warn,
        IrAssertionSeverity::Error => AssertionSeverity::Error,
        IrAssertionSeverity::Fatal => AssertionSeverity::Fatal,
    }
}

/// Builds a resolver function for a compiled signal.
pub fn build_signal_resolver(
    signal: &crate::CompiledSignal,
    world: &CompiledWorld,
) -> Option<ResolverFn> {
    if let Some(ref components) = signal.resolve_components {
        if components.iter().any(contains_entity_expression) {
            return None;
        }
        return Some(build_vector_resolver(components, &signal.value_type, world));
    }

    signal.resolve.as_ref().and_then(|expr| {
        if contains_entity_expression(expr) {
            None
        } else {
            Some(build_resolver(expr, world, signal.uses_dt_raw))
        }
    })
}

fn build_vector_resolver(
    components: &[CompiledExpr],
    value_type: &ValueType,
    world: &CompiledWorld,
) -> ResolverFn {
    let bytecodes: Vec<BytecodeChunk> = components.iter().map(|e| codegen::compile(e)).collect();
    let constants = world.constants.clone();
    let config = world.config.clone();
    let value_type = value_type.clone();

    Box::new(move |ctx| {
        let exec_ctx = ResolverContext {
            prev: ctx.prev,
            inputs: ctx.inputs,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
        };

        let results: Vec<f64> = bytecodes.iter().map(|bc| execute(bc, &exec_ctx)).collect();

        match value_type {
            ValueType::Vec2 { .. } => Value::Vec2([results[0], results[1]]),
            ValueType::Vec3 { .. } => Value::Vec3([results[0], results[1], results[2]]),
            ValueType::Vec4 { .. } => Value::Vec4([results[0], results[1], results[2], results[3]]),
            _ => Value::Scalar(results[0]),
        }
    })
}

pub fn build_resolver(
    expr: &CompiledExpr,
    world: &CompiledWorld,
    _uses_dt_raw: bool,
) -> ResolverFn {
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let exec_ctx = ResolverContext {
            prev: ctx.prev,
            inputs: ctx.inputs,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
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

/// Gets the initial scalar value for a signal from config.
pub fn get_initial_value(world: &CompiledWorld, signal_id: &SignalId) -> f64 {
    let signal_path = signal_id.to_string();
    let parts: Vec<&str> = signal_path.split('.').collect();

    if parts.len() >= 2 {
        let last = parts.last().unwrap();
        for (key, value) in &world.config {
            if key.ends_with(&format!("initial_{}", last)) {
                return *value;
            }
        }
    }
    0.0
}

/// Gets the initial value for a signal with proper type wrapping.
pub fn get_initial_signal_value(world: &CompiledWorld, signal_id: &SignalId) -> Value {
    let initial_value = get_initial_value(world, signal_id);

    let signals = world.signals();
    if let Some(signal) = signals.get(signal_id) {
        match signal.value_type {
            ValueType::Scalar { .. } => Value::Scalar(initial_value),
            ValueType::Vec2 { .. } => Value::Vec2([initial_value; 2]),
            ValueType::Vec3 { .. } => Value::Vec3([initial_value; 3]),
            ValueType::Vec4 { .. } => Value::Vec4([initial_value; 4]),
            _ => Value::Scalar(initial_value),
        }
    } else {
        Value::Scalar(initial_value)
    }
}

/// Builds an aggregate resolver for signals that depend on member signal data.
pub fn build_aggregate_resolver(expr: &CompiledExpr, world: &CompiledWorld) -> AggregateResolverFn {
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(
        move |signals: &SignalStorage, members: &MemberSignalBuffer, dt: Dt, sim_time: f64| {
            let mut ctx = MemberInterpContext {
                prev: InterpValue::Scalar(0.0),
                index: 0,
                dt: dt.seconds(),
                sim_time,
                signals,
                members,
                constants: &constants,
                config: &config,
                locals: HashMap::new(),
                entity_prefix: String::new(),
                read_current: true,
            };

            let result = interpret_expr(&expr, &mut ctx);
            Value::Scalar(result.as_f64())
        },
    )
}
