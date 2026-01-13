//! CLOSURE BUILDERS
//!
//! This module builds runtime closures from compiled IR expressions. These
//! closures capture the necessary context (constants, config, bytecode) and
//! can be invoked during simulation execution.

mod contexts;
mod member_interp;

#[cfg(test)]
mod tests;

// Re-export member interpreter types
pub use continuum_runtime::executor::{
    ScalarResolverFn as MemberResolverFn, Vec3ResolverFn as Vec3MemberResolverFn,
};
pub use member_interp::{
    InterpValue, MemberInterpContext, build_member_resolver, build_vec3_member_resolver,
    interpret_expr,
};

use indexmap::IndexMap;
use std::collections::HashMap;

use continuum_foundation::{EraId, FieldId, SignalId};
use continuum_runtime::MemberSignalBuffer;
use continuum_runtime::executor::{
    AggregateResolverFn, AssertionFn, EraConfig, FractureFn, MeasureFn, ResolverFn, TransitionFn,
    WarmupFn,
};
// Import functions crate to ensure kernels are registered

use continuum_functions as _;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::{Dt, Value};
use continuum_vm::execute;

use crate::{
    AssertionSeverity as IrAssertionSeverity, CompiledEra, CompiledExpr, CompiledWorld, codegen,
    units::Unit,
};

use contexts::{
    AssertionContext, FractureExecContext, MeasureContext, ResolverContext, SharedContextData,
    TransitionContext, WarmupContext,
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

        _ => false,
    }
}

pub fn build_resolver(
    expr: &CompiledExpr,
    constants: &IndexMap<String, (f64, Option<Unit>)>,
    config: &IndexMap<String, (f64, Option<Unit>)>,
) -> ResolverFn {
    let bytecode = codegen::compile(expr);
    let constants = constants.clone();
    let config = config.clone();

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
        result
    })
}

/// Builds a signal resolution function.
pub fn build_signal_resolver(
    signal: &crate::CompiledSignal,
    world: &CompiledWorld,
) -> Option<ResolverFn> {
    let resolve_expr = signal.resolve.as_ref()?;
    if contains_entity_expression(resolve_expr) {
        return None;
    }

    Some(build_resolver(
        resolve_expr,
        &world.constants,
        &world.config,
    ))
}

/// Builds a warmup function for a signal.
pub fn build_warmup_fn(
    expr: &CompiledExpr,
    constants: &IndexMap<String, (f64, Option<Unit>)>,
    config: &IndexMap<String, (f64, Option<Unit>)>,
) -> WarmupFn {
    let bytecode = codegen::compile(expr);
    let constants = constants.clone();
    let config = config.clone();

    Box::new(move |ctx| {
        let exec_ctx = WarmupContext {
            current: ctx.prev,
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
        };
        let result = execute(&bytecode, &exec_ctx);
        result
    })
}

/// Builds an aggregate resolver function.
pub fn build_aggregate_resolver(expr: &CompiledExpr, world: &CompiledWorld) -> AggregateResolverFn {
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |signals, members, dt, sim_time| {
        let mut ctx = MemberInterpContext {
            prev: InterpValue::Scalar(0.0),
            index: 0,
            dt: dt.seconds(),
            sim_time,
            signals,
            members,
            constants: &constants,
            config: &config,
            locals: std::collections::HashMap::new(),
            entity_prefix: String::new(),
            read_current: true,
        };

        match interpret_expr(&expr, &mut ctx) {
            InterpValue::Scalar(v) => Value::Scalar(v),
            InterpValue::Vec3(v) => Value::Vec3(v),
            InterpValue::Vec4(v) => Value::Vec4(v),
            InterpValue::Quat(v) => Value::Quat(v),
            InterpValue::Bool(b) => Value::Scalar(if b { 1.0 } else { 0.0 }),
        }
    })
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
    constants: &IndexMap<String, (f64, Option<Unit>)>,
    config: &IndexMap<String, (f64, Option<Unit>)>,
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
            if result != continuum_foundation::Value::Scalar(0.0) {
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
        if let Some(scalar) = result.as_scalar() {
            ctx.fields.emit_scalar(field_id.clone(), scalar);
        }
    }))
}

/// Builds a fracture detection function.
pub fn build_fracture(fracture: &crate::CompiledFracture, world: &CompiledWorld) -> FractureFn {
    let conditions: Vec<_> = fracture
        .conditions
        .iter()
        .map(|c| codegen::compile(c))
        .collect();
    let emits: Vec<_> = fracture
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

        // Check conditions (all must be true)
        let mut triggered = true;
        for bytecode in &conditions {
            let res = execute(bytecode, &exec_ctx);
            if !res
                .as_bool()
                .unwrap_or(res.as_scalar().unwrap_or(0.0) != 0.0)
            {
                triggered = false;
                break;
            }
        }

        if triggered {
            // Apply emissions
            let mut results = Vec::new();
            for (target, bytecode) in &emits {
                let value = execute(bytecode, &exec_ctx);
                if let Some(scalar) = value.as_scalar() {
                    results.push((target.clone(), scalar));
                }
            }
            Some(results)
        } else {
            None
        }
    })
}

/// Builds an assertion function.
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
        let res = execute(&bytecode, &exec_ctx);
        res.as_bool()
            .unwrap_or(res.as_scalar().unwrap_or(0.0) != 0.0)
    })
}

pub fn get_initial_signal_value(world: &CompiledWorld, signal_id: &SignalId) -> Value {
    let name = signal_id.to_string();

    // 1. Check exact name in config (e.g. terra.energy)
    if let Some((value, _)) = world.config.get(&name) {
        return Value::Scalar(*value);
    }

    // 2. Check name.initial (e.g. terra.energy.initial)
    if let Some((value, _)) = world.config.get(&format!("{}.initial", name)) {
        return Value::Scalar(*value);
    }

    // 3. Check initial_name (legacy/test pattern, e.g. terra.initial_energy)
    // We need to handle the case where initial is in the middle: terra.initial_energy
    // If the signal is terra.energy, it might be terra.initial_energy.
    if let Some(pos) = name.rfind('.') {
        let prefix = &name[..pos];
        let last = &name[pos + 1..];
        if let Some((value, _)) = world.config.get(&format!("{}.initial_{}", prefix, last)) {
            return Value::Scalar(*value);
        }
    } else if let Some((value, _)) = world.config.get(&format!("initial_{}", name)) {
        return Value::Scalar(*value);
    }

    // Default to 0.0 for now
    Value::Scalar(0.0)
}

/// Get initial value for a signal (legacy compatibility).
pub fn get_initial_value(world: &CompiledWorld, signal_id: &SignalId) -> f64 {
    match get_initial_signal_value(world, signal_id) {
        Value::Scalar(v) => v,
        _ => 0.0,
    }
}

/// Convert IR assertion severity to runtime severity.
pub fn convert_assertion_severity(
    severity: IrAssertionSeverity,
) -> continuum_runtime::executor::AssertionSeverity {
    match severity {
        IrAssertionSeverity::Warn => continuum_runtime::executor::AssertionSeverity::Warn,
        IrAssertionSeverity::Error => continuum_runtime::executor::AssertionSeverity::Error,
        IrAssertionSeverity::Fatal => continuum_runtime::executor::AssertionSeverity::Fatal,
    }
}

/// Evaluate an expression for initial value (legacy compatibility)
pub fn eval_initial_expr(
    expr: &CompiledExpr,
    constants: &IndexMap<String, (f64, Option<Unit>)>,
    config: &IndexMap<String, (f64, Option<Unit>)>,
) -> Value {
    // This is used for member initials which are resolved before simulation starts
    // and can only read constants and config.
    let signals = SignalStorage::default();
    let members = MemberSignalBuffer::new();

    let mut ctx = MemberInterpContext {
        prev: InterpValue::Scalar(0.0),
        index: 0,
        dt: 0.0,
        sim_time: 0.0,
        signals: &signals,
        members: &members,
        constants,
        config,
        locals: HashMap::new(),
        entity_prefix: String::new(),
        read_current: false,
    };

    match interpret_expr(expr, &mut ctx) {
        InterpValue::Scalar(v) => Value::Scalar(v),
        InterpValue::Vec3(v) => Value::Vec3(v),
        InterpValue::Vec4(v) => Value::Vec4(v),
        InterpValue::Quat(v) => Value::Quat(v),
        InterpValue::Bool(b) => Value::Scalar(if b { 1.0 } else { 0.0 }),
    }
}
