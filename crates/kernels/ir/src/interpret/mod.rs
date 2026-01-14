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

use continuum_foundation::{EntityId, EraId, FieldId, InstanceId, PrimitiveStorageClass, SignalId};
use continuum_runtime::MemberSignalBuffer;
use continuum_runtime::executor::{
    AggregateResolverFn, AssertionFn, EraConfig, FractureFn, MeasureFn, ResolverFn, Runtime,
    TransitionFn, WarmupFn,
};
use continuum_runtime::soa_storage::ValueType as MemberValueType;
use continuum_runtime::storage::{EntityInstances, EntityStorage, InstanceData, SignalStorage};
use continuum_runtime::types::{Dt, Value, WarmupConfig};
// Import functions crate to ensure kernels are registered

use continuum_functions as _;
use continuum_vm::execute;

use crate::{
    AssertionSeverity as IrAssertionSeverity, CompilationResult, CompiledEra, CompiledExpr,
    CompiledWorld, codegen, units::Unit,
};

use contexts::{
    AssertionContext, FractureExecContext, MeasureContext, ResolverContext, SharedContextData,
    TransitionContext, WarmupContext,
};

pub fn build_resolver(
    expr: &CompiledExpr,
    constants: &IndexMap<String, (f64, Option<Unit>)>,
    config: &IndexMap<String, (f64, Option<Unit>)>,
) -> ResolverFn {
    let bytecode = codegen::compile(expr);
    let constants = constants.clone();
    let config = config.clone();

    Box::new(move |ctx| {
        let mut context = ResolverContext {
            prev: ctx.prev,
            inputs: ctx.inputs,
            dt: ctx.dt.0,
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
                entities: ctx.entities,
                current_entity: None,
                self_instance: None,
                other_instance: None,
            },
        };
        execute(&bytecode, &mut context)
    })
}

/// Builds a signal resolution function.
pub fn build_signal_resolver(
    signal: &crate::CompiledSignal,
    world: &CompiledWorld,
) -> Option<ResolverFn> {
    let resolve_expr = signal.resolve.as_ref()?;
    Some(build_resolver(
        resolve_expr,
        &world.constants,
        &world.config,
    ))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MemberResolverStats {
    pub scalar_count: usize,
    pub vec3_count: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeBuildOptions {
    pub dt_override: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeBuildReport {
    pub resolver_count: usize,
    pub aggregate_count: usize,
    pub assertion_count: usize,
    pub field_count: usize,
    pub skipped_fields: usize,
    pub fracture_count: usize,
    pub member_signal_count: usize,
    pub member_initial_count: usize,
    pub member_resolvers: MemberResolverStats,
    pub max_member_instances: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeBuildError {
    #[error("failed to set member initial value for {member_id}: {error}")]
    MemberInitialization { member_id: String, error: String },
}

impl MemberResolverStats {
    pub fn total(&self) -> usize {
        self.scalar_count + self.vec3_count
    }
}

pub fn register_member_resolvers(
    runtime: &mut Runtime,
    world: &CompiledWorld,
) -> MemberResolverStats {
    let mut stats = MemberResolverStats::default();

    for (member_id, member) in world.members() {
        let Some(ref resolve_expr) = member.resolve else {
            continue;
        };

        let entity_prefix = member.entity_id.to_string();
        match member.value_type.storage_class() {
            PrimitiveStorageClass::Vec3 => {
                let resolver = build_vec3_member_resolver(
                    resolve_expr,
                    &world.constants,
                    &world.config,
                    &entity_prefix,
                );
                runtime.register_vec3_member_resolver(member_id.to_string(), resolver);
                stats.vec3_count += 1;
            }
            _ => {
                let resolver = build_member_resolver(
                    resolve_expr,
                    &world.constants,
                    &world.config,
                    &entity_prefix,
                );
                runtime.register_member_resolver(member_id.to_string(), resolver);
                stats.scalar_count += 1;
            }
        }
    }

    stats
}

pub fn build_runtime(
    world: &CompiledWorld,
    compilation: CompilationResult,
    options: RuntimeBuildOptions,
) -> Result<(Runtime, RuntimeBuildReport), RuntimeBuildError> {
    let mut report = RuntimeBuildReport::default();

    let initial_era = world
        .eras()
        .iter()
        .find(|(_, era)| era.is_initial)
        .map(|(id, _)| id.clone())
        .or_else(|| world.eras().keys().next().cloned())
        .unwrap_or_else(|| EraId::from("default"));

    let mut era_configs = build_era_configs(world);
    if let Some(dt) = options.dt_override {
        for config in era_configs.values_mut() {
            config.dt = Dt(dt);
        }
    }

    let mut runtime = Runtime::new(initial_era, era_configs, compilation.dags);

    // Register resolvers (signals + aggregates + warmups).
    for (signal_id, signal) in &world.signals() {
        if let Some(ref resolve_expr) = signal.resolve {
            let resolver = build_resolver(resolve_expr, &world.constants, &world.config);
            runtime.register_resolver(resolver);
            report.resolver_count += 1;
        }

        if let Some(ref warmup) = signal.warmup {
            let warmup_fn = build_warmup_fn(&warmup.iterate, &world.constants, &world.config);
            let config = WarmupConfig {
                max_iterations: warmup.iterations,
                convergence_epsilon: warmup.convergence,
            };
            runtime.register_warmup(signal_id.clone(), warmup_fn, config);
        }
    }

    // Register assertions.
    for (signal_id, signal) in &world.signals() {
        for assertion in &signal.assertions {
            let assertion_fn = build_assertion(&assertion.condition, world);
            let severity = convert_assertion_severity(assertion.severity);
            runtime.register_assertion(
                signal_id.clone(),
                assertion_fn,
                severity,
                assertion.message.clone(),
            );
            report.assertion_count += 1;
        }
    }

    // Register field measure functions.
    for (field_id, field) in &world.fields() {
        if let Some(ref expr) = field.measure {
            if let Some(measure_fn) = build_field_measure(field_id, expr, world) {
                runtime.register_measure_op(measure_fn);
                report.field_count += 1;
            } else {
                report.skipped_fields += 1;
            }
        }
    }

    // Register fractures.
    for (_, fracture) in &world.fractures() {
        runtime.register_fracture(build_fracture(fracture, world));
        report.fracture_count += 1;
    }

    // Initialize signals.
    for (signal_id, _) in &world.signals() {
        let value = get_initial_signal_value(world, signal_id);
        runtime.init_signal(signal_id.clone(), value);
    }

    // Initialize entities + member signals.
    let entity_counts = entity_instance_counts(world);
    for (entity_id, _entity) in &world.entities() {
        let count = entity_counts.get(entity_id).copied().unwrap_or(1);
        let mut instances = EntityInstances::new();
        for i in 0..count {
            let instance_id = InstanceId::from(format!("{}_{}", entity_id, i));
            let mut fields = indexmap::IndexMap::new();
            for (_member_id, member) in &world.members() {
                if &member.entity_id == entity_id {
                    fields.insert(
                        member.signal_name.clone(),
                        member.value_type.default_value(),
                    );
                }
            }
            instances.insert(instance_id, InstanceData::new(fields));
        }
        runtime.init_entity(entity_id.clone(), instances);
    }

    if !world.members().is_empty() {
        let max_instance_count = entity_counts.values().copied().max().unwrap_or(1);
        report.max_member_instances = max_instance_count;

        for (member_id, member) in &world.members() {
            let value_type = match member.value_type.storage_class() {
                PrimitiveStorageClass::Scalar => MemberValueType::scalar(),
                PrimitiveStorageClass::Vec2 => MemberValueType::vec2(),
                PrimitiveStorageClass::Vec3 => MemberValueType::vec3(),
                PrimitiveStorageClass::Vec4 => {
                    if member.value_type.primitive_id().name() == "Quat" {
                        MemberValueType::quat()
                    } else {
                        MemberValueType::vec4()
                    }
                }
                _ => MemberValueType::scalar(),
            };
            runtime.register_member_signal(&member_id.to_string(), value_type);
        }
        report.member_signal_count = world.members().len();

        runtime.init_member_instances(max_instance_count);

        for (entity_id, count) in &entity_counts {
            let entity_key = entity_id.to_string();
            runtime.register_entity_count(&entity_key, *count);
        }

        for (member_id, member) in &world.members() {
            if let Some(ref initial_expr) = member.initial {
                let initial_value =
                    eval_initial_expr(initial_expr, &world.constants, &world.config);
                let instance_count = entity_counts.get(&member.entity_id).copied().unwrap_or(1);
                for instance_idx in 0..instance_count {
                    if let Err(source) = runtime.set_member_signal(
                        &member_id.to_string(),
                        instance_idx,
                        initial_value.clone(),
                    ) {
                        return Err(RuntimeBuildError::MemberInitialization {
                            member_id: member_id.to_string(),
                            error: source,
                        });
                    }
                }
                report.member_initial_count += 1;
            }
        }
        if report.member_initial_count > 0 {
            runtime.commit_member_initials();
        }

        report.member_resolvers = register_member_resolvers(&mut runtime, world);
    }

    Ok((runtime, report))
}

fn entity_instance_counts(world: &CompiledWorld) -> IndexMap<EntityId, usize> {
    let mut counts = IndexMap::new();
    for (entity_id, entity) in &world.entities() {
        let count = if let Some(ref count_source) = entity.count_source {
            world
                .config
                .get(count_source)
                .map(|(v, _)| *v as usize)
                .unwrap_or(1)
        } else if let Some((min, max)) = entity.count_bounds {
            if min == max {
                min as usize
            } else {
                min as usize
            }
        } else {
            1
        };
        counts.insert(entity_id.clone(), count);
    }
    counts
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
        let mut context = WarmupContext {
            current: ctx.prev,
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
                entities: ctx.entities,
                current_entity: None,
                self_instance: None,
                other_instance: None,
            },
        };
        execute(&bytecode, &mut context)
    })
}

/// Builds an aggregate resolver function.
pub fn build_aggregate_resolver(expr: &CompiledExpr, world: &CompiledWorld) -> AggregateResolverFn {
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |signals, _entities, members, dt, sim_time| {
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

        // TODO: Switch aggregate signals to full bytecode as well.
        // For now, use the old interpreter but we could use the VM here too.
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

    Some(Box::new(
        move |signals: &SignalStorage, entities: &EntityStorage, sim_time: f64| {
            for (target_era, bytecode) in &transitions {
                let mut context = TransitionContext {
                    sim_time,
                    shared: SharedContextData {
                        constants: &constants,
                        config: &config,
                        signals,
                        entities,
                        current_entity: None,
                        self_instance: None,
                        other_instance: None,
                    },
                };

                let result = execute(bytecode, &mut context);
                if result.as_bool().unwrap_or(false) {
                    return Some(target_era.clone());
                }
            }
            None
        },
    ))
}

/// Builds a measure function for computing field values.
pub fn build_field_measure(
    field_id: &FieldId,
    expr: &CompiledExpr,
    world: &CompiledWorld,
) -> Option<MeasureFn> {
    let bytecode = codegen::compile(expr);
    let constants = world.constants.clone();
    let config = world.config.clone();
    let field_id = field_id.clone();

    Some(Box::new(move |ctx| {
        let mut context = MeasureContext {
            dt: ctx.dt.0,
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
                entities: ctx.entities,
                current_entity: None,
                self_instance: None,
                other_instance: None,
            },
        };

        let value = execute(&bytecode, &mut context);
        ctx.fields.emit(field_id.clone(), [0.0, 0.0, 0.0], value);
    }))
}

/// Builds a fracture detection function.
pub fn build_fracture(fracture: &crate::CompiledFracture, world: &CompiledWorld) -> FractureFn {
    let mut conditions = Vec::new();
    for cond in &fracture.conditions {
        conditions.push(codegen::compile(cond));
    }

    let mut emits = Vec::new();
    for emit in &fracture.emits {
        emits.push((emit.target.clone(), codegen::compile(&emit.value)));
    }

    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let mut context = FractureExecContext {
            dt: ctx.dt.0,
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
                entities: ctx.entities,
                current_entity: None,
                self_instance: None,
                other_instance: None,
            },
        };

        // Check conditions
        let mut triggered = true;
        for bytecode in &conditions {
            let val = execute(bytecode, &mut context);
            if !val.as_bool().unwrap_or(false) {
                triggered = false;
                break;
            }
        }

        if triggered {
            let mut outputs = Vec::new();
            for (target, bytecode) in &emits {
                let val = execute(bytecode, &mut context);
                outputs.push((target.clone(), val.as_scalar().unwrap_or(0.0)));
            }
            Some(outputs)
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
        let mut context = AssertionContext {
            current: ctx.current,
            prev: ctx.prev,
            dt: ctx.dt.0,
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
                entities: ctx.entities,
                current_entity: None,
                self_instance: None,
                other_instance: None,
            },
        };
        execute(&bytecode, &mut context).as_bool().unwrap_or(false)
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
