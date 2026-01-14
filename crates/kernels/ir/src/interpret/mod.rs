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

use continuum_foundation::{EraId, FieldId, PrimitiveStorageClass, SignalId};
use continuum_runtime::MemberSignalBuffer;
use continuum_runtime::executor::{
    AggregateResolverFn, AssertionFn, ChronicleFn, EmittedEvent, EraConfig, FractureFn, ImpulseFn,
    MeasureFn, ResolverFn, Runtime, TransitionFn, WarmupFn,
};
use continuum_runtime::soa_storage::ValueType as MemberValueType;
use continuum_runtime::storage::{EntityInstances, InputChannels, InstanceData};
use continuum_runtime::types::{Dt, Value, WarmupConfig};
// Import functions crate to ensure kernels are registered

use continuum_functions as _;
use continuum_runtime::storage::SignalStorage;
use continuum_vm::{BytecodeChunk, ExecutionContext, execute};

use crate::{
    AssertionSeverity as IrAssertionSeverity, CompilationResult, CompiledChronicle, CompiledEra,
    CompiledExpr, CompiledImpulse, CompiledWorld, codegen, units::Unit,
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
    pub impulse_count: usize,
    pub chronicle_count: usize,
    pub member_signal_count: usize,
    pub member_initial_count: usize,
    pub member_resolvers: MemberResolverStats,
    pub max_member_instances: usize,
    pub impulse_indices: IndexMap<continuum_foundation::ImpulseId, usize>,
    pub chronicle_indices: IndexMap<continuum_foundation::ChronicleId, usize>,
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
        if let Some(resolver) = build_signal_resolver(signal, world) {
            runtime.register_resolver(resolver);
            report.resolver_count += 1;
        } else if let Some(ref resolve_expr) = signal.resolve {
            let signal_name = signal_id.to_string();
            let placeholder: ResolverFn = Box::new(move |_ctx| {
                panic!(
                    "Signal '{}' placeholder called - aggregate signals run in Phase 3c",
                    signal_name
                );
            });
            runtime.register_resolver(placeholder);
            report.resolver_count += 1;

            let aggregate_resolver = build_aggregate_resolver(resolve_expr, world);
            runtime.register_aggregate_resolver(signal_id.clone(), aggregate_resolver);
            report.aggregate_count += 1;
        } else {
            let signal_name = signal_id.to_string();
            let placeholder: ResolverFn = Box::new(move |_ctx| {
                panic!("Signal '{}' has no resolve expression", signal_name);
            });
            runtime.register_resolver(placeholder);
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

    // Register impulses.
    for (impulse_id, impulse) in &world.impulses() {
        if let Some(handler) = build_impulse_handler(impulse, world) {
            let idx = runtime.register_impulse(handler);
            report.impulse_count += 1;
            report.impulse_indices.insert(impulse_id.clone(), idx);
        }
    }

    // Register chronicles.
    for (chronicle_id, chronicle) in &world.chronicles() {
        let idx = runtime.register_chronicle(build_chronicle_handler(chronicle, world));
        report.chronicle_count += 1;
        report.chronicle_indices.insert(chronicle_id.clone(), idx);
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
            let instance_id =
                continuum_foundation::InstanceId::from(format!("{}_{}", entity_id, i));
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

fn entity_instance_counts(
    world: &CompiledWorld,
) -> IndexMap<continuum_foundation::EntityId, usize> {
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

    Some(Box::new(
        move |signals: &SignalStorage, sim_time: f64| -> Option<continuum_foundation::EraId> {
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
        },
    ))
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
            let mut results: Vec<(continuum_foundation::SignalId, f64)> = Vec::new();
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

struct ChronicleEvalContext<'a> {
    dt: f64,
    sim_time: f64,
    shared: SharedContextData<'a>,
}

impl ExecutionContext for ChronicleEvalContext<'_> {
    fn prev(&self) -> Value {
        Value::Scalar(0.0)
    }

    fn dt_scalar(&self) -> f64 {
        self.dt
    }

    fn sim_time(&self) -> Value {
        Value::Scalar(self.sim_time)
    }

    fn inputs(&self) -> Value {
        Value::Scalar(0.0)
    }

    fn signal(&self, name: &str) -> Value {
        self.shared.signal(name)
    }

    fn constant(&self, name: &str) -> Value {
        self.shared.constant(name)
    }

    fn config(&self, name: &str) -> Value {
        self.shared.config(name)
    }

    fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
        let (namespace, function) = name
            .split_once('.')
            .unwrap_or_else(|| panic!("Kernel call '{}' is missing a namespace", name));
        continuum_kernel_registry::eval_in_namespace(namespace, function, args, self.dt)
            .unwrap_or_else(|| {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, function
                )
            })
    }
}

struct ChronicleHandlerSpec {
    event_name: String,
    condition: BytecodeChunk,
    fields: Vec<(String, BytecodeChunk)>,
}

pub fn build_chronicle_handler(
    chronicle: &CompiledChronicle,
    world: &CompiledWorld,
) -> ChronicleFn {
    let constants = world.constants.clone();
    let config = world.config.clone();
    let chronicle_id = chronicle.id.to_string();
    let handlers: Vec<ChronicleHandlerSpec> = chronicle
        .handlers
        .iter()
        .map(|handler| ChronicleHandlerSpec {
            event_name: handler.event_name.clone(),
            condition: codegen::compile(&handler.condition),
            fields: handler
                .event_fields
                .iter()
                .map(|field| (field.name.clone(), codegen::compile(&field.value)))
                .collect(),
        })
        .collect();

    Box::new(move |ctx| {
        let exec_ctx = ChronicleEvalContext {
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            shared: SharedContextData {
                constants: &constants,
                config: &config,
                signals: ctx.signals,
            },
        };

        let mut events = Vec::new();
        for handler in &handlers {
            let result = execute(&handler.condition, &exec_ctx);
            let triggered = result
                .as_bool()
                .unwrap_or(result.as_scalar().unwrap_or(0.0) != 0.0);
            if !triggered {
                continue;
            }

            let mut fields: Vec<(String, Value)> = Vec::new();
            for (name, bytecode) in &handler.fields {
                let value = execute(bytecode, &exec_ctx);
                fields.push((name.clone(), value));
            }

            events.push(EmittedEvent {
                chronicle_id: chronicle_id.clone(),
                name: handler.event_name.clone(),
                fields,
            });
        }

        events
    })
}

struct ImpulseEvalContext<'a> {
    payload: &'a Value,
    signals: &'a SignalStorage,
    channels: &'a mut InputChannels,
    dt: f64,
    sim_time: f64,
    constants: &'a IndexMap<String, (f64, Option<Unit>)>,
    config: &'a IndexMap<String, (f64, Option<Unit>)>,
    locals: HashMap<String, InterpValue>,
}

impl ImpulseEvalContext<'_> {
    fn constant(&self, name: &str) -> f64 {
        self.constants
            .get(name)
            .map(|(v, _)| *v)
            .unwrap_or_else(|| panic!("Constant '{}' not defined", name))
    }

    fn config(&self, name: &str) -> f64 {
        self.config
            .get(name)
            .map(|(v, _)| *v)
            .unwrap_or_else(|| panic!("Config value '{}' not defined", name))
    }

    fn signal(&self, name: &str) -> InterpValue {
        let id = SignalId::from(name);
        match self.signals.get(&id) {
            Some(value) => InterpValue::from_value(value),
            None => panic!("Signal '{}' not found in storage", name),
        }
    }

    fn payload_value(&self) -> InterpValue {
        InterpValue::from_value(self.payload)
    }

    fn payload_field(&self, field: &str) -> InterpValue {
        let value = match self.payload {
            Value::Map(v) => v
                .iter()
                .find(|(k, _)| k == field)
                .map(|(_, v)| v)
                .ok_or_else(|| format!("payload field '{}' not found", field)),
            _ => Err("payload is not structured; expected Map payload".to_string()),
        }
        .unwrap_or_else(|e| panic!("{}", e));

        InterpValue::from_value(value)
    }

    fn call_kernel(&self, namespace: &str, name: &str, args: &[Value]) -> Value {
        continuum_kernel_registry::eval_in_namespace(namespace, name, args, self.dt).unwrap_or_else(
            || {
                panic!(
                    "Unknown kernel function '{}.{}' - function not found in registry",
                    namespace, name
                )
            },
        )
    }
}

fn eval_impulse_function(name: &str, args: &[InterpValue]) -> InterpValue {
    match name {
        "vec2" => InterpValue::Vec3([args[0].as_f64(), args[1].as_f64(), 0.0]),
        "vec3" => InterpValue::Vec3([args[0].as_f64(), args[1].as_f64(), args[2].as_f64()]),
        _ => panic!("Unknown function '{}' in impulse apply", name),
    }
}

fn eval_impulse_expr(expr: &CompiledExpr, ctx: &mut ImpulseEvalContext) -> InterpValue {
    match expr {
        CompiledExpr::Literal(value, _) => InterpValue::Scalar(*value),
        CompiledExpr::DtRaw => InterpValue::Scalar(ctx.dt),
        CompiledExpr::SimTime => InterpValue::Scalar(ctx.sim_time),
        CompiledExpr::Signal(id) => ctx.signal(&id.to_string()),
        CompiledExpr::Const(name, _) => InterpValue::Scalar(ctx.constant(name)),
        CompiledExpr::Config(name, _) => InterpValue::Scalar(ctx.config(name)),
        CompiledExpr::Payload => ctx.payload_value(),
        CompiledExpr::PayloadField(field) => ctx.payload_field(field),
        CompiledExpr::Binary { op, left, right } => {
            let l = eval_impulse_expr(left, ctx);
            let r = eval_impulse_expr(right, ctx);
            l.binary_op(r, *op)
        }
        CompiledExpr::Unary { op, operand } => {
            let value = eval_impulse_expr(operand, ctx);
            value.unary_op(*op)
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if eval_impulse_expr(condition, ctx).as_f64() != 0.0 {
                eval_impulse_expr(then_branch, ctx)
            } else {
                eval_impulse_expr(else_branch, ctx)
            }
        }
        CompiledExpr::Let { name, value, body } => {
            let evaluated = eval_impulse_expr(value, ctx);
            ctx.locals.insert(name.clone(), evaluated);
            let result = eval_impulse_expr(body, ctx);
            ctx.locals.remove(name);
            result
        }
        CompiledExpr::Local(name) => *ctx
            .locals
            .get(name)
            .unwrap_or_else(|| panic!("Unknown local '{}'", name)),
        CompiledExpr::KernelCall {
            namespace,
            function,
            args,
        } => {
            let arg_values: Vec<Value> = args
                .iter()
                .map(|arg| eval_impulse_expr(arg, ctx).into_value())
                .collect();
            InterpValue::from_value(&ctx.call_kernel(namespace, function, &arg_values))
        }
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<_> = args.iter().map(|arg| eval_impulse_expr(arg, ctx)).collect();
            eval_impulse_function(function, &arg_values)
        }
        CompiledExpr::EmitSignal { target, value } => {
            let emitted = eval_impulse_expr(value, ctx);
            ctx.channels.accumulate(target, emitted.as_f64());
            emitted
        }
        CompiledExpr::Prev
        | CompiledExpr::Collected
        | CompiledExpr::SelfField(_)
        | CompiledExpr::EntityAccess { .. }
        | CompiledExpr::Aggregate { .. }
        | CompiledExpr::Other { .. }
        | CompiledExpr::Pairs { .. }
        | CompiledExpr::Filter { .. }
        | CompiledExpr::First { .. }
        | CompiledExpr::Nearest { .. }
        | CompiledExpr::Within { .. }
        | CompiledExpr::FieldAccess { .. } => {
            panic!("Unsupported expression in impulse apply: {:?}", expr)
        }
    }
}

pub fn build_impulse_handler(
    impulse: &CompiledImpulse,
    world: &CompiledWorld,
) -> Option<ImpulseFn> {
    let apply = impulse.apply.as_ref()?;
    let apply = apply.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Some(Box::new(move |ctx, payload| {
        let mut eval_ctx = ImpulseEvalContext {
            payload,
            signals: ctx.signals,
            channels: ctx.channels,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            constants: &constants,
            config: &config,
            locals: HashMap::new(),
        };
        let _ = eval_impulse_expr(&apply, &mut eval_ctx);
    }))
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
