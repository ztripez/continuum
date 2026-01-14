//!
//! Orchestrates DAG execution through phases.

mod assertions;
mod context;
pub mod cost_model;
pub mod kernel_registry;
pub mod l1_kernels;
pub mod l3_kernel;
pub mod lane_kernel;
pub mod lowering_strategy;
pub mod member_executor;
mod phases;
mod warmup;

// Re-export public types
pub use assertions::{AssertionChecker, AssertionFn, AssertionSeverity, SignalAssertion};
pub use context::{
    AssertContext, ChronicleContext, CollectContext, FractureContext, ImpulseContext,
    MeasureContext, ResolveContext, WarmupContext,
};
pub use kernel_registry::LaneKernelRegistry;
pub use l1_kernels::{ScalarKernelFn, ScalarL1Kernel, Vec3KernelFn, Vec3L1Kernel};
pub use l3_kernel::{
    L3Kernel, L3KernelBuilder, MemberDag, MemberDagError, ScalarL3MemberResolver,
    ScalarL3ResolverFn, Vec3L3MemberResolver, Vec3L3ResolverFn,
};
pub use lane_kernel::{LaneKernel, LaneKernelError, LaneKernelResult};
pub use lowering_strategy::{LoweringHeuristics, LoweringStrategy};
pub use member_executor::{
    ChunkConfig, MemberResolveContext, MemberSignalResolver, ScalarL1Resolver,
    ScalarResolveContext, ScalarResolverFn, Vec3L1Resolver, Vec3ResolveContext, Vec3ResolverFn,
};
pub use phases::{CollectFn, FractureFn, ImpulseFn, MeasureFn, PhaseExecutor, ResolverFn};
pub use warmup::{WarmupExecutor, WarmupFn};

use indexmap::IndexMap;

use tracing::{error, info, instrument, trace};

use crate::dag::DagSet;
use crate::error::{Error, Result};
use crate::soa_storage::{MemberSignalBuffer, ValueType as MemberValueType};
use crate::storage::{
    EmittedEventRecord, EntityInstances, EntityStorage, EventBuffer, FieldBuffer, FieldSample,
    FractureQueue, InputChannels, SignalStorage,
};
use crate::types::{
    Dt, EntityId, EraId, FieldId, SignalId, StratumId, StratumState, TickContext, Value,
    WarmupConfig, WarmupResult,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub created_at: String,
    pub seed: u64,
    pub steps: u64,
    pub stride: u64,
    pub signals: Vec<String>,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickSnapshot {
    pub tick: u64,
    pub time_seconds: f64,
    pub signals: std::collections::HashMap<String, Value>,
    pub fields: std::collections::HashMap<String, Vec<FieldSample>>,
}

#[derive(Debug, Clone)]
pub struct SnapshotOptions {
    pub output_dir: PathBuf,
    pub stride: u64,
    pub signals: Vec<SignalId>,
    pub fields: Vec<FieldId>,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub steps: u64,
    pub print_signals: bool,
    pub signals: Vec<SignalId>,
    pub snapshot: Option<SnapshotOptions>,
}

#[derive(Debug, Clone)]
pub struct RunReport {
    pub run_dir: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("runtime execution failed: {0}")]
    Execution(String),
    #[error("snapshot write failed: {0}")]
    Snapshot(String),
}

pub fn run_simulation(
    runtime: &mut Runtime,
    options: RunOptions,
) -> std::result::Result<RunReport, RunError> {
    let mut run_dir: Option<PathBuf> = None;

    if let Some(snapshot) = &options.snapshot {
        let run_id = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|t| t.as_secs())
                .unwrap_or(0)
        );
        let dir = snapshot.output_dir.join(&run_id);
        std::fs::create_dir_all(&dir).map_err(|e| RunError::Snapshot(e.to_string()))?;

        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|t| t.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string());
        let manifest = RunManifest {
            run_id: run_id.clone(),
            created_at,
            seed: snapshot.seed,
            steps: options.steps,
            stride: snapshot.stride,
            signals: snapshot.signals.iter().map(|id| id.to_string()).collect(),
            fields: snapshot.fields.iter().map(|id| id.to_string()).collect(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| RunError::Snapshot(e.to_string()))?;
        std::fs::write(dir.join("run.json"), manifest_json)
            .map_err(|e| RunError::Snapshot(e.to_string()))?;
        run_dir = Some(dir);
    }

    if !runtime.is_warmup_complete() {
        runtime
            .execute_warmup()
            .map_err(|e| RunError::Execution(e.to_string()))?;
    }

    let write_snapshot = |step: u64, runtime: &mut Runtime| -> std::result::Result<(), RunError> {
        let Some(snapshot) = &options.snapshot else {
            return Ok(());
        };
        let mut signal_values = std::collections::HashMap::new();
        for id in &snapshot.signals {
            if let Some(val) = runtime.get_signal(id) {
                signal_values.insert(id.to_string(), val.clone());
            }
        }

        let mut field_samples = std::collections::HashMap::new();
        for id in &snapshot.fields {
            if let Some(samples) = runtime.field_buffer().get_samples(id) {
                if !samples.is_empty() {
                    field_samples.insert(id.to_string(), samples.to_vec());
                }
            }
        }

        let tick_snapshot = TickSnapshot {
            tick: runtime.tick(),
            time_seconds: runtime.sim_time(),
            signals: signal_values,
            fields: field_samples,
        };

        let encoded =
            bincode::serialize(&tick_snapshot).map_err(|e| RunError::Snapshot(e.to_string()))?;
        let filename = format!("tick_{:010}.bin", step);
        let path = run_dir.as_ref().unwrap().join(filename);
        std::fs::write(path, encoded).map_err(|e| RunError::Snapshot(e.to_string()))?;

        Ok(())
    };

    for i in 0..options.steps {
        runtime
            .execute_tick()
            .map_err(|e| RunError::Execution(e.to_string()))?;

        if options.print_signals {
            let mut line = format!("Tick {:04}: ", runtime.tick());
            for id in &options.signals {
                if let Some(val) = runtime.get_signal(id) {
                    line.push_str(&format!("{}={} ", id, val));
                }
            }
            println!("{}", line);
        }

        if let Some(snapshot) = &options.snapshot {
            if i % snapshot.stride == 0 {
                write_snapshot(i, runtime)?;
            }
        }
    }

    Ok(RunReport { run_dir })
}

/// Era configuration
pub struct EraConfig {
    /// Time step for this era
    pub dt: Dt,
    /// Stratum states in this era (IndexMap for deterministic iteration order)
    pub strata: IndexMap<StratumId, StratumState>,
    /// Transition condition (returns Some(next_era) if should transition)
    pub transition: Option<TransitionFn>,
}

/// Function that evaluates era transition
pub type TransitionFn =
    Box<dyn Fn(&SignalStorage, &EntityStorage, f64) -> Option<EraId> + Send + Sync>;

/// Function that resolves an aggregate value for a signal
pub type AggregateResolverFn = Box<
    dyn Fn(&SignalStorage, &EntityStorage, &MemberSignalBuffer, Dt, f64) -> Value + Send + Sync,
>;

/// The Continuum runtime orchestrates simulation execution.
pub struct Runtime {
    /// Storage for global signals
    signals: SignalStorage,
    /// Storage for entity instances
    entities: EntityStorage,
    /// Buffers for member signals (SoA storage)
    member_signals: MemberSignalBuffer,
    /// Channels for accumulating inputs before resolution
    input_channels: InputChannels,
    /// Buffer for emitted fields
    field_buffer: FieldBuffer,
    /// Buffer for emitted events
    event_buffer: EventBuffer,
    /// Queue for signals emitted during the fracture phase
    fracture_queue: FractureQueue,
    /// Current simulation tick
    tick: u64,
    /// Accumulated simulation time in seconds
    sim_time: f64,
    /// Current era
    current_era: EraId,
    /// Current phase within the tick
    current_phase: crate::types::Phase,
    /// Era configurations (IndexMap for deterministic iteration order)
    eras: IndexMap<EraId, EraConfig>,
    /// Execution DAGs
    dags: DagSet,
    /// Phase executor
    phase_executor: crate::executor::phases::PhaseExecutor,
    /// Warmup executor
    warmup_executor: crate::executor::warmup::WarmupExecutor,
    /// Assertion checker
    assertion_checker: AssertionChecker,
    /// Registered breakpoints (SignalId)
    breakpoints: std::collections::HashSet<SignalId>,
    /// Active tick context (frozen at start of tick)
    active_tick_ctx: Option<TickContext>,
    /// Pending impulses to apply in next Collect phase (handler_idx, payload)
    pending_impulses: Vec<(usize, Value)>,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(initial_era: EraId, eras: IndexMap<EraId, EraConfig>, dags: DagSet) -> Self {
        info!(era = %initial_era, "runtime created");
        Self {
            signals: SignalStorage::default(),
            entities: EntityStorage::default(),
            member_signals: MemberSignalBuffer::new(),
            input_channels: InputChannels::default(),
            field_buffer: FieldBuffer::default(),
            event_buffer: EventBuffer::default(),
            fracture_queue: FractureQueue::default(),
            tick: 0,
            sim_time: 0.0,
            current_era: initial_era,
            current_phase: crate::types::Phase::Configure,
            eras,
            dags,
            phase_executor: crate::executor::phases::PhaseExecutor::new(),
            warmup_executor: crate::executor::warmup::WarmupExecutor::new(),
            assertion_checker: AssertionChecker::new(),
            breakpoints: std::collections::HashSet::new(),
            active_tick_ctx: None,
            pending_impulses: Vec::new(),
        }
    }

    /// Add a breakpoint for a signal
    pub fn add_breakpoint(&mut self, signal: SignalId) {
        info!(%signal, "breakpoint added");
        self.breakpoints.insert(signal);
    }

    /// Remove a breakpoint for a signal
    pub fn remove_breakpoint(&mut self, signal: &SignalId) {
        info!(%signal, "breakpoint removed");
        self.breakpoints.remove(signal);
    }

    /// Clear all breakpoints
    pub fn clear_breakpoints(&mut self) {
        info!("all breakpoints cleared");
        self.breakpoints.clear();
    }

    /// Check if a signal has a breakpoint
    pub fn has_breakpoint(&self, signal: &SignalId) -> bool {
        self.breakpoints.contains(signal)
    }

    /// Register a resolver function, returns its index
    pub fn register_resolver(&mut self, resolver: crate::executor::phases::ResolverFn) -> usize {
        self.phase_executor.register_resolver(resolver)
    }

    /// Register a collect operator, returns its index
    pub fn register_collect_op(&mut self, op: crate::executor::phases::CollectFn) -> usize {
        self.phase_executor.register_collect_op(op)
    }

    /// Register a fracture function, returns its index
    pub fn register_fracture(&mut self, fracture: crate::executor::phases::FractureFn) -> usize {
        self.phase_executor.register_fracture(fracture)
    }

    /// Register a measure operator, returns its index
    pub fn register_measure_op(&mut self, op: crate::executor::phases::MeasureFn) -> usize {
        self.phase_executor.register_measure_op(op)
    }

    /// Register a chronicle handler, returns its index
    pub fn register_chronicle(&mut self, handler: crate::executor::phases::ChronicleFn) -> usize {
        self.phase_executor.register_chronicle(handler)
    }

    /// Register an impulse handler, returns its index
    pub fn register_impulse(&mut self, handler: crate::executor::phases::ImpulseFn) -> usize {
        self.phase_executor.register_impulse(handler)
    }

    /// Inject an impulse to be applied in the next tick's Collect phase
    pub fn inject_impulse(&mut self, handler_idx: usize, payload: Value) {
        tracing::debug!(handler_idx, ?payload, "impulse injected");
        self.pending_impulses.push((handler_idx, payload));
    }

    /// Register a warmup function for a signal
    pub fn register_warmup(
        &mut self,
        signal: SignalId,
        warmup_fn: crate::executor::warmup::WarmupFn,
        config: WarmupConfig,
    ) {
        self.warmup_executor.register(signal, warmup_fn, config);
    }

    /// Register an assertion for a signal
    pub fn register_assertion(
        &mut self,
        signal: SignalId,
        condition: AssertionFn,
        severity: AssertionSeverity,
        message: Option<String>,
    ) {
        self.assertion_checker
            .register(signal, condition, severity, message);
    }

    /// Initialize a signal with a value
    pub fn init_signal(&mut self, id: SignalId, value: Value) {
        tracing::debug!(signal = %id, ?value, "signal initialized");
        self.signals.init(id, value);
    }

    /// Initialize an entity type with its instances
    pub fn init_entity(&mut self, id: EntityId, instances: EntityInstances) {
        let count = instances.count();
        tracing::debug!(entity = %id, count, "entity initialized");
        self.entities.init_entity(id, instances);
    }

    /// Register a member signal type
    pub fn register_member_signal(&mut self, signal_name: &str, value_type: MemberValueType) {
        tracing::debug!(
            signal = signal_name,
            ?value_type,
            "member signal registered"
        );
        self.member_signals
            .register_signal(signal_name.to_string(), value_type);
    }

    /// Initialize storage for all registered member signals
    pub fn init_member_instances(&mut self, instance_count: usize) {
        tracing::debug!(count = instance_count, "member instances initialized");
        self.member_signals.init_instances(instance_count);
    }

    /// Register the instance count for a specific entity.
    pub fn register_entity_count(&mut self, entity_id: &str, count: usize) {
        tracing::debug!(
            entity = entity_id,
            count,
            "entity instance count registered"
        );
        self.member_signals.register_entity_count(entity_id, count);
    }

    /// Register a scalar member resolver function
    pub fn register_member_resolver(&mut self, _signal_name: String, resolver: ScalarResolverFn) {
        tracing::debug!(signal = %_signal_name, "scalar member resolver registered");
        self.phase_executor
            .register_member_resolver(crate::executor::phases::MemberResolver::Scalar(resolver));
    }

    /// Register a Vec3 member resolver function
    pub fn register_vec3_member_resolver(
        &mut self,
        _signal_name: String,
        resolver: Vec3ResolverFn,
    ) {
        tracing::debug!(signal = %_signal_name, "vec3 member resolver registered");
        self.phase_executor
            .register_member_resolver(crate::executor::phases::MemberResolver::Vec3(resolver));
    }

    /// Register an aggregate resolver
    pub fn register_aggregate_resolver(
        &mut self,
        _signal_id: SignalId,
        resolver: AggregateResolverFn,
    ) {
        self.phase_executor.register_aggregate_resolver(resolver);
    }

    /// Set a member signal value for an instance
    pub fn set_member_signal(
        &mut self,
        signal_name: &str,
        instance_idx: usize,
        value: Value,
    ) -> std::result::Result<(), String> {
        self.member_signals
            .set_current(signal_name, instance_idx, value)
    }

    /// Commit all member initial values (swap buffers)
    pub fn commit_member_initials(&mut self) {
        self.member_signals.advance_tick();
    }

    /// Get current simulation tick
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get accumulated simulation time in seconds
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// Get current simulation era
    pub fn era(&self) -> &EraId {
        &self.current_era
    }

    /// Get value of a signal by ID
    pub fn get_signal(&self, id: &SignalId) -> Option<&Value> {
        self.signals.get_resolved(id)
    }

    /// Check if warmup has been executed
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_executor.is_complete()
    }

    /// Get access to the field buffer
    pub fn field_buffer(&self) -> &FieldBuffer {
        &self.field_buffer
    }

    /// Get access to the event buffer
    pub fn event_buffer(&self) -> &EventBuffer {
        &self.event_buffer
    }

    /// Drain the event buffer
    pub fn drain_events(&mut self) -> Vec<EmittedEventRecord> {
        self.event_buffer.drain()
    }

    /// Get access to the signals storage
    pub fn signals(&self) -> &SignalStorage {
        &self.signals
    }

    /// Get access to the entities storage
    pub fn entities(&self) -> &EntityStorage {
        &self.entities
    }

    /// Get access to the entities storage mutably
    pub fn entities_mut(&mut self) -> &mut EntityStorage {
        &mut self.entities
    }

    /// Get access to the member signals buffer
    pub fn member_signals(&self) -> &MemberSignalBuffer {
        &self.member_signals
    }

    /// Execute warmup phase
    #[instrument(skip_all, name = "warmup")]
    pub fn execute_warmup(&mut self) -> Result<WarmupResult> {
        self.warmup_executor
            .execute(&mut self.signals, &self.entities, self.sim_time)
    }

    /// Get current phase
    pub fn phase(&self) -> crate::types::Phase {
        self.current_phase
    }

    /// Execute a single phase of the simulation.
    pub fn execute_step(&mut self) -> Result<crate::types::StepResult> {
        let (dt, strata_states) = {
            let era_config = self
                .eras
                .get(&self.current_era)
                .ok_or_else(|| Error::EraNotFound(self.current_era.clone()))?;
            (era_config.dt, era_config.strata.clone())
        };

        match self.current_phase {
            crate::types::Phase::Configure => {
                self.active_tick_ctx = Some(TickContext {
                    tick: self.tick,
                    sim_time: self.sim_time,
                    dt,
                    era: self.current_era.clone(),
                });
                self.current_phase = crate::types::Phase::Collect;
            }
            crate::types::Phase::Collect => {
                trace!("phase: collect");
                self.phase_executor.execute_collect(
                    &self.current_era,
                    self.tick,
                    dt,
                    self.sim_time,
                    &strata_states,
                    &self.dags,
                    &self.signals,
                    &self.entities,
                    &mut self.input_channels,
                    &mut self.pending_impulses,
                )?;
                self.current_phase = crate::types::Phase::Resolve;
            }
            crate::types::Phase::Resolve => {
                trace!("phase: resolve");
                if let Some(signal) = self.phase_executor.execute_resolve(
                    &self.current_era,
                    self.tick,
                    dt,
                    self.sim_time,
                    &strata_states,
                    &self.dags,
                    &mut self.signals,
                    &self.entities,
                    &mut self.member_signals,
                    &mut self.input_channels,
                    &self.assertion_checker,
                    &self.breakpoints,
                )? {
                    return Ok(crate::types::StepResult::Breakpoint { signal });
                }
                self.current_phase = crate::types::Phase::Fracture;
            }
            crate::types::Phase::Fracture => {
                trace!("phase: fracture");
                self.phase_executor.execute_fracture(
                    &self.current_era,
                    dt,
                    self.sim_time,
                    &self.dags,
                    &self.signals,
                    &self.entities,
                    &mut self.fracture_queue,
                )?;
                self.current_phase = crate::types::Phase::Measure;
            }
            crate::types::Phase::Measure => {
                trace!("phase: measure");
                self.phase_executor.execute_measure(
                    &self.current_era,
                    self.tick,
                    dt,
                    self.sim_time,
                    &strata_states,
                    &self.dags,
                    &self.signals,
                    &self.entities,
                    &mut self.field_buffer,
                )?;

                self.phase_executor.execute_chronicles(
                    &self.current_era,
                    self.tick,
                    dt,
                    &strata_states,
                    &self.dags,
                    &self.signals,
                    &self.entities,
                    &mut self.event_buffer,
                )?;
                self.current_phase = crate::types::Phase::EraTransition;
            }
            crate::types::Phase::EraTransition => {
                trace!("phase: era transition");
                self.check_era_transition()?;
                self.current_phase = crate::types::Phase::PostTick;
            }
            crate::types::Phase::PostTick => {
                trace!("phase: post tick");
                self.signals.advance_tick();
                self.entities.advance_tick();
                self.member_signals.advance_tick();
                self.fracture_queue.drain_into(&mut self.input_channels);
                self.sim_time += dt.seconds();
                self.tick += 1;
                self.current_phase = crate::types::Phase::Configure;
            }
        }

        if self.current_phase == crate::types::Phase::Configure {
            let ctx = self
                .active_tick_ctx
                .take()
                .expect("tick context missing at end of tick");
            Ok(crate::types::StepResult::TickCompleted(ctx))
        } else {
            Ok(crate::types::StepResult::Continue)
        }
    }

    /// Execute a single tick
    #[instrument(skip(self), fields(tick = self.tick, era = %self.current_era))]
    pub fn execute_tick(&mut self) -> Result<TickContext> {
        loop {
            match self.execute_step()? {
                crate::types::StepResult::TickCompleted(ctx) => return Ok(ctx),
                crate::types::StepResult::Breakpoint { signal } => {
                    return Err(Error::Generic(format!(
                        "Hit breakpoint on signal '{}' during tick",
                        signal
                    )));
                }
                crate::types::StepResult::Continue => continue,
            }
        }
    }

    /// Execute until a breakpoint or tick completed
    pub fn execute_until_breakpoint(&mut self) -> Result<crate::types::StepResult> {
        loop {
            let result = self.execute_step()?;
            match result {
                crate::types::StepResult::Breakpoint { .. } => return Ok(result),
                crate::types::StepResult::TickCompleted(..) => return Ok(result),
                crate::types::StepResult::Continue => continue,
            }
        }
    }

    fn check_era_transition(&mut self) -> Result<()> {
        let era_config = self.eras.get(&self.current_era).unwrap();
        if let Some(ref transition) = era_config.transition
            && let Some(next_era) = transition(&self.signals, &self.entities, self.sim_time)
        {
            if !self.eras.contains_key(&next_era) {
                error!(era = %next_era, "transition to unknown era");
                return Err(Error::EraNotFound(next_era));
            }
            info!(from = %self.current_era, to = %next_era, "era transition");
            self.current_era = next_era;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StratumId;

    #[test]
    fn test_runtime_creation() {
        let era_id: EraId = "default".into();
        let mut eras = IndexMap::new();
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata: IndexMap::new(),
                transition: None,
            },
        );
        let runtime = Runtime::new(era_id, eras, DagSet::default());
        assert_eq!(runtime.tick(), 0);
        assert_eq!(runtime.sim_time(), 0.0);
    }
}
