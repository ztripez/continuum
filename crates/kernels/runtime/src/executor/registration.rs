//! Runtime registration and initialization methods.
//!
//! Separated from runtime core to keep the Runtime struct definition and
//! phase execution logic focused. This module contains all methods that
//! configure the runtime before or between ticks:
//! - Config/const value loading
//! - Signal type and topology initialization
//! - Signal, entity, and member signal registration
//! - Breakpoint management
//! - Impulse mapping and injection
//! - Resolver, operator, and assertion registration
//! - Warmup registration

use indexmap::IndexMap;

use crate::error::Result;
use crate::soa_storage::ValueType as MemberValueType;
use crate::storage::EntityInstances;
use crate::types::{EntityId, ImpulseId, SignalId, Value, WarmupConfig};

use super::assertions::AssertionFn;
use super::member_executor::{ScalarResolverFn, Vec3ResolverFn};
use super::runtime::{AggregateResolverFn, Runtime};
use crate::types::AssertionSeverity;

impl Runtime {
    /// Loads configuration values into the runtime.
    ///
    /// Configuration values are world-level defaults declared in `config{}` blocks
    /// that can be overridden by scenarios. They are frozen after loading and remain
    /// immutable throughout execution.
    ///
    /// # Lifecycle
    ///
    /// Called during `build_runtime()` (stage 4: Scenario Application) after compiling
    /// the world but before warmup. Values are extracted from world defaults and merged
    /// with scenario overrides (when implemented).
    ///
    /// # Access
    ///
    /// Config values are accessible in all DSL phases via `load_config("path")` and
    /// the `LoadConfig` opcode. They behave like frozen parameters, not signals (no
    /// prev/current distinction).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In world DSL:
    /// config { physics.gravity: 9.81 }
    ///
    /// // In signal resolve:
    /// signal velocity {
    ///     resolve { prev - load_config("physics.gravity") * dt }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The caller (`build_runtime`) panics if any config value contains a non-literal
    /// expression (enforcing Fail Loudly).
    pub fn set_config_values(&mut self, values: IndexMap<continuum_foundation::Path, Value>) {
        self.bytecode_executor.set_config_values(values);
    }

    /// Loads constant values into the runtime.
    ///
    /// Constant values are world-level immutable globals declared in `const{}` blocks.
    /// Unlike config, constants are NOT scenario-overridable. They are frozen after
    /// loading and remain immutable throughout execution.
    ///
    /// # Lifecycle
    ///
    /// Called during `build_runtime()` (stage 4: Scenario Application) after compiling
    /// the world but before warmup. Values are extracted from world const declarations.
    ///
    /// # Access
    ///
    /// Const values are accessible in all DSL phases via `load_const("path")` and
    /// the `LoadConst` opcode. They behave like frozen parameters, not signals (no
    /// prev/current distinction).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In world DSL:
    /// const { physics.stefan_boltzmann: 5.67e-8 }
    ///
    /// // In signal resolve:
    /// signal radiation {
    ///     resolve {
    ///         load_const("physics.stefan_boltzmann") * temp^4
    ///     }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The caller (`build_runtime`) panics if any const value contains a non-literal
    /// expression (enforcing Fail Loudly).
    pub fn set_const_values(&mut self, values: IndexMap<continuum_foundation::Path, Value>) {
        self.bytecode_executor.set_const_values(values);
    }

    /// Stores signal type information for zero value initialization.
    ///
    /// Called during `build_runtime()` (stage 4: Scenario Application) after compiling
    /// signal blocks and extracting their types. This must happen before execution begins
    /// to ensure correct zero value initialization for signal inputs.
    ///
    /// Signal types are used by the bytecode executor to create correct zero values
    /// when no inputs have been accumulated for a signal.
    pub fn set_signal_types(
        &mut self,
        types: IndexMap<crate::types::SignalId, continuum_cdsl::foundation::Type>,
    ) {
        self.bytecode_executor.set_signal_types(types);
    }

    /// Initializes spatial topologies from entity topology expressions.
    ///
    /// Called during `build_runtime()` after signal registration to populate
    /// topology storage for spatial.* kernel queries.
    pub fn initialize_topologies(
        &mut self,
        entities: &IndexMap<continuum_foundation::Path, continuum_cdsl::ast::Entity>,
    ) {
        self.bytecode_executor.initialize_topologies(entities);
    }

    /// Add a breakpoint for a signal
    pub fn add_breakpoint(&mut self, signal: SignalId) {
        tracing::info!(%signal, "breakpoint added");
        self.breakpoints.insert(signal);
    }

    /// Remove a breakpoint for a signal
    pub fn remove_breakpoint(&mut self, signal: &SignalId) {
        tracing::info!(%signal, "breakpoint removed");
        self.breakpoints.remove(signal);
    }

    /// Clear all breakpoints
    pub fn clear_breakpoints(&mut self) {
        tracing::info!("all breakpoints cleared");
        self.breakpoints.clear();
    }

    /// Add a mapping for an impulse ID to its bytecode block index
    pub fn add_impulse_mapping(&mut self, id: ImpulseId, idx: usize) {
        self.impulse_map.insert(id, idx);
    }

    /// Inject an impulse by its ID
    pub fn inject_impulse_by_id(&mut self, id: &ImpulseId, payload: Value) -> Result<()> {
        let idx = self
            .impulse_map
            .get(id)
            .ok_or_else(|| crate::error::Error::Generic(format!("Impulse '{}' not found", id)))?;
        self.inject_impulse(*idx, payload);
        Ok(())
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

    /// Initialize a global signal with a value.
    ///
    /// Global signals are stored in `MemberSignalBuffer` under the root entity.
    /// If the signal is not yet registered (e.g. in tests that bypass `build_runtime`),
    /// it is auto-registered with a `ValueType` inferred from the value.
    ///
    /// # Panics
    ///
    /// Panics if storage initialization fails (type mismatch, unsupported type).
    pub fn init_signal(&mut self, id: SignalId, value: Value) {
        tracing::debug!(signal = %id, ?value, "signal initialized");

        // Auto-register the signal if not already known.
        // This handles the test path where init_signal is called without
        // prior register_root_entity / register_global_signal / init_instances.
        let value_type = crate::soa_storage::ValueType::from_value(&value);
        self.storage
            .member_signals
            .ensure_global_signal(&id.to_string(), value_type);

        self.storage
            .member_signals
            .init_global(&id.to_string(), value)
            .unwrap_or_else(|e| panic!("failed to init global signal '{}': {}", id, e));
    }

    /// Initialize an entity type with its instances
    pub fn init_entity(&mut self, id: EntityId, instances: EntityInstances) {
        let count = instances.count();
        tracing::debug!(entity = %id, count, "entity initialized");
        self.storage.entities.init_entity(id, instances);
    }

    /// Register the root entity for global signals.
    ///
    /// Must be called before `init_member_instances`. Idempotent.
    pub fn register_root_entity(&mut self) {
        tracing::debug!("root entity registered for global signals");
        self.storage.member_signals.register_root_entity();
    }

    /// Register a global signal in the member signal buffer.
    ///
    /// Must be called before `init_member_instances`.
    pub fn register_global_signal(&mut self, signal_name: &str, value_type: MemberValueType) {
        tracing::debug!(
            signal = signal_name,
            ?value_type,
            "global signal registered in member buffer"
        );
        self.storage
            .member_signals
            .register_global_signal(signal_name, value_type);
    }

    /// Register a member signal type
    pub fn register_member_signal(&mut self, signal_name: &str, value_type: MemberValueType) {
        tracing::debug!(
            signal = signal_name,
            ?value_type,
            "member signal registered"
        );
        self.storage
            .member_signals
            .register_signal(signal_name.to_string(), value_type);
    }

    /// Initialize storage for all registered member signals
    pub fn init_member_instances(&mut self, instance_count: usize) {
        tracing::debug!(count = instance_count, "member instances initialized");
        self.storage.member_signals.init_instances(instance_count);
    }

    /// Register the instance count for a specific entity.
    pub fn register_entity_count(&mut self, entity_id: &str, count: usize) {
        tracing::debug!(
            entity = entity_id,
            count,
            "entity instance count registered"
        );
        self.storage
            .member_signals
            .register_entity_count(entity_id, count);
    }

    /// Register a scalar member resolver function
    pub fn register_member_resolver(&mut self, _signal_name: String, resolver: ScalarResolverFn) {
        tracing::debug!(signal = %_signal_name, "scalar member resolver registered");
        self.phase_executor
            .register_scalar_member_resolver(resolver);
    }

    /// Register a Vec3 member resolver function
    pub fn register_vec3_member_resolver(
        &mut self,
        _signal_name: String,
        resolver: Vec3ResolverFn,
    ) {
        tracing::debug!(signal = %_signal_name, "vec3 member resolver registered");
        self.phase_executor.register_vec3_member_resolver(resolver);
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
        self.storage
            .member_signals
            .set_current(signal_name, instance_idx, value)
    }
}
