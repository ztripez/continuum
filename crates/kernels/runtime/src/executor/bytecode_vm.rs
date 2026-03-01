//! VM execution context bridging bytecode execution to runtime storage.
//!
//! This module contains [`VMContext`] and [`EntityContext`], which implement
//! the [`ExecutionContext`] trait to connect the bytecode interpreter to
//! the runtime's signal storage, entity storage, and I/O buffers.
//!
//! Separated from [`super::bytecode`] to keep the phase orchestration
//! logic distinct from the per-opcode execution semantics.

use crate::bytecode::runtime::{ExecutionContext, ExecutionError};
use crate::executor::bytecode::VMConfig;
use crate::reductions;
use crate::soa_storage::MemberSignalBuffer;
use crate::storage::{EntityStorage, EventBuffer, FieldBuffer, FractureQueue, InputChannels};
use crate::types::{Dt, EraId, Phase, SignalId, Value};
use continuum_cdsl::foundation::{Shape, Type};
use continuum_foundation::{AggregateOp, EntityId, Mat2, Mat3, Mat4, Path};
use indexmap::IndexMap;

/// Returns the zero/identity value for a given signal type.
///
/// Used when no inputs have been accumulated for a signal during Resolve,
/// to produce a correct typed zero instead of a bare `Scalar(0.0)`.
///
/// # Panics
///
/// Panics for unsupported types (User, String, Unit, Seq) and unsupported
/// vector/matrix dimensions (enforcing fail-loudly).
pub(crate) fn zero_value_for_type(ty: &Type) -> Value {
    match ty {
        Type::Kernel(kt) => match &kt.shape {
            Shape::Scalar => Value::Scalar(0.0),
            Shape::Vector { dim } => match dim {
                2 => Value::Vec2([0.0, 0.0]),
                3 => Value::Vec3([0.0, 0.0, 0.0]),
                4 => Value::Vec4([0.0, 0.0, 0.0, 0.0]),
                _ => panic!(
                    "Unsupported vector dimension for inputs zero value: {}",
                    dim
                ),
            },
            Shape::Matrix { rows, cols } => match (rows, cols) {
                (2, 2) => Value::Mat2(Mat2([0.0; 4])),
                (3, 3) => Value::Mat3(Mat3([0.0; 9])),
                (4, 4) => Value::Mat4(Mat4([0.0; 16])),
                _ => panic!(
                    "Unsupported matrix dimensions for inputs zero value: {}x{}",
                    rows, cols
                ),
            },
            _ => panic!("Unsupported shape for inputs zero value: {:?}", kt.shape),
        },
        Type::Bool => Value::Boolean(false),
        Type::User(_) | Type::String | Type::Unit | Type::Seq(_) => {
            panic!("Unsupported type for inputs zero value: {:?}", ty)
        }
    }
}

/// Implementation of the VM execution context that bridges to runtime storage.
#[allow(dead_code)]
pub struct VMContext<'a> {
    /// Current execution phase
    pub phase: Phase,
    /// Current era identifier
    pub era: &'a EraId,
    /// Time step for the current tick
    pub dt: Dt,
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
    /// Entity instance storage
    pub entities: &'a EntityStorage,
    /// Member signal buffer (SoA per-entity state)
    pub member_signals: &'a MemberSignalBuffer,
    /// Input channels for accumulating signal contributions
    pub channels: &'a mut InputChannels,
    /// Queue for fracture-phase signal emissions (next tick)
    pub fracture_queue: &'a mut FractureQueue,
    /// Buffer for field sample emissions (Measure phase)
    pub field_buffer: &'a mut FieldBuffer,
    /// Buffer for chronicle event emissions (Measure phase)
    pub event_buffer: &'a mut EventBuffer,
    /// Target signal being resolved (if in Resolve phase)
    pub target_signal: Option<SignalId>,
    /// Cached input accumulation result (avoid re-draining)
    pub cached_inputs: Option<Value>,
    /// World configuration values loaded from config {} blocks
    pub config_values: &'a IndexMap<Path, Value>,
    /// Global simulation constants loaded from const {} blocks
    pub const_values: &'a IndexMap<Path, Value>,
    /// Spatial topologies for neighbor queries
    pub topologies: &'a crate::topology::TopologyStorage,
    /// Signal types for zero value initialization
    pub signal_types: &'a IndexMap<SignalId, Type>,
    /// Current impulse payload (only valid when executing impulse collect blocks)
    pub payload: Option<&'a Value>,
    /// Current entity context for member signal execution
    /// When Some, enables LoadSelf/LoadOther opcodes for member signal access
    pub entity_context: Option<EntityContext<'a>>,
}

/// Entity execution context for member signal bytecode.
///
/// When executing member signal resolve/initial blocks, this provides:
/// - Entity ID (e.g., "hydrology.cell")
/// - Instance index (0..N for N instances)
/// - Access to member signal buffer for self/other member reads
#[derive(Debug, Clone, Copy)]
pub struct EntityContext<'a> {
    /// The entity ID (e.g., "hydrology.cell")
    pub entity_id: &'a EntityId,
    /// The instance index being executed (0..instance_count)
    pub instance_index: usize,
    /// The member signal path being resolved (e.g., "hydrology.cell.temperature")
    pub target_member: &'a Path,
}

impl<'a> VMContext<'a> {
    /// Construct a new VM execution context.
    ///
    /// The `config` parameter carries the four immutable executor-level references
    /// (`config_values`, `const_values`, `topologies`, `signal_types`) that are
    /// identical across every VMContext in a phase method. Callers supply only the
    /// per-node / per-phase varying fields directly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: &'a VMConfig<'a>,
        phase: Phase,
        era: &'a EraId,
        dt: Dt,
        sim_time: f64,
        entities: &'a EntityStorage,
        member_signals: &'a MemberSignalBuffer,
        channels: &'a mut InputChannels,
        fracture_queue: &'a mut FractureQueue,
        field_buffer: &'a mut FieldBuffer,
        event_buffer: &'a mut EventBuffer,
        target_signal: Option<SignalId>,
        cached_inputs: Option<Value>,
        payload: Option<&'a Value>,
        entity_context: Option<EntityContext<'a>>,
    ) -> Self {
        Self {
            phase,
            era,
            dt,
            sim_time,
            entities,
            member_signals,
            channels,
            fracture_queue,
            field_buffer,
            event_buffer,
            target_signal,
            cached_inputs,
            config_values: config.config_values,
            const_values: config.const_values,
            topologies: config.topologies,
            signal_types: config.signal_types,
            payload,
            entity_context,
        }
    }
}

impl<'a> ExecutionContext for VMContext<'a> {
    fn phase(&self) -> Phase {
        self.phase
    }

    fn load_signal(&self, path: &Path) -> std::result::Result<Value, ExecutionError> {
        self.member_signals
            .get_global_or_prev(&path.to_string())
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Signal not found: {}", path),
            })
    }

    fn load_field(&self, _path: &Path) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadField not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn load_config(&self, path: &Path) -> std::result::Result<Value, ExecutionError> {
        self.config_values
            .get(path)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Config value not found: {}", path),
            })
    }

    fn load_const(&self, path: &Path) -> std::result::Result<Value, ExecutionError> {
        self.const_values
            .get(path)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Const value not found: {}", path),
            })
    }

    fn load_prev(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Resolve && self.phase != Phase::Configure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadPrev".to_string(),
                phase: self.phase,
            });
        }

        // Check if we're in entity context (member signal)
        if let Some(entity_ctx) = &self.entity_context {
            // Load from member signal buffer
            let full_path = entity_ctx.target_member.to_string();
            self.member_signals
                .get_previous(&full_path, entity_ctx.instance_index)
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!(
                        "Member signal '{}' instance {} has no previous value",
                        full_path, entity_ctx.instance_index
                    ),
                })
        } else {
            // Load from global signal storage
            let signal =
                self.target_signal
                    .as_ref()
                    .ok_or_else(|| ExecutionError::InvalidOpcode {
                        opcode: "LoadPrev requires target signal or entity context".to_string(),
                        phase: self.phase,
                    })?;
            self.member_signals
                .get_global_prev(&signal.to_string())
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!("Signal '{}' has no previous value", signal),
                })
        }
    }

    fn load_current(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Fracture && self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadCurrent".to_string(),
                phase: self.phase,
            });
        }

        // Check if we're in entity context (member signal)
        if let Some(entity_ctx) = &self.entity_context {
            // Load from member signal buffer
            let full_path = entity_ctx.target_member.to_string();
            self.member_signals
                .get_current(&full_path, entity_ctx.instance_index)
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!(
                        "Member signal '{}' instance {} has no current value",
                        full_path, entity_ctx.instance_index
                    ),
                })
        } else {
            // Load from global signal storage
            let signal =
                self.target_signal
                    .as_ref()
                    .ok_or_else(|| ExecutionError::InvalidOpcode {
                        opcode: "LoadCurrent requires target signal or entity context".to_string(),
                        phase: self.phase,
                    })?;
            self.member_signals
                .get_global_or_prev(&signal.to_string())
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!("Signal '{}' has no resolved value", signal),
                })
        }
    }

    fn load_inputs(&mut self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Resolve {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadInputs".to_string(),
                phase: self.phase,
            });
        }

        // Check if we're in entity context (member signal)
        if let Some(entity_ctx) = &self.entity_context {
            // Member signal - load from per-instance input channels
            if let Some(cached) = &self.cached_inputs {
                return Ok(cached.clone());
            }

            let value = self.channels.drain_member_sum(
                entity_ctx.entity_id,
                entity_ctx.instance_index as u32,
                entity_ctx.target_member,
            );

            // Create the appropriate Value based on the signal type
            // Look up member signal type using the full member path
            let signal_id = SignalId::from(entity_ctx.target_member.clone());
            let result = if value == 0.0 {
                // If no inputs were accumulated, return typed zero
                if let Some(ty) = self.signal_types.get(&signal_id) {
                    zero_value_for_type(ty)
                } else {
                    Value::Scalar(0.0)
                }
            } else {
                Value::Scalar(value)
            };

            self.cached_inputs = Some(result.clone());
            return Ok(result);
        }

        // Global signal - load from input channels
        let signal = self
            .target_signal
            .as_ref()
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadInputs requires target signal or entity context".to_string(),
                phase: self.phase,
            })?;

        if let Some(cached) = &self.cached_inputs {
            return Ok(cached.clone());
        }

        let value = self.channels.drain_sum(signal);

        // Create the appropriate Value based on the signal type
        let result = if value == 0.0 {
            // If no inputs were accumulated, return typed zero
            if let Some(ty) = self.signal_types.get(signal) {
                zero_value_for_type(ty)
            } else {
                Value::Scalar(0.0)
            }
        } else {
            Value::Scalar(value)
        };

        self.cached_inputs = Some(result.clone());
        Ok(result)
    }

    fn load_dt(&self) -> std::result::Result<Value, ExecutionError> {
        Ok(Value::Scalar(self.dt.0))
    }

    fn load_self(&self) -> std::result::Result<Value, ExecutionError> {
        let _entity_ctx = self
            .entity_context
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadSelf requires entity context (member signal execution)".to_string(),
                phase: self.phase,
            })?;

        // LoadSelf returns the EntitySelf marker, which FieldAccess recognizes
        // to route member signal access instead of normal field access.
        Ok(Value::EntitySelf)
    }

    fn load_other(&self) -> std::result::Result<Value, ExecutionError> {
        let entity_ctx = self
            .entity_context
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadOther requires entity context (member signal execution)".to_string(),
                phase: self.phase,
            })?;

        // LoadOther is used for accessing other member signals of the same entity instance
        // The path is determined by the opcode operand (handled by the opcode handler)
        // This stub should not be called directly - the opcode handler resolves the path
        Err(ExecutionError::InvalidOperand {
            message: format!(
                "LoadOther requires path operand (entity: {}, instance: {})",
                entity_ctx.entity_id, entity_ctx.instance_index
            ),
        })
    }

    fn load_payload(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Collect {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadPayload".to_string(),
                phase: self.phase,
            });
        }
        self.payload
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "LoadPayload called but no impulse payload is available. This block should only execute when an impulse is triggered.".to_string(),
            })
    }

    fn find_nearest(
        &self,
        seq: &[Value],
        position: Value,
    ) -> std::result::Result<Value, ExecutionError> {
        let pos = position
            .as_vec3()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Vec3".to_string(),
                found: format!("{:?}", position),
            })?;

        let mut nearest = None;
        let mut min_dist_sq = f64::MAX;

        for instance in seq {
            if let Some(inst_map) = instance.as_map() {
                let inst_pos_val = inst_map
                    .iter()
                    .find(|(name, _)| name == "position")
                    .map(|(_, v)| v);

                if let Some(Value::Vec3(inst_pos)) = inst_pos_val {
                    let dx = inst_pos[0] - pos[0];
                    let dy = inst_pos[1] - pos[1];
                    let dz = inst_pos[2] - pos[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        nearest = Some(instance.clone());
                    }
                }
            }
        }

        nearest.ok_or_else(|| ExecutionError::InvalidOperand {
            message: "No instances with 'position' field found in sequence for nearest lookup"
                .to_string(),
        })
    }

    fn filter_within(
        &self,
        seq: &[Value],
        position: Value,
        radius: Value,
    ) -> std::result::Result<Vec<Value>, ExecutionError> {
        let pos = position
            .as_vec3()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Vec3".to_string(),
                found: format!("{:?}", position),
            })?;
        let r = radius
            .as_scalar()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Scalar".to_string(),
                found: format!("{:?}", radius),
            })?;
        let r_sq = r * r;

        let mut filtered = Vec::new();
        for instance in seq {
            if let Some(inst_map) = instance.as_map() {
                let inst_pos_val = inst_map
                    .iter()
                    .find(|(name, _)| name == "position")
                    .map(|(_, v)| v);

                if let Some(Value::Vec3(inst_pos)) = inst_pos_val {
                    let dx = inst_pos[0] - pos[0];
                    let dy = inst_pos[1] - pos[1];
                    let dz = inst_pos[2] - pos[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq <= r_sq {
                        filtered.push(instance.clone());
                    }
                }
            }
        }
        Ok(filtered)
    }

    fn find_neighbors(
        &self,
        entity: &continuum_foundation::EntityId,
        instance: Value,
    ) -> std::result::Result<Vec<Value>, ExecutionError> {
        // Get topology for this entity
        let topology =
            self.topologies
                .get(entity)
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!("Entity {} has no topology defined", entity),
                })?;

        // Extract entity index from instance value
        // Instance is a map with an implicit index field
        let instance_map = instance
            .as_map()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Map (entity instance)".to_string(),
                found: format!("{:?}", instance),
            })?;

        // TODO: Need a standard way to get instance index
        // For now, assume instances have an "__index" field
        let index = instance_map
            .iter()
            .find(|(k, _)| k == "__index")
            .and_then(|(_, v)| v.as_scalar())
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Instance missing __index field".to_string(),
            })?;

        let entity_idx = crate::EntityIndex(index as usize);

        // Get neighbors from topology
        let neighbor_indices = topology.neighbors(entity_idx);

        // Convert neighbor indices to instance values
        // For now, return placeholder values with just index
        // TODO: Need to fetch actual instance data from EntityStorage
        let neighbors: Vec<Value> = neighbor_indices
            .iter()
            .map(|idx| {
                let fields = vec![("__index".to_string(), Value::Scalar(idx.0 as f64))];
                Value::Map(std::sync::Arc::new(fields))
            })
            .collect();

        Ok(neighbors)
    }

    fn emit_signal(
        &mut self,
        path: &Path,
        value: Value,
    ) -> std::result::Result<(), ExecutionError> {
        let val = value
            .as_scalar()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Only scalar signals supported for VM emit for now".to_string(),
            })?;

        let signal_id = SignalId::from(path.clone());

        match self.phase {
            Phase::Collect => {
                // Collect phase: inputs for THIS tick's resolve
                self.channels.accumulate(&signal_id, val);
            }
            Phase::Fracture => {
                // Fracture phase: queue for NEXT tick's resolve (alpha model)
                self.fracture_queue.queue(signal_id, val);
            }
            _ => {
                return Err(ExecutionError::InvalidOpcode {
                    opcode: "Emit signal only allowed in Collect or Fracture phase".to_string(),
                    phase: self.phase,
                });
            }
        }
        Ok(())
    }

    fn emit_member_signal(
        &mut self,
        entity: &EntityId,
        instance_idx: u32,
        member_path: &Path,
        value: Value,
    ) -> std::result::Result<(), ExecutionError> {
        // Phase validation
        if self.phase != Phase::Collect && self.phase != Phase::Fracture {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "EmitMember only allowed in Collect or Fracture phase".to_string(),
                phase: self.phase,
            });
        }

        // Extract scalar value (only scalars supported for now, like regular signals)
        let val = value
            .as_scalar()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Only scalar member signals supported for emit for now".to_string(),
            })?;

        // Get instance count for bounds checking
        let full_signal_path = member_path.to_string();
        let instance_count = self
            .member_signals
            .instance_count_for_signal(&full_signal_path);

        // Bounds check the instance index
        if instance_idx >= instance_count as u32 {
            return Err(ExecutionError::InvalidOperand {
                message: format!(
                    "Instance index {} out of bounds for entity {} (has {} instances)",
                    instance_idx, entity, instance_count
                ),
            });
        }

        // Accumulate to per-instance input channel
        self.channels
            .accumulate_member(entity, instance_idx, member_path, val);

        Ok(())
    }

    fn emit_field(
        &mut self,
        path: &Path,
        _pos: Value,
        value: Value,
    ) -> std::result::Result<(), ExecutionError> {
        if self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "Emit field only allowed in Measure phase".to_string(),
                phase: self.phase,
            });
        }
        let val = value
            .as_scalar()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Only scalar fields supported for VM emit for now".to_string(),
            })?;
        self.field_buffer
            .emit_scalar(crate::types::FieldId::from(path.clone()), val);
        Ok(())
    }

    fn emit_event(
        &mut self,
        chronicle_id: String,
        name: String,
        fields: Vec<(String, Value)>,
    ) -> std::result::Result<(), ExecutionError> {
        if self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "Emit event only allowed in Measure phase".to_string(),
                phase: self.phase,
            });
        }
        self.event_buffer.emit(chronicle_id, name, fields);
        Ok(())
    }

    fn spawn(
        &mut self,
        _entity: &EntityId,
        _data: Value,
    ) -> std::result::Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "Spawn not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn destroy(
        &mut self,
        _entity: &EntityId,
        _instance: Value,
    ) -> std::result::Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "Destroy not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn iter_entity(&self, entity: &EntityId) -> std::result::Result<Vec<Value>, ExecutionError> {
        // Read from MemberSignalBuffer (the single source of truth for entity state)
        let entity_str = entity.to_string();
        let instance_count = self.member_signals.instance_count_for_entity(&entity_str);

        if instance_count == 0 {
            return Err(ExecutionError::InvalidOperand {
                message: format!("Entity {} has no instances", entity),
            });
        }

        // Get all member signal names for this entity
        let signal_names = self.member_signals.signals_for_entity(&entity_str);

        if signal_names.is_empty() {
            return Err(ExecutionError::InvalidOperand {
                message: format!("Entity {} has no member signals", entity),
            });
        }

        // Build a Value::Map for each instance containing all member signals
        let mut values = Vec::with_capacity(instance_count);
        for idx in 0..instance_count {
            let mut fields = Vec::new();
            for signal in &signal_names {
                // Extract member name from full path (e.g., "hydrology.cell.temperature" -> "temperature")
                let member_name = signal.rsplit('.').next().unwrap_or(signal).to_string();

                if let Some(value) = self.member_signals.get_current(signal, idx) {
                    fields.push((member_name, value));
                }
            }
            values.push(Value::map(fields));
        }
        Ok(values)
    }

    fn reduce_aggregate(
        &self,
        op: AggregateOp,
        values: Vec<Value>,
    ) -> std::result::Result<Value, ExecutionError> {
        if values.is_empty() {
            return Err(ExecutionError::InvalidOperand {
                message: "Aggregate reduction requires at least one value".to_string(),
            });
        }

        /// Extract all values as f64 scalars, returning a type mismatch error if any fail.
        fn reduce_scalars(
            values: &[Value],
            f: fn(&[f64]) -> f64,
        ) -> std::result::Result<Value, ExecutionError> {
            let v = values
                .iter()
                .map(Value::as_scalar)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| ExecutionError::TypeMismatch {
                    expected: "Scalar values".to_string(),
                    found: format!("{values:?}"),
                })?;
            Ok(Value::Scalar(f(&v)))
        }

        /// Extract all values as bools, returning a type mismatch error if any fail.
        fn collect_bools(values: &[Value]) -> std::result::Result<Vec<bool>, ExecutionError> {
            values
                .iter()
                .map(Value::as_bool)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| ExecutionError::TypeMismatch {
                    expected: "Boolean values".to_string(),
                    found: format!("{values:?}"),
                })
        }

        match op {
            AggregateOp::Sum => {
                // Sum supports both Scalar and Vec3
                if let Some(v) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::sum(&v)))
                } else if let Some(v) = values
                    .iter()
                    .map(Value::as_vec3)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Vec3(reductions::sum_vec3(&v)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar or Vec3 values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Product => reduce_scalars(&values, reductions::product),
            AggregateOp::Max => reduce_scalars(&values, reductions::max),
            AggregateOp::Min => reduce_scalars(&values, reductions::min),
            AggregateOp::Mean => reduce_scalars(&values, reductions::mean),
            AggregateOp::Count => {
                let bools = collect_bools(&values)?;
                let count = bools.iter().filter(|&&v| v).count();
                Ok(Value::Scalar(count as f64))
            }
            AggregateOp::Any => {
                let bools = collect_bools(&values)?;
                Ok(Value::Boolean(bools.iter().any(|&v| v)))
            }
            AggregateOp::All => {
                let bools = collect_bools(&values)?;
                Ok(Value::Boolean(bools.iter().all(|&v| v)))
            }
            AggregateOp::None => {
                let bools = collect_bools(&values)?;
                Ok(Value::Boolean(!bools.iter().any(|&v| v)))
            }
            AggregateOp::First => Ok(values[0].clone()),
            AggregateOp::Map => Err(ExecutionError::InvalidOperand {
                message: "Aggregate Map is not a runtime reduction".to_string(),
            }),
        }
    }

    fn call_kernel(
        &self,
        kernel: &continuum_kernel_types::KernelId,
        args: &[Value],
    ) -> std::result::Result<Value, ExecutionError> {
        continuum_kernel_registry::eval_in_namespace(
            kernel.namespace.as_ref(),
            kernel.name.as_ref(),
            args,
            self.dt.0,
        )
        .ok_or_else(|| ExecutionError::KernelCallFailed {
            message: format!("Kernel not found: {}", kernel.qualified_name()),
        })
    }

    fn trigger_assertion_fault(
        &mut self,
        severity: Option<&str>,
        message: Option<&str>,
    ) -> std::result::Result<(), ExecutionError> {
        // Default values are intentional: assertions may omit severity/message metadata.
        // DSL syntax: `assert { condition }` uses defaults.
        // Explicit: `assert { condition } severity("warn") message("custom")` overrides.
        // The .unwrap_or() here provides documented fallback behavior, not silent error hiding.
        Err(ExecutionError::AssertionFailed {
            severity: severity.unwrap_or("error").to_string(),
            message: message.unwrap_or("assertion failed").to_string(),
        })
    }

    fn load_member_signal(&self, member_name: &str) -> std::result::Result<Value, ExecutionError> {
        let entity_ctx = self
            .entity_context
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: format!(
                    "load_member_signal('{}') requires entity context",
                    member_name
                ),
                phase: self.phase,
            })?;

        // Construct the full member signal path: entity_id.member_name
        let full_path = format!("{}.{}", entity_ctx.entity_id, member_name);

        // Load the current value from the member signal buffer
        self.member_signals
            .get_current(&full_path, entity_ctx.instance_index)
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!(
                    "Member signal '{}' at instance {} not found or not initialized",
                    full_path, entity_ctx.instance_index
                ),
            })
    }
}
