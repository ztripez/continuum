//! Bytecode executor
//!
//! Stack-based VM that executes compiled bytecode.

use crate::bytecode::{BytecodeChunk, Op, ReductionOp};
use continuum_kernel_registry::Value;

/// Execution context providing runtime values
pub trait ExecutionContext {
    /// Get previous value of current signal
    fn prev(&self) -> Value;

    /// Get previous value component by name (x, y, z, w) for vector signals
    fn prev_component(&self, component: &str) -> Value {
        // Default implementation returns the full prev value if scalar, or extracts component
        if let Some(v) = self.prev().component(component) {
            Value::Scalar(v)
        } else {
            panic!(
                "prev value {:?} has no component {}",
                self.prev(),
                component
            )
        }
    }

    /// Get dt (time step)
    fn dt(&self) -> Value {
        Value::Scalar(self.dt_scalar())
    }

    /// Get dt as scalar (helper)
    fn dt_scalar(&self) -> f64;

    /// Get accumulated simulation time in seconds
    fn sim_time(&self) -> Value;

    /// Get sum of inputs for current signal
    fn inputs(&self) -> Value;

    /// Get inputs component by name (x, y, z, w) for vector signals
    ///
    /// # Panics
    ///
    /// Panics if the component doesn't exist on the value type
    fn inputs_component(&self, component: &str) -> Value {
        let inputs = self.inputs();
        match inputs.component(component) {
            Some(v) => Value::Scalar(v),
            None => panic!(
                "inputs_component: component '{}' not found on {:?}",
                component, inputs
            ),
        }
    }

    /// Get signal value by name
    fn signal(&self, name: &str) -> Value;

    /// Get signal component by name and component (x, y, z, w)
    ///
    /// # Panics
    ///
    /// Panics if the component doesn't exist on the signal's value type
    fn signal_component(&self, name: &str, component: &str) -> Value {
        let signal = self.signal(name);
        match signal.component(component) {
            Some(v) => Value::Scalar(v),
            None => panic!(
                "signal_component: component '{}' not found on signal '{}' ({:?})",
                component, name, signal
            ),
        }
    }

    /// Get constant value by name
    fn constant(&self, name: &str) -> Value;

    /// Get config value by name
    fn config(&self, name: &str) -> Value;

    /// Call a kernel function
    fn call_kernel(&self, name: &str, args: &[Value]) -> Value;

    // === Entity access ===
    /// Get value of a member signal of the current instance
    fn self_field(&self, component: &str) -> Value;

    /// Get value of a field from a specific entity instance
    fn entity_field(&self, entity: &str, instance: &str, component: &str) -> Value;

    /// Get value of another instance in the same entity set (for n-body)
    fn other_field(&self, component: &str) -> Value;

    /// Get all instance IDs for an entity type (MUST be sorted)
    fn entity_instances(&self, entity: &str) -> Vec<String>;

    /// Set the current entity type ID for subsequent entity-relative calls
    fn set_current_entity(&mut self, entity: Option<String>);

    /// Set the 'self' instance for subsequent self_field calls (during iteration)
    fn set_self_instance(&mut self, instance: Option<String>);

    /// Set the 'other' instance for subsequent other_field calls (during pairwise iteration)
    fn set_other_instance(&mut self, instance: Option<String>);

    // === Impulse access ===
    /// Get the current impulse payload
    fn payload(&self) -> Value;

    /// Get a field from the current impulse payload
    fn payload_field(&self, component: &str) -> Value;

    /// Emit a signal from an impulse
    fn emit_signal(&self, target: &str, value: Value);
}

/// Execute bytecode with the given context
pub fn execute(chunk: &BytecodeChunk, ctx: &mut dyn ExecutionContext) -> Value {
    let mut stack: Vec<Value> = Vec::with_capacity(32);
    let mut locals: Vec<Value> = vec![Value::Scalar(0.0); chunk.local_count as usize];
    let mut ip = 0;

    while ip < chunk.ops.len() {
        match chunk.ops[ip] {
            Op::Literal(idx) => {
                stack.push(chunk.literals[idx as usize].clone());
            }

            Op::LoadPrev => {
                stack.push(ctx.prev());
            }

            Op::LoadDt => {
                stack.push(ctx.dt());
            }

            Op::LoadSimTime => {
                stack.push(ctx.sim_time());
            }

            Op::LoadInputs => {
                stack.push(ctx.inputs());
            }

            Op::LoadInputsComponent(component_idx) => {
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.inputs_component(component));
            }

            Op::LoadSignal(idx) => {
                let name = &chunk.signals[idx as usize];
                stack.push(ctx.signal(name));
            }

            Op::LoadSignalComponent(signal_idx, component_idx) => {
                let name = &chunk.signals[signal_idx as usize];
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.signal_component(name, component));
            }

            Op::LoadPrevComponent(component_idx) => {
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.prev_component(component));
            }

            Op::LoadConst(idx) => {
                let name = &chunk.constants[idx as usize];
                stack.push(ctx.constant(name));
            }

            Op::LoadConfig(idx) => {
                let name = &chunk.configs[idx as usize];
                stack.push(ctx.config(name));
            }

            Op::LoadLocal(slot) => {
                stack.push(locals[slot as usize].clone());
            }

            Op::StoreLocal(slot) => {
                let v = stack.last().expect("vm bug: stack underflow").clone();
                locals[slot as usize] = v;
            }

            Op::LoadSelfField(component_idx) => {
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.self_field(component));
            }

            Op::LoadEntityField(entity_idx, instance_idx, component_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let instance = &chunk.instances[instance_idx as usize];
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.entity_field(entity, instance, component));
            }

            Op::LoadOtherField(component_idx) => {
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.other_field(component));
            }

            Op::Aggregate(entity_idx, op, sub_chunk_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let sub_chunk = &chunk.sub_chunks[sub_chunk_idx as usize];
                let instances = ctx.entity_instances(entity);

                let mut result = match op {
                    ReductionOp::Sum | ReductionOp::Mean => Value::Scalar(0.0),
                    ReductionOp::Product => Value::Scalar(1.0),
                    ReductionOp::Min => Value::Scalar(f64::INFINITY),
                    ReductionOp::Max => Value::Scalar(f64::NEG_INFINITY),
                    ReductionOp::Count => Value::Integer(0),
                    ReductionOp::Any => Value::Boolean(false),
                    ReductionOp::All => Value::Boolean(true),
                    ReductionOp::None => Value::Boolean(true),
                };

                let count = instances.len();
                if op == ReductionOp::Count {
                    stack.push(Value::Integer(count as i64));
                } else {
                    ctx.set_current_entity(Some(entity.clone()));
                    for instance_id in instances {
                        ctx.set_self_instance(Some(instance_id.clone()));
                        let val = execute(sub_chunk, ctx);

                        match op {
                            ReductionOp::Sum | ReductionOp::Mean => {
                                result = val_add(result, val);
                            }
                            ReductionOp::Product => {
                                result = val_mul(result, val);
                            }
                            ReductionOp::Min => {
                                if val_cmp(val.clone(), result.clone()) < 0 {
                                    result = val;
                                }
                            }
                            ReductionOp::Max => {
                                if val_cmp(val.clone(), result.clone()) > 0 {
                                    result = val;
                                }
                            }
                            ReductionOp::Any => {
                                if val_truthy(&val) {
                                    result = Value::Boolean(true);
                                    break;
                                }
                            }
                            ReductionOp::All => {
                                if !val_truthy(&val) {
                                    result = Value::Boolean(false);
                                    break;
                                }
                            }
                            ReductionOp::None => {
                                if val_truthy(&val) {
                                    result = Value::Boolean(false);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }

                    if op == ReductionOp::Mean && count > 0 {
                        result = val_div(result, Value::Scalar(count as f64));
                    }

                    ctx.set_self_instance(None);
                    ctx.set_current_entity(None);
                    stack.push(result);
                }
            }

            Op::Filter(entity_idx, pred_chunk_idx, body_chunk_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let pred_chunk = &chunk.sub_chunks[pred_chunk_idx as usize];
                let body_chunk = &chunk.sub_chunks[body_chunk_idx as usize];
                let instances = ctx.entity_instances(entity);

                ctx.set_current_entity(Some(entity.clone()));
                let mut last_val = Value::Scalar(0.0);
                for instance_id in instances {
                    ctx.set_self_instance(Some(instance_id.clone()));
                    let pred = execute(pred_chunk, ctx);
                    if val_truthy(&pred) {
                        last_val = execute(body_chunk, ctx);
                    }
                }
                ctx.set_self_instance(None);
                ctx.set_current_entity(None);
                stack.push(last_val);
            }

            Op::FindFirstField(entity_idx, pred_chunk_idx, component_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let pred_chunk = &chunk.sub_chunks[pred_chunk_idx as usize];
                let component = &chunk.components[component_idx as usize];
                let instances = ctx.entity_instances(entity);

                ctx.set_current_entity(Some(entity.clone()));
                let mut result = Value::Scalar(0.0);
                for instance_id in instances {
                    ctx.set_self_instance(Some(instance_id.clone()));
                    let pred = execute(pred_chunk, ctx);
                    if val_truthy(&pred) {
                        result = ctx.self_field(component);
                        break;
                    }
                }
                ctx.set_self_instance(None);
                ctx.set_current_entity(None);
                stack.push(result);
            }

            Op::LoadNearestField(entity_idx, position_field_idx, component_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let position_field = &chunk.components[position_field_idx as usize];
                let component = &chunk.components[component_idx as usize];
                let pos = stack.pop().expect("vm bug: stack underflow");
                let instances = ctx.entity_instances(entity);

                ctx.set_current_entity(Some(entity.clone()));
                let mut nearest_id = None;
                let mut min_dist_sq = f64::INFINITY;

                for instance_id in instances {
                    ctx.set_self_instance(Some(instance_id.clone()));
                    let inst_pos = ctx.self_field(position_field);
                    let dist_sq = val_dist_sq(&pos, &inst_pos);

                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        nearest_id = Some(instance_id);
                    } else if dist_sq == min_dist_sq {
                        if let Some(ref current_nearest) = nearest_id {
                            if instance_id < *current_nearest {
                                nearest_id = Some(instance_id);
                            }
                        }
                    }
                }

                if let Some(id) = nearest_id {
                    ctx.set_self_instance(Some(id));
                    stack.push(ctx.self_field(component));
                } else {
                    stack.push(Value::Scalar(0.0));
                }
                ctx.set_self_instance(None);
                ctx.set_current_entity(None);
            }

            Op::WithinAggregate(entity_idx, position_field_idx, op, body_chunk_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let position_field = &chunk.components[position_field_idx as usize];
                let body_chunk = &chunk.sub_chunks[body_chunk_idx as usize];
                let radius = stack
                    .pop()
                    .expect("vm bug: stack underflow")
                    .as_scalar()
                    .unwrap_or(0.0);
                let radius_sq = radius * radius;
                let pos = stack.pop().expect("vm bug: stack underflow");
                let instances = ctx.entity_instances(entity);

                ctx.set_current_entity(Some(entity.clone()));
                let mut result = match op {
                    ReductionOp::Sum | ReductionOp::Mean => Value::Scalar(0.0),
                    ReductionOp::Product => Value::Scalar(1.0),
                    ReductionOp::Min => Value::Scalar(f64::INFINITY),
                    ReductionOp::Max => Value::Scalar(f64::NEG_INFINITY),
                    ReductionOp::Count => Value::Integer(0),
                    ReductionOp::Any => Value::Boolean(false),
                    ReductionOp::All => Value::Boolean(true),
                    ReductionOp::None => Value::Boolean(true),
                };

                let mut count = 0usize;
                for instance_id in instances {
                    ctx.set_self_instance(Some(instance_id.clone()));
                    let inst_pos = ctx.self_field(position_field);
                    if val_dist_sq(&pos, &inst_pos) <= radius_sq {
                        count += 1;
                        let val = execute(body_chunk, ctx);

                        match op {
                            ReductionOp::Sum | ReductionOp::Mean => {
                                result = val_add(result, val);
                            }
                            ReductionOp::Product => {
                                result = val_mul(result, val);
                            }
                            ReductionOp::Min => {
                                if val_cmp(val.clone(), result.clone()) < 0 {
                                    result = val;
                                }
                            }
                            ReductionOp::Max => {
                                if val_cmp(val.clone(), result.clone()) > 0 {
                                    result = val;
                                }
                            }
                            ReductionOp::Any => {
                                if val_truthy(&val) {
                                    result = Value::Boolean(true);
                                    break;
                                }
                            }
                            ReductionOp::All => {
                                if !val_truthy(&val) {
                                    result = Value::Boolean(false);
                                    break;
                                }
                            }
                            ReductionOp::None => {
                                if val_truthy(&val) {
                                    result = Value::Boolean(false);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                if op == ReductionOp::Count {
                    result = Value::Integer(count as i64);
                } else if op == ReductionOp::Mean && count > 0 {
                    result = val_div(result, Value::Scalar(count as f64));
                }

                ctx.set_self_instance(None);
                ctx.set_current_entity(None);
                stack.push(result);
            }

            Op::Pairs(entity_idx, body_chunk_idx) => {
                let entity = &chunk.entities[entity_idx as usize];
                let body_chunk = &chunk.sub_chunks[body_chunk_idx as usize];
                let instances = ctx.entity_instances(entity);

                ctx.set_current_entity(Some(entity.clone()));
                for i in 0..instances.len() {
                    for j in (i + 1)..instances.len() {
                        ctx.set_self_instance(Some(instances[i].clone()));
                        ctx.set_other_instance(Some(instances[j].clone()));
                        execute(body_chunk, ctx);
                    }
                }
                ctx.set_self_instance(None);
                ctx.set_other_instance(None);
                ctx.set_current_entity(None);
                stack.push(Value::Scalar(0.0));
            }

            Op::LoadPayload => {
                stack.push(ctx.payload());
            }

            Op::LoadPayloadField(component_idx) => {
                let component = &chunk.components[component_idx as usize];
                stack.push(ctx.payload_field(component));
            }

            Op::EmitSignal(signal_idx) => {
                let target = &chunk.signals[signal_idx as usize];
                let val = stack.pop().expect("vm bug: stack underflow");
                ctx.emit_signal(target, val);
            }

            Op::Add => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(val_add(l, r));
            }

            Op::Sub => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(val_sub(l, r));
            }

            Op::Mul => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(val_mul(l, r));
            }

            Op::Div => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(val_div(l, r));
            }

            Op::Pow => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(val_pow(l, r));
            }

            Op::Neg => {
                let v = stack.pop().expect("vm bug: stack underflow");
                stack.push(val_neg(v));
            }

            Op::Eq => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(l == r));
            }

            Op::Ne => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(l != r));
            }

            Op::Lt => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(val_cmp(l, r) < 0));
            }

            Op::Le => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(val_cmp(l, r) <= 0));
            }

            Op::Gt => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(val_cmp(l, r) > 0));
            }

            Op::Ge => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(val_cmp(l, r) >= 0));
            }

            Op::And => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(val_truthy(&l) && val_truthy(&r)));
            }

            Op::Or => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(val_truthy(&l) || val_truthy(&r)));
            }

            Op::Not => {
                let v = stack.pop().expect("vm bug: stack underflow");
                stack.push(Value::Boolean(!val_truthy(&v)));
            }

            Op::JumpIfZero(offset) => {
                let v = stack.pop().expect("vm bug: stack underflow");
                if !val_truthy(&v) {
                    ip += offset as usize;
                }
            }

            Op::Jump(offset) => {
                ip += offset as usize;
            }

            Op::Call { kernel, arity } => {
                let name = &chunk.kernels[kernel as usize];
                let start = stack.len().saturating_sub(arity as usize);
                let args: Vec<Value> = stack.drain(start..).collect();
                let result = ctx.call_kernel(name, &args);
                stack.push(result);
            }

            Op::Dup => {
                let v = stack.last().expect("vm bug: stack underflow").clone();
                stack.push(v);
            }

            Op::Pop => {
                stack.pop();
            }
        }
        ip += 1;
    }

    stack.pop().expect("vm bug: stack underflow")
}

// === Value Helpers ===

// Matrix operations use shared implementations from foundation crate
use continuum_foundation::matrix_ops::{
    mat2_mul, mat2_transform, mat3_mul, mat3_transform, mat4_mul, mat4_transform,
};

fn val_add(l: Value, r: Value) -> Value {
    match (&l, &r) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a + b),
        (Value::Integer(a), Value::Integer(b)) => Value::Integer(a + b),
        (Value::Scalar(a), Value::Integer(b)) => Value::Scalar(a + *b as f64),
        (Value::Integer(a), Value::Scalar(b)) => Value::Scalar(*a as f64 + b),
        // Vector addition
        (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] + b[0], a[1] + b[1]]),
        (Value::Vec3(a), Value::Vec3(b)) => Value::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]]),
        (Value::Vec4(a), Value::Vec4(b)) => {
            Value::Vec4([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
        }
        // Quaternion addition (component-wise)
        (Value::Quat(a), Value::Quat(b)) => {
            Value::Quat([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
        }
        // Matrix element-wise addition
        (Value::Mat2(a), Value::Mat2(b)) => {
            let mut result = [0.0; 4];
            for i in 0..4 {
                result[i] = a[i] + b[i];
            }
            Value::Mat2(result)
        }
        (Value::Mat3(a), Value::Mat3(b)) => {
            let mut result = [0.0; 9];
            for i in 0..9 {
                result[i] = a[i] + b[i];
            }
            Value::Mat3(result)
        }
        (Value::Mat4(a), Value::Mat4(b)) => {
            let mut result = [0.0; 16];
            for i in 0..16 {
                result[i] = a[i] + b[i];
            }
            Value::Mat4(result)
        }
        (Value::Map(a), Value::Map(b)) => {
            let mut res = a.clone();
            res.extend(b.clone());
            Value::Map(res)
        }
        _ => panic!("val_add: cannot add {:?} and {:?}", l, r),
    }
}

fn val_sub(l: Value, r: Value) -> Value {
    match (&l, &r) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a - b),
        (Value::Integer(a), Value::Integer(b)) => Value::Integer(a - b),
        (Value::Scalar(a), Value::Integer(b)) => Value::Scalar(a - *b as f64),
        (Value::Integer(a), Value::Scalar(b)) => Value::Scalar(*a as f64 - b),
        // Vector subtraction
        (Value::Vec2(a), Value::Vec2(b)) => Value::Vec2([a[0] - b[0], a[1] - b[1]]),
        (Value::Vec3(a), Value::Vec3(b)) => Value::Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]]),
        (Value::Vec4(a), Value::Vec4(b)) => {
            Value::Vec4([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
        }
        // Quaternion subtraction (component-wise)
        (Value::Quat(a), Value::Quat(b)) => {
            Value::Quat([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
        }
        // Matrix element-wise subtraction
        (Value::Mat2(a), Value::Mat2(b)) => {
            let mut result = [0.0; 4];
            for i in 0..4 {
                result[i] = a[i] - b[i];
            }
            Value::Mat2(result)
        }
        (Value::Mat3(a), Value::Mat3(b)) => {
            let mut result = [0.0; 9];
            for i in 0..9 {
                result[i] = a[i] - b[i];
            }
            Value::Mat3(result)
        }
        (Value::Mat4(a), Value::Mat4(b)) => {
            let mut result = [0.0; 16];
            for i in 0..16 {
                result[i] = a[i] - b[i];
            }
            Value::Mat4(result)
        }
        _ => panic!("val_sub: cannot subtract {:?} from {:?}", r, l),
    }
}

fn val_mul(l: Value, r: Value) -> Value {
    match (&l, &r) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a * b),
        (Value::Integer(a), Value::Integer(b)) => Value::Integer(a * b),
        (Value::Scalar(a), Value::Integer(b)) => Value::Scalar(a * *b as f64),
        (Value::Integer(a), Value::Scalar(b)) => Value::Scalar(*a as f64 * b),

        // Vec2 * Scalar
        (Value::Vec2(v), Value::Scalar(s)) => Value::Vec2([v[0] * s, v[1] * s]),
        (Value::Scalar(s), Value::Vec2(v)) => Value::Vec2([v[0] * s, v[1] * s]),
        // Vec3 * Scalar
        (Value::Vec3(v), Value::Scalar(s)) => Value::Vec3([v[0] * s, v[1] * s, v[2] * s]),
        (Value::Scalar(s), Value::Vec3(v)) => Value::Vec3([v[0] * s, v[1] * s, v[2] * s]),
        // Vec4 * Scalar
        (Value::Vec4(v), Value::Scalar(s)) => Value::Vec4([v[0] * s, v[1] * s, v[2] * s, v[3] * s]),
        (Value::Scalar(s), Value::Vec4(v)) => Value::Vec4([v[0] * s, v[1] * s, v[2] * s, v[3] * s]),

        // Quat * Quat (Hamilton product)
        (Value::Quat(a), Value::Quat(b)) => {
            let [aw, ax, ay, az] = a;
            let [bw, bx, by, bz] = b;
            Value::Quat([
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ])
        }
        // Quat * Scalar
        (Value::Quat(q), Value::Scalar(s)) => Value::Quat([q[0] * s, q[1] * s, q[2] * s, q[3] * s]),
        (Value::Scalar(s), Value::Quat(q)) => Value::Quat([q[0] * s, q[1] * s, q[2] * s, q[3] * s]),

        // Scalar * Matrix (broadcast)
        (Value::Scalar(s), Value::Mat2(m)) => {
            let mut result = *m;
            for i in 0..4 {
                result[i] *= s;
            }
            Value::Mat2(result)
        }
        (Value::Mat2(m), Value::Scalar(s)) => {
            let mut result = *m;
            for i in 0..4 {
                result[i] *= s;
            }
            Value::Mat2(result)
        }
        (Value::Scalar(s), Value::Mat3(m)) => {
            let mut result = *m;
            for i in 0..9 {
                result[i] *= s;
            }
            Value::Mat3(result)
        }
        (Value::Mat3(m), Value::Scalar(s)) => {
            let mut result = *m;
            for i in 0..9 {
                result[i] *= s;
            }
            Value::Mat3(result)
        }
        (Value::Scalar(s), Value::Mat4(m)) => {
            let mut result = *m;
            for i in 0..16 {
                result[i] *= s;
            }
            Value::Mat4(result)
        }
        (Value::Mat4(m), Value::Scalar(s)) => {
            let mut result = *m;
            for i in 0..16 {
                result[i] *= s;
            }
            Value::Mat4(result)
        }

        // Matrix * Matrix (proper matrix multiplication)
        (Value::Mat2(a), Value::Mat2(b)) => Value::Mat2(mat2_mul(*a, *b)),
        (Value::Mat3(a), Value::Mat3(b)) => Value::Mat3(mat3_mul(*a, *b)),
        (Value::Mat4(a), Value::Mat4(b)) => Value::Mat4(mat4_mul(*a, *b)),

        // Matrix * Vector (transform)
        (Value::Mat2(m), Value::Vec2(v)) => Value::Vec2(mat2_transform(*m, *v)),
        (Value::Mat3(m), Value::Vec3(v)) => Value::Vec3(mat3_transform(*m, *v)),
        (Value::Mat4(m), Value::Vec4(v)) => Value::Vec4(mat4_transform(*m, *v)),

        _ => panic!("val_mul: cannot multiply {:?} and {:?}", l, r),
    }
}

fn val_div(l: Value, r: Value) -> Value {
    match (&l, &r) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a / b),
        (Value::Integer(a), Value::Integer(b)) => Value::Integer(a / b),
        (Value::Scalar(a), Value::Integer(b)) => Value::Scalar(a / *b as f64),
        (Value::Integer(a), Value::Scalar(b)) => Value::Scalar(*a as f64 / b),
        // Vector / Scalar
        (Value::Vec2(v), Value::Scalar(s)) => Value::Vec2([v[0] / s, v[1] / s]),
        (Value::Vec3(v), Value::Scalar(s)) => Value::Vec3([v[0] / s, v[1] / s, v[2] / s]),
        (Value::Vec4(v), Value::Scalar(s)) => Value::Vec4([v[0] / s, v[1] / s, v[2] / s, v[3] / s]),
        // Quat / Scalar
        (Value::Quat(q), Value::Scalar(s)) => Value::Quat([q[0] / s, q[1] / s, q[2] / s, q[3] / s]),
        // Matrix / Scalar (element-wise division)
        (Value::Mat2(m), Value::Scalar(s)) => {
            let mut result = *m;
            for i in 0..4 {
                result[i] /= s;
            }
            Value::Mat2(result)
        }
        (Value::Mat3(m), Value::Scalar(s)) => {
            let mut result = *m;
            for i in 0..9 {
                result[i] /= s;
            }
            Value::Mat3(result)
        }
        (Value::Mat4(m), Value::Scalar(s)) => {
            let mut result = *m;
            for i in 0..16 {
                result[i] /= s;
            }
            Value::Mat4(result)
        }
        _ => panic!("val_div: cannot divide {:?} by {:?}", l, r),
    }
}

fn val_pow(l: Value, r: Value) -> Value {
    match (&l, &r) {
        (Value::Scalar(a), Value::Scalar(b)) => Value::Scalar(a.powf(*b)),
        (Value::Integer(a), Value::Integer(b)) => {
            Value::Integer((*a as f64).powf(*b as f64) as i64)
        }
        (Value::Scalar(a), Value::Integer(b)) => Value::Scalar(a.powf(*b as f64)),
        (Value::Integer(a), Value::Scalar(b)) => Value::Scalar((*a as f64).powf(*b)),
        _ => panic!("val_pow: cannot raise {:?} to power {:?}", l, r),
    }
}

fn val_neg(v: Value) -> Value {
    match &v {
        Value::Scalar(a) => Value::Scalar(-a),
        Value::Integer(a) => Value::Integer(-a),
        // Vector negation
        Value::Vec2(a) => Value::Vec2([-a[0], -a[1]]),
        Value::Vec3(a) => Value::Vec3([-a[0], -a[1], -a[2]]),
        Value::Vec4(a) => Value::Vec4([-a[0], -a[1], -a[2], -a[3]]),
        // Quaternion negation
        Value::Quat(a) => Value::Quat([-a[0], -a[1], -a[2], -a[3]]),
        // Matrix negation (element-wise)
        Value::Mat2(m) => {
            let mut result = *m;
            for i in 0..4 {
                result[i] = -result[i];
            }
            Value::Mat2(result)
        }
        Value::Mat3(m) => {
            let mut result = *m;
            for i in 0..9 {
                result[i] = -result[i];
            }
            Value::Mat3(result)
        }
        Value::Mat4(m) => {
            let mut result = *m;
            for i in 0..16 {
                result[i] = -result[i];
            }
            Value::Mat4(result)
        }
        _ => panic!("val_neg: cannot negate {:?}", v),
    }
}

fn val_cmp(l: Value, r: Value) -> i8 {
    let diff = match (&l, &r) {
        (Value::Scalar(a), Value::Scalar(b)) => a - b,
        (Value::Integer(a), Value::Integer(b)) => (a - b) as f64,
        (Value::Scalar(a), Value::Integer(b)) => a - *b as f64,
        (Value::Integer(a), Value::Scalar(b)) => *a as f64 - b,
        _ => panic!("val_cmp: cannot compare {:?} and {:?}", l, r),
    };

    if diff < 0.0 {
        -1
    } else if diff > 0.0 {
        1
    } else {
        0
    }
}

fn val_truthy(v: &Value) -> bool {
    match v {
        Value::Boolean(b) => *b,
        Value::Scalar(s) => *s != 0.0,
        Value::Integer(i) => *i != 0,
        Value::Map(m) => !m.is_empty(),
        _ => panic!("val_truthy: cannot convert {:?} to boolean", v),
    }
}

fn val_dist_sq(a: &Value, b: &Value) -> f64 {
    match (a, b) {
        (Value::Vec2(v1), Value::Vec2(v2)) => {
            let dx = v1[0] - v2[0];
            let dy = v1[1] - v2[1];
            dx * dx + dy * dy
        }
        (Value::Vec3(v1), Value::Vec3(v2)) => {
            let dx = v1[0] - v2[0];
            let dy = v1[1] - v2[1];
            let dz = v1[2] - v2[2];
            dx * dx + dy * dy + dz * dz
        }
        (Value::Vec4(v1), Value::Vec4(v2)) => {
            let dx = v1[0] - v2[0];
            let dy = v1[1] - v2[1];
            let dz = v1[2] - v2[2];
            let dw = v1[3] - v2[3];
            dx * dx + dy * dy + dz * dz + dw * dw
        }
        (Value::Scalar(s1), Value::Scalar(s2)) => {
            let ds = s1 - s2;
            ds * ds
        }
        _ => panic!(
            "val_dist_sq: cannot compute distance between {:?} and {:?}",
            a, b
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{BinaryOp, Expr, compile_expr};

    struct TestContext;

    impl ExecutionContext for TestContext {
        fn prev(&self) -> Value {
            Value::Scalar(100.0)
        }
        fn dt_scalar(&self) -> f64 {
            0.1
        }
        fn sim_time(&self) -> Value {
            Value::Scalar(10.0)
        }
        fn inputs(&self) -> Value {
            Value::Scalar(5.0)
        }
        fn signal(&self, name: &str) -> Value {
            match name {
                "temp" => Value::Scalar(25.0),
                "pressure" => Value::Scalar(101.0),
                _ => Value::Scalar(0.0),
            }
        }
        fn constant(&self, name: &str) -> Value {
            match name {
                "PI" => Value::Scalar(std::f64::consts::PI),
                _ => Value::Scalar(0.0),
            }
        }
        fn config(&self, name: &str) -> Value {
            match name {
                "scale" => Value::Scalar(2.0),
                _ => Value::Scalar(0.0),
            }
        }
        fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
            match name {
                "abs" => args
                    .first()
                    .and_then(|v| v.as_scalar())
                    .map(|v| Value::Scalar(v.abs()))
                    .unwrap_or(Value::Scalar(0.0)),
                "min" => Value::Scalar(
                    args.iter()
                        .filter_map(|v| v.as_scalar())
                        .fold(f64::INFINITY, f64::min),
                ),
                "max" => Value::Scalar(
                    args.iter()
                        .filter_map(|v| v.as_scalar())
                        .fold(f64::NEG_INFINITY, f64::max),
                ),
                _ => Value::Scalar(0.0),
            }
        }

        fn self_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_field(&self, _entity: &str, _instance: &str, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn other_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_instances(&self, _entity: &str) -> Vec<String> {
            Vec::new()
        }
        fn set_current_entity(&mut self, _entity: Option<String>) {}
        fn set_self_instance(&mut self, _instance: Option<String>) {}
        fn set_other_instance(&mut self, _instance: Option<String>) {}
        fn payload(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn payload_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn emit_signal(&self, _target: &str, _value: Value) {}
    }

    #[test]
    fn test_execute_literal() {
        let chunk = compile_expr(&Expr::Literal(42.0));
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(42.0));
    }

    #[test]
    fn test_execute_binary() {
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Literal(10.0)),
            right: Box::new(Expr::Literal(32.0)),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(42.0));
    }

    #[test]
    fn test_execute_prev() {
        let chunk = compile_expr(&Expr::Prev);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(100.0));
    }

    #[test]
    fn test_execute_signal() {
        let chunk = compile_expr(&Expr::Signal("temp".to_string()));
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(25.0));
    }

    #[test]
    fn test_execute_call() {
        let expr = Expr::Call {
            function: "abs".to_string(),
            args: vec![Expr::Literal(-5.0)],
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_execute_if_true() {
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(1.0)),
            then_branch: Box::new(Expr::Literal(10.0)),
            else_branch: Box::new(Expr::Literal(20.0)),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(10.0));
    }

    #[test]
    fn test_execute_if_false() {
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(0.0)),
            then_branch: Box::new(Expr::Literal(10.0)),
            else_branch: Box::new(Expr::Literal(20.0)),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(20.0));
    }

    #[test]
    fn test_execute_complex() {
        // prev + (temp * config.scale) = 100 + (25 * 2) = 150
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Prev),
            right: Box::new(Expr::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expr::Signal("temp".to_string())),
                right: Box::new(Expr::Config("scale".to_string())),
            }),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(150.0));
    }

    // ============================================================================
    // Vector Component Tests
    // ============================================================================

    /// Test context with vector component support
    struct VectorTestContext;

    impl ExecutionContext for VectorTestContext {
        fn prev(&self) -> Value {
            Value::Scalar(100.0)
        }

        fn prev_component(&self, component: &str) -> Value {
            match component {
                "x" => Value::Scalar(1.0),
                "y" => Value::Scalar(2.0),
                "z" => Value::Scalar(3.0),
                "w" => Value::Scalar(4.0),
                _ => Value::Scalar(0.0),
            }
        }

        fn dt_scalar(&self) -> f64 {
            0.1
        }

        fn sim_time(&self) -> Value {
            Value::Scalar(10.0)
        }

        fn inputs(&self) -> Value {
            Value::Scalar(10.0)
        }

        fn inputs_component(&self, component: &str) -> Value {
            match component {
                "x" => Value::Scalar(10.0),
                "y" => Value::Scalar(20.0),
                "z" => Value::Scalar(30.0),
                "w" => Value::Scalar(40.0),
                _ => Value::Scalar(0.0),
            }
        }

        fn signal(&self, name: &str) -> Value {
            match name {
                "velocity" => Value::Scalar(50.0),
                "position" => Value::Scalar(100.0),
                _ => Value::Scalar(0.0),
            }
        }

        fn signal_component(&self, name: &str, component: &str) -> Value {
            match (name, component) {
                ("velocity", "x") => Value::Scalar(5.0),
                ("velocity", "y") => Value::Scalar(6.0),
                ("velocity", "z") => Value::Scalar(7.0),
                ("position", "x") => Value::Scalar(100.0),
                ("position", "y") => Value::Scalar(200.0),
                ("position", "z") => Value::Scalar(300.0),
                _ => Value::Scalar(0.0),
            }
        }

        fn constant(&self, _name: &str) -> Value {
            Value::Scalar(0.0)
        }

        fn config(&self, _name: &str) -> Value {
            Value::Scalar(0.0)
        }

        fn call_kernel(&self, _name: &str, _args: &[Value]) -> Value {
            Value::Scalar(0.0)
        }

        fn self_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_field(&self, _entity: &str, _instance: &str, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn other_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_instances(&self, _entity: &str) -> Vec<String> {
            Vec::new()
        }
        fn set_current_entity(&mut self, _entity: Option<String>) {}
        fn set_self_instance(&mut self, _instance: Option<String>) {}
        fn set_other_instance(&mut self, _instance: Option<String>) {}
        fn payload(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn payload_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn emit_signal(&self, _target: &str, _value: Value) {}
    }

    #[test]
    fn test_execute_prev_component_x() {
        let chunk = compile_expr(&Expr::PrevComponent("x".to_string()));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(1.0));
    }

    #[test]
    fn test_execute_prev_component_y() {
        let chunk = compile_expr(&Expr::PrevComponent("y".to_string()));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(2.0));
    }

    #[test]
    fn test_execute_prev_component_z() {
        let chunk = compile_expr(&Expr::PrevComponent("z".to_string()));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(3.0));
    }

    #[test]
    fn test_execute_collected_component_x() {
        let chunk = compile_expr(&Expr::CollectedComponent("x".to_string()));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(10.0));
    }

    #[test]
    fn test_execute_collected_component_y() {
        let chunk = compile_expr(&Expr::CollectedComponent("y".to_string()));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(20.0));
    }

    #[test]
    fn test_execute_signal_component_velocity_x() {
        let chunk = compile_expr(&Expr::SignalComponent(
            "velocity".to_string(),
            "x".to_string(),
        ));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_execute_signal_component_velocity_y() {
        let chunk = compile_expr(&Expr::SignalComponent(
            "velocity".to_string(),
            "y".to_string(),
        ));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(6.0));
    }

    #[test]
    fn test_execute_signal_component_position_z() {
        let chunk = compile_expr(&Expr::SignalComponent(
            "position".to_string(),
            "z".to_string(),
        ));
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(300.0));
    }

    #[test]
    fn test_execute_vector_component_arithmetic() {
        // prev.x + prev.y = 1.0 + 2.0 = 3.0
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::PrevComponent("x".to_string())),
            right: Box::new(Expr::PrevComponent("y".to_string())),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(3.0));
    }

    #[test]
    fn test_execute_mixed_component_arithmetic() {
        // prev.x + collected.y + velocity.z = 1.0 + 20.0 + 7.0 = 28.0
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expr::PrevComponent("x".to_string())),
                right: Box::new(Expr::CollectedComponent("y".to_string())),
            }),
            right: Box::new(Expr::SignalComponent(
                "velocity".to_string(),
                "z".to_string(),
            )),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(28.0));
    }

    #[test]
    fn test_execute_component_in_conditional() {
        // if prev.x > 0.0 then velocity.y else position.z
        // prev.x = 1.0 > 0.0, so result = velocity.y = 6.0
        let expr = Expr::If {
            condition: Box::new(Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::PrevComponent("x".to_string())),
                right: Box::new(Expr::Literal(0.0)),
            }),
            then_branch: Box::new(Expr::SignalComponent(
                "velocity".to_string(),
                "y".to_string(),
            )),
            else_branch: Box::new(Expr::SignalComponent(
                "position".to_string(),
                "z".to_string(),
            )),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &mut VectorTestContext);
        assert_eq!(result, Value::Scalar(6.0));
    }

    #[test]
    fn test_matrix_addition() {
        // Mat2 + Mat2
        let a = Value::Mat2([1.0, 2.0, 3.0, 4.0]);
        let b = Value::Mat2([5.0, 6.0, 7.0, 8.0]);
        let result = val_add(a, b);
        assert_eq!(result, Value::Mat2([6.0, 8.0, 10.0, 12.0]));
    }

    #[test]
    fn test_matrix_subtraction() {
        // Mat3 - Mat3
        let a = Value::Mat3([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let b = Value::Mat3([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let result = val_sub(a, b);
        assert_eq!(
            result,
            Value::Mat3([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        );
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        // Mat2 * Scalar
        let m = Value::Mat2([1.0, 2.0, 3.0, 4.0]);
        let s = Value::Scalar(2.0);
        let result = val_mul(m.clone(), s.clone());
        assert_eq!(result, Value::Mat2([2.0, 4.0, 6.0, 8.0]));

        // Scalar * Mat2
        let result2 = val_mul(s, m);
        assert_eq!(result2, Value::Mat2([2.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn test_matrix_multiplication() {
        // Mat2 * Mat2 (identity test)
        let identity = Value::Mat2([1.0, 0.0, 0.0, 1.0]);
        let m = Value::Mat2([1.0, 2.0, 3.0, 4.0]);
        let result = val_mul(identity, m.clone());
        assert_eq!(result, m);

        // Mat2 * Mat2 (actual multiplication)
        let a = Value::Mat2([1.0, 2.0, 3.0, 4.0]); // [[1,3], [2,4]] in row-major
        let b = Value::Mat2([5.0, 6.0, 7.0, 8.0]); // [[5,7], [6,8]] in row-major
        let result = val_mul(a, b);
        // Expected: [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]]
        //         = [[23, 31], [34, 46]]
        // In column-major: [23, 34, 31, 46]
        assert_eq!(result, Value::Mat2([23.0, 34.0, 31.0, 46.0]));
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        // Mat2 * Vec2
        let m = Value::Mat2([1.0, 0.0, 0.0, 1.0]); // identity
        let v = Value::Vec2([3.0, 4.0]);
        let result = val_mul(m, v);
        assert_eq!(result, Value::Vec2([3.0, 4.0]));

        // Mat3 * Vec3 (scaling matrix)
        let scale = Value::Mat3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        let v3 = Value::Vec3([1.0, 2.0, 3.0]);
        let result3 = val_mul(scale, v3);
        assert_eq!(result3, Value::Vec3([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_matrix_division() {
        // Mat2 / Scalar
        let m = Value::Mat2([2.0, 4.0, 6.0, 8.0]);
        let s = Value::Scalar(2.0);
        let result = val_div(m, s);
        assert_eq!(result, Value::Mat2([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_matrix_negation() {
        // -Mat3
        let m = Value::Mat3([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = val_neg(m);
        assert_eq!(
            result,
            Value::Mat3([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0])
        );
    }

    // ============================================================================
    // Custom Position Field Tests (for spatial operations)
    // ============================================================================

    /// Test context with entity spatial operations
    struct SpatialTestContext {
        current_entity: Option<String>,
        self_instance: Option<String>,
    }

    impl SpatialTestContext {
        fn new() -> Self {
            Self {
                current_entity: None,
                self_instance: None,
            }
        }
    }

    impl ExecutionContext for SpatialTestContext {
        fn prev(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn dt_scalar(&self) -> f64 {
            0.1
        }
        fn sim_time(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn inputs(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn signal(&self, _name: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn constant(&self, _name: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn config(&self, _name: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn call_kernel(&self, _name: &str, _args: &[Value]) -> Value {
            Value::Scalar(0.0)
        }

        fn self_field(&self, component: &str) -> Value {
            // Entities with custom "location" field and "value" field
            match (&self.self_instance, component) {
                // Entity A at location (0, 0, 0) with value 10
                (Some(id), "location") if id == "entity_a" => Value::Vec3([0.0, 0.0, 0.0]),
                (Some(id), "value") if id == "entity_a" => Value::Scalar(10.0),
                // Entity B at location (1, 0, 0) with value 20
                (Some(id), "location") if id == "entity_b" => Value::Vec3([1.0, 0.0, 0.0]),
                (Some(id), "value") if id == "entity_b" => Value::Scalar(20.0),
                // Entity C at location (10, 0, 0) with value 30
                (Some(id), "location") if id == "entity_c" => Value::Vec3([10.0, 0.0, 0.0]),
                (Some(id), "value") if id == "entity_c" => Value::Scalar(30.0),
                _ => Value::Scalar(0.0),
            }
        }

        fn entity_field(&self, _entity: &str, _instance: &str, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn other_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_instances(&self, entity: &str) -> Vec<String> {
            match entity {
                "particles" => vec![
                    "entity_a".to_string(),
                    "entity_b".to_string(),
                    "entity_c".to_string(),
                ],
                _ => Vec::new(),
            }
        }
        fn set_current_entity(&mut self, entity: Option<String>) {
            self.current_entity = entity;
        }
        fn set_self_instance(&mut self, instance: Option<String>) {
            self.self_instance = instance;
        }
        fn set_other_instance(&mut self, _instance: Option<String>) {}
        fn payload(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn payload_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn emit_signal(&self, _target: &str, _value: Value) {}
    }

    #[test]
    fn test_nearest_with_custom_position_field() {
        // Find nearest entity to position (0.5, 0, 0) using "location" field
        // entity_a is at (0, 0, 0) - distance 0.5
        // entity_b is at (1, 0, 0) - distance 0.5
        // entity_c is at (10, 0, 0) - distance 9.5
        // entity_a or entity_b wins (tied at 0.5), deterministically entity_a wins (lexicographically first)

        // Build the bytecode manually to use Vec3 position
        let mut chunk = BytecodeChunk::new();
        let entity_idx = chunk.add_entity("particles");
        let position_field_idx = chunk.add_component("location");
        let component_idx = chunk.add_component("value");
        let pos_lit_idx = chunk.add_literal(Value::Vec3([0.5, 0.0, 0.0]));
        chunk.emit(Op::Literal(pos_lit_idx));
        chunk.emit(Op::LoadNearestField(
            entity_idx,
            position_field_idx,
            component_idx,
        ));

        let mut ctx = SpatialTestContext::new();
        let result = execute(&chunk, &mut ctx);

        // entity_a (at 0,0,0) and entity_b (at 1,0,0) are both 0.5 away
        // entity_a wins lexicographically, its value is 10.0
        assert_eq!(result, Value::Scalar(10.0));
    }

    #[test]
    fn test_within_aggregate_with_custom_position_field() {
        // Sum values of entities within radius 2.0 of position (0, 0, 0) using "location" field
        // entity_a at (0, 0, 0) - distance 0.0 - included
        // entity_b at (1, 0, 0) - distance 1.0 - included
        // entity_c at (10, 0, 0) - distance 10.0 - excluded
        // Sum = 10 + 20 = 30

        let mut chunk = BytecodeChunk::new();
        let entity_idx = chunk.add_entity("particles");
        let position_field_idx = chunk.add_component("location");

        // Body chunk: just load the "value" field
        let mut body_chunk = BytecodeChunk::new();
        let value_idx = body_chunk.add_component("value");
        body_chunk.emit(Op::LoadSelfField(value_idx));

        let body_idx = chunk.add_sub_chunk(body_chunk);

        // Push position (0, 0, 0)
        let pos_lit_idx = chunk.add_literal(Value::Vec3([0.0, 0.0, 0.0]));
        chunk.emit(Op::Literal(pos_lit_idx));

        // Push radius 2.0
        let radius_lit_idx = chunk.add_literal(Value::Scalar(2.0));
        chunk.emit(Op::Literal(radius_lit_idx));

        // Emit WithinAggregate with Sum reduction
        chunk.emit(Op::WithinAggregate(
            entity_idx,
            position_field_idx,
            ReductionOp::Sum,
            body_idx,
        ));

        let mut ctx = SpatialTestContext::new();
        let result = execute(&chunk, &mut ctx);

        // Sum of entity_a (10) + entity_b (20) = 30
        assert_eq!(result, Value::Scalar(30.0));
    }

    #[test]
    fn test_within_aggregate_count_with_custom_position_field() {
        // Count entities within radius 1.5 of position (0, 0, 0) using "location" field
        // entity_a at (0, 0, 0) - distance 0.0 - included
        // entity_b at (1, 0, 0) - distance 1.0 - included
        // entity_c at (10, 0, 0) - distance 10.0 - excluded
        // Count = 2

        let mut chunk = BytecodeChunk::new();
        let entity_idx = chunk.add_entity("particles");
        let position_field_idx = chunk.add_component("location");

        // Body chunk: just return 1
        let mut body_chunk = BytecodeChunk::new();
        let one_idx = body_chunk.add_literal(Value::Scalar(1.0));
        body_chunk.emit(Op::Literal(one_idx));

        let body_idx = chunk.add_sub_chunk(body_chunk);

        // Push position (0, 0, 0)
        let pos_lit_idx = chunk.add_literal(Value::Vec3([0.0, 0.0, 0.0]));
        chunk.emit(Op::Literal(pos_lit_idx));

        // Push radius 1.5
        let radius_lit_idx = chunk.add_literal(Value::Scalar(1.5));
        chunk.emit(Op::Literal(radius_lit_idx));

        // Emit WithinAggregate with Count reduction
        chunk.emit(Op::WithinAggregate(
            entity_idx,
            position_field_idx,
            ReductionOp::Count,
            body_idx,
        ));

        let mut ctx = SpatialTestContext::new();
        let result = execute(&chunk, &mut ctx);

        // Count of entity_a + entity_b = 2
        assert_eq!(result, Value::Integer(2));
    }
}
