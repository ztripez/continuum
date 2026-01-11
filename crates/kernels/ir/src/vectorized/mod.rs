//! L2 Vectorized Kernel Execution
//!
//! Executes SSA IR on arrays of values for SIMD-friendly vectorized execution.
//! This is the L2 lowering strategy that operates on entire populations at once.
//!
//! # Overview
//!
//! While L1 (instance-parallel) dispatches closures per instance with chunked
//! rayon parallelism, L2 vectorized execution interprets the SSA IR directly
//! on arrays, enabling SIMD auto-vectorization by the compiler.
//!
//! # Key Components
//!
//! - [`VRegBuffer`]: Intermediate storage for virtual register values
//! - [`L2VectorizedExecutor`]: Interprets SSA IR on arrays
//! - [`ScalarL2Kernel`]: `LaneKernel` implementation for L2 execution
//!
//! # Optimization Strategies
//!
//! 1. **Uniform Scalar Optimization**: Constants are stored as single values
//!    and broadcast only when needed for array operations
//!
//! 2. **SIMD-Friendly Loops**: Operations are structured as simple index-by-index
//!    loops that LLVM can auto-vectorize
//!
//! 3. **Cache-Friendly Access**: SoA layout ensures sequential memory access
//!    patterns for maximum cache utilization
//!
//! # SIMD Intrinsics
//!
//! The [`simd`] submodule provides hand-optimized implementations of common
//! expression patterns structured for maximum SIMD auto-vectorization.

pub mod simd;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use continuum_foundation::MemberSignalId;
use continuum_runtime::executor::{
    LaneKernel, LaneKernelError, LaneKernelResult, LoweringStrategy,
};
use continuum_runtime::soa_storage::{MemberSignalBuffer, PopulationStorage};
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::Dt;

use crate::ssa::{BlockId, SsaFunction, SsaInstruction, Terminator, VReg};
use crate::{BinaryOpIr, DtRobustOperator, IntegrationMethod, UnaryOpIr};

/// Error types for L2 vectorized execution.
#[derive(Debug, Clone)]
pub enum L2ExecutionError {
    /// A virtual register was used before being defined.
    UndefinedVReg(VReg),
    /// Array size mismatch in binary operation.
    SizeMismatch { expected: usize, got: usize },
    /// Unsupported operation for vectorized execution.
    UnsupportedOperation(String),
    /// Signal not found in storage.
    SignalNotFound(String),
    /// Block not found during execution.
    BlockNotFound(BlockId),
    /// No return value produced.
    NoReturnValue,
}

impl std::fmt::Display for L2ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            L2ExecutionError::UndefinedVReg(vreg) => {
                write!(f, "undefined virtual register: {:?}", vreg)
            }
            L2ExecutionError::SizeMismatch { expected, got } => {
                write!(f, "array size mismatch: expected {}, got {}", expected, got)
            }
            L2ExecutionError::UnsupportedOperation(op) => {
                write!(f, "unsupported operation for L2: {}", op)
            }
            L2ExecutionError::SignalNotFound(id) => {
                write!(f, "signal not found: {}", id)
            }
            L2ExecutionError::BlockNotFound(id) => {
                write!(f, "block not found: {:?}", id)
            }
            L2ExecutionError::NoReturnValue => {
                write!(f, "no return value produced")
            }
        }
    }
}

impl std::error::Error for L2ExecutionError {}

/// Buffer for storing virtual register values during L2 execution.
///
/// Supports three storage modes to optimize memory usage:
/// - `Scalar`: Full array of scalar values
/// - `Vec3`: Full array of 3D vector values
/// - `UniformScalar`: Single value that applies to all elements (broadcast)
#[derive(Debug, Clone)]
pub enum VRegBuffer {
    /// Array of scalar values (one per entity).
    Scalar(Vec<f64>),
    /// Array of Vec3 values (one per entity).
    Vec3(Vec<[f64; 3]>),
    /// Uniform scalar value (same for all entities, broadcast on demand).
    UniformScalar(f64),
}

impl VRegBuffer {
    /// Create a uniform scalar buffer (single value for all entities).
    pub fn uniform(value: f64) -> Self {
        VRegBuffer::UniformScalar(value)
    }

    /// Get the scalar at a specific index.
    pub fn get_scalar(&self, idx: usize) -> Option<f64> {
        match self {
            VRegBuffer::Scalar(arr) => arr.get(idx).copied(),
            VRegBuffer::UniformScalar(v) => Some(*v),
            VRegBuffer::Vec3(_) => None,
        }
    }

    /// Materialize to a full scalar array.
    pub fn to_scalar_array(&self, size: usize) -> Vec<f64> {
        match self {
            VRegBuffer::Scalar(arr) => arr.clone(),
            VRegBuffer::UniformScalar(v) => vec![*v; size],
            VRegBuffer::Vec3(_) => panic!("Cannot convert Vec3 to scalar array"),
        }
    }

    /// Get as scalar slice if this is a Scalar buffer.
    pub fn as_scalar_slice(&self) -> Option<&[f64]> {
        match self {
            VRegBuffer::Scalar(arr) => Some(arr),
            _ => None,
        }
    }

    /// Get as uniform scalar if this is a UniformScalar buffer.
    pub fn as_uniform(&self) -> Option<f64> {
        match self {
            VRegBuffer::UniformScalar(v) => Some(*v),
            _ => None,
        }
    }
}

/// L2 Vectorized Executor that interprets SSA IR on arrays.
///
/// This executor operates on entire populations at once, executing SSA
/// instructions as array operations that can be auto-vectorized by LLVM.
pub struct L2VectorizedExecutor {
    /// The SSA function to execute.
    ssa: Arc<SsaFunction>,
}

impl L2VectorizedExecutor {
    /// Create a new L2 executor for the given SSA function.
    pub fn new(ssa: Arc<SsaFunction>) -> Self {
        Self { ssa }
    }

    /// Execute the SSA function on scalar arrays.
    ///
    /// # Arguments
    ///
    /// * `prev_values` - Previous values for each entity
    /// * `dt` - Time delta for this tick
    /// * `signals` - Signal storage for reading other signals
    ///
    /// # Returns
    ///
    /// A vector of new values, one per entity.
    pub fn execute_scalar(
        &self,
        prev_values: &[f64],
        dt: Dt,
        signals: &SignalStorage,
    ) -> Result<Vec<f64>, L2ExecutionError> {
        self.execute_with_members(prev_values, dt, signals, None)
    }

    /// Execute the SSA function with access to member signal snapshot.
    ///
    /// This method implements **snapshot/next-state semantics** for entity resolve:
    /// - All `self.X` reads see the **snapshot** (previous tick values)
    /// - All writes go to the **next-state** buffer (current tick)
    /// - This enables full parallelism across all member signal resolvers
    ///
    /// # Arguments
    ///
    /// * `prev_values` - Previous values for the target signal (all entities)
    /// * `dt` - Time delta for this tick
    /// * `signals` - Signal storage for reading global signals
    /// * `members` - Optional member signal buffer for cross-member reads (snapshot)
    ///
    /// # Semantics
    ///
    /// When `members` is provided, `SelfField` instructions read from the
    /// **previous tick's buffer** (snapshot at resolve start), not the current
    /// buffer being written to. This ensures deterministic, parallelizable
    /// execution:
    ///
    /// ```text
    /// resolve {
    ///   self.velocity = integrate(self.velocity, forces)
    ///   self.position = integrate(self.position, self.velocity)
    ///   // self.velocity here reads PREVIOUS tick's velocity, not just-computed
    /// }
    /// ```
    ///
    /// # Returns
    ///
    /// A vector of new values, one per entity.
    pub fn execute_with_members(
        &self,
        prev_values: &[f64],
        dt: Dt,
        signals: &SignalStorage,
        members: Option<&MemberSignalBuffer>,
    ) -> Result<Vec<f64>, L2ExecutionError> {
        let population = prev_values.len();

        // Initialize virtual register storage
        let mut vregs: Vec<Option<VRegBuffer>> = vec![None; self.ssa.vreg_count as usize];

        // Start execution at block 0
        let mut current_block = BlockId(0);

        loop {
            let block = self
                .ssa
                .blocks
                .get(current_block.0 as usize)
                .ok_or(L2ExecutionError::BlockNotFound(current_block))?;

            // Execute all instructions in this block
            for inst in &block.instructions {
                self.execute_instruction(
                    inst,
                    &mut vregs,
                    prev_values,
                    dt.seconds(),
                    signals,
                    members,
                    population,
                )?;
            }

            // Handle terminator
            match &block.terminator {
                Some(Terminator::Return(vreg)) => {
                    let result = vregs[vreg.0 as usize]
                        .as_ref()
                        .ok_or(L2ExecutionError::UndefinedVReg(*vreg))?;
                    return Ok(result.to_scalar_array(population));
                }
                Some(Terminator::Jump(target)) => {
                    current_block = *target;
                }
                Some(Terminator::Branch {
                    cond: _cond,
                    then_block,
                    else_block,
                }) => {
                    // For L2, we execute both paths and merge with phi nodes
                    // This is a simplification - full predicated execution is issue #80
                    // For now, we conservatively execute the "then" path
                    // TODO: Issue #80 will add proper predicated execution
                    let _ = else_block;
                    current_block = *then_block;
                }
                None => {
                    return Err(L2ExecutionError::NoReturnValue);
                }
            }
        }
    }

    /// Execute a single SSA instruction.
    fn execute_instruction(
        &self,
        inst: &SsaInstruction,
        vregs: &mut [Option<VRegBuffer>],
        prev_values: &[f64],
        dt: f64,
        signals: &SignalStorage,
        members: Option<&MemberSignalBuffer>,
        population: usize,
    ) -> Result<(), L2ExecutionError> {
        match inst {
            SsaInstruction::LoadConst { dst, value } => {
                vregs[dst.0 as usize] = Some(VRegBuffer::uniform(*value));
            }

            SsaInstruction::LoadPrev { dst } => {
                vregs[dst.0 as usize] = Some(VRegBuffer::Scalar(prev_values.to_vec()));
            }

            SsaInstruction::LoadDt { dst } => {
                vregs[dst.0 as usize] = Some(VRegBuffer::uniform(dt));
            }

            SsaInstruction::LoadSimTime { dst } => {
                // SimTime is uniform across all entities (accumulated simulation time)
                // For now, use 0.0 as L2 doesn't track sim_time yet
                vregs[dst.0 as usize] = Some(VRegBuffer::uniform(0.0));
            }

            SsaInstruction::LoadSignal { dst, signal } => {
                // Load signal value - for now assume scalar signals are uniform
                if let Some(value) =
                    signals.get_resolved(&continuum_runtime::SignalId(signal.0.clone()))
                {
                    let scalar = value.as_scalar().unwrap_or(0.0);
                    vregs[dst.0 as usize] = Some(VRegBuffer::uniform(scalar));
                } else {
                    return Err(L2ExecutionError::SignalNotFound(signal.0.clone()));
                }
            }

            SsaInstruction::LoadCollected { dst } => {
                // Collected values - for L2, assume uniform 0.0 for now
                // Full collected support requires member-signal infrastructure
                vregs[dst.0 as usize] = Some(VRegBuffer::uniform(0.0));
            }

            SsaInstruction::SelfField { dst, field } => {
                // Self field access - read from snapshot (previous tick buffer)
                // This implements snapshot/next-state semantics: all reads see previous tick
                if let Some(member_buf) = members {
                    // Try scalar first, then vec3
                    if let Some(snapshot_slice) = member_buf.prev_scalar_slice(field) {
                        vregs[dst.0 as usize] = Some(VRegBuffer::Scalar(snapshot_slice.to_vec()));
                    } else if let Some(_vec3_slice) = member_buf.prev_vec3_slice(field) {
                        // Vec3 fields not yet fully supported in L2
                        // Return scalar 0 for now (magnitude placeholder)
                        vregs[dst.0 as usize] = Some(VRegBuffer::Scalar(vec![0.0; population]));
                    } else {
                        // Field not found in member buffer - return zeros
                        vregs[dst.0 as usize] = Some(VRegBuffer::Scalar(vec![0.0; population]));
                    }
                } else {
                    // No member buffer provided - return zeros
                    vregs[dst.0 as usize] = Some(VRegBuffer::Scalar(vec![0.0; population]));
                }
            }

            SsaInstruction::BinOp { dst, op, lhs, rhs } => {
                let lhs_buf = vregs[lhs.0 as usize]
                    .as_ref()
                    .ok_or(L2ExecutionError::UndefinedVReg(*lhs))?;
                let rhs_buf = vregs[rhs.0 as usize]
                    .as_ref()
                    .ok_or(L2ExecutionError::UndefinedVReg(*rhs))?;

                let result = self.vectorized_binop(lhs_buf, rhs_buf, *op, population)?;
                vregs[dst.0 as usize] = Some(result);
            }

            SsaInstruction::UnaryOp { dst, op, operand } => {
                let operand_buf = vregs[operand.0 as usize]
                    .as_ref()
                    .ok_or(L2ExecutionError::UndefinedVReg(*operand))?;

                let result = self.vectorized_unaryop(operand_buf, *op, population)?;
                vregs[dst.0 as usize] = Some(result);
            }

            SsaInstruction::KernelCall {
                dst,
                function,
                args,
            } => {
                let arg_bufs: Vec<&VRegBuffer> = args
                    .iter()
                    .map(|vreg| {
                        vregs[vreg.0 as usize]
                            .as_ref()
                            .ok_or(L2ExecutionError::UndefinedVReg(*vreg))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let result = self.execute_kernel(function, &arg_bufs, population)?;
                vregs[dst.0 as usize] = Some(result);
            }

            SsaInstruction::DtRobustCall {
                dst,
                operator,
                args,
                method,
            } => {
                let arg_bufs: Vec<&VRegBuffer> = args
                    .iter()
                    .map(|vreg| {
                        vregs[vreg.0 as usize]
                            .as_ref()
                            .ok_or(L2ExecutionError::UndefinedVReg(*vreg))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let result =
                    self.execute_dt_robust(*operator, &arg_bufs, *method, dt, population)?;
                vregs[dst.0 as usize] = Some(result);
            }

            SsaInstruction::Phi { dst, arms } => {
                // Phi nodes merge values from different blocks
                // For simplified L2 (without predicated execution), take first defined arm
                for (_, vreg) in arms {
                    if let Some(buf) = &vregs[vreg.0 as usize] {
                        vregs[dst.0 as usize] = Some(buf.clone());
                        break;
                    }
                }
            }

            // TODO: Support more SSA instruction types
            SsaInstruction::LoadNamedConst { dst: _, name } => {
                return Err(L2ExecutionError::UnsupportedOperation(format!(
                    "LoadNamedConst '{}' not yet supported in L2",
                    name
                )));
            }
            SsaInstruction::LoadConfig { dst: _, name } => {
                return Err(L2ExecutionError::UnsupportedOperation(format!(
                    "LoadConfig '{}' not yet supported in L2",
                    name
                )));
            }
            SsaInstruction::LoadLocal { dst: _, name } => {
                return Err(L2ExecutionError::UnsupportedOperation(format!(
                    "LoadLocal '{}' not yet supported in L2",
                    name
                )));
            }
            SsaInstruction::Call {
                dst: _,
                function,
                args: _,
            } => {
                return Err(L2ExecutionError::UnsupportedOperation(format!(
                    "Call '{}' not yet supported in L2",
                    function
                )));
            }
            SsaInstruction::FieldAccess {
                dst: _,
                object: _,
                field,
            } => {
                return Err(L2ExecutionError::UnsupportedOperation(format!(
                    "FieldAccess '{}' not yet supported in L2",
                    field
                )));
            }
            SsaInstruction::StoreLocal { .. } => {
                // Store operations don't produce output for L2 array execution
                // They would be handled differently in a full implementation
            }
            SsaInstruction::Aggregate { op, entity, .. } => {
                // Aggregate operations require entity storage access
                // Not yet supported in L2 vectorized execution
                return Err(L2ExecutionError::UnsupportedOperation(format!(
                    "Aggregate '{:?}' over '{}' not yet supported in L2",
                    op, entity
                )));
            }
        }

        Ok(())
    }

    /// Execute a binary operation on VRegBuffers.
    fn vectorized_binop(
        &self,
        lhs: &VRegBuffer,
        rhs: &VRegBuffer,
        op: BinaryOpIr,
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        match (lhs.as_uniform(), rhs.as_uniform()) {
            // Both uniform - compute single value
            (Some(l), Some(r)) => {
                let result = apply_binop(l, r, op);
                Ok(VRegBuffer::uniform(result))
            }
            // LHS is array, RHS is uniform - broadcast RHS
            (None, Some(r)) => {
                let lhs_arr = lhs.to_scalar_array(population);
                let mut result = Vec::with_capacity(population);
                vectorized_binop_broadcast_rhs(&lhs_arr, r, &mut result, op);
                Ok(VRegBuffer::Scalar(result))
            }
            // LHS is uniform, RHS is array - broadcast LHS
            (Some(l), None) => {
                let rhs_arr = rhs.to_scalar_array(population);
                let mut result = Vec::with_capacity(population);
                vectorized_binop_broadcast_lhs(l, &rhs_arr, &mut result, op);
                Ok(VRegBuffer::Scalar(result))
            }
            // Both arrays
            (None, None) => {
                let lhs_arr = lhs.to_scalar_array(population);
                let rhs_arr = rhs.to_scalar_array(population);
                let mut result = Vec::with_capacity(population);
                vectorized_binop(&lhs_arr, &rhs_arr, &mut result, op);
                Ok(VRegBuffer::Scalar(result))
            }
        }
    }

    /// Execute a unary operation on VRegBuffer.
    fn vectorized_unaryop(
        &self,
        operand: &VRegBuffer,
        op: UnaryOpIr,
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        if let Some(v) = operand.as_uniform() {
            let result = apply_unaryop(v, op);
            Ok(VRegBuffer::uniform(result))
        } else {
            let arr = operand.to_scalar_array(population);
            let mut result = Vec::with_capacity(population);
            for &x in &arr {
                result.push(apply_unaryop(x, op));
            }
            Ok(VRegBuffer::Scalar(result))
        }
    }

    /// Execute a kernel function call.
    fn execute_kernel(
        &self,
        function: &str,
        args: &[&VRegBuffer],
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        match function {
            "sqrt" => self.apply_unary_kernel(args, population, |x| x.sqrt()),
            "sin" => self.apply_unary_kernel(args, population, |x| x.sin()),
            "cos" => self.apply_unary_kernel(args, population, |x| x.cos()),
            "tan" => self.apply_unary_kernel(args, population, |x| x.tan()),
            "exp" => self.apply_unary_kernel(args, population, |x| x.exp()),
            "ln" => self.apply_unary_kernel(args, population, |x| x.ln()),
            "log10" => self.apply_unary_kernel(args, population, |x| x.log10()),
            "abs" => self.apply_unary_kernel(args, population, |x| x.abs()),
            "floor" => self.apply_unary_kernel(args, population, |x| x.floor()),
            "ceil" => self.apply_unary_kernel(args, population, |x| x.ceil()),
            "round" => self.apply_unary_kernel(args, population, |x| x.round()),
            "sign" => self.apply_unary_kernel(args, population, |x| x.signum()),
            "min" => self.apply_binary_kernel(args, population, |a, b| a.min(b)),
            "max" => self.apply_binary_kernel(args, population, |a, b| a.max(b)),
            "pow" => self.apply_binary_kernel(args, population, |a, b| a.powf(b)),
            "atan2" => self.apply_binary_kernel(args, population, |a, b| a.atan2(b)),
            "clamp" => self.apply_ternary_kernel_clamp(args, population),
            "lerp" => self.apply_ternary_kernel_lerp(args, population),
            _ => Err(L2ExecutionError::UnsupportedOperation(format!(
                "kernel function: {}",
                function
            ))),
        }
    }

    /// Apply a unary kernel function.
    fn apply_unary_kernel<F>(
        &self,
        args: &[&VRegBuffer],
        population: usize,
        f: F,
    ) -> Result<VRegBuffer, L2ExecutionError>
    where
        F: Fn(f64) -> f64,
    {
        if args.len() != 1 {
            return Err(L2ExecutionError::UnsupportedOperation(
                "unary kernel requires 1 argument".to_string(),
            ));
        }

        if let Some(v) = args[0].as_uniform() {
            Ok(VRegBuffer::uniform(f(v)))
        } else {
            let arr = args[0].to_scalar_array(population);
            let result: Vec<f64> = arr.iter().map(|&x| f(x)).collect();
            Ok(VRegBuffer::Scalar(result))
        }
    }

    /// Apply a binary kernel function.
    fn apply_binary_kernel<F>(
        &self,
        args: &[&VRegBuffer],
        population: usize,
        f: F,
    ) -> Result<VRegBuffer, L2ExecutionError>
    where
        F: Fn(f64, f64) -> f64,
    {
        if args.len() != 2 {
            return Err(L2ExecutionError::UnsupportedOperation(
                "binary kernel requires 2 arguments".to_string(),
            ));
        }

        match (args[0].as_uniform(), args[1].as_uniform()) {
            (Some(a), Some(b)) => Ok(VRegBuffer::uniform(f(a, b))),
            (Some(a), None) => {
                let b_arr = args[1].to_scalar_array(population);
                let result: Vec<f64> = b_arr.iter().map(|&b| f(a, b)).collect();
                Ok(VRegBuffer::Scalar(result))
            }
            (None, Some(b)) => {
                let a_arr = args[0].to_scalar_array(population);
                let result: Vec<f64> = a_arr.iter().map(|&a| f(a, b)).collect();
                Ok(VRegBuffer::Scalar(result))
            }
            (None, None) => {
                let a_arr = args[0].to_scalar_array(population);
                let b_arr = args[1].to_scalar_array(population);
                let result: Vec<f64> = a_arr
                    .iter()
                    .zip(b_arr.iter())
                    .map(|(&a, &b)| f(a, b))
                    .collect();
                Ok(VRegBuffer::Scalar(result))
            }
        }
    }

    /// Apply clamp kernel (ternary: value, min, max).
    fn apply_ternary_kernel_clamp(
        &self,
        args: &[&VRegBuffer],
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        if args.len() != 3 {
            return Err(L2ExecutionError::UnsupportedOperation(
                "clamp requires 3 arguments".to_string(),
            ));
        }

        // Get min and max (usually uniform)
        let min_v = args[1]
            .as_uniform()
            .unwrap_or_else(|| args[1].to_scalar_array(population)[0]);
        let max_v = args[2]
            .as_uniform()
            .unwrap_or_else(|| args[2].to_scalar_array(population)[0]);

        if let Some(val) = args[0].as_uniform() {
            Ok(VRegBuffer::uniform(val.clamp(min_v, max_v)))
        } else {
            let val_arr = args[0].to_scalar_array(population);
            let result: Vec<f64> = val_arr.iter().map(|&v| v.clamp(min_v, max_v)).collect();
            Ok(VRegBuffer::Scalar(result))
        }
    }

    /// Apply lerp kernel (ternary: a, b, t).
    fn apply_ternary_kernel_lerp(
        &self,
        args: &[&VRegBuffer],
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        if args.len() != 3 {
            return Err(L2ExecutionError::UnsupportedOperation(
                "lerp requires 3 arguments".to_string(),
            ));
        }

        // lerp(a, b, t) = a + (b - a) * t
        let a = args[0];
        let b = args[1];
        let t = args[2];

        match (a.as_uniform(), b.as_uniform(), t.as_uniform()) {
            (Some(a), Some(b), Some(t)) => Ok(VRegBuffer::uniform(a + (b - a) * t)),
            _ => {
                let a_arr = a.to_scalar_array(population);
                let b_arr = b.to_scalar_array(population);
                let t_arr = t.to_scalar_array(population);
                let result: Vec<f64> = a_arr
                    .iter()
                    .zip(b_arr.iter())
                    .zip(t_arr.iter())
                    .map(|((&a, &b), &t)| a + (b - a) * t)
                    .collect();
                Ok(VRegBuffer::Scalar(result))
            }
        }
    }

    /// Execute a dt-robust operator.
    fn execute_dt_robust(
        &self,
        operator: DtRobustOperator,
        args: &[&VRegBuffer],
        method: IntegrationMethod,
        dt: f64,
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        match operator {
            DtRobustOperator::Integrate => {
                if args.len() < 2 {
                    return Err(L2ExecutionError::UnsupportedOperation(
                        "integrate requires at least 2 arguments".to_string(),
                    ));
                }

                let prev = args[0];
                let rate = args[1];

                match method {
                    IntegrationMethod::Euler => {
                        // result = prev + rate * dt
                        let scale = dt;
                        self.integrate_euler(prev, rate, scale, population)
                    }
                    IntegrationMethod::Rk4 => {
                        // For Rk4, would need multiple evaluations
                        // For now, fall back to Euler (single evaluation approximation)
                        let scale = dt;
                        self.integrate_euler(prev, rate, scale, population)
                    }
                    IntegrationMethod::Verlet => {
                        // Velocity Verlet - for position-velocity systems
                        // For now, fall back to Euler
                        let scale = dt;
                        self.integrate_euler(prev, rate, scale, population)
                    }
                }
            }
            DtRobustOperator::Decay => {
                if args.len() < 2 {
                    return Err(L2ExecutionError::UnsupportedOperation(
                        "decay requires 2 arguments".to_string(),
                    ));
                }

                let prev = args[0];
                let half_life = args[1];

                // decay = prev * exp(-ln(2) * dt / half_life)
                match (prev.as_uniform(), half_life.as_uniform()) {
                    (Some(p), Some(h)) => {
                        let factor = (-std::f64::consts::LN_2 * dt / h).exp();
                        Ok(VRegBuffer::uniform(p * factor))
                    }
                    _ => {
                        let p_arr = prev.to_scalar_array(population);
                        let h_arr = half_life.to_scalar_array(population);
                        let result: Vec<f64> = p_arr
                            .iter()
                            .zip(h_arr.iter())
                            .map(|(&p, &h)| {
                                let factor = (-std::f64::consts::LN_2 * dt / h).exp();
                                p * factor
                            })
                            .collect();
                        Ok(VRegBuffer::Scalar(result))
                    }
                }
            }
            DtRobustOperator::Smooth => {
                if args.len() < 2 {
                    return Err(L2ExecutionError::UnsupportedOperation(
                        "smooth requires 2 arguments".to_string(),
                    ));
                }

                let prev = args[0];
                let target = args[1];
                // Default tau if not provided
                let tau = if args.len() > 2 {
                    args[2]
                        .as_uniform()
                        .unwrap_or_else(|| args[2].to_scalar_array(population)[0])
                } else {
                    1.0
                };

                // smooth = prev + (target - prev) * (1 - exp(-dt / tau))
                let factor = 1.0 - (-dt / tau).exp();

                match (prev.as_uniform(), target.as_uniform()) {
                    (Some(p), Some(t)) => Ok(VRegBuffer::uniform(p + (t - p) * factor)),
                    _ => {
                        let p_arr = prev.to_scalar_array(population);
                        let t_arr = target.to_scalar_array(population);
                        let result: Vec<f64> = p_arr
                            .iter()
                            .zip(t_arr.iter())
                            .map(|(&p, &t)| p + (t - p) * factor)
                            .collect();
                        Ok(VRegBuffer::Scalar(result))
                    }
                }
            }

            DtRobustOperator::Relax => {
                // relax(current, target, tau) - same as smooth
                if args.len() < 3 {
                    return Err(L2ExecutionError::UnsupportedOperation(
                        "relax requires 3 arguments".to_string(),
                    ));
                }
                let current = args[0];
                let target = args[1];
                let tau = args[2]
                    .as_uniform()
                    .unwrap_or_else(|| args[2].to_scalar_array(population)[0]);

                let factor = 1.0 - (-dt / tau).exp();

                match (current.as_uniform(), target.as_uniform()) {
                    (Some(c), Some(t)) => Ok(VRegBuffer::uniform(c + (t - c) * factor)),
                    _ => {
                        let c_arr = current.to_scalar_array(population);
                        let t_arr = target.to_scalar_array(population);
                        let result: Vec<f64> = c_arr
                            .iter()
                            .zip(t_arr.iter())
                            .map(|(&c, &t)| c + (t - c) * factor)
                            .collect();
                        Ok(VRegBuffer::Scalar(result))
                    }
                }
            }

            DtRobustOperator::Accumulate => {
                // accumulate(prev, delta, min, max) = clamp(prev + delta * dt, min, max)
                if args.len() < 4 {
                    return Err(L2ExecutionError::UnsupportedOperation(
                        "accumulate requires 4 arguments".to_string(),
                    ));
                }
                let prev = args[0];
                let delta = args[1];
                let min_val = args[2]
                    .as_uniform()
                    .unwrap_or_else(|| args[2].to_scalar_array(population)[0]);
                let max_val = args[3]
                    .as_uniform()
                    .unwrap_or_else(|| args[3].to_scalar_array(population)[0]);

                match (prev.as_uniform(), delta.as_uniform()) {
                    (Some(p), Some(d)) => {
                        let result = (p + d * dt).clamp(min_val, max_val);
                        Ok(VRegBuffer::uniform(result))
                    }
                    _ => {
                        let p_arr = prev.to_scalar_array(population);
                        let d_arr = delta.to_scalar_array(population);
                        let result: Vec<f64> = p_arr
                            .iter()
                            .zip(d_arr.iter())
                            .map(|(&p, &d)| (p + d * dt).clamp(min_val, max_val))
                            .collect();
                        Ok(VRegBuffer::Scalar(result))
                    }
                }
            }

            DtRobustOperator::AdvancePhase => {
                // advance_phase(phase, omega) = wrap(prev + omega * dt, 0, TAU)
                if args.len() < 2 {
                    return Err(L2ExecutionError::UnsupportedOperation(
                        "advance_phase requires 2 arguments".to_string(),
                    ));
                }
                let phase = args[0];
                let omega = args[1];

                let tau = std::f64::consts::TAU;

                match (phase.as_uniform(), omega.as_uniform()) {
                    (Some(p), Some(o)) => {
                        let result = (p + o * dt).rem_euclid(tau);
                        Ok(VRegBuffer::uniform(result))
                    }
                    _ => {
                        let p_arr = phase.to_scalar_array(population);
                        let o_arr = omega.to_scalar_array(population);
                        let result: Vec<f64> = p_arr
                            .iter()
                            .zip(o_arr.iter())
                            .map(|(&p, &o)| (p + o * dt).rem_euclid(tau))
                            .collect();
                        Ok(VRegBuffer::Scalar(result))
                    }
                }
            }

            DtRobustOperator::Damp => {
                // damp(pos, vel, target, stiffness, damping) - spring-damper system
                // This is a complex operation that needs velocity, so we return error for now
                Err(L2ExecutionError::UnsupportedOperation(
                    "damp (spring-damper) not yet implemented in L2".to_string(),
                ))
            }
        }
    }

    /// Euler integration: prev + rate * scale
    fn integrate_euler(
        &self,
        prev: &VRegBuffer,
        rate: &VRegBuffer,
        scale: f64,
        population: usize,
    ) -> Result<VRegBuffer, L2ExecutionError> {
        match (prev.as_uniform(), rate.as_uniform()) {
            (Some(p), Some(r)) => Ok(VRegBuffer::uniform(p + r * scale)),
            (Some(p), None) => {
                let r_arr = rate.to_scalar_array(population);
                let result: Vec<f64> = r_arr.iter().map(|&r| p + r * scale).collect();
                Ok(VRegBuffer::Scalar(result))
            }
            (None, Some(r)) => {
                let p_arr = prev.to_scalar_array(population);
                let result: Vec<f64> = p_arr.iter().map(|&p| p + r * scale).collect();
                Ok(VRegBuffer::Scalar(result))
            }
            (None, None) => {
                let p_arr = prev.to_scalar_array(population);
                let r_arr = rate.to_scalar_array(population);
                let result: Vec<f64> = p_arr
                    .iter()
                    .zip(r_arr.iter())
                    .map(|(&p, &r)| p + r * scale)
                    .collect();
                Ok(VRegBuffer::Scalar(result))
            }
        }
    }
}

/// L2 kernel implementation for scalar member signals.
///
/// This implements the `LaneKernel` trait using L2 vectorized execution.
pub struct ScalarL2Kernel {
    /// The member signal this kernel resolves.
    member_signal_id: MemberSignalId,
    /// Signal name for error messages.
    signal_name: String,
    /// The L2 executor.
    executor: L2VectorizedExecutor,
    /// Population hint for capacity pre-allocation.
    population_hint: usize,
}

impl ScalarL2Kernel {
    /// Create a new L2 kernel from SSA IR.
    pub fn new(
        member_signal_id: MemberSignalId,
        signal_name: String,
        ssa: Arc<SsaFunction>,
        population_hint: usize,
    ) -> Self {
        Self {
            member_signal_id,
            signal_name,
            executor: L2VectorizedExecutor::new(ssa),
            population_hint,
        }
    }
}

impl LaneKernel for ScalarL2Kernel {
    fn member_signal_id(&self) -> &MemberSignalId {
        &self.member_signal_id
    }

    fn strategy(&self) -> LoweringStrategy {
        LoweringStrategy::VectorKernel
    }

    fn population_hint(&self) -> usize {
        self.population_hint
    }

    fn execute(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError> {
        let start = std::time::Instant::now();

        // Get previous values from population storage
        let prev_values = population
            .signals()
            .prev_scalar_slice(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        let population_size = prev_values.len();

        // Get member signal buffer for snapshot semantics
        // All self.X reads will see previous tick values (snapshot)
        let member_signals = population.signals();

        // Execute L2 vectorized kernel with snapshot semantics
        let new_values = self
            .executor
            .execute_with_members(prev_values, dt, signals, Some(member_signals))
            .map_err(|e| {
                LaneKernelError::ExecutionFailed(format!(
                    "L2 kernel failed for {}: {}",
                    self.signal_name, e
                ))
            })?;

        // Write results to current buffer
        let current_slice = population
            .signals_mut()
            .scalar_slice_mut(&self.signal_name)
            .ok_or_else(|| LaneKernelError::SignalNotFound(self.signal_name.clone()))?;

        current_slice.copy_from_slice(&new_values);

        let elapsed_ns = start.elapsed().as_nanos() as u64;

        Ok(LaneKernelResult {
            instances_processed: population_size,
            execution_ns: Some(elapsed_ns),
        })
    }
}

// ============================================================================
// Helper functions for SIMD-friendly vectorized operations
// ============================================================================

/// Apply a binary operation to two scalars.
#[inline(always)]
fn apply_binop(lhs: f64, rhs: f64, op: BinaryOpIr) -> f64 {
    match op {
        BinaryOpIr::Add => lhs + rhs,
        BinaryOpIr::Sub => lhs - rhs,
        BinaryOpIr::Mul => lhs * rhs,
        BinaryOpIr::Div => lhs / rhs,
        BinaryOpIr::Pow => lhs.powf(rhs),
        BinaryOpIr::Eq => {
            if (lhs - rhs).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Ne => {
            if (lhs - rhs).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Lt => {
            if lhs < rhs {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Le => {
            if lhs <= rhs {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Gt => {
            if lhs > rhs {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Ge => {
            if lhs >= rhs {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::And => {
            if lhs != 0.0 && rhs != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Or => {
            if lhs != 0.0 || rhs != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Apply a unary operation to a scalar.
#[inline(always)]
fn apply_unaryop(x: f64, op: UnaryOpIr) -> f64 {
    match op {
        UnaryOpIr::Neg => -x,
        UnaryOpIr::Not => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// SIMD-friendly vectorized binary operation (both operands are arrays).
#[inline]
fn vectorized_binop(lhs: &[f64], rhs: &[f64], result: &mut Vec<f64>, op: BinaryOpIr) {
    match op {
        BinaryOpIr::Add => {
            for (&l, &r) in lhs.iter().zip(rhs.iter()) {
                result.push(l + r);
            }
        }
        BinaryOpIr::Sub => {
            for (&l, &r) in lhs.iter().zip(rhs.iter()) {
                result.push(l - r);
            }
        }
        BinaryOpIr::Mul => {
            for (&l, &r) in lhs.iter().zip(rhs.iter()) {
                result.push(l * r);
            }
        }
        BinaryOpIr::Div => {
            for (&l, &r) in lhs.iter().zip(rhs.iter()) {
                result.push(l / r);
            }
        }
        _ => {
            // Fall back to per-element application for other ops
            for (&l, &r) in lhs.iter().zip(rhs.iter()) {
                result.push(apply_binop(l, r, op));
            }
        }
    }
}

/// SIMD-friendly vectorized binary operation with broadcast RHS.
#[inline]
fn vectorized_binop_broadcast_rhs(lhs: &[f64], rhs: f64, result: &mut Vec<f64>, op: BinaryOpIr) {
    match op {
        BinaryOpIr::Add => {
            for &l in lhs {
                result.push(l + rhs);
            }
        }
        BinaryOpIr::Sub => {
            for &l in lhs {
                result.push(l - rhs);
            }
        }
        BinaryOpIr::Mul => {
            for &l in lhs {
                result.push(l * rhs);
            }
        }
        BinaryOpIr::Div => {
            for &l in lhs {
                result.push(l / rhs);
            }
        }
        _ => {
            for &l in lhs {
                result.push(apply_binop(l, rhs, op));
            }
        }
    }
}

/// SIMD-friendly vectorized binary operation with broadcast LHS.
#[inline]
fn vectorized_binop_broadcast_lhs(lhs: f64, rhs: &[f64], result: &mut Vec<f64>, op: BinaryOpIr) {
    match op {
        BinaryOpIr::Add => {
            for &r in rhs {
                result.push(lhs + r);
            }
        }
        BinaryOpIr::Sub => {
            for &r in rhs {
                result.push(lhs - r);
            }
        }
        BinaryOpIr::Mul => {
            for &r in rhs {
                result.push(lhs * r);
            }
        }
        BinaryOpIr::Div => {
            for &r in rhs {
                result.push(lhs / r);
            }
        }
        _ => {
            for &r in rhs {
                result.push(apply_binop(lhs, r, op));
            }
        }
    }
}
