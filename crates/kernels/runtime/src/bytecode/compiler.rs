//! Bytecode compiler from typed IR to bytecode.
//!
//! This module implements the compilation pass that converts execution blocks
//! (from `continuum_cdsl::ast::Execution`) into bytecode sequences.
//!
//! # Design Principles
//!
//! 1. **One Truth** - All IR → bytecode conversion lives here
//! 2. **Deterministic** - Stable ordering from IR structure
//! 3. **Explicit errors** - Invalid IR is a compile error, never silent
//! 4. **Phase-aware** - Validates capability and phase constraints
//!
//! # Compilation Strategy
//!
//! 1. Walk the typed expression tree (ExecutionBody)
//! 2. Emit opcodes in depth-first order
//! 3. Allocate slots for locals and temporaries
//! 4. Validate phase constraints and capabilities
//! 5. Generate explicit Return opcode

use std::collections::BTreeMap;

use continuum_cdsl::ast::{Execution, ExecutionBody, ExprKind, TypedExpr, TypedStmt};
use continuum_foundation::{Path, Phase, Value};

use crate::bytecode::opcode::{Instruction, OpcodeKind};
use crate::bytecode::operand::{BlockId, Operand, Slot};
use crate::bytecode::program::{BytecodeBlock, BytecodeProgram};

/// Result of a successful compilation pass.
///
/// A compiled block contains the bytecode program, its root entry point,
/// and metadata required by the runtime DAG for scheduling and validation.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    /// Phase this block executes in.
    ///
    /// The phase determines which opcodes are valid and which capabilities
    /// (e.g., LoadPrev, Emit) are available during execution.
    pub phase: Phase,
    /// Bytecode program containing the root block and any nested blocks (e.g., from aggregates/folds).
    pub program: BytecodeProgram,
    /// Root block id where execution begins.
    pub root: BlockId,
    /// Total number of slots required to hold all locals, signals, and temporaries.
    ///
    /// This is used by the executor to pre-allocate slot storage.
    pub slot_count: u32,
    /// Signal paths read by this block. Used for DAG dependency analysis.
    pub reads: Vec<Path>,
    /// Signal paths read with `prev`. Used to ensure history availability.
    pub temporal_reads: Vec<Path>,
    /// Signal paths emitted to by this block. Used for DAG dependency analysis.
    pub emits: Vec<Path>,
}

/// Converts typed IR execution blocks into bytecode sequences.
///
/// The compiler performs a single-pass walk of the typed IR tree, allocating
/// slots for bindings and temporaries while emitting opcodes.
pub struct Compiler {
    /// Next available slot index for allocation.
    next_slot: u32,
    /// Program being assembled during the current compilation pass.
    program: BytecodeProgram,
    /// Stack of local scopes used for name resolution.
    scopes: Vec<BTreeMap<String, Slot>>,
}

impl Compiler {
    /// Create a new compiler instance with an empty initial scope.
    pub fn new() -> Self {
        Self {
            next_slot: 0,
            program: BytecodeProgram::new(),
            scopes: vec![BTreeMap::new()],
        }
    }

    /// Compile an execution block from IR
    ///
    /// This is the main entry point for compilation. Takes a typed Execution
    /// from the IR and produces a CompiledBlock.
    ///
    /// Compiler state (slots and scopes) is reset per call.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Phase constraints are violated
    /// - Capabilities are missing
    /// - Invalid IR structure
    pub fn compile_execution(
        &mut self,
        execution: &Execution,
    ) -> Result<CompiledBlock, CompileError> {
        self.next_slot = 0;
        self.program = BytecodeProgram::new();
        self.scopes = vec![BTreeMap::new()];
        let returns_value = matches!(execution.body, ExecutionBody::Expr(_));
        let mut root_block = BytecodeBlock::new(returns_value);
        self.compile_body(&mut root_block, &execution.body)?;
        root_block
            .instructions
            .push(Instruction::new(OpcodeKind::Return, vec![]));

        let root = self.program.add_block(root_block);
        let program = std::mem::replace(&mut self.program, BytecodeProgram::new());
        let slot_count = self.next_slot;

        self.validate_program(execution.phase, &program)?;

        Ok(CompiledBlock {
            phase: execution.phase,
            program,
            root,
            slot_count,
            reads: execution.reads.clone(),
            temporal_reads: execution.temporal_reads.clone(),
            emits: execution.emits.clone(),
        })
    }

    fn compile_body(
        &mut self,
        block: &mut BytecodeBlock,
        body: &ExecutionBody,
    ) -> Result<(), CompileError> {
        match body {
            ExecutionBody::Expr(expr) => {
                self.compile_expr(block, expr)?;
                Ok(())
            }
            ExecutionBody::Statements(statements) => {
                for stmt in statements {
                    self.compile_stmt(block, stmt)?;
                }
                Ok(())
            }
        }
    }

    fn compile_stmt(
        &mut self,
        block: &mut BytecodeBlock,
        stmt: &TypedStmt,
    ) -> Result<(), CompileError> {
        use continuum_cdsl::ast::block::Stmt;

        match stmt {
            Stmt::Let { name, value, .. } => {
                self.compile_let(block, name.clone(), value, None)?;
            }
            Stmt::SignalAssign { target, value, .. } => {
                self.compile_expr(block, value)?;
                block.instructions.push(Instruction::new(
                    OpcodeKind::Emit,
                    vec![Operand::Signal(target.clone())],
                ));
            }
            Stmt::FieldAssign {
                target,
                position,
                value,
                ..
            } => {
                self.compile_expr(block, position)?;
                self.compile_expr(block, value)?;
                block.instructions.push(Instruction::new(
                    OpcodeKind::EmitField,
                    vec![Operand::Field(target.clone())],
                ));
            }
            Stmt::Expr(expr) => {
                self.compile_expr(block, expr)?;
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::Pop, vec![]));
            }
        }

        Ok(())
    }

    fn compile_expr(
        &mut self,
        block: &mut BytecodeBlock,
        expr: &TypedExpr,
    ) -> Result<(), CompileError> {
        match &expr.expr {
            ExprKind::Literal { value, .. } => {
                block.instructions.push(Instruction::new(
                    OpcodeKind::PushLiteral,
                    vec![Operand::Literal(Value::Scalar(*value))],
                ));
            }
            ExprKind::Vector(values) => {
                self.compile_vector(block, values)?;
            }
            ExprKind::Local(name) => {
                let slot = self
                    .lookup_local(name)
                    .ok_or_else(|| CompileError::InvalidIR {
                        message: format!("Unknown local binding: {name}"),
                    })?;
                block.instructions.push(Instruction::new(
                    OpcodeKind::Load,
                    vec![Operand::Slot(slot)],
                ));
            }
            ExprKind::Signal(path) => {
                block.instructions.push(Instruction::new(
                    OpcodeKind::LoadSignal,
                    vec![Operand::Signal(path.clone())],
                ));
            }
            ExprKind::Field(path) => {
                block.instructions.push(Instruction::new(
                    OpcodeKind::LoadField,
                    vec![Operand::Field(path.clone())],
                ));
            }
            ExprKind::Config(path) => {
                block.instructions.push(Instruction::new(
                    OpcodeKind::LoadConfig,
                    vec![Operand::Config(path.clone())],
                ));
            }
            ExprKind::Const(path) => {
                block.instructions.push(Instruction::new(
                    OpcodeKind::LoadConst,
                    vec![Operand::Const(path.clone())],
                ));
            }
            ExprKind::Prev => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadPrev, vec![]));
            }
            ExprKind::Current => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadCurrent, vec![]));
            }
            ExprKind::Inputs => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadInputs, vec![]));
            }
            ExprKind::Dt => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadDt, vec![]));
            }
            ExprKind::Self_ => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadSelf, vec![]));
            }
            ExprKind::Other => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadOther, vec![]));
            }
            ExprKind::Payload => {
                block
                    .instructions
                    .push(Instruction::new(OpcodeKind::LoadPayload, vec![]));
            }
            ExprKind::Let { name, value, body } => {
                self.compile_let(block, name.clone(), value, Some(body))?;
            }
            ExprKind::Aggregate {
                op,
                entity,
                binding,
                body,
            } => {
                self.compile_aggregate(block, *op, entity, binding, body)?;
            }
            ExprKind::Fold {
                entity,
                init,
                acc,
                elem,
                body,
            } => {
                self.compile_fold(block, entity, init, acc, elem, body)?;
            }
            ExprKind::Call { kernel, args } => {
                self.compile_call(block, *kernel, args)?;
            }
            ExprKind::Struct { fields, .. } => {
                self.compile_struct(block, fields)?;
            }
            ExprKind::FieldAccess { object, field } => {
                self.compile_field_access(block, object, field)?;
            }
        }

        Ok(())
    }

    fn compile_vector(
        &mut self,
        block: &mut BytecodeBlock,
        values: &[TypedExpr],
    ) -> Result<(), CompileError> {
        for value in values {
            self.compile_expr(block, value)?;
        }
        block.instructions.push(Instruction::new(
            OpcodeKind::BuildVector,
            vec![Operand::Literal(Value::Integer(values.len() as i64))],
        ));
        Ok(())
    }

    fn compile_field_access(
        &mut self,
        block: &mut BytecodeBlock,
        object: &TypedExpr,
        field: &str,
    ) -> Result<(), CompileError> {
        self.compile_expr(block, object)?;
        block.instructions.push(Instruction::new(
            OpcodeKind::FieldAccess,
            vec![Operand::String(field.to_string())],
        ));
        Ok(())
    }

    fn compile_let(
        &mut self,
        block: &mut BytecodeBlock,
        name: String,
        value: &TypedExpr,
        body: Option<&TypedExpr>,
    ) -> Result<(), CompileError> {
        let slot = self.alloc_slot();
        self.compile_expr(block, value)?;

        block.instructions.push(Instruction::new(
            OpcodeKind::Store,
            vec![Operand::Slot(slot)],
        ));
        block
            .instructions
            .push(Instruction::new(OpcodeKind::Let, vec![Operand::Slot(slot)]));

        if let Some(body_expr) = body {
            // Expression let: push scope, compile body, pop, endlet
            self.push_scope();
            self.bind_local(name, slot);
            self.compile_expr(block, body_expr)?;
            self.pop_scope();
            block
                .instructions
                .push(Instruction::new(OpcodeKind::EndLet, vec![]));
        } else {
            // Statement let: just bind in current scope
            self.bind_local(name, slot);
        }

        Ok(())
    }

    fn compile_aggregate(
        &mut self,
        block: &mut BytecodeBlock,
        op: continuum_cdsl::ast::expr::AggregateOp,
        entity: &continuum_cdsl::ast::Entity,
        binding: &String,
        body: &TypedExpr,
    ) -> Result<(), CompileError> {
        ensure_aggregate_supported(op)?;
        let binding_slot = self.alloc_slot();
        let mut aggregate_block = BytecodeBlock::new(true);
        self.push_scope();
        self.bind_local(binding.clone(), binding_slot);
        self.compile_expr(&mut aggregate_block, body)?;
        aggregate_block
            .instructions
            .push(Instruction::new(OpcodeKind::Return, vec![]));
        self.pop_scope();
        let block_id = self.program.add_block(aggregate_block);
        block.instructions.push(Instruction::new(
            OpcodeKind::Aggregate,
            vec![
                Operand::Entity(entity.clone()),
                Operand::Slot(binding_slot),
                Operand::Block(block_id),
                Operand::AggregateOp(op),
            ],
        ));
        Ok(())
    }

    fn compile_fold(
        &mut self,
        block: &mut BytecodeBlock,
        entity: &continuum_cdsl::ast::Entity,
        init: &TypedExpr,
        acc: &String,
        elem: &String,
        body: &TypedExpr,
    ) -> Result<(), CompileError> {
        let acc_slot = self.alloc_slot();
        let elem_slot = self.alloc_slot();
        self.compile_expr(block, init)?;
        block.instructions.push(Instruction::new(
            OpcodeKind::Store,
            vec![Operand::Slot(acc_slot)],
        ));

        let mut fold_block = BytecodeBlock::new(true);
        self.push_scope();
        self.bind_local(acc.clone(), acc_slot);
        self.bind_local(elem.clone(), elem_slot);
        self.compile_expr(&mut fold_block, body)?;
        fold_block
            .instructions
            .push(Instruction::new(OpcodeKind::Return, vec![]));
        self.pop_scope();
        let block_id = self.program.add_block(fold_block);
        block.instructions.push(Instruction::new(
            OpcodeKind::Fold,
            vec![
                Operand::Entity(entity.clone()),
                Operand::Slot(acc_slot),
                Operand::Slot(elem_slot),
                Operand::Block(block_id),
            ],
        ));
        Ok(())
    }

    fn compile_call(
        &mut self,
        block: &mut BytecodeBlock,
        kernel: continuum_kernel_types::KernelOp,
        args: &[TypedExpr],
    ) -> Result<(), CompileError> {
        for arg in args {
            self.compile_expr(block, arg)?;
        }
        block.instructions.push(Instruction::kernel_call(
            kernel,
            vec![Operand::Literal(Value::Integer(args.len() as i64))],
        ));
        Ok(())
    }

    fn compile_struct(
        &mut self,
        block: &mut BytecodeBlock,
        fields: &[(String, TypedExpr)],
    ) -> Result<(), CompileError> {
        for (_, value) in fields {
            self.compile_expr(block, value)?;
        }
        let operands = fields
            .iter()
            .map(|(name, _)| Operand::String(name.clone()))
            .collect();
        block
            .instructions
            .push(Instruction::new(OpcodeKind::BuildStruct, operands));
        Ok(())
    }

    fn alloc_slot(&mut self) -> Slot {
        let slot = Slot::new(self.next_slot);
        self.next_slot += 1;
        slot
    }

    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop_scope(&mut self) {
        if self.scopes.len() <= 1 {
            panic!("Compiler scope underflow");
        }
        self.scopes.pop();
    }

    fn bind_local(&mut self, name: String, slot: Slot) {
        let scope = self
            .scopes
            .last_mut()
            .expect("Compiler scope stack missing");
        scope.insert(name, slot);
    }

    fn lookup_local(&self, name: &str) -> Option<Slot> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn validate_program(
        &self,
        phase: Phase,
        program: &BytecodeProgram,
    ) -> Result<(), CompileError> {
        for block in program.blocks() {
            for instruction in &block.instructions {
                if !instruction.kind.is_valid_in_phase(phase) {
                    return Err(CompileError::PhaseViolation {
                        opcode: format!("{:?}", instruction.kind),
                        phase,
                    });
                }
                if instruction.kind.has_effect()
                    && instruction.kind.metadata().allowed_phases.is_none()
                {
                    return Err(CompileError::PhaseViolation {
                        opcode: format!("{:?}", instruction.kind),
                        phase,
                    });
                }
            }
        }
        Ok(())
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_aggregate_supported(
    op: continuum_cdsl::ast::expr::AggregateOp,
) -> Result<(), CompileError> {
    if matches!(op, continuum_cdsl::ast::expr::AggregateOp::Map) {
        return Err(CompileError::InvalidIR {
            message: "Aggregate Map is not a runtime reduction opcode".to_string(),
        });
    }
    Ok(())
}

/// Compilation error types.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    /// An opcode was used in a phase where it is not allowed (e.g., `Emit` in `Resolve`).
    #[error("Opcode {opcode} not allowed in phase {phase:?}")]
    PhaseViolation {
        /// The forbidden opcode kind.
        opcode: String,
        /// The phase where it was attempted.
        phase: Phase,
    },

    /// The typed IR structure is invalid or contains unexpected nodes.
    #[error("Invalid IR: {message}")]
    InvalidIR {
        /// Description of the IR violation.
        message: String,
    },

    /// Expression type mismatch detected during compilation.
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        /// The expected type description.
        expected: String,
        /// The actual type found.
        found: String,
    },
}

#[cfg(test)]
mod tests;
