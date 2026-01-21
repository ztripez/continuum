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

/// Result of a successful compilation pass from typed IR to bytecode.
///
/// A compiled block contains the executable bytecode program and metadata
/// required by the deterministic runtime for dependency analysis, scheduling,
/// and resource allocation.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    /// The simulation phase this block was compiled for.
    ///
    /// Opcode validity and capability access (e.g., LoadPrev, Emit) are
    /// locked to this phase.
    pub phase: Phase,
    /// The bytecode program containing the instruction sequence and nested blocks.
    pub program: BytecodeProgram,
    /// The entry point block ID within the program (typically ID 0).
    pub root: BlockId,
    /// The total number of memory slots required to execute this program.
    ///
    /// This count includes space for all local let-bindings, signal lookups,
    /// and temporary computation results across all blocks in the program.
    pub slot_count: u32,
    /// List of signal paths that this block reads from the current tick.
    /// Used by the graph builder to establish incoming causal edges.
    pub reads: Vec<Path>,
    /// List of signal paths read with the `prev` keyword.
    /// Used to ensure that history buffers are maintained for these signals.
    pub temporal_reads: Vec<Path>,
    /// List of signal paths that this block emits to.
    /// Used by the graph builder to establish outgoing causal edges.
    pub emits: Vec<Path>,
}

/// A stateful compiler that transforms typed IR execution blocks into VM bytecode.
///
/// The compiler performs a depth-first walk of the typed IR tree, allocating
/// VM slots for bindings and intermediates while emitting instructions.
///
/// # Compilation Process
/// 1. **Slot Allocation**: Assigns stable indices to all local variables.
/// 2. **Opcode Emission**: Translates IR nodes into linear instruction sequences.
/// 3. **Block Management**: Handles nested scope construction for `Aggregate` and `Fold`.
/// 4. **Validation**: Enforces phase and capability constraints defined in the IR.
pub struct Compiler {
    /// Current slot index being allocated. Incremented for each new binding.
    next_slot: u32,
    /// The program currently being assembled.
    program: BytecodeProgram,
    /// Lexical scope stack mapping variable names to their allocated slots.
    scopes: Vec<BTreeMap<String, Slot>>,
}

impl Compiler {
    /// Creates a new compiler instance with an empty initial scope.
    pub fn new() -> Self {
        Self {
            next_slot: 0,
            program: BytecodeProgram::new(),
            scopes: vec![BTreeMap::new()],
        }
    }

    /// Compiles a typed IR [`Execution`] block into a [`CompiledBlock`].
    ///
    /// This is the primary entry point for bytecode generation. It resets the
    /// compiler's internal state (slots and scopes) before starting the walk.
    ///
    /// # Errors
    ///
    /// Returns a [`CompileError`] if:
    /// - The IR violates phase constraints (e.g., `Emit` in `Resolve` phase).
    /// - Required capabilities are missing from the IR definition.
    /// - The IR structure is inconsistent or unsupported.
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

    /// Compiles an IR execution body (either a single expression or a list of statements).
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

    /// Compiles a single typed IR statement into instructions.
    fn compile_stmt(
        &mut self,
        block: &mut BytecodeBlock,
        stmt: &TypedStmt,
    ) -> Result<(), CompileError> {
        use continuum_cdsl::ast::Stmt;

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

    /// Compiles a typed IR expression into instructions.
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
                self.compile_call(block, kernel.clone(), args)?;
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

    /// Compiles a vector construction expression.
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

    /// Compiles a field access expression (e.g., `pos.x`).
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

    /// Compiles a `let` binding, either as a statement or an expression.
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

    /// Compiles an aggregate operation (sum, count, etc.) into a nested bytecode block.
    fn compile_aggregate(
        &mut self,
        block: &mut BytecodeBlock,
        op: continuum_cdsl::ast::AggregateOp,
        entity: &continuum_foundation::EntityId,
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

    /// Compiles a fold operation into a nested bytecode block.
    fn compile_fold(
        &mut self,
        block: &mut BytecodeBlock,
        entity: &continuum_foundation::EntityId,
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

    /// Compiles a kernel call into a `CallKernel` instruction.
    fn compile_call(
        &mut self,
        block: &mut BytecodeBlock,
        kernel: continuum_kernel_types::KernelId,
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

    /// Compiles a struct literal construction.
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

    /// Allocates a new VM slot index.
    fn alloc_slot(&mut self) -> Slot {
        let slot = Slot::new(self.next_slot);
        self.next_slot += 1;
        slot
    }

    /// Pushes a new lexical scope onto the stack.
    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    /// Pops the current lexical scope.
    fn pop_scope(&mut self) {
        if self.scopes.len() <= 1 {
            panic!("Compiler scope underflow");
        }
        self.scopes.pop();
    }

    /// Binds a variable name to a slot in the current scope.
    fn bind_local(&mut self, name: String, slot: Slot) {
        let scope = self
            .scopes
            .last_mut()
            .expect("Compiler scope stack missing");
        scope.insert(name, slot);
    }

    /// Looks up the slot index for a variable name in the scope stack.
    fn lookup_local(&self, name: &str) -> Option<Slot> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    /// Validates that all opcodes in the program are permitted in the target phase.
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

/// Validates that an aggregate operation is supported by the bytecode VM.
fn ensure_aggregate_supported(
    op: continuum_cdsl::ast::AggregateOp,
) -> Result<(), CompileError> {
    if matches!(op, continuum_cdsl::ast::AggregateOp::Map) {
        return Err(CompileError::InvalidIR {
            message: "Aggregate Map is not a runtime reduction opcode".to_string(),
        });
    }
    Ok(())
}

/// Compilation error types encountered during IR → bytecode translation.
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
