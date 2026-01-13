//! Lowering from CompiledExpr to SSA IR.
//!
//! Transforms tree-structured `CompiledExpr` into linear SSA form.

use std::collections::HashMap;

use crate::CompiledExpr;

use super::{BlockId, SsaFunction, SsaInstruction, Terminator, VReg};

/// Lower a CompiledExpr to SSA IR.
pub fn lower_to_ssa(expr: &CompiledExpr) -> SsaFunction {
    let mut ctx = LoweringContext::new();
    let result = ctx.lower_expr(expr, BlockId(0));
    ctx.func
        .block_mut(ctx.current_block)
        .terminate(Terminator::Return(result));
    ctx.func
}

/// Context for lowering expressions to SSA.
struct LoweringContext {
    func: SsaFunction,
    current_block: BlockId,
    /// Map from local variable names to their current VReg values.
    locals: HashMap<String, VReg>,
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            func: SsaFunction::new(),
            current_block: BlockId(0),
            locals: HashMap::new(),
        }
    }

    /// Lower an expression and return the VReg containing the result.
    fn lower_expr(&mut self, expr: &CompiledExpr, block: BlockId) -> VReg {
        self.current_block = block;

        match expr {
            CompiledExpr::Literal(value, _) => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadConst { dst, value: *value });
                dst
            }

            CompiledExpr::Prev => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadPrev { dst });
                dst
            }

            CompiledExpr::DtRaw => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadDt { dst });
                dst
            }

            CompiledExpr::SimTime => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadSimTime { dst });
                dst
            }

            CompiledExpr::Collected => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadCollected { dst });
                dst
            }

            CompiledExpr::Signal(id) => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadSignal {
                    dst,
                    signal: id.clone(),
                });
                dst
            }

            CompiledExpr::Const(_name, _) => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadConst {
                    dst,
                    value: 0.0, // Should look up actual value from world
                });
                dst
            }

            CompiledExpr::Config(name, _) => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadConfig {
                    dst,
                    name: name.clone(),
                });
                dst
            }

            CompiledExpr::Binary { op, left, right } => {
                let lhs = self.lower_expr(left, self.current_block);
                let rhs = self.lower_expr(right, self.current_block);
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::BinOp {
                    dst,
                    op: *op,
                    lhs,
                    rhs,
                });
                dst
            }

            CompiledExpr::Unary { op, operand } => {
                let operand_reg = self.lower_expr(operand, self.current_block);
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::UnaryOp {
                    dst,
                    op: *op,
                    operand: operand_reg,
                });
                dst
            }

            CompiledExpr::Call { function, args } => {
                let arg_regs: Vec<_> = args
                    .iter()
                    .map(|a| self.lower_expr(a, self.current_block))
                    .collect();
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::Call {
                    dst,
                    function: function.clone(),
                    args: arg_regs,
                });
                dst
            }

            CompiledExpr::KernelCall { function, args } => {
                let arg_regs: Vec<_> = args
                    .iter()
                    .map(|a| self.lower_expr(a, self.current_block))
                    .collect();
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::KernelCall {
                    dst,
                    function: function.clone(),
                    args: arg_regs,
                });
                dst
            }

            CompiledExpr::FieldAccess { object, field } => {
                let obj_reg = self.lower_expr(object, self.current_block);
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::FieldAccess {
                    dst,
                    object: obj_reg,
                    field: field.clone(),
                });
                dst
            }

            CompiledExpr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_reg = self.lower_expr(condition, self.current_block);
                let then_block = self.func.alloc_block();
                let else_block = self.func.alloc_block();
                let merge_block = self.func.alloc_block();

                self.func
                    .block_mut(self.current_block)
                    .terminate(Terminator::Branch {
                        cond: cond_reg,
                        then_block,
                        else_block,
                    });

                let then_result = self.lower_expr(then_branch, then_block);
                let then_exit_block = self.current_block;
                self.func
                    .block_mut(then_exit_block)
                    .terminate(Terminator::Jump(merge_block));

                let else_result = self.lower_expr(else_branch, else_block);
                let else_exit_block = self.current_block;
                self.func
                    .block_mut(else_exit_block)
                    .terminate(Terminator::Jump(merge_block));

                self.current_block = merge_block;
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::Phi {
                    dst,
                    arms: vec![
                        (then_exit_block, then_result),
                        (else_exit_block, else_result),
                    ],
                });
                dst
            }

            CompiledExpr::Let { name, value, body } => {
                let value_reg = self.lower_expr(value, self.current_block);
                let old_value = self.locals.insert(name.clone(), value_reg);
                let result = self.lower_expr(body, self.current_block);
                if let Some(old) = old_value {
                    self.locals.insert(name.clone(), old);
                } else {
                    self.locals.remove(name);
                }
                result
            }

            CompiledExpr::Local(name) => {
                if let Some(&reg) = self.locals.get(name) {
                    reg
                } else {
                    let dst = self.func.alloc_vreg();
                    self.emit(SsaInstruction::LoadLocal {
                        dst,
                        name: name.clone(),
                    });
                    dst
                }
            }

            CompiledExpr::SelfField(field) => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::SelfField {
                    dst,
                    field: field.clone(),
                });
                dst
            }

            CompiledExpr::Aggregate { op, entity, body } => {
                let body_block = self.func.alloc_block();
                let _body_result = self.lower_expr(body, body_block);
                self.current_block = self.func.alloc_block();
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::Aggregate {
                    dst,
                    op: *op,
                    entity: entity.to_string(),
                    body_block,
                });
                dst
            }

            _ => {
                let dst = self.func.alloc_vreg();
                self.emit(SsaInstruction::LoadConst { dst, value: 0.0 });
                dst
            }
        }
    }

    fn emit(&mut self, inst: SsaInstruction) {
        self.func.block_mut(self.current_block).push(inst);
    }
}
