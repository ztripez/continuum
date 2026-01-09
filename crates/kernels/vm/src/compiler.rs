//! Bytecode compiler
//!
//! Compiles CompiledExpr AST to flat bytecode.

use std::collections::HashMap;

use crate::bytecode::{BytecodeChunk, Op, SlotId};

/// Binary operator from IR
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    /// Addition (+)
    Add,
    /// Subtraction (-)
    Sub,
    /// Multiplication (*)
    Mul,
    /// Division (/)
    Div,
    /// Power (^)
    Pow,
    /// Equality (==)
    Eq,
    /// Inequality (!=)
    Ne,
    /// Less than (<)
    Lt,
    /// Less than or equal (<=)
    Le,
    /// Greater than (>)
    Gt,
    /// Greater than or equal (>=)
    Ge,
    /// Logical AND
    And,
    /// Logical OR
    Or,
}

/// Unary operator from IR
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    /// Negation (-)
    Neg,
    /// Logical NOT (!)
    Not,
}

/// Expression node for compilation
///
/// This is a simplified view that the compiler accepts.
/// The actual IR types are converted to this before compilation.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A literal floating-point value
    Literal(f64),
    /// Load the previous value of the current signal
    Prev,
    /// Load the time step (dt)
    DtRaw,
    /// Load the sum of inputs for the current signal
    Collected,
    /// Load a signal value by name
    Signal(String),
    /// Access a component of a vector signal (e.g., signal.x, signal.y)
    SignalComponent(String, String),
    /// Load a constant value by name
    Const(String),
    /// Load a configuration value by name
    Config(String),
    /// A binary operation
    Binary {
        /// The operator to apply
        op: BinaryOp,
        /// The left operand
        left: Box<Expr>,
        /// The right operand
        right: Box<Expr>,
    },
    /// A unary operation
    Unary {
        /// The operator to apply
        op: UnaryOp,
        /// The operand
        operand: Box<Expr>,
    },
    /// A function call
    Call {
        /// The name of the function to call
        function: String,
        /// The arguments to the function
        args: Vec<Expr>,
    },
    /// An if-else conditional expression
    If {
        /// The condition to test (non-zero is true)
        condition: Box<Expr>,
        /// The branch to execute if true
        then_branch: Box<Expr>,
        /// The branch to execute if false
        else_branch: Box<Expr>,
    },
    /// A let-binding (local variable)
    Let {
        /// The name of the variable
        name: String,
        /// The value to bind
        value: Box<Expr>,
        /// The expression where the binding is active
        body: Box<Expr>,
    },
    /// A reference to a local variable
    Local(String),
}

/// Compiler state
struct Compiler {
    chunk: BytecodeChunk,
    locals: HashMap<String, SlotId>,
    next_local: SlotId,
}

impl Compiler {
    fn new() -> Self {
        Self {
            chunk: BytecodeChunk::new(),
            locals: HashMap::new(),
            next_local: 0,
        }
    }

    fn compile_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal(v) => {
                self.chunk.emit(Op::Const(*v));
            }

            Expr::Prev => {
                self.chunk.emit(Op::LoadPrev);
            }

            Expr::DtRaw => {
                self.chunk.emit(Op::LoadDt);
            }

            Expr::Collected => {
                self.chunk.emit(Op::LoadInputs);
            }

            Expr::Signal(name) => {
                let idx = self.chunk.add_signal(name);
                self.chunk.emit(Op::LoadSignal(idx));
            }

            Expr::SignalComponent(signal, component) => {
                let signal_idx = self.chunk.add_signal(signal);
                let component_idx = self.chunk.add_component(component);
                self.chunk.emit(Op::LoadSignalComponent(signal_idx, component_idx));
            }

            Expr::Const(name) => {
                let idx = self.chunk.add_constant(name);
                self.chunk.emit(Op::LoadConst(idx));
            }

            Expr::Config(name) => {
                let idx = self.chunk.add_config(name);
                self.chunk.emit(Op::LoadConfig(idx));
            }

            Expr::Binary { op, left, right } => {
                // Compile left, then right, then operator
                self.compile_expr(left);
                self.compile_expr(right);
                let op_inst = match op {
                    BinaryOp::Add => Op::Add,
                    BinaryOp::Sub => Op::Sub,
                    BinaryOp::Mul => Op::Mul,
                    BinaryOp::Div => Op::Div,
                    BinaryOp::Pow => Op::Pow,
                    BinaryOp::Eq => Op::Eq,
                    BinaryOp::Ne => Op::Ne,
                    BinaryOp::Lt => Op::Lt,
                    BinaryOp::Le => Op::Le,
                    BinaryOp::Gt => Op::Gt,
                    BinaryOp::Ge => Op::Ge,
                    BinaryOp::And => Op::And,
                    BinaryOp::Or => Op::Or,
                };
                self.chunk.emit(op_inst);
            }

            Expr::Unary { op, operand } => {
                self.compile_expr(operand);
                let op_inst = match op {
                    UnaryOp::Neg => Op::Neg,
                    UnaryOp::Not => Op::Not,
                };
                self.chunk.emit(op_inst);
            }

            Expr::Call { function, args } => {
                // Push arguments left-to-right
                for arg in args {
                    self.compile_expr(arg);
                }
                let kernel = self.chunk.add_kernel(function);
                self.chunk.emit(Op::Call {
                    kernel,
                    arity: args.len() as u8,
                });
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                // Compile condition
                self.compile_expr(condition);

                // Jump to else if zero
                let jump_to_else = self.chunk.offset();
                self.chunk.emit(Op::JumpIfZero(0)); // placeholder

                // Compile then branch
                self.compile_expr(then_branch);

                // Jump over else
                let jump_over_else = self.chunk.offset();
                self.chunk.emit(Op::Jump(0)); // placeholder

                // Patch jump to else
                let else_start = self.chunk.offset();
                self.chunk
                    .patch_jump(jump_to_else, (else_start - jump_to_else - 1) as u16);

                // Compile else branch
                self.compile_expr(else_branch);

                // Patch jump over else
                let end = self.chunk.offset();
                self.chunk
                    .patch_jump(jump_over_else, (end - jump_over_else - 1) as u16);
            }

            Expr::Let { name, value, body } => {
                // Compile value
                self.compile_expr(value);

                // Allocate local slot
                let slot = self.next_local;
                self.next_local += 1;
                self.chunk.local_count = self.chunk.local_count.max(self.next_local);

                // Store to slot
                self.chunk.emit(Op::StoreLocal(slot));
                self.chunk.emit(Op::Pop); // StoreLocal doesn't pop

                // Register local
                let prev = self.locals.insert(name.clone(), slot);

                // Compile body
                self.compile_expr(body);

                // Restore previous binding (if shadowed)
                if let Some(prev_slot) = prev {
                    self.locals.insert(name.clone(), prev_slot);
                } else {
                    self.locals.remove(name);
                }
            }

            Expr::Local(name) => {
                if let Some(&slot) = self.locals.get(name) {
                    self.chunk.emit(Op::LoadLocal(slot));
                } else {
                    // Unknown local - emit 0 as fallback
                    self.chunk.emit(Op::Const(0.0));
                }
            }
        }
    }

    fn finish(self) -> BytecodeChunk {
        self.chunk
    }
}

/// Compile an expression to bytecode
pub fn compile_expr(expr: &Expr) -> BytecodeChunk {
    let mut compiler = Compiler::new();
    compiler.compile_expr(expr);
    compiler.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_literal() {
        let expr = Expr::Literal(42.0);
        let chunk = compile_expr(&expr);

        assert_eq!(chunk.ops.len(), 1);
        assert_eq!(chunk.ops[0], Op::Const(42.0));
    }

    #[test]
    fn test_compile_binary() {
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Literal(1.0)),
            right: Box::new(Expr::Literal(2.0)),
        };
        let chunk = compile_expr(&expr);

        assert_eq!(chunk.ops.len(), 3);
        assert_eq!(chunk.ops[0], Op::Const(1.0));
        assert_eq!(chunk.ops[1], Op::Const(2.0));
        assert_eq!(chunk.ops[2], Op::Add);
    }

    #[test]
    fn test_compile_call() {
        let expr = Expr::Call {
            function: "abs".to_string(),
            args: vec![Expr::Literal(-5.0)],
        };
        let chunk = compile_expr(&expr);

        assert_eq!(chunk.ops.len(), 2);
        assert_eq!(chunk.ops[0], Op::Const(-5.0));
        assert!(matches!(chunk.ops[1], Op::Call { kernel: 0, arity: 1 }));
        assert_eq!(chunk.kernels[0], "abs");
    }

    #[test]
    fn test_compile_if() {
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(1.0)),
            then_branch: Box::new(Expr::Literal(10.0)),
            else_branch: Box::new(Expr::Literal(20.0)),
        };
        let chunk = compile_expr(&expr);

        // Const(1.0), JumpIfZero, Const(10.0), Jump, Const(20.0)
        assert_eq!(chunk.ops.len(), 5);
        assert_eq!(chunk.ops[0], Op::Const(1.0));
        assert!(matches!(chunk.ops[1], Op::JumpIfZero(_)));
        assert_eq!(chunk.ops[2], Op::Const(10.0));
        assert!(matches!(chunk.ops[3], Op::Jump(_)));
        assert_eq!(chunk.ops[4], Op::Const(20.0));
    }

    #[test]
    fn test_compile_let() {
        // let a = 10.0
        // let b = 20.0
        // a + b
        let expr = Expr::Let {
            name: "a".to_string(),
            value: Box::new(Expr::Literal(10.0)),
            body: Box::new(Expr::Let {
                name: "b".to_string(),
                value: Box::new(Expr::Literal(20.0)),
                body: Box::new(Expr::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expr::Local("a".to_string())),
                    right: Box::new(Expr::Local("b".to_string())),
                }),
            }),
        };
        let chunk = compile_expr(&expr);

        // Should have local slots allocated
        assert_eq!(chunk.local_count, 2);
        // Ops: Const(10), StoreLocal(0), Pop, Const(20), StoreLocal(1), Pop, LoadLocal(0), LoadLocal(1), Add
        assert_eq!(chunk.ops.len(), 9);
    }
}
