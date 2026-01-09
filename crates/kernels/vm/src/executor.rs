//! Bytecode executor
//!
//! Stack-based VM that executes compiled bytecode.

use crate::bytecode::{BytecodeChunk, Op};

/// Execution context providing runtime values
pub trait ExecutionContext {
    /// Get previous value of current signal
    fn prev(&self) -> f64;

    /// Get dt (time step)
    fn dt(&self) -> f64;

    /// Get sum of inputs for current signal
    fn inputs(&self) -> f64;

    /// Get signal value by name
    fn signal(&self, name: &str) -> f64;

    /// Get signal component by name and component (x, y, z, w)
    fn signal_component(&self, name: &str, component: &str) -> f64 {
        // Default implementation returns the full signal value
        let _ = component;
        self.signal(name)
    }

    /// Get constant value by name
    fn constant(&self, name: &str) -> f64;

    /// Get config value by name
    fn config(&self, name: &str) -> f64;

    /// Call a kernel function
    fn call_kernel(&self, name: &str, args: &[f64]) -> f64;
}

/// Execute bytecode with the given context
pub fn execute(chunk: &BytecodeChunk, ctx: &dyn ExecutionContext) -> f64 {
    let mut stack: Vec<f64> = Vec::with_capacity(32);
    let mut locals: Vec<f64> = vec![0.0; chunk.local_count as usize];
    let mut ip = 0;

    while ip < chunk.ops.len() {
        match chunk.ops[ip] {
            Op::Const(v) => {
                stack.push(v);
            }

            Op::LoadPrev => {
                stack.push(ctx.prev());
            }

            Op::LoadDt => {
                stack.push(ctx.dt());
            }

            Op::LoadInputs => {
                stack.push(ctx.inputs());
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

            Op::LoadConst(idx) => {
                let name = &chunk.constants[idx as usize];
                stack.push(ctx.constant(name));
            }

            Op::LoadConfig(idx) => {
                let name = &chunk.configs[idx as usize];
                stack.push(ctx.config(name));
            }

            Op::LoadLocal(slot) => {
                stack.push(locals[slot as usize]);
            }

            Op::StoreLocal(slot) => {
                let v = *stack.last().expect("vm bug: stack underflow");
                locals[slot as usize] = v;
            }

            Op::Add => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(l + r);
            }

            Op::Sub => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(l - r);
            }

            Op::Mul => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(l * r);
            }

            Op::Div => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(l / r);
            }

            Op::Pow => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(l.powf(r));
            }

            Op::Neg => {
                let v = stack.pop().expect("vm bug: stack underflow");
                stack.push(-v);
            }

            Op::Eq => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if (l - r).abs() < f64::EPSILON { 1.0 } else { 0.0 });
            }

            Op::Ne => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if (l - r).abs() >= f64::EPSILON { 1.0 } else { 0.0 });
            }

            Op::Lt => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if l < r { 1.0 } else { 0.0 });
            }

            Op::Le => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if l <= r { 1.0 } else { 0.0 });
            }

            Op::Gt => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if l > r { 1.0 } else { 0.0 });
            }

            Op::Ge => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if l >= r { 1.0 } else { 0.0 });
            }

            Op::And => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if l != 0.0 && r != 0.0 { 1.0 } else { 0.0 });
            }

            Op::Or => {
                let r = stack.pop().expect("vm bug: stack underflow");
                let l = stack.pop().expect("vm bug: stack underflow");
                stack.push(if l != 0.0 || r != 0.0 { 1.0 } else { 0.0 });
            }

            Op::Not => {
                let v = stack.pop().expect("vm bug: stack underflow");
                stack.push(if v == 0.0 { 1.0 } else { 0.0 });
            }

            Op::JumpIfZero(offset) => {
                let v = stack.pop().expect("vm bug: stack underflow");
                if v == 0.0 {
                    ip += offset as usize;
                }
            }

            Op::Jump(offset) => {
                ip += offset as usize;
            }

            Op::Call { kernel, arity } => {
                let name = &chunk.kernels[kernel as usize];
                let start = stack.len().saturating_sub(arity as usize);
                let args: Vec<f64> = stack.drain(start..).collect();
                let result = ctx.call_kernel(name, &args);
                stack.push(result);
            }

            Op::Dup => {
                let v = *stack.last().expect("vm bug: stack underflow");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{compile_expr, BinaryOp, Expr};

    struct TestContext;

    impl ExecutionContext for TestContext {
        fn prev(&self) -> f64 {
            100.0
        }
        fn dt(&self) -> f64 {
            0.1
        }
        fn inputs(&self) -> f64 {
            5.0
        }
        fn signal(&self, name: &str) -> f64 {
            match name {
                "temp" => 25.0,
                "pressure" => 101.0,
                _ => 0.0,
            }
        }
        fn constant(&self, name: &str) -> f64 {
            match name {
                "PI" => std::f64::consts::PI,
                _ => 0.0,
            }
        }
        fn config(&self, name: &str) -> f64 {
            match name {
                "scale" => 2.0,
                _ => 0.0,
            }
        }
        fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
            match name {
                "abs" => args.first().map(|v| v.abs()).unwrap_or(0.0),
                "min" => args.iter().copied().fold(f64::INFINITY, f64::min),
                "max" => args.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                _ => 0.0,
            }
        }
    }

    #[test]
    fn test_execute_literal() {
        let chunk = compile_expr(&Expr::Literal(42.0));
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_execute_binary() {
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Literal(10.0)),
            right: Box::new(Expr::Literal(32.0)),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_execute_prev() {
        let chunk = compile_expr(&Expr::Prev);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_execute_signal() {
        let chunk = compile_expr(&Expr::Signal("temp".to_string()));
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_execute_call() {
        let expr = Expr::Call {
            function: "abs".to_string(),
            args: vec![Expr::Literal(-5.0)],
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_execute_if_true() {
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(1.0)),
            then_branch: Box::new(Expr::Literal(10.0)),
            else_branch: Box::new(Expr::Literal(20.0)),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_execute_if_false() {
        let expr = Expr::If {
            condition: Box::new(Expr::Literal(0.0)),
            then_branch: Box::new(Expr::Literal(10.0)),
            else_branch: Box::new(Expr::Literal(20.0)),
        };
        let chunk = compile_expr(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 20.0);
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
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 150.0);
    }
}
