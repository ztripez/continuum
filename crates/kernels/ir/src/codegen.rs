//! Bytecode Code Generation
//!
//! This module converts IR expressions ([`CompiledExpr`]) into executable
//! bytecode for the Continuum virtual machine.
//!
//! # Overview
//!
//! The code generator performs a straightforward recursive translation from
//! IR expression nodes to VM expression nodes, then delegates to the VM
//! compiler to produce actual bytecode.
//!
//! # Expression Mapping
//!
//! Most IR expression types map directly to VM equivalents:
//!
//! | IR Type | VM Type |
//! |---------|---------|
//! | `Literal(f64)` | `Literal(f64)` |
//! | `Prev` | `Prev` |
//! | `DtRaw` | `DtRaw` |
//! | `Signal(id)` | `Signal(name)` |
//! | `Binary { op, left, right }` | `Binary { op, left, right }` |
//! | etc. | etc. |
//!
//! # Entity Expressions
//!
//! Entity-related expressions (`SelfField`, `Aggregate`, `Filter`, etc.) are
//! **not** converted to bytecode. They return placeholder `Literal(0.0)` values
//! because entity operations require runtime access to entity storage, which
//! the bytecode VM does not have. Entity execution uses a separate interpreter.
//!
//! # Usage
//!
//! ```ignore
//! use continuum_ir::codegen::compile;
//!
//! let expr = CompiledExpr::Binary {
//!     op: BinaryOpIr::Add,
//!     left: Box::new(CompiledExpr::Prev),
//!     right: Box::new(CompiledExpr::Literal(1.0)),
//! };
//!
//! let bytecode = compile(&expr);
//! // Execute with: continuum_vm::execute(&bytecode, &context)
//! ```

use continuum_vm::BytecodeChunk;
use continuum_vm::compiler::{BinaryOp, Expr, UnaryOp};

use crate::{BinaryOpIr, CompiledExpr, DtRobustOperator, UnaryOpIr};

/// Converts an IR binary operator to its VM equivalent.
///
/// This is a direct 1:1 mapping as both representations use the same
/// operator semantics.
fn convert_binary_op(op: BinaryOpIr) -> BinaryOp {
    match op {
        BinaryOpIr::Add => BinaryOp::Add,
        BinaryOpIr::Sub => BinaryOp::Sub,
        BinaryOpIr::Mul => BinaryOp::Mul,
        BinaryOpIr::Div => BinaryOp::Div,
        BinaryOpIr::Pow => BinaryOp::Pow,
        BinaryOpIr::Eq => BinaryOp::Eq,
        BinaryOpIr::Ne => BinaryOp::Ne,
        BinaryOpIr::Lt => BinaryOp::Lt,
        BinaryOpIr::Le => BinaryOp::Le,
        BinaryOpIr::Gt => BinaryOp::Gt,
        BinaryOpIr::Ge => BinaryOp::Ge,
        BinaryOpIr::And => BinaryOp::And,
        BinaryOpIr::Or => BinaryOp::Or,
    }
}

/// Converts an IR unary operator to its VM equivalent.
fn convert_unary_op(op: UnaryOpIr) -> UnaryOp {
    match op {
        UnaryOpIr::Neg => UnaryOp::Neg,
        UnaryOpIr::Not => UnaryOp::Not,
    }
}

/// Converts a dt-robust operator to its function name for VM execution.
///
/// The function implementations are registered in the functions crate.
fn dt_robust_operator_name(op: DtRobustOperator) -> String {
    match op {
        DtRobustOperator::Integrate => "integrate".to_string(),
        DtRobustOperator::Decay => "decay".to_string(),
        DtRobustOperator::Relax => "relax".to_string(),
        DtRobustOperator::Accumulate => "accumulate".to_string(),
        DtRobustOperator::AdvancePhase => "advance_phase".to_string(),
        DtRobustOperator::Smooth => "smooth".to_string(),
        DtRobustOperator::Damp => "damp".to_string(),
    }
}

/// Recursively converts a [`CompiledExpr`] to a VM [`Expr`].
///
/// This function handles the translation of all expression types. Most types
/// map directly to VM equivalents, but entity expressions are not supported
/// and return placeholder values.
///
/// # Field Access
///
/// Field access on signals (e.g., `signal.pos.x`) is converted to
/// `SignalComponent` for vector component extraction.
fn convert_expr(expr: &CompiledExpr) -> Expr {
    match expr {
        CompiledExpr::Literal(v, _) => Expr::Literal(*v),
        CompiledExpr::Prev => Expr::Prev,
        CompiledExpr::DtRaw => Expr::DtRaw,
        CompiledExpr::SimTime => Expr::SimTime,
        CompiledExpr::Collected => Expr::Collected,
        CompiledExpr::Signal(id) => Expr::Signal(id.to_string()),
        CompiledExpr::Const(name, _) => Expr::Const(name.clone()),
        CompiledExpr::Config(name, _) => Expr::Config(name.clone()),
        CompiledExpr::Binary { op, left, right } => Expr::Binary {
            op: convert_binary_op(*op),
            left: Box::new(convert_expr(left)),
            right: Box::new(convert_expr(right)),
        },
        CompiledExpr::Unary { op, operand } => Expr::Unary {
            op: convert_unary_op(*op),
            operand: Box::new(convert_expr(operand)),
        },
        CompiledExpr::Call { function, args } => Expr::Call {
            function: function.clone(),
            args: args.iter().map(convert_expr).collect(),
        },
        CompiledExpr::KernelCall { function, args } => {
            // Kernel functions are converted to regular calls for the VM.
            // The VM dispatches by name; the distinction is kept in IR for
            // semantic analysis and potential GPU dispatch in the future.
            Expr::Call {
                function: format!("kernel.{}", function),
                args: args.iter().map(convert_expr).collect(),
            }
        }
        CompiledExpr::DtRobustCall {
            operator,
            args,
            method: _,
        } => {
            // Convert dt-robust operators to function calls for the VM.
            // The function name matches the operator, and the implementation
            // is provided by the functions registry.
            let function = dt_robust_operator_name(*operator);
            Expr::Call {
                function,
                args: args.iter().map(convert_expr).collect(),
            }
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => Expr::If {
            condition: Box::new(convert_expr(condition)),
            then_branch: Box::new(convert_expr(then_branch)),
            else_branch: Box::new(convert_expr(else_branch)),
        },
        CompiledExpr::Let { name, value, body } => Expr::Let {
            name: name.clone(),
            value: Box::new(convert_expr(value)),
            body: Box::new(convert_expr(body)),
        },
        CompiledExpr::Local(name) => Expr::Local(name.clone()),
        CompiledExpr::FieldAccess { object, field } => {
            // Handle field access on signals, prev, and collected (e.g., signal.x, prev.y, collected.z)
            match object.as_ref() {
                CompiledExpr::Signal(id) => {
                    // Convert to SignalComponent for vector component access
                    Expr::SignalComponent(id.to_string(), field.clone())
                }
                CompiledExpr::Prev => {
                    // Convert to PrevComponent for vector prev component access
                    Expr::PrevComponent(field.clone())
                }
                CompiledExpr::Collected => {
                    // Convert to CollectedComponent for vector inputs component access
                    Expr::CollectedComponent(field.clone())
                }
                other => {
                    panic!(
                        "Nested field access on {:?} not supported in bytecode compiler",
                        other
                    );
                }
            }
        }

        // Entity expressions are handled by the EntityExecutor at runtime, NOT the bytecode VM.
        // If these expressions reach the bytecode compiler, it indicates a bug in the
        // compilation pipeline - entity operations should be routed to EntityExecutor instead.
        CompiledExpr::SelfField(field) => {
            panic!(
                "SelfField({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                field
            );
        }
        CompiledExpr::EntityAccess {
            entity,
            instance,
            field,
        } => {
            panic!(
                "EntityAccess({}.{}.{}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity, instance, field
            );
        }
        CompiledExpr::Aggregate { op, entity, .. } => {
            panic!(
                "Aggregate({:?} over {}) reached bytecode compiler - entity expressions must use EntityExecutor",
                op, entity
            );
        }
        CompiledExpr::Other { entity, .. } => {
            panic!(
                "Other({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity
            );
        }
        CompiledExpr::Pairs { entity, .. } => {
            panic!(
                "Pairs({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity
            );
        }
        CompiledExpr::Filter { entity, .. } => {
            panic!(
                "Filter({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity
            );
        }
        CompiledExpr::First { entity, .. } => {
            panic!(
                "First({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity
            );
        }
        CompiledExpr::Nearest { entity, .. } => {
            panic!(
                "Nearest({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity
            );
        }
        CompiledExpr::Within { entity, .. } => {
            panic!(
                "Within({}) reached bytecode compiler - entity expressions must use EntityExecutor",
                entity
            );
        }

        // Impulse expressions should be handled by impulse executor
        CompiledExpr::Payload => {
            panic!(
                "Payload reached bytecode compiler - impulse expressions must use ImpulseExecutor"
            );
        }
        CompiledExpr::PayloadField(field) => {
            panic!(
                "PayloadField({}) reached bytecode compiler - impulse expressions must use ImpulseExecutor",
                field
            );
        }
        CompiledExpr::EmitSignal { target, .. } => {
            panic!(
                "EmitSignal({}) reached bytecode compiler - impulse expressions must use ImpulseExecutor",
                target
            );
        }
    }
}

/// Compiles a [`CompiledExpr`] to executable bytecode.
///
/// This is the main entry point for bytecode generation. It converts the IR
/// expression to a VM expression tree, then delegates to the VM compiler to
/// produce the final bytecode chunk.
///
/// # Returns
///
/// A [`BytecodeChunk`] that can be executed by the VM with an execution context
/// providing signal values, constants, config, and kernel functions.
///
/// # Example
///
/// ```ignore
/// let expr = CompiledExpr::Binary {
///     op: BinaryOpIr::Add,
///     left: Box::new(CompiledExpr::Prev),
///     right: Box::new(CompiledExpr::Literal(1.0)),
/// };
///
/// let chunk = compile(&expr);
/// let result = continuum_vm::execute(&chunk, &my_context);
/// ```
pub fn compile(expr: &CompiledExpr) -> BytecodeChunk {
    let vm_expr = convert_expr(expr);
    continuum_vm::compile_expr(&vm_expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompiledExpr;
    use continuum_foundation::SignalId;
    use continuum_vm::{ExecutionContext, execute};

    struct TestContext;

    impl ExecutionContext for TestContext {
        fn prev(&self) -> f64 {
            100.0
        }
        fn dt(&self) -> f64 {
            0.1
        }
        fn sim_time(&self) -> f64 {
            10.0
        }
        fn inputs(&self) -> f64 {
            5.0
        }
        fn signal(&self, name: &str) -> f64 {
            match name {
                "temp" => 25.0,
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
                _ => 0.0,
            }
        }
    }

    #[test]
    fn test_compile_literal() {
        let expr = CompiledExpr::Literal(42.0);
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_compile_binary() {
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Literal(10.0)),
            right: Box::new(CompiledExpr::Literal(32.0)),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_compile_signal() {
        let expr = CompiledExpr::Signal(SignalId::from("temp"));
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_compile_call() {
        let expr = CompiledExpr::Call {
            function: "abs".to_string(),
            args: vec![CompiledExpr::Literal(-5.0)],
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_compile_if() {
        let expr = CompiledExpr::If {
            condition: Box::new(CompiledExpr::Binary {
                op: BinaryOpIr::Gt,
                left: Box::new(CompiledExpr::Signal(SignalId::from("temp"))),
                right: Box::new(CompiledExpr::Literal(20.0)),
            }),
            then_branch: Box::new(CompiledExpr::Literal(100.0)),
            else_branch: Box::new(CompiledExpr::Literal(0.0)),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 100.0); // temp (25) > 20, so 100
    }

    #[test]
    fn test_compile_complex() {
        // prev + abs(temp - 30) * scale = 100 + abs(25-30) * 2 = 100 + 5*2 = 110
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Binary {
                op: BinaryOpIr::Mul,
                left: Box::new(CompiledExpr::Call {
                    function: "abs".to_string(),
                    args: vec![CompiledExpr::Binary {
                        op: BinaryOpIr::Sub,
                        left: Box::new(CompiledExpr::Signal(SignalId::from("temp"))),
                        right: Box::new(CompiledExpr::Literal(30.0)),
                    }],
                }),
                right: Box::new(CompiledExpr::Config("scale".to_string())),
            }),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 110.0);
    }

    #[test]
    fn test_compile_let() {
        // let a = 10.0
        // let b = 20.0
        // a + b = 30.0
        let expr = CompiledExpr::Let {
            name: "a".to_string(),
            value: Box::new(CompiledExpr::Literal(10.0)),
            body: Box::new(CompiledExpr::Let {
                name: "b".to_string(),
                value: Box::new(CompiledExpr::Literal(20.0)),
                body: Box::new(CompiledExpr::Binary {
                    op: BinaryOpIr::Add,
                    left: Box::new(CompiledExpr::Local("a".to_string())),
                    right: Box::new(CompiledExpr::Local("b".to_string())),
                }),
            }),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 30.0);
    }
}
