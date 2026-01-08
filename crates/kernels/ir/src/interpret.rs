//! IR Interpretation
//!
//! Evaluates compiled expressions and builds runtime closures.

use std::collections::HashMap;

use indexmap::IndexMap;

use continuum_foundation::{EraId, SignalId, StratumId};
use continuum_runtime::executor::{AssertionFn, AssertionSeverity, EraConfig, ResolverFn};
use continuum_runtime::operators;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::{Dt, StratumState, Value};

use crate::{
    AssertionSeverity as IrAssertionSeverity, BinaryOpIr, CompiledExpr, CompiledWorld,
    StratumStateIr, UnaryOpIr, ValueType,
};

/// Build era configurations from compiled world
pub fn build_era_configs(world: &CompiledWorld) -> HashMap<EraId, EraConfig> {
    let mut configs = HashMap::new();

    for (era_id, era) in &world.eras {
        let mut strata = HashMap::new();
        for (stratum_id, state) in &era.strata_states {
            let runtime_state = match state {
                StratumStateIr::Active => StratumState::Active,
                StratumStateIr::ActiveWithStride(s) => StratumState::ActiveWithStride(*s),
                StratumStateIr::Gated => StratumState::Gated,
            };
            strata.insert(StratumId(stratum_id.0.clone()), runtime_state);
        }

        configs.insert(
            EraId(era_id.0.clone()),
            EraConfig {
                dt: Dt(era.dt_seconds),
                strata,
                transition: None, // TODO: implement transition conditions
            },
        );
    }

    configs
}

/// Get initial value for a signal from config
pub fn get_initial_value(world: &CompiledWorld, signal_id: &SignalId) -> f64 {
    let signal_name = &signal_id.0;
    let parts: Vec<&str> = signal_name.split('.').collect();

    // Try various config key patterns
    if parts.len() >= 2 {
        let last = parts.last().unwrap();

        // Try config.<domain>.initial_<signal>
        for (key, value) in &world.config {
            if key.ends_with(&format!("initial_{}", last)) {
                return *value;
            }
        }
    }

    // Default to 0
    0.0
}

/// Get initial value for a signal based on its type
pub fn get_initial_signal_value(world: &CompiledWorld, signal_id: &SignalId) -> Value {
    let initial_value = get_initial_value(world, signal_id);

    if let Some(signal) = world.signals.get(signal_id) {
        match signal.value_type {
            ValueType::Scalar { .. } => Value::Scalar(initial_value),
            ValueType::Vec2 => Value::Vec2([initial_value; 2]),
            ValueType::Vec3 => Value::Vec3([initial_value; 3]),
            ValueType::Vec4 => Value::Vec4([initial_value; 4]),
        }
    } else {
        Value::Scalar(initial_value)
    }
}

/// Build a resolver function from a compiled expression
pub fn build_resolver(expr: &CompiledExpr, world: &CompiledWorld, uses_dt_raw: bool) -> ResolverFn {
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let dt = if uses_dt_raw {
            ctx.dt.seconds()
        } else {
            ctx.dt.seconds()
        };

        let result = eval_expr(&expr, ctx.prev, ctx.inputs, dt, &constants, &config, ctx.signals);
        Value::Scalar(result)
    })
}

/// Build an assertion function from a compiled expression
pub fn build_assertion(expr: &CompiledExpr, world: &CompiledWorld) -> AssertionFn {
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let dt = ctx.dt.seconds();
        // Assertions evaluate against current (post-resolve) value
        let result = eval_assertion_expr(
            &expr,
            ctx.current,
            ctx.prev,
            dt,
            &constants,
            &config,
            ctx.signals,
        );
        result != 0.0
    })
}

/// Convert IR assertion severity to runtime severity
pub fn convert_assertion_severity(severity: IrAssertionSeverity) -> AssertionSeverity {
    match severity {
        IrAssertionSeverity::Warn => AssertionSeverity::Warn,
        IrAssertionSeverity::Error => AssertionSeverity::Error,
        IrAssertionSeverity::Fatal => AssertionSeverity::Fatal,
    }
}

/// Evaluate a compiled expression in resolver context
fn eval_expr(
    expr: &CompiledExpr,
    prev: &Value,
    inputs: f64,
    dt: f64,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    signals: &SignalStorage,
) -> f64 {
    match expr {
        CompiledExpr::Literal(v) => *v,
        CompiledExpr::Prev => prev.as_scalar().unwrap_or(0.0),
        CompiledExpr::DtRaw => dt,
        CompiledExpr::SumInputs => inputs,
        CompiledExpr::Signal(id) => {
            let runtime_id = SignalId(id.0.clone());
            signals
                .get(&runtime_id)
                .and_then(|v| v.as_scalar())
                .unwrap_or(0.0)
        }
        CompiledExpr::Const(name) => constants.get(name).copied().unwrap_or(0.0),
        CompiledExpr::Config(name) => config.get(name).copied().unwrap_or(0.0),
        CompiledExpr::Binary { op, left, right } => {
            let l = eval_expr(left, prev, inputs, dt, constants, config, signals);
            let r = eval_expr(right, prev, inputs, dt, constants, config, signals);
            eval_binary_op(*op, l, r)
        }
        CompiledExpr::Unary { op, operand } => {
            let v = eval_expr(operand, prev, inputs, dt, constants, config, signals);
            eval_unary_op(*op, v)
        }
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<f64> = args
                .iter()
                .map(|a| eval_expr(a, prev, inputs, dt, constants, config, signals))
                .collect();
            eval_function_call(function, &arg_values, dt)
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = eval_expr(condition, prev, inputs, dt, constants, config, signals);
            if cond != 0.0 {
                eval_expr(then_branch, prev, inputs, dt, constants, config, signals)
            } else {
                eval_expr(else_branch, prev, inputs, dt, constants, config, signals)
            }
        }
        CompiledExpr::Let { value, body, .. } => {
            // For now, just evaluate body (let bindings need local environment)
            let _val = eval_expr(value, prev, inputs, dt, constants, config, signals);
            eval_expr(body, prev, inputs, dt, constants, config, signals)
        }
        CompiledExpr::FieldAccess { .. } => {
            // Field access on expressions - not yet supported
            0.0
        }
    }
}

/// Evaluate a compiled expression in assertion context
fn eval_assertion_expr(
    expr: &CompiledExpr,
    current: &Value,
    prev: &Value,
    dt: f64,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    signals: &SignalStorage,
) -> f64 {
    match expr {
        CompiledExpr::Literal(v) => *v,
        // In assertions, 'prev' refers to the current (post-resolve) value being asserted
        CompiledExpr::Prev => current.as_scalar().unwrap_or(0.0),
        CompiledExpr::DtRaw => dt,
        CompiledExpr::SumInputs => 0.0, // Not used in assertions
        CompiledExpr::Signal(id) => {
            let runtime_id = SignalId(id.0.clone());
            signals
                .get(&runtime_id)
                .and_then(|v| v.as_scalar())
                .unwrap_or(0.0)
        }
        CompiledExpr::Const(name) => constants.get(name).copied().unwrap_or(0.0),
        CompiledExpr::Config(name) => config.get(name).copied().unwrap_or(0.0),
        CompiledExpr::Binary { op, left, right } => {
            let l = eval_assertion_expr(left, current, prev, dt, constants, config, signals);
            let r = eval_assertion_expr(right, current, prev, dt, constants, config, signals);
            eval_binary_op(*op, l, r)
        }
        CompiledExpr::Unary { op, operand } => {
            let v = eval_assertion_expr(operand, current, prev, dt, constants, config, signals);
            eval_unary_op(*op, v)
        }
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<f64> = args
                .iter()
                .map(|a| eval_assertion_expr(a, current, prev, dt, constants, config, signals))
                .collect();
            // Assertions don't use dt-dependent functions
            eval_assertion_function_call(function, &arg_values)
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond =
                eval_assertion_expr(condition, current, prev, dt, constants, config, signals);
            if cond != 0.0 {
                eval_assertion_expr(then_branch, current, prev, dt, constants, config, signals)
            } else {
                eval_assertion_expr(else_branch, current, prev, dt, constants, config, signals)
            }
        }
        CompiledExpr::Let { value, body, .. } => {
            let _val =
                eval_assertion_expr(value, current, prev, dt, constants, config, signals);
            eval_assertion_expr(body, current, prev, dt, constants, config, signals)
        }
        CompiledExpr::FieldAccess { .. } => 0.0,
    }
}

/// Evaluate a binary operation
fn eval_binary_op(op: BinaryOpIr, l: f64, r: f64) -> f64 {
    match op {
        BinaryOpIr::Add => l + r,
        BinaryOpIr::Sub => l - r,
        BinaryOpIr::Mul => l * r,
        BinaryOpIr::Div => l / r,
        BinaryOpIr::Pow => l.powf(r),
        BinaryOpIr::Eq => {
            if (l - r).abs() < f64::EPSILON {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Ne => {
            if (l - r).abs() >= f64::EPSILON {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Lt => {
            if l < r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Le => {
            if l <= r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Gt => {
            if l > r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Ge => {
            if l >= r {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::And => {
            if l != 0.0 && r != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        BinaryOpIr::Or => {
            if l != 0.0 || r != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Evaluate a unary operation
fn eval_unary_op(op: UnaryOpIr, v: f64) -> f64 {
    match op {
        UnaryOpIr::Neg => -v,
        UnaryOpIr::Not => {
            if v == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Evaluate a function call in resolver context (includes dt-dependent functions)
fn eval_function_call(function: &str, args: &[f64], dt: f64) -> f64 {
    match function {
        "decay" => {
            if args.len() >= 2 {
                operators::decay(args[0], args[1], dt)
            } else {
                0.0
            }
        }
        "relax" => {
            if args.len() >= 3 {
                operators::relax(args[0], args[1], args[2], dt)
            } else {
                0.0
            }
        }
        "integrate" => {
            if args.len() >= 2 {
                operators::integrate(args[0], args[1], dt)
            } else {
                0.0
            }
        }
        "clamp" => {
            if args.len() >= 3 {
                args[0].clamp(args[1], args[2])
            } else {
                0.0
            }
        }
        "min" => args.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => args.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "abs" => args.first().map(|v| v.abs()).unwrap_or(0.0),
        "sqrt" => args.first().map(|v| v.sqrt()).unwrap_or(0.0),
        "sin" => args.first().map(|v| v.sin()).unwrap_or(0.0),
        "cos" => args.first().map(|v| v.cos()).unwrap_or(0.0),
        "exp" => args.first().map(|v| v.exp()).unwrap_or(0.0),
        "ln" => args.first().map(|v| v.ln()).unwrap_or(0.0),
        "log10" => args.first().map(|v| v.log10()).unwrap_or(0.0),
        "pow" => {
            if args.len() >= 2 {
                args[0].powf(args[1])
            } else {
                0.0
            }
        }
        _ => {
            tracing::warn!("unknown function '{}'", function);
            0.0
        }
    }
}

/// Evaluate a function call in assertion context (pure math only)
fn eval_assertion_function_call(function: &str, args: &[f64]) -> f64 {
    match function {
        "abs" => args.first().map(|v| v.abs()).unwrap_or(0.0),
        "min" => args.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => args.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "sqrt" => args.first().map(|v| v.sqrt()).unwrap_or(0.0),
        "sin" => args.first().map(|v| v.sin()).unwrap_or(0.0),
        "cos" => args.first().map(|v| v.cos()).unwrap_or(0.0),
        "exp" => args.first().map(|v| v.exp()).unwrap_or(0.0),
        "ln" => args.first().map(|v| v.ln()).unwrap_or(0.0),
        "log10" => args.first().map(|v| v.log10()).unwrap_or(0.0),
        "pow" => {
            if args.len() >= 2 {
                args[0].powf(args[1])
            } else {
                0.0
            }
        }
        "clamp" => {
            if args.len() >= 3 {
                args[0].clamp(args[1], args[2])
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_binary_op() {
        assert_eq!(eval_binary_op(BinaryOpIr::Add, 2.0, 3.0), 5.0);
        assert_eq!(eval_binary_op(BinaryOpIr::Sub, 5.0, 3.0), 2.0);
        assert_eq!(eval_binary_op(BinaryOpIr::Mul, 2.0, 3.0), 6.0);
        assert_eq!(eval_binary_op(BinaryOpIr::Div, 6.0, 2.0), 3.0);
        assert_eq!(eval_binary_op(BinaryOpIr::Lt, 2.0, 3.0), 1.0);
        assert_eq!(eval_binary_op(BinaryOpIr::Lt, 3.0, 2.0), 0.0);
        assert_eq!(eval_binary_op(BinaryOpIr::Ge, 3.0, 3.0), 1.0);
    }

    #[test]
    fn test_eval_unary_op() {
        assert_eq!(eval_unary_op(UnaryOpIr::Neg, 5.0), -5.0);
        assert_eq!(eval_unary_op(UnaryOpIr::Not, 0.0), 1.0);
        assert_eq!(eval_unary_op(UnaryOpIr::Not, 1.0), 0.0);
    }

    #[test]
    fn test_eval_function_call() {
        assert_eq!(eval_function_call("abs", &[-5.0], 1.0), 5.0);
        assert_eq!(eval_function_call("min", &[3.0, 1.0, 2.0], 1.0), 1.0);
        assert_eq!(eval_function_call("max", &[3.0, 1.0, 2.0], 1.0), 3.0);
        assert_eq!(eval_function_call("clamp", &[5.0, 0.0, 3.0], 1.0), 3.0);
    }
}
