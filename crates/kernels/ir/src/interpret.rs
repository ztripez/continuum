//! IR Interpretation
//!
//! Evaluates compiled expressions and builds runtime closures.

use std::collections::HashMap;

use indexmap::IndexMap;

use continuum_foundation::{EraId, FieldId, SignalId, StratumId};
use continuum_runtime::executor::{
    AssertionFn, AssertionSeverity, EraConfig, FractureFn, MeasureFn, ResolverFn, TransitionFn,
};
use continuum_runtime::operators;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::{Dt, StratumState, Value};

use crate::{
    AssertionSeverity as IrAssertionSeverity, BinaryOpIr, CompiledEra, CompiledExpr,
    CompiledFracture, CompiledWorld, StratumStateIr, UnaryOpIr, ValueType,
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

        // Build transition function if there are transitions
        let transition = build_transition_fn(era, &world.constants, &world.config);

        configs.insert(
            EraId(era_id.0.clone()),
            EraConfig {
                dt: Dt(era.dt_seconds),
                strata,
                transition,
            },
        );
    }

    configs
}

/// Build a transition function for an era
fn build_transition_fn(
    era: &CompiledEra,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
) -> Option<TransitionFn> {
    if era.transitions.is_empty() {
        return None;
    }

    // Clone the data we need for the closure
    let transitions: Vec<_> = era
        .transitions
        .iter()
        .map(|t| (t.target_era.clone(), t.condition.clone()))
        .collect();
    let constants = constants.clone();
    let config = config.clone();

    Some(Box::new(move |signals: &SignalStorage| {
        // Evaluate each transition condition in order
        // First matching condition wins
        for (target_era, condition) in &transitions {
            let result = eval_transition_expr(condition, &constants, &config, signals);
            if result != 0.0 {
                return Some(EraId(target_era.0.clone()));
            }
        }
        None
    }))
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

/// Build a measure function for a field
///
/// Field measure expressions evaluate against current signal values and emit to the field buffer.
pub fn build_field_measure(
    field_id: &FieldId,
    expr: &CompiledExpr,
    world: &CompiledWorld,
) -> MeasureFn {
    let field_id = FieldId(field_id.0.clone());
    let expr = expr.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        let result = eval_measure_expr(&expr, ctx.dt.seconds(), &constants, &config, ctx.signals);
        ctx.fields.emit_scalar(field_id.clone(), result);
    })
}

/// Build a fracture detection function
///
/// Fractures check conditions and emit to signals when triggered.
/// Returns `Some(emits)` if all conditions pass, `None` otherwise.
pub fn build_fracture(fracture: &CompiledFracture, world: &CompiledWorld) -> FractureFn {
    let conditions = fracture.conditions.clone();
    let emits = fracture.emits.clone();
    let constants = world.constants.clone();
    let config = world.config.clone();

    Box::new(move |ctx| {
        // Check all conditions - all must be non-zero
        for condition in &conditions {
            let result =
                eval_fracture_expr(condition, ctx.dt.seconds(), &constants, &config, ctx.signals);
            if result == 0.0 {
                return None;
            }
        }

        // All conditions passed - evaluate emit expressions
        let outputs: Vec<(continuum_runtime::SignalId, f64)> = emits
            .iter()
            .map(|emit| {
                let value = eval_fracture_expr(
                    &emit.value,
                    ctx.dt.seconds(),
                    &constants,
                    &config,
                    ctx.signals,
                );
                (continuum_runtime::SignalId(emit.target.0.clone()), value)
            })
            .collect();

        Some(outputs)
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

/// Evaluate a compiled expression in transition context (signals only, no prev/dt)
fn eval_transition_expr(
    expr: &CompiledExpr,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    signals: &SignalStorage,
) -> f64 {
    match expr {
        CompiledExpr::Literal(v) => *v,
        // Prev and DtRaw don't make sense in transition conditions
        CompiledExpr::Prev => 0.0,
        CompiledExpr::DtRaw => 0.0,
        CompiledExpr::SumInputs => 0.0,
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
            let l = eval_transition_expr(left, constants, config, signals);
            let r = eval_transition_expr(right, constants, config, signals);
            eval_binary_op(*op, l, r)
        }
        CompiledExpr::Unary { op, operand } => {
            let v = eval_transition_expr(operand, constants, config, signals);
            eval_unary_op(*op, v)
        }
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<f64> = args
                .iter()
                .map(|a| eval_transition_expr(a, constants, config, signals))
                .collect();
            // Transitions use pure math functions only
            eval_assertion_function_call(function, &arg_values)
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = eval_transition_expr(condition, constants, config, signals);
            if cond != 0.0 {
                eval_transition_expr(then_branch, constants, config, signals)
            } else {
                eval_transition_expr(else_branch, constants, config, signals)
            }
        }
        CompiledExpr::Let { value, body, .. } => {
            let _val = eval_transition_expr(value, constants, config, signals);
            eval_transition_expr(body, constants, config, signals)
        }
        CompiledExpr::FieldAccess { .. } => 0.0,
    }
}

/// Evaluate a compiled expression in measure context (current signals, with dt)
fn eval_measure_expr(
    expr: &CompiledExpr,
    dt: f64,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    signals: &SignalStorage,
) -> f64 {
    match expr {
        CompiledExpr::Literal(v) => *v,
        // Prev doesn't make sense in measure context (we have current values)
        CompiledExpr::Prev => 0.0,
        CompiledExpr::DtRaw => dt,
        CompiledExpr::SumInputs => 0.0,
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
            let l = eval_measure_expr(left, dt, constants, config, signals);
            let r = eval_measure_expr(right, dt, constants, config, signals);
            eval_binary_op(*op, l, r)
        }
        CompiledExpr::Unary { op, operand } => {
            let v = eval_measure_expr(operand, dt, constants, config, signals);
            eval_unary_op(*op, v)
        }
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<f64> = args
                .iter()
                .map(|a| eval_measure_expr(a, dt, constants, config, signals))
                .collect();
            eval_assertion_function_call(function, &arg_values)
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = eval_measure_expr(condition, dt, constants, config, signals);
            if cond != 0.0 {
                eval_measure_expr(then_branch, dt, constants, config, signals)
            } else {
                eval_measure_expr(else_branch, dt, constants, config, signals)
            }
        }
        CompiledExpr::Let { value, body, .. } => {
            let _val = eval_measure_expr(value, dt, constants, config, signals);
            eval_measure_expr(body, dt, constants, config, signals)
        }
        CompiledExpr::FieldAccess { .. } => 0.0,
    }
}

/// Evaluate a compiled expression in fracture context (current signals, with dt)
fn eval_fracture_expr(
    expr: &CompiledExpr,
    dt: f64,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    signals: &SignalStorage,
) -> f64 {
    // Fracture evaluation is identical to measure evaluation - access to current signals and dt
    eval_measure_expr(expr, dt, constants, config, signals)
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

    #[test]
    fn test_eval_transition_expr() {
        use continuum_runtime::storage::SignalStorage;

        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut signals = SignalStorage::default();

        // Initialize a signal
        signals.init(SignalId::from("temp"), Value::Scalar(100.0));

        // Test literal
        let expr = CompiledExpr::Literal(1.0);
        assert_eq!(eval_transition_expr(&expr, &constants, &config, &signals), 1.0);

        // Test signal reference
        let expr = CompiledExpr::Signal(continuum_foundation::SignalId::from("temp"));
        assert_eq!(eval_transition_expr(&expr, &constants, &config, &signals), 100.0);

        // Test comparison: temp < 200
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Lt,
            left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from("temp"))),
            right: Box::new(CompiledExpr::Literal(200.0)),
        };
        assert_eq!(eval_transition_expr(&expr, &constants, &config, &signals), 1.0); // true

        // Test comparison: temp < 50
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Lt,
            left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from("temp"))),
            right: Box::new(CompiledExpr::Literal(50.0)),
        };
        assert_eq!(eval_transition_expr(&expr, &constants, &config, &signals), 0.0); // false
    }

    #[test]
    fn test_build_transition_fn() {
        use crate::{CompiledEra, CompiledTransition};
        use continuum_runtime::storage::SignalStorage;

        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut signals = SignalStorage::default();

        // Initialize signal at 100
        signals.init(SignalId::from("temp"), Value::Scalar(100.0));

        // Create an era with a transition when temp < 50
        let era = CompiledEra {
            id: continuum_foundation::EraId::from("test"),
            is_initial: true,
            is_terminal: false,
            title: None,
            dt_seconds: 1.0,
            strata_states: IndexMap::new(),
            transitions: vec![CompiledTransition {
                target_era: continuum_foundation::EraId::from("next_era"),
                condition: CompiledExpr::Binary {
                    op: BinaryOpIr::Lt,
                    left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from("temp"))),
                    right: Box::new(CompiledExpr::Literal(50.0)),
                },
            }],
        };

        let transition_fn = build_transition_fn(&era, &constants, &config).unwrap();

        // Signal at 100, should not transition (100 < 50 is false)
        assert!(transition_fn(&signals).is_none());

        // Update signal to 30
        signals.set_current(SignalId::from("temp"), Value::Scalar(30.0));

        // Signal at 30, should transition (30 < 50 is true)
        let result = transition_fn(&signals);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "next_era");
    }

    #[test]
    fn test_build_fracture() {
        use crate::{CompiledEmit, CompiledFracture};
        use continuum_runtime::executor::FractureContext;
        use continuum_runtime::storage::SignalStorage;
        use continuum_runtime::types::Dt;

        let world = CompiledWorld {
            constants: IndexMap::new(),
            config: IndexMap::new(),
            strata: IndexMap::new(),
            eras: IndexMap::new(),
            signals: IndexMap::new(),
            fields: IndexMap::new(),
            operators: IndexMap::new(),
            impulses: IndexMap::new(),
            fractures: IndexMap::new(),
        };

        // Create a fracture that triggers when temp > 100 and emits to energy
        let fracture = CompiledFracture {
            id: continuum_foundation::FractureId::from("test_fracture"),
            reads: vec![continuum_foundation::SignalId::from("temp")],
            conditions: vec![CompiledExpr::Binary {
                op: BinaryOpIr::Gt,
                left: Box::new(CompiledExpr::Signal(continuum_foundation::SignalId::from(
                    "temp",
                ))),
                right: Box::new(CompiledExpr::Literal(100.0)),
            }],
            emits: vec![CompiledEmit {
                target: continuum_foundation::SignalId::from("energy"),
                value: CompiledExpr::Literal(50.0),
            }],
        };

        let fracture_fn = build_fracture(&fracture, &world);

        let mut signals = SignalStorage::default();
        signals.init(SignalId::from("temp"), Value::Scalar(50.0));

        let ctx = FractureContext {
            signals: &signals,
            dt: Dt(1.0),
        };

        // Temp is 50, condition (temp > 100) is false, should not trigger
        assert!(fracture_fn(&ctx).is_none());

        // Update temp to 150
        signals.set_current(SignalId::from("temp"), Value::Scalar(150.0));
        let ctx = FractureContext {
            signals: &signals,
            dt: Dt(1.0),
        };

        // Temp is 150, condition (temp > 100) is true, should trigger
        let result = fracture_fn(&ctx);
        assert!(result.is_some());
        let emits = result.unwrap();
        assert_eq!(emits.len(), 1);
        assert_eq!(emits[0].0, SignalId::from("energy"));
        assert_eq!(emits[0].1, 50.0);
    }
}
