//! Member signal interpreter.
//!
//! Provides a baseline CPU interpreter for member signals resolved per instance.
//! Used for prototyping and as a fallback when JIT/vectorized execution
//! is not available.

use std::collections::HashMap;

use continuum_foundation::SignalId;
pub use continuum_runtime::executor::{
    ScalarResolverFn as MemberResolverFn, Vec3ResolverFn as Vec3MemberResolverFn,
};
use continuum_runtime::soa_storage::MemberSignalBuffer;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::types::Value;
use indexmap::IndexMap;

use crate::CompiledExpr;

/// Intermediate value during interpretation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpValue {
    Scalar(f64),
    Vec3([f64; 3]),
    Bool(bool),
}

impl InterpValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            InterpValue::Scalar(v) => *v,
            InterpValue::Vec3(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
            InterpValue::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn as_vec3(&self) -> [f64; 3] {
        match self {
            InterpValue::Scalar(v) => [*v, 0.0, 0.0],
            InterpValue::Vec3(v) => *v,
            InterpValue::Bool(b) => [if *b { 1.0 } else { 0.0 }, 0.0, 0.0],
        }
    }

    pub fn binary_op(&self, other: InterpValue, op: crate::BinaryOpIr) -> InterpValue {
        use crate::BinaryOpIr::*;
        match op {
            Add => match (self, other) {
                (InterpValue::Scalar(a), InterpValue::Scalar(b)) => InterpValue::Scalar(a + b),
                (InterpValue::Vec3(a), InterpValue::Vec3(b)) => {
                    InterpValue::Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                }
                _ => InterpValue::Scalar(self.as_f64() + other.as_f64()),
            },
            Sub => match (self, other) {
                (InterpValue::Scalar(a), InterpValue::Scalar(b)) => InterpValue::Scalar(a - b),
                (InterpValue::Vec3(a), InterpValue::Vec3(b)) => {
                    InterpValue::Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
                }
                _ => InterpValue::Scalar(self.as_f64() - other.as_f64()),
            },
            Mul => match (self, other) {
                (InterpValue::Scalar(a), InterpValue::Scalar(b)) => InterpValue::Scalar(a * b),
                (InterpValue::Scalar(s), InterpValue::Vec3(v)) => {
                    InterpValue::Vec3([v[0] * s, v[1] * s, v[2] * s])
                }
                (InterpValue::Vec3(v), InterpValue::Scalar(s)) => {
                    InterpValue::Vec3([v[0] * s, v[1] * s, v[2] * s])
                }
                _ => InterpValue::Scalar(self.as_f64() * other.as_f64()),
            },
            Div => match (self, other) {
                (InterpValue::Scalar(a), InterpValue::Scalar(b)) => InterpValue::Scalar(a / b),
                (InterpValue::Vec3(v), InterpValue::Scalar(s)) => {
                    InterpValue::Vec3([v[0] / s, v[1] / s, v[2] / s])
                }
                _ => InterpValue::Scalar(self.as_f64() / other.as_f64()),
            },
            Eq => InterpValue::Bool(self.as_f64() == other.as_f64()),
            Ne => InterpValue::Bool(self.as_f64() != other.as_f64()),
            Lt => InterpValue::Bool(self.as_f64() < other.as_f64()),
            Le => InterpValue::Bool(self.as_f64() <= other.as_f64()),
            Gt => InterpValue::Bool(self.as_f64() > other.as_f64()),
            Ge => InterpValue::Bool(self.as_f64() >= other.as_f64()),
            And => InterpValue::Bool(self.as_f64() != 0.0 && other.as_f64() != 0.0),
            Or => InterpValue::Bool(self.as_f64() != 0.0 || other.as_f64() != 0.0),
            _ => InterpValue::Scalar(0.0),
        }
    }

    pub fn unary_op(&self, op: crate::UnaryOpIr) -> InterpValue {
        use crate::UnaryOpIr::*;
        match op {
            Neg => match self {
                InterpValue::Scalar(v) => InterpValue::Scalar(-v),
                InterpValue::Vec3(v) => InterpValue::Vec3([-v[0], -v[1], -v[2]]),
                InterpValue::Bool(b) => InterpValue::Scalar(if *b { -1.0 } else { 0.0 }),
            },
            Not => InterpValue::Bool(self.as_f64() == 0.0),
        }
    }
}

/// Context for member expression interpretation.
pub struct MemberInterpContext<'a> {
    pub prev: InterpValue,
    pub index: usize,
    pub dt: f64,
    pub sim_time: f64,
    pub signals: &'a SignalStorage,
    pub members: &'a MemberSignalBuffer,
    pub constants: &'a IndexMap<String, (f64, Option<crate::units::Unit>)>,
    pub config: &'a IndexMap<String, (f64, Option<crate::units::Unit>)>,
    pub locals: HashMap<String, InterpValue>,
    pub entity_prefix: String,
    pub read_current: bool,
}

impl MemberInterpContext<'_> {
    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).map(|(v, _)| *v).unwrap_or(0.0)
    }
    fn config(&self, name: &str) -> f64 {
        self.config.get(name).map(|(v, _)| *v).unwrap_or(0.0)
    }
    fn signal(&self, name: &str) -> InterpValue {
        let id = SignalId::from(name);
        match self.signals.get(&id) {
            Some(Value::Scalar(v)) => InterpValue::Scalar(*v),
            Some(Value::Vec3(v)) => InterpValue::Vec3(*v),
            _ => InterpValue::Scalar(0.0),
        }
    }
    fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
        continuum_kernel_registry::eval(name, args, self.dt).unwrap_or(0.0)
    }
}

pub fn interpret_expr(expr: &CompiledExpr, ctx: &mut MemberInterpContext) -> InterpValue {
    match expr {
        CompiledExpr::Literal(v, _) => InterpValue::Scalar(*v),
        CompiledExpr::Prev => ctx.prev,
        CompiledExpr::DtRaw => InterpValue::Scalar(ctx.dt),
        CompiledExpr::SimTime => InterpValue::Scalar(ctx.sim_time),
        CompiledExpr::Collected => InterpValue::Scalar(0.0),
        CompiledExpr::Signal(id) => ctx.signal(&id.to_string()),
        CompiledExpr::Const(name, _) => InterpValue::Scalar(ctx.constant(name)),
        CompiledExpr::Config(name, _) => InterpValue::Scalar(ctx.config(name)),
        CompiledExpr::Binary { op, left, right } => {
            let l = interpret_expr(left, ctx);
            let r = interpret_expr(right, ctx);
            l.binary_op(r, *op)
        }
        CompiledExpr::Unary { op, operand } => {
            let v = interpret_expr(operand, ctx);
            v.unary_op(*op)
        }
        CompiledExpr::Call { function, args } => {
            let arg_vals: Vec<_> = args.iter().map(|a| interpret_expr(a, ctx)).collect();
            eval_function(function, &arg_vals, ctx)
        }
        CompiledExpr::KernelCall { function, args } => {
            let arg_vals: Vec<f64> = args
                .iter()
                .map(|a| interpret_expr(a, ctx).as_f64())
                .collect();
            let name = format!("kernel.{}", function);
            InterpValue::Scalar(ctx.call_kernel(&name, &arg_vals))
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if interpret_expr(condition, ctx).as_f64() != 0.0 {
                interpret_expr(then_branch, ctx)
            } else {
                interpret_expr(else_branch, ctx)
            }
        }
        CompiledExpr::Let { name, value, body } => {
            let val = interpret_expr(value, ctx);
            ctx.locals.insert(name.clone(), val);
            let result = interpret_expr(body, ctx);
            ctx.locals.remove(name);
            result
        }
        CompiledExpr::Local(name) => *ctx.locals.get(name).unwrap_or(&InterpValue::Scalar(0.0)),
        CompiledExpr::SelfField(name) => {
            let full_name = format!("{}.{}", ctx.entity_prefix, name);
            let val = if ctx.read_current {
                ctx.members.get_current(&full_name, ctx.index)
            } else {
                ctx.members.get_previous(&full_name, ctx.index)
            };
            match val {
                Some(Value::Scalar(v)) => InterpValue::Scalar(v),
                Some(Value::Vec3(v)) => InterpValue::Vec3(v),
                _ => InterpValue::Scalar(0.0),
            }
        }
        _ => InterpValue::Scalar(0.0),
    }
}

fn eval_function(name: &str, args: &[InterpValue], _ctx: &MemberInterpContext) -> InterpValue {
    match name {
        "vec2" => InterpValue::Vec3([args[0].as_f64(), args[1].as_f64(), 0.0]),
        "vec3" => InterpValue::Vec3([args[0].as_f64(), args[1].as_f64(), args[2].as_f64()]),
        _ => InterpValue::Scalar(0.0),
    }
}

pub fn build_member_resolver(
    expr: &CompiledExpr,
    constants: &IndexMap<String, (f64, Option<crate::units::Unit>)>,
    config: &IndexMap<String, (f64, Option<crate::units::Unit>)>,
    entity_prefix: &str,
) -> MemberResolverFn {
    let expr = expr.clone();
    let constants = constants.clone();
    let config = config.clone();
    let entity_prefix = entity_prefix.to_string();

    Box::new(move |ctx| {
        let mut interp_ctx = MemberInterpContext {
            prev: InterpValue::Scalar(ctx.prev),
            index: ctx.index.0,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            signals: ctx.signals,
            members: ctx.members,
            constants: &constants,
            config: &config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.clone(),
            read_current: false,
        };

        interpret_expr(&expr, &mut interp_ctx).as_f64()
    })
}

pub fn build_vec3_member_resolver(
    expr: &CompiledExpr,
    constants: &IndexMap<String, (f64, Option<crate::units::Unit>)>,
    config: &IndexMap<String, (f64, Option<crate::units::Unit>)>,
    entity_prefix: &str,
) -> Vec3MemberResolverFn {
    let expr = expr.clone();
    let constants = constants.clone();
    let config = config.clone();
    let entity_prefix = entity_prefix.to_string();

    Box::new(move |ctx| {
        let mut interp_ctx = MemberInterpContext {
            prev: InterpValue::Vec3(ctx.prev),
            index: ctx.index.0,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            signals: ctx.signals,
            members: ctx.members,
            constants: &constants,
            config: &config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.clone(),
            read_current: false,
        };

        interpret_expr(&expr, &mut interp_ctx).as_vec3()
    })
}
