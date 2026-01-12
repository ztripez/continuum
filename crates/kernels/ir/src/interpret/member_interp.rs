//! Member signal expression interpreter.
//!
//! This module provides an interpreter for evaluating member signal expressions.
//! Unlike global signals that use bytecode compilation, member signals require
//! direct interpretation because they access per-instance data via `self.*`
//! (SelfField) expressions which cannot be compiled to bytecode.

use indexmap::IndexMap;
use std::collections::HashMap;

use continuum_foundation::SignalId;
use continuum_runtime::executor::member_executor::{ScalarResolveContext, Vec3ResolveContext};
use continuum_runtime::soa_storage::MemberSignalBuffer;
use continuum_runtime::storage::SignalStorage;
use continuum_vm::ExecutionContext;

use crate::{AggregateOpIr, BinaryOpIr, CompiledExpr, DtRobustOperator, UnaryOpIr};

// ============================================================================
// Interpreter Value Type
// ============================================================================

/// Value type for the interpreter, supporting multiple numeric types.
#[derive(Debug, Clone, Copy)]
pub enum InterpValue {
    /// Single f64 scalar value
    Scalar(f64),
    /// 3D vector [x, y, z]
    Vec3([f64; 3]),
}

impl InterpValue {
    #[inline]
    pub fn as_scalar(self) -> f64 {
        match self {
            InterpValue::Scalar(v) => v,
            InterpValue::Vec3(_) => panic!("Expected scalar, got Vec3"),
        }
    }

    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            InterpValue::Scalar(v) => v,
            InterpValue::Vec3(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
        }
    }

    #[inline]
    pub fn as_vec3(self) -> [f64; 3] {
        match self {
            InterpValue::Vec3(v) => v,
            InterpValue::Scalar(v) => [v, v, v],
        }
    }

    pub fn component(&self, name: &str) -> f64 {
        match self {
            InterpValue::Scalar(v) => *v,
            InterpValue::Vec3(v) => match name {
                "x" => v[0],
                "y" => v[1],
                "z" => v[2],
                _ => panic!("Unknown Vec3 component: {}", name),
            },
        }
    }

    pub fn binary_op(self, other: Self, op: BinaryOpIr) -> Self {
        match (self, other) {
            (InterpValue::Scalar(l), InterpValue::Scalar(r)) => {
                InterpValue::Scalar(eval_binary_scalar(op, l, r))
            }
            (InterpValue::Vec3(l), InterpValue::Vec3(r)) => InterpValue::Vec3([
                eval_binary_scalar(op, l[0], r[0]),
                eval_binary_scalar(op, l[1], r[1]),
                eval_binary_scalar(op, l[2], r[2]),
            ]),
            (InterpValue::Scalar(s), InterpValue::Vec3(v)) => InterpValue::Vec3([
                eval_binary_scalar(op, s, v[0]),
                eval_binary_scalar(op, s, v[1]),
                eval_binary_scalar(op, s, v[2]),
            ]),
            (InterpValue::Vec3(v), InterpValue::Scalar(s)) => InterpValue::Vec3([
                eval_binary_scalar(op, v[0], s),
                eval_binary_scalar(op, v[1], s),
                eval_binary_scalar(op, v[2], s),
            ]),
        }
    }

    pub fn unary_op(self, op: UnaryOpIr) -> Self {
        match self {
            InterpValue::Scalar(v) => InterpValue::Scalar(eval_unary_scalar(op, v)),
            InterpValue::Vec3(v) => InterpValue::Vec3([
                eval_unary_scalar(op, v[0]),
                eval_unary_scalar(op, v[1]),
                eval_unary_scalar(op, v[2]),
            ]),
        }
    }
}

fn eval_binary_scalar(op: BinaryOpIr, l: f64, r: f64) -> f64 {
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

fn eval_unary_scalar(op: UnaryOpIr, v: f64) -> f64 {
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

// ============================================================================
// Interpreter Context
// ============================================================================

pub struct MemberInterpContext<'a> {
    pub prev: InterpValue,
    pub index: usize,
    pub dt: f64,
    pub sim_time: f64,
    pub signals: &'a SignalStorage,
    pub members: &'a MemberSignalBuffer,
    pub constants: &'a IndexMap<String, f64>,
    pub config: &'a IndexMap<String, f64>,
    pub locals: HashMap<String, InterpValue>,
    pub entity_prefix: String,
    pub read_current: bool,
}

impl<'a> MemberInterpContext<'a> {
    pub fn from_scalar_context(
        ctx: &'a ScalarResolveContext<'a>,
        constants: &'a IndexMap<String, f64>,
        config: &'a IndexMap<String, f64>,
        entity_prefix: &str,
    ) -> Self {
        Self {
            prev: InterpValue::Scalar(ctx.prev),
            index: ctx.index.0,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            signals: ctx.signals,
            members: ctx.members,
            constants,
            config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.to_string(),
            read_current: false,
        }
    }

    pub fn from_vec3_context(
        ctx: &'a Vec3ResolveContext<'a>,
        constants: &'a IndexMap<String, f64>,
        config: &'a IndexMap<String, f64>,
        entity_prefix: &str,
    ) -> Self {
        Self {
            prev: InterpValue::Vec3(ctx.prev),
            index: ctx.index.0,
            dt: ctx.dt.seconds(),
            sim_time: ctx.sim_time,
            signals: ctx.signals,
            members: ctx.members,
            constants,
            config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.to_string(),
            read_current: false,
        }
    }

    fn signal(&self, name: &str) -> InterpValue {
        let runtime_id = SignalId::from(name);
        match self.signals.get(&runtime_id) {
            Some(v) => {
                if let Some(s) = v.as_scalar() {
                    InterpValue::Scalar(s)
                } else if let Some(v3) = v.as_vec3() {
                    InterpValue::Vec3(v3)
                } else {
                    panic!("Unsupported signal type for interpreter: {}", name)
                }
            }
            None => panic!("Signal '{}' not found", name),
        }
    }

    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId::from(name);
        match self.signals.get(&runtime_id) {
            Some(v) => v.component(component).unwrap_or(0.0),
            None => panic!("Signal '{}' not found", name),
        }
    }

    fn self_field(&self, field: &str) -> InterpValue {
        let full_path = format!("{}.{}", self.entity_prefix, field);
        let value = if self.read_current {
            self.members.get_current(&full_path, self.index)
        } else {
            self.members.get_previous(&full_path, self.index)
        };
        match value {
            Some(v) => {
                if let Some(s) = v.as_scalar() {
                    InterpValue::Scalar(s)
                } else if let Some(v3) = v.as_vec3() {
                    InterpValue::Vec3(v3)
                } else {
                    panic!("Unsupported member field type: {}", full_path)
                }
            }
            None => InterpValue::Scalar(0.0),
        }
    }

    fn self_field_component(&self, field: &str, component: &str) -> f64 {
        let full_path = format!("{}.{}", self.entity_prefix, field);
        let value = if self.read_current {
            self.members.get_current(&full_path, self.index)
        } else {
            self.members.get_previous(&full_path, self.index)
        };
        value.and_then(|v| v.component(component)).unwrap_or(0.0)
    }

    fn constant(&self, name: &str) -> f64 {
        self.constants.get(name).copied().unwrap_or(0.0)
    }

    fn config(&self, name: &str) -> f64 {
        self.config.get(name).copied().unwrap_or(0.0)
    }
}

impl ExecutionContext for MemberInterpContext<'_> {
    fn prev(&self) -> f64 {
        self.prev.as_scalar()
    }
    fn prev_component(&self, component: &str) -> f64 {
        self.prev.component(component)
    }
    fn dt(&self) -> f64 {
        self.dt
    }
    fn sim_time(&self) -> f64 {
        self.sim_time
    }
    fn inputs(&self) -> f64 {
        0.0
    }
    fn signal(&self, name: &str) -> f64 {
        self.signal(name).as_scalar()
    }
    fn signal_component(&self, name: &str, component: &str) -> f64 {
        self.signal_component(name, component)
    }
    fn constant(&self, name: &str) -> f64 {
        self.constant(name)
    }
    fn config(&self, name: &str) -> f64 {
        self.config(name)
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
        CompiledExpr::Const(name) => InterpValue::Scalar(ctx.constant(name)),
        CompiledExpr::Config(name) => InterpValue::Scalar(ctx.config(name)),
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
        CompiledExpr::DtRobustCall { operator, args, .. } => {
            let arg_vals: Vec<f64> = args
                .iter()
                .map(|a| interpret_expr(a, ctx).as_f64())
                .collect();
            let name = match operator {
                DtRobustOperator::Integrate => "integrate",
                DtRobustOperator::Decay => "decay",
                DtRobustOperator::Relax => "relax",
                DtRobustOperator::Accumulate => "accumulate",
                DtRobustOperator::AdvancePhase => "advance_phase",
                DtRobustOperator::Smooth => "smooth",
                DtRobustOperator::Damp => "damp",
            };
            InterpValue::Scalar(ctx.call_kernel(name, &arg_vals))
        }
        CompiledExpr::FieldAccess { object, field } => match object.as_ref() {
            CompiledExpr::Signal(id) => {
                InterpValue::Scalar(ctx.signal_component(&id.to_string(), field))
            }
            CompiledExpr::Prev => InterpValue::Scalar(ctx.prev.component(field)),
            CompiledExpr::SelfField(f) => InterpValue::Scalar(ctx.self_field_component(f, field)),
            _ => {
                let v = interpret_expr(object, ctx);
                InterpValue::Scalar(v.component(field))
            }
        },
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = interpret_expr(condition, ctx).as_f64();
            if cond != 0.0 {
                interpret_expr(then_branch, ctx)
            } else {
                interpret_expr(else_branch, ctx)
            }
        }
        CompiledExpr::Let { name, value, body } => {
            let val = interpret_expr(value, ctx);
            ctx.locals.insert(name.clone(), val);
            let res = interpret_expr(body, ctx);
            ctx.locals.remove(name);
            res
        }
        CompiledExpr::Local(name) => ctx
            .locals
            .get(name)
            .cloned()
            .unwrap_or(InterpValue::Scalar(0.0)),
        CompiledExpr::SelfField(field) => ctx.self_field(field),
        CompiledExpr::Aggregate { op, entity, body } => {
            let instance_count = ctx.members.instance_count_for_entity(&entity.to_string());
            let mut results = Vec::with_capacity(instance_count);
            let original_prefix = ctx.entity_prefix.clone();
            let original_index = ctx.index;
            let original_read_current = ctx.read_current;

            ctx.read_current = true;
            ctx.entity_prefix = entity.to_string();
            for i in 0..instance_count {
                ctx.index = i;
                results.push(interpret_expr(body, ctx).as_f64());
            }
            ctx.entity_prefix = original_prefix;
            ctx.index = original_index;
            ctx.read_current = original_read_current;
            InterpValue::Scalar(reduce_results(results, *op))
        }
        _ => InterpValue::Scalar(0.0),
    }
}

fn eval_function(name: &str, args: &[InterpValue], ctx: &MemberInterpContext) -> InterpValue {
    match name {
        "vec3" | "Vec3" if args.len() == 3 => {
            InterpValue::Vec3([args[0].as_f64(), args[1].as_f64(), args[2].as_f64()])
        }
        "clamp" if args.len() == 3 => {
            InterpValue::Scalar(args[0].as_f64().clamp(args[1].as_f64(), args[2].as_f64()))
        }
        _ => {
            let scalar_args: Vec<f64> = args.iter().map(|a| a.as_f64()).collect();
            InterpValue::Scalar(ctx.call_kernel(name, &scalar_args))
        }
    }
}

fn reduce_results(results: Vec<f64>, op: AggregateOpIr) -> f64 {
    match op {
        AggregateOpIr::Sum => results.iter().sum(),
        AggregateOpIr::Mean => {
            if results.is_empty() {
                0.0
            } else {
                results.iter().sum::<f64>() / results.len() as f64
            }
        }
        AggregateOpIr::Min => results.iter().copied().fold(f64::INFINITY, f64::min),
        AggregateOpIr::Max => results.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        AggregateOpIr::Count => results.len() as f64,
        _ => 0.0,
    }
}

pub type MemberResolverFn = Box<dyn Fn(&ScalarResolveContext) -> f64 + Send + Sync>;
pub type Vec3MemberResolverFn = Box<dyn Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync>;

pub fn build_member_resolver(
    expr: &CompiledExpr,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    entity_prefix: &str,
) -> MemberResolverFn {
    let expr = expr.clone();
    let constants = constants.clone();
    let config = config.clone();
    let entity_prefix = entity_prefix.to_string();

    Box::new(move |ctx| {
        let mut interp_ctx =
            MemberInterpContext::from_scalar_context(ctx, &constants, &config, &entity_prefix);
        interpret_expr(&expr, &mut interp_ctx).as_scalar()
    })
}

pub fn build_vec3_member_resolver(
    expr: &CompiledExpr,
    constants: &IndexMap<String, f64>,
    config: &IndexMap<String, f64>,
    entity_prefix: &str,
) -> Vec3MemberResolverFn {
    let expr = expr.clone();
    let constants = constants.clone();
    let config = config.clone();
    let entity_prefix = entity_prefix.to_string();

    Box::new(move |ctx| {
        let mut interp_ctx =
            MemberInterpContext::from_vec3_context(ctx, &constants, &config, &entity_prefix);
        interpret_expr(&expr, &mut interp_ctx).as_vec3()
    })
}
