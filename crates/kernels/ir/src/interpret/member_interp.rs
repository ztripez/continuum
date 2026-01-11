//! Member signal expression interpreter.
//!
//! This module provides an interpreter for evaluating member signal expressions.
//! Unlike global signals that use bytecode compilation, member signals require
//! direct interpretation because they access per-instance data via `self.*`
//! (SelfField) expressions which cannot be compiled to bytecode.
//!
//! # Architecture
//!
//! ```text
//! CompiledExpr (with SelfField) → interpret_expr() → f64 result
//!                                      ↑
//!                            MemberInterpContext
//!                            ├─ prev: f64
//!                            ├─ index: usize
//!                            ├─ dt: f64
//!                            ├─ signals: &SignalStorage
//!                            ├─ members: &MemberSignalBuffer
//!                            ├─ constants: &IndexMap<String, f64>
//!                            └─ config: &IndexMap<String, f64>
//! ```
//!
//! # Why Interpretation?
//!
//! The bytecode VM (`continuum_vm`) deliberately panics on entity expressions:
//!
//! ```ignore
//! CompiledExpr::SelfField(field) => {
//!     panic!("SelfField({}) reached bytecode compiler...");
//! }
//! ```
//!
//! This is intentional - member signals execute per-instance in parallel, and
//! the bytecode VM lacks the context for per-instance data access. This
//! interpreter fills that gap by evaluating expressions with full member context.

use std::collections::HashMap;

use indexmap::IndexMap;

use continuum_runtime::executor::member_executor::ScalarResolveContext;
use continuum_runtime::soa_storage::MemberSignalBuffer;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::SignalId;

use crate::{BinaryOpIr, CompiledExpr, DtRobustOperator, UnaryOpIr};

/// Context for interpreting member expressions.
///
/// This context provides all data needed to evaluate a member signal expression
/// for a single entity instance. It wraps a `ScalarResolveContext` with additional
/// data needed for interpretation.
pub struct MemberInterpContext<'a> {
    /// Previous tick's value for this member signal instance
    pub prev: f64,
    /// Entity instance index
    pub index: usize,
    /// Time step in seconds
    pub dt: f64,
    /// Read-only access to global signals
    pub signals: &'a SignalStorage,
    /// Read-only access to member signal buffer
    pub members: &'a MemberSignalBuffer,
    /// World constants
    pub constants: &'a IndexMap<String, f64>,
    /// World config values
    pub config: &'a IndexMap<String, f64>,
    /// Local variable bindings (for `let` expressions)
    pub locals: HashMap<String, f64>,
    /// Entity prefix for constructing full member paths (e.g., "terra.plate")
    /// Used to convert short field names like "age" to full paths like "terra.plate.age"
    pub entity_prefix: String,
}

impl<'a> MemberInterpContext<'a> {
    /// Create a context from a scalar resolve context and world data.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The scalar resolve context from the runtime
    /// * `constants` - World constants
    /// * `config` - World config values
    /// * `entity_prefix` - The entity path prefix (e.g., "terra.plate" for member "terra.plate.age")
    pub fn from_resolve_context(
        ctx: &'a ScalarResolveContext<'a>,
        constants: &'a IndexMap<String, f64>,
        config: &'a IndexMap<String, f64>,
        entity_prefix: &str,
    ) -> Self {
        Self {
            prev: ctx.prev,
            index: ctx.index.0,
            dt: ctx.dt.seconds(),
            signals: ctx.signals,
            members: ctx.members,
            constants,
            config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.to_string(),
        }
    }

    /// Get a signal value by name.
    fn signal(&self, name: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        match self.signals.get(&runtime_id) {
            Some(v) => v.as_scalar().unwrap_or_else(|| {
                panic!(
                    "Signal '{}' exists but is not a scalar - cannot convert to f64",
                    name
                )
            }),
            None => panic!("Signal '{}' not found in storage", name),
        }
    }

    /// Get a signal component by name and component (x, y, z, w).
    fn signal_component(&self, name: &str, component: &str) -> f64 {
        let runtime_id = SignalId(name.to_string());
        match self.signals.get(&runtime_id) {
            Some(v) => v.component(component).unwrap_or_else(|| {
                panic!(
                    "Signal '{}' has no component '{}' - expected vector with x/y/z/w components",
                    name, component
                )
            }),
            None => panic!("Signal '{}' not found in storage", name),
        }
    }

    /// Get a member field value for the current instance.
    ///
    /// Constructs the full member path from the entity prefix and field name.
    /// For example, if entity_prefix is "terra.plate" and field is "age",
    /// looks up "terra.plate.age".
    fn self_field(&self, field: &str) -> f64 {
        let full_path = format!("{}.{}", self.entity_prefix, field);
        self.members
            .get_previous(&full_path, self.index)
            .and_then(|v| v.as_scalar())
            .unwrap_or_else(|| {
                panic!(
                    "Member field '{}' (full path: '{}') not found for instance {} or is not a scalar",
                    field, full_path, self.index
                )
            })
    }

    /// Get a member field component for the current instance.
    ///
    /// Constructs the full member path from the entity prefix and field name.
    fn self_field_component(&self, field: &str, component: &str) -> f64 {
        let full_path = format!("{}.{}", self.entity_prefix, field);
        self.members
            .get_previous(&full_path, self.index)
            .and_then(|v| v.component(component))
            .unwrap_or_else(|| {
                panic!(
                    "Member field '{}' (full path: '{}') has no component '{}' for instance {}",
                    field, full_path, component, self.index
                )
            })
    }

    /// Get a constant value by name.
    fn constant(&self, name: &str) -> f64 {
        self.constants
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("Constant '{}' not defined", name))
    }

    /// Get a config value by name.
    fn config(&self, name: &str) -> f64 {
        self.config
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("Config value '{}' not defined", name))
    }
}

/// Interpret a compiled expression in a member context.
///
/// This function recursively evaluates a `CompiledExpr` tree, handling all
/// expression types including entity-specific ones like `SelfField`.
///
/// # Arguments
///
/// * `expr` - The compiled expression to evaluate
/// * `ctx` - The member interpretation context
///
/// # Returns
///
/// The evaluated scalar result.
///
/// # Panics
///
/// Panics if the expression cannot be evaluated (missing signals, type mismatches, etc.)
pub fn interpret_expr(expr: &CompiledExpr, ctx: &mut MemberInterpContext) -> f64 {
    match expr {
        // Leaf expressions
        CompiledExpr::Literal(v) => *v,
        CompiledExpr::Prev => ctx.prev,
        CompiledExpr::DtRaw => ctx.dt,
        CompiledExpr::Collected => 0.0, // Members don't use collected inputs
        CompiledExpr::Signal(id) => ctx.signal(&id.0),
        CompiledExpr::Const(name) => ctx.constant(name),
        CompiledExpr::Config(name) => ctx.config(name),
        CompiledExpr::Local(name) => ctx.locals.get(name).copied().unwrap_or_else(|| {
            panic!("Local variable '{}' not found", name)
        }),

        // Entity-specific expressions
        CompiledExpr::SelfField(field) => ctx.self_field(field),

        // Binary operations
        CompiledExpr::Binary { op, left, right } => {
            let l = interpret_expr(left, ctx);
            let r = interpret_expr(right, ctx);
            eval_binary_op(*op, l, r)
        }

        // Unary operations
        CompiledExpr::Unary { op, operand } => {
            let v = interpret_expr(operand, ctx);
            eval_unary_op(*op, v)
        }

        // Function calls
        CompiledExpr::Call { function, args } => {
            let arg_values: Vec<f64> = args.iter().map(|a| interpret_expr(a, ctx)).collect();
            eval_function(function, &arg_values)
        }

        // Kernel calls
        CompiledExpr::KernelCall { function, args } => {
            let arg_values: Vec<f64> = args.iter().map(|a| interpret_expr(a, ctx)).collect();
            let kernel_name = format!("kernel.{}", function);
            continuum_kernel_registry::eval(&kernel_name, &arg_values, ctx.dt)
                .unwrap_or_else(|| panic!("Unknown kernel function '{}'", kernel_name))
        }

        // Dt-robust operators
        CompiledExpr::DtRobustCall {
            operator,
            args,
            method: _,
        } => {
            let arg_values: Vec<f64> = args.iter().map(|a| interpret_expr(a, ctx)).collect();
            eval_dt_robust(*operator, &arg_values, ctx.dt)
        }

        // Conditional
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = interpret_expr(condition, ctx);
            if cond != 0.0 {
                interpret_expr(then_branch, ctx)
            } else {
                interpret_expr(else_branch, ctx)
            }
        }

        // Let bindings
        CompiledExpr::Let { name, value, body } => {
            let val = interpret_expr(value, ctx);
            ctx.locals.insert(name.clone(), val);
            let result = interpret_expr(body, ctx);
            ctx.locals.remove(name);
            result
        }

        // Field access on signals/prev
        CompiledExpr::FieldAccess { object, field } => match object.as_ref() {
            CompiledExpr::Signal(id) => ctx.signal_component(&id.0, field),
            CompiledExpr::Prev => {
                // For member signals, prev is a scalar - component access not supported
                panic!(
                    "Component access on scalar prev not supported for member signals (field: {})",
                    field
                )
            }
            CompiledExpr::SelfField(member_field) => {
                // Access component of a vector member field
                ctx.self_field_component(member_field, field)
            }
            _ => panic!("Unsupported field access base: {:?}", object),
        },

        // Entity aggregate operations - not yet implemented
        CompiledExpr::Aggregate { op, entity, .. } => {
            panic!(
                "Aggregate({:?} over {}) not yet implemented in member interpreter",
                op, entity.0
            )
        }

        // Other entity operations - not yet implemented
        CompiledExpr::EntityAccess {
            entity,
            instance,
            field,
        } => {
            panic!(
                "EntityAccess({}.{}.{}) not yet implemented in member interpreter",
                entity.0, instance.0, field
            )
        }
        CompiledExpr::Other { entity, .. } => {
            panic!(
                "Other({}) not yet implemented in member interpreter",
                entity.0
            )
        }
        CompiledExpr::Pairs { entity, .. } => {
            panic!(
                "Pairs({}) not yet implemented in member interpreter",
                entity.0
            )
        }
        CompiledExpr::Filter { entity, .. } => {
            panic!(
                "Filter({}) not yet implemented in member interpreter",
                entity.0
            )
        }
        CompiledExpr::First { entity, .. } => {
            panic!(
                "First({}) not yet implemented in member interpreter",
                entity.0
            )
        }
        CompiledExpr::Nearest { entity, .. } => {
            panic!(
                "Nearest({}) not yet implemented in member interpreter",
                entity.0
            )
        }
        CompiledExpr::Within { entity, .. } => {
            panic!(
                "Within({}) not yet implemented in member interpreter",
                entity.0
            )
        }

        // Impulse expressions - not applicable to member signals
        CompiledExpr::Payload => panic!("Payload not supported in member expressions"),
        CompiledExpr::PayloadField(f) => {
            panic!("PayloadField({}) not supported in member expressions", f)
        }
        CompiledExpr::EmitSignal { target, .. } => {
            panic!(
                "EmitSignal({}) not supported in member expressions",
                target.0
            )
        }
    }
}

/// Evaluate a binary operation.
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

/// Evaluate a unary operation.
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

/// Evaluate a built-in function call.
fn eval_function(name: &str, args: &[f64]) -> f64 {
    match name {
        // Math functions
        "abs" => args.first().map(|v| v.abs()).unwrap_or(0.0),
        "sin" => args.first().map(|v| v.sin()).unwrap_or(0.0),
        "cos" => args.first().map(|v| v.cos()).unwrap_or(0.0),
        "tan" => args.first().map(|v| v.tan()).unwrap_or(0.0),
        "asin" => args.first().map(|v| v.asin()).unwrap_or(0.0),
        "acos" => args.first().map(|v| v.acos()).unwrap_or(0.0),
        "atan" => args.first().map(|v| v.atan()).unwrap_or(0.0),
        "sqrt" => args.first().map(|v| v.sqrt()).unwrap_or(0.0),
        "exp" => args.first().map(|v| v.exp()).unwrap_or(0.0),
        "ln" | "log" => args.first().map(|v| v.ln()).unwrap_or(0.0),
        "log10" => args.first().map(|v| v.log10()).unwrap_or(0.0),
        "floor" => args.first().map(|v| v.floor()).unwrap_or(0.0),
        "ceil" => args.first().map(|v| v.ceil()).unwrap_or(0.0),
        "round" => args.first().map(|v| v.round()).unwrap_or(0.0),
        "sign" | "signum" => args.first().map(|v| v.signum()).unwrap_or(0.0),
        "min" => args.iter().copied().fold(f64::INFINITY, f64::min),
        "max" => args.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "clamp" if args.len() == 3 => args[0].clamp(args[1], args[2]),
        "lerp" if args.len() == 3 => args[0] + (args[1] - args[0]) * args[2],
        "pow" if args.len() == 2 => args[0].powf(args[1]),
        "atan2" if args.len() == 2 => args[0].atan2(args[1]),

        // Vector normalization (for scalar, just return 1.0 if non-zero, 0.0 otherwise)
        // This is used when computing normalized directions from scalar values
        "normalize" if args.len() == 1 => {
            if args[0].abs() > f64::EPSILON { 1.0 } else { 0.0 }
        }

        // Vector length (for single scalar, just return abs)
        "length" if args.len() == 1 => args[0].abs(),
        "length" if args.len() == 2 => (args[0] * args[0] + args[1] * args[1]).sqrt(),
        "length" if args.len() == 3 => (args[0] * args[0] + args[1] * args[1] + args[2] * args[2]).sqrt(),

        // Try kernel registry for unknown functions
        _ => continuum_kernel_registry::eval(name, args, 0.0)
            .unwrap_or_else(|| panic!("Unknown function '{}' with {} args", name, args.len())),
    }
}

/// Evaluate a dt-robust operator.
fn eval_dt_robust(op: DtRobustOperator, args: &[f64], dt: f64) -> f64 {
    let fn_name = match op {
        DtRobustOperator::Integrate => "integrate",
        DtRobustOperator::Decay => "decay",
        DtRobustOperator::Relax => "relax",
        DtRobustOperator::Accumulate => "accumulate",
        DtRobustOperator::AdvancePhase => "advance_phase",
        DtRobustOperator::Smooth => "smooth",
        DtRobustOperator::Damp => "damp",
    };

    continuum_kernel_registry::eval(fn_name, args, dt)
        .unwrap_or_else(|| panic!("Dt-robust function '{}' not found in registry", fn_name))
}

/// Type alias for member resolver functions.
///
/// A member resolver takes a scalar resolve context and returns the new value.
pub type MemberResolverFn = Box<dyn Fn(&ScalarResolveContext) -> f64 + Send + Sync>;

/// Build a member resolver function from a compiled expression.
///
/// This function creates a closure that evaluates the expression using the
/// interpreter. The closure captures constants, config, and entity_prefix for
/// efficient access.
///
/// # Arguments
///
/// * `expr` - The compiled resolve expression
/// * `constants` - World constants
/// * `config` - World config values
/// * `entity_prefix` - The entity path prefix (e.g., "terra.plate" for "terra.plate.age")
///
/// # Returns
///
/// A boxed function that can be called with a `ScalarResolveContext` to
/// compute the new member signal value.
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

    Box::new(move |ctx: &ScalarResolveContext| {
        let mut interp_ctx = MemberInterpContext::from_resolve_context(ctx, &constants, &config, &entity_prefix);
        interpret_expr(&expr, &mut interp_ctx)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_foundation::SignalId;
    use continuum_runtime::executor::member_executor::MemberResolveContext;
    use continuum_runtime::soa_storage::ValueType;
    use continuum_runtime::types::{Dt, Value};
    use continuum_runtime::vectorized::EntityIndex;

    fn create_test_signals() -> SignalStorage {
        let mut storage = SignalStorage::default();
        storage.init("global.temp".into(), Value::Scalar(25.0));
        storage.init("global.scale".into(), Value::Scalar(2.0));
        storage
    }

    // Test entity prefix used in tests
    const TEST_ENTITY_PREFIX: &str = "test.entity";

    fn create_test_members(count: usize) -> MemberSignalBuffer {
        let mut buffer = MemberSignalBuffer::new();
        // Use full paths (entity_prefix.field_name) for storage
        buffer.register_signal(format!("{}.age", TEST_ENTITY_PREFIX), ValueType::Scalar);
        buffer.register_signal(format!("{}.mass", TEST_ENTITY_PREFIX), ValueType::Scalar);
        buffer.init_instances(count);

        // Set some previous values
        for i in 0..count {
            buffer.set_current(&format!("{}.age", TEST_ENTITY_PREFIX), i, Value::Scalar((i + 1) as f64 * 10.0));
            buffer.set_current(&format!("{}.mass", TEST_ENTITY_PREFIX), i, Value::Scalar(100.0 + i as f64));
        }
        buffer.advance_tick();

        buffer
    }

    fn create_test_context<'a>(
        prev: f64,
        index: usize,
        signals: &'a SignalStorage,
        members: &'a MemberSignalBuffer,
        constants: &'a IndexMap<String, f64>,
        config: &'a IndexMap<String, f64>,
    ) -> MemberInterpContext<'a> {
        MemberInterpContext {
            prev,
            index,
            dt: 0.1,
            signals,
            members,
            constants,
            config,
            locals: HashMap::new(),
            entity_prefix: TEST_ENTITY_PREFIX.to_string(),
        }
    }

    #[test]
    fn test_literal() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(0.0, 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Literal(42.0);
        assert_eq!(interpret_expr(&expr, &mut ctx), 42.0);
    }

    #[test]
    fn test_prev() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(123.0, 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Prev;
        assert_eq!(interpret_expr(&expr, &mut ctx), 123.0);
    }

    #[test]
    fn test_self_field() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(0.0, 1, &signals, &members, &constants, &config);

        // Instance 1 has age = 20.0 (from setup: (1+1)*10 = 20)
        let expr = CompiledExpr::SelfField("age".to_string());
        assert_eq!(interpret_expr(&expr, &mut ctx), 20.0);
    }

    #[test]
    fn test_binary_add() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(100.0, 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx), 101.0);
    }

    #[test]
    fn test_signal_access() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(0.0, 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Signal(SignalId::from("global.temp"));
        assert_eq!(interpret_expr(&expr, &mut ctx), 25.0);
    }

    #[test]
    fn test_complex_expression() {
        // prev + self.age * signal * 0.1
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(50.0, 0, &signals, &members, &constants, &config);

        // Instance 0: age = 10.0, temp = 25.0
        // Result: 50 + 10 * 25 * 0.1 = 50 + 25 = 75
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Binary {
                op: BinaryOpIr::Mul,
                left: Box::new(CompiledExpr::Binary {
                    op: BinaryOpIr::Mul,
                    left: Box::new(CompiledExpr::SelfField("age".to_string())),
                    right: Box::new(CompiledExpr::Signal(SignalId::from("global.temp"))),
                }),
                right: Box::new(CompiledExpr::Literal(0.1)),
            }),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx), 75.0);
    }

    #[test]
    fn test_build_member_resolver() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();

        // Build resolver: prev + 1.0
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        };

        let resolver = build_member_resolver(&expr, &constants, &config, TEST_ENTITY_PREFIX);

        let ctx = MemberResolveContext {
            prev: 100.0,
            index: EntityIndex(0),
            signals: &signals,
            members: &members,
            dt: Dt(0.1),
        };

        assert_eq!(resolver(&ctx), 101.0);
    }

    #[test]
    fn test_resolver_with_self_field() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();

        // Build resolver: prev + self.mass
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::SelfField("mass".to_string())),
        };

        let resolver = build_member_resolver(&expr, &constants, &config, TEST_ENTITY_PREFIX);

        // Instance 1 has mass = 101.0 (100.0 + 1)
        let ctx = MemberResolveContext {
            prev: 50.0,
            index: EntityIndex(1),
            signals: &signals,
            members: &members,
            dt: Dt(0.1),
        };

        assert_eq!(resolver(&ctx), 151.0); // 50 + 101 = 151
    }
}
