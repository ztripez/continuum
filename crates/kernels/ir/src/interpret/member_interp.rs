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
//! CompiledExpr (with SelfField) → interpret_expr() → InterpValue result
//!                                      ↑
//!                            MemberInterpContext
//!                            ├─ prev: InterpValue
//!                            ├─ index: usize
//!                            ├─ dt: f64
//!                            ├─ signals: &SignalStorage
//!                            ├─ members: &MemberSignalBuffer
//!                            ├─ constants: &IndexMap<String, f64>
//!                            └─ config: &IndexMap<String, f64>
//! ```
//!
//! # Value Types
//!
//! The interpreter supports multiple value types through [`InterpValue`]:
//! - `Scalar(f64)` - Single floating-point values
//! - `Vec3([f64; 3])` - 3D vectors (x, y, z components)
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

use continuum_runtime::executor::member_executor::{ScalarResolveContext, Vec3ResolveContext};
use continuum_runtime::soa_storage::MemberSignalBuffer;
use continuum_runtime::storage::SignalStorage;
use continuum_runtime::SignalId;

use crate::{AggregateOpIr, BinaryOpIr, CompiledExpr, DtRobustOperator, UnaryOpIr};

// ============================================================================
// Interpreter Value Type
// ============================================================================

/// Value type for the interpreter, supporting multiple numeric types.
///
/// This enum allows the interpreter to handle both scalar and vector
/// operations transparently, extracting the appropriate type at the end.
#[derive(Debug, Clone, Copy)]
pub enum InterpValue {
    /// Single f64 scalar value
    Scalar(f64),
    /// 3D vector [x, y, z]
    Vec3([f64; 3]),
}

impl InterpValue {
    /// Extract as scalar, panicking if not a scalar.
    #[inline]
    pub fn as_scalar(self) -> f64 {
        match self {
            InterpValue::Scalar(v) => v,
            InterpValue::Vec3(_) => panic!("Expected scalar, got Vec3"),
        }
    }

    /// Extract as Vec3, panicking if not a Vec3.
    #[inline]
    pub fn as_vec3(self) -> [f64; 3] {
        match self {
            InterpValue::Vec3(v) => v,
            InterpValue::Scalar(_) => panic!("Expected Vec3, got scalar"),
        }
    }

    /// Check if this is a scalar value.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        matches!(self, InterpValue::Scalar(_))
    }

    /// Check if this is a Vec3 value.
    #[inline]
    pub fn is_vec3(&self) -> bool {
        matches!(self, InterpValue::Vec3(_))
    }

    /// Get a component by name (x, y, z for Vec3, or the scalar value itself).
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
}

impl Default for InterpValue {
    fn default() -> Self {
        InterpValue::Scalar(0.0)
    }
}

impl From<f64> for InterpValue {
    fn from(v: f64) -> Self {
        InterpValue::Scalar(v)
    }
}

impl From<[f64; 3]> for InterpValue {
    fn from(v: [f64; 3]) -> Self {
        InterpValue::Vec3(v)
    }
}

// ============================================================================
// Interpreter Context
// ============================================================================

/// Context for interpreting member expressions.
///
/// This context provides all data needed to evaluate a member signal expression
/// for a single entity instance. It wraps a resolve context with additional
/// data needed for interpretation.
pub struct MemberInterpContext<'a> {
    /// Previous tick's value for this member signal instance
    pub prev: InterpValue,
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
    pub locals: HashMap<String, InterpValue>,
    /// Entity prefix for constructing full member paths (e.g., "terra.plate")
    /// Used to convert short field names like "age" to full paths like "terra.plate.age"
    pub entity_prefix: String,
    /// Whether to read current tick values (for aggregates after member resolution)
    /// or previous tick values (for member resolution during the tick)
    pub read_current: bool,
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
            signals: ctx.signals,
            members: ctx.members,
            constants,
            config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.to_string(),
            read_current: false, // During member resolution, read previous tick values
        }
    }

    /// Create a context from a Vec3 resolve context and world data.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The Vec3 resolve context from the runtime
    /// * `constants` - World constants
    /// * `config` - World config values
    /// * `entity_prefix` - The entity path prefix (e.g., "terra.plate" for member "terra.plate.position")
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
            signals: ctx.signals,
            members: ctx.members,
            constants,
            config,
            locals: HashMap::new(),
            entity_prefix: entity_prefix.to_string(),
            read_current: false, // During member resolution, read previous tick values
        }
    }

    /// Get a signal value by name as InterpValue.
    fn signal(&self, name: &str) -> InterpValue {
        let runtime_id = SignalId(name.to_string());
        match self.signals.get(&runtime_id) {
            Some(continuum_runtime::types::Value::Scalar(v)) => InterpValue::Scalar(*v),
            Some(continuum_runtime::types::Value::Vec3(v)) => InterpValue::Vec3(*v),
            Some(v) => {
                // For other vector types, extract as scalar if possible
                if let Some(s) = v.as_scalar() {
                    InterpValue::Scalar(s)
                } else {
                    panic!(
                        "Signal '{}' has unsupported type for interpreter: {:?}",
                        name, v
                    )
                }
            }
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
    ///
    /// Uses `get_current()` if `read_current` is true (for aggregate evaluation),
    /// otherwise uses `get_previous()` (for member resolution during tick).
    fn self_field(&self, field: &str) -> InterpValue {
        let full_path = format!("{}.{}", self.entity_prefix, field);
        let value = if self.read_current {
            self.members.get_current(&full_path, self.index)
        } else {
            self.members.get_previous(&full_path, self.index)
        };
        match value {
            Some(continuum_runtime::types::Value::Scalar(v)) => InterpValue::Scalar(v),
            Some(continuum_runtime::types::Value::Vec3(v)) => InterpValue::Vec3(v),
            Some(v) => {
                if let Some(s) = v.as_scalar() {
                    InterpValue::Scalar(s)
                } else {
                    panic!(
                        "Member field '{}' (full path: '{}') has unsupported type: {:?}",
                        field, full_path, v
                    )
                }
            }
            None => panic!(
                "Member field '{}' (full path: '{}') not found for instance {}",
                field, full_path, self.index
            ),
        }
    }

    /// Get a member field component for the current instance.
    ///
    /// Constructs the full member path from the entity prefix and field name.
    /// Uses `get_current()` if `read_current` is true (for aggregate evaluation),
    /// otherwise uses `get_previous()` (for member resolution during tick).
    fn self_field_component(&self, field: &str, component: &str) -> f64 {
        let full_path = format!("{}.{}", self.entity_prefix, field);
        let value = if self.read_current {
            self.members.get_current(&full_path, self.index)
        } else {
            self.members.get_previous(&full_path, self.index)
        };
        value
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

// ============================================================================
// Expression Interpreter
// ============================================================================

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
/// The evaluated result as [`InterpValue`] (scalar or Vec3).
///
/// # Panics
///
/// Panics if the expression cannot be evaluated (missing signals, type mismatches, etc.)
pub fn interpret_expr(expr: &CompiledExpr, ctx: &mut MemberInterpContext) -> InterpValue {
    match expr {
        // Leaf expressions
        CompiledExpr::Literal(v) => InterpValue::Scalar(*v),
        CompiledExpr::Prev => ctx.prev,
        CompiledExpr::DtRaw => InterpValue::Scalar(ctx.dt),
        CompiledExpr::Collected => InterpValue::Scalar(0.0), // Members don't use collected inputs
        CompiledExpr::Signal(id) => ctx.signal(&id.0),
        CompiledExpr::Const(name) => InterpValue::Scalar(ctx.constant(name)),
        CompiledExpr::Config(name) => InterpValue::Scalar(ctx.config(name)),
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
            let arg_values: Vec<InterpValue> = args.iter().map(|a| interpret_expr(a, ctx)).collect();
            eval_function(function, &arg_values)
        }

        // Kernel calls
        CompiledExpr::KernelCall { function, args } => {
            let arg_values: Vec<f64> = args.iter()
                .map(|a| interpret_expr(a, ctx).as_scalar())
                .collect();
            let kernel_name = format!("kernel.{}", function);
            let result = continuum_kernel_registry::eval(&kernel_name, &arg_values, ctx.dt)
                .unwrap_or_else(|| panic!("Unknown kernel function '{}'", kernel_name));
            InterpValue::Scalar(result)
        }

        // Dt-robust operators
        CompiledExpr::DtRobustCall {
            operator,
            args,
            method: _,
        } => {
            let arg_values: Vec<f64> = args.iter()
                .map(|a| interpret_expr(a, ctx).as_scalar())
                .collect();
            InterpValue::Scalar(eval_dt_robust(*operator, &arg_values, ctx.dt))
        }

        // Conditional
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = interpret_expr(condition, ctx).as_scalar();
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

        // Field access on signals/prev/self fields (extracts component from vectors)
        CompiledExpr::FieldAccess { object, field } => match object.as_ref() {
            CompiledExpr::Signal(id) => InterpValue::Scalar(ctx.signal_component(&id.0, field)),
            CompiledExpr::Prev => {
                // For Vec3 prev, extract component; for scalar, this is an error
                InterpValue::Scalar(ctx.prev.component(field))
            }
            CompiledExpr::SelfField(member_field) => {
                // Access component of a vector member field
                InterpValue::Scalar(ctx.self_field_component(member_field, field))
            }
            _ => {
                // Evaluate the object and extract component
                let obj_value = interpret_expr(object, ctx);
                InterpValue::Scalar(obj_value.component(field))
            }
        },

        // Entity aggregate operations: sum/mean/min/max/count over entity instances
        CompiledExpr::Aggregate { op, entity, body } => {
            let instance_count = ctx.members.instance_count();
            if instance_count == 0 {
                // Return identity for empty aggregations
                return InterpValue::Scalar(match op {
                    AggregateOpIr::Sum => 0.0,
                    AggregateOpIr::Product => 1.0,
                    AggregateOpIr::Min => f64::INFINITY,
                    AggregateOpIr::Max => f64::NEG_INFINITY,
                    AggregateOpIr::Mean => 0.0,
                    AggregateOpIr::Count => 0.0,
                    AggregateOpIr::Any => 0.0,
                    AggregateOpIr::All => 1.0,
                    AggregateOpIr::None => 1.0, // True (1.0) if no values are non-zero (vacuously true for empty set)
                });
            }

            // Save original context
            let original_prefix = ctx.entity_prefix.clone();
            let original_index = ctx.index;

            // Set entity context for the aggregation
            // The entity ID (e.g., "terra.plate") becomes the prefix for self.* lookups
            ctx.entity_prefix = entity.0.clone();

            // Collect values by evaluating body for each instance
            let values: Vec<f64> = (0..instance_count)
                .map(|i| {
                    ctx.index = i;
                    interpret_expr(body, ctx).as_scalar()
                })
                .collect();

            // Restore original context
            ctx.entity_prefix = original_prefix;
            ctx.index = original_index;

            // Aggregate the collected values
            let result = match op {
                AggregateOpIr::Sum => values.iter().sum(),
                AggregateOpIr::Product => values.iter().product(),
                AggregateOpIr::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
                AggregateOpIr::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                AggregateOpIr::Mean => {
                    let sum: f64 = values.iter().sum();
                    sum / values.len() as f64
                }
                AggregateOpIr::Count => values.len() as f64,
                AggregateOpIr::Any => {
                    if values.iter().any(|v| v.abs() > f64::EPSILON) { 1.0 } else { 0.0 }
                }
                AggregateOpIr::All => {
                    if values.iter().all(|v| v.abs() > f64::EPSILON) { 1.0 } else { 0.0 }
                }
                AggregateOpIr::None => {
                    // True (1.0) if no values are non-zero
                    if values.iter().all(|v| v.abs() <= f64::EPSILON) { 1.0 } else { 0.0 }
                }
            };
            InterpValue::Scalar(result)
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

// ============================================================================
// Binary Operations
// ============================================================================

/// Evaluate a binary operation on InterpValues.
///
/// Supports scalar-scalar, scalar-vec3, vec3-scalar, and vec3-vec3 operations.
fn eval_binary_op(op: BinaryOpIr, l: InterpValue, r: InterpValue) -> InterpValue {
    match (l, r) {
        // Scalar-Scalar
        (InterpValue::Scalar(l), InterpValue::Scalar(r)) => {
            InterpValue::Scalar(eval_binary_scalar(op, l, r))
        }
        // Vec3-Vec3
        (InterpValue::Vec3(l), InterpValue::Vec3(r)) => {
            InterpValue::Vec3(eval_binary_vec3(op, l, r))
        }
        // Scalar-Vec3 (broadcast scalar to all components)
        (InterpValue::Scalar(s), InterpValue::Vec3(v)) => {
            InterpValue::Vec3([
                eval_binary_scalar(op, s, v[0]),
                eval_binary_scalar(op, s, v[1]),
                eval_binary_scalar(op, s, v[2]),
            ])
        }
        // Vec3-Scalar (broadcast scalar to all components)
        (InterpValue::Vec3(v), InterpValue::Scalar(s)) => {
            InterpValue::Vec3([
                eval_binary_scalar(op, v[0], s),
                eval_binary_scalar(op, v[1], s),
                eval_binary_scalar(op, v[2], s),
            ])
        }
    }
}

/// Evaluate a binary operation on scalars.
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

/// Evaluate a binary operation on Vec3s.
fn eval_binary_vec3(op: BinaryOpIr, l: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    [
        eval_binary_scalar(op, l[0], r[0]),
        eval_binary_scalar(op, l[1], r[1]),
        eval_binary_scalar(op, l[2], r[2]),
    ]
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Evaluate a unary operation on InterpValue.
fn eval_unary_op(op: UnaryOpIr, v: InterpValue) -> InterpValue {
    match v {
        InterpValue::Scalar(s) => InterpValue::Scalar(eval_unary_scalar(op, s)),
        InterpValue::Vec3(v) => InterpValue::Vec3([
            eval_unary_scalar(op, v[0]),
            eval_unary_scalar(op, v[1]),
            eval_unary_scalar(op, v[2]),
        ]),
    }
}

/// Evaluate a unary operation on a scalar.
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
// Function Calls
// ============================================================================

/// Evaluate a built-in function call with InterpValue args.
fn eval_function(name: &str, args: &[InterpValue]) -> InterpValue {
    match name {
        // Vec3 constructor
        "Vec3" if args.len() == 3 => {
            InterpValue::Vec3([
                args[0].as_scalar(),
                args[1].as_scalar(),
                args[2].as_scalar(),
            ])
        }

        // Vector length - returns scalar
        "length" if args.len() == 1 => {
            match args[0] {
                InterpValue::Scalar(v) => InterpValue::Scalar(v.abs()),
                InterpValue::Vec3(v) => {
                    InterpValue::Scalar((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                }
            }
        }

        // Vector normalize - preserves type
        "normalize" if args.len() == 1 => {
            match args[0] {
                InterpValue::Scalar(v) => {
                    InterpValue::Scalar(if v.abs() > f64::EPSILON { v.signum() } else { 0.0 })
                }
                InterpValue::Vec3(v) => {
                    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if len > f64::EPSILON {
                        InterpValue::Vec3([v[0] / len, v[1] / len, v[2] / len])
                    } else {
                        InterpValue::Vec3([0.0, 0.0, 0.0])
                    }
                }
            }
        }

        // Dot product - returns scalar
        "dot" if args.len() == 2 => {
            match (&args[0], &args[1]) {
                (InterpValue::Vec3(a), InterpValue::Vec3(b)) => {
                    InterpValue::Scalar(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
                }
                _ => {
                    let a = args[0].as_scalar();
                    let b = args[1].as_scalar();
                    InterpValue::Scalar(a * b)
                }
            }
        }

        // Cross product - returns Vec3
        "cross" if args.len() == 2 => {
            let a = args[0].as_vec3();
            let b = args[1].as_vec3();
            InterpValue::Vec3([
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ])
        }

        // All other functions work on scalars
        _ => {
            let scalar_args: Vec<f64> = args.iter().map(|a| a.as_scalar()).collect();
            InterpValue::Scalar(eval_scalar_function(name, &scalar_args))
        }
    }
}

/// Evaluate a scalar function call.
fn eval_scalar_function(name: &str, args: &[f64]) -> f64 {
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

// ============================================================================
// Member Resolver Builders
// ============================================================================

/// Type alias for scalar member resolver functions.
///
/// A scalar member resolver takes a scalar resolve context and returns f64.
pub type MemberResolverFn = Box<dyn Fn(&ScalarResolveContext) -> f64 + Send + Sync>;

/// Type alias for Vec3 member resolver functions.
///
/// A Vec3 member resolver takes a Vec3 resolve context and returns [f64; 3].
pub type Vec3MemberResolverFn = Box<dyn Fn(&Vec3ResolveContext) -> [f64; 3] + Send + Sync>;

/// Build a scalar member resolver function from a compiled expression.
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
        let mut interp_ctx = MemberInterpContext::from_scalar_context(ctx, &constants, &config, &entity_prefix);
        interpret_expr(&expr, &mut interp_ctx).as_scalar()
    })
}

/// Build a Vec3 member resolver function from a compiled expression.
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
/// * `entity_prefix` - The entity path prefix (e.g., "terra.plate" for "terra.plate.position")
///
/// # Returns
///
/// A boxed function that can be called with a `Vec3ResolveContext` to
/// compute the new member signal value.
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

    Box::new(move |ctx: &Vec3ResolveContext| {
        let mut interp_ctx = MemberInterpContext::from_vec3_context(ctx, &constants, &config, &entity_prefix);
        interpret_expr(&expr, &mut interp_ctx).as_vec3()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_foundation::SignalId;
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
        buffer.register_signal(format!("{}.position", TEST_ENTITY_PREFIX), ValueType::Vec3);
        buffer.init_instances(count);

        // Set some previous values
        for i in 0..count {
            buffer.set_current(&format!("{}.age", TEST_ENTITY_PREFIX), i, Value::Scalar((i + 1) as f64 * 10.0));
            buffer.set_current(&format!("{}.mass", TEST_ENTITY_PREFIX), i, Value::Scalar(100.0 + i as f64));
            buffer.set_current(&format!("{}.position", TEST_ENTITY_PREFIX), i, Value::Vec3([i as f64, 0.0, 0.0]));
        }
        buffer.advance_tick();

        buffer
    }

    fn create_test_context<'a>(
        prev: InterpValue,
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
            read_current: false, // Tests use previous tick values like member resolution
        }
    }

    #[test]
    fn test_literal() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Literal(42.0);
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 42.0);
    }

    #[test]
    fn test_prev() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(123.0), 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Prev;
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 123.0);
    }

    #[test]
    fn test_self_field() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 1, &signals, &members, &constants, &config);

        // Instance 1 has age = 20.0 (from setup: (1+1)*10 = 20)
        let expr = CompiledExpr::SelfField("age".to_string());
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 20.0);
    }

    #[test]
    fn test_binary_add() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(100.0), 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 101.0);
    }

    #[test]
    fn test_signal_access() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Signal(SignalId::from("global.temp"));
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 25.0);
    }

    #[test]
    fn test_complex_expression() {
        // prev + self.age * signal * 0.1
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(50.0), 0, &signals, &members, &constants, &config);

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
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 75.0);
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

        let ctx = ScalarResolveContext {
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
        let ctx = ScalarResolveContext {
            prev: 50.0,
            index: EntityIndex(1),
            signals: &signals,
            members: &members,
            dt: Dt(0.1),
        };

        assert_eq!(resolver(&ctx), 151.0); // 50 + 101 = 151
    }

    #[test]
    fn test_aggregate_sum() {
        use continuum_foundation::EntityId;

        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        // Sum of all ages: 10 + 20 + 30 = 60
        let expr = CompiledExpr::Aggregate {
            op: AggregateOpIr::Sum,
            entity: EntityId::from(TEST_ENTITY_PREFIX),
            body: Box::new(CompiledExpr::SelfField("age".to_string())),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 60.0);
    }

    #[test]
    fn test_aggregate_mean() {
        use continuum_foundation::EntityId;

        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        // Mean of all ages: (10 + 20 + 30) / 3 = 20
        let expr = CompiledExpr::Aggregate {
            op: AggregateOpIr::Mean,
            entity: EntityId::from(TEST_ENTITY_PREFIX),
            body: Box::new(CompiledExpr::SelfField("age".to_string())),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 20.0);
    }

    #[test]
    fn test_aggregate_min_max() {
        use continuum_foundation::EntityId;

        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        // Min of all ages: 10
        let min_expr = CompiledExpr::Aggregate {
            op: AggregateOpIr::Min,
            entity: EntityId::from(TEST_ENTITY_PREFIX),
            body: Box::new(CompiledExpr::SelfField("age".to_string())),
        };
        assert_eq!(interpret_expr(&min_expr, &mut ctx).as_scalar(), 10.0);

        // Max of all ages: 30
        let max_expr = CompiledExpr::Aggregate {
            op: AggregateOpIr::Max,
            entity: EntityId::from(TEST_ENTITY_PREFIX),
            body: Box::new(CompiledExpr::SelfField("age".to_string())),
        };
        assert_eq!(interpret_expr(&max_expr, &mut ctx).as_scalar(), 30.0);
    }

    #[test]
    fn test_aggregate_count() {
        use continuum_foundation::EntityId;

        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        // Count of all instances: 3
        let expr = CompiledExpr::Aggregate {
            op: AggregateOpIr::Count,
            entity: EntityId::from(TEST_ENTITY_PREFIX),
            body: Box::new(CompiledExpr::Literal(1.0)), // body is ignored for count
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 3.0);
    }

    // ========================================================================
    // Vec3 Tests
    // ========================================================================

    #[test]
    fn test_vec3_prev() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Vec3([1.0, 2.0, 3.0]), 0, &signals, &members, &constants, &config);

        let expr = CompiledExpr::Prev;
        assert_eq!(interpret_expr(&expr, &mut ctx).as_vec3(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vec3_binary_add() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Vec3([1.0, 2.0, 3.0]), 0, &signals, &members, &constants, &config);

        // prev + prev (Vec3 + Vec3)
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Prev),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_vec3(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vec3_scalar_multiply() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Vec3([1.0, 2.0, 3.0]), 0, &signals, &members, &constants, &config);

        // prev * 2.0 (Vec3 * scalar)
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Mul,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(2.0)),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_vec3(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vec3_component_access() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Vec3([1.0, 2.0, 3.0]), 0, &signals, &members, &constants, &config);

        // prev.y
        let expr = CompiledExpr::FieldAccess {
            object: Box::new(CompiledExpr::Prev),
            field: "y".to_string(),
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 2.0);
    }

    #[test]
    fn test_vec3_self_field() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 1, &signals, &members, &constants, &config);

        // Instance 1 has position = [1.0, 0.0, 0.0]
        let expr = CompiledExpr::SelfField("position".to_string());
        assert_eq!(interpret_expr(&expr, &mut ctx).as_vec3(), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vec3_length_function() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Vec3([3.0, 4.0, 0.0]), 0, &signals, &members, &constants, &config);

        // length(prev) = sqrt(9 + 16 + 0) = 5
        let expr = CompiledExpr::Call {
            function: "length".to_string(),
            args: vec![CompiledExpr::Prev],
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_scalar(), 5.0);
    }

    #[test]
    fn test_vec3_normalize_function() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Vec3([3.0, 4.0, 0.0]), 0, &signals, &members, &constants, &config);

        // normalize(prev) = [0.6, 0.8, 0.0]
        let expr = CompiledExpr::Call {
            function: "normalize".to_string(),
            args: vec![CompiledExpr::Prev],
        };
        let result = interpret_expr(&expr, &mut ctx).as_vec3();
        assert!((result[0] - 0.6).abs() < 1e-10);
        assert!((result[1] - 0.8).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_constructor() {
        let signals = create_test_signals();
        let members = create_test_members(3);
        let constants = IndexMap::new();
        let config = IndexMap::new();
        let mut ctx = create_test_context(InterpValue::Scalar(0.0), 0, &signals, &members, &constants, &config);

        // Vec3(1.0, 2.0, 3.0)
        let expr = CompiledExpr::Call {
            function: "Vec3".to_string(),
            args: vec![
                CompiledExpr::Literal(1.0),
                CompiledExpr::Literal(2.0),
                CompiledExpr::Literal(3.0),
            ],
        };
        assert_eq!(interpret_expr(&expr, &mut ctx).as_vec3(), [1.0, 2.0, 3.0]);
    }
}
