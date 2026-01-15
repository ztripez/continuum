//! Layered Expression Architecture for Continuum IR
//!
//! This module implements a clean separation between:
//! - **ScalarExpr**: Pure mathematical/logical operations that compile to VM bytecode
//! - **StreamOp**: Collection and spatial operations that work on entity streams
//!
//! This separation enables:
//! - Simpler bytecode compilation (only ScalarExpr needs VM mapping)
//! - Better analysis (spatial ops only in Measure phase)
//! - Cleaner interpretation and optimization

use super::{AggregateOp, BinaryOp, UnaryOp};
use continuum_foundation::{EntityId, InstanceId, SignalId};

/// A layered expression that combines scalar math with stream operations
#[derive(Debug, Clone)]
pub enum LayeredExpr {
    /// Pure mathematical/logical expression
    Scalar(ScalarExpr),
    /// Collection/spatial stream operation  
    Stream(StreamOp),
}

/// Pure mathematical and logical expressions that compile directly to VM bytecode
///
/// These expressions represent:
/// - Arithmetic: +, -, *, /, ^
/// - Logic: ==, !=, <, <=, >, >=, &&, ||, !
/// - Control flow: if-then-else, let bindings
/// - Value access: constants, signals, fields
/// - Function calls: user functions and kernel operations
#[derive(Debug, Clone)]
pub enum ScalarExpr {
    // ===== LITERALS & VALUES =====
    /// Literal numeric value
    Literal(f64),
    /// Previous signal value (for time integration)
    Prev,
    /// Raw time delta (dt_raw)
    DtRaw,
    /// Current simulation time
    SimTime,
    /// Collected impulse values
    Collected,

    // ===== REFERENCES =====
    /// Reference to a signal by ID
    Signal(SignalId),
    /// Reference to a constant by name
    Const(String),
    /// Reference to a config value by name
    Config(String),
    /// Reference to a let-bound local variable
    Local(String),
    /// Access to impulse payload
    Payload,
    /// Access to named field in impulse payload
    PayloadField(String),
    /// Access to named field of current entity instance
    SelfField(String),

    // ===== ARITHMETIC & LOGIC =====
    /// Binary operation (math or comparison)
    Binary {
        op: BinaryOp,
        left: Box<ScalarExpr>,
        right: Box<ScalarExpr>,
    },
    /// Unary operation (negation, logical not)
    Unary {
        op: UnaryOp,
        operand: Box<ScalarExpr>,
    },

    // ===== CONTROL FLOW =====
    /// Conditional expression
    If {
        condition: Box<ScalarExpr>,
        then_branch: Box<ScalarExpr>,
        else_branch: Box<ScalarExpr>,
    },
    /// Let binding with local scope
    Let {
        name: String,
        value: Box<ScalarExpr>,
        body: Box<ScalarExpr>,
    },

    // ===== FUNCTION CALLS =====
    /// User-defined function call
    Call {
        function: String,
        args: Vec<ScalarExpr>,
    },
    /// Engine kernel function call
    KernelCall {
        namespace: String,
        function: String,
        args: Vec<ScalarExpr>,
    },

    // ===== FIELD ACCESS =====
    /// Access named field of an object
    FieldAccess {
        object: Box<ScalarExpr>,
        field: String,
    },

    // ===== STREAM RESULT =====
    /// Result of a stream operation (bridges to StreamOp)
    Stream(Box<StreamOp>),
}

/// Stream operations that work on collections or spatial data
///
/// These operations:
/// - Work on entity collections or spatial queries
/// - Can only appear in the Measure phase (for spatial ops)
/// - Often produce scalar results via aggregation
/// - Use ScalarExpr for their logic/predicates
#[derive(Debug, Clone)]
pub enum StreamOp {
    // ===== ENTITY ACCESS =====
    /// Access a specific entity instance field
    EntityAccess {
        entity: EntityId,
        instance: InstanceId,
        field: String,
    },

    // ===== COLLECTION OPERATIONS =====
    /// Aggregate operation over entity collection
    Aggregate {
        op: AggregateOp,
        entity: EntityId,
        body: Box<ScalarExpr>,
    },
    /// Iterate over other instances of same entity type
    Other {
        entity: EntityId,
        body: Box<ScalarExpr>,
    },
    /// Iterate over pairs of entity instances
    Pairs {
        entity: EntityId,
        body: Box<ScalarExpr>,
    },
    /// Filter entity collection by predicate
    Filter {
        entity: EntityId,
        predicate: Box<ScalarExpr>,
        body: Box<ScalarExpr>,
    },
    /// Find first entity matching predicate
    First {
        entity: EntityId,
        predicate: Box<ScalarExpr>,
    },

    // ===== SPATIAL OPERATIONS =====
    /// Find nearest entity to a position
    Nearest {
        entity: EntityId,
        position: Box<ScalarExpr>,
    },
    /// Find entities within radius of position
    Within {
        entity: EntityId,
        position: Box<ScalarExpr>,
        radius: Box<ScalarExpr>,
        body: Box<ScalarExpr>,
    },

    // ===== SIGNAL EMISSION (FRACTURES) =====
    /// Emit signal value (fracture response)
    EmitSignal {
        target: SignalId,
        value: Box<ScalarExpr>,
    },
}

impl From<ScalarExpr> for LayeredExpr {
    fn from(expr: ScalarExpr) -> Self {
        LayeredExpr::Scalar(expr)
    }
}

impl From<StreamOp> for LayeredExpr {
    fn from(expr: StreamOp) -> Self {
        LayeredExpr::Stream(expr)
    }
}

impl LayeredExpr {
    /// Check if this expression is a pure scalar (no stream operations)
    pub fn is_pure_scalar(&self) -> bool {
        match self {
            LayeredExpr::Scalar(scalar) => scalar.is_pure_scalar(),
            LayeredExpr::Stream(_) => false,
        }
    }

    /// Check if this expression contains spatial operations (Nearest, Within)
    pub fn has_spatial_ops(&self) -> bool {
        match self {
            LayeredExpr::Scalar(scalar) => scalar.has_spatial_ops(),
            LayeredExpr::Stream(stream) => stream.has_spatial_ops(),
        }
    }

    /// Extract all signal dependencies from this expression
    pub fn signal_dependencies(&self) -> Vec<SignalId> {
        match self {
            LayeredExpr::Scalar(scalar) => scalar.signal_dependencies(),
            LayeredExpr::Stream(stream) => stream.signal_dependencies(),
        }
    }
}

impl ScalarExpr {
    /// Check if this scalar expression is purely mathematical (no stream operations)
    pub fn is_pure_scalar(&self) -> bool {
        use ScalarExpr::*;
        match self {
            // Pure values
            Literal(_) | Prev | DtRaw | SimTime | Collected => true,
            Signal(_) | Const(_) | Config(_) | Local(_) => true,
            Payload | PayloadField(_) | SelfField(_) => true,

            // Recursive cases
            Binary { left, right, .. } => left.is_pure_scalar() && right.is_pure_scalar(),
            Unary { operand, .. } => operand.is_pure_scalar(),
            If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                condition.is_pure_scalar()
                    && then_branch.is_pure_scalar()
                    && else_branch.is_pure_scalar()
            }
            Let { value, body, .. } => value.is_pure_scalar() && body.is_pure_scalar(),
            Call { args, .. } | KernelCall { args, .. } => {
                args.iter().all(|arg| arg.is_pure_scalar())
            }

            FieldAccess { object, .. } => object.is_pure_scalar(),

            // Contains stream operation
            Stream(_) => false,
        }
    }

    /// Check if this expression contains spatial operations
    pub fn has_spatial_ops(&self) -> bool {
        use ScalarExpr::*;
        match self {
            // Base cases - no spatial ops
            Literal(_) | Prev | DtRaw | SimTime | Collected => false,
            Signal(_) | Const(_) | Config(_) | Local(_) => false,
            Payload | PayloadField(_) | SelfField(_) => false,

            // Recursive cases
            Binary { left, right, .. } => left.has_spatial_ops() || right.has_spatial_ops(),
            Unary { operand, .. } => operand.has_spatial_ops(),
            If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                condition.has_spatial_ops()
                    || then_branch.has_spatial_ops()
                    || else_branch.has_spatial_ops()
            }
            Let { value, body, .. } => value.has_spatial_ops() || body.has_spatial_ops(),
            Call { args, .. } | KernelCall { args, .. } => {
                args.iter().any(|arg| arg.has_spatial_ops())
            }

            FieldAccess { object, .. } => object.has_spatial_ops(),

            // Check stream operations
            Stream(stream_op) => stream_op.has_spatial_ops(),
        }
    }

    /// Extract all signal dependencies
    pub fn signal_dependencies(&self) -> Vec<SignalId> {
        use ScalarExpr::*;
        let mut deps = Vec::new();

        match self {
            Signal(id) => deps.push(id.clone()),
            Binary { left, right, .. } => {
                deps.extend(left.signal_dependencies());
                deps.extend(right.signal_dependencies());
            }
            Unary { operand, .. } => deps.extend(operand.signal_dependencies()),
            If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                deps.extend(condition.signal_dependencies());
                deps.extend(then_branch.signal_dependencies());
                deps.extend(else_branch.signal_dependencies());
            }
            Let { value, body, .. } => {
                deps.extend(value.signal_dependencies());
                deps.extend(body.signal_dependencies());
            }
            Call { args, .. } | KernelCall { args, .. } => {
                for arg in args {
                    deps.extend(arg.signal_dependencies());
                }
            }

            FieldAccess { object, .. } => deps.extend(object.signal_dependencies()),
            Stream(stream_op) => deps.extend(stream_op.signal_dependencies()),

            // Base cases with no dependencies
            _ => {}
        }

        deps
    }
}

impl StreamOp {
    /// Check if this stream operation contains spatial operations
    pub fn has_spatial_ops(&self) -> bool {
        use StreamOp::*;
        match self {
            // Spatial operations
            Nearest { .. } => true,
            Within {
                position,
                radius,
                body,
                ..
            } => {
                // Within is spatial, but also check its components
                true || position.has_spatial_ops()
                    || radius.has_spatial_ops()
                    || body.has_spatial_ops()
            }

            // Collection operations with potential spatial content in body/predicate
            Aggregate { body, .. } => body.has_spatial_ops(),
            Other { body, .. } => body.has_spatial_ops(),
            Pairs { body, .. } => body.has_spatial_ops(),
            Filter {
                predicate, body, ..
            } => predicate.has_spatial_ops() || body.has_spatial_ops(),
            First { predicate, .. } => predicate.has_spatial_ops(),
            EmitSignal { value, .. } => value.has_spatial_ops(),

            // Entity access is not spatial
            EntityAccess { .. } => false,
        }
    }

    /// Extract all signal dependencies from this stream operation
    pub fn signal_dependencies(&self) -> Vec<SignalId> {
        use StreamOp::*;
        let mut deps = Vec::new();

        match self {
            EntityAccess { .. } => {} // No direct signal dependencies
            Aggregate { body, .. } => deps.extend(body.signal_dependencies()),
            Other { body, .. } => deps.extend(body.signal_dependencies()),
            Pairs { body, .. } => deps.extend(body.signal_dependencies()),
            Filter {
                predicate, body, ..
            } => {
                deps.extend(predicate.signal_dependencies());
                deps.extend(body.signal_dependencies());
            }
            First { predicate, .. } => deps.extend(predicate.signal_dependencies()),
            Nearest { position, .. } => deps.extend(position.signal_dependencies()),
            Within {
                position,
                radius,
                body,
                ..
            } => {
                deps.extend(position.signal_dependencies());
                deps.extend(radius.signal_dependencies());
                deps.extend(body.signal_dependencies());
            }
            EmitSignal { target, value, .. } => {
                deps.push(target.clone());
                deps.extend(value.signal_dependencies());
            }
        }

        deps
    }
}
