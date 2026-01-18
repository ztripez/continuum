//! Declaration AST types for CDSL top-level constructs.
//!
//! This module defines the AST types that result from parsing top-level
//! declarations in CDSL source files. These types represent the parsed form
//! before name resolution, type checking, and lowering to IR.
//!
//! # Design
//!
//! - **Declaration** - Thin wrapper enum for all top-level declaration types
//! - **Stmt** - Statement types for block bodies (let, assignment, expr)
//! - **BlockBody** - Either expression or statement list
//! - Structural declarations: TypeDecl, WorldDecl, EraDecl, etc.
//! - Entry types: ConstEntry, ConfigEntry (with type annotations and spans)
//!
//! # Architecture Alignment
//!
//! Per compiler manifesto:
//! - Parser produces Node<I> directly (no intermediate parsed types)
//! - All structural metadata preserved (spans, doc comments)
//! - Types explicit everywhere (no Option<TypeExpr> for const/config)

use crate::ast::node::{Entity, Node, Stratum};
use crate::ast::untyped::{Expr, TypeExpr};
use crate::foundation::{EntityId, Path, Span};

/// Top-level declaration variants returned by parser.
///
/// This is a thin wrapper that allows the parser to return different
/// declaration types from a single `declarations_parser()` function.
/// Each variant contains the fully-constructed declaration with all
/// metadata preserved.
#[derive(Debug, Clone)]
pub enum Declaration {
    /// Global primitive (signal, field, operator, impulse, fracture, chronicle)
    Node(Node<()>),

    /// Per-entity primitive (member signal/field/etc)
    Member(Node<EntityId>),

    /// Entity declaration (index space)
    Entity(Entity),

    /// Stratum declaration (execution lane)
    Stratum(Stratum),

    /// Era declaration (execution regime)
    Era(EraDecl),

    /// Type declaration (custom struct)
    Type(TypeDecl),

    /// World block (metadata + warmup policy)
    World(WorldDecl),

    /// Top-level const block
    Const(Vec<ConstEntry>),

    /// Top-level config block
    Config(Vec<ConfigEntry>),
}

/// Statement in a block body.
///
/// Statements appear in blocks with effect capabilities (Collect, Apply, Emit).
/// Blocks in pure phases (Resolve, Measure, Assert) use expression bodies.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Let binding: `let x = expr`
    ///
    /// Introduces a local variable visible in subsequent statements.
    /// Unlike `let...in` expressions, this doesn't have a body scope.
    Let {
        /// Variable name
        name: String,
        /// Value expression
        value: Expr,
        /// Source location
        span: Span,
    },

    /// Signal assignment: `signal.path <- expr`
    ///
    /// Emits a value to a signal's input accumulator.
    /// Valid only in blocks with Emit capability (Collect, Apply, WarmUp).
    SignalAssign {
        /// Target signal path
        target: Path,
        /// Value to emit
        value: Expr,
        /// Source location
        span: Span,
    },

    /// Field assignment: `field.path <- position, value`
    ///
    /// Emits a positioned sample to a field.
    /// Valid only in Measure phase with Emit capability.
    FieldAssign {
        /// Target field path
        target: Path,
        /// Position expression (Vec2/Vec3 or other spatial coordinate)
        position: Expr,
        /// Value expression (the field data at this position)
        value: Expr,
        /// Source location
        span: Span,
    },

    /// Expression statement
    ///
    /// An expression evaluated for its side effects (usually a function call).
    Expr(Expr),
}

/// Block body - either single expression or statement list.
///
/// The body kind is determined by the block's phase capabilities:
/// - Pure phases (Resolve, Iterate, Assert): Expression
/// - Effect phases (Collect, Apply, Emit): Statements
#[derive(Debug, Clone, PartialEq)]
pub enum BlockBody {
    /// Single expression (pure phases)
    ///
    /// Used in: resolve, iterate, measure (when simple), assert
    Expression(Expr),

    /// Statement list (effect phases)
    ///
    /// Used in: collect, apply, emit, when
    Statements(Vec<Stmt>),
}

/// Attribute parsed from source.
///
/// Attributes are parsed generically as `: name(args)` or `: name`.
/// Validation of attribute names and argument types happens in the analyzer,
/// not the parser. This keeps the parser simple and allows for future
/// attribute additions without parser changes.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    /// Attribute name (e.g., "title", "strata", "dt")
    pub name: String,

    /// Attribute arguments (empty for flag attributes like `:initial`)
    pub args: Vec<Expr>,

    /// Source location
    pub span: Span,
}

/// Custom struct type declaration.
///
/// Defines a user type that can be used in type annotations.
/// Example:
/// ```cdsl
/// type ImpactEvent {
///     mass: Scalar<kg>
///     velocity: Vec3<m/s>
///     location: Vec2<rad>
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TypeDecl {
    /// Type name (unqualified identifier)
    pub name: String,

    /// Field declarations
    pub fields: Vec<TypeField>,

    /// Source location
    pub span: Span,

    /// Doc comment (captured from `///` comments preceding declaration)
    pub doc: Option<String>,
}

/// Field in a custom type declaration.
#[derive(Debug, Clone)]
pub struct TypeField {
    /// Field name
    pub name: String,

    /// Field type
    pub type_expr: TypeExpr,

    /// Source location
    pub span: Span,
}

/// World declaration with metadata and warmup policy.
///
/// Defines world-level configuration that affects the entire simulation.
/// Example:
/// ```cdsl
/// world terra {
///     :title("Terra Simulation")
///     :version("1.0.0")
///     warmup {
///         :max_iterations(1000)
///         :converged(maths.max_delta(signals) < 1e-6)
///         :on_timeout(fail)
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct WorldDecl {
    /// World path
    pub path: Path,

    /// Human-readable title
    pub title: Option<String>,

    /// Version string
    pub version: Option<String>,

    /// Warmup policy for iterative equilibration
    pub warmup: Option<WarmupPolicy>,

    /// Source location
    pub span: Span,

    /// Doc comment
    pub doc: Option<String>,
}

/// Warmup policy for iterative equilibration.
///
/// Per compiler manifesto, this must use expression-based convergence
/// predicate, not a simple float threshold.
#[derive(Debug, Clone)]
pub struct WarmupPolicy {
    /// Expression that evaluates to true when converged.
    ///
    /// Example: `maths.max_delta(signals) < 1e-6`
    /// This is a typed expression, not a constant.
    pub converged: Expr,

    /// Maximum iterations before timeout (required).
    pub max_iterations: u32,

    /// Behavior on timeout (required, defaults to Fail).
    pub on_timeout: WarmupTimeout,

    /// Source location
    pub span: Span,
}

/// Behavior when warmup times out without converging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarmupTimeout {
    /// Emit error diagnostic and abort compilation
    Fail,

    /// Continue with unconverged state (emit warning)
    Continue,
}

impl Default for WarmupTimeout {
    fn default() -> Self {
        Self::Fail
    }
}

/// Const entry with full metadata.
///
/// Per architect feedback: type annotations must be required for const entries.
/// No `Option<TypeExpr>` - types are explicit.
#[derive(Debug, Clone)]
pub struct ConstEntry {
    /// Constant path
    pub path: Path,

    /// Constant value
    pub value: Expr,

    /// Type annotation (required)
    pub type_expr: TypeExpr,

    /// Source location
    pub span: Span,

    /// Doc comment
    pub doc: Option<String>,
}

/// Config entry with full metadata.
///
/// Config entries have required types and optional defaults.
#[derive(Debug, Clone)]
pub struct ConfigEntry {
    /// Config path
    pub path: Path,

    /// Default value (optional - can be overridden by scenario)
    pub default: Option<Expr>,

    /// Type annotation (required)
    pub type_expr: TypeExpr,

    /// Source location
    pub span: Span,

    /// Doc comment
    pub doc: Option<String>,
}

/// Era declaration (parsed form).
///
/// Defines an execution regime with dt, stratum policies, and transitions.
/// Example:
/// ```cdsl
/// era formation {
///     :initial
///     :dt(1_000_000<yr>)
///     strata {
///         tectonics: active
///         climate: gated
///     }
///     transition stable when {
///         signal.mantle.temperature < 1500<K>
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct EraDecl {
    /// Era path
    pub path: Path,

    /// Base timestep expression
    ///
    /// Can be constant or computed. If None, inherits from parent era or world default.
    pub dt: Option<Expr>,

    /// Is this the initial era (simulation starts here)?
    pub is_initial: bool,

    /// Is this a terminal era (simulation can end here)?
    pub is_terminal: bool,

    /// Stratum activation policies for this era
    pub strata_policy: Vec<StratumPolicyEntry>,

    /// Transitions to other eras
    pub transitions: Vec<TransitionDecl>,

    /// Source location
    pub span: Span,

    /// Doc comment
    pub doc: Option<String>,
}

/// Stratum policy entry in an era.
///
/// Defines how a stratum behaves in this era (active, gated)
/// and optionally overrides its cadence.
#[derive(Debug, Clone)]
pub struct StratumPolicyEntry {
    /// Stratum path
    pub stratum: Path,

    /// Activation state
    pub state: StratumState,

    /// Optional stride override for this era
    ///
    /// If None, uses the stratum's declared cadence.
    pub stride: Option<u32>,

    /// Source location
    pub span: Span,
}

/// Stratum activation state within an era.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratumState {
    /// Stratum executes in this era
    Active,

    /// Stratum does not execute in this era
    Gated,
}

/// Transition declaration.
///
/// Defines a transition to another era when conditions are met.
/// Example:
/// ```cdsl
/// transition stable when {
///     signal.temp < 1000<K>
///     signal.time > 1e9<s>
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TransitionDecl {
    /// Target era path
    pub target: Path,

    /// Transition conditions (all must be true)
    ///
    /// Each condition is a boolean expression. Transition fires when
    /// all conditions evaluate to true.
    pub conditions: Vec<Expr>,

    /// Source location
    pub span: Span,
}

/// Warmup block within a signal declaration.
///
/// Contains warmup-specific attributes and the iterate expression.
/// Example:
/// ```cdsl
/// warmup {
///     :iterations(100)
///     :convergence(1e-6)
///     iterate { prev * 0.9 }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct WarmupBlock {
    /// Warmup attributes (iterations, convergence, etc.)
    pub attrs: Vec<Attribute>,

    /// Iterate expression (warmup logic)
    pub iterate: Expr,

    /// Source location
    pub span: Span,
}

/// When block for fractures.
///
/// Contains conditions that trigger fracture detection.
/// Example:
/// ```cdsl
/// when {
///     signal.temp > 350<K>
///     signal.pressure > 100<atm>
/// }
/// ```
#[derive(Debug, Clone)]
pub struct WhenBlock {
    /// Condition expressions (all must be true for when block to fire)
    pub conditions: Vec<Expr>,

    /// Source location
    pub span: Span,
}

/// Observe block for chronicles.
///
/// Contains when/emit pairs for pattern detection.
/// Example:
/// ```cdsl
/// observe {
///     when signal.diversity < -0.5 {
///         emit event.extinction { severity: 0.8 }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ObserveBlock {
    /// When clauses with associated emit blocks
    pub when_clauses: Vec<ObserveWhen>,

    /// Source location
    pub span: Span,
}

/// When clause within an observe block.
#[derive(Debug, Clone)]
pub struct ObserveWhen {
    /// Condition expression
    pub condition: Expr,

    /// Emit statements when condition is true
    pub emit_block: Vec<Stmt>,

    /// Source location
    pub span: Span,
}
