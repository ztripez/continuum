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

use crate::ast::block::Stmt;
use crate::ast::node::{Entity, Node, Stratum};
use crate::ast::untyped::{Expr, TypeExpr};
use crate::foundation::{EntityId, Path, Span};

/// Top-level declaration variants returned by parser.
///
/// This is a thin wrapper that allows the parser to return different
/// declaration types from a single `declarations_parser()` function.
/// Each variant contains the fully-constructed declaration with all
/// metadata preserved.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Attribute parsed from source.
///
/// Attributes are parsed generically as `: name(args)` or `: name`.
/// Validation of attribute names and argument types happens in the analyzer,
/// not the parser. This keeps the parser simple and allows for future
/// attribute additions without parser changes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldDecl {
    /// World path
    pub path: Path,

    /// Human-readable title
    pub title: Option<String>,

    /// Version string
    pub version: Option<String>,

    /// Raw warmup attributes from world declaration
    ///
    /// Semantic analysis validates these and builds a WarmupPolicy.
    /// Parser preserves raw syntax; analyzer validates and interprets.
    pub warmup: Option<RawWarmupPolicy>,

    /// Parsed attributes from source
    ///
    /// Raw attributes for semantic analysis to interpret.
    /// Common attributes:
    /// - :title("name") - sets human-readable title
    /// - :version("1.0.0") - sets version string
    ///
    /// Semantic analysis extracts and validates these.
    pub attributes: Vec<Attribute>,

    /// Source location
    pub span: Span,

    /// Doc comment
    pub doc: Option<String>,

    /// Auto-generate debug fields for all signals
    pub debug: bool,
}

/// Warmup policy for iterative equilibration.
///
/// Per compiler manifesto, this must use expression-based convergence
/// predicate, not a simple float threshold.
///
/// **Parser/Semantic Boundary Issue:** This is currently built by the parser
/// from raw attributes. Should be moved to semantic analysis phase.
/// Fields are Optional to avoid silent defaults when attributes are missing/invalid.
/// Semantic analysis validates required fields and applies proper defaults.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarmupPolicy {
    /// Expression that evaluates to true when converged.
    ///
    /// Example: `maths.max_delta(signals) < 1e-6`
    /// This is a typed expression, not a constant.
    /// None if `:converged(...)` attribute is missing or invalid.
    pub converged: Option<Expr>,

    /// Maximum iterations before timeout.
    ///
    /// None if `:max_iterations(...)` attribute is missing or invalid.
    pub max_iterations: Option<u32>,

    /// Behavior on timeout.
    ///
    /// None if `:on_timeout(...)` attribute is missing or invalid.
    pub on_timeout: Option<WarmupTimeout>,

    /// Source location
    pub span: Span,
}

/// Raw warmup attributes from world declaration.
///
/// Parser extracts these from `world { warmup { :attr(...) } }` blocks
/// and stores them as raw attributes for semantic analysis to validate
/// and convert to WarmupPolicy.
///
/// This preserves the parser/semantic boundary: parser handles syntax,
/// semantic analysis validates and interprets.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RawWarmupPolicy {
    /// Raw attributes from warmup block
    pub attributes: Vec<Attribute>,

    /// Source location
    pub span: Span,
}

/// Behavior when warmup times out without converging.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EraDecl {
    /// Era path
    pub path: Path,

    /// Base timestep expression.
    ///
    /// Derived from `:dt(...)` attributes during parsing.
    pub dt: Option<Expr>,

    /// Stratum activation policies for this era
    pub strata_policy: Vec<StratumPolicyEntry>,

    /// Transitions to other eras
    pub transitions: Vec<TransitionDecl>,

    /// Parsed attributes from source
    ///
    /// Raw attributes for semantic analysis to interpret.
    /// Common attributes:
    /// - :initial - marks this as the starting era
    /// - :terminal - marks this as a valid end state
    ///
    /// Semantic analysis extracts and validates these.
    pub attributes: Vec<Attribute>,

    /// Source location
    pub span: Span,

    /// Doc comment
    pub doc: Option<String>,
}

/// Stratum policy entry in an era.
///
/// Defines how a stratum behaves in this era (active, gated)
/// and optionally overrides its cadence.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StratumPolicyEntry {
    /// Stratum path
    pub stratum: Path,

    /// Activation state identifier (raw string from parser)
    ///
    /// Parser preserves the raw identifier string (e.g., "active", "gated").
    /// Semantic analysis interprets this into StratumState enum.
    pub state_name: String,

    /// Optional stride override for this era
    ///
    /// If None, uses the stratum's declared cadence.
    pub stride: Option<u32>,

    /// Source location
    pub span: Span,
}

/// Stratum activation state within an era.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObserveBlock {
    /// When clauses with associated emit blocks
    pub when_clauses: Vec<ObserveWhen>,

    /// Source location
    pub span: Span,
}

/// When clause within an observe block.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObserveWhen {
    /// Condition expression
    pub condition: Expr,

    /// Emit statements when condition is true
    pub emit_block: Vec<Stmt>,

    /// Source location
    pub span: Span,
}
