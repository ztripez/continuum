//! Abstract Syntax Tree (AST) for the Continuum DSL.
//!
//! This module defines the typed representation of parsed DSL source code.
//! The AST preserves source spans for all nodes, enabling precise error
//! reporting and IDE hover text.
//!
//! # Structure
//!
//! A [`CompilationUnit`] contains a list of top-level [`Item`]s, which include:
//! - **Configuration**: [`ConstBlock`], [`ConfigBlock`] for compile-time values
//! - **Types**: [`TypeDef`] for custom type declarations
//! - **Functions**: [`FnDef`] for pure inlined expressions
//! - **Simulation Structure**: [`StrataDef`], [`EraDef`] for time organization
//! - **Signals**: [`SignalDef`] for authoritative state
//! - **Member Signals**: [`MemberDef`] for per-entity authoritative state
//! - **Observation**: [`FieldDef`] for derived measurements
//! - **Operators**: [`OperatorDef`] for phase-specific logic
//! - **Events**: [`ImpulseDef`], [`FractureDef`], [`ChronicleDef`]
//! - **Collections**: [`EntityDef`] for index spaces
//!
//! # Module Organization
//!
//! The AST is split into three modules:
//! - [`expr`] - Expression types for computations and data flow
//! - [`items`] - Top-level definition types (signals, fields, etc.)
//! - This module - Core types (Span, Spanned, Path) and the Item enum
//!
//! # Span Tracking
//!
//! All nodes are wrapped in [`Spanned<T>`] which associates the AST node
//! with its byte range in the source file. This enables:
//! - Precise error messages pointing to exact source locations
//! - IDE features like go-to-definition and hover documentation
//! - Source mapping for compiled IR
//!
//! # Example
//!
//! ```ignore
//! use continuum_dsl::{parse, ast::*};
//!
//! let src = r#"
//!     signal.terra.temperature {
//!         : Scalar<K, 50..1000>
//!         : strata(terra.thermal)
//!         resolve { prev + 0.1 }
//!     }
//! "#;
//!
//! let (unit, errors) = parse(src);
//! let unit = unit.unwrap();
//! for item in &unit.items {
//!     if let Item::SignalDef(sig) = &item.node {
//!         println!("Signal: {}", sig.path.node);
//!     }
//! }
//! ```

pub mod expr;
pub mod items;
pub mod visitor;

use std::ops::Range as StdRange;

// Re-export all types for convenience
pub use expr::*;
pub use items::*;
pub use visitor::{
    AstTransformer, AstVisitor, ExprVisitor, SpannedExprVisitor, uses_dt_raw, walk_ast_expr,
    walk_expr, walk_spanned_expr,
};

/// Source span representing a byte range in the source file.
///
/// Used for error reporting and source mapping.
pub type Span = StdRange<usize>;

/// A spanned AST node that associates a value with its source location.
///
/// All significant AST nodes are wrapped in `Spanned` to preserve their
/// position in the source file for error reporting and IDE features.
///
/// # Example
///
/// ```ignore
/// let path = Spanned::new(
///     Path::new(vec!["terra".into(), "temperature".into()]),
///     0..18
/// );
/// assert_eq!(path.node.to_string(), "terra.temperature");
/// assert_eq!(path.span, 0..18);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    /// The wrapped AST node.
    pub node: T,
    /// Byte range in source (start..end).
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Creates a new spanned node.
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

/// A complete DSL compilation unit representing a parsed source file.
///
/// Contains all top-level items defined in the source. Multiple compilation
/// units from different files are merged during world loading.
#[derive(Debug, Clone, Default)]
pub struct CompilationUnit {
    /// Module-level documentation from `//!` comments at the top of the file.
    pub module_doc: Option<String>,
    /// All top-level items in declaration order.
    pub items: Vec<Spanned<Item>>,
}

/// Top-level items that can appear in DSL source files.
///
/// Each variant corresponds to a distinct declaration type in the DSL.
/// Items are processed in a specific order during compilation regardless
/// of their declaration order in source.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// World manifest definition: `world.terra { ... }`.
    WorldDef(WorldDef),
    /// Compile-time constant definitions: `const { physics.g: 9.81 }`.
    ConstBlock(ConstBlock),
    /// Runtime configuration values: `config { thermal.tau: 1000.0 }`.
    ConfigBlock(ConfigBlock),
    /// Custom type declaration: `type Vec2 { x: Scalar<m>, y: Scalar<m> }`.
    TypeDef(TypeDef),
    /// User-defined function: `fn.math.lerp(a, b, t) { a + (b - a) * t }`.
    FnDef(FnDef),
    /// Time stratum definition: `strata.terra { : stride(10) }`.
    StrataDef(StrataDef),
    /// Era definition: `era.main { : initial : dt(1 <yr>) }`.
    EraDef(EraDef),
    /// Signal (authoritative state): `signal.terra.temp { resolve { prev } }`.
    SignalDef(SignalDef),
    /// Field (derived measurement): `field.terra.surface { measure { ... } }`.
    FieldDef(FieldDef),
    /// Operator (phase logic): `operator.terra.diffuse { collect { ... } }`.
    OperatorDef(OperatorDef),
    /// Impulse (external event): `impulse.stellar.flare { apply { ... } }`.
    ImpulseDef(ImpulseDef),
    /// Fracture (tension detector): `fracture.terra.quake { when { ... } }`.
    FractureDef(FractureDef),
    /// Chronicle (observer): `chronicle.stellar.events { observe { ... } }`.
    ChronicleDef(ChronicleDef),
    /// Entity (index space): `entity.stellar.moon { : count(config.n) }`.
    EntityDef(EntityDef),
    /// Member signal (per-entity state): `member.stellar.moon.mass { resolve { ... } }`.
    MemberDef(MemberDef),
    /// Analyzer (post-hoc analysis): `analyzer.terra.hypsometric { compute { ... } }`.
    AnalyzerDef(AnalyzerDef),
}

pub use continuum_foundation::Path;

// === Types ===

/// Type expression representing a value's shape and constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    /// Primitive type declaration resolved via the registry.
    Primitive(PrimitiveTypeExpr),
    /// Reference to a named type: `OrbitalElements`.
    Named(String),
}

/// Primitive type expression with registry-driven parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimitiveTypeExpr {
    /// Primitive type identifier.
    pub id: continuum_foundation::PrimitiveTypeId,
    /// Supplied parameter values.
    pub params: Vec<PrimitiveParamValue>,
    /// Mathematical constraints on the tensor.
    pub constraints: Vec<TensorConstraint>,
    /// Aggregate constraints on sequence elements.
    pub seq_constraints: Vec<SeqConstraint>,
}

/// Parameter value for a primitive type declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveParamValue {
    Unit(String),
    Range(Range),
    Magnitude(Range),
    Rows(u8),
    Cols(u8),
    Width(u32),
    Height(u32),
    ElementType(Box<TypeExpr>),
}

impl PrimitiveParamValue {
    /// Returns the param kind for this value.
    pub fn kind(&self) -> continuum_foundation::PrimitiveParamKind {
        use continuum_foundation::PrimitiveParamKind;

        match self {
            PrimitiveParamValue::Unit(_) => PrimitiveParamKind::Unit,
            PrimitiveParamValue::Range(_) => PrimitiveParamKind::Range,
            PrimitiveParamValue::Magnitude(_) => PrimitiveParamKind::Magnitude,
            PrimitiveParamValue::Rows(_) => PrimitiveParamKind::Rows,
            PrimitiveParamValue::Cols(_) => PrimitiveParamKind::Cols,
            PrimitiveParamValue::Width(_) => PrimitiveParamKind::Width,
            PrimitiveParamValue::Height(_) => PrimitiveParamKind::Height,
            PrimitiveParamValue::ElementType(_) => PrimitiveParamKind::ElementType,
        }
    }
}

/// Numeric range for value bounds validation.
///
/// Used in type expressions to constrain valid values.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    /// Minimum allowed value (inclusive).
    pub min: f64,
    /// Maximum allowed value (inclusive).
    pub max: f64,
}

/// Constraints for tensor types.
///
/// These constraints enforce mathematical properties of matrix values.
///
/// # Examples
///
/// ```text
/// : Tensor<3,3,Pa>
///   : symmetric
///   : positive_definite
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorConstraint {
    /// Matrix must be symmetric (A = A^T).
    Symmetric,
    /// Matrix must be positive definite (all eigenvalues > 0).
    PositiveDefinite,
}

/// Constraints for sequence types.
///
/// These constraints enforce aggregate properties over sequence elements.
///
/// # Examples
///
/// ```text
/// : Seq<Scalar<kg>>
///   : each(1e20..1e28)
///   : sum(1e25..1e30)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum SeqConstraint {
    /// Each element must be within the given range.
    Each(Range),
    /// The sum of all elements must be within the given range.
    Sum(Range),
}
