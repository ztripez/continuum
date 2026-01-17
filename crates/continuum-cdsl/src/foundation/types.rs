//! Type system for the Continuum DSL
//!
//! The type system distinguishes:
//! - **Kernel types** — numeric types with shape, units, and bounds
//! - **User types** — product types (structs) defined in DSL
//! - **Bool** — distinct from Scalar for logical operations
//! - **Unit** — void type for side effects (emit, spawn, log)
//! - **Seq<T>** — intermediate collection type (must be consumed by aggregates)
//!
//! # Examples
//!
//! ```
//! # use continuum_cdsl::foundation::types::*;
//! # use continuum_foundation::TypeId;
//! // Bool type
//! let bool_ty = Type::Bool;
//! assert!(bool_ty.is_bool());
//!
//! // Unit type (for effects)
//! let unit_ty = Type::Unit;
//! assert!(unit_ty.is_unit());
//!
//! // User-defined type
//! let vec3_id = TypeId::from("Vec3");
//! let user_ty = Type::User(vec3_id);
//! assert!(user_ty.is_user());
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

use super::{Path, Shape, Unit};
use continuum_foundation::TypeId;

/// A type in the Continuum type system.
///
/// Types are used for compile-time validation and code generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    /// Numeric kernel type (shape + unit + optional bounds)
    Kernel(KernelType),

    /// User-defined product type (struct)
    User(UserTypeId),

    /// Boolean type (true/false)
    ///
    /// Distinct from Scalar for type safety.
    Bool,

    /// Unit type (void)
    ///
    /// Result of side effects: emit, spawn, destroy, log.
    /// Only valid at statement position.
    Unit,

    /// Sequence type (intermediate only)
    ///
    /// Produced by `map`, consumed by aggregates.
    /// Cannot be stored, returned, or used in let bindings
    /// except when immediately consumed.
    Seq(Box<Type>),
}

/// Kernel type — numeric type with physics.
///
/// Combines geometric structure (Shape), physical dimensions (Unit),
/// and optional value constraints (Bounds).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelType {
    /// Geometric structure
    pub shape: Shape,
    /// Physical dimensions
    pub unit: Unit,
    /// Value constraints (None = unbounded)
    pub bounds: Option<Bounds>,
}

/// Value constraints for kernel types.
///
/// Bounds are compile-time tracked and runtime-validated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    /// Minimum value (None = unbounded below)
    pub min: Option<f64>,
    /// Maximum value (None = unbounded above)
    pub max: Option<f64>,
}

/// User-defined product type.
///
/// These are struct-like types defined in DSL with named fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserType {
    /// Unique identifier for this type
    pub id: UserTypeId,
    /// Type name (path)
    pub name: Path,
    /// Named fields with their types
    pub fields: Vec<(String, Type)>,
}

/// Type alias for user type IDs (re-exported from foundation)
pub type UserTypeId = TypeId;

impl Type {
    /// Check if this is a kernel type.
    pub fn is_kernel(&self) -> bool {
        matches!(self, Type::Kernel(_))
    }

    /// Check if this is a user type.
    pub fn is_user(&self) -> bool {
        matches!(self, Type::User(_))
    }

    /// Check if this is Bool.
    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool)
    }

    /// Check if this is Unit.
    pub fn is_unit(&self) -> bool {
        matches!(self, Type::Unit)
    }

    /// Check if this is a Seq type.
    pub fn is_seq(&self) -> bool {
        matches!(self, Type::Seq(_))
    }

    /// Get the inner type of a Seq, if this is a Seq.
    pub fn seq_inner(&self) -> Option<&Type> {
        match self {
            Type::Seq(inner) => Some(inner),
            _ => None,
        }
    }

    /// Create a Bool type.
    pub const fn bool() -> Self {
        Type::Bool
    }

    /// Create a Unit type.
    pub const fn unit() -> Self {
        Type::Unit
    }

    /// Create a Seq type wrapping another type.
    pub fn seq(inner: Type) -> Self {
        Type::Seq(Box::new(inner))
    }

    /// Create a user type reference.
    pub fn user(id: UserTypeId) -> Self {
        Type::User(id)
    }

    /// Create a kernel type.
    pub fn kernel(shape: Shape, unit: Unit, bounds: Option<Bounds>) -> Self {
        Type::Kernel(KernelType {
            shape,
            unit,
            bounds,
        })
    }
}

impl KernelType {
    /// Create a new kernel type.
    pub fn new(shape: Shape, unit: Unit, bounds: Option<Bounds>) -> Self {
        Self {
            shape,
            unit,
            bounds,
        }
    }

    /// Create a dimensionless scalar.
    pub fn dimensionless_scalar() -> Self {
        Self::new(Shape::Scalar, Unit::DIMENSIONLESS, None)
    }

    /// Get the shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the unit.
    pub fn unit(&self) -> &Unit {
        &self.unit
    }

    /// Get the bounds.
    pub fn bounds(&self) -> Option<&Bounds> {
        self.bounds.as_ref()
    }
}

impl Bounds {
    /// Create unbounded constraints.
    pub const fn unbounded() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// Create bounds with both min and max.
    pub const fn range(min: f64, max: f64) -> Self {
        Self {
            min: Some(min),
            max: Some(max),
        }
    }

    /// Create bounds with only minimum.
    pub const fn min(min: f64) -> Self {
        Self {
            min: Some(min),
            max: None,
        }
    }

    /// Create bounds with only maximum.
    pub const fn max(max: f64) -> Self {
        Self {
            min: None,
            max: Some(max),
        }
    }

    /// Check if a value is within bounds.
    pub fn contains(&self, value: f64) -> bool {
        if let Some(min) = self.min {
            if value < min {
                return false;
            }
        }
        if let Some(max) = self.max {
            if value > max {
                return false;
            }
        }
        true
    }

    /// Check if bounds are unbounded.
    pub fn is_unbounded(&self) -> bool {
        self.min.is_none() && self.max.is_none()
    }
}

impl UserType {
    /// Create a new user type.
    pub fn new(id: UserTypeId, name: Path, fields: Vec<(String, Type)>) -> Self {
        Self { id, name, fields }
    }

    /// Get the type ID.
    pub fn id(&self) -> &UserTypeId {
        &self.id
    }

    /// Get the type name.
    pub fn name(&self) -> &Path {
        &self.name
    }

    /// Get the fields.
    pub fn fields(&self) -> &[(String, Type)] {
        &self.fields
    }

    /// Look up a field by name.
    pub fn field(&self, name: &str) -> Option<&Type> {
        self.fields
            .iter()
            .find(|(field_name, _)| field_name == name)
            .map(|(_, ty)| ty)
    }

    /// Get the number of fields.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Kernel(kt) => write!(f, "{}", kt),
            Type::User(id) => write!(f, "{}", id),
            Type::Bool => write!(f, "Bool"),
            Type::Unit => write!(f, "Unit"),
            Type::Seq(inner) => write!(f, "Seq<{}>", inner),
        }
    }
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}<{}>", self.shape, self.unit)?;
        if let Some(bounds) = &self.bounds {
            write!(f, " {}", bounds)?;
        }
        Ok(())
    }
}

impl fmt::Display for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.min, self.max) {
            (Some(min), Some(max)) => write!(f, "[{}..{}]", min, max),
            (Some(min), None) => write!(f, "[{}..]", min),
            (None, Some(max)) => write!(f, "[..{}]", max),
            (None, None) => write!(f, "unbounded"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_constructors() {
        let bool_ty = Type::bool();
        assert!(bool_ty.is_bool());

        let unit_ty = Type::unit();
        assert!(unit_ty.is_unit());

        let seq_ty = Type::seq(Type::Bool);
        assert!(seq_ty.is_seq());
        assert_eq!(seq_ty.seq_inner(), Some(&Type::Bool));
    }

    #[test]
    fn test_bounds() {
        let unbounded = Bounds::unbounded();
        assert!(unbounded.is_unbounded());
        assert!(unbounded.contains(f64::INFINITY));
        assert!(unbounded.contains(f64::NEG_INFINITY));

        let range = Bounds::range(0.0, 1.0);
        assert!(range.contains(0.5));
        assert!(!range.contains(-0.1));
        assert!(!range.contains(1.1));

        let min_only = Bounds::min(0.0);
        assert!(min_only.contains(100.0));
        assert!(!min_only.contains(-1.0));
    }

    #[test]
    fn test_user_type() {
        let fields = vec![("x".to_string(), Type::Bool), ("y".to_string(), Type::Unit)];
        let id = TypeId::from("TestType");
        let user_ty = UserType::new(id, Path::from("TestType"), fields);

        assert_eq!(user_ty.field_count(), 2);
        assert_eq!(user_ty.field("x"), Some(&Type::Bool));
        assert_eq!(user_ty.field("y"), Some(&Type::Unit));
        assert_eq!(user_ty.field("z"), None);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(Type::Bool.to_string(), "Bool");
        assert_eq!(Type::Unit.to_string(), "Unit");

        let seq = Type::seq(Type::Bool);
        assert_eq!(seq.to_string(), "Seq<Bool>");
    }

    #[test]
    fn test_bounds_display() {
        assert_eq!(Bounds::unbounded().to_string(), "unbounded");
        assert_eq!(Bounds::range(0.0, 1.0).to_string(), "[0..1]");
        assert_eq!(Bounds::min(0.0).to_string(), "[0..]");
        assert_eq!(Bounds::max(100.0).to_string(), "[..100]");
    }
}
