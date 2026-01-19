//! Type resolution for CDSL type and unit syntax.
//!
//! Translates parsed [`TypeExpr`](crate::ast::TypeExpr) and
//! [`UnitExpr`](crate::ast::UnitExpr) nodes into semantic [`Type`] and [`Unit`]
//! values used by later validation passes.
//!
//! # What This Pass Does
//!
//! 1. **TypeExpr → Type** - Resolves untyped type syntax to semantic types
//! 2. **UnitExpr → Unit** - Resolves unit syntax to dimensional units
//! 3. **User type lookup** - Resolves type names to TypeIds via [`TypeTable`]
//! 4. **Dimensional analysis** - Validates and computes unit arithmetic
//!
//! # What This Pass Does NOT Do
//!
//! - **No full type inference** - Complex bidirectional inference deferred to validation
//! - **No type checking** - Compatibility validation happens in later passes
//! - **No kernel call resolution** - Return types resolved during validation
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Validation
//!                                         ^^^^^^
//!                                      YOU ARE HERE
//! ```
//!
//! # Examples
//!
//! ```rust
//! use continuum_cdsl::resolve::types::{resolve_type_expr, TypeTable};
//! use continuum_cdsl::ast::{TypeExpr, UnitExpr};
//! use continuum_cdsl::foundation::Span;
//!
//! let table = TypeTable::new();
//! let span = Span::new(0, 0, 10, 1);
//!
//! // Resolve Vector<3, m>
//! let ty = TypeExpr::Vector {
//!     dim: 3,
//!     unit: Some(UnitExpr::Base("m".into())),
//! };
//! let resolved = resolve_type_expr(&ty, &table, span).unwrap();
//! assert!(resolved.is_kernel());
//! ```

use crate::ast::{TypeExpr, UnitExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{
    Path, Shape, Span, Type, Unit, UnitDimensions, UnitKind, UserType, UserTypeId,
};
use std::collections::HashMap;

/// Registry of user-defined types keyed by fully-qualified [`Path`](crate::foundation::Path).
///
/// `TypeTable` assigns stable [`UserTypeId`](crate::foundation::TypeId) values and supports
/// lookup by name during type resolution. The table does not perform validation;
/// it only stores declarations.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_cdsl::foundation::{Path, UserType, TypeId};
///
/// let mut table = TypeTable::new();
/// let path = Path::from("Vec3");
/// let type_id = TypeId::from("Vec3");
/// let user_type = UserType::new(type_id.clone(), path.clone(), vec![]);
/// table.register(user_type);
///
/// assert!(table.contains(&path));
/// assert_eq!(table.get_id(&path), Some(type_id));
/// ```
#[derive(Debug, Default)]
pub struct TypeTable {
    /// Map from type path to UserType definition
    types: HashMap<Path, UserType>,

    /// Map from type path to TypeId (for quick lookups)
    type_ids: HashMap<Path, UserTypeId>,
}

impl TypeTable {
    /// Create an empty type table
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a user-defined type
    pub fn register(&mut self, user_type: UserType) {
        let path = user_type.name().clone();
        let id = user_type.id().clone();
        self.types.insert(path.clone(), user_type);
        self.type_ids.insert(path, id);
    }

    /// Looks up a user type by [`Path`](crate::foundation::Path).
    ///
    /// # Parameters
    ///
    /// - `path`: Fully-qualified type name.
    ///
    /// # Returns
    ///
    /// The stored [`UserType`] if present; otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::types::TypeTable;
    /// use continuum_cdsl::foundation::{Path, UserType, TypeId};
    ///
    /// let mut table = TypeTable::new();
    /// let path = Path::from("Foo");
    /// table.register(UserType::new(TypeId::from("Foo"), path.clone(), vec![]));
    ///
    /// assert!(table.get(&path).is_some());
    /// ```
    pub fn get(&self, path: &Path) -> Option<&UserType> {
        self.types.get(path)
    }

    /// Looks up a user type identifier by [`Path`](crate::foundation::Path).
    ///
    /// # Parameters
    ///
    /// - `path`: Fully-qualified type name.
    ///
    /// # Returns
    ///
    /// The [`UserTypeId`](crate::foundation::TypeId) if present; otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::types::TypeTable;
    /// use continuum_cdsl::foundation::{Path, UserType, TypeId};
    ///
    /// let mut table = TypeTable::new();
    /// let path = Path::from("Foo");
    /// let id = TypeId::from("Foo");
    /// table.register(UserType::new(id.clone(), path.clone(), vec![]));
    ///
    /// assert_eq!(table.get_id(&path), Some(id));
    /// ```
    pub fn get_id(&self, path: &Path) -> Option<UserTypeId> {
        self.type_ids.get(path).cloned()
    }

    /// Tests whether a user type exists in the table.
    ///
    /// # Parameters
    ///
    /// - `path`: Fully-qualified type name.
    ///
    /// # Returns
    ///
    /// `true` if the type is registered, otherwise `false`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::types::TypeTable;
    /// use continuum_cdsl::foundation::{Path, UserType, TypeId};
    ///
    /// let mut table = TypeTable::new();
    /// let path = Path::from("Foo");
    /// table.register(UserType::new(TypeId::from("Foo"), path.clone(), vec![]));
    ///
    /// assert!(table.contains(&path));
    /// ```
    pub fn contains(&self, path: &Path) -> bool {
        self.types.contains_key(path)
    }

    /// Iterate over all registered user types.
    ///
    /// # Returns
    ///
    /// Iterator over references to all [`UserType`] definitions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::types::TypeTable;
    /// use continuum_cdsl::foundation::{Path, UserType, TypeId};
    ///
    /// let mut table = TypeTable::new();
    /// table.register(UserType::new(TypeId::from("Foo"), Path::from("Foo"), vec![]));
    ///
    /// assert_eq!(table.iter().count(), 1);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &UserType> {
        self.types.values()
    }

    /// Look up a user type by its [`UserTypeId`].
    ///
    /// # Parameters
    ///
    /// - `id`: The type identifier to look up.
    ///
    /// # Returns
    ///
    /// The [`UserType`] if present; otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::types::TypeTable;
    /// use continuum_cdsl::foundation::{Path, UserType, TypeId};
    ///
    /// let mut table = TypeTable::new();
    /// let id = TypeId::from("Foo");
    /// table.register(UserType::new(id.clone(), Path::from("Foo"), vec![]));
    ///
    /// assert!(table.get_by_id(&id).is_some());
    /// ```
    pub fn get_by_id(&self, id: &UserTypeId) -> Option<&UserType> {
        self.types.values().find(|user_type| user_type.id() == id)
    }
}

/// Resolves a parsed [`TypeExpr`](crate::ast::TypeExpr) into a semantic [`Type`].
///
/// Converts untyped type syntax from the CDSL AST into semantic type values,
/// including resolving unit expressions and looking up user-defined types.
///
/// # Parameters
///
/// - `type_expr`: Parsed type syntax from the CDSL AST.
/// - `type_table`: Registry of user-defined types for name lookup.
/// - `span`: Source location for error reporting.
///
/// # Returns
///
/// A resolved [`Type`] suitable for later validation and IR lowering.
///
/// # Errors
///
/// Returns [`CompileError`](crate::error::CompileError) if:
/// - Unit expression syntax is invalid
/// - A user type name is unknown
/// - Vector or matrix dimensions are zero
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::resolve::types::{resolve_type_expr, TypeTable};
/// use continuum_cdsl::ast::{TypeExpr, UnitExpr};
/// use continuum_cdsl::foundation::Span;
///
/// let table = TypeTable::new();
/// let span = Span::new(0, 10, 20, 1);
///
/// // Resolve Scalar<m>
/// let expr = TypeExpr::Scalar {
///     unit: Some(UnitExpr::Base("m".into())),
/// };
/// let ty = resolve_type_expr(&expr, &table, span).unwrap();
/// assert!(ty.is_kernel());
/// ```
pub fn resolve_type_expr(
    type_expr: &TypeExpr,
    type_table: &TypeTable,
    span: Span,
) -> Result<Type, CompileError> {
    match type_expr {
        TypeExpr::Scalar { unit } => {
            let resolved_unit = resolve_unit_expr(unit.as_ref(), span)?;
            Ok(Type::kernel(Shape::Scalar, resolved_unit, None))
        }

        TypeExpr::Vector { dim, unit } => {
            validate_nonzero_dim(*dim, "Vector", span)?;
            let resolved_unit = resolve_unit_expr(unit.as_ref(), span)?;
            Ok(Type::kernel(
                Shape::Vector { dim: *dim },
                resolved_unit,
                None,
            ))
        }

        TypeExpr::Matrix { rows, cols, unit } => {
            if *rows == 0 || *cols == 0 {
                return Err(CompileError::new(
                    ErrorKind::DimensionMismatch,
                    span,
                    "Matrix dimensions must be greater than 0".to_string(),
                ));
            }
            let resolved_unit = resolve_unit_expr(unit.as_ref(), span)?;
            Ok(Type::kernel(
                Shape::Matrix {
                    rows: *rows,
                    cols: *cols,
                },
                resolved_unit,
                None,
            ))
        }

        TypeExpr::User(path) => {
            let type_id = type_table.get_id(path).ok_or_else(|| {
                CompileError::new(
                    ErrorKind::UnknownType,
                    span,
                    format!("Unknown user type: {}", path),
                )
            })?;
            Ok(Type::user(type_id))
        }

        TypeExpr::Bool => Ok(Type::Bool),
    }
}

/// Resolves a parsed [`UnitExpr`](crate::ast::UnitExpr) into a semantic [`Unit`].
///
/// Converts unit syntax to dimensional unit values with dimensional exponents.
/// Handles unit arithmetic (multiply, divide, power) and validates kind compatibility.
///
/// # Parameters
///
/// - `unit_expr`: Optional unit syntax; `None` yields dimensionless units.
/// - `span`: Source location for error reporting.
///
/// # Returns
///
/// A resolved [`Unit`] with dimensional exponents computed from the expression.
///
/// # Errors
///
/// Returns [`CompileError`](crate::error::CompileError) if:
/// - A base unit name is unrecognized
/// - Unit arithmetic is invalid for affine or logarithmic units
/// - Dimensional exponent math overflows `i8` bounds
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::resolve::types::resolve_unit_expr;
/// use continuum_cdsl::ast::UnitExpr;
/// use continuum_cdsl::foundation::Span;
///
/// let span = Span::new(0, 10, 20, 1);
///
/// // Resolve m/s
/// let expr = UnitExpr::Divide(
///     Box::new(UnitExpr::Base("m".into())),
///     Box::new(UnitExpr::Base("s".into())),
/// );
/// let unit = resolve_unit_expr(Some(&expr), span).unwrap();
/// assert_eq!(unit.dims().length, 1);
/// assert_eq!(unit.dims().time, -1);
/// ```
pub fn resolve_unit_expr(unit_expr: Option<&UnitExpr>, span: Span) -> Result<Unit, CompileError> {
    match unit_expr {
        None => Ok(Unit::DIMENSIONLESS),
        Some(UnitExpr::Dimensionless) => Ok(Unit::DIMENSIONLESS),
        Some(UnitExpr::Base(name)) => resolve_base_unit(name, span),
        Some(UnitExpr::Multiply(lhs, rhs)) => {
            let lhs_unit = resolve_unit_expr(Some(lhs), span)?;
            let rhs_unit = resolve_unit_expr(Some(rhs), span)?;
            multiply_units(&lhs_unit, &rhs_unit, span)
        }
        Some(UnitExpr::Divide(numerator, denominator)) => {
            let num_unit = resolve_unit_expr(Some(numerator), span)?;
            let den_unit = resolve_unit_expr(Some(denominator), span)?;
            divide_units(&num_unit, &den_unit, span)
        }
        Some(UnitExpr::Power(base, exponent)) => {
            let base_unit = resolve_unit_expr(Some(base), span)?;
            power_unit(&base_unit, *exponent, span)
        }
    }
}

/// Validate that a dimension value is non-zero
///
/// Helper to reduce duplication in dimension validation.
fn validate_nonzero_dim(dim: u8, type_name: &str, span: Span) -> Result<(), CompileError> {
    if dim == 0 {
        return Err(CompileError::new(
            ErrorKind::DimensionMismatch,
            span,
            format!("{} dimension must be greater than 0", type_name),
        ));
    }
    Ok(())
}

/// Resolve a base unit name to a Unit
///
/// Maps unit symbols (m, kg, s, etc.) to their Unit definitions.
fn resolve_base_unit(name: &str, span: Span) -> Result<Unit, CompileError> {
    let unit = match name {
        // SI base units
        "m" => Unit::meters(),
        "kg" => Unit::kilograms(),
        "s" => Unit::seconds(),
        "K" => Unit::kelvin(),
        "A" => Unit::amperes(),
        "mol" => Unit::moles(),
        "cd" => Unit::candelas(),
        "rad" => Unit::radians(),

        _ => {
            return Err(CompileError::new(
                ErrorKind::InvalidUnit,
                span,
                format!("Unknown base unit: {}", name),
            ));
        }
    };
    Ok(unit)
}

/// Multiply two units
///
/// Combines dimensional exponents and validates kind compatibility.
fn multiply_units(lhs: &Unit, rhs: &Unit, span: Span) -> Result<Unit, CompileError> {
    // Only multiplicative units can be multiplied
    if !lhs.is_multiplicative() || !rhs.is_multiplicative() {
        return Err(CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            "Cannot multiply non-multiplicative units (affine/logarithmic)".to_string(),
        ));
    }

    let dims = add_dimensions(lhs.dims(), rhs.dims(), span)?;
    Ok(Unit::new(UnitKind::Multiplicative, dims))
}

/// Divide two units
///
/// Subtracts dimensional exponents and validates kind compatibility.
fn divide_units(numerator: &Unit, denominator: &Unit, span: Span) -> Result<Unit, CompileError> {
    // Only multiplicative units can be divided
    if !numerator.is_multiplicative() || !denominator.is_multiplicative() {
        return Err(CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            "Cannot divide non-multiplicative units (affine/logarithmic)".to_string(),
        ));
    }

    let dims = subtract_dimensions(numerator.dims(), denominator.dims(), span)?;
    Ok(Unit::new(UnitKind::Multiplicative, dims))
}

/// Raise a unit to a power
///
/// Multiplies all dimensional exponents by the exponent.
fn power_unit(base: &Unit, exponent: i8, span: Span) -> Result<Unit, CompileError> {
    // Only multiplicative units can be raised to powers
    if !base.is_multiplicative() {
        return Err(CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            "Cannot raise non-multiplicative units to powers (affine/logarithmic)".to_string(),
        ));
    }

    let dims = scale_dimensions(base.dims(), exponent, span)?;
    Ok(Unit::new(UnitKind::Multiplicative, dims))
}

/// Add dimensional exponents (for multiplication)
fn add_dimensions(
    lhs: &UnitDimensions,
    rhs: &UnitDimensions,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_i8_op(lhs.length, rhs.length, "length", span, i8::checked_add)?,
        mass: checked_i8_op(lhs.mass, rhs.mass, "mass", span, i8::checked_add)?,
        time: checked_i8_op(lhs.time, rhs.time, "time", span, i8::checked_add)?,
        temperature: checked_i8_op(
            lhs.temperature,
            rhs.temperature,
            "temperature",
            span,
            i8::checked_add,
        )?,
        current: checked_i8_op(lhs.current, rhs.current, "current", span, i8::checked_add)?,
        amount: checked_i8_op(lhs.amount, rhs.amount, "amount", span, i8::checked_add)?,
        luminosity: checked_i8_op(
            lhs.luminosity,
            rhs.luminosity,
            "luminosity",
            span,
            i8::checked_add,
        )?,
        angle: checked_i8_op(lhs.angle, rhs.angle, "angle", span, i8::checked_add)?,
    })
}

/// Subtract dimensional exponents (for division)
fn subtract_dimensions(
    lhs: &UnitDimensions,
    rhs: &UnitDimensions,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_i8_op(lhs.length, rhs.length, "length", span, i8::checked_sub)?,
        mass: checked_i8_op(lhs.mass, rhs.mass, "mass", span, i8::checked_sub)?,
        time: checked_i8_op(lhs.time, rhs.time, "time", span, i8::checked_sub)?,
        temperature: checked_i8_op(
            lhs.temperature,
            rhs.temperature,
            "temperature",
            span,
            i8::checked_sub,
        )?,
        current: checked_i8_op(lhs.current, rhs.current, "current", span, i8::checked_sub)?,
        amount: checked_i8_op(lhs.amount, rhs.amount, "amount", span, i8::checked_sub)?,
        luminosity: checked_i8_op(
            lhs.luminosity,
            rhs.luminosity,
            "luminosity",
            span,
            i8::checked_sub,
        )?,
        angle: checked_i8_op(lhs.angle, rhs.angle, "angle", span, i8::checked_sub)?,
    })
}

/// Scale dimensional exponents (for power)
fn scale_dimensions(
    dims: &UnitDimensions,
    scale: i8,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_i8_op(dims.length, scale, "length", span, i8::checked_mul)?,
        mass: checked_i8_op(dims.mass, scale, "mass", span, i8::checked_mul)?,
        time: checked_i8_op(dims.time, scale, "time", span, i8::checked_mul)?,
        temperature: checked_i8_op(
            dims.temperature,
            scale,
            "temperature",
            span,
            i8::checked_mul,
        )?,
        current: checked_i8_op(dims.current, scale, "current", span, i8::checked_mul)?,
        amount: checked_i8_op(dims.amount, scale, "amount", span, i8::checked_mul)?,
        luminosity: checked_i8_op(dims.luminosity, scale, "luminosity", span, i8::checked_mul)?,
        angle: checked_i8_op(dims.angle, scale, "angle", span, i8::checked_mul)?,
    })
}

/// Checked i8 arithmetic with dimension name in error
///
/// Generic helper to reduce duplication across add/sub/mul operations.
fn checked_i8_op<F>(a: i8, b: i8, dim_name: &str, span: Span, op: F) -> Result<i8, CompileError>
where
    F: FnOnce(i8, i8) -> Option<i8>,
{
    op(a, b).ok_or_else(|| {
        CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            format!("Dimension exponent overflow for {}", dim_name),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::TypeId;

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    #[test]
    fn test_resolve_scalar_type() {
        let type_table = TypeTable::new();

        // Scalar with no unit (dimensionless)
        let scalar_type = TypeExpr::Scalar { unit: None };
        let resolved = resolve_type_expr(&scalar_type, &type_table, test_span()).unwrap();
        assert!(resolved.is_kernel());

        // Scalar with meters
        let scalar_m = TypeExpr::Scalar {
            unit: Some(UnitExpr::Base("m".to_string())),
        };
        let resolved = resolve_type_expr(&scalar_m, &type_table, test_span()).unwrap();
        assert!(resolved.is_kernel());
    }

    #[test]
    fn test_resolve_vector_type() {
        let type_table = TypeTable::new();

        // Vector<3, m>
        let vector_type = TypeExpr::Vector {
            dim: 3,
            unit: Some(UnitExpr::Base("m".to_string())),
        };
        let resolved = resolve_type_expr(&vector_type, &type_table, test_span()).unwrap();

        // Verify it's a kernel type with correct shape
        let Type::Kernel(kernel) = resolved else {
            panic!("Expected kernel type");
        };
        assert_eq!(kernel.shape, Shape::Vector { dim: 3 });
        assert_eq!(kernel.unit.dims().length, 1);
        assert!(kernel.unit.is_multiplicative());
    }

    #[test]
    fn test_resolve_vector_zero_dim_fails() {
        let type_table = TypeTable::new();

        // Vector<0, m> should fail
        let vector_type = TypeExpr::Vector {
            dim: 0,
            unit: Some(UnitExpr::Base("m".to_string())),
        };
        let err = resolve_type_expr(&vector_type, &type_table, test_span()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::DimensionMismatch);
        assert!(err.message.contains("Vector dimension"));
    }

    #[test]
    fn test_resolve_matrix_type() {
        let type_table = TypeTable::new();

        // Matrix<3, 3, kg>
        let matrix_type = TypeExpr::Matrix {
            rows: 3,
            cols: 3,
            unit: Some(UnitExpr::Base("kg".to_string())),
        };
        let resolved = resolve_type_expr(&matrix_type, &type_table, test_span()).unwrap();

        // Verify it's a kernel type with correct shape
        let Type::Kernel(kernel) = resolved else {
            panic!("Expected kernel type");
        };
        assert_eq!(kernel.shape, Shape::Matrix { rows: 3, cols: 3 });
        assert_eq!(kernel.unit.dims().mass, 1);
        assert!(kernel.unit.is_multiplicative());
    }

    #[test]
    fn test_matrix_zero_rows_or_cols_fails() {
        let type_table = TypeTable::new();

        // Matrix with zero rows
        let zero_rows = TypeExpr::Matrix {
            rows: 0,
            cols: 3,
            unit: Some(UnitExpr::Base("kg".to_string())),
        };
        let err = resolve_type_expr(&zero_rows, &type_table, test_span()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::DimensionMismatch);
        assert!(err.message.contains("Matrix dimensions"));

        // Matrix with zero cols
        let zero_cols = TypeExpr::Matrix {
            rows: 3,
            cols: 0,
            unit: Some(UnitExpr::Base("kg".to_string())),
        };
        let err = resolve_type_expr(&zero_cols, &type_table, test_span()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::DimensionMismatch);
        assert!(err.message.contains("Matrix dimensions"));
    }

    #[test]
    fn test_resolve_bool_type() {
        let type_table = TypeTable::new();

        let bool_type = TypeExpr::Bool;
        let resolved = resolve_type_expr(&bool_type, &type_table, test_span()).unwrap();
        assert!(resolved.is_bool());
    }

    #[test]
    fn test_resolve_user_type() {
        let mut type_table = TypeTable::new();

        // Register a user type
        let path = Path::from("Vec3");
        let type_id = TypeId::from("Vec3");
        let user_type = UserType::new(type_id.clone(), path.clone(), vec![]);
        type_table.register(user_type);

        // Resolve reference to it
        let user_type_expr = TypeExpr::User(path);
        let resolved = resolve_type_expr(&user_type_expr, &type_table, test_span()).unwrap();

        // Verify it's a user type with correct ID
        assert!(resolved.is_user());
        if let Type::User(resolved_id) = resolved {
            assert_eq!(resolved_id, type_id);
        } else {
            panic!("Expected user type");
        }
    }

    #[test]
    fn test_resolve_unknown_user_type_fails() {
        let type_table = TypeTable::new();

        // Try to resolve unknown type
        let user_type_expr = TypeExpr::User(Path::from("UnknownType"));
        let err = resolve_type_expr(&user_type_expr, &type_table, test_span()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownType);
        assert!(err.message.contains("Unknown user type"));
    }

    #[test]
    fn test_resolve_base_units() {
        let span = test_span();
        assert!(resolve_base_unit("m", span).is_ok());
        assert!(resolve_base_unit("kg", span).is_ok());
        assert!(resolve_base_unit("s", span).is_ok());
        assert!(resolve_base_unit("K", span).is_ok());
        assert!(resolve_base_unit("A", span).is_ok());
        assert!(resolve_base_unit("mol", span).is_ok());
        assert!(resolve_base_unit("cd", span).is_ok());
        assert!(resolve_base_unit("rad", span).is_ok());

        // Unknown unit should fail with proper error
        let err = resolve_base_unit("xyz", span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("Unknown base unit"));
    }

    #[test]
    fn test_unknown_unit_via_unit_expr() {
        let unit_expr = UnitExpr::Base("nope".to_string());
        let err = resolve_unit_expr(Some(&unit_expr), test_span()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("Unknown base unit"));
    }

    #[test]
    fn test_unit_multiplication() {
        // m * s = m·s
        let m = Unit::meters();
        let s = Unit::seconds();
        let result = multiply_units(&m, &s, test_span()).unwrap();
        assert_eq!(result.dims().length, 1);
        assert_eq!(result.dims().time, 1);
    }

    #[test]
    fn test_unit_division() {
        // m / s = m/s
        let m = Unit::meters();
        let s = Unit::seconds();
        let result = divide_units(&m, &s, test_span()).unwrap();
        assert_eq!(result.dims().length, 1);
        assert_eq!(result.dims().time, -1);
    }

    #[test]
    fn test_unit_power() {
        // m^2
        let m = Unit::meters();
        let result = power_unit(&m, 2, test_span()).unwrap();
        assert_eq!(result.dims().length, 2);
    }

    #[test]
    fn test_resolve_compound_unit() {
        // m/s
        let unit_expr = UnitExpr::Divide(
            Box::new(UnitExpr::Base("m".to_string())),
            Box::new(UnitExpr::Base("s".to_string())),
        );
        let resolved = resolve_unit_expr(Some(&unit_expr), test_span()).unwrap();
        assert_eq!(resolved.dims().length, 1);
        assert_eq!(resolved.dims().time, -1);
    }

    #[test]
    fn test_resolve_complex_unit() {
        // kg*m/s^2 (force unit, Newton)
        let kg_m = UnitExpr::Multiply(
            Box::new(UnitExpr::Base("kg".to_string())),
            Box::new(UnitExpr::Base("m".to_string())),
        );
        let s_squared = UnitExpr::Power(Box::new(UnitExpr::Base("s".to_string())), 2);
        let unit_expr = UnitExpr::Divide(Box::new(kg_m), Box::new(s_squared));

        let resolved = resolve_unit_expr(Some(&unit_expr), test_span()).unwrap();
        assert_eq!(resolved.dims().mass, 1);
        assert_eq!(resolved.dims().length, 1);
        assert_eq!(resolved.dims().time, -2);
    }

    #[test]
    fn test_dimensionless_unit() {
        let resolved = resolve_unit_expr(None, test_span()).unwrap();
        assert!(resolved.is_dimensionless());

        let explicit = resolve_unit_expr(Some(&UnitExpr::Dimensionless), test_span()).unwrap();
        assert!(explicit.is_dimensionless());
    }

    #[test]
    fn test_type_table_operations() {
        let mut table = TypeTable::new();

        let path = Path::from("TestType");
        let type_id = TypeId::from("TestType");
        let user_type = UserType::new(type_id.clone(), path.clone(), vec![]);

        assert!(!table.contains(&path));

        table.register(user_type);

        assert!(table.contains(&path));
        assert_eq!(table.get_id(&path), Some(type_id));
        assert!(table.get(&path).is_some());
    }

    #[test]
    fn test_error_spans_are_preserved() {
        let type_table = TypeTable::new();
        let span = Span::new(1, 100, 200, 10);

        // Test vector zero dimension error preserves span
        let vector_type = TypeExpr::Vector {
            dim: 0,
            unit: Some(UnitExpr::Base("m".to_string())),
        };
        let err = resolve_type_expr(&vector_type, &type_table, span).unwrap_err();
        assert_eq!(err.span, span);
        assert_eq!(err.kind, ErrorKind::DimensionMismatch);

        // Test unknown user type error preserves span
        let user_type_expr = TypeExpr::User(Path::from("UnknownType"));
        let err = resolve_type_expr(&user_type_expr, &type_table, span).unwrap_err();
        assert_eq!(err.span, span);
        assert_eq!(err.kind, ErrorKind::UnknownType);

        // Test unknown unit error preserves span
        let unit_expr = UnitExpr::Base("xyz".to_string());
        let err = resolve_unit_expr(Some(&unit_expr), span).unwrap_err();
        assert_eq!(err.span, span);
        assert_eq!(err.kind, ErrorKind::InvalidUnit);

        // Test non-multiplicative unit multiplication error preserves span
        let celsius = Unit::celsius();
        let kelvin = Unit::kelvin();
        let err = multiply_units(&celsius, &kelvin, span).unwrap_err();
        assert_eq!(err.span, span);
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
    }

    #[test]
    fn test_non_multiplicative_unit_arithmetic_fails() {
        let span = test_span();
        let celsius = Unit::celsius();
        let kelvin = Unit::kelvin();

        // Cannot multiply affine units
        let err = multiply_units(&celsius, &kelvin, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("Cannot multiply non-multiplicative"));

        // Cannot divide affine units
        let err = divide_units(&celsius, &kelvin, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("Cannot divide non-multiplicative"));

        // Cannot raise affine units to powers
        let err = power_unit(&celsius, 2, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("Cannot raise non-multiplicative"));
    }

    #[test]
    fn test_dimension_overflow_fails() {
        let span = test_span();

        // Multiply overflow
        let left = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 100);
        let right = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 100);
        let unit_expr = UnitExpr::Multiply(Box::new(left), Box::new(right));
        let err = resolve_unit_expr(Some(&unit_expr), span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("overflow"));

        // Divide overflow (subtraction)
        let big = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 120);
        let tiny = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), -120);
        let divide = UnitExpr::Divide(Box::new(big), Box::new(tiny));
        let err = resolve_unit_expr(Some(&divide), span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("overflow"));

        // Power overflow (scaling)
        let huge_power = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 127);
        let power = UnitExpr::Power(Box::new(huge_power), 2);
        let err = resolve_unit_expr(Some(&power), span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("overflow"));
    }

    #[test]
    fn test_unit_power_with_zero_and_negative_exponents() {
        let span = test_span();

        // m^0 = dimensionless
        let zero = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 0);
        let resolved = resolve_unit_expr(Some(&zero), span).unwrap();
        assert!(resolved.is_dimensionless());

        // m^-1 = 1/m
        let neg = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), -1);
        let resolved = resolve_unit_expr(Some(&neg), span).unwrap();
        assert_eq!(resolved.dims().length, -1);
    }

    #[test]
    fn test_vector_matrix_none_unit_is_dimensionless() {
        let type_table = TypeTable::new();
        let span = test_span();

        // Vector with no unit
        let vector = TypeExpr::Vector { dim: 2, unit: None };
        let resolved = resolve_type_expr(&vector, &type_table, span).unwrap();
        let Type::Kernel(kernel) = resolved else {
            panic!("Expected kernel type");
        };
        assert!(kernel.unit.is_dimensionless());
        assert_eq!(kernel.shape, Shape::Vector { dim: 2 });

        // Matrix with no unit
        let matrix = TypeExpr::Matrix {
            rows: 2,
            cols: 2,
            unit: None,
        };
        let resolved = resolve_type_expr(&matrix, &type_table, span).unwrap();
        let Type::Kernel(kernel) = resolved else {
            panic!("Expected kernel type");
        };
        assert!(kernel.unit.is_dimensionless());
        assert_eq!(kernel.shape, Shape::Matrix { rows: 2, cols: 2 });
    }

    #[test]
    fn test_scalar_type_details() {
        let type_table = TypeTable::new();
        let span = test_span();

        // Scalar with no unit
        let scalar = TypeExpr::Scalar { unit: None };
        let resolved = resolve_type_expr(&scalar, &type_table, span).unwrap();
        let Type::Kernel(kernel) = resolved else {
            panic!("Expected kernel type");
        };
        assert_eq!(kernel.shape, Shape::Scalar);
        assert!(kernel.unit.is_dimensionless());

        // Scalar with meters
        let scalar_m = TypeExpr::Scalar {
            unit: Some(UnitExpr::Base("m".to_string())),
        };
        let resolved = resolve_type_expr(&scalar_m, &type_table, span).unwrap();
        let Type::Kernel(kernel) = resolved else {
            panic!("Expected kernel type");
        };
        assert_eq!(kernel.shape, Shape::Scalar);
        assert_eq!(kernel.unit.dims().length, 1);
        assert!(kernel.unit.is_multiplicative());
    }

    #[test]
    fn test_scalar_compound_unit_resolution() {
        let type_table = TypeTable::new();
        let span = test_span();

        // Scalar<kg*m/s^2> (force unit, Newton)
        let kg_m = UnitExpr::Multiply(
            Box::new(UnitExpr::Base("kg".to_string())),
            Box::new(UnitExpr::Base("m".to_string())),
        );
        let s2 = UnitExpr::Power(Box::new(UnitExpr::Base("s".to_string())), 2);
        let unit = UnitExpr::Divide(Box::new(kg_m), Box::new(s2));
        let scalar = TypeExpr::Scalar { unit: Some(unit) };

        let Type::Kernel(kernel) = resolve_type_expr(&scalar, &type_table, span).unwrap() else {
            panic!("Expected kernel type");
        };
        assert_eq!(kernel.shape, Shape::Scalar);
        assert_eq!(kernel.unit.dims().mass, 1);
        assert_eq!(kernel.unit.dims().length, 1);
        assert_eq!(kernel.unit.dims().time, -2);
        // Verify other dimensions remain zero
        assert_eq!(kernel.unit.dims().temperature, 0);
        assert_eq!(kernel.unit.dims().current, 0);
        assert_eq!(kernel.unit.dims().amount, 0);
        assert_eq!(kernel.unit.dims().luminosity, 0);
        assert_eq!(kernel.unit.dims().angle, 0);
    }

    #[test]
    fn test_unit_cancellation_and_identity() {
        let span = test_span();

        // m / m => dimensionless
        let cancel = UnitExpr::Divide(
            Box::new(UnitExpr::Base("m".to_string())),
            Box::new(UnitExpr::Base("m".to_string())),
        );
        let resolved = resolve_unit_expr(Some(&cancel), span).unwrap();
        assert!(resolved.is_dimensionless());

        // 1 * m => m
        let identity = UnitExpr::Multiply(
            Box::new(UnitExpr::Dimensionless),
            Box::new(UnitExpr::Base("m".to_string())),
        );
        let resolved = resolve_unit_expr(Some(&identity), span).unwrap();
        assert_eq!(resolved.dims().length, 1);
        assert!(resolved.is_multiplicative());
        // Verify other dimensions remain zero
        assert_eq!(resolved.dims().mass, 0);
        assert_eq!(resolved.dims().time, 0);
    }

    #[test]
    fn test_base_unit_dimensions() {
        let span = test_span();

        // Meters
        let m = resolve_base_unit("m", span).unwrap();
        assert_eq!(m.dims().length, 1);
        assert_eq!(m.dims().mass, 0);
        assert_eq!(m.dims().time, 0);
        assert!(m.is_multiplicative());

        // Kilograms
        let kg = resolve_base_unit("kg", span).unwrap();
        assert_eq!(kg.dims().mass, 1);
        assert_eq!(kg.dims().length, 0);
        assert_eq!(kg.dims().time, 0);
        assert!(kg.is_multiplicative());

        // Seconds
        let s = resolve_base_unit("s", span).unwrap();
        assert_eq!(s.dims().time, 1);
        assert_eq!(s.dims().length, 0);
        assert_eq!(s.dims().mass, 0);
        assert!(s.is_multiplicative());
    }

    #[test]
    fn test_scalar_invalid_unit_propagates_error() {
        let type_table = TypeTable::new();
        let span = Span::new(3, 30, 40, 2);

        let scalar = TypeExpr::Scalar {
            unit: Some(UnitExpr::Base("bad".to_string())),
        };

        let err = resolve_type_expr(&scalar, &type_table, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert_eq!(err.span, span);
        assert!(err.message.contains("Unknown base unit"));
    }

    #[test]
    fn test_vector_invalid_unit_propagates_error() {
        let type_table = TypeTable::new();
        let span = Span::new(4, 50, 60, 2);

        let vector = TypeExpr::Vector {
            dim: 3,
            unit: Some(UnitExpr::Base("invalid".to_string())),
        };

        let err = resolve_type_expr(&vector, &type_table, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert_eq!(err.span, span);
        assert!(err.message.contains("Unknown base unit"));
    }

    #[test]
    fn test_matrix_invalid_unit_propagates_error() {
        let type_table = TypeTable::new();
        let span = Span::new(5, 70, 80, 2);

        let matrix = TypeExpr::Matrix {
            rows: 2,
            cols: 2,
            unit: Some(UnitExpr::Base("xyz".to_string())),
        };

        let err = resolve_type_expr(&matrix, &type_table, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert_eq!(err.span, span);
        assert!(err.message.contains("Unknown base unit"));
    }

    #[test]
    fn test_overflow_errors_preserve_span() {
        let span = Span::new(6, 90, 100, 3);

        // Test multiply overflow span
        let left = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 100);
        let right = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 100);
        let multiply = UnitExpr::Multiply(Box::new(left), Box::new(right));
        let err = resolve_unit_expr(Some(&multiply), span).unwrap_err();
        assert_eq!(err.span, span);
        assert!(err.message.contains("overflow"));

        // Test power overflow span
        let huge_power = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 127);
        let power = UnitExpr::Power(Box::new(huge_power), 2);
        let err = resolve_unit_expr(Some(&power), span).unwrap_err();
        assert_eq!(err.span, span);
        assert!(err.message.contains("overflow"));
    }
}
