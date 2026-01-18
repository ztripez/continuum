//! Type resolution pass
//!
//! Resolves `TypeExpr` → `Type` and performs basic type inference.
//!
//! # What This Pass Does
//!
//! 1. **TypeExpr → Type** - Resolves untyped type syntax to semantic types
//! 2. **UnitExpr → Unit** - Resolves unit syntax to dimensional units
//! 3. **User type lookup** - Resolves type names to TypeIds
//! 4. **Kernel type resolution** - Derives return types from kernel signatures
//!
//! # What This Pass Does NOT Do
//!
//! - **No full type inference** - Complex bidirectional inference deferred to validation
//! - **No type checking** - Compatibility validation happens in later passes
//! - **No unit arithmetic** - Complex dimensional analysis deferred
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
//! ```cdsl
//! signal velocity : Vector<3, m/s>  // TypeExpr → Type::Kernel(Vector<3, m/s>)
//! type OrbitalElements {            // Registers user type
//!     semi_major_axis: Scalar<m>
//! }
//! ```

use crate::ast::{TypeExpr, UnitExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{
    Path, Shape, Span, Type, Unit, UnitDimensions, UnitKind, UserType, UserTypeId,
};
use std::collections::HashMap;

/// Type table for user-defined types
///
/// Maps type names to their definitions and assigns unique TypeIds.
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

    /// Look up a type by path
    pub fn get(&self, path: &Path) -> Option<&UserType> {
        self.types.get(path)
    }

    /// Look up a type ID by path
    pub fn get_id(&self, path: &Path) -> Option<UserTypeId> {
        self.type_ids.get(path).cloned()
    }

    /// Check if a type exists
    pub fn contains(&self, path: &Path) -> bool {
        self.types.contains_key(path)
    }
}

/// Resolve a TypeExpr to a Type
///
/// Converts untyped type syntax from the parser into semantic Type values.
/// This includes resolving unit expressions and looking up user types.
///
/// # Errors
///
/// Returns `CompileError` if:
/// - Unit expression is invalid
/// - User type reference doesn't exist
/// - Vector/Matrix dimensions are invalid
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
            if *dim == 0 {
                return Err(CompileError::new(
                    ErrorKind::DimensionMismatch,
                    span,
                    "Vector dimension must be greater than 0".to_string(),
                ));
            }
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

/// Resolve a UnitExpr to a Unit
///
/// Converts unit syntax to dimensional unit values with proper exponents.
///
/// # Errors
///
/// Returns `CompileError` if:
/// - Base unit name is unrecognized
/// - Unit arithmetic is invalid
/// - Exponent is out of range for i8
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
        length: checked_add_i8(lhs.length, rhs.length, "length", span)?,
        mass: checked_add_i8(lhs.mass, rhs.mass, "mass", span)?,
        time: checked_add_i8(lhs.time, rhs.time, "time", span)?,
        temperature: checked_add_i8(lhs.temperature, rhs.temperature, "temperature", span)?,
        current: checked_add_i8(lhs.current, rhs.current, "current", span)?,
        amount: checked_add_i8(lhs.amount, rhs.amount, "amount", span)?,
        luminosity: checked_add_i8(lhs.luminosity, rhs.luminosity, "luminosity", span)?,
        angle: checked_add_i8(lhs.angle, rhs.angle, "angle", span)?,
    })
}

/// Subtract dimensional exponents (for division)
fn subtract_dimensions(
    lhs: &UnitDimensions,
    rhs: &UnitDimensions,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_sub_i8(lhs.length, rhs.length, "length", span)?,
        mass: checked_sub_i8(lhs.mass, rhs.mass, "mass", span)?,
        time: checked_sub_i8(lhs.time, rhs.time, "time", span)?,
        temperature: checked_sub_i8(lhs.temperature, rhs.temperature, "temperature", span)?,
        current: checked_sub_i8(lhs.current, rhs.current, "current", span)?,
        amount: checked_sub_i8(lhs.amount, rhs.amount, "amount", span)?,
        luminosity: checked_sub_i8(lhs.luminosity, rhs.luminosity, "luminosity", span)?,
        angle: checked_sub_i8(lhs.angle, rhs.angle, "angle", span)?,
    })
}

/// Scale dimensional exponents (for power)
fn scale_dimensions(
    dims: &UnitDimensions,
    scale: i8,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_mul_i8(dims.length, scale, "length", span)?,
        mass: checked_mul_i8(dims.mass, scale, "mass", span)?,
        time: checked_mul_i8(dims.time, scale, "time", span)?,
        temperature: checked_mul_i8(dims.temperature, scale, "temperature", span)?,
        current: checked_mul_i8(dims.current, scale, "current", span)?,
        amount: checked_mul_i8(dims.amount, scale, "amount", span)?,
        luminosity: checked_mul_i8(dims.luminosity, scale, "luminosity", span)?,
        angle: checked_mul_i8(dims.angle, scale, "angle", span)?,
    })
}

/// Checked addition for i8 with dimension name in error
fn checked_add_i8(a: i8, b: i8, dim_name: &str, span: Span) -> Result<i8, CompileError> {
    a.checked_add(b).ok_or_else(|| {
        CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            format!("Dimension exponent overflow for {}", dim_name),
        )
    })
}

/// Checked subtraction for i8 with dimension name in error
fn checked_sub_i8(a: i8, b: i8, dim_name: &str, span: Span) -> Result<i8, CompileError> {
    a.checked_sub(b).ok_or_else(|| {
        CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            format!("Dimension exponent overflow for {}", dim_name),
        )
    })
}

/// Checked multiplication for i8 with dimension name in error
fn checked_mul_i8(a: i8, b: i8, dim_name: &str, span: Span) -> Result<i8, CompileError> {
    a.checked_mul(b).ok_or_else(|| {
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

        // Create units with extreme exponents that will overflow when multiplied
        let left = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 100);
        let right = UnitExpr::Power(Box::new(UnitExpr::Base("m".to_string())), 100);
        let unit_expr = UnitExpr::Multiply(Box::new(left), Box::new(right));

        let err = resolve_unit_expr(Some(&unit_expr), span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("overflow"));
    }
}
