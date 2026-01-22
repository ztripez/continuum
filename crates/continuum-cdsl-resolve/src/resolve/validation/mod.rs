//! Type validation for typed CDSL expressions.
//!
//! Validates typed expressions for semantic correctness after type resolution.
//! This pass checks type compatibility, kernel signatures, bounds, and unit consistency.
//!
//! # What This Pass Does
//!
//! 1. **Type compatibility** - Verifies operand types match expected types
//! 2. **Kernel validation** - Checks kernel calls against signatures
//! 3. **Bounds checking** - Validates values satisfy min/max constraints
//! 4. **Unit validation** - Ensures unit consistency in operations
//! 5. **Struct validation** - Verifies struct field types match declarations
//!
//! # What This Pass Does NOT Do
//!
//! - **No type inference** - Types must already be assigned (by type resolution)
//! - **No name resolution** - Names must already be resolved
//! - **No code generation** - Validation only, no IR output
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Res → Type Res → Validation → Compilation
//!                                             ^^^^^^
//!                                          YOU ARE HERE
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::validation::{validate_expr, ValidationContext};
//! use continuum_cdsl::ast::TypedExpr;
//!
//! let ctx = ValidationContext::new(type_table, kernel_registry);
//! let errors = validate_expr(&typed_expr, &ctx);
//! if errors.is_empty() {
//!     println!("Expression is valid!");
//! }
//! ```

use crate::resolve::types::TypeTable;
use continuum_cdsl_ast::KernelRegistry;

mod constraints;
mod kernels;
mod phases;
mod seq;
mod structs;
mod types;

// Re-export public API
pub use phases::validate_node;
pub use seq::validate_seq_escape;
pub use types::validate_expr;

/// Context for validating typed CDSL expressions using user-defined types and kernel signatures.
///
/// `ValidationContext` provides read-only access to user type definitions and kernel
/// signatures that expression validation needs when checking struct construction, field
/// access, and kernel calls.
///
/// # Parameters
///
/// - `type_table`: Reference to the [`TypeTable`] containing user-defined types.
/// - `kernel_registry`: Reference to the [`KernelRegistry`] containing kernel signatures.
///
/// # Returns
///
/// A `ValidationContext` that borrows the provided registries for lookups during validation.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::KernelRegistry;
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_cdsl::resolve::validation::ValidationContext;
///
/// let type_table = TypeTable::new();
/// let kernel_registry = KernelRegistry::global();
/// let ctx = ValidationContext::new(&type_table, kernel_registry);
/// let _ = ctx;
/// ```
pub struct ValidationContext<'a> {
    /// User type definitions for struct validation
    pub type_table: &'a TypeTable,

    /// Kernel signatures for kernel call validation
    pub kernel_registry: &'a KernelRegistry,
}

impl<'a> ValidationContext<'a> {
    /// Create a new validation context.
    ///
    /// # Parameters
    ///
    /// - `type_table`: User type definitions for struct validation.
    /// - `kernel_registry`: Kernel signatures for kernel call validation.
    ///
    /// # Returns
    ///
    /// A new validation context ready for use.
    pub fn new(type_table: &'a TypeTable, kernel_registry: &'a KernelRegistry) -> Self {
        Self {
            type_table,
            kernel_registry,
        }
    }
}
