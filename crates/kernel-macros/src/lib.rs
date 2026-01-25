// Allow unwrap in tests only
#![cfg_attr(test, allow(clippy::unwrap_used))]

//! Proc-macro for registering kernel functions.
//!
//! # Usage
//!
//! ```ignore
//! use continuum_kernel_macros::kernel_fn;
//! use continuum_foundation::{Dt, Value};
//!
//! /// Exponential decay toward zero
//! #[kernel_fn(namespace = "dt")]
//! pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
//!     value * 0.5_f64.powf(dt / halflife)
//! }
//!
//! /// Absolute value (pure, no dt)
//! #[kernel_fn(namespace = "maths")]
//! pub fn abs(x: f64) -> f64 {
//!     x.abs()
//! }
//!
//! /// Generic Vector support
//! #[kernel_fn(namespace = "maths")]
//! pub fn vec_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
//!     [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

mod codegen;
mod parsing;
mod shared;
mod validation;

use codegen::{constant, runtime, signature, vectorized};
use parsing::{extract_doc_comments, KernelFnArgs, VectorizedKernelArgs};
use validation::{analyze_parameters, has_type_constraints, validate_type_constraints};

/// Register a function as a kernel callable from DSL.
///
/// This macro generates dual registration for kernel functions:
/// - Runtime registration in `KERNELS` distributed slice (for VM execution)
/// - Compile-time registration in `KERNEL_SIGNATURES` distributed slice (for type checking)
///
/// # Required Arguments
///
/// - `namespace = "..."`: Namespace tag (e.g. "maths", "vector", "dt", "logic", "compare")
///
/// # Optional Arguments (Runtime Only)
///
/// - `name = "..."`: DSL name (defaults to function name)
/// - `category = "..."`: Category tag (defaults to "math")
/// - `variadic`: Mark as variadic (takes `&[Value]` or `&[f64]`)
/// - `vectorized`: Mark as having vectorized implementation available
/// - `unit_inference = "..."`: Unit derivation rule ("dimensionless", "preserve_first", "sqrt", "integrate", "decay")
/// - `pattern_hint = "..."`: Hint for optimizer ("clamping", "decay", "integration")
/// - `requires_uses = "..."`: Required `use` statement key
/// - `requires_uses_hint = "..."`: Error message hint for missing `use`
///
/// # Type Constraint Arguments (Compile-Time Signatures)
///
/// When ANY type constraint attribute is provided, ALL must be provided (no implicit defaults):
///
/// - `purity = Pure | Effect`: Effect discipline
///   - `Pure`: No side effects, deterministic, can be used in any phase
///   - `Effect`: Mutates state (emit, spawn, destroy) or produces artifacts (log)
///
/// - `shape_in = [...]`: Parameter shape constraints (array, one per parameter)
///   - `Any`: Accept any shape
///   - `AnyScalar`: Accept only scalar values
///   - `AnyVector`: Accept vectors of any dimension
///   - `AnyMatrix`: Accept matrices of any dimensions
///   - `SameAs(N)`: Must match parameter N's shape
///   - `BroadcastWith(N)`: Must be broadcastable with parameter N
///   - `Exact(Shape::Scalar)`: Must be exactly this shape
///
/// - `unit_in = [...]`: Parameter unit constraints (array, one per parameter)
///   - `UnitAny`: Accept any unit
///   - `UnitDimensionless`: Must be dimensionless
///   - `Angle`: Must be an angle (radians)
///   - `UnitSameAs(N)`: Must match parameter N's unit
///   - `UnitExact(Unit::...)`: Must be exactly this unit
///
/// - `shape_out = ...`: Return shape derivation
///   - `ShapeSameAs(N)`: Returns same shape as parameter N
///   - `Scalar`: Always returns a scalar
///   - `FromBroadcast(N, M)`: Returns broadcast result of parameters N and M
///   - `ShapeExact(Shape::...)`: Always returns this exact shape
///
/// - `unit_out = ...`: Return unit derivation
///   - `UnitDerivSameAs(N)`: Returns same unit as parameter N
///   - `Dimensionless`: Always returns dimensionless value
///   - `Multiply(N, M)`: Returns unit_N * unit_M
///   - `Divide(N, M)`: Returns unit_N / unit_M
///   - `Sqrt(N)`: Returns sqrt(unit_N)
///   - `UnitDerivExact(Unit::...)`: Always returns this exact unit
///
/// # Detection
///
/// - If the last parameter is `Dt`, the function is dt-dependent
/// - If the first parameter is `&[f64]` or `&[Value]` and `variadic` is set, it's variadic
/// - Otherwise, arity is the number of parameters
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_kernel_macros::kernel_fn;
/// use continuum_kernel_types::prelude::*;
///
/// // Simple pure function with type constraints
/// #[kernel_fn(
///     namespace = "maths",
///     purity = Pure,
///     shape_in = [Any, SameAs(0)],
///     unit_in = [UnitAny, UnitSameAs(0)],
///     shape_out = ShapeSameAs(0),
///     unit_out = UnitDerivSameAs(0)
/// )]
/// pub fn add(a: f64, b: f64) -> f64 {
///     a + b
/// }
///
/// // Boolean logic with dimensionless output
/// #[kernel_fn(
///     namespace = "logic",
///     purity = Pure,
///     shape_in = [Any, Any],
///     unit_in = [UnitDimensionless, UnitDimensionless],
///     shape_out = ShapeSameAs(0),
///     unit_out = Dimensionless
/// )]
/// pub fn and(a: bool, b: bool) -> bool {
///     a && b
/// }
///
/// // Comparison returning dimensionless boolean
/// #[kernel_fn(
///     namespace = "compare",
///     purity = Pure,
///     shape_in = [Any, SameAs(0)],
///     unit_in = [UnitAny, UnitSameAs(0)],
///     shape_out = ShapeSameAs(0),
///     unit_out = Dimensionless
/// )]
/// pub fn eq(a: f64, b: f64) -> bool {
///     (a - b).abs() < f64::EPSILON
/// }
///
/// // Legacy syntax (runtime-only, no type constraints)
/// #[kernel_fn(namespace = "maths")]
/// pub fn abs(x: f64) -> f64 {
///     x.abs()
/// }
/// ```
///
/// # Errors
///
/// Compile-time errors are emitted when:
/// - `namespace` attribute is missing
/// - Type constraint attributes are partially provided (all-or-nothing required)
/// - `shape_in` or `unit_in` arity doesn't match parameter count
/// - Invalid Rust syntax in constraint expressions
#[proc_macro_attribute]
pub fn kernel_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as KernelFnArgs);
    let func = parse_macro_input!(item as ItemFn);

    let expanded = generate_kernel_registration(&args, &func);

    match expanded {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate_kernel_registration(
    args: &KernelFnArgs,
    func: &ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    // Extract doc comments
    let doc = extract_doc_comments(func);

    // Analyze parameters
    let analysis = analyze_parameters(func);

    // Validate type constraints if present
    validate_type_constraints(args, func, analysis.user_params.len())?;

    // Generate runtime registration (always generated)
    let runtime_reg = runtime::generate_runtime_registration(args, func, &analysis, &doc);

    // Generate compile-time signature registration if type constraints are provided
    let signature_reg = if has_type_constraints(args) {
        signature::generate_signature_registration(args, func, &analysis)?
    } else {
        quote! {}
    };

    // Generate constant alias registrations if marked as constant
    let constant_regs = constant::generate_constant_registrations(args, &func.sig.ident);

    Ok(quote! {
        #func

        #runtime_reg

        #signature_reg

        #constant_regs
    })
}

/// Register a vectorized implementation for an existing kernel function.
///
/// This macro should be used to register high-performance vectorized implementations
/// for functions that already have a scalar implementation registered with `kernel_fn`.
///
/// # Arguments
///
/// - `name = "..."` (optional): The name of the existing kernel function (defaults to function name)
/// - `namespace = "..."` (required): Namespace tag (e.g. "maths", "vector", "dt")
///
/// # Expected Signature
///
/// The function must have signature:
/// - Pure: `fn(args: &[&VRegBuffer], population: usize) -> Result<VRegBuffer, Error>`
/// - WithDt: `fn(args: &[&VRegBuffer], dt: Dt, population: usize) -> Result<VRegBuffer, Error>`
///
/// # Example
///
/// ```ignore
/// #[vectorized_kernel_fn(name = "integrate", namespace = "dt")]
/// pub fn integrate_vectorized(
///     args: &[&VRegBuffer],
///     dt: Dt,
///     population: usize
/// ) -> Result<VRegBuffer, Box<dyn std::error::Error + Send + Sync>> {
///     // Implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn vectorized_kernel_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as VectorizedKernelArgs);
    let func = parse_macro_input!(item as ItemFn);

    let expanded = vectorized::generate_vectorized_registration(&args, &func);

    match expanded {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}
