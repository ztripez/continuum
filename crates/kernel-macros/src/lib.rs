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
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, ItemFn, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
};

/// Arguments to the kernel_fn attribute
struct KernelFnArgs {
    name: Option<String>,
    namespace: String,
    category: String,
    variadic: bool,
    vectorized: bool,
    unit_inference: Option<String>,
    pattern_hints: Vec<String>,
    requires_uses: Option<String>,
    requires_uses_hint: Option<String>,
    // New Rust-syntax type constraints (optional, for compile-time signatures)
    purity: Option<Expr>,
    shape_in: Option<Vec<Expr>>,
    unit_in: Option<Vec<Expr>>,
    shape_out: Option<Expr>,
    unit_out: Option<Expr>,
}

/// Arguments to the vectorized_kernel_fn attribute
struct VectorizedKernelArgs {
    name: Option<String>,
    namespace: String,
}

impl Parse for KernelFnArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut namespace = None;
        let mut category = String::from("math");
        let mut variadic = false;
        let mut vectorized = false;
        let mut unit_inference = None;
        let mut pattern_hints = Vec::new();
        let mut requires_uses = None;
        let mut requires_uses_hint = None;
        let mut purity = None;
        let mut shape_in = None;
        let mut unit_in = None;
        let mut shape_out = None;
        let mut unit_out = None;

        let args = Punctuated::<KernelArg, Token![,]>::parse_terminated(input)?;
        for arg in args {
            match arg {
                KernelArg::Name(n) => name = Some(n),
                KernelArg::Namespace(n) => namespace = Some(n),
                KernelArg::Category(c) => category = c,
                KernelArg::Variadic => variadic = true,
                KernelArg::Vectorized => vectorized = true,
                KernelArg::UnitInference(u) => unit_inference = Some(u),
                KernelArg::PatternHint(h) => pattern_hints.push(h),
                KernelArg::RequiresUses(r) => requires_uses = Some(r),
                KernelArg::RequiresUsesHint(h) => requires_uses_hint = Some(h),
                KernelArg::Purity(p) => purity = Some(p),
                KernelArg::ShapeIn(s) => shape_in = Some(s),
                KernelArg::UnitIn(u) => unit_in = Some(u),
                KernelArg::ShapeOut(s) => shape_out = Some(s),
                KernelArg::UnitOut(u) => unit_out = Some(u),
            }
        }

        let namespace =
            namespace.ok_or_else(|| input.error("missing `namespace = \"...\"` argument"))?;
        Ok(KernelFnArgs {
            name,
            namespace,
            category,
            variadic,
            vectorized,
            unit_inference,
            pattern_hints,
            requires_uses,
            requires_uses_hint,
            purity,
            shape_in,
            unit_in,
            shape_out,
            unit_out,
        })
    }
}

impl Parse for VectorizedKernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut namespace = None;

        let args = Punctuated::<KernelArg, Token![,]>::parse_terminated(input)?;
        for arg in args {
            match arg {
                KernelArg::Name(n) => name = Some(n),
                KernelArg::Namespace(n) => namespace = Some(n),
                KernelArg::Category(_)
                | KernelArg::Variadic
                | KernelArg::Vectorized
                | KernelArg::UnitInference(_)
                | KernelArg::PatternHint(_)
                | KernelArg::RequiresUses(_)
                | KernelArg::RequiresUsesHint(_)
                | KernelArg::Purity(_)
                | KernelArg::ShapeIn(_)
                | KernelArg::UnitIn(_)
                | KernelArg::ShapeOut(_)
                | KernelArg::UnitOut(_) => {}
            }
        }

        let namespace =
            namespace.ok_or_else(|| input.error("missing `namespace = \"...\"` argument"))?;
        Ok(VectorizedKernelArgs { name, namespace })
    }
}

enum KernelArg {
    Name(String),
    Namespace(String),
    Category(String),
    Variadic,
    Vectorized,
    UnitInference(String),
    PatternHint(String),
    RequiresUses(String),
    RequiresUsesHint(String),
    // New Rust-syntax type constraints
    Purity(Expr),
    ShapeIn(Vec<Expr>),
    UnitIn(Vec<Expr>),
    ShapeOut(Expr),
    UnitOut(Expr),
}

impl Parse for KernelArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        match ident.to_string().as_str() {
            "name" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::Name(lit.value()))
            }
            "namespace" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::Namespace(lit.value()))
            }
            "category" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::Category(lit.value()))
            }
            "unit_inference" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::UnitInference(lit.value()))
            }
            "pattern_hint" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::PatternHint(lit.value()))
            }
            "requires_uses" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::RequiresUses(lit.value()))
            }
            "requires_uses_hint" => {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::RequiresUsesHint(lit.value()))
            }
            "variadic" => Ok(KernelArg::Variadic),
            "vectorized" => Ok(KernelArg::Vectorized),
            // New Rust-syntax type constraints (token forwarding)
            "purity" => {
                input.parse::<Token![=]>()?;
                let expr: Expr = input.parse()?;
                Ok(KernelArg::Purity(expr))
            }
            "shape_in" => {
                input.parse::<Token![=]>()?;
                let content;
                syn::bracketed!(content in input);
                let exprs = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
                Ok(KernelArg::ShapeIn(exprs.into_iter().collect()))
            }
            "unit_in" => {
                input.parse::<Token![=]>()?;
                let content;
                syn::bracketed!(content in input);
                let exprs = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
                Ok(KernelArg::UnitIn(exprs.into_iter().collect()))
            }
            "shape_out" => {
                input.parse::<Token![=]>()?;
                let expr: Expr = input.parse()?;
                Ok(KernelArg::ShapeOut(expr))
            }
            "unit_out" => {
                input.parse::<Token![=]>()?;
                let expr: Expr = input.parse()?;
                Ok(KernelArg::UnitOut(expr))
            }
            other => Err(syn::Error::new(
                ident.span(),
                format!("unknown argument: {}", other),
            )),
        }
    }
}

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
    let fn_name = &func.sig.ident;
    let dsl_name = args.name.clone().unwrap_or_else(|| fn_name.to_string());
    let namespace = &args.namespace;
    let category = &args.category;
    let variadic = args.variadic;
    let vectorized = args.vectorized;
    let unit_inference = &args.unit_inference;
    let pattern_hints = &args.pattern_hints;
    let requires_uses = &args.requires_uses;
    let requires_uses_hint = &args.requires_uses_hint;

    // Extract doc comments
    let doc = func
        .attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                attr.meta.require_name_value().ok().map(|nv| {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        s.value().trim().to_string()
                    } else {
                        String::new()
                    }
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    // Analyze parameters
    let params: Vec<_> = func.sig.inputs.iter().collect();

    // Check if last param is Dt
    let has_dt = params.last().map_or(false, |p| {
        if let syn::FnArg::Typed(pat) = p {
            if let syn::Type::Path(tp) = pat.ty.as_ref() {
                return tp
                    .path
                    .segments
                    .last()
                    .map_or(false, |seg| seg.ident == "Dt");
            }
        }
        false
    });

    // Extract parameters and their types (excluding dt)
    let user_params: Vec<(&Ident, &syn::Type)> = params
        .iter()
        .take(params.len() - if has_dt { 1 } else { 0 })
        .filter_map(|p| {
            if let syn::FnArg::Typed(pat) = p {
                if let syn::Pat::Ident(pi) = pat.pat.as_ref() {
                    return Some((&pi.ident, pat.ty.as_ref()));
                }
            }
            None
        })
        .collect();

    // Names for signature string
    let param_names: Vec<String> = user_params.iter().map(|(id, _)| id.to_string()).collect();

    // Build signature string
    let signature = if variadic {
        format!("{}(...) -> Value", dsl_name)
    } else {
        format!("{}({}) -> Value", dsl_name, param_names.join(", "))
    };

    // Calculate arity (excluding dt)
    let arity = if variadic {
        quote! { ::continuum_kernel_registry::Arity::Variadic }
    } else {
        let count = param_names.len();
        quote! { ::continuum_kernel_registry::Arity::Fixed(#count) }
    };

    // Generate the wrapper function and registration
    let descriptor_name = format_ident!("__KERNEL_{}", fn_name.to_string().to_uppercase());

    let (wrapper, impl_variant) = if variadic {
        // Variadic: fn(&[Value]) -> Value or fn(&[Value], Dt) -> Value

        let first_param_type = user_params.first().map(|(_, ty)| ty);
        let is_value_slice = if let Some(syn::Type::Reference(tr)) = first_param_type {
            if let syn::Type::Slice(ts) = tr.elem.as_ref() {
                if let syn::Type::Path(tp) = ts.elem.as_ref() {
                    tp.path
                        .segments
                        .last()
                        .map_or(false, |s| s.ident == "Value")
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        let (call_prelude, call_expr) = if is_value_slice {
            (quote! {}, quote! { args })
        } else {
            (
                quote! {
                    let converted_args: Vec<f64> = args.iter()
                         .map(|v| <f64 as ::continuum_kernel_registry::FromValue>::from_value(v).expect("Variadic kernel expects scalar f64 arguments"))
                         .collect();
                },
                quote! { &converted_args },
            )
        };

        if has_dt {
            (
                quote! {
                    fn wrapper(args: &[::continuum_kernel_registry::Value], dt: ::continuum_kernel_registry::Dt) -> ::continuum_kernel_registry::Value {
                        #call_prelude
                        let result = #fn_name(#call_expr, dt);
                        ::continuum_kernel_registry::IntoValue::into_value(result)
                    }
                },
                quote! { ::continuum_kernel_registry::KernelImpl::WithDt(wrapper) },
            )
        } else {
            (
                quote! {
                    fn wrapper(args: &[::continuum_kernel_registry::Value]) -> ::continuum_kernel_registry::Value {
                        #call_prelude
                        let result = #fn_name(#call_expr);
                        ::continuum_kernel_registry::IntoValue::into_value(result)
                    }
                },
                quote! { ::continuum_kernel_registry::KernelImpl::Pure(wrapper) },
            )
        }
    } else {
        // Fixed arity: generate wrapper that unpacks args using FromValue
        let param_indices: Vec<_> = (0..user_params.len()).map(syn::Index::from).collect();
        let param_types: Vec<_> = user_params.iter().map(|(_, ty)| ty).collect();

        // Create identifiers for the temporary variables
        let arg_val_idents: Vec<_> = (0..user_params.len())
            .map(|i| format_ident!("arg_val_{}", i))
            .collect();
        let arg_idents: Vec<_> = (0..user_params.len())
            .map(|i| format_ident!("arg_{}", i))
            .collect();

        let unpack_stmts = quote! {
            #(
                let #arg_val_idents = &args[#param_indices];
                let #arg_idents = <#param_types as ::continuum_kernel_registry::FromValue>::from_value(#arg_val_idents)
                    .expect(&format!("Kernel argument type mismatch for argument {} ({})", #param_indices, stringify!(#param_types)));
            )*
        };

        let arg_names = quote! { #(#arg_idents),* };

        if has_dt {
            (
                quote! {
                    fn wrapper(args: &[::continuum_kernel_registry::Value], dt: ::continuum_kernel_registry::Dt) -> ::continuum_kernel_registry::Value {
                        #unpack_stmts
                        let result = #fn_name(#arg_names, dt);
                        ::continuum_kernel_registry::IntoValue::into_value(result)
                    }
                },
                quote! { ::continuum_kernel_registry::KernelImpl::WithDt(wrapper) },
            )
        } else {
            (
                quote! {
                    fn wrapper(args: &[::continuum_kernel_registry::Value]) -> ::continuum_kernel_registry::Value {
                        #unpack_stmts
                        let result = #fn_name(#arg_names);
                        ::continuum_kernel_registry::IntoValue::into_value(result)
                    }
                },
                quote! { ::continuum_kernel_registry::KernelImpl::Pure(wrapper) },
            )
        }
    };

    // Determine vectorized implementation (placeholder for now - will be set by separate macro)
    let vectorized_impl = if vectorized {
        quote! { None } // TODO: Update when we add vectorized_kernel_fn macro
    } else {
        quote! { None }
    };

    // Parse unit inference specification
    let unit_inference_value = if let Some(ui_str) = unit_inference {
        match ui_str.as_str() {
            "dimensionless" => {
                quote! { ::continuum_kernel_registry::UnitInference::Dimensionless { requires_angle: false } }
            }
            "dimensionless_angle" => {
                quote! { ::continuum_kernel_registry::UnitInference::Dimensionless { requires_angle: true } }
            }
            "preserve_first" => {
                quote! { ::continuum_kernel_registry::UnitInference::PreserveFirst }
            }
            "sqrt" => quote! { ::continuum_kernel_registry::UnitInference::Sqrt },
            "integrate" => quote! { ::continuum_kernel_registry::UnitInference::Integrate },
            "decay" => quote! { ::continuum_kernel_registry::UnitInference::Decay },
            _ => quote! { ::continuum_kernel_registry::UnitInference::None },
        }
    } else {
        quote! { ::continuum_kernel_registry::UnitInference::None }
    };

    // Parse pattern hints
    let mut clamping = false;
    let mut decay = false;
    let mut integration = false;
    for hint in pattern_hints {
        match hint.as_str() {
            "clamping" => clamping = true,
            "decay" => decay = true,
            "integration" => integration = true,
            _ => {}
        }
    }
    let pattern_hints_value = quote! {
        ::continuum_kernel_registry::PatternHints {
            clamping: #clamping,
            decay: #decay,
            integration: #integration,
        }
    };

    // Build requires_uses field for runtime descriptor
    let requires_uses_value = if let (Some(key), Some(hint)) = (requires_uses, requires_uses_hint) {
        quote! {
            Some(::continuum_kernel_registry::RequiresUses {
                key: #key,
                hint: #hint,
            })
        }
    } else {
        quote! { None }
    };

    // Build requires_uses field for compile-time signature
    let requires_uses_signature =
        if let (Some(key), Some(hint)) = (requires_uses, requires_uses_hint) {
            quote! {
                Some(::continuum_kernel_types::RequiresUses {
                    key: #key,
                    hint: #hint,
                })
            }
        } else {
            quote! { None }
        };

    // Structural validation for new type constraint attributes (if present)
    // When ANY constraint attribute is provided, ALL must be provided (no implicit defaults)
    let any_constraint_present = args.purity.is_some()
        || args.shape_in.is_some()
        || args.unit_in.is_some()
        || args.shape_out.is_some()
        || args.unit_out.is_some();

    if any_constraint_present {
        // Reject constraints on variadic functions (they take &[Value], can't type-check per-arg)
        if args.variadic {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "variadic functions cannot have type constraints (they take &[Value] at runtime)",
            ));
        }

        // Reject unit_inference when type constraints are present (ambiguous)
        if args.unit_inference.is_some() {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "unit_inference cannot be used with type constraints (use unit_out instead)",
            ));
        }

        // Require all constraint attributes when any are provided
        if args.purity.is_none() {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "purity attribute required when type constraints are provided",
            ));
        }
        if args.shape_in.is_none() {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "shape_in attribute required when type constraints are provided",
            ));
        }
        if args.unit_in.is_none() {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "unit_in attribute required when type constraints are provided",
            ));
        }
        if args.shape_out.is_none() {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "shape_out attribute required when type constraints are provided",
            ));
        }
        if args.unit_out.is_none() {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "unit_out attribute required when type constraints are provided",
            ));
        }

        let param_count = user_params.len();
        let shape_in_vec = args.shape_in.as_ref().unwrap();
        let unit_in_vec = args.unit_in.as_ref().unwrap();

        // Validate shape_in arity matches parameter count
        if shape_in_vec.len() != param_count {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                format!(
                    "shape_in has {} constraints but function has {} parameters",
                    shape_in_vec.len(),
                    param_count
                ),
            ));
        }

        // Validate unit_in arity matches parameter count
        if unit_in_vec.len() != param_count {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                format!(
                    "unit_in has {} constraints but function has {} parameters",
                    unit_in_vec.len(),
                    param_count
                ),
            ));
        }
    }

    // Generate compile-time signature registration if type constraints are provided
    let signature_registration = if any_constraint_present {
        let signature_name = format_ident!("__KERNEL_SIG_{}", fn_name.to_string().to_uppercase());

        // Extract constraint expressions (all guaranteed to be Some at this point)
        let purity_expr = args.purity.as_ref().unwrap();
        let shape_in_exprs: Vec<_> = args
            .shape_in
            .as_ref()
            .unwrap()
            .iter()
            .map(|e| quote! { #e })
            .collect();
        let unit_in_exprs: Vec<_> = args
            .unit_in
            .as_ref()
            .unwrap()
            .iter()
            .map(|e| quote! { #e })
            .collect();
        let shape_out_expr = args.shape_out.as_ref().unwrap();
        let unit_out_expr = args.unit_out.as_ref().unwrap();

        // Derive value type from function return type
        // We can't reliably parse the type AST from declarative macro expansions,
        // so we convert to string and check the token representation
        let value_type_expr = match &func.sig.output {
            syn::ReturnType::Type(_, ty) => {
                let ty_str = quote! { #ty }.to_string();
                if ty_str.trim() == "bool" {
                    quote! { ::continuum_kernel_types::ValueType::Bool }
                } else {
                    // Default to F64 for f64, f32, or other numeric types
                    quote! { ::continuum_kernel_types::ValueType::F64 }
                }
            }
            _ => quote! { ::continuum_kernel_types::ValueType::F64 },
        };

        quote! {
            #[allow(non_upper_case_globals)]
            #[::continuum_kernel_registry::linkme::distributed_slice(::continuum_kernel_types::KERNEL_SIGNATURES)]
            static #signature_name: ::continuum_kernel_types::KernelSignature = {
                use ::continuum_kernel_types::prelude::*;
                ::continuum_kernel_types::KernelSignature {
                    id: ::continuum_kernel_types::KernelId::new(#namespace, #dsl_name),
                    purity: #purity_expr,
                    params: &[
                        #(::continuum_kernel_types::KernelParam {
                            name: #param_names,
                            shape: #shape_in_exprs,
                            unit: #unit_in_exprs,
                        }),*
                    ],
                    returns: ::continuum_kernel_types::KernelReturn {
                        shape: #shape_out_expr,
                        unit: #unit_out_expr,
                        value_type: #value_type_expr,
                    },
                    requires_uses: #requires_uses_signature,
                }
            };
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        #func

        #[allow(non_upper_case_globals)]
        #[::continuum_kernel_registry::linkme::distributed_slice(::continuum_kernel_registry::KERNELS)]
        static #descriptor_name: ::continuum_kernel_registry::KernelDescriptor = {
            #wrapper
            ::continuum_kernel_registry::KernelDescriptor {
                namespace: #namespace,
                name: #dsl_name,
                signature: #signature,
                doc: #doc,
                category: #category,
                arity: #arity,
                implementation: #impl_variant,
                vectorized_impl: #vectorized_impl,
                unit_inference: #unit_inference_value,
                pattern_hints: #pattern_hints_value,
                requires_uses: #requires_uses_value,
            }
        };

        #signature_registration
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

    let expanded = generate_vectorized_registration(&args, &func);

    match expanded {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate_vectorized_registration(
    args: &VectorizedKernelArgs,
    func: &ItemFn,
) -> syn::Result<proc_macro2::TokenStream> {
    let fn_name = &func.sig.ident;
    let dsl_name = args.name.clone().unwrap_or_else(|| fn_name.to_string());
    let namespace = &args.namespace;

    let params: Vec<_> = func.sig.inputs.iter().collect();
    let has_dt = params.iter().any(|p| {
        if let syn::FnArg::Typed(pat) = p {
            if let syn::Type::Path(tp) = pat.ty.as_ref() {
                return tp
                    .path
                    .segments
                    .last()
                    .map_or(false, |seg| seg.ident == "Dt");
            }
        }
        false
    });

    let descriptor_name = format_ident!("__VECTOR_KERNEL_{}", fn_name.to_string().to_uppercase());

    let (wrapper, impl_variant) = if has_dt {
        (
            quote! {
                fn wrapper(
                    args: &[&::continuum_kernel_registry::VRegBuffer],
                    dt: ::continuum_kernel_registry::Dt,
                    population: usize,
                ) -> ::continuum_kernel_registry::VectorizedResult<::continuum_kernel_registry::VRegBuffer> {
                    #fn_name(args, dt, population)
                }
            },
            quote! { ::continuum_kernel_registry::VectorizedImpl::WithDt(wrapper) },
        )
    } else {
        (
            quote! {
                fn wrapper(
                    args: &[&::continuum_kernel_registry::VRegBuffer],
                    population: usize,
                ) -> ::continuum_kernel_registry::VectorizedResult<::continuum_kernel_registry::VRegBuffer> {
                    #fn_name(args, population)
                }
            },
            quote! { ::continuum_kernel_registry::VectorizedImpl::Pure(wrapper) },
        )
    };

    Ok(quote! {
        #func

        #[allow(non_upper_case_globals)]
        #[::continuum_kernel_registry::linkme::distributed_slice(::continuum_kernel_registry::VECTOR_KERNELS)]
        static #descriptor_name: ::continuum_kernel_registry::VectorizedKernelDescriptor = {
            #wrapper
            ::continuum_kernel_registry::VectorizedKernelDescriptor {
                namespace: #namespace,
                name: #dsl_name,
                implementation: #impl_variant,
            }
        };
    })
}
