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
    Ident, ItemFn, LitStr, Token,
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

        let args = Punctuated::<KernelArg, Token![,]>::parse_terminated(input)?;
        for arg in args {
            match arg {
                KernelArg::Name(n) => name = Some(n),
                KernelArg::Namespace(n) => namespace = Some(n),
                KernelArg::Category(c) => category = c,
                KernelArg::Variadic => variadic = true,
                KernelArg::Vectorized => vectorized = true,
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
                KernelArg::Category(_) | KernelArg::Variadic | KernelArg::Vectorized => {}
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
            "variadic" => Ok(KernelArg::Variadic),
            "vectorized" => Ok(KernelArg::Vectorized),
            other => Err(syn::Error::new(
                ident.span(),
                format!("unknown argument: {}", other),
            )),
        }
    }
}

/// Register a function as a kernel callable from DSL.
///
/// # Arguments
///
/// - `name = "..."` (optional): The name used in DSL expressions (defaults to function name)
/// - `namespace = "..."` (required): Namespace tag (e.g. "maths", "vector", "dt")
/// - `category = "..."` (optional): Category tag (defaults to "math")
/// - `variadic` (optional): Mark as variadic (takes `&[Value]` or `&[f64]`)
/// - `vectorized` (optional): Mark as having vectorized implementation available
///
/// # Detection
///
/// - If the last parameter is `Dt`, the function is dt-dependent
/// - If the first parameter is `&[f64]` or `&[Value]` and `variadic` is set, it's variadic
/// - Otherwise, arity is the number of parameters
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
            }
        };
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
