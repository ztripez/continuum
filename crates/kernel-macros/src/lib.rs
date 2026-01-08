//! Proc-macro for registering kernel functions.
//!
//! # Usage
//!
//! ```ignore
//! use continuum_kernel_macros::kernel_fn;
//! use continuum_foundation::Dt;
//!
//! /// Exponential decay toward zero
//! #[kernel_fn(name = "decay")]
//! pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
//!     value * 0.5_f64.powf(dt / halflife)
//! }
//!
//! /// Absolute value (pure, no dt)
//! #[kernel_fn(name = "abs")]
//! pub fn abs(x: f64) -> f64 {
//!     x.abs()
//! }
//!
//! /// Variadic sum
//! #[kernel_fn(name = "sum", variadic)]
//! pub fn sum(args: &[f64]) -> f64 {
//!     args.iter().sum()
//! }
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Ident, ItemFn, LitStr, Token,
};

/// Arguments to the kernel_fn attribute
struct KernelFnArgs {
    name: String,
    variadic: bool,
}

impl Parse for KernelFnArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut variadic = false;

        let args = Punctuated::<KernelArg, Token![,]>::parse_terminated(input)?;
        for arg in args {
            match arg {
                KernelArg::Name(n) => name = Some(n),
                KernelArg::Variadic => variadic = true,
            }
        }

        let name = name.ok_or_else(|| input.error("missing `name = \"...\"` argument"))?;
        Ok(KernelFnArgs { name, variadic })
    }
}

enum KernelArg {
    Name(String),
    Variadic,
}

impl Parse for KernelArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        match ident.to_string().as_str() {
            "name" => {
                let _: Token![=] = input.parse()?;
                let lit: LitStr = input.parse()?;
                Ok(KernelArg::Name(lit.value()))
            }
            "variadic" => Ok(KernelArg::Variadic),
            other => Err(syn::Error::new(ident.span(), format!("unknown argument: {}", other))),
        }
    }
}

/// Register a function as a kernel callable from DSL.
///
/// # Arguments
///
/// - `name = "..."` (required): The name used in DSL expressions
/// - `variadic` (optional): Mark as variadic (takes `&[f64]` instead of individual args)
///
/// # Detection
///
/// - If the last parameter is `Dt`, the function is dt-dependent
/// - If the first parameter is `&[f64]` and `variadic` is set, it's variadic
/// - Otherwise, arity is the number of f64 parameters
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
    let dsl_name = &args.name;
    let variadic = args.variadic;

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
                return tp.path.segments.last().map_or(false, |seg| seg.ident == "Dt");
            }
        }
        false
    });

    // Calculate arity (excluding dt)
    let arity = if variadic {
        quote! { continuum_kernel_registry::Arity::Variadic }
    } else {
        let count = params.len() - if has_dt { 1 } else { 0 };
        quote! { continuum_kernel_registry::Arity::Fixed(#count) }
    };

    // Generate the wrapper function and registration
    let descriptor_name = format_ident!("__KERNEL_{}", fn_name.to_string().to_uppercase());

    let (wrapper, impl_variant) = if variadic {
        // Variadic: fn(&[f64]) -> f64 or fn(&[f64], Dt) -> f64
        if has_dt {
            (
                quote! {
                    fn wrapper(args: &[f64], dt: continuum_kernel_registry::Dt) -> f64 {
                        #fn_name(args, dt)
                    }
                },
                quote! { continuum_kernel_registry::KernelImpl::WithDt(wrapper) },
            )
        } else {
            (
                quote! {
                    fn wrapper(args: &[f64]) -> f64 {
                        #fn_name(args)
                    }
                },
                quote! { continuum_kernel_registry::KernelImpl::Pure(wrapper) },
            )
        }
    } else {
        // Fixed arity: generate wrapper that unpacks args
        let param_count = params.len() - if has_dt { 1 } else { 0 };
        let arg_indices: Vec<_> = (0..param_count).map(syn::Index::from).collect();

        if has_dt {
            (
                quote! {
                    fn wrapper(args: &[f64], dt: continuum_kernel_registry::Dt) -> f64 {
                        #fn_name(#(args[#arg_indices]),*, dt)
                    }
                },
                quote! { continuum_kernel_registry::KernelImpl::WithDt(wrapper) },
            )
        } else {
            (
                quote! {
                    fn wrapper(args: &[f64]) -> f64 {
                        #fn_name(#(args[#arg_indices]),*)
                    }
                },
                quote! { continuum_kernel_registry::KernelImpl::Pure(wrapper) },
            )
        }
    };

    Ok(quote! {
        #func

        #[allow(non_upper_case_globals)]
        #[::continuum_kernel_registry::linkme::distributed_slice(::continuum_kernel_registry::KERNELS)]
        static #descriptor_name: ::continuum_kernel_registry::KernelDescriptor = {
            #wrapper
            ::continuum_kernel_registry::KernelDescriptor {
                name: #dsl_name,
                doc: #doc,
                arity: #arity,
                implementation: #impl_variant,
            }
        };
    })
}
