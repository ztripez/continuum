//! Runtime kernel descriptor generation.
//!
//! Generates the distributed slice registration for kernel functions,
//! including wrapper functions that handle Value conversion and dispatch.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, ItemFn};

use crate::{parsing::KernelFnArgs, validation::ParameterAnalysis};

/// Generate runtime kernel registration code
pub(crate) fn generate_runtime_registration(
    args: &KernelFnArgs,
    func: &ItemFn,
    analysis: &ParameterAnalysis<'_>,
    doc: &str,
) -> TokenStream {
    let fn_name = &func.sig.ident;
    let dsl_name = args.name.clone().unwrap_or_else(|| fn_name.to_string());
    let namespace = &args.namespace;
    let category = &args.category;
    let variadic = args.variadic;

    let descriptor_name = format_ident!("__KERNEL_{}", fn_name.to_string().to_uppercase());

    // Build signature string
    let signature = if variadic {
        format!("{}(...) -> Value", dsl_name)
    } else {
        format!("{}({}) -> Value", dsl_name, analysis.param_names.join(", "))
    };

    // Calculate arity (excluding dt)
    let arity = if variadic {
        quote! { ::continuum_kernel_registry::Arity::Variadic }
    } else {
        let count = analysis.param_names.len();
        quote! { ::continuum_kernel_registry::Arity::Fixed(#count) }
    };

    // Generate wrapper and implementation variant
    let (wrapper, impl_variant) = if variadic {
        generate_variadic_wrapper(fn_name, &analysis.user_params, analysis.has_dt)
    } else {
        generate_fixed_arity_wrapper(fn_name, &analysis.user_params, analysis.has_dt)
    };

    // Parse unit inference specification
    let unit_inference_value = generate_unit_inference(&args.unit_inference);

    // Parse pattern hints
    let pattern_hints_value = generate_pattern_hints(&args.pattern_hints);

    // Build requires_uses field
    let requires_uses_value = generate_requires_uses(&args.requires_uses, &args.requires_uses_hint);

    // Vectorized implementation (registered separately by vectorized_kernel_fn)
    let vectorized_impl = quote! { None };

    quote! {
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
    }
}

/// Generate wrapper for variadic kernel functions
fn generate_variadic_wrapper(
    fn_name: &Ident,
    user_params: &[(&Ident, &syn::Type)],
    has_dt: bool,
) -> (TokenStream, TokenStream) {
    let first_param_type = user_params.first().map(|(_, ty)| ty);
    let is_value_slice = if let Some(syn::Type::Reference(tr)) = first_param_type {
        if let syn::Type::Slice(ts) = tr.elem.as_ref() {
            if let syn::Type::Path(tp) = ts.elem.as_ref() {
                tp.path.segments.last().is_some_and(|s| s.ident == "Value")
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
}

/// Generate wrapper for fixed-arity kernel functions
fn generate_fixed_arity_wrapper(
    fn_name: &Ident,
    user_params: &[(&Ident, &syn::Type)],
    has_dt: bool,
) -> (TokenStream, TokenStream) {
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
}

/// Generate unit inference value from string specification
fn generate_unit_inference(unit_inference: &Option<String>) -> TokenStream {
    if let Some(ui_str) = unit_inference {
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
    }
}

/// Generate pattern hints from list of hint strings
fn generate_pattern_hints(pattern_hints: &[String]) -> TokenStream {
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
    quote! {
        ::continuum_kernel_registry::PatternHints {
            clamping: #clamping,
            decay: #decay,
            integration: #integration,
        }
    }
}

/// Generate requires_uses field from optional key and hint
fn generate_requires_uses(
    requires_uses: &Option<String>,
    requires_uses_hint: &Option<String>,
) -> TokenStream {
    if let (Some(key), Some(hint)) = (requires_uses, requires_uses_hint) {
        quote! {
            Some(::continuum_kernel_registry::RequiresUses {
                key: #key,
                hint: #hint,
            })
        }
    } else {
        quote! { None }
    }
}
