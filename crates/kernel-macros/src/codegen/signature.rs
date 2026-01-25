//! Compile-time signature registration for kernel functions.
//!
//! Generates distributed slice registration for compile-time type checking
//! when type constraint attributes are provided.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

use crate::{
    parsing::KernelFnArgs,
    shared::{parse_return_value_type, KernelReturnValueType},
    validation::ParameterAnalysis,
};

/// Generate compile-time signature registration if type constraints are provided
pub(crate) fn generate_signature_registration(
    args: &KernelFnArgs,
    func: &ItemFn,
    analysis: &ParameterAnalysis<'_>,
) -> syn::Result<TokenStream> {
    let fn_name = &func.sig.ident;
    let dsl_name = args.name.clone().unwrap_or_else(|| fn_name.to_string());
    let namespace = &args.namespace;

    let signature_name = format_ident!("__KERNEL_SIG_{}", fn_name.to_string().to_uppercase());

    // Extract constraint expressions (all guaranteed to be Some at this point)
    let purity_expr = args
        .purity
        .as_ref()
        .expect("BUG: purity should be Some when generating signature");
    let shape_in_exprs: Vec<_> = args
        .shape_in
        .as_ref()
        .expect("BUG: shape_in should be Some when generating signature")
        .iter()
        .map(|e| quote! { #e })
        .collect();
    let unit_in_exprs: Vec<_> = args
        .unit_in
        .as_ref()
        .expect("BUG: unit_in should be Some when generating signature")
        .iter()
        .map(|e| quote! { #e })
        .collect();
    let shape_out_expr = args
        .shape_out
        .as_ref()
        .expect("BUG: shape_out should be Some when generating signature");
    let unit_out_expr = args
        .unit_out
        .as_ref()
        .expect("BUG: unit_out should be Some when generating signature");

    // Derive value type from function return type
    let value_type_expr = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => {
            let value_type = parse_return_value_type(ty)?;
            match value_type {
                KernelReturnValueType::Scalar => {
                    quote! { ::continuum_kernel_types::ValueType::Scalar }
                }
                KernelReturnValueType::Vec2 => {
                    quote! { ::continuum_kernel_types::ValueType::Vec2 }
                }
                KernelReturnValueType::Vec3 => {
                    quote! { ::continuum_kernel_types::ValueType::Vec3 }
                }
                KernelReturnValueType::Vec4 => {
                    quote! { ::continuum_kernel_types::ValueType::Vec4 }
                }
                KernelReturnValueType::Quat => {
                    quote! { ::continuum_kernel_types::ValueType::Quat }
                }
                KernelReturnValueType::Mat2 => {
                    quote! { ::continuum_kernel_types::ValueType::Mat2 }
                }
                KernelReturnValueType::Mat3 => {
                    quote! { ::continuum_kernel_types::ValueType::Mat3 }
                }
                KernelReturnValueType::Mat4 => {
                    quote! { ::continuum_kernel_types::ValueType::Mat4 }
                }
                KernelReturnValueType::Tensor => {
                    quote! { ::continuum_kernel_types::ValueType::Tensor }
                }
                KernelReturnValueType::Bool => {
                    quote! { ::continuum_kernel_types::ValueType::Bool }
                }
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                &func.sig.ident,
                "kernel functions must declare a return type",
            ));
        }
    };

    // Build requires_uses field for compile-time signature
    let requires_uses_signature =
        if let (Some(key), Some(hint)) = (&args.requires_uses, &args.requires_uses_hint) {
            quote! {
                Some(::continuum_kernel_types::RequiresUses {
                    key: #key,
                    hint: #hint,
                })
            }
        } else {
            quote! { None }
        };

    let param_names = &analysis.param_names;

    Ok(quote! {
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
    })
}
