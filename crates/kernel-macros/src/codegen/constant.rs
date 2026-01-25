//! Constant alias registration for kernel functions.
//!
//! Handles generation of distributed slice registration for constant aliases
//! (e.g., PI, π, TAU, τ) that desugar to kernel function calls.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::parsing::KernelFnArgs;

/// Generate constant alias registrations if marked as constant
pub(crate) fn generate_constant_registrations(args: &KernelFnArgs, fn_name: &Ident) -> TokenStream {
    if !args.constant || args.aliases.is_empty() {
        return quote! {};
    }

    let dsl_name = args.name.clone().unwrap_or_else(|| fn_name.to_string());
    let namespace = &args.namespace;

    let mut regs = Vec::new();
    for (i, alias) in args.aliases.iter().enumerate() {
        let const_name = format_ident!("__{}_CONSTANT_{}", fn_name.to_string().to_uppercase(), i);
        regs.push(quote! {
            #[allow(non_upper_case_globals)]
            #[::continuum_kernel_registry::linkme::distributed_slice(::continuum_kernel_registry::KERNEL_CONSTANTS)]
            static #const_name: ::continuum_kernel_registry::KernelConstant =
                ::continuum_kernel_registry::KernelConstant {
                    alias: #alias,
                    namespace: #namespace,
                    kernel_name: #dsl_name,
                };
        });
    }
    quote! { #(#regs)* }
}
