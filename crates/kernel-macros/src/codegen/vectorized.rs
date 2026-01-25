//! Vectorized kernel registration generation.
//!
//! Generates distributed slice registration for high-performance vectorized
//! implementations of kernel functions.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

use crate::parsing::VectorizedKernelArgs;

/// Generate vectorized kernel registration code
pub(crate) fn generate_vectorized_registration(
    args: &VectorizedKernelArgs,
    func: &ItemFn,
) -> syn::Result<TokenStream> {
    let fn_name = &func.sig.ident;
    let dsl_name = args.name.clone().unwrap_or_else(|| fn_name.to_string());
    let namespace = &args.namespace;

    let params: Vec<_> = func.sig.inputs.iter().collect();
    let has_dt = params.iter().any(|p| {
        if let syn::FnArg::Typed(pat) = p
            && let syn::Type::Path(tp) = pat.ty.as_ref()
        {
            return tp.path.segments.last().is_some_and(|seg| seg.ident == "Dt");
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
