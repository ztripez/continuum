//! Shared types and utilities used across kernel macro modules.

use quote::quote;

/// Enum representing the return value type category of a kernel function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelReturnValueType {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Quat,
    Mat2,
    Mat3,
    Mat4,
    Tensor,
    Bool,
}

/// Parse a Rust type into a KernelReturnValueType
///
/// Recognizes:
/// - `f64`, `i64`, `Dt` → Scalar
/// - `[f64; 2]` → Vec2
/// - `[f64; 3]` → Vec3
/// - `[f64; 4]` → Vec4
/// - `Quat` → Quat
/// - `Mat2`, `Mat3`, `Mat4` → corresponding matrix types
/// - `TensorData` → Tensor
/// - `bool` → Bool
pub(crate) fn parse_return_value_type(ty: &syn::Type) -> syn::Result<KernelReturnValueType> {
    match ty {
        syn::Type::Path(type_path) => {
            let ident = type_path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string());
            match ident.as_deref() {
                Some("bool") => Ok(KernelReturnValueType::Bool),
                Some("f64") | Some("i64") | Some("Dt") => Ok(KernelReturnValueType::Scalar),
                Some("Quat") => Ok(KernelReturnValueType::Quat),
                Some("Mat2") => Ok(KernelReturnValueType::Mat2),
                Some("Mat3") => Ok(KernelReturnValueType::Mat3),
                Some("Mat4") => Ok(KernelReturnValueType::Mat4),
                Some("TensorData") => Ok(KernelReturnValueType::Tensor),
                _ => Err(syn::Error::new_spanned(
                    ty,
                    format!(
                        "unsupported kernel return type `{}`; expected scalar, vector, matrix, tensor, or bool",
                        quote! { #ty }
                    ),
                )),
            }
        }
        syn::Type::Group(group) => parse_return_value_type(&group.elem),
        syn::Type::Paren(paren) => parse_return_value_type(&paren.elem),
        syn::Type::Reference(reference) => parse_return_value_type(&reference.elem),
        syn::Type::Array(array) => match &array.len {
            syn::Expr::Lit(expr) => {
                if let syn::Lit::Int(int) = &expr.lit {
                    match int.base10_parse::<usize>().map_err(|err| {
                        syn::Error::new_spanned(int, format!("invalid array length: {}", err))
                    })? {
                        2 => Ok(KernelReturnValueType::Vec2),
                        3 => Ok(KernelReturnValueType::Vec3),
                        4 => Ok(KernelReturnValueType::Vec4),
                        _ => Err(syn::Error::new_spanned(
                            ty,
                            "unsupported array length for kernel return (expected 2, 3, or 4)",
                        )),
                    }
                } else {
                    Err(syn::Error::new_spanned(
                        ty,
                        "kernel return arrays must use literal lengths",
                    ))
                }
            }
            _ => Err(syn::Error::new_spanned(
                ty,
                "kernel return arrays must use literal lengths",
            )),
        },
        syn::Type::Slice(_) | syn::Type::Tuple(_) => Err(syn::Error::new_spanned(
            ty,
            "unsupported kernel return type; use fixed-size arrays for vectors",
        )),
        syn::Type::Verbatim(tokens) => {
            let parsed = syn::parse2::<syn::Type>(tokens.clone()).map_err(|err| {
                syn::Error::new_spanned(
                    ty,
                    format!("unsupported kernel return type `{}`: {}", tokens, err),
                )
            })?;
            parse_return_value_type(&parsed)
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            format!(
                "unsupported kernel return type `{}`; expected scalar, vector, matrix, tensor, or bool",
                quote! { #ty }
            ),
        )),
    }
}
