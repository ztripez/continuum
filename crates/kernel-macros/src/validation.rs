//! Parameter analysis and constraint validation for kernel functions.

use syn::{Ident, ItemFn};

use crate::parsing::KernelFnArgs;

/// Analyzed parameter information extracted from function signature
pub(crate) struct ParameterAnalysis<'a> {
    /// Whether the function has a `Dt` parameter (last parameter)
    pub(crate) has_dt: bool,
    /// User-facing parameters (excluding Dt if present)
    pub(crate) user_params: Vec<(&'a Ident, &'a syn::Type)>,
    /// Parameter names for signature generation
    pub(crate) param_names: Vec<String>,
}

/// Analyze function parameters and extract relevant information
pub(crate) fn analyze_parameters(func: &ItemFn) -> ParameterAnalysis<'_> {
    let params: Vec<_> = func.sig.inputs.iter().collect();

    // Check if last param is Dt
    let has_dt = params.last().is_some_and(|p| {
        if let syn::FnArg::Typed(pat) = p
            && let syn::Type::Path(tp) = pat.ty.as_ref()
        {
            return tp.path.segments.last().is_some_and(|seg| seg.ident == "Dt");
        }
        false
    });

    // Extract parameters and their types (excluding dt)
    let user_params: Vec<(&Ident, &syn::Type)> = params
        .iter()
        .take(params.len() - if has_dt { 1 } else { 0 })
        .filter_map(|p| {
            if let syn::FnArg::Typed(pat) = p
                && let syn::Pat::Ident(pi) = pat.pat.as_ref()
            {
                return Some((&pi.ident, pat.ty.as_ref()));
            }
            None
        })
        .collect();

    // Names for signature string
    let param_names: Vec<String> = user_params.iter().map(|(id, _)| id.to_string()).collect();

    ParameterAnalysis {
        has_dt,
        user_params,
        param_names,
    }
}

/// Validate type constraints (structural validation only)
///
/// When ANY constraint attribute is provided, ALL must be provided (no implicit defaults).
/// This also validates arity matching between constraints and parameters.
pub(crate) fn validate_type_constraints(
    args: &KernelFnArgs,
    func: &ItemFn,
    param_count: usize,
) -> syn::Result<()> {
    let any_constraint_present = args.purity.is_some()
        || args.shape_in.is_some()
        || args.unit_in.is_some()
        || args.shape_out.is_some()
        || args.unit_out.is_some();

    if !any_constraint_present {
        return Ok(());
    }

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

    let shape_in_vec = args
        .shape_in
        .as_ref()
        .expect("BUG: shape_in should be Some when any_constraint_present is true");
    let unit_in_vec = args
        .unit_in
        .as_ref()
        .expect("BUG: unit_in should be Some when any_constraint_present is true");

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

    Ok(())
}

/// Check whether type constraints are present
pub(crate) fn has_type_constraints(args: &KernelFnArgs) -> bool {
    args.purity.is_some()
        || args.shape_in.is_some()
        || args.unit_in.is_some()
        || args.shape_out.is_some()
        || args.unit_out.is_some()
}
