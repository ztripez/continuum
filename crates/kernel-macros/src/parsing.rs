//! Parsing logic for kernel_fn attribute arguments.

use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Expr, Ident, ItemFn, LitStr, Token,
};

/// Arguments to the kernel_fn attribute
pub(crate) struct KernelFnArgs {
    pub(crate) name: Option<String>,
    pub(crate) namespace: String,
    pub(crate) category: String,
    pub(crate) variadic: bool,
    #[allow(dead_code)] // Parsed but not currently used in code generation
    pub(crate) vectorized: bool,
    pub(crate) unit_inference: Option<String>,
    pub(crate) pattern_hints: Vec<String>,
    pub(crate) requires_uses: Option<String>,
    pub(crate) requires_uses_hint: Option<String>,
    // Rust-syntax type constraints (optional, for compile-time signatures)
    pub(crate) purity: Option<Expr>,
    pub(crate) shape_in: Option<Vec<Expr>>,
    pub(crate) unit_in: Option<Vec<Expr>>,
    pub(crate) shape_out: Option<Expr>,
    pub(crate) unit_out: Option<Expr>,
    // Mark this as a constant for desugaring bare identifiers
    pub(crate) constant: bool,
    // Aliases for desugaring bare identifiers (e.g., PI, π, TAU, τ)
    pub(crate) aliases: Vec<String>,
}

/// Arguments to the vectorized_kernel_fn attribute
pub(crate) struct VectorizedKernelArgs {
    pub(crate) name: Option<String>,
    pub(crate) namespace: String,
}

/// Individual argument parsed from the attribute
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
    // Rust-syntax type constraints
    Purity(Expr),
    ShapeIn(Vec<Expr>),
    UnitIn(Vec<Expr>),
    ShapeOut(Expr),
    UnitOut(Expr),
    // Constant marking
    Constant,
    // Aliases for desugaring
    Aliases(Vec<String>),
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
        let mut constant = false;
        let mut aliases = Vec::new();

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
                KernelArg::Constant => constant = true,
                KernelArg::Aliases(a) => aliases = a,
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
            constant,
            aliases,
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
                | KernelArg::UnitOut(_)
                | KernelArg::Constant
                | KernelArg::Aliases(_) => {}
            }
        }

        let namespace =
            namespace.ok_or_else(|| input.error("missing `namespace = \"...\"` argument"))?;
        Ok(VectorizedKernelArgs { name, namespace })
    }
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
            "constant" => Ok(KernelArg::Constant),
            // Rust-syntax type constraints (token forwarding)
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
            "aliases" => {
                input.parse::<Token![=]>()?;
                let content;
                syn::bracketed!(content in input);
                let lits = Punctuated::<LitStr, Token![,]>::parse_terminated(&content)?;
                Ok(KernelArg::Aliases(
                    lits.into_iter().map(|lit| lit.value()).collect(),
                ))
            }
            other => Err(syn::Error::new(
                ident.span(),
                format!("unknown argument: {}", other),
            )),
        }
    }
}

/// Extract doc comments from a function
pub(crate) fn extract_doc_comments(func: &ItemFn) -> String {
    func.attrs
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
        .join(" ")
}
