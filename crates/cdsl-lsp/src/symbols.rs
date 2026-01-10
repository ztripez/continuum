//! Symbol index for CDSL documents.
//!
//! Provides position-based symbol lookup for hover and go-to-definition.
//!
//! # Architecture
//!
//! The [`SymbolIndex`] maintains two collections:
//!
//! - **Definitions**: Top-level items (signals, fields, operators, etc.) with their spans
//! - **References**: Symbol references within expressions (e.g., `signal.core.temp` inside
//!   a resolve block) with their spans
//!
//! When looking up a symbol at a position, references are checked first (they're more
//! specific), then definitions. This ensures hovering over `signal.core.temp` inside
//! a `signal.thermal.gradient` resolve block shows info about `core.temp`, not the
//! containing `thermal.gradient`.
//!
//! # Reference Collection
//!
//! References are collected using [`SpannedExprVisitor`] from the DSL crate. The
//! [`RefCollector`] visitor walks all expressions and records signal, field, const,
//! config, and entity references with their source spans.

use std::ops::Range;

use continuum_dsl::ast::{
    ChronicleDef, CompilationUnit, EntityDef, EraDef, Expr, FieldDef, FnDef, FractureDef,
    ImpulseDef, Item, OperatorBody, OperatorDef, Path, SignalDef, Spanned, SpannedExprVisitor,
    StrataDef, TypeDef, TypeExpr,
};

/// Information about a symbol for hover display.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// The kind of symbol (signal, field, fn, etc.).
    pub kind: SymbolKind,
    /// The full path of the symbol.
    pub path: String,
    /// Documentation from `///` comments.
    pub doc: Option<String>,
    /// Type expression if applicable.
    pub ty: Option<String>,
    /// Human-readable title (from `: title("...")` attribute).
    pub title: Option<String>,
    /// Mathematical symbol (from `: symbol("...")` attribute).
    pub symbol: Option<String>,
    /// Strata path if applicable.
    pub strata: Option<String>,
    /// Additional info specific to the symbol kind.
    pub extra: Vec<(String, String)>,
}

/// The kind of symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Signal,
    Field,
    Operator,
    Function,
    Type,
    Strata,
    Era,
    Impulse,
    Fracture,
    Chronicle,
    Entity,
}

impl SymbolKind {
    /// Returns the display name for this kind.
    pub fn display_name(&self) -> &'static str {
        match self {
            SymbolKind::Signal => "signal",
            SymbolKind::Field => "field",
            SymbolKind::Operator => "operator",
            SymbolKind::Function => "fn",
            SymbolKind::Type => "type",
            SymbolKind::Strata => "strata",
            SymbolKind::Era => "era",
            SymbolKind::Impulse => "impulse",
            SymbolKind::Fracture => "fracture",
            SymbolKind::Chronicle => "chronicle",
            SymbolKind::Entity => "entity",
        }
    }
}

/// An indexed symbol with its source span.
#[derive(Debug)]
struct IndexedSymbol {
    span: Range<usize>,
    info: SymbolInfo,
}

/// A reference to another symbol (e.g., signal.foo inside an expression).
#[derive(Debug)]
struct SymbolReference {
    span: Range<usize>,
    kind: SymbolKind,
    target_path: String,
}

/// Completion info for a symbol.
#[derive(Debug, Clone)]
pub struct CompletionInfo<'a> {
    /// The kind of symbol.
    pub kind: SymbolKind,
    /// The full path of the symbol (e.g., "terra.surface.temp").
    pub path: &'a str,
    /// Documentation from `///` comments.
    pub doc: Option<&'a str>,
    /// Type expression string.
    pub ty: Option<&'a str>,
    /// Human-readable title.
    pub title: Option<&'a str>,
}

/// Reference info for semantic tokens.
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    /// The kind of symbol being referenced.
    pub kind: SymbolKind,
}

/// Symbol index for a document.
#[derive(Debug, Default)]
pub struct SymbolIndex {
    symbols: Vec<IndexedSymbol>,
    references: Vec<SymbolReference>,
    #[allow(dead_code)]
    module_doc: Option<String>,
}

impl SymbolIndex {
    /// Build a symbol index from a compilation unit.
    pub fn from_ast(ast: &CompilationUnit) -> Self {
        let mut index = SymbolIndex {
            symbols: Vec::new(),
            references: Vec::new(),
            module_doc: ast.module_doc.clone(),
        };

        for item in &ast.items {
            index.index_item(item);
        }

        index
    }

    /// Find the symbol at the given byte offset.
    pub fn find_at_offset(&self, offset: usize) -> Option<&SymbolInfo> {
        // First check if offset is within a reference (these are more specific)
        if let Some(reference) = self
            .references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
        {
            // Look up the referenced symbol's definition
            return self.find_definition(reference.kind, &reference.target_path);
        }

        // Fall back to finding the containing definition
        self.symbols
            .iter()
            .filter(|s| s.span.contains(&offset))
            .min_by_key(|s| s.span.end - s.span.start)
            .map(|s| &s.info)
    }

    /// Find a symbol definition by kind and path.
    fn find_definition(&self, kind: SymbolKind, path: &str) -> Option<&SymbolInfo> {
        self.symbols
            .iter()
            .find(|s| s.info.kind == kind && s.info.path == path)
            .map(|s| &s.info)
    }

    /// Find the definition span for the symbol at the given offset.
    ///
    /// Returns the byte range of the definition if found. Used for go-to-definition.
    pub fn find_definition_span(&self, offset: usize) -> Option<Range<usize>> {
        // First check if offset is within a reference
        if let Some(reference) = self
            .references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
        {
            // Find the definition's span
            return self
                .symbols
                .iter()
                .find(|s| s.info.kind == reference.kind && s.info.path == reference.target_path)
                .map(|s| s.span.clone());
        }

        // If on a definition itself, return its own span
        self.symbols
            .iter()
            .filter(|s| s.span.contains(&offset))
            .min_by_key(|s| s.span.end - s.span.start)
            .map(|s| s.span.clone())
    }

    /// Get the module documentation.
    #[allow(dead_code)]
    pub fn module_doc(&self) -> Option<&str> {
        self.module_doc.as_deref()
    }

    /// Get all symbols for document outline.
    ///
    /// Returns symbol info with spans for the document symbols provider.
    pub fn get_all_symbols(&self) -> impl Iterator<Item = (&SymbolInfo, &Range<usize>)> {
        self.symbols.iter().map(|s| (&s.info, &s.span))
    }

    /// Get all symbols for completion.
    ///
    /// Returns tuples of (kind, path, doc, type_info) for each defined symbol.
    pub fn get_completions(&self) -> impl Iterator<Item = CompletionInfo<'_>> {
        self.symbols.iter().map(|s| CompletionInfo {
            kind: s.info.kind,
            path: &s.info.path,
            doc: s.info.doc.as_deref(),
            ty: s.info.ty.as_deref(),
            title: s.info.title.as_deref(),
        })
    }

    /// Get all references for semantic tokens.
    ///
    /// Returns reference info with kind and span for syntax highlighting.
    pub fn get_all_references(&self) -> impl Iterator<Item = (ReferenceInfo, &Range<usize>)> {
        self.references
            .iter()
            .map(|r| (ReferenceInfo { kind: r.kind }, &r.span))
    }

    /// Find all references to the symbol at the given offset.
    ///
    /// Returns spans of all references (not including the definition itself).
    pub fn find_references(&self, offset: usize) -> Vec<Range<usize>> {
        // First, find what symbol we're looking for
        let target = if let Some(reference) = self
            .references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
        {
            // On a reference - find all refs to the same target
            Some((reference.kind, reference.target_path.clone()))
        } else if let Some(symbol) = self
            .symbols
            .iter()
            .filter(|s| s.span.contains(&offset))
            .min_by_key(|s| s.span.end - s.span.start)
        {
            // On a definition - find all refs to this definition
            Some((symbol.info.kind, symbol.info.path.clone()))
        } else {
            None
        };

        if let Some((kind, path)) = target {
            self.references
                .iter()
                .filter(|r| r.kind == kind && r.target_path == path)
                .map(|r| r.span.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    fn index_item(&mut self, item: &Spanned<Item>) {
        match &item.node {
            Item::SignalDef(def) => self.index_signal(def, item.span.clone()),
            Item::FieldDef(def) => self.index_field(def, item.span.clone()),
            Item::OperatorDef(def) => self.index_operator(def, item.span.clone()),
            Item::FnDef(def) => self.index_fn(def, item.span.clone()),
            Item::TypeDef(def) => self.index_type(def, item.span.clone()),
            Item::StrataDef(def) => self.index_strata(def, item.span.clone()),
            Item::EraDef(def) => self.index_era(def, item.span.clone()),
            Item::ImpulseDef(def) => self.index_impulse(def, item.span.clone()),
            Item::FractureDef(def) => self.index_fracture(def, item.span.clone()),
            Item::ChronicleDef(def) => self.index_chronicle(def, item.span.clone()),
            Item::EntityDef(def) => self.index_entity(def, item.span.clone()),
            Item::ConstBlock(_) | Item::ConfigBlock(_) => {
                // Config blocks don't have individual symbol identity
            }
        }
    }

    fn index_signal(&mut self, def: &SignalDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if def.dt_raw {
            extra.push(("dt_raw".to_string(), "true".to_string()));
        }

        if let Some(ref assertions) = def.assertions {
            extra.push((
                "assertions".to_string(),
                format!("{} assertion(s)", assertions.assertions.len()),
            ));
            // Index references in assertion expressions
            for assertion in &assertions.assertions {
                self.index_expr(&assertion.condition);
            }
        }

        if !def.tensor_constraints.is_empty() {
            let constraints: Vec<_> = def
                .tensor_constraints
                .iter()
                .map(|c| format!("{:?}", c).to_lowercase())
                .collect();
            extra.push(("constraints".to_string(), constraints.join(", ")));
        }

        // Index references in resolve block
        if let Some(ref resolve) = def.resolve {
            self.index_expr(&resolve.body);
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Signal,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: def.ty.as_ref().map(|t| format_type_expr(&t.node)),
                title: def.title.as_ref().map(|t| t.node.clone()),
                symbol: def.symbol.as_ref().map(|s| s.node.clone()),
                strata: def.strata.as_ref().map(|s| s.node.to_string()),
                extra,
            },
        });
    }

    fn index_field(&mut self, def: &FieldDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if let Some(ref topology) = def.topology {
            extra.push(("topology".to_string(), format!("{:?}", topology.node)));
        }

        // Index references in measure block
        if let Some(ref measure) = def.measure {
            self.index_expr(&measure.body);
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Field,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: def.ty.as_ref().map(|t| format_type_expr(&t.node)),
                title: def.title.as_ref().map(|t| t.node.clone()),
                symbol: def.symbol.as_ref().map(|s| s.node.clone()),
                strata: def.strata.as_ref().map(|s| s.node.to_string()),
                extra,
            },
        });
    }

    fn index_operator(&mut self, def: &OperatorDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if let Some(ref phase) = def.phase {
            extra.push(("phase".to_string(), format!("{:?}", phase.node)));
        }

        if let Some(ref assertions) = def.assertions {
            extra.push((
                "assertions".to_string(),
                format!("{} assertion(s)", assertions.assertions.len()),
            ));
            for assertion in &assertions.assertions {
                self.index_expr(&assertion.condition);
            }
        }

        // Index references in operator body
        if let Some(ref body) = def.body {
            match body {
                OperatorBody::Warmup(expr) => self.index_expr(expr),
                OperatorBody::Collect(expr) => self.index_expr(expr),
                OperatorBody::Measure(expr) => self.index_expr(expr),
            }
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Operator,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: None,
                title: None,
                symbol: None,
                strata: def.strata.as_ref().map(|s| s.node.to_string()),
                extra,
            },
        });
    }

    fn index_fn(&mut self, def: &FnDef, span: Range<usize>) {
        let mut extra = Vec::new();

        // Add parameter info
        if !def.params.is_empty() {
            let params: Vec<_> = def
                .params
                .iter()
                .map(|p| {
                    if let Some(ref ty) = p.ty {
                        format!("{}: {}", p.name.node, format_type_expr(&ty.node))
                    } else {
                        p.name.node.clone()
                    }
                })
                .collect();
            extra.push(("params".to_string(), params.join(", ")));
        }

        // Add generics info
        if !def.generics.is_empty() {
            let generics: Vec<_> = def.generics.iter().map(|g| g.node.clone()).collect();
            extra.push(("generics".to_string(), generics.join(", ")));
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Function,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: def
                    .return_type
                    .as_ref()
                    .map(|t| format_type_expr(&t.node)),
                title: None,
                symbol: None,
                strata: None,
                extra,
            },
        });
    }

    fn index_type(&mut self, def: &TypeDef, span: Range<usize>) {
        let mut extra = Vec::new();

        // Add field info
        if !def.fields.is_empty() {
            let fields: Vec<_> = def
                .fields
                .iter()
                .map(|f| format!("{}: {}", f.name.node, format_type_expr(&f.ty.node)))
                .collect();
            extra.push(("fields".to_string(), fields.join(", ")));
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Type,
                path: def.name.node.clone(),
                doc: def.doc.clone(),
                ty: None,
                title: None,
                symbol: None,
                strata: None,
                extra,
            },
        });
    }

    fn index_strata(&mut self, def: &StrataDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if let Some(ref stride) = def.stride {
            extra.push(("stride".to_string(), stride.node.to_string()));
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Strata,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: None,
                title: def.title.as_ref().map(|t| t.node.clone()),
                symbol: def.symbol.as_ref().map(|s| s.node.clone()),
                strata: None,
                extra,
            },
        });
    }

    fn index_era(&mut self, def: &EraDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if def.is_initial {
            extra.push(("initial".to_string(), "true".to_string()));
        }
        if def.is_terminal {
            extra.push(("terminal".to_string(), "true".to_string()));
        }
        if let Some(ref dt) = def.dt {
            extra.push((
                "dt".to_string(),
                format!("{:?} {}", dt.node.value, dt.node.unit),
            ));
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Era,
                path: def.name.node.clone(),
                doc: def.doc.clone(),
                ty: None,
                title: def.title.as_ref().map(|t| t.node.clone()),
                symbol: None,
                strata: None,
                extra,
            },
        });
    }

    fn index_impulse(&mut self, def: &ImpulseDef, span: Range<usize>) {
        // Index references in apply block
        if let Some(ref apply) = def.apply {
            self.index_expr(&apply.body);
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Impulse,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: def.payload_type.as_ref().map(|t| format_type_expr(&t.node)),
                title: None,
                symbol: None,
                strata: None,
                extra: Vec::new(),
            },
        });
    }

    fn index_fracture(&mut self, def: &FractureDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if !def.conditions.is_empty() {
            extra.push((
                "conditions".to_string(),
                format!("{} condition(s)", def.conditions.len()),
            ));
            // Index references in conditions
            for condition in &def.conditions {
                self.index_expr(condition);
            }
        }
        if !def.emit.is_empty() {
            extra.push((
                "emit".to_string(),
                format!("{} emission(s)", def.emit.len()),
            ));
            // Index references in emit statements
            for emit in &def.emit {
                // The target path is a signal reference
                self.references.push(SymbolReference {
                    span: emit.target.span.clone(),
                    kind: SymbolKind::Signal,
                    target_path: emit.target.node.to_string(),
                });
                // Index the value expression too
                self.index_expr(&emit.value);
            }
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Fracture,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: None,
                title: None,
                symbol: None,
                strata: None,
                extra,
            },
        });
    }

    fn index_chronicle(&mut self, def: &ChronicleDef, span: Range<usize>) {
        // Index references in observe handlers
        if let Some(ref observe) = def.observe {
            for handler in &observe.handlers {
                self.index_expr(&handler.condition);
                for (_, field_expr) in &handler.event_fields {
                    self.index_expr(field_expr);
                }
            }
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Chronicle,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: None,
                title: None,
                symbol: None,
                strata: None,
                extra: Vec::new(),
            },
        });
    }

    fn index_entity(&mut self, def: &EntityDef, span: Range<usize>) {
        let mut extra = Vec::new();

        if !def.schema.is_empty() {
            extra.push((
                "schema".to_string(),
                format!("{} field(s)", def.schema.len()),
            ));
        }
        if let Some(ref bounds) = def.count_bounds {
            extra.push(("count".to_string(), format!("{}..{}", bounds.min, bounds.max)));
        }

        self.symbols.push(IndexedSymbol {
            span,
            info: SymbolInfo {
                kind: SymbolKind::Entity,
                path: def.path.node.to_string(),
                doc: def.doc.clone(),
                ty: None,
                title: None,
                symbol: None,
                strata: def.strata.as_ref().map(|s| s.node.to_string()),
                extra,
            },
        });
    }

    /// Index references within an expression using the visitor pattern.
    fn index_expr(&mut self, expr: &Spanned<Expr>) {
        let mut collector = RefCollector {
            references: &mut self.references,
        };
        collector.walk(expr);
    }
}

/// Visitor that collects symbol references from expressions.
///
/// Implements [`SpannedExprVisitor`] to walk expression trees and record all
/// symbol references (signals, fields, entities, consts, configs) with their
/// source spans. The visitor only overrides the reference visit methods;
/// all other expression types use the default behavior which continues traversal.
///
/// # Example
///
/// ```ignore
/// let mut collector = RefCollector { references: &mut self.references };
/// collector.walk(&resolve_block.body);
/// // self.references now contains all symbol references in the resolve block
/// ```
struct RefCollector<'a> {
    references: &'a mut Vec<SymbolReference>,
}

impl SpannedExprVisitor for RefCollector<'_> {
    fn visit_signal_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Signal,
            target_path: path.to_string(),
        });
        true
    }

    fn visit_const_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
        // Consts don't have definitions we can look up yet, but index anyway
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Signal, // We use Signal as a fallback
            target_path: format!("const.{}", path),
        });
        true
    }

    fn visit_config_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Signal,
            target_path: format!("config.{}", path),
        });
        true
    }

    fn visit_field_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Field,
            target_path: path.to_string(),
        });
        true
    }

    fn visit_entity_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Entity,
            target_path: path.to_string(),
        });
        true
    }
}

/// Format a type expression for display.
fn format_type_expr(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Scalar { unit, range } => {
            if let Some(r) = range {
                format!("Scalar<{}, {}..{}>", unit, r.min, r.max)
            } else {
                format!("Scalar<{}>", unit)
            }
        }
        TypeExpr::Vector {
            dim,
            unit,
            magnitude,
        } => {
            if let Some(m) = magnitude {
                format!("Vec{}<{}, magnitude: {}..{}>", dim, unit, m.min, m.max)
            } else {
                format!("Vec{}<{}>", dim, unit)
            }
        }
        TypeExpr::Tensor {
            rows,
            cols,
            unit,
            constraints,
        } => {
            if constraints.is_empty() {
                format!("Tensor<{}, {}, {}>", rows, cols, unit)
            } else {
                let c: Vec<_> = constraints
                    .iter()
                    .map(|c| format!("{:?}", c).to_lowercase())
                    .collect();
                format!("Tensor<{}, {}, {}> : {}", rows, cols, unit, c.join(", "))
            }
        }
        TypeExpr::Grid {
            width,
            height,
            element_type,
        } => {
            format!(
                "Grid<{}, {}, {}>",
                width,
                height,
                format_type_expr(element_type)
            )
        }
        TypeExpr::Seq {
            element_type,
            constraints,
        } => {
            if constraints.is_empty() {
                format!("Seq<{}>", format_type_expr(element_type))
            } else {
                format!(
                    "Seq<{}> ({} constraints)",
                    format_type_expr(element_type),
                    constraints.len()
                )
            }
        }
        TypeExpr::Named(name) => name.clone(),
    }
}

/// Debug: get all indexed references (for testing).
#[cfg(test)]
impl SymbolIndex {
    fn get_references(&self) -> &[SymbolReference] {
        &self.references
    }
}

#[cfg(test)]
impl SymbolReference {
    fn span(&self) -> &Range<usize> {
        &self.span
    }
    fn target_path(&self) -> &str {
        &self.target_path
    }
}

/// Format symbol info as markdown for hover display.
pub fn format_hover_markdown(info: &SymbolInfo) -> String {
    let mut parts = Vec::new();

    // Header: kind.path
    let header = format!("**{}.{}**", info.kind.display_name(), info.path);
    parts.push(header);

    // Title if present
    if let Some(ref title) = info.title {
        parts.push(format!("*{}*", title));
    }

    // Type if present
    if let Some(ref ty) = info.ty {
        parts.push(format!("Type: `{}`", ty));
    }

    // Symbol if present
    if let Some(ref symbol) = info.symbol {
        parts.push(format!("Symbol: {}", symbol));
    }

    // Strata if present
    if let Some(ref strata) = info.strata {
        parts.push(format!("Strata: `{}`", strata));
    }

    // Extra info
    for (key, value) in &info.extra {
        parts.push(format!("{}: {}", key, value));
    }

    // Documentation
    if let Some(ref doc) = info.doc {
        parts.push(String::new()); // Empty line before doc
        parts.push("---".to_string());
        parts.push(doc.clone());
    }

    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_refs_are_indexed() {
        let src = r#"
signal.core.temp {
    : Scalar<K>
    resolve { prev }
}

signal.thermal.gradient {
    : Scalar<K>
    resolve {
        let core = signal.core.temp in
        core
    }
}
"#;
        let (ast, errors) = continuum_dsl::parse(src);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let index = SymbolIndex::from_ast(&ast.unwrap());
        let refs = index.get_references();

        // Should have at least one reference to core.temp
        let core_temp_refs: Vec<_> = refs
            .iter()
            .filter(|r| r.target_path() == "core.temp")
            .collect();

        assert!(
            !core_temp_refs.is_empty(),
            "Expected reference to core.temp, found refs: {:?}",
            refs.iter()
                .map(|r| (r.target_path(), r.span()))
                .collect::<Vec<_>>()
        );

        // Verify the span is correct - find the position of "signal.core.temp" in the resolve
        let pos = src.find("signal.core.temp in").unwrap();
        let ref_span = core_temp_refs[0].span();

        assert!(
            ref_span.contains(&pos),
            "Reference span {:?} should contain position {} (signal.core.temp)",
            ref_span,
            pos
        );
    }

    #[test]
    fn test_hover_finds_referenced_signal() {
        let src = r#"
signal.core.temp {
    : Scalar<K>
    : title("Core Temperature")
    resolve { prev }
}

signal.thermal.gradient {
    : Scalar<K>
    resolve {
        let core = signal.core.temp in
        core
    }
}
"#;
        let (ast, errors) = continuum_dsl::parse(src);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let index = SymbolIndex::from_ast(&ast.unwrap());

        // Find position of "signal.core.temp" in the resolve block
        let pos = src.find("signal.core.temp in").unwrap() + 7; // middle of "core"

        let info = index.find_at_offset(pos);
        assert!(info.is_some(), "Should find symbol at position {}", pos);

        let info = info.unwrap();
        assert_eq!(
            info.path, "core.temp",
            "Should find core.temp, not {:?}",
            info.path
        );
        assert_eq!(
            info.title,
            Some("Core Temperature".to_string()),
            "Should have title from core.temp definition"
        );
    }

    #[test]
    fn test_goto_definition() {
        let src = r#"
signal.core.temp {
    : Scalar<K>
    resolve { prev }
}

signal.thermal.gradient {
    : Scalar<K>
    resolve {
        let core = signal.core.temp in
        core
    }
}
"#;
        let (ast, errors) = continuum_dsl::parse(src);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let index = SymbolIndex::from_ast(&ast.unwrap());

        // Find position of the reference "signal.core.temp"
        let ref_pos = src.find("signal.core.temp in").unwrap() + 7;

        // Go-to-definition should return the span of signal.core.temp definition
        let def_span = index.find_definition_span(ref_pos);
        assert!(def_span.is_some(), "Should find definition span");

        let def_span = def_span.unwrap();

        // The definition span should point to the signal.core.temp block
        let def_text = &src[def_span.clone()];
        assert!(
            def_text.starts_with("signal.core.temp"),
            "Definition span should start with 'signal.core.temp', got: {:?}",
            &def_text[..50.min(def_text.len())]
        );
    }

    #[test]
    fn test_find_references() {
        let src = r#"
signal.core.temp {
    : Scalar<K>
    resolve { prev }
}

signal.thermal.gradient {
    : Scalar<K>
    resolve {
        let core = signal.core.temp in
        core
    }
}

fracture.test {
    when { signal.core.temp < 100 }
    emit { signal.core.temp <- 5.0 }
}
"#;
        let (ast, errors) = continuum_dsl::parse(src);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let index = SymbolIndex::from_ast(&ast.unwrap());

        // Find references from the definition
        let def_pos = src.find("signal.core.temp {").unwrap() + 7; // "core" in definition
        let refs = index.find_references(def_pos);

        // Should find 3 references (in gradient resolve, fracture when, fracture emit)
        assert_eq!(
            refs.len(),
            3,
            "Expected 3 references to core.temp, found {}",
            refs.len()
        );

        // Find references from a reference
        let ref_pos = src.find("signal.core.temp in").unwrap() + 7;
        let refs_from_ref = index.find_references(ref_pos);

        assert_eq!(
            refs_from_ref.len(),
            3,
            "Should find same references from a reference"
        );
    }

    #[test]
    fn test_document_symbols() {
        let src = r#"
signal.core.temp {
    : Scalar<K>
    resolve { prev }
}

field.thermal.display {
    : Scalar<K>
    measure { signal.core.temp }
}

strata.thermal {
    : stride(1)
}
"#;
        let (ast, errors) = continuum_dsl::parse(src);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let index = SymbolIndex::from_ast(&ast.unwrap());
        let symbols: Vec<_> = index.get_all_symbols().collect();

        assert_eq!(symbols.len(), 3, "Should have 3 symbols");

        let paths: Vec<_> = symbols.iter().map(|(info, _)| &info.path).collect();
        assert!(paths.contains(&&"core.temp".to_string()));
        assert!(paths.contains(&&"thermal.display".to_string()));
        assert!(paths.contains(&&"thermal".to_string()));
    }

    #[test]
    fn test_get_completions() {
        let src = r#"
/// Temperature of the core.
signal.core.temp {
    : Scalar<K>
    : title("Core Temperature")
    resolve { prev }
}

/// Surface temperature display.
field.thermal.display {
    : Scalar<K>
    measure { signal.core.temp }
}

fn.math.double(x) { x * 2 }
"#;
        let (ast, errors) = continuum_dsl::parse(src);
        assert!(errors.is_empty(), "Parse errors: {:?}", errors);

        let index = SymbolIndex::from_ast(&ast.unwrap());
        let completions: Vec<_> = index.get_completions().collect();

        assert_eq!(completions.len(), 3, "Should have 3 completions");

        // Check signal completion
        let signal = completions.iter().find(|c| c.path == "core.temp").unwrap();
        assert_eq!(signal.kind, SymbolKind::Signal);
        assert_eq!(signal.doc, Some("Temperature of the core."));
        assert_eq!(signal.ty, Some("Scalar<K>"));
        assert_eq!(signal.title, Some("Core Temperature"));

        // Check field completion
        let field = completions
            .iter()
            .find(|c| c.path == "thermal.display")
            .unwrap();
        assert_eq!(field.kind, SymbolKind::Field);
        assert_eq!(field.doc, Some("Surface temperature display."));

        // Check function completion
        let func = completions.iter().find(|c| c.path == "math.double").unwrap();
        assert_eq!(func.kind, SymbolKind::Function);
    }
}
