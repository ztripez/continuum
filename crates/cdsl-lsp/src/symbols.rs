//! Symbol index for CDSL documents.
//!
//! Provides position-based symbol lookup for hover and go-to-definition.

use std::ops::Range;

use continuum_compiler::dsl::ast::{
    CompilationUnit, ConfigBlock, ConstBlock, Expr, Item, Literal, OperatorBody, Path, Spanned,
    SpannedExprVisitor,
};
use continuum_compiler::ir::{
    CompiledNode, CompiledWorld, NodeKind, PrimitiveParamKind, PrimitiveParamSpec, ValueType,
    ValueTypeParamValue,
};
use continuum_kernel_registry as kernel_registry;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    Member,
    World,
    Const,
    Config,
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
            SymbolKind::Member => "member",
            SymbolKind::World => "world",
            SymbolKind::Const => "const",
            SymbolKind::Config => "config",
        }
    }
}

/// An indexed symbol with its source span.
#[derive(Debug)]
pub struct IndexedSymbol {
    /// Full span of the definition block.
    pub span: Range<usize>,
    /// Span of just the path (e.g., "terra.temp" in "signal.terra.temp").
    pub path_span: Range<usize>,
    pub info: SymbolInfo,
}

/// A reference to another symbol (e.g., signal.foo inside an expression).
#[derive(Debug)]
pub struct SymbolReference {
    pub span: Range<usize>,
    pub kind: SymbolKind,
    pub target_path: String,
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

/// A function parameter for signature help.
#[derive(Debug, Clone)]
pub struct FnParamInfo {
    /// Parameter name.
    pub name: String,
    /// Parameter type (formatted string).
    pub ty: Option<String>,
}

/// Function signature for signature help.
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Function path (e.g., "math.lerp").
    pub path: String,
    /// Documentation comment.
    pub doc: Option<String>,
    /// Generic type parameters.
    pub generics: Vec<String>,
    /// Function parameters.
    pub params: Vec<FnParamInfo>,
    /// Return type (formatted string).
    pub return_type: Option<String>,
}

/// Reference validation info for undefined reference diagnostics.
#[derive(Debug, Clone)]
pub struct ReferenceValidationInfo {
    /// The span of the reference in source.
    pub span: Range<usize>,
    /// The kind of symbol being referenced.
    pub kind: SymbolKind,
    /// The target path (e.g., "core.temp").
    pub target_path: String,
}

/// Symbol index for a document.
#[derive(Debug, Default)]
pub struct SymbolIndex {
    pub symbols: Vec<IndexedSymbol>,
    pub references: Vec<SymbolReference>,
    pub functions: Vec<FunctionSignature>,
    #[allow(dead_code)]
    pub module_doc: Option<String>,
}

impl SymbolIndex {
    /// Build a symbol index from an AST and its lowered IR.
    pub fn new(ast: &CompilationUnit, world: &CompiledWorld) -> Self {
        let mut index = SymbolIndex {
            symbols: Vec::new(),
            references: Vec::new(),
            functions: Vec::new(),
            module_doc: ast.module_doc.clone(),
        };

        // 1. Index references from AST (since they have spans)
        for item in &ast.items {
            index.index_ast_references(item);
        }

        // 2. Index definitions from IR (since they have resolved info)
        for node in world.nodes.values() {
            index.index_compiled_node(node);
        }

        // 3. Index Const/Config from AST (since they aren't nodes in IR yet)
        for item in &ast.items {
            match &item.node {
                Item::ConstBlock(block) => index.index_const_block(block),
                Item::ConfigBlock(block) => index.index_config_block(block),
                _ => {}
            }
        }

        index
    }

    /// (Backward Compatibility) Build from AST only.
    pub fn from_ast(ast: &CompilationUnit) -> Self {
        // Lower to IR for indexing (individual file context)
        let world = continuum_compiler::ir::lower(ast).unwrap_or_else(|_| CompiledWorld {
            constants: indexmap::IndexMap::new(),
            config: indexmap::IndexMap::new(),
            nodes: indexmap::IndexMap::new(),
        });
        Self::new(ast, &world)
    }

    fn index_ast_references(&mut self, item: &Spanned<Item>) {
        match &item.node {
            Item::SignalDef(def) => {
                if let Some(ref resolve) = def.resolve {
                    self.index_expr(&resolve.body);
                }
                if let Some(ref assertions) = def.assertions {
                    for assertion in &assertions.assertions {
                        self.index_expr(&assertion.condition);
                    }
                }
            }
            Item::FieldDef(def) => {
                if let Some(ref measure) = def.measure {
                    self.index_expr(&measure.body);
                }
            }
            Item::OperatorDef(def) => {
                if let Some(ref body) = def.body {
                    match body {
                        OperatorBody::Warmup(expr) => self.index_expr(expr),
                        OperatorBody::Collect(expr) => self.index_expr(expr),
                        OperatorBody::Measure(expr) => self.index_expr(expr),
                    }
                }
                if let Some(ref assertions) = def.assertions {
                    for assertion in &assertions.assertions {
                        self.index_expr(&assertion.condition);
                    }
                }
            }
            Item::FnDef(def) => {
                self.index_expr(&def.body);
            }
            Item::ImpulseDef(def) => {
                if let Some(ref apply) = def.apply {
                    self.index_expr(&apply.body);
                }
            }
            Item::FractureDef(def) => {
                for condition in &def.conditions {
                    self.index_expr(condition);
                }
                if let Some(ref emit) = def.emit {
                    self.index_expr(emit);
                }
            }
            Item::ChronicleDef(def) => {
                if let Some(ref observe) = def.observe {
                    for handler in &observe.handlers {
                        self.index_expr(&handler.condition);
                        for (_, field_expr) in &handler.event_fields {
                            self.index_expr(field_expr);
                        }
                    }
                }
            }
            Item::MemberDef(def) => {
                if let Some(ref initial) = def.initial {
                    self.index_expr(&initial.body);
                }
                if let Some(ref resolve) = def.resolve {
                    self.index_expr(&resolve.body);
                }
                if let Some(ref assertions) = def.assertions {
                    for assertion in &assertions.assertions {
                        self.index_expr(&assertion.condition);
                    }
                }
            }
            Item::EraDef(def) => {
                for transition in &def.transitions {
                    for condition in &transition.conditions {
                        self.index_expr(condition);
                    }
                }
            }
            _ => {}
        }
    }

    fn index_compiled_node(&mut self, node: &CompiledNode) {
        let mut extra = Vec::new();
        let kind = match &node.kind {
            NodeKind::Signal(props) => {
                if props.uses_dt_raw {
                    extra.push(("dt_raw".to_string(), "true".to_string()));
                }
                SymbolKind::Signal
            }
            NodeKind::Field(props) => {
                extra.push(("topology".to_string(), format!("{:?}", props.topology)));
                SymbolKind::Field
            }
            NodeKind::Operator(props) => {
                extra.push(("phase".to_string(), format!("{:?}", props.phase)));
                SymbolKind::Operator
            }
            NodeKind::Function(_) => SymbolKind::Function,
            NodeKind::Type(_) => SymbolKind::Type,
            NodeKind::Stratum(_) => SymbolKind::Strata,
            NodeKind::Era(_) => SymbolKind::Era,
            NodeKind::Impulse(_) => SymbolKind::Impulse,
            NodeKind::Fracture(_) => SymbolKind::Fracture,
            NodeKind::Entity(_) => SymbolKind::Entity,
            NodeKind::Member(_) => SymbolKind::Member,
            NodeKind::Chronicle(_) => SymbolKind::Chronicle,
        };

        let (ty, title, symbol) = match &node.kind {
            NodeKind::Signal(p) => (
                Some(format_value_type(&p.value_type)),
                p.title.clone(),
                p.symbol.clone(),
            ),
            NodeKind::Field(p) => (
                Some(format_value_type(&p.value_type)),
                p.title.clone(),
                None,
            ),
            NodeKind::Member(p) => (
                Some(format_value_type(&p.value_type)),
                p.title.clone(),
                p.symbol.clone(),
            ),
            NodeKind::Era(p) => (None, p.title.clone(), None),
            NodeKind::Stratum(p) => (None, p.title.clone(), p.symbol.clone()),
            _ => (None, None, None),
        };

        let path_span = node.span.clone();

        self.symbols.push(IndexedSymbol {
            span: node.span.clone(),
            path_span,
            info: SymbolInfo {
                kind,
                path: node.id.to_string(),
                doc: None, // TODO: IR nodes should carry doc comments
                ty,
                title,
                symbol,
                strata: node.stratum.as_ref().map(|s| s.to_string()),
                extra,
            },
        });

        if let NodeKind::Function(p) = &node.kind {
            self.functions.push(FunctionSignature {
                path: node.id.to_string(),
                doc: None,
                generics: Vec::new(),
                params: p
                    .params
                    .iter()
                    .map(|name| FnParamInfo {
                        name: name.clone(),
                        ty: None,
                    })
                    .collect(),
                return_type: None,
            });
        }
    }

    fn index_const_block(&mut self, block: &ConstBlock) {
        for entry in &block.entries {
            let value_str = format_literal(&entry.value.node);
            let type_str = if let Some(ref unit) = entry.unit {
                format!("{} <{}>", value_str, unit.node)
            } else {
                value_str
            };

            self.symbols.push(IndexedSymbol {
                span: entry.path.span.clone(),
                path_span: entry.path.span.clone(),
                info: SymbolInfo {
                    kind: SymbolKind::Const,
                    path: entry.path.node.to_string(),
                    doc: entry.doc.clone(),
                    ty: Some(type_str),
                    title: None,
                    symbol: None,
                    strata: None,
                    extra: vec![],
                },
            });
        }
    }

    fn index_config_block(&mut self, block: &ConfigBlock) {
        for entry in &block.entries {
            let value_str = format_literal(&entry.value.node);
            let type_str = if let Some(ref unit) = entry.unit {
                format!("{} <{}>", value_str, unit.node)
            } else {
                value_str
            };

            self.symbols.push(IndexedSymbol {
                span: entry.path.span.clone(),
                path_span: entry.path.span.clone(),
                info: SymbolInfo {
                    kind: SymbolKind::Config,
                    path: entry.path.node.to_string(),
                    doc: entry.doc.clone(),
                    ty: Some(type_str),
                    title: None,
                    symbol: None,
                    strata: None,
                    extra: vec![],
                },
            });
        }
    }

    fn index_expr(&mut self, expr: &Spanned<Expr>) {
        let mut collector = RefCollector {
            references: &mut self.references,
        };
        collector.walk(expr);
    }

    pub fn find_at_offset(&self, offset: usize) -> Option<&SymbolInfo> {
        if let Some(reference) = self
            .references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
        {
            return self.find_definition(reference.kind, &reference.target_path);
        }

        self.symbols
            .iter()
            .filter(|s| s.span.contains(&offset))
            .min_by_key(|s| s.span.end - s.span.start)
            .map(|s| &s.info)
    }

    pub fn find_definition(&self, kind: SymbolKind, path: &str) -> Option<&SymbolInfo> {
        self.symbols
            .iter()
            .find(|s| s.info.kind == kind && s.info.path == path)
            .map(|s| &s.info)
    }

    pub fn get_reference_at_offset(&self, offset: usize) -> Option<(SymbolKind, &str)> {
        self.references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
            .map(|r| (r.kind, r.target_path.as_str()))
    }

    pub fn get_references_for_validation(&self) -> Vec<ReferenceValidationInfo> {
        self.references
            .iter()
            .map(|r| ReferenceValidationInfo {
                span: r.span.clone(),
                kind: r.kind,
                target_path: r.target_path.clone(),
            })
            .collect()
    }

    pub fn has_symbol_kind(&self, kind: SymbolKind) -> bool {
        self.symbols.iter().any(|s| s.info.kind == kind)
    }

    pub fn get_all_definitions(&self) -> impl Iterator<Item = (&SymbolInfo, Range<usize>)> {
        self.symbols.iter().map(|s| (&s.info, s.path_span.clone()))
    }

    pub fn find_definition_span(&self, offset: usize) -> Option<Range<usize>> {
        if let Some(reference) = self
            .references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
        {
            return self
                .symbols
                .iter()
                .find(|s| s.info.kind == reference.kind && s.info.path == reference.target_path)
                .map(|s| s.span.clone());
        }

        self.symbols
            .iter()
            .filter(|s| s.span.contains(&offset))
            .min_by_key(|s| s.span.end - s.span.start)
            .map(|s| s.span.clone())
    }

    pub fn get_all_symbols(&self) -> impl Iterator<Item = (&SymbolInfo, &Range<usize>)> {
        self.symbols.iter().map(|s| (&s.info, &s.span))
    }

    pub fn get_completions(&self) -> impl Iterator<Item = CompletionInfo<'_>> {
        self.symbols.iter().map(|s| CompletionInfo {
            kind: s.info.kind,
            path: &s.info.path,
            doc: s.info.doc.as_deref(),
            ty: s.info.ty.as_deref(),
            title: s.info.title.as_deref(),
        })
    }

    pub fn get_all_references(&self) -> impl Iterator<Item = (ReferenceInfo, &Range<usize>)> {
        self.references
            .iter()
            .map(|r| (ReferenceInfo { kind: r.kind }, &r.span))
    }

    pub fn get_symbol_path_spans(&self) -> impl Iterator<Item = (SymbolKind, &Range<usize>)> {
        self.symbols.iter().map(|s| (s.info.kind, &s.path_span))
    }

    pub fn get_function_signature(&self, path: &str) -> Option<&FunctionSignature> {
        self.functions.iter().find(|f| f.path == path)
    }

    pub fn find_references(&self, offset: usize) -> Vec<Range<usize>> {
        let target = if let Some(reference) = self
            .references
            .iter()
            .filter(|r| r.span.contains(&offset))
            .min_by_key(|r| r.span.end - r.span.start)
        {
            Some((reference.kind, reference.target_path.clone()))
        } else if let Some(symbol) = self
            .symbols
            .iter()
            .filter(|s| s.span.contains(&offset))
            .min_by_key(|s| s.span.end - s.span.start)
        {
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
}

fn format_literal(lit: &Literal) -> String {
    match lit {
        Literal::Integer(i) => format!("{}", i),
        Literal::Float(f) => format!("{}", f),
        Literal::String(s) => format!("\"{}\"", s),
        Literal::Bool(b) => format!("{}", b),
    }
}

fn format_value_type(ty: &ValueType) -> String {
    let def = ty.primitive_def();
    let mut positional: Vec<_> = def.params.iter().filter(|p| p.position.is_some()).collect();
    positional.sort_by_key(|spec| spec.position);

    let mut parts = Vec::new();
    for spec in positional {
        if let Some(value) = format_param_value(ty, spec) {
            parts.push(value);
        }
    }

    for spec in def.params.iter().filter(|p| p.position.is_none()) {
        if let Some(value) = format_param_value(ty, spec) {
            parts.push(format!("{}: {}", spec.name, value));
        }
    }

    let mut name = def.name.to_string();
    if !parts.is_empty() {
        name.push('<');
        name.push_str(&parts.join(", "));
        name.push('>');
    }

    if def.name == "Seq" {
        name.push_str(&format!(" ({} constraints)", ty.seq_constraints.len()));
    }

    name
}

fn format_param_value(ty: &ValueType, spec: &PrimitiveParamSpec) -> Option<String> {
    match spec.kind {
        PrimitiveParamKind::Unit => match ty.param_value(PrimitiveParamKind::Unit) {
            Some(ValueTypeParamValue::Unit(unit)) => Some(unit.to_string()),
            _ if spec.optional => Some("1".to_string()),
            _ => None,
        },
        PrimitiveParamKind::Range => match ty.param_value(PrimitiveParamKind::Range) {
            Some(ValueTypeParamValue::Range(range)) => {
                Some(format!("{}..{}", range.min, range.max))
            }
            _ => None,
        },
        PrimitiveParamKind::Magnitude => match ty.param_value(PrimitiveParamKind::Magnitude) {
            Some(ValueTypeParamValue::Magnitude(range)) => {
                Some(format!("{}..{}", range.min, range.max))
            }
            _ => None,
        },
        PrimitiveParamKind::Rows => match ty.param_value(PrimitiveParamKind::Rows) {
            Some(ValueTypeParamValue::Rows(value)) => Some(value.to_string()),
            _ => None,
        },
        PrimitiveParamKind::Cols => match ty.param_value(PrimitiveParamKind::Cols) {
            Some(ValueTypeParamValue::Cols(value)) => Some(value.to_string()),
            _ => None,
        },
        PrimitiveParamKind::Width => match ty.param_value(PrimitiveParamKind::Width) {
            Some(ValueTypeParamValue::Width(value)) => Some(value.to_string()),
            _ => None,
        },
        PrimitiveParamKind::Height => match ty.param_value(PrimitiveParamKind::Height) {
            Some(ValueTypeParamValue::Height(value)) => Some(value.to_string()),
            _ => None,
        },
        PrimitiveParamKind::ElementType => match ty.param_value(PrimitiveParamKind::ElementType) {
            Some(ValueTypeParamValue::ElementType(element)) => Some(format_value_type(element)),
            _ => None,
        },
    }
}

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
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Const,
            target_path: path.to_string(),
        });
        true
    }

    fn visit_config_ref(&mut self, span: Range<usize>, path: &Path) -> bool {
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Config,
            target_path: path.to_string(),
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

    fn visit_emit_signal(&mut self, span: Range<usize>, target: &Path) -> bool {
        self.references.push(SymbolReference {
            span,
            kind: SymbolKind::Signal,
            target_path: target.to_string(),
        });
        true
    }
}

pub fn format_hover_markdown(info: &SymbolInfo) -> String {
    let mut parts = Vec::new();
    let header = format!("**{}.{}**", info.kind.display_name(), info.path);
    parts.push(header);

    if let Some(ref title) = info.title {
        parts.push(format!("*{}*", title));
    }

    if let Some(ref ty) = info.ty {
        parts.push(format!("Type: `{}`", ty));
    }

    if let Some(ref symbol) = info.symbol {
        parts.push(format!("Symbol: {}", symbol));
    }

    if let Some(ref strata) = info.strata {
        parts.push(format!("Strata: `{}`", strata));
    }

    for (key, value) in &info.extra {
        parts.push(format!("{}: {}", key, value));
    }

    if let Some(ref doc) = info.doc {
        parts.push(String::new());
        parts.push("---".to_string());
        parts.push(doc.clone());
    }

    parts.join("\n\n")
}

pub fn get_builtin_hover(name: &str) -> Option<String> {
    let (namespace, function) = name.split_once('.')?;
    kernel_registry::get_in_namespace(namespace, function).map(|k| {
        format!(
            "**{}.{}** (built-in)\n\n`{}`\n\n---\n\n{}",
            k.namespace, k.name, k.signature, k.doc
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_refs_are_indexed() {
        let src = r#"
            world test {}
            era main { : initial }
            strata test {}
            signal core.temp {
                : Scalar<K>
                : strata(test)
                resolve { prev }
            }

            signal thermal.gradient {
                : Scalar<K>
                : strata(test)
                resolve {
                    let core = signal.core.temp in
                    core
                }
            }
        "#;
        let (ast, errors) = continuum_compiler::dsl::parse(src);
        assert!(errors.is_empty());
        let index = SymbolIndex::from_ast(&ast.unwrap());
        let refs = index.get_all_references().collect::<Vec<_>>();
        assert!(refs.iter().any(|(info, _)| info.kind == SymbolKind::Signal));
    }

    #[test]
    fn test_hover_finds_referenced_signal() {
        let src = r#"
            world test {}
            era main { : initial }
            strata test {}
            signal core.temp {
                : Scalar<K>
                : strata(test)
                : title("Core Temperature")
                resolve { prev }
            }

            signal thermal.gradient {
                : Scalar<K>
                : strata(test)
                resolve {
                    let core = signal.core.temp in
                    core
                }
            }
        "#;
        let (ast, errors) = continuum_compiler::dsl::parse(src);
        assert!(errors.is_empty());

        let index = SymbolIndex::from_ast(&ast.unwrap());

        let pos = src.find("signal.core.temp in").unwrap() + 7;

        let info = index.find_at_offset(pos);
        assert!(info.is_some());

        let info = info.unwrap();
        assert_eq!(info.path, "core.temp");
        assert_eq!(info.title, Some("Core Temperature".to_string()));
    }

    #[test]
    fn test_goto_definition() {
        let src = r#"
            world test {}
            era main { : initial }
            strata test {}
            signal core.temp {
                : Scalar<K>
                : strata(test)
                resolve { prev }
            }

            signal thermal.gradient {
                : Scalar<K>
                : strata(test)
                resolve {
                    let core = signal.core.temp in
                    core
                }
            }
        "#;
        let (ast, errors) = continuum_compiler::dsl::parse(src);
        assert!(errors.is_empty());

        let index = SymbolIndex::from_ast(&ast.unwrap());

        let ref_pos = src.find("signal.core.temp in").unwrap() + 7;

        let def_span = index.find_definition_span(ref_pos);
        assert!(def_span.is_some());

        let def_span = def_span.unwrap();
        let def_text = &src[def_span.clone()];
        assert!(def_text.starts_with("signal core.temp"));
    }

    #[test]
    fn test_find_references() {
        let src = r#"
            world test {}
            era main { : initial }
            strata test {}
            signal core.temp {
                : Scalar<K>
                : strata(test)
                resolve { prev }
            }

            signal thermal.gradient {
                : Scalar<K>
                : strata(test)
                resolve {
                    let core = signal.core.temp in
                    core
                }
            }

            fracture test {
                : strata(test)
                when { signal.core.temp < 100.0 }
                emit { signal.core.temp <- 5.0 }
            }
        "#;
        let (ast, errors) = continuum_compiler::dsl::parse(src);
        assert!(errors.is_empty());

        let index = SymbolIndex::from_ast(&ast.unwrap());

        let def_pos = src.find("signal core.temp {").unwrap() + 7;
        let refs = index.find_references(def_pos);

        assert_eq!(refs.len(), 3);
    }

    #[test]
    fn test_document_symbols() {
        let src = r#"
            world test {}
            era main { : initial }
            strata thermal { : stride(1) }
            signal core.temp {
                : Scalar<K>
                : strata(thermal)
                resolve { prev }
            }

            field thermal.display {
                : Scalar<K>
                : strata(thermal)
                measure { signal.core.temp }
            }
        "#;
        let (ast, errors) = continuum_compiler::dsl::parse(src);
        assert!(errors.is_empty());

        let index = SymbolIndex::from_ast(&ast.unwrap());
        let symbols: Vec<_> = index.get_all_symbols().collect();

        assert!(symbols.len() >= 3);
        let paths: Vec<_> = symbols.iter().map(|(info, _)| &info.path).collect();
        assert!(paths.contains(&&"core.temp".to_string()));
        assert!(paths.contains(&&"thermal.display".to_string()));
        assert!(paths.contains(&&"thermal".to_string()));
    }

    #[test]
    fn test_get_completions() {
        let src = r#"
            world test {}
            era main { : initial }
            strata test {}
            /// Temperature of the core.
            signal core.temp {
                : Scalar<K>
                : strata(test)
                : title("Core Temperature")
                resolve { prev }
            }

            /// Surface temperature display.
            field thermal.display {
                : Scalar<K>
                : strata(test)
                measure { signal.core.temp }
            }
        "#;
        let (ast, errors) = continuum_compiler::dsl::parse(src);
        assert!(errors.is_empty());

        let index = SymbolIndex::from_ast(&ast.unwrap());
        let completions: Vec<_> = index.get_completions().collect();

        let signal = completions.iter().find(|c| c.path == "core.temp").unwrap();
        assert_eq!(signal.kind, SymbolKind::Signal);
        assert_eq!(signal.ty, Some("Scalar<K>"));
        assert_eq!(signal.title, Some("Core Temperature"));
    }

    #[test]
    fn test_builtin_hover() {
        assert!(get_builtin_hover("maths.clamp").is_some());
        let hover = get_builtin_hover("maths.clamp").unwrap();
        assert!(hover.contains("clamp"));
        assert!(hover.contains("(built-in)"));
    }
}
