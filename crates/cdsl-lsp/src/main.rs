//! CDSL Language Server
//!
//! A Language Server Protocol implementation for the Continuum DSL.
//!
//! # Features
//!
//! - **Diagnostics**: Real-time parse error reporting and undefined reference warnings
//! - **Hover**: Symbol information with documentation, cross-file lookup, built-in docs
//! - **Go-to-definition**: Jump to signal/field/operator definitions (F12 or Ctrl+Click)
//! - **Find references**: Find all usages of a symbol (Shift+F12)
//! - **Document symbols**: Navigate to any symbol in the file (Ctrl+Shift+O)
//! - **Completion**: World-aware completion with all signals, fields, etc.
//! - **Formatting**: Code formatting on save or on demand
//! - **Inlay hints**: Type hints for symbol references
//! - **Rename**: Rename symbols across all files (F2)
//! - **Folding**: Collapse blocks (signals, fields, functions, etc.)

mod formatter;
mod symbols;

use std::path::Path;

use dashmap::DashMap;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use walkdir::WalkDir;

use continuum_compiler::dsl::ast::{CompilationUnit, Expr, Item, OperatorBody, Spanned};
use continuum_compiler::dsl::parse;
use symbols::{
    ReferenceValidationInfo, SymbolIndex, SymbolKind as CdslSymbolKind, format_hover_markdown,
    get_builtin_hover,
};

/// Semantic token types for syntax highlighting.
/// These indices must match the order in the legend.
const SEMANTIC_TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::VARIABLE,  // 0: signal
    SemanticTokenType::PROPERTY,  // 1: field
    SemanticTokenType::METHOD,    // 2: operator
    SemanticTokenType::FUNCTION,  // 3: function
    SemanticTokenType::TYPE,      // 4: type
    SemanticTokenType::NAMESPACE, // 5: strata
    SemanticTokenType::NAMESPACE, // 6: era
    SemanticTokenType::EVENT,     // 7: impulse
    SemanticTokenType::EVENT,     // 8: fracture
    SemanticTokenType::CLASS,     // 9: chronicle
    SemanticTokenType::CLASS,     // 10: entity
    SemanticTokenType::KEYWORD,   // 11: keyword
    SemanticTokenType::NUMBER,    // 12: number
    SemanticTokenType::STRING,    // 13: string/unit
    SemanticTokenType::COMMENT,   // 14: comment
    SemanticTokenType::PARAMETER, // 15: const
    SemanticTokenType::PARAMETER, // 16: config
    SemanticTokenType::VARIABLE,  // 17: member
    SemanticTokenType::NAMESPACE, // 18: world
];

/// Semantic token modifiers.
const SEMANTIC_TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::DEFINITION,
    SemanticTokenModifier::DECLARATION,
];

/// The CDSL language server backend.
struct Backend {
    /// LSP client for sending notifications.
    client: Client,
    /// Workspace root folders.
    workspace_roots: RwLock<Vec<Url>>,
    /// Cached document contents (for open documents).
    documents: DashMap<Url, String>,
    /// Cached symbol indices for ALL world files (not just open ones).
    symbol_indices: DashMap<Url, SymbolIndex>,
}

impl Backend {
    /// Parse a document and publish diagnostics.
    async fn parse_and_publish_diagnostics(&self, uri: Url, text: &str) {
        let (ast, errors) = parse(text);

        // Convert parse errors to LSP diagnostics
        let mut diagnostics: Vec<Diagnostic> = errors
            .iter()
            .map(|err| {
                let span = err.span();
                let (start_line, start_char) = offset_to_position(text, span.start);
                let (end_line, end_char) = offset_to_position(text, span.end);

                Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("cdsl".to_string()),
                    message: format!("{}", err.reason()),
                    ..Default::default()
                }
            })
            .collect();

        // Build symbol index from AST and collect hints
        if let Some(ref ast) = ast {
            let index = SymbolIndex::from_ast(ast);

            // Get references for validation before inserting
            let refs_to_validate = index.get_references_for_validation();

            self.symbol_indices.insert(uri.clone(), index);

            // Collect clamp usage hints
            let clamp_spans = collect_clamp_usages(ast);
            for span in clamp_spans {
                let (start_line, start_char) = offset_to_position(text, span.start);
                let (end_line, end_char) = offset_to_position(text, span.end);

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(DiagnosticSeverity::HINT),
                    source: Some("cdsl".to_string()),
                    message: "Consider using assertions instead of clamp. \
                        Clamping can hide simulation errors by silently correcting values."
                        .to_string(),
                    tags: Some(vec![DiagnosticTag::UNNECESSARY]),
                    ..Default::default()
                });
            }

            // Check for undefined references
            let undefined = self.find_undefined_references(&refs_to_validate);
            for undef in undefined {
                let (start_line, start_char) = offset_to_position(text, undef.span.start);
                let (end_line, end_char) = offset_to_position(text, undef.span.end);

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(DiagnosticSeverity::WARNING),
                    source: Some("cdsl".to_string()),
                    message: format!(
                        "Undefined {}: '{}'",
                        undef.kind.display_name(),
                        undef.target_path
                    ),
                    ..Default::default()
                });
            }

            // Check for missing required attributes
            let missing_attrs = collect_missing_attributes(ast);
            for attr in missing_attrs {
                let (start_line, start_char) = offset_to_position(text, attr.span.start);
                let (end_line, end_char) = offset_to_position(text, attr.span.end);

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(attr.severity),
                    source: Some("cdsl".to_string()),
                    message: format!("Missing {} for '{}'", attr.attribute, attr.symbol_path),
                    ..Default::default()
                });
            }
        }

        // Publish diagnostics
        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }

    /// Scan workspace for all .cdsl files and index them.
    async fn scan_workspace(&self) {
        let roots = self.workspace_roots.read().await;

        for root in roots.iter() {
            if let Ok(path) = root.to_file_path() {
                self.scan_directory(&path).await;
            }
        }
    }

    /// Scan a directory recursively for .cdsl files.
    async fn scan_directory(&self, dir: &Path) {
        for entry in WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().map(|e| e == "cdsl").unwrap_or(false) {
                self.index_file(path).await;
            }
        }
    }

    /// Index a single .cdsl file (without publishing diagnostics).
    async fn index_file(&self, path: &Path) {
        let uri = match Url::from_file_path(path) {
            Ok(uri) => uri,
            Err(_) => return,
        };

        // Skip if already indexed from an open document
        if self.documents.contains_key(&uri) {
            return;
        }

        // Read and parse the file
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(_) => return,
        };

        let (ast, _errors) = parse(&text);

        if let Some(ref ast) = ast {
            let index = SymbolIndex::from_ast(ast);
            self.symbol_indices.insert(uri, index);
        }
    }

    /// Find undefined references by checking against all indexed symbols.
    ///
    /// Returns references that don't have a corresponding definition in any
    /// indexed file.
    fn find_undefined_references(
        &self,
        refs: &[ReferenceValidationInfo],
    ) -> Vec<ReferenceValidationInfo> {
        refs.iter()
            .filter(|r| {
                // Check if definition exists in any indexed file
                !self.symbol_indices.iter().any(|entry| {
                    entry
                        .value()
                        .find_definition(r.kind, &r.target_path)
                        .is_some()
                })
            })
            .cloned()
            .collect()
    }

    /// Check if any indexed file contains a world definition.
    fn has_world_definition(&self) -> bool {
        self.symbol_indices
            .iter()
            .any(|entry| entry.value().has_symbol_kind(CdslSymbolKind::World))
    }

    /// Find duplicate symbol definitions across all indexed files.
    ///
    /// Returns a list of (uri, path, span, kind) tuples for symbols that are
    /// defined more than once. Only the second and subsequent definitions are
    /// included (the first definition is considered canonical).
    fn find_duplicate_symbols(&self) -> Vec<(Url, String, std::ops::Range<usize>, CdslSymbolKind)> {
        use std::collections::HashMap;

        // Collect all symbols: path -> Vec<(uri, span, kind)>
        let mut all_symbols: HashMap<String, Vec<(Url, std::ops::Range<usize>, CdslSymbolKind)>> =
            HashMap::new();

        for entry in self.symbol_indices.iter() {
            let uri = entry.key().clone();
            for (info, span) in entry.value().get_all_definitions() {
                all_symbols.entry(info.path.clone()).or_default().push((
                    uri.clone(),
                    span,
                    info.kind,
                ));
            }
        }

        // Find paths with more than one definition
        let mut duplicates = Vec::new();
        for (path, occurrences) in all_symbols {
            if occurrences.len() > 1 {
                // Skip the first (canonical) definition, report the rest
                for (uri, span, kind) in occurrences.into_iter().skip(1) {
                    duplicates.push((uri, path.clone(), span, kind));
                }
            }
        }

        duplicates
    }

    /// Find unused symbols (defined but never referenced) in a specific file.
    ///
    /// Returns a list of (path, span, kind) for symbols that have no references
    /// anywhere in the workspace.
    fn find_unused_symbols_in_file(
        &self,
        uri: &Url,
    ) -> Vec<(String, std::ops::Range<usize>, CdslSymbolKind)> {
        let index = match self.symbol_indices.get(uri) {
            Some(idx) => idx,
            None => return vec![],
        };

        // Collect all references across the entire workspace
        let mut all_refs: std::collections::HashSet<(String, CdslSymbolKind)> =
            std::collections::HashSet::new();

        for entry in self.symbol_indices.iter() {
            for ref_info in entry.value().get_references_for_validation().iter() {
                all_refs.insert((ref_info.target_path.clone(), ref_info.kind));
            }
        }

        // Find definitions in this file that have no references
        let mut unused = Vec::new();
        for (info, span) in index.get_all_definitions() {
            // Skip certain kinds that are entry points or config
            match info.kind {
                CdslSymbolKind::World
                | CdslSymbolKind::Era
                | CdslSymbolKind::Const
                | CdslSymbolKind::Config => continue,
                _ => {}
            }

            if !all_refs.contains(&(info.path.clone(), info.kind)) {
                unused.push((info.path.clone(), span, info.kind));
            }
        }

        unused
    }

    /// Run workspace-level validation and publish diagnostics.
    ///
    /// This should be called after workspace scanning is complete.
    async fn validate_workspace(&self) {
        // Check for missing world definition
        if !self.has_world_definition() {
            // Find the first indexed file to publish the error
            if let Some(entry) = self.symbol_indices.iter().next() {
                let uri = entry.key().clone();
                let text = self
                    .documents
                    .get(&uri)
                    .map(|v| v.clone())
                    .unwrap_or_default();

                // Get existing diagnostics for this file (if any)
                let mut diagnostics = Vec::new();

                // Add error at position 0,0
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position::new(0, 0),
                        end: Position::new(0, 0),
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("cdsl".to_string()),
                    message: "Missing world definition. Every world must have exactly one \
                        `world.name { }` declaration."
                        .to_string(),
                    ..Default::default()
                });

                // Re-parse to get any existing diagnostics
                if !text.is_empty() {
                    let (_, errors) = parse(&text);
                    for err in &errors {
                        let span = err.span();
                        let (start_line, start_char) = offset_to_position(&text, span.start);
                        let (end_line, end_char) = offset_to_position(&text, span.end);

                        diagnostics.push(Diagnostic {
                            range: Range {
                                start: Position::new(start_line, start_char),
                                end: Position::new(end_line, end_char),
                            },
                            severity: Some(DiagnosticSeverity::ERROR),
                            source: Some("cdsl".to_string()),
                            message: format!("{}", err.reason()),
                            ..Default::default()
                        });
                    }
                }

                self.client
                    .publish_diagnostics(uri, diagnostics, None)
                    .await;
            }
        }

        // Check for duplicate symbols
        let duplicates = self.find_duplicate_symbols();
        // Group duplicates by URI
        let mut duplicates_by_uri: std::collections::HashMap<
            Url,
            Vec<(String, std::ops::Range<usize>, CdslSymbolKind)>,
        > = std::collections::HashMap::new();

        for (uri, path, span, kind) in duplicates {
            duplicates_by_uri
                .entry(uri)
                .or_default()
                .push((path, span, kind));
        }

        // Publish warnings for each file with duplicates
        for (uri, file_duplicates) in duplicates_by_uri {
            let text = self
                .documents
                .get(&uri)
                .map(|v| v.clone())
                .or_else(|| {
                    uri.to_file_path()
                        .ok()
                        .and_then(|p| std::fs::read_to_string(p).ok())
                })
                .unwrap_or_default();

            // Re-parse to get existing diagnostics
            let (_, errors) = parse(&text);
            let mut diagnostics: Vec<Diagnostic> = errors
                .iter()
                .map(|err| {
                    let span = err.span();
                    let (start_line, start_char) = offset_to_position(&text, span.start);
                    let (end_line, end_char) = offset_to_position(&text, span.end);

                    Diagnostic {
                        range: Range {
                            start: Position::new(start_line, start_char),
                            end: Position::new(end_line, end_char),
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        source: Some("cdsl".to_string()),
                        message: format!("{}", err.reason()),
                        ..Default::default()
                    }
                })
                .collect();

            // Add duplicate warnings
            for (path, span, kind) in file_duplicates {
                let (start_line, start_char) = offset_to_position(&text, span.start);
                let (end_line, end_char) = offset_to_position(&text, span.end);

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(DiagnosticSeverity::WARNING),
                    source: Some("cdsl".to_string()),
                    message: format!(
                        "Duplicate {} definition: '{}' is already defined elsewhere",
                        kind.display_name(),
                        path
                    ),
                    ..Default::default()
                });
            }

            self.client
                .publish_diagnostics(uri, diagnostics, None)
                .await;
        }

        // Check for unused symbols in all files
        for entry in self.symbol_indices.iter() {
            let uri = entry.key().clone();
            let unused = self.find_unused_symbols_in_file(&uri);

            if unused.is_empty() {
                continue;
            }

            let text = self
                .documents
                .get(&uri)
                .map(|v| v.clone())
                .or_else(|| {
                    uri.to_file_path()
                        .ok()
                        .and_then(|p| std::fs::read_to_string(p).ok())
                })
                .unwrap_or_default();

            // Re-parse to get existing diagnostics
            let (_, errors) = parse(&text);
            let mut diagnostics: Vec<Diagnostic> = errors
                .iter()
                .map(|err| {
                    let span = err.span();
                    let (start_line, start_char) = offset_to_position(&text, span.start);
                    let (end_line, end_char) = offset_to_position(&text, span.end);

                    Diagnostic {
                        range: Range {
                            start: Position::new(start_line, start_char),
                            end: Position::new(end_line, end_char),
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        source: Some("cdsl".to_string()),
                        message: format!("{}", err.reason()),
                        ..Default::default()
                    }
                })
                .collect();

            // Add unused symbol hints
            for (path, span, kind) in unused {
                let (start_line, start_char) = offset_to_position(&text, span.start);
                let (end_line, end_char) = offset_to_position(&text, span.end);

                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(DiagnosticSeverity::HINT),
                    source: Some("cdsl".to_string()),
                    message: format!(
                        "Unused {}: '{}' is never referenced",
                        kind.display_name(),
                        path
                    ),
                    tags: Some(vec![DiagnosticTag::UNNECESSARY]),
                    ..Default::default()
                });
            }

            self.client
                .publish_diagnostics(uri, diagnostics, None)
                .await;
        }
    }
}

/// Convert a byte offset to line/column position.
fn offset_to_position(text: &str, offset: usize) -> (u32, u32) {
    let mut line = 0u32;
    let mut col = 0u32;
    let mut current_offset = 0;

    for ch in text.chars() {
        if current_offset >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
        current_offset += ch.len_utf8();
    }

    (line, col)
}

/// Convert an LSP position (line/column) to a byte offset.
fn position_to_offset(text: &str, position: Position) -> usize {
    let mut current_line = 0u32;
    let mut current_col = 0u32;
    let mut offset = 0;

    for ch in text.chars() {
        if current_line == position.line && current_col == position.character {
            return offset;
        }
        if ch == '\n' {
            if current_line == position.line {
                // Position is beyond line end, return end of line
                return offset;
            }
            current_line += 1;
            current_col = 0;
        } else {
            current_col += 1;
        }
        offset += ch.len_utf8();
    }

    offset
}

/// Get the identifier at cursor position.
///
/// Returns the full identifier under or before the cursor (e.g., "clamp" or "math.lerp").
/// Used for hover on built-in functions.
fn get_word_at_cursor(text: &str, offset: usize) -> Option<String> {
    if offset > text.len() {
        return None;
    }

    // Find start of identifier (going backwards)
    let before = &text[..offset];
    let start = before
        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + 1)
        .unwrap_or(0);

    // Find end of identifier (going forwards)
    let after = &text[offset..];
    let end_rel = after
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(after.len());
    let end = offset + end_rel;

    if start >= end {
        return None;
    }

    let word = &text[start..end];
    if word.is_empty() {
        None
    } else {
        // Extract just the last segment (the function name) for builtin lookup
        Some(word.split('.').next_back().unwrap_or(word).to_string())
    }
}

/// Get the completion prefix (the text being typed before the cursor).
///
/// Returns the text from the start of the current "word" to the cursor position.
/// A word includes alphanumeric characters, underscores, and dots.
fn get_completion_prefix(text: &str, position: Position) -> String {
    let offset = position_to_offset(text, position);
    let before_cursor = &text[..offset];

    // Find the start of the current word (including dots for paths)
    let start = before_cursor
        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + 1)
        .unwrap_or(0);

    before_cursor[start..].to_string()
}

/// Detect if the prefix starts with a symbol kind (e.g., "signal.", "field.").
///
/// Returns the kind and prefix length if found.
fn detect_kind_prefix(prefix: &str) -> Option<(CdslSymbolKind, usize)> {
    let prefixes = [
        ("signal.", CdslSymbolKind::Signal),
        ("field.", CdslSymbolKind::Field),
        ("operator.", CdslSymbolKind::Operator),
        ("fn.", CdslSymbolKind::Function),
        ("type.", CdslSymbolKind::Type),
        ("strata.", CdslSymbolKind::Strata),
        ("era.", CdslSymbolKind::Era),
        ("impulse.", CdslSymbolKind::Impulse),
        ("fracture.", CdslSymbolKind::Fracture),
        ("chronicle.", CdslSymbolKind::Chronicle),
        ("entity.", CdslSymbolKind::Entity),
    ];

    for (pat, kind) in prefixes {
        if prefix.starts_with(pat) {
            return Some((kind, pat.len()));
        }
    }
    None
}

/// Find function call context for signature help.
///
/// Searches backwards from cursor to find if we're inside a `fn.path.name(...)` call.
/// Returns the function path (without "fn." prefix) and the active parameter index.
fn find_function_call_context(text_before: &str) -> Option<(String, usize)> {
    // Find the most recent unclosed '(' that follows "fn."
    let mut paren_depth = 0;
    let mut last_fn_call_start = None;

    // Scan backwards through the text
    let chars: Vec<char> = text_before.chars().collect();
    let mut i = chars.len();

    while i > 0 {
        i -= 1;
        match chars[i] {
            ')' => paren_depth += 1,
            '(' => {
                if paren_depth > 0 {
                    paren_depth -= 1;
                } else {
                    // Found an unclosed '(' - check if it's a fn call
                    // Look backwards for "fn." pattern
                    let before_paren = &text_before[..i];
                    if let Some(fn_start) = before_paren.rfind("fn.") {
                        // Extract the function path between "fn." and "("
                        let path_start = fn_start + 3; // Skip "fn."
                        let path = &text_before[path_start..i];

                        // Validate it's a valid path (alphanumeric, underscore, dots)
                        if path
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                            && !path.is_empty()
                        {
                            last_fn_call_start = Some((path.to_string(), i + 1));
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // If we found a function call, count commas to find active parameter
    if let Some((fn_path, args_start)) = last_fn_call_start {
        let args_text = &text_before[args_start..];
        let mut comma_count = 0usize;
        let mut depth = 0usize;

        for ch in args_text.chars() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => comma_count += 1,
                _ => {}
            }
        }

        return Some((fn_path, comma_count));
    }

    None
}

/// Collect all spans where `clamp` is used in the AST.
///
/// Detects both method calls (value.clamp(min, max)) and function calls.
fn collect_clamp_usages(ast: &CompilationUnit) -> Vec<std::ops::Range<usize>> {
    let mut spans = Vec::new();

    for item in &ast.items {
        collect_clamp_in_item(&item.node, &mut spans);
    }

    spans
}

fn collect_clamp_in_item(item: &Item, spans: &mut Vec<std::ops::Range<usize>>) {
    match item {
        Item::SignalDef(def) => {
            if let Some(ref resolve) = def.resolve {
                collect_clamp_in_expr(&resolve.body, spans);
            }
            if let Some(ref assertions) = def.assertions {
                for assertion in &assertions.assertions {
                    collect_clamp_in_expr(&assertion.condition, spans);
                }
            }
        }
        Item::FieldDef(def) => {
            if let Some(ref measure) = def.measure {
                collect_clamp_in_expr(&measure.body, spans);
            }
        }
        Item::OperatorDef(def) => {
            if let Some(ref body) = def.body {
                use OperatorBody;
                match body {
                    OperatorBody::Warmup(expr)
                    | OperatorBody::Collect(expr)
                    | OperatorBody::Measure(expr) => {
                        collect_clamp_in_expr(expr, spans);
                    }
                }
            }
        }
        Item::FnDef(def) => {
            collect_clamp_in_expr(&def.body, spans);
        }
        Item::ImpulseDef(def) => {
            if let Some(ref apply) = def.apply {
                collect_clamp_in_expr(&apply.body, spans);
            }
        }
        Item::FractureDef(def) => {
            for condition in &def.conditions {
                collect_clamp_in_expr(condition, spans);
            }
            for emit in &def.emit {
                collect_clamp_in_expr(&emit.value, spans);
            }
        }
        Item::ChronicleDef(def) => {
            if let Some(ref observe) = def.observe {
                for handler in &observe.handlers {
                    collect_clamp_in_expr(&handler.condition, spans);
                    for (_, field_expr) in &handler.event_fields {
                        collect_clamp_in_expr(field_expr, spans);
                    }
                }
            }
        }
        Item::EraDef(def) => {
            for transition in &def.transitions {
                for condition in &transition.conditions {
                    collect_clamp_in_expr(condition, spans);
                }
            }
        }
        _ => {}
    }
}

fn collect_clamp_in_expr(expr: &Spanned<Expr>, spans: &mut Vec<std::ops::Range<usize>>) {
    match &expr.node {
        Expr::MethodCall {
            object,
            method,
            args,
        } => {
            if method == "clamp" {
                spans.push(expr.span.clone());
            }
            collect_clamp_in_expr(object, spans);
            for arg in args {
                collect_clamp_in_expr(&arg.value, spans);
            }
        }
        Expr::Call { function, args } => {
            // Check if function is a path ending in "clamp"
            if let Expr::Path(path) = &function.node {
                if path.segments.last().map(|s| s.as_str()) == Some("clamp") {
                    spans.push(expr.span.clone());
                }
            }
            collect_clamp_in_expr(function, spans);
            for arg in args {
                collect_clamp_in_expr(&arg.value, spans);
            }
        }
        Expr::Binary { left, right, .. } => {
            collect_clamp_in_expr(left, spans);
            collect_clamp_in_expr(right, spans);
        }
        Expr::Unary { operand, .. } => {
            collect_clamp_in_expr(operand, spans);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_clamp_in_expr(condition, spans);
            collect_clamp_in_expr(then_branch, spans);
            if let Some(else_expr) = else_branch {
                collect_clamp_in_expr(else_expr, spans);
            }
        }
        Expr::Let { value, body, .. } => {
            collect_clamp_in_expr(value, spans);
            collect_clamp_in_expr(body, spans);
        }
        Expr::Block(exprs) => {
            for e in exprs {
                collect_clamp_in_expr(e, spans);
            }
        }
        Expr::FieldAccess { object, .. } => {
            collect_clamp_in_expr(object, spans);
        }
        Expr::For { iter, body, .. } => {
            collect_clamp_in_expr(iter, spans);
            collect_clamp_in_expr(body, spans);
        }
        Expr::Struct(fields) => {
            for (_, value) in fields {
                collect_clamp_in_expr(value, spans);
            }
        }
        Expr::EmitSignal { value, .. } => {
            collect_clamp_in_expr(value, spans);
        }
        Expr::EmitField {
            position, value, ..
        } => {
            collect_clamp_in_expr(position, spans);
            collect_clamp_in_expr(value, spans);
        }
        Expr::Map { sequence, function } => {
            collect_clamp_in_expr(sequence, spans);
            collect_clamp_in_expr(function, spans);
        }
        Expr::Fold {
            sequence,
            init,
            function,
        } => {
            collect_clamp_in_expr(sequence, spans);
            collect_clamp_in_expr(init, spans);
            collect_clamp_in_expr(function, spans);
        }
        Expr::Filter { predicate, .. } => {
            collect_clamp_in_expr(predicate, spans);
        }
        Expr::First { predicate, .. } => {
            collect_clamp_in_expr(predicate, spans);
        }
        Expr::Aggregate { body, .. } => {
            collect_clamp_in_expr(body, spans);
        }
        Expr::EntityAccess { instance, .. } => {
            collect_clamp_in_expr(instance, spans);
        }
        Expr::Nearest { position, .. } => {
            collect_clamp_in_expr(position, spans);
        }
        Expr::Within {
            position, radius, ..
        } => {
            collect_clamp_in_expr(position, spans);
            collect_clamp_in_expr(radius, spans);
        }
        // Terminal expressions - no children
        Expr::Literal(_)
        | Expr::LiteralWithUnit { .. }
        | Expr::Path(_)
        | Expr::Prev
        | Expr::PrevField(_)
        | Expr::DtRaw
        | Expr::Payload
        | Expr::PayloadField(_)
        | Expr::SignalRef(_)
        | Expr::ConstRef(_)
        | Expr::ConfigRef(_)
        | Expr::FieldRef(_)
        | Expr::Collected
        | Expr::MathConst(_)
        | Expr::SelfField(_)
        | Expr::EntityRef(_)
        | Expr::Other(_)
        | Expr::Pairs(_) => {}
    }
}

/// Information about a missing required attribute.
struct MissingAttributeInfo {
    span: std::ops::Range<usize>,
    symbol_path: String,
    attribute: &'static str,
    severity: DiagnosticSeverity,
}

/// Collect diagnostics for missing required attributes in the AST.
fn collect_missing_attributes(ast: &CompilationUnit) -> Vec<MissingAttributeInfo> {
    let mut diagnostics = Vec::new();

    for item in &ast.items {
        match &item.node {
            Item::SignalDef(def) => {
                // Signal should have strata
                if def.strata.is_none() {
                    diagnostics.push(MissingAttributeInfo {
                        span: def.path.span.clone(),
                        symbol_path: def.path.node.to_string(),
                        attribute: "strata",
                        severity: DiagnosticSeverity::WARNING,
                    });
                }
                // Signal should have resolve block
                if def.resolve.is_none() {
                    diagnostics.push(MissingAttributeInfo {
                        span: def.path.span.clone(),
                        symbol_path: def.path.node.to_string(),
                        attribute: "resolve block",
                        severity: DiagnosticSeverity::WARNING,
                    });
                }
            }
            Item::FieldDef(def) => {
                // Field should have strata
                if def.strata.is_none() {
                    diagnostics.push(MissingAttributeInfo {
                        span: def.path.span.clone(),
                        symbol_path: def.path.node.to_string(),
                        attribute: "strata",
                        severity: DiagnosticSeverity::WARNING,
                    });
                }
                // Field should have measure block
                if def.measure.is_none() {
                    diagnostics.push(MissingAttributeInfo {
                        span: def.path.span.clone(),
                        symbol_path: def.path.node.to_string(),
                        attribute: "measure block",
                        severity: DiagnosticSeverity::WARNING,
                    });
                }
            }
            Item::OperatorDef(def) => {
                // Operator should have strata
                if def.strata.is_none() {
                    diagnostics.push(MissingAttributeInfo {
                        span: def.path.span.clone(),
                        symbol_path: def.path.node.to_string(),
                        attribute: "strata",
                        severity: DiagnosticSeverity::WARNING,
                    });
                }
            }
            Item::MemberDef(def) => {
                // Member should have strata
                if def.strata.is_none() {
                    diagnostics.push(MissingAttributeInfo {
                        span: def.path.span.clone(),
                        symbol_path: def.path.node.to_string(),
                        attribute: "strata",
                        severity: DiagnosticSeverity::WARNING,
                    });
                }
            }
            _ => {}
        }
    }

    diagnostics
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
        // Store workspace roots for scanning
        let mut roots = self.workspace_roots.write().await;
        if let Some(folders) = params.workspace_folders {
            for folder in folders {
                roots.push(folder.uri);
            }
        } else if let Some(root_uri) = params.root_uri {
            roots.push(root_uri);
        }
        drop(roots);

        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                document_formatting_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                references_provider: Some(OneOf::Left(true)),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                semantic_tokens_provider: Some(
                    SemanticTokensServerCapabilities::SemanticTokensOptions(
                        SemanticTokensOptions {
                            legend: SemanticTokensLegend {
                                token_types: SEMANTIC_TOKEN_TYPES.to_vec(),
                                token_modifiers: SEMANTIC_TOKEN_MODIFIERS.to_vec(),
                            },
                            full: Some(SemanticTokensFullOptions::Bool(true)),
                            range: None,
                            ..Default::default()
                        },
                    ),
                ),
                inlay_hint_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Right(RenameOptions {
                    prepare_provider: Some(true),
                    work_done_progress_options: Default::default(),
                })),
                folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "cdsl-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "CDSL language server initialized")
            .await;

        // Scan workspace for all .cdsl files
        self.scan_workspace().await;

        let count = self.symbol_indices.len();
        self.client
            .log_message(
                MessageType::INFO,
                format!("Indexed {} .cdsl files in workspace", count),
            )
            .await;

        // Run workspace-level validation (missing world, duplicates)
        self.validate_workspace().await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;

        self.documents.insert(uri.clone(), text.clone());
        self.parse_and_publish_diagnostics(uri, &text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;

        // We use FULL sync, so there's only one change with the full content
        if let Some(change) = params.content_changes.into_iter().next() {
            self.documents.insert(uri.clone(), change.text.clone());
            self.parse_and_publish_diagnostics(uri, &change.text).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);
        self.symbol_indices.remove(&uri);

        // Clear diagnostics
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        // Get document text
        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        // Get the prefix being typed (text from start of word to cursor)
        let prefix = get_completion_prefix(&doc, position);

        // Check if prefix matches a symbol kind (e.g., "signal.", "field.")
        let kind_filter = detect_kind_prefix(&prefix);

        let mut items = Vec::new();

        // If we have a kind prefix, show hierarchical path completion
        if let Some((kind, prefix_len)) = kind_filter {
            // Get the path part after "signal." etc.
            let path_prefix = &prefix[prefix_len..];
            // Split by dots, filtering empty segments (from trailing dots like "core.")
            let prefix_segments: Vec<&str> =
                path_prefix.split('.').filter(|s| !s.is_empty()).collect();
            let prefix_depth = prefix_segments.len();

            // Track unique next segments: segment -> Option<(full_path, type, title, doc)>
            struct CompletionData {
                full_path: String,
                ty: Option<String>,
                title: Option<String>,
                doc: Option<String>,
            }
            let mut seen_segments: std::collections::HashMap<String, Option<CompletionData>> =
                std::collections::HashMap::new();

            // Collect completions from ALL indexed files in the world
            for entry in self.symbol_indices.iter() {
                for info in entry.value().get_completions().filter(|c| c.kind == kind) {
                    let path_segments: Vec<&str> = info.path.split('.').collect();

                    // Check if this path matches the prefix
                    let matches_prefix = prefix_segments
                        .iter()
                        .zip(path_segments.iter())
                        .all(|(p, s)| s.starts_with(*p));

                    if !matches_prefix {
                        continue;
                    }

                    // Get the next segment to show
                    if let Some(next_segment) = path_segments.get(prefix_depth) {
                        let is_final = path_segments.len() == prefix_depth + 1;

                        // Build the completion info
                        if is_final {
                            // This is the final segment - show full info
                            seen_segments.insert(
                                next_segment.to_string(),
                                Some(CompletionData {
                                    full_path: info.path.to_string(),
                                    ty: info.ty.map(|s| s.to_string()),
                                    title: info.title.map(|s| s.to_string()),
                                    doc: info.doc.map(|s| s.to_string()),
                                }),
                            );
                        } else {
                            // Intermediate segment - just a namespace
                            seen_segments
                                .entry(next_segment.to_string())
                                .or_insert(None);
                        }
                    }
                }
            }

            // Build completion items from unique segments
            for (segment, data) in seen_segments {
                let is_final = data.is_some();
                let (detail, documentation) = match data {
                    Some(d) => {
                        // Build detail line
                        let detail = match (&d.ty, &d.title) {
                            (Some(ty), Some(title)) => format!("{} — {}", ty, title),
                            (Some(ty), None) => ty.clone(),
                            (None, Some(title)) => title.clone(),
                            (None, None) => kind.display_name().to_string(),
                        };

                        // Build rich markdown documentation
                        let mut doc_parts =
                            vec![format!("**{}.{}**", kind.display_name(), d.full_path)];
                        if let Some(ref title) = d.title {
                            doc_parts.push(format!("*{}*", title));
                        }
                        if let Some(ref ty) = d.ty {
                            doc_parts.push(format!("Type: `{}`", ty));
                        }
                        if let Some(ref doc) = d.doc {
                            doc_parts.push("---".to_string());
                            doc_parts.push(doc.clone());
                        }

                        let doc_markdown = doc_parts.join("\n\n");
                        (detail, Some(doc_markdown))
                    }
                    None => ("namespace".to_string(), None),
                };

                items.push(CompletionItem {
                    label: segment,
                    kind: Some(if is_final {
                        cdsl_kind_to_completion_kind(kind)
                    } else {
                        CompletionItemKind::MODULE
                    }),
                    detail: Some(detail),
                    documentation: documentation.map(|d| {
                        Documentation::MarkupContent(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: d,
                        })
                    }),
                    ..Default::default()
                });
            }
        } else {
            // No kind prefix - show keywords and all symbols
            items.extend(vec![
                // Top-level declarations
                completion_item("signal", CompletionItemKind::KEYWORD, "Signal declaration"),
                completion_item("field", CompletionItemKind::KEYWORD, "Field declaration"),
                completion_item(
                    "fracture",
                    CompletionItemKind::KEYWORD,
                    "Fracture declaration",
                ),
                completion_item(
                    "impulse",
                    CompletionItemKind::KEYWORD,
                    "Impulse declaration",
                ),
                completion_item(
                    "chronicle",
                    CompletionItemKind::KEYWORD,
                    "Chronicle declaration",
                ),
                completion_item("entity", CompletionItemKind::KEYWORD, "Entity declaration"),
                completion_item(
                    "operator",
                    CompletionItemKind::KEYWORD,
                    "Operator declaration",
                ),
                completion_item("strata", CompletionItemKind::KEYWORD, "Strata declaration"),
                completion_item("era", CompletionItemKind::KEYWORD, "Era declaration"),
                completion_item("const", CompletionItemKind::KEYWORD, "Const block"),
                completion_item("config", CompletionItemKind::KEYWORD, "Config block"),
                completion_item("fn", CompletionItemKind::KEYWORD, "Function definition"),
                completion_item("type", CompletionItemKind::KEYWORD, "Type definition"),
                // Block keywords
                completion_item("resolve", CompletionItemKind::KEYWORD, "Resolve block"),
                completion_item("measure", CompletionItemKind::KEYWORD, "Measure block"),
                completion_item("when", CompletionItemKind::KEYWORD, "When condition"),
                completion_item("emit", CompletionItemKind::KEYWORD, "Emit block"),
                completion_item("assert", CompletionItemKind::KEYWORD, "Assert block"),
                // Expression keywords
                completion_item("if", CompletionItemKind::KEYWORD, "Conditional expression"),
                completion_item("else", CompletionItemKind::KEYWORD, "Else branch"),
                completion_item("let", CompletionItemKind::KEYWORD, "Let binding"),
                completion_item("in", CompletionItemKind::KEYWORD, "In expression"),
                completion_item("prev", CompletionItemKind::VARIABLE, "Previous value"),
                completion_item("dt_raw", CompletionItemKind::VARIABLE, "Raw time delta"),
                completion_item("collected", CompletionItemKind::VARIABLE, "Collected value"),
                completion_item("payload", CompletionItemKind::VARIABLE, "Impulse payload"),
                // Types
                completion_item("Scalar", CompletionItemKind::TYPE_PARAMETER, "Scalar type"),
                completion_item("Vector", CompletionItemKind::TYPE_PARAMETER, "Vector type"),
                completion_item("Tensor", CompletionItemKind::TYPE_PARAMETER, "Tensor type"),
                // Built-in functions
                completion_item(
                    "clamp",
                    CompletionItemKind::FUNCTION,
                    "Clamp value to range",
                ),
                completion_item("min", CompletionItemKind::FUNCTION, "Minimum of values"),
                completion_item("max", CompletionItemKind::FUNCTION, "Maximum of values"),
                completion_item("abs", CompletionItemKind::FUNCTION, "Absolute value"),
                completion_item("exp", CompletionItemKind::FUNCTION, "Exponential"),
                completion_item("ln", CompletionItemKind::FUNCTION, "Natural logarithm"),
                completion_item("log", CompletionItemKind::FUNCTION, "Logarithm"),
                completion_item("sqrt", CompletionItemKind::FUNCTION, "Square root"),
                completion_item("sin", CompletionItemKind::FUNCTION, "Sine"),
                completion_item("cos", CompletionItemKind::FUNCTION, "Cosine"),
                completion_item("tan", CompletionItemKind::FUNCTION, "Tangent"),
            ]);
            // Note: Full symbol paths are NOT shown here.
            // Type "signal.", "field.", etc. to get hierarchical path completion.
        }

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        // Get document and symbol index
        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        // Convert LSP position to byte offset
        let offset = position_to_offset(&doc, position);

        // Find the definition span
        if let Some(def_span) = index.find_definition_span(offset) {
            let (start_line, start_char) = offset_to_position(&doc, def_span.start);
            let (end_line, end_char) = offset_to_position(&doc, def_span.end);

            return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                uri: uri.clone(),
                range: Range {
                    start: Position::new(start_line, start_char),
                    end: Position::new(end_line, end_char),
                },
            })));
        }

        Ok(None)
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        // Get document content
        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        // Convert LSP position to byte offset
        let offset = position_to_offset(&doc, position);

        // Try to find symbol in current file's index
        if let Some(index) = self.symbol_indices.get(uri) {
            // First try direct lookup in current file
            if let Some(info) = index.find_at_offset(offset) {
                let markdown = format_hover_markdown(info);
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: markdown,
                    }),
                    range: None,
                }));
            }

            // Check if we're on a reference to a symbol defined in another file
            if let Some((kind, path)) = index.get_reference_at_offset(offset) {
                // Search all indexed files for the definition
                for entry in self.symbol_indices.iter() {
                    if let Some(info) = entry.value().find_definition(kind, path) {
                        let markdown = format_hover_markdown(info);
                        return Ok(Some(Hover {
                            contents: HoverContents::Markup(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: markdown,
                            }),
                            range: None,
                        }));
                    }
                }
            }
        }

        // Try to find a built-in function at cursor
        if let Some(word) = get_word_at_cursor(&doc, offset) {
            if let Some(hover_md) = get_builtin_hover(&word) {
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: hover_md,
                    }),
                    range: None,
                }));
            }
        }

        Ok(None)
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = &params.text_document.uri;

        // Get the document content
        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        // Format the document
        let formatted = formatter::format(&doc);

        // If no changes, return empty
        if formatted == doc {
            return Ok(Some(vec![]));
        }

        // Calculate the range for the entire document
        let line_count = doc.lines().count() as u32;
        let last_line_len = doc.lines().last().map(|l| l.len() as u32).unwrap_or(0);

        let edit = TextEdit {
            range: Range {
                start: Position::new(0, 0),
                end: Position::new(line_count, last_line_len),
            },
            new_text: formatted,
        };

        Ok(Some(vec![edit]))
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let symbols: Vec<SymbolInformation> = index
            .get_all_symbols()
            .map(|(info, span)| {
                let (start_line, start_char) = offset_to_position(&doc, span.start);
                let (end_line, end_char) = offset_to_position(&doc, span.end);

                #[allow(deprecated)] // location field is deprecated but required
                SymbolInformation {
                    name: format!("{}.{}", info.kind.display_name(), info.path),
                    kind: symbol_kind_to_lsp(info.kind),
                    tags: None,
                    deprecated: None,
                    location: Location {
                        uri: uri.clone(),
                        range: Range {
                            start: Position::new(start_line, start_char),
                            end: Position::new(end_line, end_char),
                        },
                    },
                    container_name: None,
                }
            })
            .collect();

        Ok(Some(DocumentSymbolResponse::Flat(symbols)))
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);
        let ref_spans = index.find_references(offset);

        if ref_spans.is_empty() {
            return Ok(None);
        }

        let locations: Vec<Location> = ref_spans
            .into_iter()
            .map(|span| {
                let (start_line, start_char) = offset_to_position(&doc, span.start);
                let (end_line, end_char) = offset_to_position(&doc, span.end);

                Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                }
            })
            .collect();

        Ok(Some(locations))
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        // Get text up to cursor position
        let offset = position_to_offset(&doc, position);
        let text_before = &doc[..offset];

        // Find the function call context by looking for "fn.path.name(" pattern
        // and counting parentheses/commas
        let (fn_path, active_param) = match find_function_call_context(text_before) {
            Some(ctx) => ctx,
            None => return Ok(None),
        };

        // Search all indexed files for the function
        for entry in self.symbol_indices.iter() {
            if let Some(sig) = entry.value().get_function_signature(&fn_path) {
                // Build the signature label
                let params_str: Vec<String> = sig
                    .params
                    .iter()
                    .map(|p| {
                        if let Some(ref ty) = p.ty {
                            format!("{}: {}", p.name, ty)
                        } else {
                            p.name.clone()
                        }
                    })
                    .collect();

                let generics_str = if sig.generics.is_empty() {
                    String::new()
                } else {
                    format!("<{}>", sig.generics.join(", "))
                };

                let return_str = sig
                    .return_type
                    .as_ref()
                    .map(|t| format!(" -> {}", t))
                    .unwrap_or_default();

                let label = format!(
                    "fn.{}{}({}){}",
                    sig.path,
                    generics_str,
                    params_str.join(", "),
                    return_str
                );

                // Build parameter information with offsets
                let mut param_infos = Vec::new();
                let params_start = label.find('(').unwrap_or(0) + 1;
                let mut current_offset = params_start;

                for (i, param_str) in params_str.iter().enumerate() {
                    let param_start = current_offset;
                    let param_end = param_start + param_str.len();

                    param_infos.push(ParameterInformation {
                        label: ParameterLabel::LabelOffsets([param_start as u32, param_end as u32]),
                        documentation: None,
                    });

                    current_offset = param_end;
                    if i < params_str.len() - 1 {
                        current_offset += 2; // ", "
                    }
                }

                let signature = SignatureInformation {
                    label,
                    documentation: sig.doc.as_ref().map(|d| {
                        Documentation::MarkupContent(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: d.clone(),
                        })
                    }),
                    parameters: Some(param_infos),
                    active_parameter: Some(
                        active_param.min(sig.params.len().saturating_sub(1)) as u32
                    ),
                };

                return Ok(Some(SignatureHelp {
                    signatures: vec![signature],
                    active_signature: Some(0),
                    active_parameter: Some(
                        active_param.min(sig.params.len().saturating_sub(1)) as u32
                    ),
                }));
            }
        }

        Ok(None)
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = params.query.to_lowercase();
        let mut results = Vec::new();

        // Search all indexed files for matching symbols
        for entry in self.symbol_indices.iter() {
            let uri = entry.key().clone();
            let index = entry.value();

            let doc = match self.documents.get(&uri) {
                Some(doc) => doc.clone(),
                None => continue,
            };

            for (info, span) in index.get_all_definitions() {
                // Match symbol path against query (fuzzy match)
                let path_lower = info.path.to_lowercase();
                if query.is_empty() || path_lower.contains(&query) {
                    let (start_line, start_char) = offset_to_position(&doc, span.start);
                    let (end_line, end_char) = offset_to_position(&doc, span.end);

                    #[allow(deprecated)]
                    results.push(SymbolInformation {
                        name: info.path.clone(),
                        kind: symbol_kind_to_lsp(info.kind),
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: uri.clone(),
                            range: Range {
                                start: Position::new(start_line, start_char),
                                end: Position::new(end_line, end_char),
                            },
                        },
                        container_name: Some(format!("{:?}", info.kind)),
                    });
                }
            }
        }

        // Sort by relevance: exact prefix matches first, then by path length
        results.sort_by(|a, b| {
            let a_starts = a.name.to_lowercase().starts_with(&query);
            let b_starts = b.name.to_lowercase().starts_with(&query);
            match (a_starts, b_starts) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.len().cmp(&b.name.len()),
            }
        });

        // Limit results to prevent overwhelming the UI
        results.truncate(100);

        Ok(Some(results))
    }

    async fn semantic_tokens_full(
        &self,
        params: SemanticTokensParams,
    ) -> Result<Option<SemanticTokensResult>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let mut tokens: Vec<(usize, usize, u32, u32)> = Vec::new(); // (start, len, type, modifiers)

        // Collect tokens from symbol definition paths (e.g., "terra.temp" in "signal.terra.temp")
        for (kind, span) in index.get_symbol_path_spans() {
            let token_type = match kind {
                CdslSymbolKind::Signal => 0,
                CdslSymbolKind::Field => 1,
                CdslSymbolKind::Operator => 2,
                CdslSymbolKind::Function => 3,
                CdslSymbolKind::Type => 4,
                CdslSymbolKind::Strata => 5,
                CdslSymbolKind::Era => 6,
                CdslSymbolKind::Impulse => 7,
                CdslSymbolKind::Fracture => 8,
                CdslSymbolKind::Chronicle => 9,
                CdslSymbolKind::Entity => 10,
                CdslSymbolKind::Const => 15,
                CdslSymbolKind::Config => 16,
                CdslSymbolKind::Member => 17,
                CdslSymbolKind::World => 18,
            };
            // Mark as definition
            tokens.push((span.start, span.end - span.start, token_type, 0b11));
        }

        // Collect tokens from references
        for (info, span) in index.get_all_references() {
            let token_type = match info.kind {
                CdslSymbolKind::Signal => 0,
                CdslSymbolKind::Field => 1,
                CdslSymbolKind::Operator => 2,
                CdslSymbolKind::Function => 3,
                CdslSymbolKind::Type => 4,
                CdslSymbolKind::Strata => 5,
                CdslSymbolKind::Era => 6,
                CdslSymbolKind::Impulse => 7,
                CdslSymbolKind::Fracture => 8,
                CdslSymbolKind::Chronicle => 9,
                CdslSymbolKind::Entity => 10,
                CdslSymbolKind::Const => 15,
                CdslSymbolKind::Config => 16,
                CdslSymbolKind::Member => 17,
                CdslSymbolKind::World => 18,
            };
            tokens.push((span.start, span.end - span.start, token_type, 0));
        }

        // Add keyword tokens
        let keywords = [
            ("signal", 11),
            ("field", 11),
            ("operator", 11),
            ("strata", 11),
            ("era", 11),
            ("fn", 11),
            ("type", 11),
            ("const", 11),
            ("config", 11),
            ("impulse", 11),
            ("fracture", 11),
            ("chronicle", 11),
            ("entity", 11),
            ("member", 11),
            ("world", 11),
            ("resolve", 11),
            ("measure", 11),
            ("collect", 11),
            ("when", 11),
            ("emit", 11),
            ("apply", 11),
            ("assert", 11),
            ("observe", 11),
            ("schema", 11),
            ("transition", 11),
            ("policy", 11),
            ("if", 11),
            ("else", 11),
            ("let", 11),
            ("in", 11),
        ];

        for (kw, token_type) in keywords {
            let mut search_start = 0;
            while let Some(pos) = doc[search_start..].find(kw) {
                let abs_pos = search_start + pos;
                // Check this is a whole word (not part of identifier)
                let before_ok = abs_pos == 0
                    || !doc[..abs_pos]
                        .chars()
                        .last()
                        .map(|c| c.is_alphanumeric() || c == '_')
                        .unwrap_or(false);
                let after_ok = abs_pos + kw.len() >= doc.len()
                    || !doc[abs_pos + kw.len()..]
                        .chars()
                        .next()
                        .map(|c| c.is_alphanumeric() || c == '_')
                        .unwrap_or(false);

                if before_ok && after_ok {
                    // Check not inside a comment or string
                    let line_start = doc[..abs_pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
                    let line_before = &doc[line_start..abs_pos];
                    if !line_before.contains("//") {
                        tokens.push((abs_pos, kw.len(), token_type, 0));
                    }
                }
                search_start = abs_pos + kw.len();
            }
        }

        // Add comment tokens
        let mut in_doc_comment = false;
        for (line_idx, line) in doc.lines().enumerate() {
            let line_start = doc
                .lines()
                .take(line_idx)
                .map(|l| l.len() + 1)
                .sum::<usize>();
            if let Some(pos) = line.find("//") {
                let comment_start = line_start + pos;
                let comment_len = line.len() - pos;
                tokens.push((comment_start, comment_len, 14, 0));
                in_doc_comment = line[pos..].starts_with("//!");
            }
        }
        let _ = in_doc_comment; // silence warning

        // Sort tokens by position
        tokens.sort_by_key(|t| t.0);

        // Convert to delta encoding
        let mut data = Vec::new();
        let mut prev_line = 0u32;
        let mut prev_start = 0u32;

        for (start, len, token_type, modifiers) in tokens {
            let (line, col) = offset_to_position(&doc, start);

            let delta_line = line - prev_line;
            let delta_start = if delta_line == 0 {
                col - prev_start
            } else {
                col
            };

            data.push(SemanticToken {
                delta_line,
                delta_start,
                length: len as u32,
                token_type,
                token_modifiers_bitset: modifiers,
            });

            prev_line = line;
            prev_start = col;
        }

        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data,
        })))
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let mut hints = Vec::new();

        // Add type hints for symbol references
        for ref_info in index.get_references_for_validation() {
            // Look up the definition to get its type
            let type_str = self.symbol_indices.iter().find_map(|entry| {
                entry
                    .value()
                    .find_definition(ref_info.kind, &ref_info.target_path)
                    .and_then(|info| info.ty.clone())
            });

            if let Some(ty) = type_str {
                let (line, col) = offset_to_position(&doc, ref_info.span.end);
                hints.push(InlayHint {
                    position: Position::new(line, col),
                    label: InlayHintLabel::String(format!(": {}", ty)),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: None,
                    padding_left: Some(false),
                    padding_right: Some(true),
                    data: None,
                });
            }
        }

        Ok(Some(hints))
    }

    async fn prepare_rename(
        &self,
        params: TextDocumentPositionParams,
    ) -> Result<Option<PrepareRenameResponse>> {
        let uri = &params.text_document.uri;
        let position = params.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        // Check if we're on a renameable symbol
        if let Some(info) = index.find_at_offset(offset) {
            // Only allow renaming user-defined symbols (not built-ins)
            match info.kind {
                CdslSymbolKind::Signal
                | CdslSymbolKind::Field
                | CdslSymbolKind::Operator
                | CdslSymbolKind::Function
                | CdslSymbolKind::Const
                | CdslSymbolKind::Config => {
                    // Return the range of the symbol path
                    // For now, just return the current word range
                    let range = get_word_range(&doc, offset);
                    let (start_line, start_col) = offset_to_position(&doc, range.start);
                    let (end_line, end_col) = offset_to_position(&doc, range.end);
                    return Ok(Some(PrepareRenameResponse::Range(Range {
                        start: Position::new(start_line, start_col),
                        end: Position::new(end_line, end_col),
                    })));
                }
                _ => return Ok(None),
            }
        }

        Ok(None)
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;
        let new_name = &params.new_name;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        // Find the symbol being renamed
        let (kind, old_path) = if let Some((k, p)) = index.get_reference_at_offset(offset) {
            (k, p.to_string())
        } else if let Some(info) = index.find_at_offset(offset) {
            (info.kind, info.path.clone())
        } else {
            return Ok(None);
        };

        // Collect all edits across all files
        let mut changes: std::collections::HashMap<Url, Vec<TextEdit>> =
            std::collections::HashMap::new();

        for entry in self.symbol_indices.iter() {
            let file_uri = entry.key().clone();
            let file_index = entry.value();

            // Get the document content
            let file_doc = if &file_uri == uri {
                doc.clone()
            } else if let Some(d) = self.documents.get(&file_uri) {
                d.clone()
            } else if let Ok(path) = file_uri.to_file_path() {
                std::fs::read_to_string(path).unwrap_or_default()
            } else {
                continue;
            };

            let mut file_edits = Vec::new();

            // Find references to rename
            for ref_info in file_index.get_references_for_validation() {
                if ref_info.kind == kind && ref_info.target_path == old_path {
                    let (start_line, start_col) =
                        offset_to_position(&file_doc, ref_info.span.start);
                    let (end_line, end_col) = offset_to_position(&file_doc, ref_info.span.end);
                    file_edits.push(TextEdit {
                        range: Range {
                            start: Position::new(start_line, start_col),
                            end: Position::new(end_line, end_col),
                        },
                        new_text: format!("{}.{}", kind.display_name(), new_name),
                    });
                }
            }

            // Find definition to rename
            for (info, span) in file_index.get_all_symbols() {
                if info.kind == kind && info.path == old_path {
                    let (start_line, start_col) = offset_to_position(&file_doc, span.start);
                    let (end_line, end_col) = offset_to_position(&file_doc, span.end);
                    // The span includes the whole definition, we need just the path part
                    // For now, we'll let the user edit manually or use a more precise span
                    // This is a simplification - full rename would need AST modification
                    file_edits.push(TextEdit {
                        range: Range {
                            start: Position::new(start_line, start_col),
                            end: Position::new(end_line, end_col),
                        },
                        new_text: new_name.to_string(),
                    });
                }
            }

            if !file_edits.is_empty() {
                changes.insert(file_uri, file_edits);
            }
        }

        if changes.is_empty() {
            return Ok(None);
        }

        Ok(Some(WorkspaceEdit {
            changes: Some(changes),
            document_changes: None,
            change_annotations: None,
        }))
    }

    async fn folding_range(&self, params: FoldingRangeParams) -> Result<Option<Vec<FoldingRange>>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let (ast, _) = parse(&doc);
        let ast = match ast {
            Some(ast) => ast,
            None => return Ok(None),
        };

        let mut ranges = Vec::new();

        // Add folding ranges for each top-level item
        for item in &ast.items {
            let (start_line, _) = offset_to_position(&doc, item.span.start);
            let (end_line, _) = offset_to_position(&doc, item.span.end);

            // Only create fold if it spans multiple lines
            if end_line > start_line {
                ranges.push(FoldingRange {
                    start_line,
                    start_character: None,
                    end_line,
                    end_character: None,
                    kind: Some(FoldingRangeKind::Region),
                    collapsed_text: Some(get_item_collapsed_text(&item.node)),
                });
            }

            // Add folding for nested blocks within items
            add_nested_folding_ranges(&doc, &item.node, &mut ranges);
        }

        // Add folding for comment blocks
        add_comment_folding_ranges(&doc, &mut ranges);

        if ranges.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ranges))
        }
    }
}

/// Get collapsed text preview for a folded item.
fn get_item_collapsed_text(item: &Item) -> String {
    match item {
        Item::SignalDef(def) => format!("signal.{} {{ ... }}", def.path.node),
        Item::FieldDef(def) => format!("field.{} {{ ... }}", def.path.node),
        Item::OperatorDef(def) => format!("operator.{} {{ ... }}", def.path.node),
        Item::FnDef(def) => format!("fn.{} {{ ... }}", def.path.node),
        Item::TypeDef(def) => format!("type.{} {{ ... }}", def.name.node),
        Item::StrataDef(def) => format!("strata.{} {{ ... }}", def.path.node),
        Item::EraDef(def) => format!("era.{} {{ ... }}", def.name.node),
        Item::ImpulseDef(def) => format!("impulse.{} {{ ... }}", def.path.node),
        Item::FractureDef(def) => format!("fracture.{} {{ ... }}", def.path.node),
        Item::ChronicleDef(def) => format!("chronicle.{} {{ ... }}", def.path.node),
        Item::EntityDef(def) => format!("entity.{} {{ ... }}", def.path.node),
        Item::MemberDef(def) => format!("member.{} {{ ... }}", def.path.node),
        Item::WorldDef(def) => format!("world.{} {{ ... }}", def.path.node),
        Item::ConstBlock(_) => "const { ... }".to_string(),
        Item::ConfigBlock(_) => "config { ... }".to_string(),
    }
}

/// Add folding ranges for nested blocks (resolve, config, assert, etc.)
fn add_nested_folding_ranges(doc: &str, item: &Item, ranges: &mut Vec<FoldingRange>) {
    match item {
        Item::SignalDef(def) => {
            if let Some(ref resolve) = def.resolve {
                add_block_folding(doc, resolve.body.span.start, resolve.body.span.end, ranges);
            }
            if let Some(ref assertions) = def.assertions {
                for assertion in &assertions.assertions {
                    add_block_folding(
                        doc,
                        assertion.condition.span.start,
                        assertion.condition.span.end,
                        ranges,
                    );
                }
            }
            if !def.local_config.is_empty() {
                // Local config block - find its span from entries
                if let (Some(first), Some(last)) =
                    (def.local_config.first(), def.local_config.last())
                {
                    add_block_folding(doc, first.path.span.start, last.value.span.end, ranges);
                }
            }
        }
        Item::FieldDef(def) => {
            if let Some(ref measure) = def.measure {
                add_block_folding(doc, measure.body.span.start, measure.body.span.end, ranges);
            }
        }
        Item::OperatorDef(def) => {
            if let Some(ref body) = def.body {
                // Extract span from the OperatorBody enum variant
                let expr_span = match body {
                    OperatorBody::Warmup(expr) => &expr.span,
                    OperatorBody::Collect(expr) => &expr.span,
                    OperatorBody::Measure(expr) => &expr.span,
                };
                add_block_folding(doc, expr_span.start, expr_span.end, ranges);
            }
        }
        Item::FnDef(def) => {
            add_block_folding(doc, def.body.span.start, def.body.span.end, ranges);
        }
        Item::FractureDef(def) => {
            for cond in &def.conditions {
                add_block_folding(doc, cond.span.start, cond.span.end, ranges);
            }
            for emission in &def.emit {
                add_block_folding(
                    doc,
                    emission.value.span.start,
                    emission.value.span.end,
                    ranges,
                );
            }
        }
        Item::MemberDef(def) => {
            if let Some(ref resolve) = def.resolve {
                add_block_folding(doc, resolve.body.span.start, resolve.body.span.end, ranges);
            }
        }
        _ => {}
    }
}

/// Add a folding range for a block if it spans multiple lines.
fn add_block_folding(doc: &str, start: usize, end: usize, ranges: &mut Vec<FoldingRange>) {
    let (start_line, _) = offset_to_position(doc, start);
    let (end_line, _) = offset_to_position(doc, end);

    if end_line > start_line {
        ranges.push(FoldingRange {
            start_line,
            start_character: None,
            end_line,
            end_character: None,
            kind: Some(FoldingRangeKind::Region),
            collapsed_text: None,
        });
    }
}

/// Add folding ranges for consecutive comment blocks.
fn add_comment_folding_ranges(doc: &str, ranges: &mut Vec<FoldingRange>) {
    let lines: Vec<&str> = doc.lines().collect();
    let mut comment_start: Option<u32> = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let is_comment = trimmed.starts_with("//") || trimmed.starts_with('#');

        if is_comment {
            if comment_start.is_none() {
                comment_start = Some(i as u32);
            }
        } else if let Some(start) = comment_start {
            let end = i as u32 - 1;
            if end > start {
                ranges.push(FoldingRange {
                    start_line: start,
                    start_character: None,
                    end_line: end,
                    end_character: None,
                    kind: Some(FoldingRangeKind::Comment),
                    collapsed_text: Some("// ...".to_string()),
                });
            }
            comment_start = None;
        }
    }

    // Handle comment block at end of file
    if let Some(start) = comment_start {
        let end = lines.len() as u32 - 1;
        if end > start {
            ranges.push(FoldingRange {
                start_line: start,
                start_character: None,
                end_line: end,
                end_character: None,
                kind: Some(FoldingRangeKind::Comment),
                collapsed_text: Some("// ...".to_string()),
            });
        }
    }
}

/// Get the word range at a byte offset.
fn get_word_range(text: &str, offset: usize) -> std::ops::Range<usize> {
    if offset > text.len() {
        return offset..offset;
    }

    // Find start of word
    let before = &text[..offset];
    let start = before
        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + 1)
        .unwrap_or(0);

    // Find end of word
    let after = &text[offset..];
    let end_rel = after
        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .unwrap_or(after.len());

    start..(offset + end_rel)
}

/// Convert our SymbolKind to LSP SymbolKind.
fn symbol_kind_to_lsp(kind: symbols::SymbolKind) -> SymbolKind {
    match kind {
        symbols::SymbolKind::Signal => SymbolKind::VARIABLE,
        symbols::SymbolKind::Field => SymbolKind::FIELD,
        symbols::SymbolKind::Operator => SymbolKind::OPERATOR,
        symbols::SymbolKind::Function => SymbolKind::FUNCTION,
        symbols::SymbolKind::Type => SymbolKind::STRUCT,
        symbols::SymbolKind::Strata => SymbolKind::NAMESPACE,
        symbols::SymbolKind::Era => SymbolKind::NAMESPACE,
        symbols::SymbolKind::Impulse => SymbolKind::EVENT,
        symbols::SymbolKind::Fracture => SymbolKind::EVENT,
        symbols::SymbolKind::Chronicle => SymbolKind::CLASS,
        symbols::SymbolKind::Entity => SymbolKind::CLASS,
        symbols::SymbolKind::Member => SymbolKind::VARIABLE,
        symbols::SymbolKind::World => SymbolKind::MODULE,
        symbols::SymbolKind::Const => SymbolKind::CONSTANT,
        symbols::SymbolKind::Config => SymbolKind::PROPERTY,
    }
}

/// Convert our SymbolKind to LSP CompletionItemKind.
fn cdsl_kind_to_completion_kind(kind: CdslSymbolKind) -> CompletionItemKind {
    match kind {
        CdslSymbolKind::Signal => CompletionItemKind::VARIABLE,
        CdslSymbolKind::Field => CompletionItemKind::FIELD,
        CdslSymbolKind::Operator => CompletionItemKind::METHOD,
        CdslSymbolKind::Function => CompletionItemKind::FUNCTION,
        CdslSymbolKind::Type => CompletionItemKind::STRUCT,
        CdslSymbolKind::Strata => CompletionItemKind::MODULE,
        CdslSymbolKind::Era => CompletionItemKind::MODULE,
        CdslSymbolKind::Impulse => CompletionItemKind::EVENT,
        CdslSymbolKind::Fracture => CompletionItemKind::EVENT,
        CdslSymbolKind::Chronicle => CompletionItemKind::CLASS,
        CdslSymbolKind::Entity => CompletionItemKind::CLASS,
        CdslSymbolKind::Member => CompletionItemKind::VARIABLE,
        CdslSymbolKind::World => CompletionItemKind::MODULE,
        CdslSymbolKind::Const => CompletionItemKind::CONSTANT,
        CdslSymbolKind::Config => CompletionItemKind::PROPERTY,
    }
}

fn completion_item(label: &str, kind: CompletionItemKind, detail: &str) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        workspace_roots: RwLock::new(Vec::new()),
        documents: DashMap::new(),
        symbol_indices: DashMap::new(),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}
