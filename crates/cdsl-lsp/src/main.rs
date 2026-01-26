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

use continuum_functions as _;
use continuum_kernel_registry as kernel_registry;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dashmap::DashMap;
use indexmap::IndexMap;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use walkdir::WalkDir;

// Engine types - pure transport layer (no custom shadow structures)
use continuum_cdsl::ast::{Node, RoleData, RoleId, World};
use continuum_cdsl::foundation::{Path as EnginePath, Span as EngineSpan};
use continuum_cdsl::{compile_from_memory, CompileError, CompileResultWithSources, SourceMap};

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
];

const SEMANTIC_TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::DECLARATION,
    SemanticTokenModifier::DEFINITION,
    SemanticTokenModifier::READONLY,
];

struct Backend {
    client: Client,
    /// Workspace root folders.
    workspace_roots: RwLock<Vec<Url>>,
    /// Cached document contents (for open documents).
    documents: DashMap<Url, String>,
    /// Compiled worlds for ALL files (not just open ones).
    /// Pure transport - we store engine World directly, no shadow structures.
    worlds: DashMap<Url, World>,
}

impl Backend {
    /// Parse a document and publish diagnostics.
    /// Uses compile_from_memory() to compile in-memory source and stores World directly.
    async fn parse_and_publish_diagnostics(&self, uri: Url, text: &str) {
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return,
        };

        // Compile from memory using new API
        let mut sources = IndexMap::new();
        sources.insert(path.clone(), text.to_string());

        let compile_result = compile_from_memory(sources);

        let diagnostics: Vec<Diagnostic> = match compile_result {
            Ok(compiled_world) => {
                // Success - store the World
                self.worlds.insert(uri.clone(), compiled_world.world);
                Vec::new() // No errors
            }
            Err((source_map, errors)) => {
                // Compilation failed - convert errors to LSP diagnostics
                errors
                    .iter()
                    .filter_map(|error| {
                        // Get the source file for this error
                        let file = source_map.file(&error.span);
                        
                        // Only show errors from the current file
                        if file.path != path {
                            return None;
                        }

                        // Convert engine span (byte offsets) to LSP positions (line/col)
                        let (start_line, start_char) = offset_to_position(text, error.span.start as usize);
                        let (end_line, end_char) = offset_to_position(text, error.span.end as usize);

                        Some(Diagnostic {
                            range: Range {
                                start: Position::new(start_line, start_char),
                                end: Position::new(end_line, end_char),
                            },
                            severity: Some(DiagnosticSeverity::ERROR),
                            source: Some("cdsl".to_string()),
                            message: error.message.clone(),
                            ..Default::default()
                        })
                    })
                    .collect()
            }
        };

        self.client
            .publish_diagnostics(uri.clone(), diagnostics, None)
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
            if entry.path().extension().is_some_and(|e| e == "cdsl") {
                self.index_file(entry.path()).await;
            }
        }
    }

    /// Index a single .cdsl file (without publishing diagnostics).
    async fn index_file(&self, path: &Path) {
        let uri = match Url::from_file_path(path) {
            Ok(uri) => uri,
            Err(_) => return,
        };

        if self.documents.contains_key(&uri) {
            return;
        }

        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(_) => return,
        };

        // Compile the file using new API
        let mut sources = IndexMap::new();
        sources.insert(path.to_path_buf(), text);

        if let Ok(compiled_world) = compile_from_memory(sources) {
            self.worlds.insert(uri, compiled_world.world);
        }
    }

    // COMMENTED OUT - needs rewrite for engine types
    // /// Find undefined references (legacy, now handled by unified compiler)
    // #[allow(dead_code)]
    // fn find_undefined_references(
    //     &self,
    //     refs: &[ReferenceValidationInfo],
    // ) -> Vec<ReferenceValidationInfo> {
    //     refs.iter()
    //         .filter(|r| {
    //             !self.worlds.iter().any(|entry| {
    //                 // TODO: implement using World directly
    //                 false
    //             })
    //         })
    //         .cloned()
    //         .collect()
    // }

    /// Check if the workspace has a world definition.
    fn has_world_definition(&self) -> bool {
        // With new API, if we have any compiled world, we have a world definition
        !self.worlds.is_empty()
    }

    // COMMENTED OUT - needs rewrite for engine types
    // /// Find all unused symbols across the entire world.
    // #[allow(dead_code)]
    // fn find_unused_symbols_in_file(
    //     &self,
    //     uri: &Url,
    // ) -> Vec<(String, std::ops::Range<usize>, RoleId)> {
    //     // TODO: implement by walking World.globals/members
    //     vec![]
    // }

    /// Run workspace-level validation and publish diagnostics.
    async fn validate_workspace(&self) {
        // Validation is now handled by the unified compiler during compilation
        // No separate validation step needed
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> Result<InitializeResult> {
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

        self.scan_workspace().await;
        self.validate_workspace().await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.documents.insert(
            params.text_document.uri.clone(),
            params.text_document.text.clone(),
        );
        self.parse_and_publish_diagnostics(params.text_document.uri, &params.text_document.text)
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        if let Some(change) = params.content_changes.first() {
            self.documents
                .insert(params.text_document.uri.clone(), change.text.clone());
            self.parse_and_publish_diagnostics(params.text_document.uri, &change.text)
                .await;
        }
    }

    async fn did_save(&self, params: DidSaveTextDocumentParams) {
        if let Some(text) = params.text {
            self.parse_and_publish_diagnostics(params.text_document.uri, &text)
                .await;
        } else if let Some(doc) = self.documents.get(&params.text_document.uri) {
            self.parse_and_publish_diagnostics(params.text_document.uri, doc.value())
                .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);
        self.worlds.remove(&uri);

        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let prefix = get_completion_prefix(&doc, position);
        let kind_filter = detect_kind_prefix(&prefix);

        let mut items = Vec::new();

        if let Some((kind, prefix_len)) = kind_filter {
            let path_prefix = &prefix[prefix_len..];
            let prefix_segments: Vec<&str> =
                path_prefix.split('.').filter(|s| !s.is_empty()).collect();
            let prefix_depth = prefix_segments.len();

            struct CompletionData {
                ty: Option<String>,
                title: Option<String>,
                doc: Option<String>,
            }
            let mut seen_segments: std::collections::HashMap<String, Option<CompletionData>> =
                std::collections::HashMap::new();

            for entry in self.symbol_indices.iter() {
                for info in entry.value().get_completions().filter(|c| c.kind == kind) {
                    let path_segments: Vec<&str> = info.path.split('.').collect();

                    let matches_prefix = prefix_segments
                        .iter()
                        .zip(path_segments.iter())
                        .all(|(p, s)| s.starts_with(*p));

                    if !matches_prefix {
                        continue;
                    }

                    if let Some(next_segment) = path_segments.get(prefix_depth) {
                        let is_final = path_segments.len() == prefix_depth + 1;

                        if is_final {
                            seen_segments.insert(
                                next_segment.to_string(),
                                Some(CompletionData {
                                    ty: info.ty.map(|s| s.to_string()),
                                    title: info.title.map(|s| s.to_string()),
                                    doc: info.doc.map(|s| s.to_string()),
                                }),
                            );
                        } else {
                            seen_segments
                                .entry(next_segment.to_string())
                                .or_insert(None);
                        }
                    }
                }
            }

            for (segment, data) in seen_segments {
                let is_final = data.is_some();
                let (detail, documentation) = match data {
                    Some(d) => {
                        let detail = match (&d.ty, &d.title) {
                            (Some(ty), Some(title)) => format!("{} — {}", ty, title),
                            (Some(ty), None) => ty.clone(),
                            (None, Some(title)) => title.clone(),
                            (None, None) => kind.display_name().to_string(),
                        };

                        let mut doc_parts = Vec::new();
                        if let Some(title) = d.title {
                            doc_parts.push(format!("*{}*", title));
                        }
                        if let Some(ty) = d.ty {
                            doc_parts.push(format!("Type: `{}`", ty));
                        }
                        if let Some(doc) = d.doc {
                            doc_parts.push(doc);
                        }

                        (Some(detail), Some(doc_parts.join("\n\n")))
                    }
                    None => (Some("Namespace".to_string()), None),
                };

                items.push(CompletionItem {
                    label: segment,
                    kind: Some(if is_final {
                        match kind {
                            CdslSymbolKind::Signal => CompletionItemKind::VARIABLE,
                            CdslSymbolKind::Field => CompletionItemKind::PROPERTY,
                            CdslSymbolKind::Operator => CompletionItemKind::METHOD,
                            CdslSymbolKind::Function => CompletionItemKind::FUNCTION,
                            CdslSymbolKind::Type => CompletionItemKind::TYPE_PARAMETER,
                            CdslSymbolKind::Entity => CompletionItemKind::CLASS,
                            _ => CompletionItemKind::FIELD,
                        }
                    } else {
                        CompletionItemKind::MODULE
                    }),
                    detail,
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
            items.extend(vec![
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
                completion_item("resolve", CompletionItemKind::KEYWORD, "Resolve block"),
                completion_item("measure", CompletionItemKind::KEYWORD, "Measure block"),
                completion_item("when", CompletionItemKind::KEYWORD, "When condition"),
                completion_item("emit", CompletionItemKind::KEYWORD, "Emit block"),
                completion_item("assert", CompletionItemKind::KEYWORD, "Assert block"),
                completion_item("if", CompletionItemKind::KEYWORD, "Conditional expression"),
                completion_item("else", CompletionItemKind::KEYWORD, "Else branch"),
                completion_item("let", CompletionItemKind::KEYWORD, "Let binding"),
                completion_item("in", CompletionItemKind::KEYWORD, "In expression"),
                completion_item("prev", CompletionItemKind::VARIABLE, "Previous value"),
                completion_item("dt_raw", CompletionItemKind::VARIABLE, "Raw time delta"),
                completion_item("collected", CompletionItemKind::VARIABLE, "Collected value"),
                completion_item("payload", CompletionItemKind::VARIABLE, "Impulse payload"),
                completion_item("Scalar", CompletionItemKind::TYPE_PARAMETER, "Scalar type"),
                completion_item("Vector", CompletionItemKind::TYPE_PARAMETER, "Vector type"),
                completion_item("Tensor", CompletionItemKind::TYPE_PARAMETER, "Tensor type"),
            ]);

            for (namespace, name) in kernel_registry::all_names() {
                if let Some(k) = kernel_registry::get_in_namespace(namespace, name) {
                    items.push(CompletionItem {
                        label: format!("{}.{}", namespace, name),
                        kind: Some(CompletionItemKind::FUNCTION),
                        detail: Some(k.signature.to_string()),
                        documentation: Some(Documentation::MarkupContent(MarkupContent {
                            kind: MarkupKind::Markdown,
                            value: k.doc.to_string(),
                        })),
                        ..Default::default()
                    });
                }
            }
        }

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = &params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

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

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        if let Some(index) = self.symbol_indices.get(uri) {
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

            if let Some((kind, path)) = index.get_reference_at_offset(offset) {
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

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let formatted = formatter::format(&doc);

        if formatted == doc {
            return Ok(Some(vec![]));
        }

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

                #[allow(deprecated)]
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

        let offset = position_to_offset(&doc, position);
        let text_before = &doc[..offset];

        let (fn_path, active_param) = match find_function_call_context(text_before) {
            Some(ctx) => ctx,
            None => return Ok(None),
        };

        for entry in self.symbol_indices.iter() {
            if let Some(sig) = entry.value().get_function_signature(&fn_path) {
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

                return Ok(Some(SignatureHelp {
                    signatures: vec![SignatureInformation {
                        label,
                        documentation: sig.doc.as_ref().map(|d| {
                            Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: d.clone(),
                            })
                        }),
                        parameters: Some(param_infos),
                        active_parameter: Some(
                            active_param.min(sig.params.len().saturating_sub(1)) as u32,
                        ),
                    }],
                    active_signature: Some(0),
                    active_parameter: Some(
                        active_param.min(sig.params.len().saturating_sub(1)) as u32
                    ),
                }));
            }
        }

        if let Some((namespace, function)) = fn_path.split_once('.')
            && let Some(k) = kernel_registry::get_in_namespace(namespace, function)
        {
            let label = k.signature;
            let mut param_infos = Vec::new();

            if let Some(start) = label.find('(') {
                if let Some(end) = label.find(')') {
                    let params_part = &label[start + 1..end];
                    let params: Vec<_> = params_part.split(',').map(|s| s.trim()).collect();

                    let mut current_offset = start + 1;
                    for param in &params {
                        if param.is_empty() {
                            continue;
                        }
                        if let Some(pos) = label[current_offset..].find(param) {
                            let param_start = pos + current_offset;
                            let param_end = param_start + param.len();

                            param_infos.push(ParameterInformation {
                                label: ParameterLabel::LabelOffsets([
                                    param_start as u32,
                                    param_end as u32,
                                ]),
                                documentation: None,
                            });

                            current_offset = param_end;
                        }
                    }

                    return Ok(Some(SignatureHelp {
                        signatures: vec![SignatureInformation {
                            label: label.to_string(),
                            documentation: Some(Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: k.doc.to_string(),
                            })),
                            parameters: Some(param_infos),
                            active_parameter: Some(
                                active_param.min(params.len().saturating_sub(1)) as u32,
                            ),
                        }],
                        active_signature: Some(0),
                        active_parameter: Some(
                            active_param.min(params.len().saturating_sub(1)) as u32
                        ),
                    }));
                }
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

        for entry in self.symbol_indices.iter() {
            let uri = entry.key().clone();
            let index = entry.value();

            let doc = match self.documents.get(&uri) {
                Some(doc) => doc.clone(),
                None => continue,
            };

            for (info, span) in index.get_all_definitions() {
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

        results.sort_by(|a, b| {
            let a_starts = a.name.to_lowercase().starts_with(&query);
            let b_starts = b.name.to_lowercase().starts_with(&query);
            match (a_starts, b_starts) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.len().cmp(&b.name.len()),
            }
        });

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

        let mut tokens: Vec<(usize, usize, u32, u32)> = Vec::new();

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
                _ => 11,
            };
            tokens.push((span.start, span.end - span.start, token_type, 1));
        }

        for (info, span) in index.get_all_references() {
            let token_type = match info.kind {
                CdslSymbolKind::Signal => 0,
                CdslSymbolKind::Field => 1,
                CdslSymbolKind::Const => 11,
                CdslSymbolKind::Config => 11,
                CdslSymbolKind::Entity => 10,
                _ => 11,
            };
            tokens.push((span.start, span.end - span.start, token_type, 0));
        }

        let keywords = [
            "signal",
            "field",
            "operator",
            "fn",
            "type",
            "strata",
            "era",
            "impulse",
            "fracture",
            "chronicle",
            "entity",
            "world",
            "const",
            "config",
            "policy",
            "version",
            "initial",
            "terminal",
            "stride",
            "title",
            "symbol",
            "uses",
            "active",
            "converge",
            "warmup",
            "iterate",
            "phase",
            "magnitude",
            "symmetric",
            "positive_definite",
            "topology",
            "min",
            "max",
            "mean",
            "sum",
            "product",
            "any",
            "all",
            "none",
            "first",
            "nearest",
            "within",
            "other",
            "pairs",
            "filter",
            "event",
            "observe",
            "apply",
            "when",
            "emit",
            "assert",
            "resolve",
            "measure",
            "collect",
            "transition",
            "gated",
            "dt",
            "to",
            "warn",
            "error",
            "fatal",
            "Scalar",
            "Vec2",
            "Vec3",
            "Vec4",
            "Vector",
            "Tensor",
            "Grid",
            "Seq",
            "if",
            "else",
            "let",
            "in",
            "prev",
            "dt_raw",
            "collected",
            "payload",
        ];

        for kw in keywords {
            let mut search_start = 0;
            while let Some(pos) = doc[search_start..].find(kw) {
                let abs_pos = search_start + pos;
                let before_ok = abs_pos == 0
                    || !doc[..abs_pos]
                        .chars()
                        .next_back()
                        .unwrap()
                        .is_alphanumeric();
                let after_ok = abs_pos + kw.len() >= doc.len()
                    || !doc[abs_pos + kw.len()..]
                        .chars()
                        .next()
                        .unwrap()
                        .is_alphanumeric();

                if before_ok && after_ok {
                    let line_start = doc[..abs_pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
                    let line_before = &doc[line_start..abs_pos];
                    if !line_before.trim().starts_with("///") {
                        tokens.push((abs_pos, kw.len(), 11, 0));
                    }
                }
                search_start = abs_pos + kw.len();
            }
        }

        tokens.sort_by_key(|t| t.0);

        let mut last_line = 0;
        let mut last_start = 0;
        let mut data = Vec::new();

        for (start, len, token_type, modifiers) in tokens {
            let (line, col) = offset_to_position(&doc, start);
            let delta_line = line - last_line;
            let delta_start = if delta_line == 0 {
                col - last_start
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

            last_line = line;
            last_start = col;
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

        for ref_info in index.get_references_for_validation() {
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

        if let Some(info) = index.find_at_offset(offset) {
            match info.kind {
                CdslSymbolKind::Signal
                | CdslSymbolKind::Field
                | CdslSymbolKind::Operator
                | CdslSymbolKind::Function
                | CdslSymbolKind::Const
                | CdslSymbolKind::Config => {
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
        let new_name = params.new_name;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        let target =
            if let Some(reference) = index.references.iter().find(|r| r.span.contains(&offset)) {
                Some((reference.kind, reference.target_path.clone()))
            } else if let Some(symbol) = index.symbols.iter().find(|s| s.span.contains(&offset)) {
                Some((symbol.info.kind, symbol.info.path.clone()))
            } else {
                None
            };

        if let Some((kind, path)) = target {
            let mut changes = HashMap::new();

            for entry in self.symbol_indices.iter() {
                let uri = entry.key().clone();
                let index = entry.value();
                let mut edits = Vec::new();

                for symbol in &index.symbols {
                    if symbol.info.kind == kind && symbol.info.path == path {
                        let (line, col) = offset_to_position(&doc, symbol.path_span.start);
                        edits.push(TextEdit {
                            range: Range {
                                start: Position::new(line, col),
                                end: Position::new(
                                    line,
                                    col + symbol.info.path.split('.').last().unwrap().len() as u32,
                                ),
                            },
                            new_text: new_name.clone(),
                        });
                    }
                }

                for reference in &index.references {
                    if reference.kind == kind && reference.target_path == path {
                        let (line, col) = offset_to_position(&doc, reference.span.start);
                        edits.push(TextEdit {
                            range: Range {
                                start: Position::new(line, col),
                                end: Position::new(
                                    line,
                                    col + reference.target_path.split('.').last().unwrap().len()
                                        as u32,
                                ),
                            },
                            new_text: new_name.clone(),
                        });
                    }
                }

                if !edits.is_empty() {
                    changes.insert(uri, edits);
                }
            }

            return Ok(Some(WorkspaceEdit {
                changes: Some(changes),
                ..Default::default()
            }));
        }

        Ok(None)
    }

    async fn folding_range(&self, params: FoldingRangeParams) -> Result<Option<Vec<FoldingRange>>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let index = match self.symbol_indices.get(uri) {
            Some(index) => index,
            None => return Ok(None),
        };

        let mut ranges = Vec::new();

        for symbol in &index.symbols {
            let (start_line, _) = offset_to_position(&doc, symbol.span.start);
            let (end_line, _) = offset_to_position(&doc, symbol.span.end);

            if end_line > start_line {
                ranges.push(FoldingRange {
                    start_line,
                    start_character: None,
                    end_line,
                    end_character: None,
                    kind: None,
                    collapsed_text: None,
                });
            }
        }

        Ok(Some(ranges))
    }
}

fn offset_to_position(text: &str, offset: usize) -> (u32, u32) {
    let mut line = 0;
    let mut col = 0;
    for (i, c) in text.chars().enumerate() {
        if i == offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn position_to_offset(text: &str, pos: Position) -> usize {
    let mut line = 0;
    let mut col = 0;
    for (i, c) in text.chars().enumerate() {
        if line == pos.line && col == pos.character {
            return i;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    text.len()
}

fn get_completion_prefix(text: &str, pos: Position) -> String {
    let offset = position_to_offset(text, pos);
    let mut start = offset;
    while start > 0 {
        let c = text.as_bytes()[start - 1];
        if !c.is_ascii_alphanumeric() && c != b'.' && c != b'_' {
            break;
        }
        start -= 1;
    }
    text[start..offset].to_string()
}

fn detect_kind_prefix(prefix: &str) -> Option<(RoleId, usize)> {
    let prefixes = [
        ("signal.", RoleId::Signal),
        ("field.", RoleId::Field),
        ("operator.", RoleId::Operator),
        ("impulse.", RoleId::Impulse),
        ("fracture.", RoleId::Fracture),
        ("chronicle.", RoleId::Chronicle),
    ];

    for (pat, kind) in prefixes {
        if prefix.starts_with(pat) {
            return Some((kind, pat.len()));
        }
    }
    None
}

fn find_function_call_context(text_before: &str) -> Option<(String, usize)> {
    let mut paren_depth = 0;
    let mut last_call_start = None;

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
                    let before_paren = &text_before[..i];

                    if let Some(fn_start) = before_paren.rfind("fn.") {
                        let path_start = fn_start + 3;
                        let path = &text_before[path_start..i];

                        if path
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                            && !path.is_empty()
                        {
                            last_call_start = Some((path.to_string(), i + 1));
                            break;
                        }
                    }

                    let id_start = before_paren
                        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
                        .map(|idx| idx + 1)
                        .unwrap_or(0);
                    let id = &before_paren[id_start..];
                    if !id.is_empty()
                        && id
                            .chars()
                            .all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                    {
                        last_call_start = Some((id.to_string(), i + 1));
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    if let Some((fn_path, args_start)) = last_call_start {
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
            if let Some(ref emit) = def.emit {
                collect_clamp_in_expr(emit, spans);
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
            ..
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
        Expr::Nearest { position, .. } => {
            collect_clamp_in_expr(position, spans);
        }
        Expr::Within {
            position, radius, ..
        } => {
            collect_clamp_in_expr(position, spans);
            collect_clamp_in_expr(radius, spans);
        }
        _ => {}
    }
}

/// Convert engine RoleId to LSP SymbolKind for protocol transport
fn role_to_lsp_symbol_kind(role: RoleId) -> SymbolKind {
    match role {
        RoleId::Signal => SymbolKind::VARIABLE,
        RoleId::Field => SymbolKind::PROPERTY,
        RoleId::Operator => SymbolKind::METHOD,
        RoleId::Impulse => SymbolKind::EVENT,
        RoleId::Fracture => SymbolKind::EVENT,
        RoleId::Chronicle => SymbolKind::CLASS,
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

fn get_word_at_cursor(text: &str, offset: usize) -> Option<String> {
    let start = text[..offset]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = text[offset..]
        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + offset)
        .unwrap_or(text.len());
    if start < end {
        Some(text[start..end].to_string())
    } else {
        None
    }
}

fn get_word_range(text: &str, offset: usize) -> std::ops::Range<usize> {
    let start = text[..offset]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = text[offset..]
        .find(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(|i| i + offset)
        .unwrap_or(text.len());
    start..end
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        workspace_roots: RwLock::new(Vec::new()),
        documents: DashMap::new(),
        worlds: DashMap::new(),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
