//! CDSL Language Server
//!
//! A Language Server Protocol implementation for the Continuum DSL.
//!
//! # Features
//!
//! - **Diagnostics**: Real-time parse error reporting
//! - **Hover**: Symbol information with documentation
//! - **Go-to-definition**: Jump to signal/field/operator definitions (F12 or Ctrl+Click)
//! - **Find references**: Find all usages of a symbol (Shift+F12)
//! - **Document symbols**: Navigate to any symbol in the file (Ctrl+Shift+O)
//! - **Completion**: World-aware completion with all signals, fields, etc.
//! - **Formatting**: Code formatting on save or on demand

mod formatter;
mod symbols;

use std::path::Path;

use dashmap::DashMap;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use walkdir::WalkDir;

use continuum_dsl::parse;
use symbols::{format_hover_markdown, SymbolIndex, SymbolKind as CdslSymbolKind};

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
        let diagnostics: Vec<Diagnostic> = errors
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

        // Build symbol index from AST
        if let Some(ref ast) = ast {
            let index = SymbolIndex::from_ast(ast);
            self.symbol_indices.insert(uri.clone(), index);
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

        // If we have a kind prefix, only show symbols of that kind from ALL files
        if let Some((kind, _prefix_len)) = kind_filter {
            // Collect completions from ALL indexed files in the world
            for entry in self.symbol_indices.iter() {
                for info in entry.value().get_completions().filter(|c| c.kind == kind) {
                    let lsp_kind = cdsl_kind_to_completion_kind(info.kind);

                    // Build detail string with type and title
                    let detail = match (info.ty, info.title) {
                        (Some(ty), Some(title)) => format!("{} - {}", ty, title),
                        (Some(ty), None) => ty.to_string(),
                        (None, Some(title)) => title.to_string(),
                        (None, None) => info.kind.display_name().to_string(),
                    };

                    items.push(CompletionItem {
                        label: info.path.to_string(),
                        kind: Some(lsp_kind),
                        detail: Some(detail),
                        documentation: info.doc.map(|d| {
                            Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: d.to_string(),
                            })
                        }),
                        ..Default::default()
                    });
                }
            }
        } else {
            // No kind prefix - show keywords and all symbols
            items.extend(vec![
                // Top-level declarations
                completion_item("signal", CompletionItemKind::KEYWORD, "Signal declaration"),
                completion_item("field", CompletionItemKind::KEYWORD, "Field declaration"),
                completion_item("fracture", CompletionItemKind::KEYWORD, "Fracture declaration"),
                completion_item("impulse", CompletionItemKind::KEYWORD, "Impulse declaration"),
                completion_item("chronicle", CompletionItemKind::KEYWORD, "Chronicle declaration"),
                completion_item("entity", CompletionItemKind::KEYWORD, "Entity declaration"),
                completion_item("operator", CompletionItemKind::KEYWORD, "Operator declaration"),
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
                completion_item("clamp", CompletionItemKind::FUNCTION, "Clamp value to range"),
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

            // Add all world symbols with full path
            for entry in self.symbol_indices.iter() {
                for info in entry.value().get_completions() {
                    let lsp_kind = cdsl_kind_to_completion_kind(info.kind);
                    let label = format!("{}.{}", info.kind.display_name(), info.path);

                    let detail = match (info.ty, info.title) {
                        (Some(ty), Some(title)) => format!("{} - {}", ty, title),
                        (Some(ty), None) => ty.to_string(),
                        (None, Some(title)) => title.to_string(),
                        (None, None) => info.kind.display_name().to_string(),
                    };

                    items.push(CompletionItem {
                        label,
                        kind: Some(lsp_kind),
                        detail: Some(detail),
                        documentation: info.doc.map(|d| {
                            Documentation::MarkupContent(MarkupContent {
                                kind: MarkupKind::Markdown,
                                value: d.to_string(),
                            })
                        }),
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

        // Find symbol at position
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
