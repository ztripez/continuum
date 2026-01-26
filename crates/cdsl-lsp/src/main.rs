//! CDSL Language Server - Minimal working version during migration
//!
//! This is a minimal LSP that compiles and provides basic functionality:
//! - Document lifecycle (open/change/save/close)
//! - Diagnostics from compilation errors
//! - Document formatting
//!
//! TODO: Progressively re-enable handlers using World iteration pattern

mod formatter;

use continuum_functions as _;
use continuum_kernel_registry as kernel_registry;
use std::path::{Path, PathBuf};

use dashmap::DashMap;
use indexmap::IndexMap;
use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};
use walkdir::WalkDir;

// Engine types - pure transport layer (no custom shadow structures)
use continuum_cdsl::ast::{Index as NodeIndex, Node, World};
use continuum_cdsl::{compile_from_memory};

/// Semantic token types for syntax highlighting.
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
    workspace_roots: RwLock<Vec<Url>>,
    documents: DashMap<Url, String>,
    /// Compiled worlds mapped by world root directory - pure transport, no shadow structures
    /// Key: world root path, Value: compiled World
    worlds: DashMap<PathBuf, World>,
}

impl Backend {
    /// Get the compiled world for a given file URI.
    /// 
    /// This finds the world root for the file and returns the cached compiled world.
    fn get_world_for_uri(&self, uri: &Url) -> Option<dashmap::mapref::one::Ref<PathBuf, World>> {
        let path = uri.to_file_path().ok()?;
        let world_root = self.find_world_root(&path)?;
        self.worlds.get(&world_root)
    }

    /// Find the world root directory for a given file.
    /// 
    /// A world root is a directory containing either:
    /// - A `world.yaml` file, OR
    /// - A `.cdsl` file with a `world` declaration
    /// 
    /// We search upwards from the file's directory until we find one.
    fn find_world_root(&self, file_path: &Path) -> Option<PathBuf> {
        let mut current = file_path.parent()?;
        
        loop {
            // Check for world.yaml
            if current.join("world.yaml").exists() {
                return Some(current.to_path_buf());
            }
            
            // Check for any .cdsl file with world declaration
            if let Ok(entries) = std::fs::read_dir(current) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.extension().is_some_and(|e| e == "cdsl") {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            // Simple check for "world" declaration
                            if content.contains("world ") {
                                return Some(current.to_path_buf());
                            }
                        }
                    }
                }
            }
            
            // Move up to parent directory
            current = current.parent()?;
        }
    }
    
    /// Collect all .cdsl files in a world directory tree.
    fn collect_world_files(&self, world_root: &Path) -> IndexMap<PathBuf, String> {
        let mut sources = IndexMap::new();
        
        for entry in WalkDir::new(world_root)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().extension().is_some_and(|e| e == "cdsl") {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    sources.insert(entry.path().to_path_buf(), content);
                }
            }
        }
        
        sources
    }

    /// Parse a document and publish diagnostics for entire world.
    /// 
    /// When any file in a world changes, we recompile the ENTIRE world
    /// (all .cdsl files in the world root directory tree) and distribute
    /// diagnostics back to each file.
    async fn parse_and_publish_diagnostics(&self, uri: Url, _text: &str) {
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return,
        };

        // Find the world root for this file
        let world_root = match self.find_world_root(&path) {
            Some(root) => root,
            None => {
                // No world root found - treat as standalone file for now
                self.client.log_message(MessageType::WARNING, format!(
                    "No world root found for {}, treating as standalone file", 
                    path.display()
                )).await;
                return;
            }
        };

        // Collect all .cdsl files in the world
        let sources = self.collect_world_files(&world_root);
        
        if sources.is_empty() {
            self.client.log_message(MessageType::WARNING, 
                format!("No .cdsl files found in world root {}", world_root.display())
            ).await;
            return;
        }

        self.client.log_message(MessageType::INFO, 
            format!("Compiling world at {} with {} files", world_root.display(), sources.len())
        ).await;

        // Compile the entire world
        let compile_result = compile_from_memory(sources);

        match compile_result {
            Ok(compiled_world) => {
                // Success - store the compiled world
                self.worlds.insert(world_root.clone(), compiled_world.world);
                
                // Clear diagnostics for all files in this world
                for path in self.collect_world_files(&world_root).keys() {
                    if let Ok(file_uri) = Url::from_file_path(path) {
                        self.client
                            .publish_diagnostics(file_uri, vec![], None)
                            .await;
                    }
                }
            }
            Err((source_map, errors)) => {
                // Compilation failed - group diagnostics by file
                let mut diagnostics_by_file: std::collections::HashMap<PathBuf, Vec<Diagnostic>> = 
                    std::collections::HashMap::new();
                
                for error in errors {
                    // Get the source file for this error
                    let file = source_map.file(&error.span);
                    let file_path = file.path.clone();
                    
                    // Read the file content to convert byte offsets to line/col
                    let file_content = match std::fs::read_to_string(&file_path) {
                        Ok(content) => content,
                        Err(_) => continue,
                    };
                    
                    // Convert engine span (byte offsets) to LSP positions (line/col)
                    let (start_line, start_char) = offset_to_position(&file_content, error.span.start as usize);
                    let (end_line, end_char) = offset_to_position(&file_content, error.span.end as usize);

                    let diagnostic = Diagnostic {
                        range: Range {
                            start: Position::new(start_line, start_char),
                            end: Position::new(end_line, end_char),
                        },
                        severity: Some(DiagnosticSeverity::ERROR),
                        source: Some("cdsl".to_string()),
                        message: error.message.clone(),
                        ..Default::default()
                    };
                    
                    diagnostics_by_file.entry(file_path).or_default().push(diagnostic);
                }
                
                // Publish diagnostics for each file
                for (file_path, diagnostics) in &diagnostics_by_file {
                    if let Ok(file_uri) = Url::from_file_path(file_path) {
                        self.client
                            .publish_diagnostics(file_uri, diagnostics.clone(), None)
                            .await;
                    }
                }
                
                // Clear diagnostics for files with no errors
                for path in self.collect_world_files(&world_root).keys() {
                    if !diagnostics_by_file.contains_key(path) {
                        if let Ok(file_uri) = Url::from_file_path(path) {
                            self.client
                                .publish_diagnostics(file_uri, vec![], None)
                                .await;
                        }
                    }
                }
            }
        }
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

        // Store the document text for later use
        self.documents.insert(uri, text);
        
        // Note: We don't compile individual files during indexing anymore.
        // Compilation happens when files are opened/changed and compiles the entire world.
    }

    /// Check if the workspace has a world definition.
    fn has_world_definition(&self) -> bool {
        !self.worlds.is_empty()
    }

    /// Run workspace-level validation (now handled by unified compiler).
    async fn validate_workspace(&self) {
        // Validation is now handled during compilation
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
                document_formatting_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                definition_provider: Some(OneOf::Left(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                references_provider: Some(OneOf::Left(true)),
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
                workspace_symbol_provider: Some(OneOf::Left(true)),
                folding_range_provider: Some(FoldingRangeProviderCapability::Simple(true)),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                inlay_hint_provider: Some(OneOf::Left(true)),
                rename_provider: Some(OneOf::Right(RenameOptions {
                    prepare_provider: Some(true),
                    work_done_progress_options: Default::default(),
                })),
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
        
        // Note: We don't remove worlds on file close since worlds are shared
        // across all files in the world directory. Worlds are cached by root path.

        self.client.publish_diagnostics(uri, vec![], None).await;
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

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        // Search globals
        for (_path, node) in world.globals.iter() {
            if node.span.start as usize <= offset && offset <= node.span.end as usize {
                let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
                let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                })));
            }
        }

        // Search members
        for (_path, node) in world.members.iter() {
            if node.span.start as usize <= offset && offset <= node.span.end as usize {
                let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
                let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

                return Ok(Some(GotoDefinitionResponse::Scalar(Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                })));
            }
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

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        // Search globals (signals, fields, operators, etc.)
        for (_path, node) in world.globals.iter() {
            if node.span.start as usize <= offset && offset <= node.span.end as usize {
                return Ok(Some(format_hover_from_node(node)));
            }
        }

        // Search members (per-entity primitives)
        for (_path, node) in world.members.iter() {
            if node.span.start as usize <= offset && offset <= node.span.end as usize {
                return Ok(Some(format_hover_from_node(node)));
            }
        }

        // Search entities
        for (_path, entity) in world.entities.iter() {
            if entity.span.start as usize <= offset && offset <= entity.span.end as usize {
                let mut parts = Vec::new();
                parts.push(format!("Entity: `{}`", entity.path));
                if let Some(ref doc) = entity.doc {
                    parts.push(doc.clone());
                }
                
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: parts.join("\n\n"),
                    }),
                    range: None,
                }));
            }
        }

        Ok(None)
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let _position = params.text_document_position.position;

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let mut items = Vec::new();

        // Add all globals as completion items
        for (_path, node) in world.globals.iter() {
            let role_name = node.role_id().spec().name;
            
            let kind = match node.role_id() {
                continuum_cdsl::ast::RoleId::Signal => CompletionItemKind::VARIABLE,
                continuum_cdsl::ast::RoleId::Field => CompletionItemKind::PROPERTY,
                continuum_cdsl::ast::RoleId::Operator => CompletionItemKind::METHOD,
                continuum_cdsl::ast::RoleId::Impulse => CompletionItemKind::EVENT,
                continuum_cdsl::ast::RoleId::Fracture => CompletionItemKind::EVENT,
                continuum_cdsl::ast::RoleId::Chronicle => CompletionItemKind::CLASS,
            };

            let detail = if let Some(ref output) = node.output {
                format!("{:?}", output) // TODO: proper Type formatting
            } else {
                role_name.to_string()
            };

            let documentation = node.doc.as_ref().map(|doc| {
                Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: doc.clone(),
                })
            });

            items.push(CompletionItem {
                label: node.path.to_string(),
                kind: Some(kind),
                detail: Some(detail),
                documentation,
                ..Default::default()
            });
        }

        // Add keywords
        items.extend(vec![
            completion_item("signal", CompletionItemKind::KEYWORD, "Signal declaration"),
            completion_item("field", CompletionItemKind::KEYWORD, "Field declaration"),
            completion_item("operator", CompletionItemKind::KEYWORD, "Operator declaration"),
            completion_item("impulse", CompletionItemKind::KEYWORD, "Impulse declaration"),
            completion_item("fracture", CompletionItemKind::KEYWORD, "Fracture declaration"),
            completion_item("chronicle", CompletionItemKind::KEYWORD, "Chronicle declaration"),
            completion_item("entity", CompletionItemKind::KEYWORD, "Entity declaration"),
            completion_item("strata", CompletionItemKind::KEYWORD, "Strata declaration"),
            completion_item("era", CompletionItemKind::KEYWORD, "Era declaration"),
            completion_item("resolve", CompletionItemKind::KEYWORD, "Resolve block"),
            completion_item("measure", CompletionItemKind::KEYWORD, "Measure block"),
            completion_item("collect", CompletionItemKind::KEYWORD, "Collect block"),
            completion_item("when", CompletionItemKind::KEYWORD, "When condition"),
            completion_item("emit", CompletionItemKind::KEYWORD, "Emit block"),
            completion_item("assert", CompletionItemKind::KEYWORD, "Assert block"),
            completion_item("if", CompletionItemKind::KEYWORD, "Conditional"),
            completion_item("else", CompletionItemKind::KEYWORD, "Else branch"),
            completion_item("let", CompletionItemKind::KEYWORD, "Let binding"),
            completion_item("prev", CompletionItemKind::VARIABLE, "Previous value"),
            completion_item("dt", CompletionItemKind::VARIABLE, "Time delta"),
            completion_item("collected", CompletionItemKind::VARIABLE, "Collected value"),
        ]);

        Ok(Some(CompletionResponse::Array(items)))
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

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let mut symbols = Vec::new();

        // Add all globals (signals, fields, operators, etc.)
        for (_path, node) in world.globals.iter() {
            let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
            let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

            #[allow(deprecated)]
            symbols.push(SymbolInformation {
                name: node.path.to_string(),
                kind: role_to_lsp_symbol_kind(node.role_id()),
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
            });
        }

        // Add all members
        for (_path, node) in world.members.iter() {
            let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
            let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

            #[allow(deprecated)]
            symbols.push(SymbolInformation {
                name: format!("{} (member)", node.path),
                kind: role_to_lsp_symbol_kind(node.role_id()),
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
            });
        }

        // Add entities
        for (_path, entity) in world.entities.iter() {
            let (start_line, start_char) = offset_to_position(&doc, entity.span.start as usize);
            let (end_line, end_char) = offset_to_position(&doc, entity.span.end as usize);

            #[allow(deprecated)]
            symbols.push(SymbolInformation {
                name: entity.path.to_string(),
                kind: SymbolKind::CLASS,
                tags: None,
                deprecated: None,
                location: Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                },
                container_name: Some("entity".to_string()),
            });
        }

        Ok(Some(DocumentSymbolResponse::Flat(symbols)))
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        let uri = &params.text_document_position.text_document.uri;
        let position = params.text_document_position.position;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);
        let mut locations = Vec::new();

        // Find the symbol at cursor
        for (_path, node) in world.globals.iter() {
            if node.span.start as usize <= offset && offset <= node.span.end as usize {
                // Found the symbol - return its definition location
                let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
                let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

                locations.push(Location {
                    uri: uri.clone(),
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                });

                // TODO: Search for actual references in expressions
                // This requires walking the AST expressions to find Path references
                // For now, we just return the definition itself
                break;
            }
        }

        if locations.is_empty() {
            Ok(None)
        } else {
            Ok(Some(locations))
        }
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

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let mut tokens: Vec<(usize, usize, u32)> = Vec::new(); // (start, length, token_type)

        // Token type indices (must match SEMANTIC_TOKEN_TYPES order)
        // 0: signal, 1: field, 2: operator, 3: function, 4: type,
        // 5: strata, 6: era, 7: impulse, 8: fracture, 9: chronicle, 10: entity, 11: keyword

        // Add tokens for all globals
        for (_path, node) in world.globals.iter() {
            let token_type = match node.role_id() {
                continuum_cdsl::ast::RoleId::Signal => 0,
                continuum_cdsl::ast::RoleId::Field => 1,
                continuum_cdsl::ast::RoleId::Operator => 2,
                continuum_cdsl::ast::RoleId::Impulse => 7,
                continuum_cdsl::ast::RoleId::Fracture => 8,
                continuum_cdsl::ast::RoleId::Chronicle => 9,
            };

            tokens.push((
                node.span.start as usize,
                (node.span.end - node.span.start) as usize,
                token_type,
            ));
        }

        // Add tokens for entities
        for (_path, entity) in world.entities.iter() {
            tokens.push((
                entity.span.start as usize,
                (entity.span.end - entity.span.start) as usize,
                10, // entity
            ));
        }

        // Sort by position
        tokens.sort_by_key(|t| t.0);

        // Convert to LSP semantic tokens format (delta encoding)
        let mut data = Vec::new();
        let mut last_line = 0;
        let mut last_start = 0;

        for (start, len, token_type) in tokens {
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
                token_modifiers_bitset: 0,
            });

            last_line = line;
            last_start = col;
        }

        Ok(Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data,
        })))
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<Vec<SymbolInformation>>> {
        let query = params.query.to_lowercase();
        let mut results = Vec::new();

        // Search all indexed worlds
        for entry in self.worlds.iter() {
            let world_root = entry.key();
            let world = entry.value();

            // For each node in the world, find its file and create a symbol
            // Search globals
            for (_path, node) in world.globals.iter() {
                let path_str = node.path.to_string().to_lowercase();
                if query.is_empty() || path_str.contains(&query) {
                    // Get the file for this node
                    let node_file = match &node.file {
                        Some(f) => f,
                        None => continue,
                    };
                    
                    // Read the file to convert spans
                    let doc = match std::fs::read_to_string(node_file) {
                        Ok(content) => content,
                        Err(_) => continue,
                    };
                    
                    let uri = match Url::from_file_path(node_file) {
                        Ok(u) => u,
                        Err(_) => continue,
                    };
                    
                    let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
                    let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

                    #[allow(deprecated)]
                    results.push(SymbolInformation {
                        name: node.path.to_string(),
                        kind: role_to_lsp_symbol_kind(node.role_id()),
                        tags: None,
                        deprecated: None,
                        location: Location {
                            uri: uri.clone(),
                            range: Range {
                                start: Position::new(start_line, start_char),
                                end: Position::new(end_line, end_char),
                            },
                        },
                        container_name: Some(node.role_id().spec().name.to_string()),
                    });
                }
            }

            // TODO: Search entities
            // Entities don't have file field, need SourceMap to resolve their locations
            // For now, we only search globals and members which have file information
        }

        // Sort by relevance (exact prefix matches first)
        results.sort_by(|a, b| {
            let a_starts = a.name.to_lowercase().starts_with(&query);
            let b_starts = b.name.to_lowercase().starts_with(&query);
            match (a_starts, b_starts) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.name.len().cmp(&b.name.len()),
            }
        });

        results.truncate(100); // Limit results

        Ok(Some(results))
    }

    async fn folding_range(&self, params: FoldingRangeParams) -> Result<Option<Vec<FoldingRange>>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let mut ranges = Vec::new();

        // Create folding ranges for all globals
        for (_path, node) in world.globals.iter() {
            let (start_line, _) = offset_to_position(&doc, node.span.start as usize);
            let (end_line, _) = offset_to_position(&doc, node.span.end as usize);

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

        // Create folding ranges for entities
        for (_path, entity) in world.entities.iter() {
            let (start_line, _) = offset_to_position(&doc, entity.span.start as usize);
            let (end_line, _) = offset_to_position(&doc, entity.span.end as usize);

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

        Ok(Some(ranges))
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let _uri = &params.text_document_position_params.text_document.uri;
        let _position = params.text_document_position_params.position;

        // TODO: Parse cursor context to find function call and active parameter
        // For now, return None (no signature help)
        // Full implementation requires:
        // 1. Finding function call at cursor
        // 2. Determining active parameter index
        // 3. Looking up function signature in World or kernel registry
        
        Ok(None)
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> Result<Option<Vec<InlayHint>>> {
        let uri = &params.text_document.uri;

        let doc = match self.documents.get(uri) {
            Some(doc) => doc.clone(),
            None => return Ok(None),
        };

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let mut hints = Vec::new();

        // Add type hints for nodes with resolved output types
        for (_path, node) in world.globals.iter() {
            if let Some(ref output_type) = node.output {
                let (line, col) = offset_to_position(&doc, node.span.end as usize);
                
                hints.push(InlayHint {
                    position: Position::new(line, col),
                    label: InlayHintLabel::String(format!(": {:?}", output_type)),
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

        let world = match self.get_world_for_uri(uri) {
            Some(world) => world,
            None => return Ok(None),
        };

        let offset = position_to_offset(&doc, position);

        // Check if cursor is on a renameable symbol
        for (_path, node) in world.globals.iter() {
            if node.span.start as usize <= offset && offset <= node.span.end as usize {
                let (start_line, start_char) = offset_to_position(&doc, node.span.start as usize);
                let (end_line, end_char) = offset_to_position(&doc, node.span.end as usize);

                return Ok(Some(PrepareRenameResponse::Range(Range {
                    start: Position::new(start_line, start_char),
                    end: Position::new(end_line, end_char),
                })));
            }
        }

        Ok(None)
    }

    async fn rename(&self, params: RenameParams) -> Result<Option<WorkspaceEdit>> {
        // TODO: Implement full rename
        // Requires:
        // 1. Finding all references to the symbol
        // 2. Creating edits for all occurrences
        // 3. Handling cross-file renames
        // For now, return None (rename not supported)
        
        let _uri = &params.text_document_position.text_document.uri;
        let _position = params.text_document_position.position;
        let _new_name = params.new_name;

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
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Helper to create completion items
fn completion_item(label: &str, kind: CompletionItemKind, detail: &str) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

/// Convert engine RoleId to LSP SymbolKind for protocol transport
fn role_to_lsp_symbol_kind(role: continuum_cdsl::ast::RoleId) -> SymbolKind {
    use continuum_cdsl::ast::RoleId;
    match role {
        RoleId::Signal => SymbolKind::VARIABLE,
        RoleId::Field => SymbolKind::PROPERTY,
        RoleId::Operator => SymbolKind::METHOD,
        RoleId::Impulse => SymbolKind::EVENT,
        RoleId::Fracture => SymbolKind::EVENT,
        RoleId::Chronicle => SymbolKind::CLASS,
    }
}

/// Format hover markdown from an engine Node
fn format_hover_from_node<I: NodeIndex>(node: &Node<I>) -> Hover {
    let mut parts = Vec::new();

    // Title (if present)
    if let Some(ref title) = node.title {
        parts.push(format!("**{}**", title));
    }

    // Role and path
    let role_name = node.role_id().spec().name;
    parts.push(format!("{}: `{}`", role_name, node.path));

    // Type (if resolved)
    if let Some(ref output) = node.output {
        parts.push(format!("Type: `{:?}`", output)); // TODO: proper Type formatting
    }

    // Documentation
    if let Some(ref doc) = node.doc {
        parts.push(doc.clone());
    }

    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: parts.join("\n\n"),
        }),
        range: None,
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

// =============================================================================
// Main
// =============================================================================

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
