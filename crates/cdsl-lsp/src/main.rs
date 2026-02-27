//! CDSL Language Server - Minimal working version during migration
//!
//! This is a minimal LSP that compiles and provides basic functionality:
//! - Document lifecycle (open/change/save/close)
//! - Diagnostics from compilation errors
//! - Document formatting
//!
//! TODO: Progressively re-enable handlers using World iteration pattern

mod formatter;
mod handlers;
mod helpers;

use continuum_functions as _;
use std::path::{Path, PathBuf};

use dashmap::DashMap;
use indexmap::IndexMap;
use tokio::sync::RwLock;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LspService, Server};
use walkdir::WalkDir;

// Engine types - pure transport layer (no custom shadow structures)
use continuum_cdsl::ast::World;
use continuum_cdsl::compile_from_memory;

use crate::helpers::offset_to_position;

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
                self.client
                    .log_message(
                        MessageType::WARNING,
                        format!(
                            "No world root found for {}, treating as standalone file",
                            path.display()
                        ),
                    )
                    .await;
                return;
            }
        };

        // Collect all .cdsl files in the world
        let sources = self.collect_world_files(&world_root);

        if sources.is_empty() {
            self.client
                .log_message(
                    MessageType::WARNING,
                    format!(
                        "No .cdsl files found in world root {}",
                        world_root.display()
                    ),
                )
                .await;
            return;
        }

        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "Compiling world at {} with {} files",
                    world_root.display(),
                    sources.len()
                ),
            )
            .await;

        // Compile the entire world
        let compile_result = compile_from_memory(sources);

        match compile_result {
            Ok(compiled_world) => {
                // Success - store the compiled world
                self.worlds
                    .insert(world_root.clone(), compiled_world.world);

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
                    let (start_line, start_char) =
                        offset_to_position(&file_content, error.span.start as usize);
                    let (end_line, end_char) =
                        offset_to_position(&file_content, error.span.end as usize);

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

                    diagnostics_by_file
                        .entry(file_path)
                        .or_default()
                        .push(diagnostic);
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
