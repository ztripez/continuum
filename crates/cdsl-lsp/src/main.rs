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

/// Find the world root directory for a given file (pure filesystem operation).
///
/// A world root is a directory containing either:
/// - A `world.yaml` file, OR
/// - A `.cdsl` file with a `world` declaration
///
/// Searches upward from the file's parent directory.
fn find_world_root(file_path: &Path) -> Option<PathBuf> {
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
                if path.extension().is_some_and(|e| e == "cdsl")
                    && let Ok(content) = std::fs::read_to_string(&path) {
                        // Simple check for "world" declaration
                        if content.contains("world ") {
                            return Some(current.to_path_buf());
                        }
                    }
            }
        }

        // Move up to parent directory
        current = current.parent()?;
    }
}

/// Collect all `.cdsl` files in a world directory tree (pure filesystem operation).
///
/// Walks the directory recursively and reads each `.cdsl` file's content.
fn collect_world_files(world_root: &Path) -> IndexMap<PathBuf, String> {
    let mut sources = IndexMap::new();

    for entry in WalkDir::new(world_root)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().extension().is_some_and(|e| e == "cdsl")
            && let Ok(content) = std::fs::read_to_string(entry.path()) {
                sources.insert(entry.path().to_path_buf(), content);
            }
    }

    sources
}

/// Scan a directory recursively for `.cdsl` files and return their contents.
///
/// Returns `(uri, text)` pairs for each discovered file.
fn scan_directory_sync(dir: &Path) -> Vec<(PathBuf, String)> {
    let mut results = Vec::new();
    for entry in WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.path().extension().is_some_and(|e| e == "cdsl")
            && let Ok(text) = std::fs::read_to_string(entry.path()) {
                results.push((entry.path().to_path_buf(), text));
            }
    }
    results
}

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
    /// Uses cached world roots to avoid filesystem I/O on every handler call.
    /// Falls back to filesystem lookup if no cached world matches.
    fn get_world_for_uri(&self, uri: &Url) -> Option<dashmap::mapref::one::Ref<'_, PathBuf, World>> {
        let path = uri.to_file_path().ok()?;

        // Fast path: check if the file's path is under any known world root
        let cached_root = self.worlds.iter().find_map(|entry| {
            if path.starts_with(entry.key()) {
                Some(entry.key().clone())
            } else {
                None
            }
        });

        if let Some(root) = cached_root {
            return self.worlds.get(&root);
        }

        // Slow path: walk filesystem to find world root (blocking, but only
        // happens once per unknown path before compilation caches the world)
        let world_root = find_world_root(&path)?;
        self.worlds.get(&world_root)
    }

    /// Parse a document and publish diagnostics for entire world.
    ///
    /// When any file in a world changes, we recompile the ENTIRE world
    /// (all .cdsl files in the world root directory tree) and distribute
    /// diagnostics back to each file.
    ///
    /// Filesystem I/O and compilation are offloaded to a blocking thread
    /// to avoid stalling the tokio event loop.
    async fn parse_and_publish_diagnostics(&self, uri: Url, _text: &str) {
        let path = match uri.to_file_path() {
            Ok(p) => p,
            Err(_) => return,
        };

        // Offload filesystem discovery + compilation to blocking thread
        let compile_task_result = tokio::task::spawn_blocking(move || {
            // Find the world root for this file
            let world_root = match find_world_root(&path) {
                Some(root) => root,
                None => return Err(format!(
                    "No world root found for {}, treating as standalone file",
                    path.display()
                )),
            };

            // Collect all .cdsl files in the world
            let sources = collect_world_files(&world_root);

            if sources.is_empty() {
                return Err(format!(
                    "No .cdsl files found in world root {}",
                    world_root.display()
                ));
            }

            let file_count = sources.len();

            // Compile the entire world (CPU-heavy)
            let compile_result = compile_from_memory(sources);

            Ok((world_root, compile_result, file_count))
        })
        .await;

        // Handle spawn_blocking join error
        let (world_root, compile_result, file_count) = match compile_task_result {
            Ok(Ok(tuple)) => tuple,
            Ok(Err(warning)) => {
                self.client
                    .log_message(MessageType::WARNING, warning)
                    .await;
                return;
            }
            Err(join_err) => {
                self.client
                    .log_message(
                        MessageType::ERROR,
                        format!("Compilation task panicked: {}", join_err),
                    )
                    .await;
                return;
            }
        };

        self.client
            .log_message(
                MessageType::INFO,
                format!(
                    "Compiling world at {} with {} files",
                    world_root.display(),
                    file_count
                ),
            )
            .await;

        match compile_result {
            Ok(compiled_world) => {
                // Success - store the compiled world
                self.worlds
                    .insert(world_root.clone(), compiled_world.world);

                // Clear diagnostics for all files in this world
                // Collect file paths on blocking thread, then publish on async side
                let world_root_clone = world_root.clone();
                let file_paths = tokio::task::spawn_blocking(move || {
                    collect_world_files(&world_root_clone)
                        .into_keys()
                        .collect::<Vec<_>>()
                })
                .await
                .unwrap_or_default();

                for path in file_paths {
                    if let Ok(file_uri) = Url::from_file_path(&path) {
                        self.client
                            .publish_diagnostics(file_uri, vec![], None)
                            .await;
                    }
                }
            }
            Err((source_map, errors)) => {
                // Compilation failed - group diagnostics by file
                // Read file contents on blocking thread for span conversion
                let diagnostics_by_file = tokio::task::spawn_blocking(move || {
                    let mut diagnostics_by_file: std::collections::HashMap<PathBuf, Vec<Diagnostic>> =
                        std::collections::HashMap::new();

                    for error in errors {
                        let file = source_map.file(&error.span);
                        let file_path = file.path.clone();

                        let file_content = match std::fs::read_to_string(&file_path) {
                            Ok(content) => content,
                            Err(_) => continue,
                        };

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

                    diagnostics_by_file
                })
                .await
                .unwrap_or_default();

                // Publish diagnostics for each file
                for (file_path, diagnostics) in &diagnostics_by_file {
                    if let Ok(file_uri) = Url::from_file_path(file_path) {
                        self.client
                            .publish_diagnostics(file_uri, diagnostics.clone(), None)
                            .await;
                    }
                }

                // Clear diagnostics for files with no errors
                let world_root_clone = world_root.clone();
                let file_paths = tokio::task::spawn_blocking(move || {
                    collect_world_files(&world_root_clone)
                        .into_keys()
                        .collect::<Vec<_>>()
                })
                .await
                .unwrap_or_default();

                for path in file_paths {
                    if !diagnostics_by_file.contains_key(&path)
                        && let Ok(file_uri) = Url::from_file_path(&path) {
                            self.client
                                .publish_diagnostics(file_uri, vec![], None)
                                .await;
                        }
                }
            }
        }
    }

    /// Scan workspace for all `.cdsl` files and index them.
    ///
    /// Filesystem I/O is offloaded to a blocking thread.
    async fn scan_workspace(&self) {
        let roots = self.workspace_roots.read().await;

        for root in roots.iter() {
            if let Ok(path) = root.to_file_path() {
                let dir = path.clone();
                let files = tokio::task::spawn_blocking(move || scan_directory_sync(&dir))
                    .await
                    .unwrap_or_default();

                for (file_path, text) in files {
                    let uri = match Url::from_file_path(&file_path) {
                        Ok(uri) => uri,
                        Err(_) => continue,
                    };

                    if !self.documents.contains_key(&uri) {
                        self.documents.insert(uri, text);
                    }
                }
            }
        }
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
