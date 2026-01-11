//! Continuum Compiler
//!
//! Unified entry point for the Continuum DSL compilation pipeline.
//! Consolidates parsing, validation, and lowering into a single API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use continuum_dsl::ast::CompilationUnit;
pub use continuum_ir::CompiledWorld;

/// A unified diagnostic message from any phase of the compiler.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Human-readable error message.
    pub message: String,
    /// Source location (byte range), if available.
    pub span: Option<std::ops::Range<usize>>,
    /// Path to the source file, if available.
    pub file: Option<PathBuf>,
    /// Severity of the diagnostic.
    pub severity: Severity,
}

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Hint,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
            file: None,
            severity: Severity::Error,
        }
    }

    pub fn with_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.file = Some(path.into());
        self
    }

    pub fn with_span(mut self, span: std::ops::Range<usize>) -> Self {
        self.span = Some(span);
        self
    }
}

/// Primary entry point for compiling a Continuum world from source.
///
/// This function executes the full compilation pipeline:
/// 1. **Parse**: Each file in the source map is parsed into an AST.
/// 2. **Merge**: All ASTs are merged into a single compilation unit.
/// 3. **Validate**: Semantic validation is run on the merged AST.
/// 4. **Lower**: The AST is lowered to typed IR (CompiledWorld).
///
/// # Arguments
///
/// * `source_map` - A map of file paths to their CDSL source content.
///
/// # Returns
///
/// * `Ok(CompiledWorld)` - The successfully lowered IR.
/// * `Err(Vec<Diagnostic>)` - A list of all diagnostics encountered during any phase.
pub fn compile(source_map: &HashMap<PathBuf, &str>) -> Result<CompiledWorld, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let mut merged_unit = CompilationUnit::default();
    let mut has_world_def = false;

    // 1. Parse all files
    for (path, source) in source_map {
        let (result, parse_errors) = continuum_dsl::parse(source);

        for err in parse_errors {
            let span = err.span();
            diagnostics.push(Diagnostic {
                message: format!("{}", err.reason()),
                span: Some(span.start..span.end),
                file: Some(path.clone()),
                severity: Severity::Error,
            });
        }

        if let Some(unit) = result {
            for item in &unit.items {
                if matches!(item.node, continuum_dsl::ast::Item::WorldDef(_)) {
                    if has_world_def {
                        diagnostics.push(Diagnostic {
                            message:
                                "multiple world definitions found (already defined in another file)"
                                    .to_string(),
                            span: Some(item.span.clone()),
                            file: Some(path.clone()),
                            severity: Severity::Error,
                        });
                    }
                    has_world_def = true;
                }
            }
            merged_unit.items.extend(unit.items);
        }
    }

    // Stop if we have parse errors
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(diagnostics);
    }

    // Verify world definition exists
    if !has_world_def {
        diagnostics.push(Diagnostic::error(
            "no world definition found (missing world { } block)",
        ));
        return Err(diagnostics);
    }

    // 2. Validate merged unit
    let validation_errors = continuum_dsl::validate(&merged_unit);
    for err in validation_errors {
        diagnostics.push(Diagnostic {
            message: err.message,
            span: Some(err.span),
            file: None, // We lost file info during merge, but spans are still valid
            severity: Severity::Error,
        });
    }

    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return Err(diagnostics);
    }

    // 3. Lower to IR
    match continuum_ir::lower(&merged_unit) {
        Ok(world) => Ok(world),
        Err(err) => {
            diagnostics.push(Diagnostic {
                message: format!("{}", err),
                span: None, // LowerError currently doesn't provide spans
                file: None,
                severity: Severity::Error,
            });
            Err(diagnostics)
        }
    }
}

/// Helper function to load and compile a world from a directory.
pub fn compile_from_dir(world_dir: &Path) -> Result<CompiledWorld, Vec<Diagnostic>> {
    let mut source_map = HashMap::new();
    let files = collect_cdsl_files(world_dir);

    let mut sources = Vec::new();
    for path in &files {
        match std::fs::read_to_string(path) {
            Ok(content) => sources.push(content),
            Err(e) => {
                return Err(vec![Diagnostic::error(format!(
                    "failed to read {}: {}",
                    path.display(),
                    e
                ))]);
            }
        }
    }

    for (path, content) in files.iter().zip(sources.iter()) {
        source_map.insert(path.clone(), content.as_str());
    }

    compile(&source_map)
}

fn collect_cdsl_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_cdsl_files_recursive(dir, &mut files);
    files.sort();
    files
}

fn collect_cdsl_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_cdsl_files_recursive(&path, files);
            } else if path.extension().is_some_and(|e| e == "cdsl") {
                files.push(path);
            }
        }
    }
}
