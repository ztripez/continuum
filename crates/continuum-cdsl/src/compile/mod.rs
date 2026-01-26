use crate::pipeline;
use crate::CompileError;
use crate::CompiledWorld;
use crate::Token;
use crate::{SourceMap, Span};
use continuum_cdsl_parser::parse_declarations_with_spans;
use continuum_cdsl_resolve::error::ErrorKind;
use indexmap::IndexMap;
use logos::Logos;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Normalize Unicode superscripts to caret notation (e.g., m² → m^2).
///
/// This allows terra CDSL files to use readable superscript notation
/// while keeping the lexer/parser simple.
///
/// Note: Negative superscripts must be replaced before positive ones
/// to avoid partial replacements (e.g., ⁻¹ → ^-1, not ⁻^1).
fn normalize_superscripts(source: &str) -> String {
    source
        // Negative superscripts first (multi-char sequences)
        .replace("⁻¹", "^-1")
        .replace("⁻²", "^-2")
        .replace("⁻³", "^-3")
        .replace("⁻⁴", "^-4")
        .replace("⁻⁵", "^-5")
        .replace("⁻⁶", "^-6")
        .replace("⁻⁷", "^-7")
        .replace("⁻⁸", "^-8")
        .replace("⁻⁹", "^-9")
        // Positive superscripts (single chars)
        .replace('⁰', "^0")
        .replace('¹', "^1")
        .replace('²', "^2")
        .replace('³', "^3")
        .replace('⁴', "^4")
        .replace('⁵', "^5")
        .replace('⁶', "^6")
        .replace('⁷', "^7")
        .replace('⁸', "^8")
        .replace('⁹', "^9")
}

/// Result type for compilation with source map.
pub type CompileResultWithSources = Result<CompiledWorld, (SourceMap, Vec<CompileError>)>;

/// Compiles a Continuum world from a root directory, returning source map on error.
///
/// This is the high-level public API for the CDSL compiler.
/// It performs the following steps:
/// 1. Discovers all `*.cdsl` files in the root directory (recursive).
/// 2. Sorts files by path to ensure deterministic symbol registration.
/// 3. Reads all source files and builds a [`SourceMap`].
/// 4. Lexes and parses all files into a unified list of declarations.
/// 5. Runs the full resolution and validation pipeline.
/// 6. Assembles the final [`CompiledWorld`].
///
/// # Parameters
/// - `root`: Path to the world root directory.
///
/// # Errors
/// Returns a tuple of [`SourceMap`] and list of [`CompileError`] if any stage fails.
/// The source map can be used with [`DiagnosticFormatter`] to produce rich error messages.
pub fn compile_with_sources(root: &Path) -> CompileResultWithSources {
    eprintln!(
        "compile_with_sources: discovering .cdsl files in {:?}",
        root
    );
    let mut source_map = SourceMap::new();
    let mut declarations = Vec::new();
    let mut all_errors = Vec::new();

    // 1. Discover files
    let mut cdsl_files = Vec::new();
    for entry in WalkDir::new(root) {
        match entry {
            Ok(e) => {
                if e.path().extension().map_or(false, |ext| ext == "cdsl") {
                    cdsl_files.push(e.path().to_path_buf());
                }
            }
            Err(err) => {
                all_errors.push(CompileError::new(
                    ErrorKind::Internal,
                    Span::new(0, 0, 0, 1),
                    format!("Directory traversal error: {}", err),
                ));
            }
        }
    }

    // Ensure deterministic order
    cdsl_files.sort();

    eprintln!("Found {} .cdsl files", cdsl_files.len());

    if cdsl_files.is_empty() {
        return Err((
            source_map,
            vec![CompileError::new(
                ErrorKind::Internal,
                Span::new(0, 0, 0, 1),
                format!("No .cdsl files found in {}", root.display()),
            )],
        ));
    }

    // 2. Lex & Parse each file
    eprintln!("Lexing and parsing files...");
    for file_path in cdsl_files {
        eprintln!("  Processing: {:?}", file_path);
        let source = match std::fs::read_to_string(&file_path) {
            Ok(s) => s,
            Err(e) => {
                all_errors.push(CompileError::new(
                    ErrorKind::Internal,
                    Span::new(0, 0, 0, 1),
                    format!("Failed to read file {}: {}", file_path.display(), e),
                ));
                continue;
            }
        };

        let file_id = source_map.add_file(file_path.clone(), source.clone());

        // Normalize superscripts to caret notation before lexing
        let normalized = normalize_superscripts(&source);

        // Lexing
        let mut lexer = Token::lexer(&normalized);
        let mut tokens_with_spans = Vec::new();
        let mut lex_failed = false;

        while let Some(result) = lexer.next() {
            let span = lexer.span();
            match result {
                Ok(token) => {
                    tokens_with_spans.push((token, span));
                }
                Err(_) => {
                    all_errors.push(CompileError::new(
                        ErrorKind::Syntax,
                        Span::new(file_id, span.start as u32, span.end as u32, 0),
                        format!("Invalid token in {}", file_path.display()),
                    ));
                    lex_failed = true;
                }
            }
        }

        if lex_failed {
            eprintln!("    Lexing failed, skipping parse");
            continue;
        }

        // Parsing
        eprintln!("    Parsing {} tokens...", tokens_with_spans.len());
        match parse_declarations_with_spans(&tokens_with_spans, file_id) {
            Ok(decls) => {
                eprintln!("    Parsed {} declarations", decls.len());
                declarations.extend(decls)
            }
            Err(errors) => {
                eprintln!("    Parse failed with {} errors", errors.len());
                for err in errors {
                    // Map ParseError to CompileError
                    all_errors.push(CompileError::new(ErrorKind::Syntax, err.span, err.message));
                }
            }
        }
    }

    eprintln!(
        "Lex/parse complete. Total declarations: {}",
        declarations.len()
    );

    if !all_errors.is_empty() {
        eprintln!("Returning {} errors from lex/parse", all_errors.len());
        return Err((source_map, all_errors));
    }

    // 3. Pipeline Resolution
    eprintln!("Running resolution pipeline...");
    let result = pipeline::compile(declarations).map_err(|errors| (source_map, errors));
    eprintln!("Pipeline complete");
    result
}

/// Compiles a Continuum world from in-memory sources, returning source map on error.
///
/// This API is designed for LSP/tooling use where source content is already in memory
/// (e.g., dirty editor buffers). It bypasses file discovery and reads directly from
/// the provided sources map.
///
/// # Parameters
/// - `sources`: Map of file paths to source content. Paths should be absolute or
///   relative to the world root. Files are processed in deterministic order (sorted by path).
///
/// # Returns
/// - `Ok(CompiledWorld)` on success
/// - `Err((SourceMap, Vec<CompileError>))` on failure, with source map for diagnostics
///
/// # Examples
/// ```rust,ignore
/// use indexmap::IndexMap;
/// use std::path::PathBuf;
/// use continuum_cdsl::compile_from_memory;
///
/// let mut sources = IndexMap::new();
/// sources.insert(PathBuf::from("world.cdsl"), "world demo { }".to_string());
/// sources.insert(PathBuf::from("signals.cdsl"), "signal.foo { }".to_string());
///
/// let result = compile_from_memory(sources);
/// ```
///
/// # Errors
/// Returns compilation errors with source map for rich diagnostics.
pub fn compile_from_memory(sources: IndexMap<PathBuf, String>) -> CompileResultWithSources {
    eprintln!(
        "compile_from_memory: compiling {} source files",
        sources.len()
    );

    let mut source_map = SourceMap::new();
    let mut declarations = Vec::new();
    let mut all_errors = Vec::new();

    if sources.is_empty() {
        return Err((
            source_map,
            vec![CompileError::new(
                ErrorKind::Internal,
                Span::new(0, 0, 0, 1),
                "No source files provided".to_string(),
            )],
        ));
    }

    // Process files in deterministic order (sorted by path)
    let mut sorted_sources: Vec<(PathBuf, String)> = sources.into_iter().collect();
    sorted_sources.sort_by(|a, b| a.0.cmp(&b.0));

    eprintln!("Lexing and parsing {} files...", sorted_sources.len());

    for (file_path, source) in sorted_sources {
        eprintln!("  Processing: {:?}", file_path);

        let file_id = source_map.add_file(file_path.clone(), source.clone());

        // Normalize superscripts to caret notation before lexing
        let normalized = normalize_superscripts(&source);

        // Lexing
        let mut lexer = Token::lexer(&normalized);
        let mut tokens_with_spans = Vec::new();
        let mut lex_failed = false;

        while let Some(result) = lexer.next() {
            let span = lexer.span();
            match result {
                Ok(token) => {
                    tokens_with_spans.push((token, span));
                }
                Err(_) => {
                    all_errors.push(CompileError::new(
                        ErrorKind::Syntax,
                        Span::new(file_id, span.start as u32, span.end as u32, 0),
                        format!("Invalid token in {}", file_path.display()),
                    ));
                    lex_failed = true;
                }
            }
        }

        if lex_failed {
            eprintln!("    Lexing failed, skipping parse");
            continue;
        }

        // Parsing
        eprintln!("    Parsing {} tokens...", tokens_with_spans.len());
        match parse_declarations_with_spans(&tokens_with_spans, file_id) {
            Ok(decls) => {
                eprintln!("    Parsed {} declarations", decls.len());
                declarations.extend(decls)
            }
            Err(errors) => {
                eprintln!("    Parse failed with {} errors", errors.len());
                for err in errors {
                    // Map ParseError to CompileError
                    all_errors.push(CompileError::new(ErrorKind::Syntax, err.span, err.message));
                }
            }
        }
    }

    eprintln!(
        "Lex/parse complete. Total declarations: {}",
        declarations.len()
    );

    if !all_errors.is_empty() {
        eprintln!("Returning {} errors from lex/parse", all_errors.len());
        return Err((source_map, all_errors));
    }

    // Pipeline Resolution
    eprintln!("Running resolution pipeline...");
    let result = pipeline::compile(declarations).map_err(|errors| (source_map, errors));
    eprintln!("Pipeline complete");
    result
}

/// Compiles a Continuum world from a root directory (without source map in error).
///
/// This is a compatibility wrapper around [`compile_with_sources`] that discards
/// the source map on error. For better error messages, use [`compile_with_sources`]
/// and format errors with [`DiagnosticFormatter`].
///
/// # Parameters
/// - `root`: Path to the world root directory.
///
/// # Errors
/// Returns a list of [`CompileError`] if any stage of the pipeline fails.
pub fn compile(root: &Path) -> Result<CompiledWorld, Vec<CompileError>> {
    compile_with_sources(root).map_err(|(_, errors)| errors)
}

/// Serializes a [`CompiledWorld`] to a MessagePack byte vector.
pub fn serialize_world(world: &CompiledWorld) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(world)
}

/// Deserializes a [`CompiledWorld`] from a MessagePack byte slice.
pub fn deserialize_world(data: &[u8]) -> Result<CompiledWorld, rmp_serde::decode::Error> {
    rmp_serde::from_slice(data)
}

/// Formats compilation errors with source context.
pub fn format_errors(errors: &[CompileError], _source_map: &SourceMap) -> String {
    errors
        .iter()
        .map(|e| format!("Error: {}", e.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests;
