use crate::CompileError;
use crate::CompiledWorld;
use crate::Token;
use crate::parse_declarations;
use crate::pipeline;
use crate::{SourceMap, Span};
use continuum_cdsl_resolve::error::ErrorKind;
use logos::Logos;
use std::path::Path;
use walkdir::WalkDir;

/// Compiles a Continuum world from a root directory.
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
/// Returns a list of [`CompileError`] if any stage of the pipeline fails.
pub fn compile(root: &Path) -> Result<CompiledWorld, Vec<CompileError>> {
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

    if cdsl_files.is_empty() {
        return Err(vec![CompileError::new(
            ErrorKind::Internal,
            Span::new(0, 0, 0, 1),
            format!("No .cdsl files found in {}", root.display()),
        )]);
    }

    // 2. Lex & Parse each file
    for file_path in cdsl_files {
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

        // Lexing
        let mut lexer = Token::lexer(&source);
        let mut tokens = Vec::new();
        let mut lex_failed = false;

        while let Some(result) = lexer.next() {
            let span = lexer.span();
            match result {
                Ok(token) => {
                    tokens.push(token);
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
            continue;
        }

        // Parsing
        match parse_declarations(&tokens, file_id) {
            Ok(decls) => declarations.extend(decls),
            Err(errors) => {
                for err in errors {
                    // Map ParseError to CompileError
                    all_errors.push(CompileError::new(ErrorKind::Syntax, err.span, err.message));
                }
            }
        }
    }

    if !all_errors.is_empty() {
        return Err(all_errors);
    }

    // 3. Pipeline Resolution
    pipeline::compile(declarations)
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
