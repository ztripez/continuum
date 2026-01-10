//! World Loading.
//!
//! This module provides functions to load Continuum worlds from the filesystem.
//! A world is a directory containing either a `world.yaml` manifest or a DSL
//! file with a `world` definition, plus other `.cdsl` files.
//!
//! # World Structure
//!
//! A valid world directory must contain:
//! - `world.yaml` OR a `world` definition in a `.cdsl` file
//! - One or more `*.cdsl` files - DSL source files
//!
//! # Loading Process
//!
//! 1. Validate that the directory exists
//! 2. Recursively collect all `.cdsl` files, sorted by path for determinism
//! 3. Parse and validate each file individually
//! 4. Merge all compilation units into a single [`CompilationUnit`]
//! 5. Verify that a world definition exists (manifest or DSL)
//!
//! # Example
//!
//! ```ignore
//! use continuum_dsl::loader::load_world;
//! use std::path::Path;
//!
//! let result = load_world(Path::new("worlds/earth"))?;
//! println!("Loaded {} files with {} items",
//!     result.files.len(),
//!     result.unit.items.len()
//! );
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use crate::ast::{CompilationUnit, Item};
use crate::{parse, validate, ValidationError};

/// Errors that can occur during world loading.
///
/// These errors cover the full loading pipeline: filesystem access, parsing,
/// and semantic validation. Each variant preserves the file path where the
/// error occurred to aid debugging.
#[derive(Debug)]
pub enum LoadError {
    /// The specified world directory doesn't exist or isn't a directory.
    ///
    /// The path passed to [`load_world`] must be an existing directory.
    InvalidWorldDir(PathBuf),

    /// No `world` DSL definition was found.
    ///
    /// Every valid world must have a world definition block in a DSL file.
    /// Support for `world.yaml` has been removed.
    MissingWorldDefinition(PathBuf),

    /// Failed to read a file from disk.
    ///
    /// This wraps I/O errors from the filesystem, such as permission
    /// denied or file not found (for files discovered but then deleted).
    ReadError {
        /// Path to the file that couldn't be read.
        path: PathBuf,
        /// The underlying I/O error.
        error: std::io::Error,
    },

    /// Syntax errors occurred while parsing a `.cdsl` file.
    ///
    /// Parse errors indicate malformed DSL syntax that the parser
    /// couldn't understand. Each error message includes location information.
    ParseErrors {
        /// Path to the file with parse errors.
        path: PathBuf,
        /// Human-readable parse error messages.
        errors: Vec<String>,
    },

    /// Semantic validation errors in a parsed file.
    ///
    /// Validation errors indicate well-formed syntax that violates
    /// semantic rules, such as referencing undefined functions or
    /// having inconsistent type usage.
    ValidationErrors {
        /// Path to the file with validation errors.
        path: PathBuf,
        /// The validation errors found.
        errors: Vec<ValidationError>,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::InvalidWorldDir(path) => {
                write!(f, "'{}' is not a valid directory", path.display())
            }
            LoadError::MissingWorldDefinition(path) => {
                write!(f, "no world definition found in '{}' (missing world {{ }} block in .cdsl)", path.display())
            }
            LoadError::ReadError { path, error } => {
                write!(f, "error reading {}: {}", path.display(), error)
            }
            LoadError::ParseErrors { path, errors } => {
                write!(f, "parse errors in {}:", path.display())?;
                for err in errors {
                    write!(f, "\n  - {}", err)?;
                }
                Ok(())
            }
            LoadError::ValidationErrors { path, errors } => {
                write!(f, "validation errors in {}:", path.display())?;
                for err in errors {
                    write!(f, "\n  - {}", err)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Result of loading a world
pub struct LoadResult {
    /// The merged compilation unit
    pub unit: CompilationUnit,
    /// Paths of all loaded .cdsl files
    pub files: Vec<PathBuf>,
}

/// Collect all .cdsl files in a directory (recursive)
pub fn collect_cdsl_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_cdsl_files_recursive(dir, &mut files);
    files.sort();
    files
}

fn collect_cdsl_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
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

/// Load and parse a single .cdsl file
pub fn load_file(path: &Path) -> Result<CompilationUnit, LoadError> {
    let source = fs::read_to_string(path).map_err(|e| LoadError::ReadError {
        path: path.to_path_buf(),
        error: e,
    })?;

    let (result, parse_errors) = parse(&source);

    if !parse_errors.is_empty() {
        return Err(LoadError::ParseErrors {
            path: path.to_path_buf(),
            errors: parse_errors.iter().map(|e| e.to_string()).collect(),
        });
    }

    let unit = result.ok_or_else(|| LoadError::ParseErrors {
        path: path.to_path_buf(),
        errors: vec!["failed to parse".to_string()],
    })?;

    let validation_errors = validate(&unit);
    if !validation_errors.is_empty() {
        return Err(LoadError::ValidationErrors {
            path: path.to_path_buf(),
            errors: validation_errors,
        });
    }

    Ok(unit)
}

/// Load all .cdsl files from a world directory
///
/// Verifies the directory exists, then collects and parses all .cdsl files,
/// merging them into a single compilation unit. Ensures a world definition
/// exists in the DSL.
pub fn load_world(world_dir: &Path) -> Result<LoadResult, LoadError> {
    // Validate world directory
    if !world_dir.exists() || !world_dir.is_dir() {
        return Err(LoadError::InvalidWorldDir(world_dir.to_path_buf()));
    }

    // Collect .cdsl files
    let files = collect_cdsl_files(world_dir);

    // Parse and merge all files
    let mut merged_unit = CompilationUnit::default();
    let mut has_world_def = false;

    for file in &files {
        let unit = load_file(file)?;
        for item in &unit.items {
            if matches!(item.node, Item::WorldDef(_)) {
                if has_world_def {
                    return Err(LoadError::ValidationErrors {
                        path: file.clone(),
                        errors: vec![ValidationError {
                            message: "multiple world definitions found (already defined in another file)".to_string(),
                            span: item.span.clone(),
                        }],
                    });
                }
                has_world_def = true;
            }
        }
        merged_unit.items.extend(unit.items);
    }

    // Must have a world definition
    if !has_world_def {
        return Err(LoadError::MissingWorldDefinition(world_dir.to_path_buf()));
    }

    Ok(LoadResult {
        unit: merged_unit,
        files,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_collect_cdsl_files() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        // Create some .cdsl files
        File::create(root.join("a.cdsl")).unwrap();
        File::create(root.join("b.cdsl")).unwrap();
        File::create(root.join("other.txt")).unwrap();

        // Create a subdirectory with more files
        fs::create_dir(root.join("sub")).unwrap();
        File::create(root.join("sub/c.cdsl")).unwrap();

        let files = collect_cdsl_files(root);

        assert_eq!(files.len(), 3);
        assert!(files.iter().all(|f| f.extension().unwrap() == "cdsl"));
    }

    #[test]
    fn test_load_file_simple() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.cdsl");

        let mut file = File::create(&path).unwrap();
        writeln!(file, "strata.test {{ }}").unwrap();
        writeln!(file, "era.main {{ : initial }}").unwrap();

        let unit = load_file(&path).unwrap();
        assert_eq!(unit.items.len(), 2);
    }

    #[test]
    fn test_load_world() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        // Create a .cdsl file with world definition
        let mut file = File::create(root.join("world.cdsl")).unwrap();
        writeln!(file, "world.terra {{ : title(\"Test World\") }}").unwrap();
        writeln!(file, "strata.test {{ }}").unwrap();
        writeln!(file, "era.main {{ : initial }}").unwrap();

        let result = load_world(root).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.unit.items.len(), 3);
        match &result.unit.items[0].node {
            Item::WorldDef(def) => assert_eq!(def.title.as_ref().unwrap().node, "Test World"),
            _ => panic!("expected WorldDef"),
        }
    }

    #[test]
    fn test_load_world_missing_def() {
        let dir = tempdir().unwrap();
        let result = load_world(dir.path());
        assert!(matches!(result, Err(LoadError::MissingWorldDefinition(_))));
    }

    #[test]
    fn test_load_world_invalid_dir() {
        let result = load_world(Path::new("/nonexistent/path/to/world"));
        assert!(matches!(result, Err(LoadError::InvalidWorldDir(_))));
    }

    #[test]
    fn test_load_file_parse_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.cdsl");

        let mut file = File::create(&path).unwrap();
        writeln!(file, "this is not valid cdsl syntax {{{{").unwrap();

        let result = load_file(&path);
        assert!(matches!(result, Err(LoadError::ParseErrors { .. })));
    }

    #[test]
    fn test_load_world_with_parse_error() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        // Even with parse error, we don't need world.yaml anymore to attempt loading
        let mut file = File::create(root.join("bad.cdsl")).unwrap();
        writeln!(file, "invalid syntax {{{{").unwrap();

        let result = load_world(root);
        assert!(matches!(result, Err(LoadError::ParseErrors { .. })));
    }

    #[test]
    fn test_load_error_display() {
        // Test Display impl for each error variant
        let err = LoadError::InvalidWorldDir(PathBuf::from("/test"));
        assert!(err.to_string().contains("not a valid directory"));

        let err = LoadError::MissingWorldDefinition(PathBuf::from("/test"));
        assert!(err.to_string().contains("no world definition"));

        let err = LoadError::ReadError {
            path: PathBuf::from("/test"),
            error: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
        };
        assert!(err.to_string().contains("error reading"));

        let err = LoadError::ParseErrors {
            path: PathBuf::from("/test"),
            errors: vec!["syntax error".to_string()],
        };
        let msg = err.to_string();
        assert!(msg.contains("parse errors"));
        assert!(msg.contains("syntax error"));

        let err = LoadError::ValidationErrors {
            path: PathBuf::from("/test"),
            errors: vec![],
        };
        assert!(err.to_string().contains("validation errors"));
    }

    #[test]
    fn test_collect_cdsl_files_empty_dir() {
        let dir = tempdir().unwrap();
        let files = collect_cdsl_files(dir.path());
        assert!(files.is_empty());
    }

    #[test]
    fn test_collect_cdsl_files_sorted() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        // Create files in non-alphabetical order
        File::create(root.join("z.cdsl")).unwrap();
        File::create(root.join("a.cdsl")).unwrap();
        File::create(root.join("m.cdsl")).unwrap();

        let files = collect_cdsl_files(root);

        assert_eq!(files.len(), 3);
        // Files should be sorted
        assert!(files[0].ends_with("a.cdsl"));
        assert!(files[1].ends_with("m.cdsl"));
        assert!(files[2].ends_with("z.cdsl"));
    }

    #[test]
    fn test_load_world_duplicate_def() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        let mut file1 = File::create(root.join("world1.cdsl")).unwrap();
        writeln!(file1, "world.terra {{ : title(\"One\") }}").unwrap();

        let mut file2 = File::create(root.join("world2.cdsl")).unwrap();
        writeln!(file2, "world.mars {{ : title(\"Two\") }}").unwrap();

        let result = load_world(root);
        assert!(matches!(result, Err(LoadError::ValidationErrors { .. })));
    }
}
