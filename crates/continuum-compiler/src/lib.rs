//! Continuum Compiler
//!
//! Unified entry point for the Continuum DSL compilation pipeline.
//! Consolidates parsing, validation, and lowering into a single API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use continuum_dsl as dsl;
pub use continuum_ir as ir;

pub use dsl::ast::CompilationUnit;
pub use ir::CompiledWorld;

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

/// Result of a compilation, including the lowered world and any diagnostics.
pub struct CompileResult {
    /// The successfully lowered world (if no errors occurred).
    pub world: Option<CompiledWorld>,
    /// All diagnostics (errors, warnings, hints) produced during compilation.
    pub diagnostics: Vec<Diagnostic>,
    /// Map of file paths to their source content (for line/col mapping).
    pub sources: HashMap<PathBuf, String>,
}

impl CompileResult {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    pub fn success(self) -> Result<CompiledWorld, Vec<Diagnostic>> {
        if self.has_errors() || self.world.is_none() {
            Err(self.diagnostics)
        } else {
            Ok(self.world.unwrap())
        }
    }

    /// Format all diagnostics into a human-readable string with line/column info.
    pub fn format_diagnostics(&self) -> String {
        let mut output = String::new();
        for diag in &self.diagnostics {
            let severity = match diag.severity {
                Severity::Error => "error",
                Severity::Warning => "warning",
                Severity::Hint => "hint",
            };

            let loc = if let (Some(file), Some(span)) = (&diag.file, &diag.span) {
                if let Some(source) = self.sources.get(file) {
                    let (line, col) = offset_to_line_col(source, span.start);
                    format!("{}:{}:{}", file.display(), line + 1, col + 1)
                } else {
                    format!("{}:at {:?}", file.display(), span)
                }
            } else {
                "unknown".to_string()
            };

            output.push_str(&format!("{}: {}: {}\n", loc, severity, diag.message));
        }
        output
    }
}

fn offset_to_line_col(text: &str, offset: usize) -> (u32, u32) {
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

/// Primary entry point for compiling a Continuum world from source.
pub fn compile(source_map: &HashMap<PathBuf, &str>) -> CompileResult {
    let mut diagnostics = Vec::new();
    let mut units = Vec::new();
    let mut has_world_def = false;
    let mut sources = HashMap::new();

    for (path, source) in source_map {
        sources.insert(path.clone(), source.to_string());
    }

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
            units.push((path.clone(), unit));
        }
    }

    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return CompileResult {
            world: None,
            diagnostics,
            sources,
        };
    }

    // Verify world definition exists
    if !has_world_def {
        diagnostics.push(Diagnostic::error(
            "no world definition found (missing world { } block)",
        ));
        return CompileResult {
            world: None,
            diagnostics,
            sources,
        };
    }

    // 2. Lower to IR (this also performs validation)
    let world = match continuum_ir::lower_multi(units.iter().map(|(p, u)| (p.clone(), u)).collect())
    {
        Ok(world) => world,
        Err(err) => {
            diagnostics.push(Diagnostic {
                message: format!("{}", err),
                span: None,
                file: None,
                severity: Severity::Error,
            });
            return CompileResult {
                world: None,
                diagnostics,
                sources,
            };
        }
    };

    // 3. Advanced Analysis
    // Cycle Detection (Error)
    let cycles = continuum_ir::analysis::cycles::find_cycles(&world);
    for cycle in cycles {
        let path_str = cycle
            .path
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" -> ");

        // Use file info from the first node in the cycle
        let first_node = world.nodes.get(&cycle.path[0]);
        let file = first_node.and_then(|n| n.file.clone());

        diagnostics.push(Diagnostic {
            message: format!("circular dependency detected: {}", path_str),
            span: Some(cycle.spans[0].clone()),
            file,
            severity: Severity::Error,
        });
    }

    // Dead Code Analysis (Warning)
    let dead_code = continuum_ir::analysis::dead_code::find_dead_code(&world);
    for signal_id in dead_code.unused_signals {
        // Find node for this signal
        let node = world.nodes.get(&signal_id);
        let span = node.map(|n| n.span.clone());
        let file = node.and_then(|n| n.file.clone());

        diagnostics.push(Diagnostic {
            message: format!("unused signal: {}", signal_id),
            span,
            file,
            severity: Severity::Warning,
        });
    }

    // Dimensional Analysis (Warning)
    let dim_diagnostics = continuum_ir::analysis::dimensions::analyze_dimensions(&world);
    for diag in dim_diagnostics {
        let node = world.nodes.get(&diag.path);
        let span = node.map(|n| n.span.clone());
        let file = node.and_then(|n| n.file.clone());

        diagnostics.push(Diagnostic {
            message: diag.message,
            span,
            file,
            severity: Severity::Warning,
        });
    }

    CompileResult {
        world: Some(world),
        diagnostics,
        sources,
    }
}

/// Helper function to load and compile a world from a directory, returning full CompileResult.
pub fn compile_from_dir_result(world_dir: &Path) -> CompileResult {
    let files = collect_cdsl_files(world_dir);
    let mut source_map = HashMap::new();
    let mut sources = Vec::new();

    for path in &files {
        match std::fs::read_to_string(path) {
            Ok(content) => sources.push(content),
            Err(e) => {
                return CompileResult {
                    world: None,
                    diagnostics: vec![Diagnostic::error(format!(
                        "failed to read {}: {}",
                        path.display(),
                        e
                    ))],
                    sources: HashMap::new(),
                };
            }
        }
    }

    for (path, content) in files.iter().zip(sources.iter()) {
        source_map.insert(path.clone(), content.as_str());
    }

    compile(&source_map)
}

/// Helper function to load and compile a world from a directory.
pub fn compile_from_dir(world_dir: &Path) -> Result<CompiledWorld, Vec<Diagnostic>> {
    compile_from_dir_result(world_dir).success()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_cycle_detection() {
        let mut source_map = HashMap::new();
        source_map.insert(
            PathBuf::from("main.cdsl"),
            r#"
            world.test { }
            era.main { : initial }
            strata.test {}

            signal.a { : Scalar : strata(test) resolve { signal.b } }
            signal.b { : Scalar : strata(test) resolve { signal.a } }
            "#,
        );

        let result = compile(&source_map);
        for d in &result.diagnostics {
            println!("Diagnostic: {}", d.message);
        }
        assert!(result.has_errors());
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.message.contains("circular dependency detected"))
        );
    }

    #[test]
    fn test_compile_dead_code_warning() {
        let mut source_map = HashMap::new();
        source_map.insert(
            PathBuf::from("main.cdsl"),
            r#"
            world.test { }
            era.main { : initial }
            strata.test {}

            signal.used { : Scalar : strata(test) resolve { prev } }
            signal.unused { : Scalar : strata(test) resolve { prev } }

            field.out { : Scalar : strata(test) measure { signal.used } }
            "#,
        );

        let result = compile(&source_map);
        for d in &result.diagnostics {
            println!("Diagnostic: {:?}", d);
        }
        assert!(!result.has_errors());
        assert!(result.world.is_some());

        let warnings: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .collect();
        assert!(
            warnings
                .iter()
                .any(|d| d.message.contains("unused signal: unused"))
        );
        assert!(
            !warnings
                .iter()
                .any(|d| d.message.contains("unused signal: used"))
        );
    }

    #[test]
    fn test_compile_dimensional_warning() {
        let mut source_map = HashMap::new();
        source_map.insert(
            PathBuf::from("main.cdsl"),
            r#"
            world.test { }
            era.main { : initial }
            strata.test {}

            signal.length { : Scalar<m> : strata(test) resolve { 10.0 } }
            signal.time { : Scalar<s> : strata(test) resolve { 2.0 } }
            
            // This should warn: adding m and s
            signal.bad { : Scalar<m> : strata(test) resolve { signal.length + signal.time } }
            
            // This should be fine: m/s
            signal.velocity { : Scalar<m/s> : strata(test) resolve { signal.length / signal.time } }
            
            field.out { : Scalar measure { signal.velocity } }
            "#,
        );

        let result = compile(&source_map);
        assert!(!result.has_errors());

        let warnings: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .collect();
        assert!(
            warnings
                .iter()
                .any(|d| d.message.contains("incompatible units"))
        );
    }
}
