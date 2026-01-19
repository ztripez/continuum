//! Compile-time error reporting and diagnostics.
//!
//! This module provides a structured error system for the CDSL compiler.
//! Errors are rich diagnostics with source locations, messages, and optional hints.
//!
//! # Design
//!
//! - `CompileError` — single diagnostic with primary and optional secondary spans
//! - `ErrorKind` — categorizes errors by compiler phase
//! - `Severity` — error, warning, or note
//! - `DiagnosticFormatter` — formats diagnostics with source snippets
//!
//! # Examples
//!
//! ```
//! # use continuum_cdsl::error::*;
//! # use continuum_cdsl::foundation::{Span, Path};
//! # let span = Span::new(0, 0, 5, 1);
//! # let path = Path::from_str("test.signal");
//! let error = CompileError::new(
//!     ErrorKind::DuplicateName,
//!     span,
//!     format!("duplicate definition of '{}'", path)
//! );
//! ```

use crate::foundation::{SourceMap, Span};
use std::fmt;

/// Compilation diagnostic with source location and message.
///
/// Each diagnostic has:
/// - Primary span (where the error occurred)
/// - Error kind (categorizes the error)
/// - Message (human-readable explanation)
/// - Optional secondary labels (related code locations)
/// - Optional notes (additional context or suggestions)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    /// Category of this error
    pub kind: ErrorKind,
    /// Severity level
    pub severity: Severity,
    /// Primary source location
    pub span: Span,
    /// Primary error message
    pub message: String,
    /// Additional labeled spans
    pub labels: Vec<Label>,
    /// Additional notes or hints
    pub notes: Vec<String>,
}

/// Category of compilation error.
///
/// Errors are categorized by the compiler phase that detected them.
/// This enables filtering, statistics, and phase-specific error recovery.
///
/// # Invariant
///
/// The discriminant values must match the ERROR_KIND_NAMES array indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ErrorKind {
    // Parse phase
    /// Syntax error (invalid token, unexpected EOF, etc.)
    Syntax = 0,

    // Name resolution phase
    /// Undefined symbol (signal, field, config, etc.)
    UndefinedName = 1,
    /// Duplicate definition at same path
    DuplicateName = 2,
    /// Ambiguous reference (could refer to multiple symbols)
    AmbiguousName = 3,

    // Type resolution phase
    /// Type mismatch (expected X, got Y)
    TypeMismatch = 4,
    /// Unknown type name
    UnknownType = 5,
    /// Recursive type definition
    RecursiveType = 6,

    // Shape validation
    /// Shape mismatch (Vec2 vs Vec3, etc.)
    ShapeMismatch = 7,
    /// Dimension mismatch in matrix operations
    DimensionMismatch = 8,

    // Unit validation
    /// Unit mismatch (m/s vs K, etc.)
    UnitMismatch = 9,
    /// Invalid unit expression
    InvalidUnit = 10,

    // Kernel validation
    /// Unknown kernel name
    UnknownKernel = 11,
    /// Wrong number of arguments to kernel
    WrongArgCount = 12,
    /// Argument shape doesn't satisfy kernel constraint
    InvalidKernelShape = 13,
    /// Argument unit doesn't satisfy kernel constraint
    InvalidKernelUnit = 14,

    // Capability validation
    /// Capability not available in this phase (e.g., prev in measure)
    MissingCapability = 15,
    /// Invalid use of capability (e.g., emit with wrong target)
    InvalidCapability = 16,
    /// Phase boundary violation (e.g., field read in kernel execution)
    PhaseBoundaryViolation = 23,

    // Effect validation
    /// Effect (emit) used in pure phase
    EffectInPureContext = 17,

    // Structure validation
    /// Circular dependency detected
    CyclicDependency = 18,
    /// Path collision (signal.x conflicts with signal field x)
    PathCollision = 19,

    // Execution compilation
    /// Cannot compile to executable form
    CompilationFailed = 20,

    // Uses declaration validation
    /// Missing uses() declaration for dangerous function
    MissingUsesDeclaration = 21,

    // Generic
    /// Internal compiler error (bug in compiler)
    Internal = 24,
}

/// Human-readable names for error kinds.
///
/// Index matches ErrorKind discriminant.
const ERROR_KIND_NAMES: &[&str] = &[
    "syntax error",             // 0: Syntax
    "undefined name",           // 1: UndefinedName
    "duplicate name",           // 2: DuplicateName
    "ambiguous name",           // 3: AmbiguousName
    "type mismatch",            // 4: TypeMismatch
    "unknown type",             // 5: UnknownType
    "recursive type",           // 6: RecursiveType
    "shape mismatch",           // 7: ShapeMismatch
    "dimension mismatch",       // 8: DimensionMismatch
    "unit mismatch",            // 9: UnitMismatch
    "invalid unit",             // 10: InvalidUnit
    "unknown kernel",           // 11: UnknownKernel
    "wrong argument count",     // 12: WrongArgCount
    "invalid kernel shape",     // 13: InvalidKernelShape
    "invalid kernel unit",      // 14: InvalidKernelUnit
    "missing capability",       // 15: MissingCapability
    "invalid capability",       // 16: InvalidCapability
    "effect in pure context",   // 17: EffectInPureContext
    "cyclic dependency",        // 18: CyclicDependency
    "path collision",           // 19: PathCollision
    "compilation failed",       // 20: CompilationFailed
    "missing uses declaration", // 21: MissingUsesDeclaration
    "reserved",                 // 22: (reserved)
    "phase boundary violation", // 23: PhaseBoundaryViolation
    "internal compiler error",  // 24: Internal
];

/// Diagnostic severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational note (not an error)
    Note,
    /// Warning (code is valid but suspicious)
    Warning,
    /// Error (compilation cannot proceed)
    Error,
}

/// Secondary labeled span in a diagnostic.
///
/// Used to point to related code locations (e.g., "first defined here").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    /// Source location
    pub span: Span,
    /// Label text
    pub message: String,
}

impl CompileError {
    /// Creates a new error diagnostic.
    ///
    /// # Parameters
    ///
    /// * `kind` - Error category
    /// * `span` - Primary source location
    /// * `message` - Human-readable error message
    ///
    /// # Returns
    ///
    /// A new error with severity `Error` and no secondary labels or notes.
    pub fn new(kind: ErrorKind, span: Span, message: String) -> Self {
        Self::with_severity(kind, Severity::Error, span, message)
    }

    /// Creates a new warning diagnostic.
    ///
    /// # Parameters
    ///
    /// * `kind` - Error category
    /// * `span` - Primary source location
    /// * `message` - Human-readable warning message
    ///
    /// # Returns
    ///
    /// A new diagnostic with severity `Warning`.
    pub fn warning(kind: ErrorKind, span: Span, message: String) -> Self {
        Self::with_severity(kind, Severity::Warning, span, message)
    }

    /// Creates a new note diagnostic.
    ///
    /// # Parameters
    ///
    /// * `kind` - Error category
    /// * `span` - Primary source location
    /// * `message` - Informational message
    ///
    /// # Returns
    ///
    /// A new diagnostic with severity `Note`.
    pub fn note(kind: ErrorKind, span: Span, message: String) -> Self {
        Self::with_severity(kind, Severity::Note, span, message)
    }

    /// Internal constructor with explicit severity.
    fn with_severity(kind: ErrorKind, severity: Severity, span: Span, message: String) -> Self {
        Self {
            kind,
            severity,
            span,
            message,
            labels: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Adds a secondary labeled span.
    ///
    /// # Parameters
    ///
    /// * `span` - Related source location
    /// * `message` - Label text (e.g., "first defined here")
    ///
    /// # Returns
    ///
    /// Self (for chaining).
    pub fn with_label(mut self, span: Span, message: String) -> Self {
        self.labels.push(Label { span, message });
        self
    }

    /// Adds a note or hint.
    ///
    /// # Parameters
    ///
    /// * `note` - Additional context or suggestion
    ///
    /// # Returns
    ///
    /// Self (for chaining).
    pub fn with_note(mut self, note: String) -> Self {
        self.notes.push(note);
        self
    }
}

impl ErrorKind {
    /// Returns a human-readable name for this error kind.
    ///
    /// # Parameters
    ///
    /// (none - uses `self`)
    ///
    /// # Returns
    ///
    /// String slice with the error kind name.
    pub fn name(self) -> &'static str {
        ERROR_KIND_NAMES[self as usize]
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Note => write!(f, "note"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}: {}",
            self.severity,
            self.kind.name(),
            self.message
        )
    }
}

impl std::error::Error for CompileError {}

/// Result type for compilation operations.
pub type CompileResult<T> = Result<T, CompileError>;

/// Formats diagnostics with source code context.
///
/// Produces rich error messages with:
/// - File path and line/column location
/// - Source code snippet
/// - Visual indicators (^^^) under error spans
/// - Secondary labels
/// - Notes and hints
///
/// # Examples
///
/// ```
/// # use continuum_cdsl::error::*;
/// # use continuum_cdsl::foundation::{Span, SourceMap};
/// # use std::path::PathBuf;
/// let mut sources = SourceMap::new();
/// let file_id = sources.add_file(PathBuf::from("test.cdsl"), "let x = foo".to_string());
/// let span = Span::new(file_id, 8, 11, 1);
///
/// let error = CompileError::new(
///     ErrorKind::UndefinedName,
///     span,
///     "undefined symbol 'foo'".to_string()
/// );
///
/// let formatter = DiagnosticFormatter::new(&sources);
/// let formatted = formatter.format(&error);
/// ```
pub struct DiagnosticFormatter<'a> {
    sources: &'a SourceMap,
}

impl<'a> DiagnosticFormatter<'a> {
    /// Creates a new diagnostic formatter.
    ///
    /// # Parameters
    ///
    /// * `sources` - SourceMap containing all source files
    ///
    /// # Returns
    ///
    /// New formatter instance.
    pub fn new(sources: &'a SourceMap) -> Self {
        Self { sources }
    }

    /// Formats a diagnostic as a string with source context.
    ///
    /// # Parameters
    ///
    /// * `error` - The diagnostic to format
    ///
    /// # Returns
    ///
    /// Formatted string with source location, snippet, and labels.
    pub fn format(&self, error: &CompileError) -> String {
        let mut output = String::new();

        // Header: severity and message
        output.push_str(&format!(
            "{}: {}: {}\n",
            error.severity,
            error.kind.name(),
            error.message
        ));

        // Location and snippet
        let file_path = self.sources.file_path(&error.span);
        let (line, col) = self.sources.line_col(&error.span);
        output.push_str(&format!("  --> {}:{}:{}\n", file_path.display(), line, col));

        // Source line
        let file = self.sources.file(&error.span);
        if let Some(source_line) = file.line_text(line) {
            output.push_str(&format!("   |\n"));
            output.push_str(&format!("{:3} | {}\n", line, source_line));

            // Underline
            let start_col = col as usize;
            let span_len = (error.span.end - error.span.start) as usize;
            let end_col = (start_col + span_len).min(source_line.len() + 1);
            let underline = " ".repeat(start_col.saturating_sub(1))
                + &"^".repeat(end_col.saturating_sub(start_col).max(1));
            output.push_str(&format!("   | {}\n", underline));
        }

        // Secondary labels
        for label in &error.labels {
            output.push_str(&format!("   = note: {}\n", label.message));

            let (label_line, label_col) = self.sources.line_col(&label.span);
            let label_path = self.sources.file_path(&label.span);
            output.push_str(&format!(
                "     at {}:{}:{}\n",
                label_path.display(),
                label_line,
                label_col
            ));
        }

        // Notes
        for note in &error.notes {
            output.push_str(&format!("   = help: {}\n", note));
        }

        output
    }

    /// Formats multiple diagnostics.
    ///
    /// # Parameters
    ///
    /// * `errors` - Slice of diagnostics to format
    ///
    /// # Returns
    ///
    /// Formatted string with all diagnostics separated by blank lines.
    pub fn format_all(&self, errors: &[CompileError]) -> String {
        errors
            .iter()
            .map(|e| self.format(e))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::SourceMap;
    use std::path::PathBuf;

    fn dummy_span() -> Span {
        Span::new(0, 0, 5, 1)
    }

    fn test_sources() -> SourceMap {
        let mut sources = SourceMap::new();
        sources.add_file(
            PathBuf::from("test.cdsl"),
            "let x = foo\nlet y = bar".to_string(),
        );
        sources
    }

    #[test]
    fn test_error_creation() {
        let err = CompileError::new(
            ErrorKind::DuplicateName,
            dummy_span(),
            "duplicate signal 'foo'".to_string(),
        );

        assert_eq!(err.kind, ErrorKind::DuplicateName);
        assert_eq!(err.severity, Severity::Error);
        assert_eq!(err.message, "duplicate signal 'foo'");
        assert!(err.labels.is_empty());
        assert!(err.notes.is_empty());
    }

    #[test]
    fn test_warning_creation() {
        let warn = CompileError::warning(
            ErrorKind::TypeMismatch,
            dummy_span(),
            "unused variable".to_string(),
        );

        assert_eq!(warn.severity, Severity::Warning);
    }

    #[test]
    fn test_note_creation() {
        let note = CompileError::note(
            ErrorKind::TypeMismatch,
            dummy_span(),
            "consider using explicit type".to_string(),
        );

        assert_eq!(note.severity, Severity::Note);
    }

    #[test]
    fn test_error_with_label() {
        let err = CompileError::new(
            ErrorKind::DuplicateName,
            dummy_span(),
            "duplicate signal".to_string(),
        )
        .with_label(dummy_span(), "first defined here".to_string());

        assert_eq!(err.labels.len(), 1);
        assert_eq!(err.labels[0].message, "first defined here");
    }

    #[test]
    fn test_error_with_note() {
        let err = CompileError::new(
            ErrorKind::TypeMismatch,
            dummy_span(),
            "type mismatch".to_string(),
        )
        .with_note("expected Scalar, got Vec2".to_string());

        assert_eq!(err.notes.len(), 1);
        assert_eq!(err.notes[0], "expected Scalar, got Vec2");
    }

    #[test]
    fn test_error_chaining() {
        let err = CompileError::new(
            ErrorKind::DuplicateName,
            dummy_span(),
            "duplicate signal 'foo'".to_string(),
        )
        .with_label(dummy_span(), "first defined here".to_string())
        .with_note("rename one of the signals".to_string())
        .with_label(dummy_span(), "conflicts with this".to_string());

        assert_eq!(err.labels.len(), 2);
        assert_eq!(err.notes.len(), 1);
    }

    #[test]
    fn test_error_kind_names() {
        assert_eq!(ErrorKind::Syntax.name(), "syntax error");
        assert_eq!(ErrorKind::TypeMismatch.name(), "type mismatch");
        assert_eq!(ErrorKind::MissingCapability.name(), "missing capability");
        assert_eq!(ErrorKind::Internal.name(), "internal compiler error");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Note < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn test_error_display() {
        let err = CompileError::new(
            ErrorKind::TypeMismatch,
            dummy_span(),
            "expected Scalar, got Vec2".to_string(),
        );

        let display = format!("{}", err);
        assert!(display.contains("error"));
        assert!(display.contains("type mismatch"));
        assert!(display.contains("expected Scalar, got Vec2"));
    }

    #[test]
    fn test_all_error_kinds_have_names() {
        // Ensure all ErrorKind variants have names
        let kinds = vec![
            ErrorKind::Syntax,
            ErrorKind::UndefinedName,
            ErrorKind::DuplicateName,
            ErrorKind::AmbiguousName,
            ErrorKind::TypeMismatch,
            ErrorKind::UnknownType,
            ErrorKind::RecursiveType,
            ErrorKind::ShapeMismatch,
            ErrorKind::DimensionMismatch,
            ErrorKind::UnitMismatch,
            ErrorKind::InvalidUnit,
            ErrorKind::UnknownKernel,
            ErrorKind::WrongArgCount,
            ErrorKind::InvalidKernelShape,
            ErrorKind::InvalidKernelUnit,
            ErrorKind::MissingCapability,
            ErrorKind::InvalidCapability,
            ErrorKind::EffectInPureContext,
            ErrorKind::CyclicDependency,
            ErrorKind::PathCollision,
            ErrorKind::CompilationFailed,
            ErrorKind::Internal,
        ];

        for kind in kinds {
            assert!(!kind.name().is_empty());
        }
    }

    #[test]
    fn test_formatter_basic() {
        let sources = test_sources();
        let span = Span::new(0, 8, 11, 1); // "foo"

        let error = CompileError::new(
            ErrorKind::UndefinedName,
            span,
            "undefined symbol 'foo'".to_string(),
        );

        let formatter = DiagnosticFormatter::new(&sources);
        let formatted = formatter.format(&error);

        assert!(formatted.contains("error"));
        assert!(formatted.contains("undefined name"));
        assert!(formatted.contains("undefined symbol 'foo'"));
        assert!(formatted.contains("test.cdsl:1:9"));
        assert!(formatted.contains("let x = foo"));
    }

    #[test]
    fn test_formatter_with_label() {
        let sources = test_sources();
        let primary_span = Span::new(0, 8, 11, 1); // "foo" on line 1
        let label_span = Span::new(0, 20, 23, 2); // "bar" on line 2

        let error = CompileError::new(
            ErrorKind::DuplicateName,
            primary_span,
            "duplicate symbol 'foo'".to_string(),
        )
        .with_label(label_span, "first defined here".to_string());

        let formatter = DiagnosticFormatter::new(&sources);
        let formatted = formatter.format(&error);

        assert!(formatted.contains("duplicate symbol 'foo'"));
        assert!(formatted.contains("first defined here"));
        assert!(formatted.contains("test.cdsl:2:")); // label location
    }

    #[test]
    fn test_formatter_with_note() {
        let sources = test_sources();
        let span = Span::new(0, 8, 11, 1);

        let error = CompileError::new(ErrorKind::TypeMismatch, span, "type mismatch".to_string())
            .with_note("expected Scalar, got Vec2".to_string());

        let formatter = DiagnosticFormatter::new(&sources);
        let formatted = formatter.format(&error);

        assert!(formatted.contains("type mismatch"));
        assert!(formatted.contains("help: expected Scalar, got Vec2"));
    }

    #[test]
    fn test_formatter_multiple_errors() {
        let sources = test_sources();

        let errors = vec![
            CompileError::new(
                ErrorKind::UndefinedName,
                Span::new(0, 8, 11, 1),
                "undefined 'foo'".to_string(),
            ),
            CompileError::new(
                ErrorKind::UndefinedName,
                Span::new(0, 20, 23, 2),
                "undefined 'bar'".to_string(),
            ),
        ];

        let formatter = DiagnosticFormatter::new(&sources);
        let formatted = formatter.format_all(&errors);

        assert!(formatted.contains("undefined 'foo'"));
        assert!(formatted.contains("undefined 'bar'"));
    }

    #[test]
    fn test_warning_formatting() {
        let sources = test_sources();
        let span = Span::new(0, 8, 11, 1);

        let warning =
            CompileError::warning(ErrorKind::TypeMismatch, span, "unused variable".to_string());

        let formatter = DiagnosticFormatter::new(&sources);
        let formatted = formatter.format(&warning);

        assert!(formatted.contains("warning"));
        assert!(!formatted.contains("error:"));
    }
}
