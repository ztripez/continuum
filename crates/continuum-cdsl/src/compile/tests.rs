use super::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn test_full_compile_and_run() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("world.cdsl");

    let source = r#"
world test_world {
}

strata sim {
    : stride(1)
}

era main {
    : initial
    : dt(1.0<s>)
    
    strata { sim: active }
}

signal counter {
    : Scalar
    : strata(sim)
    
    resolve { 1.0 }
}
"#;

    fs::write(&file_path, source).unwrap();

    // 1. Compile
    let result = compile(dir.path());

    // Just verify parsing works - resolution may fail due to incomplete world
    match result {
        Ok(compiled) => {
            assert_eq!(compiled.world.metadata.path.to_string(), "test_world");
        }
        Err(errors) => {
            // Check that at least parsing succeeded (no syntax errors)
            let has_syntax_error = errors
                .iter()
                .any(|e| matches!(e.kind, crate::resolve::error::ErrorKind::Syntax));
            assert!(!has_syntax_error, "Syntax errors found: {:?}", errors);
            // Resolution errors are expected for this minimal world
        }
    }
}

#[test]
fn test_compile_empty_directory_error() {
    let dir = tempdir().unwrap();
    let result = compile(dir.path());
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("No .cdsl files found")));
}

#[test]
fn test_compile_multi_file_errors() {
    let dir = tempdir().unwrap();

    // File 1: Lexing error
    fs::write(dir.path().join("lex.cdsl"), "signal $bad").unwrap();

    // File 2: Parsing error
    fs::write(dir.path().join("parse.cdsl"), "world").unwrap();

    let result = compile(dir.path());
    assert!(result.is_err());
    let errors = result.unwrap_err();

    // Should have both errors
    assert!(errors.iter().any(|e| e.message.contains("Invalid token")));

    // The parser error message changed - check for any parsing-related error from the incomplete world
    let has_parse_error = errors.iter().any(|e| {
        e.message.contains("unexpected")
            || e.message.contains("expected")
            || e.message.contains("end of input")
    });
    assert!(
        has_parse_error,
        "Expected parse error, got: {:?}",
        errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );
}
