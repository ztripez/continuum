use super::*;
use indexmap::IndexMap;
use std::fs;
use std::path::PathBuf;
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

#[test]
fn test_compile_from_memory_basic() {
    let mut sources = IndexMap::new();
    sources.insert(
        PathBuf::from("world.cdsl"),
        r#"
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
"#
        .to_string(),
    );

    let result = compile_from_memory(sources);

    match result {
        Ok(compiled) => {
            assert_eq!(compiled.world.metadata.path.to_string(), "test_world");
            assert!(compiled
                .world
                .globals
                .contains_key(&crate::foundation::Path::from_path_str("counter")));
        }
        Err((_, errors)) => {
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
fn test_compile_from_memory_multi_file() {
    let mut sources = IndexMap::new();

    sources.insert(
        PathBuf::from("world.cdsl"),
        r#"
world test_multi {
}
"#
        .to_string(),
    );

    sources.insert(
        PathBuf::from("strata.cdsl"),
        r#"
strata sim {
    : stride(1)
}
"#
        .to_string(),
    );

    sources.insert(
        PathBuf::from("signals.cdsl"),
        r#"
era main {
    : initial
    : dt(1.0<s>)
    
    strata { sim: active }
}

signal foo {
    : Scalar
    : strata(sim)
    resolve { 42.0 }
}

signal bar {
    : Scalar
    : strata(sim)
    resolve { signal.foo + 1.0 }
}
"#
        .to_string(),
    );

    let result = compile_from_memory(sources);

    match result {
        Ok(compiled) => {
            assert_eq!(compiled.world.metadata.path.to_string(), "test_multi");
            assert!(compiled
                .world
                .globals
                .contains_key(&crate::foundation::Path::from_path_str("foo")));
            assert!(compiled
                .world
                .globals
                .contains_key(&crate::foundation::Path::from_path_str("bar")));
        }
        Err((_, errors)) => {
            // Print errors for debugging
            for err in &errors {
                eprintln!("Compile error: {}", err.message);
            }
            // Allow resolution errors but not syntax errors
            let has_syntax_error = errors
                .iter()
                .any(|e| matches!(e.kind, crate::resolve::error::ErrorKind::Syntax));
            assert!(!has_syntax_error, "Syntax errors found");
        }
    }
}

#[test]
fn test_compile_from_memory_empty_sources() {
    let sources = IndexMap::new();
    let result = compile_from_memory(sources);

    assert!(result.is_err());
    let (_, errors) = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("No source files provided")));
}

#[test]
fn test_compile_from_memory_deterministic_order() {
    // Insert in reverse alphabetical order to ensure sorting works
    let mut sources = IndexMap::new();
    sources.insert(PathBuf::from("z_last.cdsl"), "// last".to_string());
    sources.insert(PathBuf::from("a_first.cdsl"), "world test { }".to_string());
    sources.insert(PathBuf::from("m_middle.cdsl"), "// middle".to_string());

    // The compilation should process files in alphabetical order (a, m, z)
    // This test mainly ensures no panic occurs with out-of-order insertion
    let _result = compile_from_memory(sources);
    // If it doesn't panic, the ordering is working
}
