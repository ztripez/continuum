# Continuum DSL

Parser, AST, and validation for the Continuum Domain-Specific Language (`.cdsl`).

This crate is responsible for turning raw source text into a structured, validated Abstract Syntax Tree (AST). It handles parsing, syntax error reporting, and initial structure validation.

## Components

- **`parser`**: Chumsky-based parser for `.cdsl` files.
- **`ast`**: Typed Abstract Syntax Tree definitions (`SignalDef`, `EntityDef`, `MemberDef`, `OperatorDef`, `Expr`, etc.) with source spans.
- **`validate`**: Initial semantic validation pass (naming collisions, phase correctness).
- **`loader`**: Utilities for discovering and loading DSL files from a world directory.

## Usage

```rust
use continuum_dsl::loader::load_world;

let (compilation_unit, errors) = load_world("examples/terra")?;
```
