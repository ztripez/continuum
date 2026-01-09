# Continuum Tools

CLI tools for interacting with the Continuum engine.

This crate provides the binary entry points for running simulations, analyzing outputs, and linting DSL code.

## Binaries

### `world-run`
Executes a simulation world.
```bash
cargo run --bin world-run -- <WORLD_DIR>
```

### `dsl-lint`
Parses and validates DSL files, reporting syntax and semantic errors.
```bash
cargo run --bin dsl-lint -- <FILE_OR_DIR>
```

### `world-load`
Test utility to verify that a full world directory can be loaded and compiled to IR without error.
```bash
cargo run --bin world-load -- <WORLD_DIR>
```
