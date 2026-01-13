# Continuum DSL for Visual Studio Code

This extension provides language support for the Continuum DSL (`.cdsl`).

## Features

- **Syntax Highlighting**: Declarative simulation syntax highlighting.
- **Language Server (LSP)**: Real-time diagnostics, symbol indexing, and navigation.
- **Debugger (DAP)**: Interactive simulation debugging, breakpoints, and state inspection.

## Installation

### Prerequisites

You must build the Continuum toolchain to use the LSP and DAP features:

```bash
cargo build -p cdsl-lsp -p cdsl-dap
```

The extension will automatically look for the binaries in your `target/debug` directory.

## Configuration

If the binaries are not found automatically, you can configure the paths manually in your VS Code settings:

- `cdsl.server.path`: Path to `cdsl-lsp`
- `cdsl.dap.path`: Path to `cdsl-dap`
