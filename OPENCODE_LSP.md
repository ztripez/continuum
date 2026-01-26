# CDSL Language Server for OpenCode

This project includes a Language Server Protocol (LSP) implementation for the Continuum DSL (CDSL) that works with OpenCode/Claude Code.

## Features

The CDSL LSP provides:
- ✅ **Real-time diagnostics** - Compilation errors and warnings as you type
- ✅ **Hover information** - Documentation, type info, and signatures
- ✅ **Go to definition** - Jump to signal, field, operator definitions
- ✅ **Code completion** - Autocomplete for symbols and keywords
- ✅ **Document symbols** - File outline (Ctrl+Shift+O)
- ✅ **Workspace symbols** - Cross-file search (Ctrl+T)
- ✅ **Semantic highlighting** - Syntax highlighting based on semantic meaning
- ✅ **Folding ranges** - Code folding for blocks
- ✅ **Inlay hints** - Type annotations
- ✅ **Document formatting** - Auto-format CDSL files

## Setup

### 1. Build the LSP Server

```bash
cargo build --package cdsl-lsp --release
```

This creates the LSP binary at `./target/release/cdsl-lsp`.

### 2. Configure OpenCode

The repository includes `opencode.json` which automatically configures the LSP:

```json
{
  "lsp": {
    "cdsl": {
      "enabled": true,
      "command": ["./target/release/cdsl-lsp"],
      "filetypes": [".cdsl"],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

### 3. Use the LSP

Simply open OpenCode in the project directory:

```bash
cd /path/to/continuum
opencode .
```

The LSP will automatically activate when you open `.cdsl` files.

## Testing

Try opening a CDSL file to test the LSP features:

```bash
opencode examples/terra/terra.cdsl
```

You should see:
- Syntax highlighting
- Hover over symbols for documentation
- Ctrl+Click to jump to definitions
- Real-time error diagnostics

## Troubleshooting

### LSP Not Starting

Check the OpenCode output panel (View > Output > CDSL Language Server) for errors.

Common issues:
- LSP binary not built: Run `cargo build --package cdsl-lsp --release`
- Permissions: Ensure `target/release/cdsl-lsp` is executable
- Path issues: `opencode.json` uses relative path, must run OpenCode from project root

### Verbose Logging

Enable debug logging by editing `opencode.json`:

```json
{
  "lsp": {
    "cdsl": {
      "env": {
        "RUST_LOG": "debug"
      }
    }
  }
}
```

### Manual LSP Start (Testing)

You can test the LSP server manually:

```bash
./target/release/cdsl-lsp
```

The LSP uses stdio transport, so it will wait for LSP protocol messages on stdin.

## Architecture

The CDSL LSP follows the **pure transport layer** architecture:

- **No custom data structures** - Uses engine types (`World`, `Node<I>`, `RoleId`) directly
- **Protocol conversion only** - Converts engine types to LSP protocol types
- **Real-time compilation** - Compiles CDSL on every change for diagnostics
- **Stores compiled worlds** - Caches `World` per document for fast lookups

See `crates/cdsl-lsp/AGENTS.md` for architecture details.

## VS Code Extension

For VS Code (not OpenCode), use the extension in `tools/vscode-cdsl/`:

```bash
cd tools/vscode-cdsl
npm install
npm run compile
code --install-extension .
```

The VS Code extension uses the same LSP server but packages it differently.

## Related Files

- `opencode.json` - OpenCode LSP configuration
- `crates/cdsl-lsp/` - LSP server implementation
- `crates/cdsl-lsp/AGENTS.md` - Architecture rules
- `tools/vscode-cdsl/` - VS Code extension (alternative to OpenCode)
