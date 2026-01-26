# LSP/DAP Migration Guide: continuum_compiler → continuum_cdsl

This document maps the old `continuum_compiler` API to the new `continuum_cdsl` API for migrating cdsl-lsp, cdsl-dap, and tests crates.

## Core Principle

**The LSP/DAP must be pure transport** - they should NOT define their own symbol types or indexing structures. Instead, they should surface what the engine provides:
- Use `RoleId` from engine (NOT custom `SymbolKind`)
- Use `Node<I>` and `World` from engine for symbol information
- Use `compile()` from `continuum_cdsl` for compilation

## Import Changes

### Old Imports (continuum_compiler)
```rust
use continuum_compiler::dsl::ast::{
    CompilationUnit, ConfigBlock, ConstBlock, Expr, Item, Literal, 
    OperatorBody, Path, Spanned, SpannedExprVisitor,
};
use continuum_compiler::ir::{
    CompiledNode, CompiledWorld, NodeKind, PrimitiveParamKind, 
    PrimitiveParamSpec, ValueType, ValueTypeParamValue,
};
use continuum_compiler::compile;
```

### New Imports (continuum_cdsl)
```rust
// AST types
use continuum_cdsl::ast::{
    Declaration, Expr, Node, World, RoleId, RoleData,
    Entity, Stratum, Era,
};

// Foundation types
use continuum_cdsl::foundation::{
    Path, Span, Type, KernelType,
};

// Compilation
use continuum_cdsl::{compile_with_sources, CompileResultWithSources};

// Parser (if needed for incremental parsing)
use continuum_cdsl::{parse_declarations, parse_expr, ParseError};
```

## Type Mappings

| Old Type (continuum_compiler) | New Type (continuum_cdsl) | Notes |
|-------------------------------|---------------------------|-------|
| `CompilationUnit` | `Vec<Declaration>` | Declarations are now a flat list |
| `CompiledWorld` | `World` | Contains `globals`, `members`, `entities`, `strata`, `eras` |
| `CompiledNode` | `Node<()>` or `Node<EntityId>` | Unified node structure, generic over index |
| `NodeKind` | `RoleId` | Enum: Signal, Field, Operator, Impulse, Fracture, Chronicle |
| `Item` | `Declaration` | Top-level declarations |
| `ValueType` | `Type` | Type system types |

## Symbol Indexing Changes

### Old Approach (WRONG - violates "transport only" rule)
```rust
// cdsl-lsp/src/symbols.rs defined its own SymbolKind:
pub enum SymbolKind {
    Signal, Field, Operator, Function, Type, Strata, Era,
    Impulse, Fracture, Chronicle, Entity, Member, World, Const, Config,
}
```

### New Approach (CORRECT - use engine types)
```rust
// Use RoleId from engine
use continuum_cdsl::ast::RoleId;

// RoleId already defines:
// pub enum RoleId { Signal, Field, Operator, Impulse, Fracture, Chronicle }

// For display names, use the engine's method:
let display_name = role_id.spec().name;
```

## World Structure

### Old Structure (continuum_compiler)
```rust
struct CompiledWorld {
    // Flat list of nodes
    nodes: Vec<CompiledNode>,
}
```

### New Structure (continuum_cdsl)
```rust
pub struct World {
    pub metadata: WorldDecl,
    pub initial_era: Option<EraId>,
    
    // Nodes are organized by scope
    pub globals: IndexMap<Path, Node<()>>,      // Global signals, fields, operators
    pub members: IndexMap<Path, Node<EntityId>>, // Per-entity members
    
    // Structural declarations
    pub entities: IndexMap<Path, Entity>,
    pub strata: IndexMap<Path, Stratum>,
    pub eras: IndexMap<Path, Era>,
    
    // Original declarations (for diagnostics)
    pub declarations: Vec<Declaration>,
}
```

## Node<I> Structure

The new `Node<I>` is generic over index type `I`:
- `Node<()>` - Global primitive (signal, field, operator)
- `Node<EntityId>` - Per-entity member

### Key Fields for LSP
```rust
pub struct Node<I: Index> {
    // Identity
    pub path: Path,              // Hierarchical path (e.g., "terra.temperature")
    pub span: Span,              // Source location
    pub file: Option<PathBuf>,   // Source file
    
    // Documentation
    pub doc: Option<String>,     // /// doc comments
    pub title: Option<String>,   // : title("...") attribute
    pub symbol: Option<String>,  // : symbol("...") attribute
    
    // Role
    pub role: RoleData,          // Signal, Field, Operator, etc.
    
    // Type information
    pub output: Option<Type>,    // What this node produces
    pub inputs: Vec<(String, Type)>, // What this node receives
    
    // Execution context
    pub stratum: Option<StratumId>, // Execution lane
    pub initial: Option<f64>,    // Initial value (signals only)
    
    // Indexing (global vs per-entity)
    pub index: I,                // () or EntityId
}
```

## Compilation API

### Old API
```rust
let result = continuum_compiler::compile(&sources);
// Returns Result<CompiledWorld, Vec<Error>>
```

### New API
```rust
use continuum_cdsl::{compile_with_sources, CompileResultWithSources};
use std::path::PathBuf;
use indexmap::IndexMap;

let sources: IndexMap<PathBuf, String> = /* ... */;
let result: CompileResultWithSources = compile_with_sources(&sources);

// Result structure:
pub struct CompileResultWithSources {
    pub world: Option<World>,
    pub sources: IndexMap<PathBuf, SourceFile>,
    pub errors: Vec<Error>,
}
```

## Symbol Extraction Example

### Old Approach (custom SymbolIndex)
```rust
// Build custom index from CompiledWorld
let mut index = SymbolIndex::new();
for node in world.nodes {
    let kind = match node.kind {
        NodeKind::Signal => SymbolKind::Signal,
        NodeKind::Field => SymbolKind::Field,
        // ... manual mapping
    };
    index.add(kind, node.path, node.span, /* ... */);
}
```

### New Approach (iterate engine types)
```rust
use continuum_cdsl::ast::World;

// Iterate globals (signals, fields, operators)
for (path, node) in &world.globals {
    let role_id = node.role.id(); // RoleId from engine
    let display_name = role_id.spec().name; // "signal", "field", etc.
    
    // All metadata is on the node
    let info = SymbolInfo {
        path: node.path.to_string(),
        role: role_id,
        doc: node.doc.clone(),
        title: node.title.clone(),
        symbol: node.symbol.clone(),
        type_info: node.output.as_ref().map(|t| format!("{:?}", t)),
        span: node.span,
    };
    
    // Index by span for hover/goto
    symbols_by_position.insert(node.span, info);
}

// Iterate members (per-entity)
for (path, member) in &world.members {
    let entity_path = &member.index.0; // EntityId contains entity path
    // Same as above, but mark as member
}

// Iterate entities
for (path, entity) in &world.entities {
    // entity.path, entity.doc, entity.span
}

// Iterate strata
for (path, stratum) in &world.strata {
    // stratum.path, stratum.cadence, etc.
}

// Iterate eras
for (path, era) in &world.eras {
    // era.path, era.phase_schedule, etc.
}
```

## RoleId vs Custom SymbolKind

### Why RoleId is Superior

1. **Single Source of Truth** - RoleId is defined in the engine, used throughout the compiler
2. **Capabilities Built-in** - `role_id.spec()` provides phase and capability information
3. **Display Names** - `role_id.spec().name` gives canonical display name
4. **No Duplication** - LSP doesn't maintain parallel enum

### Mapping Non-Role Symbols

For structural declarations that aren't primitives:

```rust
pub enum SymbolCategory {
    Primitive(RoleId),  // Signal, Field, Operator, Impulse, Fracture, Chronicle
    Entity,             // Entity declaration
    Stratum,            // Stratum declaration
    Era,                // Era declaration
    Config,             // Config entry
    Const,              // Const entry
    World,              // World declaration
}
```

This keeps the role system clean while allowing LSP to represent all symbols.

## Hover Info Generation

### Old
```rust
fn generate_hover(node: &CompiledNode) -> String {
    match node.kind {
        NodeKind::Signal => format!("signal {}\n{:?}", node.path, node.ty),
        // ... manual formatting
    }
}
```

### New
```rust
fn generate_hover(node: &Node<I>) -> String {
    let role_name = node.role.id().spec().name;
    let mut parts = vec![
        format!("{} {}", role_name, node.path),
    ];
    
    if let Some(title) = &node.title {
        parts.push(format!("  {}", title));
    }
    
    if let Some(output) = &node.output {
        parts.push(format!("  : {:?}", output));
    }
    
    if let Some(doc) = &node.doc {
        parts.push(format!("\n{}", doc));
    }
    
    parts.join("\n")
}
```

## Action Items for Migration

1. **Remove custom SymbolKind** from `cdsl-lsp/src/symbols.rs`
2. **Use RoleId** from `continuum_cdsl::ast::RoleId`
3. **Rewrite SymbolIndex::new()** to iterate `World.globals`, `World.members`, etc.
4. **Update hover/completion** to use `node.doc`, `node.title`, `node.symbol`
5. **Update imports** throughout LSP and DAP crates
6. **Test with real CDSL files** to ensure hover/goto/completion work

## Benefits of New Architecture

1. **Type Safety** - `Node<()>` vs `Node<EntityId>` is compile-time enforced
2. **No Duplication** - LSP uses engine types directly
3. **Automatic Updates** - When engine adds capabilities, LSP gets them free
4. **Simpler Code** - No manual mapping between compiler IR and LSP types
5. **True Transport** - LSP becomes a thin layer over engine primitives
