# LSP Architecture Rules

## Fundamental Constraint: Pure Transport Layer

The CDSL Language Server is a **pure transport layer**. It exists ONLY to:
1. Receive LSP protocol requests
2. Call engine primitives
3. Convert engine responses to LSP protocol types
4. Send LSP protocol responses

## Forbidden Patterns

### ❌ NO Custom Data Structures Over Engine Types

**Examples of violations:**
- Custom `SymbolKind` enum duplicating engine's `RoleId`
- Custom `SymbolInfo` struct duplicating engine's `Node<I>`
- Custom `SymbolIndex` building shadow structure over `World`
- Custom `TypeInfo` duplicating engine's `Type`

**Why forbidden:**
- Violates One Truth (OT) principle
- Creates maintenance burden (must sync with engine changes)
- Adds unnecessary abstraction layer
- Increases complexity and bug surface

### ❌ NO Hardcoded Enums or Role Lists

**Wrong:**
```rust
enum SymbolKind {
    Signal, Field, Operator, // ...
}
```

**Right:**
```rust
// Use engine's RoleId directly
use continuum_cdsl::ast::RoleId;

fn display_name(role: RoleId) -> &'static str {
    role.spec().name  // "signal", "field", etc. from engine
}
```

## Allowed Patterns

### ✅ Lookup Functions Operating on Engine Types

```rust
// Find node at position - returns engine type
fn find_node_at_position<'a>(
    world: &'a World, 
    offset: usize
) -> Option<&'a Node<()>> {
    world.globals
        .values()
        .find(|node| node.span.contains(offset))
}
```

### ✅ Protocol Conversion Traits

```rust
// Convert engine type → LSP protocol type
trait ToLspSymbolKind {
    fn to_lsp_symbol_kind(&self) -> lsp_types::SymbolKind;
}

impl ToLspSymbolKind for RoleId {
    fn to_lsp_symbol_kind(&self) -> lsp_types::SymbolKind {
        match self {
            RoleId::Signal => lsp_types::SymbolKind::VARIABLE,
            RoleId::Field => lsp_types::SymbolKind::FIELD,
            RoleId::Operator => lsp_types::SymbolKind::FUNCTION,
            // ...
        }
    }
}
```

### ✅ Formatting/Display Utilities

```rust
// Format engine type for hover display
fn format_hover(node: &Node<impl Index>) -> String {
    let role_name = node.role.id().spec().name;
    let mut parts = vec![format!("{} {}", role_name, node.path)];
    
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

## LSP Handler Architecture

Each LSP handler should:
1. Parse LSP request
2. Call engine directly (compile, lookup, etc.)
3. Operate on engine types (`World`, `Node<I>`, `RoleId`)
4. Convert to LSP protocol types
5. Return LSP response

**Example: Hover**
```rust
fn handle_hover(params: HoverParams) -> Option<Hover> {
    // 1. Get position from LSP request
    let offset = position_to_offset(params.position)?;
    
    // 2. Call engine (compile returns World)
    let world = compile_document()?;
    
    // 3. Find in engine types
    let node = find_node_at_position(&world, offset)?;
    
    // 4. Format using engine data
    let hover_text = format_hover(node);
    
    // 5. Return LSP response
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: hover_text,
        }),
        range: Some(span_to_lsp_range(&node.span)),
    })
}
```

## Engine Types to Use

| Need | Engine Type | Location |
|------|-------------|----------|
| Compiled world | `World` | `continuum_cdsl::ast::World` |
| Symbol info | `Node<()>`, `Node<EntityId>` | `continuum_cdsl::ast::Node` |
| Symbol kind | `RoleId` | `continuum_cdsl::ast::RoleId` |
| Type info | `Type` | `continuum_cdsl::foundation::Type` |
| Source location | `Span` | `continuum_cdsl::foundation::Span` |
| Path | `Path` | `continuum_cdsl::foundation::Path` |
| Entity index | `EntityId` | `continuum_cdsl::ast::EntityId` |
| Stratum | `Stratum` | `continuum_cdsl::ast::Stratum` |
| Era | `Era` | `continuum_cdsl::ast::Era` |

## Symbol Lookup Pattern

```rust
// Iterate engine collections - no shadow index
fn find_symbol_at_offset(world: &World, offset: usize) -> Option<SymbolLookup> {
    // Check globals (signals, fields, operators)
    for (path, node) in &world.globals {
        if node.span.contains(offset) {
            return Some(SymbolLookup::Global(node));
        }
    }
    
    // Check members (per-entity)
    for (path, member) in &world.members {
        if member.span.contains(offset) {
            return Some(SymbolLookup::Member(member));
        }
    }
    
    // Check entities
    for (path, entity) in &world.entities {
        if entity.span.contains(offset) {
            return Some(SymbolLookup::Entity(entity));
        }
    }
    
    // Check strata
    for (path, stratum) in &world.strata {
        if stratum.span.contains(offset) {
            return Some(SymbolLookup::Stratum(stratum));
        }
    }
    
    // Check eras
    for (path, era) in &world.eras {
        if era.span.contains(offset) {
            return Some(SymbolLookup::Era(era));
        }
    }
    
    None
}

// Enum holding references to engine types - NOT copies
enum SymbolLookup<'a> {
    Global(&'a Node<()>),
    Member(&'a Node<EntityId>),
    Entity(&'a Entity),
    Stratum(&'a Stratum),
    Era(&'a Era),
}
```

## Reference Tracking

For "find all references", there are two options:

### Option 1: Engine Tracks References (Preferred)
- Resolver records reference spans during compilation
- `World` contains `references: Vec<Reference>` where `Reference { span, target: Path }`
- LSP queries `world.references` directly

### Option 2: LSP Computes References Inline
- Walk `World.declarations` AST
- Extract references from expressions
- Return spans
- **Only acceptable if engine doesn't provide this**

## Completion Strategy

```rust
fn completions(world: &World, prefix: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    
    // Add global symbols
    for (path, node) in &world.globals {
        if path.to_string().starts_with(prefix) {
            items.push(CompletionItem {
                label: path.to_string(),
                kind: Some(node.role.id().to_lsp_symbol_kind()),
                detail: node.output.as_ref().map(|t| format!("{:?}", t)),
                documentation: node.doc.clone().map(Documentation::String),
                ..Default::default()
            });
        }
    }
    
    // Same for members, entities, strata, eras...
    
    items
}
```

## Rule Summary

1. **Use engine types** - `World`, `Node<I>`, `RoleId`, `Type`, `Span`, `Path`
2. **No shadow structures** - Iterate engine collections directly
3. **Protocol conversion only** - Traits/functions to convert engine → LSP types
4. **Derive display names** - Use `RoleId::spec().name`, not hardcoded strings
5. **Return engine references** - `&Node<I>`, not copies or custom structs

When in doubt: **If the engine provides it, use it. If the engine doesn't provide it, ask why not.**
