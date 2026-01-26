# DAP Architecture Rules

## Fundamental Constraint: Pure Transport Layer

The CDSL Debug Adapter is a **pure transport layer**. It exists ONLY to:
1. Receive DAP (Debug Adapter Protocol) requests
2. Call engine primitives
3. Convert engine responses to DAP protocol types
4. Send DAP protocol responses

## Forbidden Patterns

### ❌ NO Custom Data Structures Over Engine Types

**Examples of violations:**
- Custom `SignalInfo` struct duplicating engine's `Node<I>`
- Custom `StackFrame` duplicating engine's execution context
- Custom `BreakpointInfo` duplicating engine's debug metadata
- Shadow indexes over engine structures

**Why forbidden:**
- Violates One Truth (OT) principle
- Creates maintenance burden (must sync with engine changes)
- Adds unnecessary abstraction layer
- Increases complexity and bug surface

### ❌ NO Hardcoded Enums or Lists

**Wrong:**
```rust
enum PrimitiveKind {
    Signal, Field, Operator, // ...
}
```

**Right:**
```rust
// Use engine's RoleId directly
use continuum_cdsl::ast::RoleId;

fn debug_name(role: RoleId) -> &'static str {
    role.spec().name  // From engine
}
```

## Allowed Patterns

### ✅ Debug State Queries Using Engine Types

```rust
// Query signal values - returns engine types
fn get_signal_value(
    runtime: &Runtime,
    path: &Path,
) -> Result<f64, Error> {
    runtime.get_signal_value(path)  // Engine provides this
}
```

### ✅ Protocol Conversion Traits

```rust
// Convert engine execution context → DAP stack frame
trait ToDapStackFrame {
    fn to_dap_stack_frame(&self) -> dap_types::StackFrame;
}

impl ToDapStackFrame for ExecutionContext {
    fn to_dap_stack_frame(&self) -> dap_types::StackFrame {
        dap_types::StackFrame {
            id: self.frame_id,
            name: self.node_path.to_string(),
            source: self.file.as_ref().map(|f| dap_types::Source {
                path: Some(f.to_string_lossy().to_string()),
                ..Default::default()
            }),
            line: self.span.start_line as i64,
            column: self.span.start_col as i64,
            ..Default::default()
        }
    }
}
```

### ✅ Debug Utilities

```rust
// Format engine value for DAP display
fn format_variable(node: &Node<impl Index>, value: f64) -> dap_types::Variable {
    let role_name = node.role.id().spec().name;
    dap_types::Variable {
        name: node.path.to_string(),
        value: format!("{} ({})", value, node.output.as_ref()
            .map(|t| format!("{:?}", t))
            .unwrap_or_else(|| "unknown".to_string())),
        type_: Some(role_name.to_string()),
        ..Default::default()
    }
}
```

## DAP Handler Architecture

Each DAP handler should:
1. Parse DAP request
2. Call engine debug primitives
3. Operate on engine types (`Runtime`, `World`, `Node<I>`)
4. Convert to DAP protocol types
5. Return DAP response

**Example: Scopes**
```rust
fn handle_scopes(frame_id: i64) -> Result<Vec<Scope>, Error> {
    // 1. Get frame from engine
    let frame = runtime.get_frame(frame_id)?;
    
    // 2. Query engine for available scopes
    let signals = runtime.get_visible_signals(&frame)?;
    let locals = runtime.get_local_bindings(&frame)?;
    
    // 3. Convert to DAP scopes
    Ok(vec![
        Scope {
            name: "Signals".to_string(),
            variables_reference: encode_scope_ref(frame_id, ScopeKind::Signals),
            ..Default::default()
        },
        Scope {
            name: "Locals".to_string(),
            variables_reference: encode_scope_ref(frame_id, ScopeKind::Locals),
            ..Default::default()
        },
    ])
}
```

## Engine Types to Use

| Need | Engine Type | Location |
|------|-------------|----------|
| Runtime state | `Runtime` | `continuum_runtime::Runtime` |
| Compiled world | `World` | `continuum_cdsl::ast::World` |
| Node info | `Node<()>`, `Node<EntityId>` | `continuum_cdsl::ast::Node` |
| Execution context | `ExecutionContext` | Engine runtime |
| Signal values | Via `Runtime::get_signal_value()` | Engine API |
| Breakpoint support | Engine debug hooks | Engine runtime |

## Breakpoint Strategy

```rust
// Set breakpoint - call engine directly
fn set_breakpoint(
    runtime: &mut Runtime,
    path: &Path,
    line: u32,
) -> Result<Breakpoint, Error> {
    // 1. Find node at line in World
    let world = runtime.world();
    let node = world.globals.values()
        .find(|n| n.span.start_line == line && n.file.as_deref() == Some(path))
        .ok_or(Error::NodeNotFound)?;
    
    // 2. Ask engine to set breakpoint
    let bp_id = runtime.set_breakpoint(node.path.clone())?;
    
    // 3. Return DAP breakpoint
    Ok(Breakpoint {
        id: Some(bp_id as i64),
        verified: true,
        line: Some(line as i64),
        source: Some(Source {
            path: Some(path.to_string_lossy().to_string()),
            ..Default::default()
        }),
        ..Default::default()
    })
}
```

## Variable Inspection Strategy

```rust
// Get variables - query engine directly
fn get_variables(
    runtime: &Runtime,
    variables_ref: i64,
) -> Result<Vec<Variable>, Error> {
    let (frame_id, scope_kind) = decode_scope_ref(variables_ref)?;
    let frame = runtime.get_frame(frame_id)?;
    
    match scope_kind {
        ScopeKind::Signals => {
            // Get signal nodes from World
            let world = runtime.world();
            let mut vars = Vec::new();
            
            for (path, node) in &world.globals {
                if matches!(node.role, RoleData::Signal) {
                    // Query current value from runtime
                    if let Ok(value) = runtime.get_signal_value(path) {
                        vars.push(format_variable(node, value));
                    }
                }
            }
            
            Ok(vars)
        }
        ScopeKind::Locals => {
            // Get local bindings from engine
            runtime.get_local_bindings(&frame)?
                .into_iter()
                .map(|(name, value)| Variable {
                    name,
                    value: format!("{:?}", value),
                    type_: Some("local".to_string()),
                    ..Default::default()
                })
                .collect()
        }
    }
}
```

## Rule Summary

1. **Use engine types** - `Runtime`, `World`, `Node<I>`, `ExecutionContext`
2. **No shadow structures** - Query engine state directly
3. **Protocol conversion only** - Traits/functions to convert engine → DAP types
4. **Engine provides debug info** - Breakpoints, step, frame inspection
5. **Return engine references** - Don't copy engine data into custom structs

When in doubt: **If the engine provides it, use it. If the engine doesn't provide it, ask why not.**
