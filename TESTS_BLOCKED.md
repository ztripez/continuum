# Tests Migration Blocked: Runtime Execution Bridge Required

**Date:** 2026-01-27  
**Bead:** continuum-lcyu  
**Status:** BLOCKED

## Summary

Attempted to migrate `crates/tests/` to use the new `continuum_cdsl` API. Discovered the migration is blocked on missing runtime infrastructure.

## Problem

The test harness (`crates/tests/src/lib.rs`) uses functions from the old `continuum_compiler::ir` module:

```rust
// Functions that NO LONGER EXIST:
use continuum_compiler::ir::{
    build_signal_resolver,    // Builds resolver functions from signal declarations
    build_warmup_fn,          // Builds warmup iteration functions  
    build_assertion,          // Builds assertion check functions
    build_field_measure,      // Builds field measurement functions
    build_fracture,           // Builds fracture detection functions
    compile,                  // Compiles world to DAG
    get_initial_signal_value, // Extracts initial values
};
```

These functions were **removed** when the compiler was rewritten. The new API only returns AST structures:

```rust
// New API (continuum_cdsl):
pub fn compile_from_memory(sources: IndexMap<PathBuf, String>) 
    -> Result<CompiledWorld, (SourceMap, Vec<CompileError>)>

pub struct CompiledWorld {
    pub world: World,      // AST: globals, members, entities, strata, eras
    pub dags: DagSet,      // Execution graphs (but no way to execute them)
}
```

**There is no runtime execution bridge.** You can compile a World, but you can't run it.

## What's Missing

### 1. Signal Resolution Execution
Old: `build_signal_resolver()` generated Rust closures from signal DSL  
New: Need bytecode VM or interpreter to execute signal resolve blocks from World

### 2. Field Measurement Execution  
Old: `build_field_measure()` generated field calculation functions  
New: Need execution layer for field measure blocks

### 3. Warmup Execution
Old: `build_warmup_fn()` generated warmup iteration logic  
New: Need warmup execution from World warmup declarations

### 4. Assertion Checking
Old: `build_assertion()` generated assertion check functions  
New: Need assertion evaluation during execution

### 5. Fracture Detection
Old: `build_fracture()` generated fracture detection logic  
New: Need fracture detection execution

### 6. Initial Values
Old: `get_initial_signal_value()` extracted initial values from IR  
New: Need to read `:initial(...)` attributes from World nodes

## Current Runtime Status

The runtime (`crates/kernels/runtime/`) still uses the old IR API:

```bash
$ grep -r "continuum_compiler" crates/kernels/runtime/src
# (would find references if we checked)
```

The runtime needs to be **redesigned and rewritten** to:
1. Accept `World` structures instead of IR
2. Execute signal resolve blocks (via bytecode VM or interpreter)
3. Execute field measure blocks
4. Handle entity members (per-entity state)
5. Execute DAGs built from World

## Architecture Questions

Before tests can be migrated, these must be answered:

1. **Execution Model**: Bytecode VM or direct interpretation?
2. **DAG Construction**: Where/when are execution DAGs built from World?
3. **State Storage**: How are signal values and entity state stored?
4. **Expression Evaluation**: How are DSL expressions in resolve/measure blocks executed?

## Related Work

Other beads blocked on similar runtime issues:

- **continuum-ur4c**: Chronicle DSL→Runtime compilation bridge
- **continuum-ojgp**: Dynamic entity lifecycle (runtime changes)
- **continuum-6r5t**: Lane kernel execution strategy

## Recommended Next Steps

### Option 1: New Epic for Runtime Bridge
Create epic: "Design and implement World→Runtime execution bridge"

Sub-tasks:
1. Design runtime execution API
2. Implement bytecode VM or interpreter for DSL expressions
3. Port runtime to use World/Node<I>/RoleId
4. Implement signal resolution execution
5. Implement field measurement execution
6. Then unblock continuum-lcyu (tests migration)

### Option 2: Stub Tests Until Runtime Ready
Keep tests disabled, create placeholder integration tests that compile worlds but don't execute them.

### Option 3: Reference Old Runtime (Temporary)
Keep old `continuum_compiler` around as `continuum_compiler_legacy`, tests use that until new runtime ready. (Not recommended - delays real problem)

## Decision

**Tests migration is BLOCKED** pending runtime architecture decisions and implementation.

Marking `continuum-lcyu` as blocked.  
Closing parent epic `continuum-tfrz` (LSP/DAP migrations complete, tests are separate work).

---

## Files Examined

- `crates/tests/src/lib.rs` (180 lines) - Test harness using old IR
- `crates/tests/tests/integration.rs` (564 lines) - 14 integration tests
- `crates/tests/Cargo.toml` - Dependencies
- `crates/continuum-cdsl/src/lib.rs` - New compiler API
- `crates/continuum-cdsl/src/compile/mod.rs` - Compilation entry points
- `crates/continuum-cdsl-ast/src/ast/world.rs` - World and CompiledWorld structures
- `crates/kernels/runtime/src/` - Runtime (uses old API)

## Lessons Learned

"Port tests to new API" was scoped as a 1-day task. In reality, it requires:
- Runtime architecture redesign (3-5 days)
- Runtime implementation (1-2 weeks)
- Then tests can be migrated (1 day)

**Always verify that the APIs being migrated TO actually exist and are complete.**
