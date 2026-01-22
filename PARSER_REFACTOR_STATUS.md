# Parser Refactoring Status

## Current State

**Branch**: `epic/cdsl-compile-time-optimization`
**Status**: Parser compiles, review complete, ready for refactoring

### Completed
- ✅ Hand-written recursive descent parser (~2k LOC) - COMPILES
- ✅ Split into modular crates (lexer/parser/ast/resolve) - ALL COMPILE
- ✅ Facade crate updated with re-exports
- ✅ All 5 review agents completed analysis
- ✅ 10 beads issues created for findings

### Blocking Issues (P0)
1. `continuum-8nbt` - Fix broken integration tests
2. `continuum-xeim` - Add expression precedence tests
3. `continuum-6o3e` - Add declaration round-trip tests
4. `continuum-t734` - Add error handling tests
5. `continuum-eb82` - Fix DagSet duplication (One Truth)
6. **`continuum-5zht` - Split decl.rs (793 LOC) ← IN PROGRESS**
7. `continuum-s2vi` - Split expr.rs (606 LOC)
8. `continuum-k3zm` - Fix error hiding violations

### High Priority (P1)
9. `continuum-2b3h` - Add documentation with examples
10. `continuum-pdt1` - Extract DRY helpers

## God Module Split Plan

### decl.rs → decl/ (793 LOC → 5 files)

**Target structure**:
```
parser/decl/
  mod.rs          - dispatch + common helpers (~100 LOC)
  primitives.rs   - signal, field, operator, impulse, fracture, chronicle (~200 LOC)
  entities.rs     - entity, member (~100 LOC)
  time.rs         - stratum, era (~150 LOC)
  definitions.rs  - type, const, config (~150 LOC)
  world.rs        - world, warmup, policy (~100 LOC)
```

**Common helpers to extract** (in mod.rs):
```rust
// Parse attributes (repeated 11 times)
pub(super) fn parse_attributes(stream: &mut TokenStream) -> Result<Vec<Attribute>, ParseError>

// Parse node declaration (eliminates ~200 LOC duplication)
pub(super) fn parse_node_declaration(
    stream: &mut TokenStream,
    keyword: Token,
    role: RoleData,
) -> Result<Declaration, ParseError>

// Attribute helper
pub(super) fn parse_attribute(stream: &mut TokenStream) -> Result<Attribute, ParseError>
```

**Steps**:
1. Create `decl/mod.rs` with dispatch function + helpers
2. Create `decl/primitives.rs` - extract 6 node parsers + refactor with helper
3. Create `decl/entities.rs` - extract entity + member
4. Create `decl/time.rs` - extract stratum + era + related helpers
5. Create `decl/definitions.rs` - extract type + const + config
6. Create `decl/world.rs` - extract world + warmup + policy
7. Delete old `decl.rs`
8. Update `parser/mod.rs` to use `pub mod decl;`
9. Test compilation
10. Commit

### expr.rs → expr/ (606 LOC → 5 files)

**Target structure**:
```
parser/expr/
  mod.rs        - public API + dispatch (~50 LOC)
  pratt.rs      - Pratt parser, binary/unary ops (~150 LOC)
  atoms.rs      - Literals, identifiers, parenthesized (~100 LOC)
  special.rs    - if/let/filter (~100 LOC)
  spatial.rs    - nearest, within, pairs (~100 LOC)
  aggregate.rs  - agg.* operations (~100 LOC)
```

**Common helpers**:
```rust
// Token to identifier (repeated 3 times)
pub(super) fn token_to_identifier(token: &Token) -> Option<String>
```

**Steps**: Similar to decl.rs split

## Next Session Actions

1. **Continue decl.rs split** (2-3 hours remaining)
2. **Split expr.rs** (2-3 hours)
3. **Fix error hiding violations** (2 hours)
4. **Extract DRY helpers** (included in splits)
5. **Commit and push**

## Estimated Time to P0 Complete

- God module splits: 5 hours (40% done)
- Error hiding fixes: 2 hours
- Test infrastructure: 11 hours
- DagSet fix: 3 hours
- **Total remaining: ~21 hours**

## Testing Strategy

Once splits complete:
1. Ensure all crates still compile
2. Run existing integration tests (will fail, need fixing)
3. Begin test implementation per `continuum-xeim`, `continuum-6o3e`, `continuum-t734`

## Review Summary

See individual agent reports for details:
- **Documentation**: B grade (84%), needs context-free docs + examples
- **Tests**: F grade (0%), ZERO coverage on 2k LOC
- **Error Handling**: D grade, 6 violations of fail-loud principle
- **Code Hygiene**: D grade, 2 god modules + ~300 LOC duplication
- **Architecture**: B+ grade, 1 critical One Truth violation (DagSet)

**Overall**: NOT PRODUCTION READY until P0 issues resolved
