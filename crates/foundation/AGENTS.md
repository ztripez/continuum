# Foundation Crate

Core foundational utilities shared across all Continuum crates.

## Purpose

This crate provides primitives that must be consistent across the entire system:
- Entity identifiers (SignalId, EraId, etc.)
- Stable hashing for deterministic ID generation
- Other foundational utilities

## Rules

### Determinism is Non-Negotiable
- All functions that could affect simulation reproducibility must be deterministic
- The `stable_hash` module uses FNV-1a with frozen regression values
- Never change hash constants or algorithm - this breaks replay compatibility

### No Runtime Dependencies
- This crate must have zero dependencies on other Continuum crates
- Keep external dependencies minimal
- Everything here is foundational - it cannot depend on higher-level abstractions

### ID Types
- All ID types are newtype wrappers around `String`
- They must implement: `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`, `PartialOrd`, `Ord`, `Display`, `From<&str>`
- IDs are immutable once created

### Adding New Utilities
- Only add utilities that are truly foundational (needed by 2+ crates)
- Prefer `const fn` where possible for compile-time evaluation
- Document stability guarantees for anything affecting determinism

## Structure

```
src/
├── lib.rs          # Re-exports
├── ids.rs          # Entity identifier types
└── stable_hash.rs  # FNV-1a deterministic hashing
```

## Testing

- `stable_hash` has frozen regression values - tests will fail if hashing changes
- Run `cargo test -p continuum-foundation` before any changes
