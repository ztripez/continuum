# Entity & Impulse Determinism

This document defines the strict deterministic semantics for entity iteration, spatial queries, and impulse execution. These rules ensure that simulation outcomes remain identical regardless of parallel execution, memory layout, or hardware architecture.

---

## 1. Entity Identity & Ordering

Entities are identified by a stable `InstanceId` (string).
All iteration over entity sets **MUST** follow strict **lexical order** of `InstanceId`.

### 1.1 Iteration
When iterating over `entity.path` (e.g. in `agg.sum` or `map`):
- The order is defined by sorting `InstanceId` strings.
- This order is invariant across runs and platforms.
- `IndexMap` or sorted vectors must be used; `HashMap` iteration is forbidden.

### 1.2 Access
- `entity[i]`: Access by numerical index into the **sorted** instance list. `entity[0]` is the instance with the lexicographically first ID.
- `entity["name"]`: Direct lookup by `InstanceId`.

---

## 2. Set Operations

### 2.1 Filter
`filter(entity, predicate)`
- Output sequence preserves the input order (lexical `InstanceId`).
- Stability: If `A` comes before `B` in input, and both match, `A` comes before `B` in output.

### 2.2 Pairs
`pairs(entity)`
- Generates unique pairs `(a, b)` where `a` comes before `b` in the sorted sequence.
- Iteration order:
  - Outer loop: `i` from `0` to `N-2`
  - Inner loop: `j` from `i+1` to `N-1`
  - Yields: `(instances[i], instances[j])`
- Self-pairs `(a, a)` are never generated.
- Symmetric pairs `(b, a)` are never generated (implicit in `(a, b)` for symmetric physics).

### 2.3 Other
`other(entity)`
- Used within a member context of instance `S` (self).
- Returns strict lexical iteration of all instances `X` where `X != S`.

---

## 3. Spatial Operations

Spatial queries involve floating-point distance checks. To preserve determinism:

### 3.1 Nearest
`nearest(entity, position)`
- Metric: Euclidean distance squared (avoid sqrt for comparison).
- Tie-breaking: If multiple instances have the exact same distance (bitwise identical float):
  - Select the one with the **lexicographically lowest `InstanceId`**.
  - This ensures stability even with grid-aligned or coincident entities.

### 3.2 Within
`within(entity, position, radius)`
- Metric: Euclidean distance squared <= radius squared.
- Output order: Strict **lexical order** of `InstanceId` among matches.
- Filtering is independent of distance sorting.

---

## 4. Selection Operations

### 4.1 First
`first(entity, predicate)`
- Returns the first instance in **lexical order** that satisfies the predicate.
- Short-circuiting is permitted (stop after first match).

---

## 5. Aggregates

Floating-point arithmetic is not associative. Order matters.

### 5.1 Accumulation (Sum, Product, Mean)
- Accumulation **MUST** proceed in strict lexical order of `InstanceId`.
- Parallel reduction is permitted **ONLY** if the recombination tree is fixed and stable (e.g., fixed-chunk tree reduction).
- Simple linear accumulation (left-to-right) is the baseline correctness model.

### 5.2 Min / Max
- Standard float comparison.
- NaN propagation rules must be consistent (e.g. `f64::max` or `f64::min`).
- Tie-breaking: If values are equal, the result is the value; identity of the source doesn't strictly matter for the *value*, but if returning the *instance* (argmax), lowest `InstanceId` wins.

### 5.3 Boolean (Any, All, None)
- **Any**: True if at least one match. Short-circuit allowed.
- **All**: True if all match. Short-circuit allowed on first false.
- **None**: True if zero matches. Short-circuit allowed on first true.

---

## 6. Impulse Semantics

Impulses introduce external causality but must be processed deterministically.

- **Ordering**: If multiple impulses occur in the same tick:
  - They are processed in the order defined by the **Input Channel** accumulation (which must be stable).
  - If multiple impulses target the same signal, the signal's input accumulator defines the combining logic (sum, overwrite, etc.).
- **Payloads**:
  - Payload fields (`payload.mass`) are read-only during the impulse apply block.
  - Payloads are authoritative values provided by the scenario/driver.

---

## 7. Implementation Requirements

1. **Storage**: `EntityStorage` must use `IndexMap<InstanceId, Data>` or `Vec<(InstanceId, Data)>` sorted by ID.
2. **Bytecode**: VM iterators must abstract over this ordering.
3. **Validation**: Tests must verify that shuffling memory layout or insert order does not change iteration order.
