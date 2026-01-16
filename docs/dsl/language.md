# Continuum DSL — Language

This document defines the **Continuum Domain-Specific Language (DSL)**.

The DSL is used to declare **causal structure** in a World.
It is not a scripting language and not a runtime control surface.

---

## 1. Purpose of the DSL

The Continuum DSL exists to make simulation logic:

- declarative
- analyzable
- deterministic
- schedulable as a DAG
- safe under parallel execution
- explicit about causality

The DSL describes **what exists and how it interacts**, not how to execute it.

---

## 2. What the DSL Is (and Is Not)

### The DSL **is**:
- a declarative modeling language
- statically analyzed and compiled
- phase-aware
- signal-centric
- deterministic by construction

### The DSL **is not**:
- a general-purpose programming language
- an imperative script runner
- a runtime control mechanism
- a replacement for the engine
- a place to hide side effects

If a construct cannot be scheduled as a DAG node,
it does not belong in the DSL.

---

## 3. Source Files and Scope

DSL source files use the extension:

```

*.cdsl

```

All DSL files under the World root are:
- discovered recursively
- ordered lexicographically by path
- compiled together as a single unit

There is no include system.
There is no conditional loading.

World structure is defined by the filesystem.

---

## 4. Declarations

The DSL is composed of **top-level declarations**.

Each declaration introduces a named simulation entity.

Supported declaration kinds:

- `signal` — authoritative resolved values
- `entity` — pure index spaces (identity only)
- `member` — per-entity authoritative state
- `operator` — phase-tagged logic blocks
- `field` — observable derived data
- `impulse` — external causal inputs
- `fracture` — emergent tension detectors
- `chronicle` — observer-only interpretation rules

Declarations are **order-independent**.
Dependencies are inferred, not declared manually.

---

## 5. Identifiers and Namespacing

All identifiers are:
- lowercase
- dot-separated
- globally unique within a World

Examples:
```

terra.climate.temperature
stellar.orbit.period
biosphere.biomass.total

```

Identifiers are semantic and stable.
They are part of replay identity.

---

## 6. Phases and Execution Context

Every executable declaration is associated with a **phase**.

Phases are fixed and invariant:

1. Configure
2. Collect
3. Resolve
4. Fracture
5. Measure

Phase constraints are enforced at compile time.

If a declaration attempts an operation not allowed in its phase,
compilation fails.

---

## 7. Signals

A `signal` declares an **authoritative, resolved value**.

Signals:
- resolve exactly once per tick (per stratum)
- may read resolved signals from the previous tick
- may receive inputs accumulated during Collect/Fracture
- must be deterministic

Signals do not mutate directly.
They resolve from inputs.

### 7.1 Dangerous Function Declarations

Some kernel functions are considered dangerous because they silently mask errors or hide problems (violating principle 7: "Fail Hard, Never Mask Errors"). These functions require explicit opt-in via `: uses()` declarations.

**Currently supported on:**
- Signals (`: uses(maths.clamping)`)
- Members (`: uses(maths.clamping)`)

**Planned support:**
- Fractures
- Operators
- Impulses

**Example:**

```cdsl
signal terra.surface.albedo {
    : Scalar<1>
    : uses(maths.clamping)  // Explicit opt-in required
    
    resolve {
        maths.clamp(prev + delta, 0.0, 1.0)  // OK - declared
    }
}
```

If you use a dangerous function without declaring it, compilation will fail with an error explaining why the function is dangerous and what alternatives exist.

**Recommended approach:** Use assertions instead of clamping:

```cdsl
signal terra.surface.albedo {
    : Scalar<1, 0.0..1.0>
    
    resolve {
        prev + delta  // No clamp - fail if out of bounds
    }
    
    assert {
        prev >= 0.0 : fatal, "albedo below physical minimum"
        prev <= 1.0 : fatal, "albedo above physical maximum"
    }
}
```

(See `signals.md` for full semantics.)

---

## 8. Operators

An `operator` declares an executable logic block.

Operators:
- execute in a specific phase
- may read resolved signals
- may emit signal inputs or fields (phase-dependent)
- must be side-effect free outside declared outputs

Operators exist to express logic that does not belong inside a signal resolver.

---

## 9. Fields

A `field` declares **observable data**.

Fields:
- are emitted during the Measure phase
- are derived from resolved signals
- are consumed only by observers

Fields must never influence causality.

(See `fields.md` for full semantics.)

---

## 10. Impulses

An `impulse` declares an **external causal input**.

Impulses:
- have typed payloads
- are applied during Collect
- contribute signal inputs
- are scheduled via Scenarios

Impulses are part of the causal model.

(See `impulses.md`.)

---

## 11. Fractures

A `fracture` declares an **emergent tension detector**.

Fractures:
- run during the Fracture phase
- inspect resolved signals
- emit additional signal inputs when conditions hold

Fractures are not scripted events.
They respond to emergent state.

(See `fractures.md`.)

---

## 12. Chronicles

A `chronicle` declares an **observer-only interpretation rule**.

Chronicles:
- observe fields or resolved signals
- detect patterns or conditions
- record annotations or intervals
- never influence execution

Chronicles exist outside causality.

(See `chronicles.md`.)

---

## 13. Expressions

DSL expressions are:
- pure
- deterministic
- side-effect free

Expressions may:
- perform arithmetic
- call kernel-provided functions
- construct values

Expressions must not:
- perform I/O
- access global state
- depend on execution order

---

## 14. Kernel Functions

The DSL may call namespaced kernel functions (e.g. `maths.*`, `vector.*`, `dt.*`, `physics.*`).

Kernel functions:
- are engine-provided primitives
- have fixed semantics
- may have multiple backend implementations
- must declare determinism guarantees

The DSL does not specify:
- CPU vs GPU
- memory layout
- dispatch mechanics

---

## 15. Type System

The DSL is statically typed.

- All values have known types at compile time
- No implicit type coercion
- Units are part of the type system

If a value cannot be typed, compilation fails.

(See `dsl/types-and-units.md`.)

---

## 16. Dependency Inference

Dependencies are inferred by analyzing reads and writes.

Authors do **not** declare dependencies manually.

From the DSL, the compiler derives:
- signal read sets
- operator dependencies
- phase ordering
- stratum constraints

Manual dependency declaration is forbidden.

---

## 17. Errors and Assertions

The DSL supports explicit assertions.

Assertions:
- validate invariants
- do not modify values
- emit faults on failure

Silent correction is forbidden by default.

(See `dsl/assertions.md`.)

---

## 18. What the DSL Must Never Do

The DSL must never:
- mutate state outside declared outputs
- depend on wall-clock time
- depend on execution order
- access observer data
- hide nondeterminism

If a construct cannot be reasoned about statically,
it does not belong in the DSL.

---

## Summary

- The DSL declares causal structure
- It is declarative and analyzable
- All dependencies are inferred
- Execution is derived, not authored
- Determinism and observer boundaries are enforced at compile time

If logic cannot be expressed declaratively,
the model is wrong.
```
