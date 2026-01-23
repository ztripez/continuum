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
- `analyzer` — post-hoc analysis queries on field snapshots

Declarations are **order-independent**.
Dependencies are inferred, not declared manually.

---

## 5. Config and Const Blocks

### 5.1 Const Block

A `const` block declares **global simulation constants**.

```cdsl
const {
    physics.boltzmann: 1.380649e-23 <J/K>
    physics.speed_of_light: 299792458 <m/s>
    physics.stefan_boltzmann: 5.67e-8
}
```

Const values:
- Are immutable global constants
- Never change during execution
- Are NOT overridable by scenarios
- Are loaded once during world initialization (lifecycle stage 4)
- Must be compile-time literals (Scalar or Vec3 with literal components)
- Are accessible via `load_const("path")` in all execution phases

**Non-literal expressions are forbidden:**

```cdsl
const {
    valid: 42.0                    // ✓ Literal scalar
    valid_vec: [1.0, 2.0, 3.0]    // ✓ Literal Vec3
    invalid: 1.0 + 2.0             // ✗ Kernel call - PANIC
    invalid_ref: config.base       // ✗ Reference - PANIC
}
```

Attempting to use non-literal expressions causes `build_runtime()` to panic with
a clear error message identifying the problematic const path.

### 5.2 Config Block

A `config` block declares **configuration parameters with world defaults**.

```cdsl
config {
    thermal.decay_halflife: 1.42e17 <s>
    thermal.initial_temp: 5500.0 <K>
    thermal.coupling_factor: 0.01
}
```

Config values:
- Define world-level default parameters
- Scenarios may override config values (const values are NOT overridable)
- Are frozen at scenario application (lifecycle stage 4)
- Remain immutable during execution
- Must be compile-time literals (Scalar or Vec3 with literal components)
- Are accessible via `load_config("path")` in all execution phases

**Config vs Const:**

| Feature | Config | Const |
|---------|--------|-------|
| **Overridable** | ✓ Scenarios may override | ✗ Never overridable |
| **Use case** | Parameterizing simulations | Physical constants |
| **Mutability** | Frozen at scenario load | Frozen at world load |
| **Example** | Decay rates, thresholds | Boltzmann constant, π |

**Usage in DSL:**

```cdsl
signal core.temp {
    : Scalar<K>
    
    resolve {
        let gravity = load_config("physics.gravity")
        let boltzmann = load_const("physics.boltzmann")
        
        prev * boltzmann / gravity
    }
}
```

Both `load_config()` and `load_const()` return frozen values and are available
in all execution phases. They are NOT signals (no prev/current distinction).

---

## 6. Identifiers and Namespacing

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

## 7. Phases and Execution Context

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

## 8. Signals

A `signal` declares an **authoritative, resolved value**.

Signals:
- resolve exactly once per tick (per stratum)
- may read resolved signals from the previous tick
- may receive inputs accumulated during Collect/Fracture
- must be deterministic

Signals do not mutate directly.
They resolve from inputs.

### 8.1 Dangerous Function Declarations

Some kernel functions are considered dangerous because they silently mask errors or hide problems (violating principle 7: "Fail Hard, Never Mask Errors"). These functions require explicit opt-in via `: uses()` declarations.

**Supported on all executable primitives:**
- Signals (`: uses(maths.clamping)`)
- Members (`: uses(maths.clamping)`)
- Fractures (`: uses(maths.clamping)`)
- Operators (`: uses(maths.clamping)`)
- Impulses (`: uses(maths.clamping)`)

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

## 9. Operators

An `operator` declares an executable logic block.

Operators:
- execute in a specific phase
- may read resolved signals
- may emit signal inputs or fields (phase-dependent)
- must be side-effect free outside declared outputs

Operators exist to express logic that does not belong inside a signal resolver.

---

## 10. Fields

A `field` declares **observable data**.

Fields:
- are emitted during the Measure phase
- are derived from resolved signals
- are consumed only by observers

Fields must never influence causality.

(See `fields.md` for full semantics.)

---

## 11. Impulses

An `impulse` declares an **external causal input**.

Impulses:
- have typed payloads
- are applied during Collect
- contribute signal inputs
- are scheduled via Scenarios

Impulses are part of the causal model.

(See `impulses.md`.)

---

## 12. Fractures

A `fracture` declares an **emergent tension detector**.

Fractures:
- run during the Fracture phase (after Resolve, before Measure)
- inspect resolved signals from current tick
- accumulate signal inputs for the next tick's Collect phase

Fractures use:
- `when { }` blocks for tension conditions
- `collect { }` blocks for input accumulation

Fractures are not scripted events.
They respond to emergent state.

(See `fractures.md`.)

---

## 13. Chronicles

A `chronicle` declares an **observer-only interpretation rule**.

Chronicles:
- observe fields or resolved signals
- detect patterns or conditions
- record annotations or intervals
- never influence execution

Chronicles exist outside causality.

(See `chronicles.md`.)

---

## 14. Analyzers

An `analyzer` declares a **post-hoc analysis query on field snapshots**.

Analyzers:
- read field data after simulation completes
- compute derived analysis results
- produce structured JSON output
- optionally include validation checks with severity levels
- never influence causality
- are invoked via CLI: `continuum analyze run <name>`

Analyzers are pure observers that run outside the simulation loop on captured field data.

(See `analyzers.md`.)

---

## 15. Expressions

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

## 16. Kernel Functions

The DSL may call namespaced kernel functions (e.g. `maths.*`, `vector.*`, `dt.*`, `physics.*`, `stats.*`).

Kernel functions:
- are engine-provided primitives
- have fixed semantics
- may have multiple backend implementations
- must declare determinism guarantees

The DSL does not specify:
- CPU vs GPU
- memory layout
- dispatch mechanics

Analyzer-specific kernels (available only in analyzer compute blocks):
- `stats.*` — statistical functions (mean, median, correlation, histogram)
- `field.samples()` — field data access
- `util.*` — spatial utilities (latitude, longitude, distance)

---

## 17. Type System

The DSL is statically typed.

- All values have known types at compile time
- No implicit type coercion
- Units are part of the type system
- JSON output from analyzers is dynamically typed (matches JSON schema)

If a value cannot be typed, compilation fails.

(See `dsl/types-and-units.md`.)

---

## 18. Dependency Inference

Dependencies are inferred by analyzing reads and writes.

Authors do **not** declare dependencies manually.

From the DSL, the compiler derives:
- signal read sets
- operator dependencies
- phase ordering
- stratum constraints

Manual dependency declaration is forbidden.

Analyzers declare field requirements explicitly via `: requires(fields: [...])`.

---

## 19. Errors and Assertions

The DSL supports explicit assertions.

Assertions:
- validate invariants
- do not modify values
- emit faults on failure

Silent correction is forbidden by default.

(See `dsl/assertions.md`.)

---

## 20. What the DSL Must Never Do

The DSL must never:
- mutate state outside declared outputs
- depend on wall-clock time
- depend on execution order
- access observer data (except in analyzers, which read field data post-hoc)
- hide nondeterminism

If a construct cannot be reasoned about statically,
it does not belong in the DSL.

---

## 21. Summary

- The DSL declares causal structure
- It is declarative and analyzable
- All dependencies are inferred
- Execution is derived, not authored
- Determinism and observer boundaries are enforced at compile time

If logic cannot be expressed declaratively,
the model is wrong.
```
