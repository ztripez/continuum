# Continuum DSL — Language

This document defines the **Continuum Domain-Specific Language (DSL)**.

The DSL is used to declare **causal structure and mathematical relationships**
in a Continuum World.

It is not a scripting language.
It is not a runtime control surface.
It is not an engine API.

---

## 1. Purpose of the DSL

The DSL exists to let authors express **high-level mathematical and physical models**
directly, without:

- execution plumbing
- scheduling concerns
- backend selection
- bindings or dispatch
- manual dependency wiring

The DSL describes **what exists and how it relates**, not how it runs.

From DSL source, the system derives:
- a typed intermediate representation (IR)
- a deterministic execution DAG
- a safe parallel execution plan

---

## 2. Design Principles

The DSL is designed to be:

- **Declarative** — structure before behavior
- **Math-first** — equations over mechanics
- **Deterministic** — replayable by construction
- **Analyzable** — all dependencies inferred
- **Schedulable** — reducible to a DAG
- **Observer-safe** — causality and observation are separated

If a construct cannot be statically reasoned about,
it does not belong in the DSL.

---

## 3. What the DSL Is (and Is Not)

### The DSL **is**:
- a declarative modeling language
- statically typed and unit-checked
- phase-aware
- signal-centric
- deterministic by construction

### The DSL **is not**:
- a general-purpose programming language
- an imperative update loop
- a scripting environment
- a runtime configuration mechanism
- a place to hide side effects

If logic depends on execution order or mutable state,
it is invalid.

---

## 4. Source Files and Compilation Unit

DSL source files use the extension:

```

*.cdsl

```

All DSL files under a World root are:
- discovered recursively
- ordered lexicographically by path
- compiled together as a single unit

There are:
- no includes
- no conditional loading
- no partial compilation

World structure is defined by the filesystem.

---

## 5. Declarations

The DSL consists of **top-level declarations**.

Each declaration introduces a named entity into the model.

Supported declaration kinds:

- `signal` — authoritative resolved values
- `operator` — phase-tagged executable logic
- `field` — observable output
- `impulse` — external causal input
- `fracture` — emergent tension detector
- `chronicle` — observer-only interpretation

Declarations are **order-independent**.
Dependencies are inferred from usage.

---

## 6. Identifiers and Namespacing

All identifiers are:
- lowercase
- dot-separated
- globally unique within a World

Examples:
```

stellar.orbit.period
terra.climate.temperature
biosphere.biomass.total

```

Identifiers are semantic and stable.
They are part of replay identity.

---

## 7. Phases and Execution Context

All executable logic is associated with a **phase**.

Phases are fixed and invariant:

1. Configure
2. Collect
3. Resolve
4. Fracture
5. Measure

Phase rules are enforced at compile time.

If a declaration performs an operation not allowed in its phase,
compilation fails.

---

## 8. Signals

A `signal` declares an **authoritative value** in the simulation.

Signals:
- resolve exactly once per tick (per stratum)
- produce a single resolved value
- may read resolved signals from the previous tick
- may consume accumulated inputs
- must be deterministic

Signals do not mutate.
They resolve from inputs.

(See `signals.md`.)

---

## 9. Operators

An `operator` declares a block of executable logic.

Operators:
- execute in a specific phase
- may read resolved signals
- may emit signal inputs or fields (phase-dependent)
- must be side-effect free outside declared outputs

Operators exist to express logic that does not belong
inside a signal resolver but still participates in causality.

---

## 10. Fields

A `field` declares **observable data**.

Fields:
- are emitted during the Measure phase
- are derived from resolved signals
- are consumed only by observers

Fields never influence execution.

(See `fields.md`.)

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
- run during the Fracture phase
- inspect resolved signals
- emit additional signal inputs when conditions hold

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

## 14. Expressions

DSL expressions are:

- pure
- deterministic
- side-effect free

Expressions may:
- perform arithmetic
- call kernel-provided functions
- construct structured values

Expressions must not:
- perform I/O
- access global state
- depend on execution order

---

## 15. Math-First Semantics

The DSL is designed to express **high-level mathematics directly**.

### Vector and Matrix Algebra
Built-in support includes:
- elementwise arithmetic
- dot and cross products
- norms, normalization
- matrix–vector and matrix–matrix multiplication
- deterministic reductions

All operations are statically typed and unit-checked.

### Deterministic Sequence Algebra
When the model exposes **stable-ordered sequences**
(e.g. stars, moons, layers):

The DSL provides deterministic constructs such as:
- `map(sequence, f)`
- `fold(sequence, init, f)`
- commutative accumulators (`sum`, `max`, etc.)

Iteration over unstable or unordered collections is forbidden.

### Units-Aware Mathematics
Math functions enforce dimensional correctness:
- trigonometric functions require angles
- exponential/logarithmic functions require dimensionless inputs
- powers and roots preserve unit algebra

Invalid dimensional operations fail at compile time.

---
## 16. dt-Robust Operators and Relaxation

Many physical and mathematical processes must behave consistently
across different `dt` values.

The DSL provides **dt-robust operators** to express such behavior
explicitly and declaratively.

These operators are:
- mathematically defined
- deterministic
- explicit in intent
- invariant under changes in `dt` (within numerical limits)

They are not implicit corrections.

### Relaxation and Smoothing

Common patterns such as decay, relaxation, and smoothing are expressed
via dedicated operators, for example:

- exponential relaxation toward a target
- bounded accumulation
- asymptotic decay

Conceptually:

- relaxation expresses *rates*, not steps
- smoothing expresses *continuous processes*, not frame-based hacks

The meaning of these operators does not depend on tick frequency.

### Explicit, Not Implicit

The engine never applies dt relaxation implicitly.

If a signal requires dt-robust behavior:
- the author must use a dt-aware operator
- the intent must be visible in the DSL

If a signal becomes unstable when `dt` changes,
that instability is a modeling error unless a dt-robust operator is used.

### Determinism and Validation

dt-robust operators:
- must be strictly deterministic
- must declare their mathematical form
- must not hide runaway behavior

Assertions still apply.
If a value diverges despite using a dt-robust operator,
the simulation must fail visibly.

---

## 17. Kernel Primitives

The DSL may call namespaced kernel functions (e.g. `maths.*`, `vector.*`, `dt.*`, `physics.*`).

Kernel functions are:
- engine-provided primitives
- strongly typed and unit-checked
- deterministic in causal phases
- backend-agnostic (CPU/GPU selection is automatic)

The DSL never exposes:
- bindings
- memory layout
- dispatch
- shaders
- backend choice

Kernel primitives exist to express complex math
without scaffolding.

---

## 18. Type System and Units

The DSL is statically typed.
Units are part of the type system.

- No implicit coercions
- No implicit unit conversions
- No untyped "magic numbers" in unit contexts

### SI Base Dimensions

All physical units are represented as combinations of 7 SI base dimensions:

| Dimension | Symbol | Base Unit |
|-----------|--------|-----------|
| Length | L | meter (m) |
| Mass | M | kilogram (kg) |
| Time | T | second (s) |
| Electric current | I | ampere (A) |
| Temperature | Θ | kelvin (K) |
| Amount of substance | N | mole (mol) |
| Luminous intensity | J | candela (cd) |

Plus an additional tracked dimension for angles (radians).

### Derived Units

Derived units are automatically decomposed to base dimensions:

| Unit | Decomposition | Dimensions |
|------|---------------|------------|
| `N` (Newton) | kg·m/s² | M¹·L¹·T⁻² |
| `Pa` (Pascal) | kg/(m·s²) | M¹·L⁻¹·T⁻² |
| `J` (Joule) | kg·m²/s² | M¹·L²·T⁻² |
| `W` (Watt) | kg·m²/s³ | M¹·L²·T⁻³ |
| `Hz` (Hertz) | 1/s | T⁻¹ |
| `V` (Volt) | kg·m²/(A·s³) | M¹·L²·T⁻³·I⁻¹ |

### Dimensional Equivalence

Units with the same base dimensions are **automatically equivalent**:

```
J = W·s = N·m = kg·m²/s²
```

The compiler normalizes all units to base dimensions.
If two expressions have the same dimensional signature,
they are type-compatible.

Examples:
- `<J>` and `<W·s>` are interchangeable (both are M¹·L²·T⁻²)
- `<Pa>` and `<N/m²>` are interchangeable (both are M¹·L⁻¹·T⁻²)
- `<W/m²>` is M¹·T⁻³ (power flux / irradiance)

### Dimensional Algebra

Units follow algebraic rules during operations:

| Operation | Rule | Example |
|-----------|------|---------|
| Multiply | dimensions add | m × m = m² |
| Divide | dimensions subtract | m / s = m·s⁻¹ |
| Power | dimensions scale | (m/s)² = m²/s² |
| Add/Subtract | dimensions must match | m + m ✓, m + s ✗ |
| sqrt | dimensions halve (must be even) | √(m²) = m |

### Unit Annotations

Units are annotated with angle brackets:

```
config {
    max_depth: 11000.0 <m>
    max_power: 1e15 <W>
    density: 2700.0 <kg/m³>
}

resolve {
    maths.clamp(prev + collected, 0.0 <W>, config.max_power)
}
```

Supported annotation formats:
- Base units: `<m>`, `<kg>`, `<s>`, `<K>`, `<A>`, `<mol>`, `<cd>`
- Derived units: `<N>`, `<Pa>`, `<J>`, `<W>`, `<Hz>`, `<V>`, `<Ω>`
- Compound units: `<m/s>`, `<kg/m³>`, `<W/m²>`, `<N·m>`
- Exponents: `<m²>`, `<s⁻¹>`, `<m^2>`, `<s^-1>`

### Compile-Time Enforcement

Invalid dimensional operations fail at compile time:

```
# ERROR: cannot add m and s
signal.distance + signal.time

# ERROR: sqrt requires even exponents
maths.sqrt(signal.velocity)  # velocity is m/s, cannot sqrt

# ERROR: exp requires dimensionless input
maths.exp(signal.temperature)  # must divide by reference temp first
```

### Dimensionless Quantities

Some quantities are intentionally dimensionless:

```
: Scalar<1>                    # pure number
: Scalar<1, 0..1>              # fraction/ratio
: Vec4<1, magnitude: 1>        # unit quaternion
```

Transcendental functions (exp, log, sin, cos) require dimensionless inputs

---

## 19. Dependency Inference

All dependencies are inferred.

From DSL source, the compiler derives:
- signal read sets
- operator dependencies
- phase ordering
- stratum constraints

Manual dependency declaration is forbidden.

If the compiler cannot infer dependencies,
the model is invalid.

---

## 20. Assertions and Faults

The DSL supports explicit assertions.

Assertions:
- validate invariants
- never modify values
- emit structured faults on failure

Silent correction is forbidden by default.

(See `dsl/assertions.md`.)

---

## 21. What the DSL Must Never Do

The DSL must never:
- mutate hidden state
- depend on wall-clock time
- depend on execution order
- read observer data
- hide nondeterminism
- encode backend or scheduling concerns

If logic cannot be reasoned about statically,
it does not belong in the DSL.

---

## Summary

- The DSL declares causal structure
- It is math-first, declarative, and analyzable
- All dependencies are inferred
- Execution is derived, not authored
- Determinism and observer boundaries are enforced at compile time

If a model requires plumbing or scaffolding,
the abstraction has been violated.
