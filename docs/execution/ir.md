# Intermediate Representation (IR)

This document defines the **Intermediate Representation (IR)** used by Continuum
to bridge between the **DSL** and the **execution DAG**.

The IR is the **semantic contract** between authoring and execution.

---

## 1. Purpose of the IR

The Continuum IR exists to:

- represent declared simulation structure precisely
- remove syntactic detail from the DSL
- make dependencies explicit and analyzable
- provide a stable input for DAG construction
- separate *what is modeled* from *how it is executed*

The IR is not executable by itself.
It is a **semantic description of work**.

---

## 2. Position in the Pipeline

The compilation pipeline is:

```

DSL source
↓
AST (syntax-focused)
↓
Typed IR (semantic)
↓
Execution DAG (scheduled)
↓
Runtime execution

```

The IR is the **last author-facing artifact**.
Everything after it is mechanical.

---

## 3. IR Is Fully Typed

All IR elements are **fully typed**.

This includes:
- signal outputs
- operator inputs and outputs
- field payloads
- impulse schemas
- time and unit references

There is no type inference at runtime.

If something cannot be typed at compile time, it does not belong in the IR.

---

## 4. What the IR Represents

The IR represents **declared simulation entities**, not execution order.

At minimum, the IR contains:

- Signals
- Operators
- Fields
- Impulses
- Observer rules (lens / chronicles)
- Assertions
- Time references

The IR describes:
- what exists
- what it reads
- what it writes
- in which phase it is allowed to execute

It does **not** describe:
- scheduling
- parallelism
- memory layout
- backend choice

---

## 5. Signals in the IR

Each signal in the IR specifies:

- a stable identifier
- output type
- phase (`Resolve`)
- stratum membership
- the set of other signals it reads
- assertion rules

Signal reads are **inferred**, not declared.

Signals define the authoritative state of the model.

---

## 6. Operators in the IR

Operators represent executable logic blocks.

Each operator specifies:

- a stable identifier
- execution phase
- stratum membership
- the set of signals it reads
- the set of effects it may produce
  - signal inputs
  - fields (if in Measure)
- assertion rules

Operators are phase-constrained.
An operator that violates phase rules is invalid.

---

## 7. Fields in the IR

Fields are observer-only constructs.

Each field specifies:

- a stable identifier
- payload type
- topology and reconstruction metadata
- emission constraints (phase = Measure)

Fields may be referenced only by observer logic.

Kernel phases must not reference fields in the IR.

---

## 8. Impulses in the IR

Impulses represent external inputs.

Each impulse specifies:

- a stable identifier
- payload schema
- application phase and constraints
- valid target scope

Impulses are part of the causal model.
They must be represented explicitly in the IR.

---

## 9. Assertions in the IR

Assertions are first-class IR elements.

Each assertion specifies:
- the condition to check
- the scope (signal or operator)
- the failure classification

Assertions do not:
- modify values
- affect execution order

Assertions produce **faults**, not corrections.

---

## 10. Dependency Information

The IR explicitly captures dependencies.

Dependencies include:
- signal reads
- operator reads
- phase constraints
- stratum constraints

This dependency information is the **sole input** to DAG construction.

No dependency information is inferred later.

---

## 11. IR Stability

The IR must be:

- deterministic
- stable across compilations
- independent of file order beyond declared structure

Two identical DSL inputs must produce identical IR.

If IR changes due to nondeterministic factors, the system is incorrect.

---

## 12. IR Validation

Before DAG construction, the IR must be validated.

Validation includes:
- type correctness
- phase correctness
- absence of illegal field access
- absence of illegal signal writes
- assertion well-formedness

Invalid IR must fail compilation.

---

## 13. What the IR Is Not

The IR is **not**:
- executable code
- a scripting language
- a task graph
- a runtime structure
- backend-specific

It is a **semantic model**.

---

## Summary

- IR is the semantic bridge between DSL and execution
- IR is fully typed and deterministic
- IR represents *what exists*, not *how it runs*
- Dependencies are explicit and complete
- DAG construction relies exclusively on IR

If behavior cannot be expressed in the IR,
it cannot be executed.
