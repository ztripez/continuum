Here is a **clean, enforceable `PRINCIPLES.md`** that sits *between* the manifesto and the system docs.

This is written to be **normative**, **reviewable**, and **durable**, without leaking implementation details or old engine assumptions.

---

```md
# PRINCIPLES

This document defines the **non-negotiable design principles** of Continuum.

These principles are **enforceable**.  
They guide architecture, APIs, crate boundaries, and reviews.

If a design violates a principle, it is incorrect — regardless of convenience, performance, or precedent.

---

## 1. Causality Is Primary

Continuum models **causal systems**.

- Simulation behavior must arise from declared causes.
- No outcome may be authored directly.
- No result may be injected externally without being modeled as a cause.

If a result appears without a cause, the system is incorrect.

---

## 2. Graph-First Execution

All simulation behavior must be representable as a **deterministic execution graph**.

- Logic is declared, not handwritten.
- Dependencies are inferred, not manually ordered.
- Execution order is derived from causality, not author intent.

If behavior cannot be scheduled as a DAG, it does not belong in the system.

---

## 3. Determinism Is Mandatory

Determinism is not optional.

- All ordering must be stable and explicit.
- All randomness must be derived from an explicit, labeled seed.
- Identical inputs must produce identical outputs.

If behavior depends on thread timing, hash order, platform quirks, or ambient entropy, it violates this principle.

---

## 4. Signals Are Authority

Authoritative simulation state is expressed as **signals**.

- Kernel execution may read only resolved signals and authoritative state.
- If something influences causality, it must be a signal.
- Derived or sampled data must never feed back into causality.

Signals define **what is true** in the simulation.

---

## 5. Fields Are Observation

Fields are **measurements**, not state.

- Fields may be derived, reconstructed, sampled, or visualized.
- Kernel execution must never depend on fields.
- Observers may read fields; they may not influence simulation behavior.

Observation must never affect causality.

---

## 6. Observer Boundary Is Absolute

Observers are strictly non-causal.

- Observers must be removable without changing outcomes.
- Visualization, analytics, logging, and inspection are observers.
- No observer may write signals or alter execution order.

If removing an observer changes results, the boundary has been violated.

---

## 7. Fail Hard, Never Mask Errors

Silent error masking is forbidden.

- Do not silently clamp, wrap, or "fix" invalid values.
- Invalid states must be detected and surfaced explicitly.
- A visible failure is always preferable to silent corruption.

Correctness is more important than continuity.

**Enforcement:** Functions that silently mask errors (e.g., `maths.clamp`, `maths.saturate`) require explicit `: uses()` declarations in DSL. This forces conscious decision-making when using dangerous patterns. See `@docs/dsl/assertions.md` for alternatives.

---

## 8. Validation First, Correction Explicit

Validation is the default posture.

- Bounds, invariants, and sanity conditions must be checked.
- Corrective behavior (clamping, damping, wrapping) must be:
  - explicit
  - local
  - visible in declarations

There are no implicit safety nets.

---

## 9. No Compatibility That Preserves Incorrectness

Backward compatibility must not preserve broken semantics.

- Do not retain old behavior if it masked errors.
- Do not provide silent migration paths.
- Breaking changes that improve correctness are acceptable and expected.

Users must opt into correctness consciously.

---

## 10. Parallelism Must Not Change Meaning

Parallel execution is an optimization, not a semantic feature.

- Parallelism must never change outcomes.
- Execution order must be well-defined even when work runs concurrently.
- If correctness depends on serial execution, enforce serialization explicitly.

Concurrency must not introduce ambiguity.

---

## 11. Time Is Explicit and Controlled

Time advancement is part of the model.

- dt, cadence, and multi-rate execution must be explicit.
- Changes in temporal resolution must not alter semantics.
- Derived time units must be stable and reproducible.

If changing dt changes meaning, the model is incorrect.

---

## 12. Context Is Configuration, Not Logic

Configuration defines **how a system runs**, not **what it is**.

- Manifests configure time, eras, and policy.
- Simulation structure lives in declarations.
- Configuration must not encode logic.

Logic belongs in the model, not in configuration files.

---

## 13. Correctness Over Convenience

When forced to choose:

- Prefer correctness over performance.
- Prefer explicitness over brevity.
- Prefer breaking changes over silent failure.

The system must be trusted before it can be optimized.

---

## 14. Concepts Must Generalize

Core concepts must not exist to serve a single world or domain.

- If a concept only exists because “this case needs it”, it does not belong in the core.
- World-specific behavior must be expressed using general mechanisms.

The system must remain domain-agnostic.

---

## 15. Observation Gives Meaning, Not Truth

Simulation produces **causal history**.

Meaning is assigned by observers.

- The system does not interpret results.
- Observers contextualize, label, and visualize.
- Truth belongs to causality; meaning belongs to observation.

Never confuse the two.

---

## Closing Rule

When in doubt, ask:

> **Does this preserve causality, determinism, and observability — without hiding errors?**

If the answer is not a clear “yes”, the design is wrong.
```

---

If you want next, I can:

* turn these principles into a **PR review checklist**, or
* map each principle to **specific compiler/runtime enforcement points** so nothing stays aspirational.
