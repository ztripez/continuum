# Determinism

This document defines the **determinism guarantees** of Continuum execution.

Determinism is a **fundamental property** of the system.
It is not a configuration option and not a best-effort goal.

If determinism is violated, the execution is incorrect.

---

## 1. What Determinism Means

Continuum execution is deterministic if:

> Given the same **world specification**, **scenario specification**, **initial seed**,  
> and **execution policy**, the system produces identical causal history.

Determinism applies to:
- signal values
- era transitions
- execution order
- fault behavior

Determinism does **not** require identical performance or execution speed.

---

## 2. Determinism Scope

Determinism is defined over **causal history**, not observation.

- Signals must be identical
- Execution ordering must be identical
- Observers may differ in presentation, but not in input data

If removing observers changes signal values or execution order,
determinism has been violated.

---

## 3. Stable Inputs

A deterministic **run** is fully specified by:

- the **world specification**
  - merged `world.yaml`
  - compiled DSL IR
- the **scenario specification**
  - initial signal values
  - signal parameters
  - declared configuration choices
- the initial world seed
- the sequence of eras
- the sequence of `dt` values

No other input may influence execution.

Wall-clock time, thread scheduling, execution speed, and hardware topology
must not affect results.

---

## 4. World vs Scenario Determinism

Determinism operates at two levels:

### World determinism
- The same World specification must always produce the same execution graph.
- World identity is independent of Scenario or Run.

### Scenario determinism
- The same Scenario applied to the same World must produce the same initial state.
- Changing a Scenario is expected to change results.
- Re-running the same Scenario must never do so.

---

## 5. Ordering Is Explicit

All ordering in the system must be explicit and stable.

This includes:
- file discovery and loading order
- symbol registration order
- IR construction order
- execution DAG construction
- topological level ordering
- reduction and accumulation order

If any ordering depends on:
- hash map iteration
- pointer address
- thread timing
- platform-specific behavior

the system is incorrect.

---

## 6. Randomness Is Derived

All stochastic behavior must be derived from the world seed.

- Random values must be generated via labeled derivation
- Labels must be stable and semantic
- No ambient or implicit randomness is allowed

Examples of forbidden sources:
- system RNGs
- default hashers
- time-based seeds

If randomness cannot be replayed exactly, it must not exist.

---

## 7. Deterministic Parallelism

Parallel execution must not change outcomes.

- Nodes may execute concurrently only if causally independent
- Reduction order must be fixed and deterministic
- Parallel scheduling must not affect visible results

Parallelism is an optimization, never a semantic feature.

If correctness depends on serial execution, serialization must be explicit.

---

## 8. Determinism Across Backends

Different execution backends (e.g. CPU vs GPU) must preserve semantics.

- Kernel implementations must declare their determinism guarantees
- Strict-deterministic kernels must produce bitwise-identical results
- Relaxed-deterministic kernels may differ numerically, but only in non-causal phases

Kernel execution that influences causality must be strict-deterministic.

---

## 9. Faults and Determinism

Fault behavior must be deterministic.

- The same fault must occur at the same tick
- Fault ordering must be stable
- Budget exhaustion must be deterministic

A fault must never appear or disappear due to scheduling or timing differences.

---

## 10. Pre-Execution Determinism

All pre-execution steps must be deterministic.

This includes:
- scenario application
- initial signal validation
- execution graph selection
- warmup or pre-causal stabilization

Pre-execution steps must be:
- bounded
- deterministic
- independent of observers or external state

If pre-execution results differ across runs, execution must fail.

---

## 11. What Determinism Is Not

Determinism does **not** require:
- identical floating-point rounding across architectures (unless causal)
- identical performance characteristics
- identical observer output formatting

Determinism is about **meaning**, not presentation.

---

## 12. Detecting Determinism Violations

The system should make determinism violations visible.

Examples include:
- replay mismatch detection
- cross-backend comparison
- invariant checks
- stable hash comparison of causal history

A determinism violation is always a bug.

---

## Summary

- Determinism is mandatory
- World and Scenario are both deterministic inputs
- All inputs must be explicit
- Ordering must be stable
- Randomness must be derived
- Parallelism must not change meaning
- Faults must be reproducible

If two runs with the same World, Scenario, seed, and policy
produce different causal history, the execution model is wrong.
