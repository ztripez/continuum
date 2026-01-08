1# Strata

This document defines **Strata** in Continuum.

Strata are the mechanism by which **multi-rate execution** is expressed.
They are a core part of the execution model, not a domain concept.

---

## 1. What a Stratum Is

A **Stratum** is a named execution lane with its own temporal cadence.

A stratum:
- participates in the simulation timeline
- executes at a defined stride relative to ticks
- contains signals and operators assigned to it
- is scheduled deterministically

Strata exist to model systems that evolve at different rates.

---

## 2. Why Strata Exist

Many simulations involve processes with vastly different timescales.

Examples:
- orbital mechanics vs climate
- tectonics vs ecology
- slow accumulators vs fast feedback loops

Without strata:
- slow systems would be oversampled
- fast systems would be numerically unstable
- coupling would be inefficient or incorrect

Strata allow **correct temporal resolution without changing the model**.

---

## 3. Strata and Time

Strata do not define time themselves.

- Time advances globally via ticks and `dt`
- Strata define *how often* logic runs relative to ticks
- A stratum may skip ticks based on its stride or cadence

Changing a stratum’s cadence changes **sampling frequency**, not meaning.

If meaning changes, the model is incorrect.

---

## 4. Strata and Execution

Strata affect execution structure.

- Execution graphs are constructed per:
```

(phase × stratum × era)

```
- Each stratum has its own DAG
- Strata may execute or be gated depending on the active era

Strata do not:
- define logic
- define causality
- define ordering across strata

They define **when** logic is eligible to run.

---

## 5. Assigning Work to Strata

Signals and operators are assigned to exactly one stratum.

A stratum assignment:
- is declared in the DSL
- is fixed at compile time
- must not change at runtime

A signal or operator must not span multiple strata.

---

## 6. Strata Coupling

Strata may interact via signals.

- Signals in one stratum may read signals from another
- Reads always observe the most recently resolved value
- No implicit interpolation or extrapolation is performed

All cross-strata coupling must be explicit and deterministic.

---

## 7. Strata and Eras

Eras control **which strata are active**.

Within an era:
- some strata may be active
- some strata may be gated (paused)
- some strata may run at reduced cadence

Eras do not change stratum identity or structure.
They select execution policy.

---

## 8. Strata and Determinism

Strata behavior must be deterministic.

Given:
- the same World
- the same Scenario
- the same era sequence
- the same tick

strata execution must produce identical results.

Skipping or executing a stratum must never depend on runtime conditions
other than declared era policy.

---

## 9. What Strata Are Not

Strata are **not**:
- domains
- threads
- processes
- performance hints
- scheduling heuristics

They are **semantic execution structure**.

---

## Summary

- Strata define multi-rate execution
- They control *when* logic runs, not *what* it does
- Strata are fixed at compile time
- Eras gate and modulate strata
- Cross-strata interaction is explicit and deterministic

If a change in stratum configuration alters meaning,
the abstraction has been violated.
