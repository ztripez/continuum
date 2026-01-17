//! WarmUp phase policy
//!
//! The WarmUp phase runs tick phases (Configure → Collect → Resolve → Fracture → Measure → Assert)
//! repeatedly until the simulation reaches a stable state, as defined by a convergence predicate.
//!
//! # Design Principles
//!
//! ## Determinism
//!
//! WarmUp iteration count is part of the replayable execution trace. Given the same:
//! - World definition
//! - Scenario parameters
//! - Random seed
//!
//! WarmUp always runs the **exact same number of iterations** and produces **identical state**.
//!
//! ## Safety
//!
//! WarmUp must terminate:
//! - **Convergence:** Predicate evaluates to `true` → WarmUp completes
//! - **Timeout:** `max_iterations` reached → behavior determined by `on_timeout`
//!
//! ## Use Cases
//!
//! WarmUp is used when initial conditions need to "settle" before the main simulation:
//! - **Thermal equilibrium:** Run until temperature distribution stabilizes
//! - **Gravitational settling:** Run until orbital elements converge
//! - **Pressure equilibrium:** Run until fluid pressures balance
//!
//! # Examples
//!
//! ```cdsl
//! world terra {
//!     warmup {
//!         // Converge when mantle temperature change is < 0.01K per tick
//!         converged: maths.abs(mantle.temperature - prev) < 0.01<K>
//!         max_iterations: 1000
//!         on_timeout: fault  // Halt if not converged after 1000 ticks
//!     }
//! }
//! ```
//!
//! # Execution Flow
//!
//! ```text
//! CollectConfig → Initialize → WarmUp (loop until converged)
//!                                 ↓
//!                          Configure → Collect → Resolve → Fracture → Measure → Assert
//!                                 ↓
//!                          Evaluate converged predicate
//!                                 ↓
//!                          true → Continue to main simulation
//!                          false → Repeat tick phases (up to max_iterations)
//! ```

use super::TypedExpr;

/// WarmUp phase policy
///
/// Defines how the WarmUp phase determines convergence and handles timeout.
/// WarmUp runs tick phases repeatedly until the convergence predicate evaluates
/// to `true` or `max_iterations` is reached.
///
/// # Fields
///
/// - **converged**: Boolean expression evaluated after each tick to check convergence
/// - **max_iterations**: Maximum number of ticks before forced termination
/// - **on_timeout**: Behavior when `max_iterations` is reached without convergence
///
/// # Convergence Predicate
///
/// The `converged` expression:
/// - Must evaluate to [`Type::Bool`](crate::foundation::Type::Bool)
/// - Evaluated after each tick (after Assert phase)
/// - Has access to all resolved signals and `prev` values
/// - Should compare current state with previous state to detect stability
///
/// # Determinism Guarantee
///
/// Given the same world + scenario + seed, WarmUp:
/// - Always runs the same number of iterations
/// - Always produces identical final state
/// - Iteration count is part of the execution trace
///
/// # Examples
///
/// ## Thermal Equilibrium
///
/// ```cdsl
/// warmup {
///     converged: maths.abs(mantle.temperature - prev) < 0.01<K>
///     max_iterations: 1000
///     on_timeout: fault
/// }
/// ```
///
/// ## Orbital Settling
///
/// ```cdsl
/// warmup {
///     converged: all(bodies, |b| {
///         vector.norm(b.velocity - b.prev.velocity) < 1.0<m/s>
///     })
///     max_iterations: 500
///     on_timeout: warn
/// }
/// ```
///
/// ## Multi-Condition Convergence
///
/// ```cdsl
/// warmup {
///     converged: {
///         let temp_stable = maths.abs(core.temp - prev) < 0.1<K>
///         let pressure_stable = maths.abs(mantle.pressure - prev) < 1.0<Pa>
///         temp_stable && pressure_stable
///     }
///     max_iterations: 2000
///     on_timeout: fault
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WarmUpPolicy {
    /// Convergence predicate
    ///
    /// Boolean expression evaluated after each WarmUp tick. When this evaluates
    /// to `true`, WarmUp completes and the simulation proceeds to the main loop.
    ///
    /// # Type Requirements
    ///
    /// - Must evaluate to [`Type::Bool`](crate::foundation::Type::Bool)
    /// - Type checking ensures this at compile time
    ///
    /// # Available Context
    ///
    /// The predicate can access:
    /// - All resolved signals (current tick values)
    /// - `prev` values (previous tick or initial values)
    /// - Config and const values
    ///
    /// # Common Patterns
    ///
    /// - **Delta threshold:** `maths.abs(signal - prev) < threshold`
    /// - **Aggregate stability:** `all(entities, |e| condition)`
    /// - **Multiple conditions:** `cond1 && cond2 && cond3`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// // Simple delta check
    /// converged: maths.abs(temperature - prev) < 0.01<K>
    ///
    /// // Vector magnitude check
    /// converged: vector.norm(velocity - prev) < 0.1<m/s>
    ///
    /// // Aggregate over entities
    /// converged: all(plates, |p| p.stress < 1e6<Pa>)
    /// ```
    pub converged: TypedExpr,

    /// Maximum WarmUp iterations
    ///
    /// Safety limit to prevent infinite loops. If WarmUp runs this many ticks
    /// without `converged` evaluating to `true`, behavior is determined by
    /// `on_timeout`.
    ///
    /// # Choosing a Value
    ///
    /// - **Fast settling:** 100-500 iterations
    /// - **Moderate settling:** 500-2000 iterations
    /// - **Slow settling:** 2000+ iterations
    ///
    /// Set high enough that normal convergence completes, but low enough to
    /// catch divergence or oscillation bugs.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// max_iterations: 1000  // Typical for thermal settling
    /// max_iterations: 500   // Fast mechanical equilibrium
    /// max_iterations: 5000  // Slow climate stabilization
    /// ```
    pub max_iterations: u32,

    /// Timeout behavior
    ///
    /// Determines what happens when `max_iterations` is reached without convergence.
    ///
    /// # Policy Decision
    ///
    /// - **Fault:** Use when convergence is required for correctness
    /// - **Warn:** Use when approximate convergence is acceptable
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// on_timeout: fault  // Halt simulation if not converged
    /// on_timeout: warn   // Continue with warning if not converged
    /// ```
    pub on_timeout: WarmUpTimeout,
}

impl WarmUpPolicy {
    /// Create a new WarmUp policy
    ///
    /// # Parameters
    ///
    /// - `converged`: Boolean expression for convergence check
    /// - `max_iterations`: Maximum number of WarmUp ticks
    /// - `on_timeout`: Behavior when max_iterations is reached
    ///
    /// # Returns
    ///
    /// A WarmUp policy ready for use in world configuration.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let policy = WarmUpPolicy::new(
    ///     converged_expr,
    ///     1000,
    ///     WarmUpTimeout::Fault,
    /// );
    /// ```
    pub fn new(converged: TypedExpr, max_iterations: u32, on_timeout: WarmUpTimeout) -> Self {
        Self {
            converged,
            max_iterations,
            on_timeout,
        }
    }
}

/// WarmUp timeout behavior
///
/// Determines what happens when WarmUp reaches `max_iterations` without
/// the `converged` predicate evaluating to `true`.
///
/// # Determinism
///
/// The timeout behavior itself is deterministic - given the same world + scenario + seed,
/// the timeout will occur at the same iteration. However, the *consequences* differ:
///
/// - **Fault:** Deterministically halts with a fatal error
/// - **Warn:** Deterministically continues (with a warning in the log)
///
/// # Examples
///
/// ```cdsl
/// // Strict convergence required
/// warmup {
///     converged: condition
///     max_iterations: 1000
///     on_timeout: fault  // Halt if not converged
/// }
///
/// // Best-effort convergence
/// warmup {
///     converged: condition
///     max_iterations: 500
///     on_timeout: warn   // Continue with warning
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarmUpTimeout {
    /// Emit fatal fault and halt simulation
    ///
    /// Use this when convergence is **required for correctness**. If WarmUp
    /// doesn't converge, the simulation state is invalid and continuing would
    /// produce wrong results.
    ///
    /// # Behavior
    ///
    /// - Emits a structured fault with:
    ///   - Fault kind: `WarmUpTimeout`
    ///   - Iteration count: `max_iterations`
    ///   - Final predicate value (if available)
    /// - Halts simulation immediately
    /// - Returns error to caller
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// on_timeout: fault
    /// ```
    Fault,

    /// Emit warning and continue with current state
    ///
    /// Use this when approximate convergence is **acceptable**. WarmUp will
    /// stop at `max_iterations` and continue to the main simulation loop,
    /// even if not fully converged.
    ///
    /// # Behavior
    ///
    /// - Emits a structured warning with:
    ///   - Warning kind: `WarmUpTimeout`
    ///   - Iteration count: `max_iterations`
    ///   - Final predicate value (if available)
    /// - Continues to main simulation loop
    /// - Current state becomes initial state for main loop
    ///
    /// # Use Cases
    ///
    /// - Fast settling where "close enough" is fine
    /// - Debugging (allow continuation to inspect state)
    /// - Optional optimization (WarmUp improves but isn't required)
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// on_timeout: warn
    /// ```
    Warn,
}

impl WarmUpTimeout {
    /// Check if this timeout behavior halts the simulation
    ///
    /// # Returns
    ///
    /// - `true` for `Fault` (halts simulation)
    /// - `false` for `Warn` (continues simulation)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::ast::WarmUpTimeout;
    ///
    /// assert!(WarmUpTimeout::Fault.is_fatal());
    /// assert!(!WarmUpTimeout::Warn.is_fatal());
    /// ```
    pub fn is_fatal(self) -> bool {
        matches!(self, Self::Fault)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{KernelType, Shape, Span, Type, Unit};

    fn make_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn bool_expr() -> TypedExpr {
        TypedExpr::new(super::super::ExprKind::Prev, Type::Bool, make_span())
    }

    #[test]
    fn warmup_policy_creation() {
        let policy = WarmUpPolicy::new(bool_expr(), 1000, WarmUpTimeout::Fault);
        assert_eq!(policy.max_iterations, 1000);
        assert_eq!(policy.on_timeout, WarmUpTimeout::Fault);
    }

    #[test]
    fn warmup_timeout_is_fatal() {
        assert!(WarmUpTimeout::Fault.is_fatal());
        assert!(!WarmUpTimeout::Warn.is_fatal());
    }

    #[test]
    fn warmup_timeout_variants() {
        let fault = WarmUpTimeout::Fault;
        let warn = WarmUpTimeout::Warn;
        assert_ne!(fault, warn);
    }

    #[test]
    fn warmup_policy_clone() {
        let policy = WarmUpPolicy::new(bool_expr(), 500, WarmUpTimeout::Warn);
        let cloned = policy.clone();
        assert_eq!(policy.max_iterations, cloned.max_iterations);
        assert_eq!(policy.on_timeout, cloned.on_timeout);
    }

    #[test]
    fn warmup_policy_equality() {
        let policy1 = WarmUpPolicy::new(bool_expr(), 1000, WarmUpTimeout::Fault);
        let policy2 = WarmUpPolicy::new(bool_expr(), 1000, WarmUpTimeout::Fault);
        // Policies compare equal if fields match
        assert_eq!(policy1.max_iterations, policy2.max_iterations);
        assert_eq!(policy1.on_timeout, policy2.on_timeout);
    }
}
