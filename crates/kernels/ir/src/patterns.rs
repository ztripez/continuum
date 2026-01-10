//! Signal Pattern Batching for L2 Vectorized Execution
//!
//! This module implements expression pattern extraction and signal grouping
//! for L2 lowering. Signals with identical expression patterns are batched
//! together for SIMD-friendly vectorized execution.
//!
//! # Overview
//!
//! The pattern batching process:
//!
//! 1. **Pattern Extraction**: Analyze signal resolve expressions to extract
//!    canonical patterns like `prev + collected` or `clamp(prev + collected, min, max)`
//!
//! 2. **Signal Grouping**: Group signals by:
//!    - Same stratum
//!    - Same expression pattern
//!    - Same value type (f64, Vec3, etc.)
//!
//! 3. **Batched Execution**: Generate SIMD-vectorized resolvers that process
//!    all signals in a group together
//!
//! # Pattern Coverage
//!
//! | Pattern | Estimated Coverage | Vectorization Benefit |
//! |---------|-------------------|----------------------|
//! | `prev + collected` | 30-40% | HIGH |
//! | `clamp(prev + collected, a, b)` | 20-30% | HIGH |
//! | `decay(prev, h) + collected` | 10-15% | MEDIUM |
//! | Linear transforms | 5-10% | HIGH |
//! | Complex/unique | 10-20% | None (fallback) |
//!
//! # Determinism
//!
//! Batched resolution produces bitwise-identical results to sequential resolution.
//! Signals within a batch are processed in stable ID order.
//!
//! # L2 Kernel Generation
//!
//! The module provides functions to generate L2 kernels from compiled expressions:
//!
//! - [`generate_l2_kernel`]: Creates a ScalarL2Kernel from a member signal's resolve expression
//! - [`should_use_l2`]: Heuristic to determine if L2 is beneficial for a given pattern/population

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use indexmap::IndexMap;

use continuum_foundation::{SignalId, StratumId};
use continuum_runtime::types::EntityId;
use continuum_runtime::vectorized::MemberSignalId;

use crate::ssa::lower_to_ssa;
use crate::vectorized::ScalarL2Kernel;
use crate::{BinaryOpIr, CompiledExpr, CompiledWorld, DtRobustOperator, ValueType};

/// Minimum batch size for SIMD vectorization (matches typical SIMD width).
///
/// Groups smaller than this threshold fall back to standard per-signal resolution.
pub const MIN_BATCH_SIZE: usize = 4;

/// A canonical expression pattern that can be batched.
///
/// Patterns represent the *shape* of an expression, abstracting away
/// concrete parameter values. Signals with the same pattern shape can
/// be executed together with different parameter values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExpressionPattern {
    /// Simple accumulator: `prev + collected`
    ///
    /// This is the most common pattern, representing additive accumulation.
    SimpleAccumulator,

    /// Clamped accumulator: `clamp(prev + collected, min, max)`
    ///
    /// Accumulation with bounds checking, common for bounded quantities.
    ClampedAccumulator {
        /// Whether a minimum bound is applied.
        has_min: bool,
        /// Whether a maximum bound is applied.
        has_max: bool,
    },

    /// Decay accumulator: `decay(prev, halflife) + collected`
    ///
    /// Exponential decay with optional additive input.
    DecayAccumulator {
        /// Whether collected values are added after decay.
        has_collected: bool,
    },

    /// Linear transform: `a * prev + b * collected + c`
    ///
    /// General linear combination of prev and collected.
    LinearTransform,

    /// Simple decay: `decay(prev, halflife)`
    ///
    /// Pure exponential decay without addition.
    SimpleDecay,

    /// Relaxation toward target: `relax(prev, target, tau)`
    ///
    /// Exponential approach to a target value.
    Relaxation,

    /// Integration: `integrate(prev, rate)`
    ///
    /// Euler integration of a rate.
    Integration,

    /// Prev-only passthrough: `prev`
    ///
    /// Signal value is unchanged, just carried forward.
    Passthrough,

    /// Constant value (no prev dependency).
    Constant,

    /// Unique pattern that doesn't match any canonical form.
    ///
    /// The hash is computed from the expression structure for grouping
    /// identical unique patterns together.
    Custom(u64),
}

impl ExpressionPattern {
    /// Returns true if this pattern supports SIMD batching.
    pub fn supports_batching(&self) -> bool {
        match self {
            ExpressionPattern::SimpleAccumulator
            | ExpressionPattern::ClampedAccumulator { .. }
            | ExpressionPattern::DecayAccumulator { .. }
            | ExpressionPattern::LinearTransform
            | ExpressionPattern::SimpleDecay
            | ExpressionPattern::Relaxation
            | ExpressionPattern::Integration
            | ExpressionPattern::Passthrough
            | ExpressionPattern::Constant => true,
            ExpressionPattern::Custom(_) => false,
        }
    }

    /// Returns the expected vectorization benefit level.
    pub fn vectorization_benefit(&self) -> VectorizationBenefit {
        match self {
            ExpressionPattern::SimpleAccumulator => VectorizationBenefit::High,
            ExpressionPattern::ClampedAccumulator { .. } => VectorizationBenefit::High,
            ExpressionPattern::LinearTransform => VectorizationBenefit::High,
            ExpressionPattern::Passthrough => VectorizationBenefit::High,
            ExpressionPattern::Constant => VectorizationBenefit::High,
            ExpressionPattern::DecayAccumulator { .. } => VectorizationBenefit::Medium,
            ExpressionPattern::SimpleDecay => VectorizationBenefit::Medium,
            ExpressionPattern::Relaxation => VectorizationBenefit::Medium,
            ExpressionPattern::Integration => VectorizationBenefit::Medium,
            ExpressionPattern::Custom(_) => VectorizationBenefit::None,
        }
    }
}

/// Expected benefit from vectorizing a pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorizationBenefit {
    /// High benefit: trivial SIMD operations (add, mul, clamp).
    High,
    /// Medium benefit: involves transcendental functions (exp, ln).
    Medium,
    /// No benefit: complex or unpredictable control flow.
    None,
}

/// A batch of signals that share the same expression pattern.
///
/// All signals in a batch can be executed together using SIMD
/// vectorized operations.
#[derive(Debug, Clone)]
pub struct SignalBatch {
    /// The canonical expression pattern shared by all signals.
    pub pattern: ExpressionPattern,
    /// Stratum binding (all signals in batch share this).
    pub stratum: StratumId,
    /// Value type (all signals in batch share this).
    pub value_type: ValueTypeCategory,
    /// Signal IDs in this batch, in stable sort order.
    pub signal_ids: Vec<SignalId>,
}

impl SignalBatch {
    /// Returns true if this batch meets the minimum size for SIMD benefit.
    pub fn is_vectorizable(&self) -> bool {
        self.signal_ids.len() >= MIN_BATCH_SIZE && self.pattern.supports_batching()
    }
}

/// Simplified value type category for batching.
///
/// Signals must have the same type category to be batched together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueTypeCategory {
    /// Scalar f64 value.
    Scalar,
    /// 2D vector.
    Vec2,
    /// 3D vector.
    Vec3,
    /// 4D vector.
    Vec4,
    /// Tensor (matrices).
    Tensor,
    /// Grid (2D array).
    Grid,
    /// Sequence.
    Seq,
}

impl From<&ValueType> for ValueTypeCategory {
    fn from(vt: &ValueType) -> Self {
        match vt {
            ValueType::Scalar { .. } => ValueTypeCategory::Scalar,
            ValueType::Vec2 { .. } => ValueTypeCategory::Vec2,
            ValueType::Vec3 { .. } => ValueTypeCategory::Vec3,
            ValueType::Vec4 { .. } => ValueTypeCategory::Vec4,
            ValueType::Tensor { .. } => ValueTypeCategory::Tensor,
            ValueType::Grid { .. } => ValueTypeCategory::Grid,
            ValueType::Seq { .. } => ValueTypeCategory::Seq,
        }
    }
}

/// A grouping key for batching signals together.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BatchKey {
    stratum: StratumId,
    pattern: ExpressionPattern,
    value_type: ValueTypeCategory,
}

/// Extract the expression pattern from a compiled expression.
///
/// This function analyzes the structure of a resolve expression and
/// returns a canonical pattern that describes its computational shape.
pub fn extract_pattern(expr: &CompiledExpr) -> ExpressionPattern {
    // Try to match known patterns in order of specificity
    if let Some(pattern) = try_match_clamped_accumulator(expr) {
        return pattern;
    }

    if let Some(pattern) = try_match_decay_accumulator(expr) {
        return pattern;
    }

    if let Some(pattern) = try_match_simple_accumulator(expr) {
        return pattern;
    }

    if let Some(pattern) = try_match_linear_transform(expr) {
        return pattern;
    }

    if let Some(pattern) = try_match_simple_decay(expr) {
        return pattern;
    }

    if let Some(pattern) = try_match_relaxation(expr) {
        return pattern;
    }

    if let Some(pattern) = try_match_integration(expr) {
        return pattern;
    }

    if try_match_passthrough(expr) {
        return ExpressionPattern::Passthrough;
    }

    if try_match_constant(expr) {
        return ExpressionPattern::Constant;
    }

    // No known pattern matched - compute hash for unique pattern grouping
    ExpressionPattern::Custom(compute_expr_hash(expr))
}

/// Try to match `prev + collected` pattern.
fn try_match_simple_accumulator(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    match expr {
        CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left,
            right,
        } => {
            let left_is_prev = matches!(left.as_ref(), CompiledExpr::Prev);
            let right_is_collected = matches!(right.as_ref(), CompiledExpr::Collected);
            let left_is_collected = matches!(left.as_ref(), CompiledExpr::Collected);
            let right_is_prev = matches!(right.as_ref(), CompiledExpr::Prev);

            if (left_is_prev && right_is_collected) || (left_is_collected && right_is_prev) {
                return Some(ExpressionPattern::SimpleAccumulator);
            }
            None
        }
        _ => None,
    }
}

/// Try to match `clamp(prev + collected, min, max)` pattern.
fn try_match_clamped_accumulator(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    match expr {
        CompiledExpr::KernelCall { function, args } if function == "clamp" && args.len() == 3 => {
            // Check if first arg is prev + collected
            if try_match_simple_accumulator(&args[0]).is_some() {
                let has_min = !matches!(&args[1], CompiledExpr::Literal(v) if *v == f64::NEG_INFINITY);
                let has_max = !matches!(&args[2], CompiledExpr::Literal(v) if *v == f64::INFINITY);
                return Some(ExpressionPattern::ClampedAccumulator { has_min, has_max });
            }
            None
        }
        CompiledExpr::Call { function, args } if function == "clamp" && args.len() == 3 => {
            if try_match_simple_accumulator(&args[0]).is_some() {
                let has_min = !matches!(&args[1], CompiledExpr::Literal(v) if *v == f64::NEG_INFINITY);
                let has_max = !matches!(&args[2], CompiledExpr::Literal(v) if *v == f64::INFINITY);
                return Some(ExpressionPattern::ClampedAccumulator { has_min, has_max });
            }
            None
        }
        _ => None,
    }
}

/// Try to match `decay(prev, halflife) + collected` pattern.
fn try_match_decay_accumulator(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    match expr {
        CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left,
            right,
        } => {
            let decay_on_left = matches!(
                left.as_ref(),
                CompiledExpr::DtRobustCall { operator: DtRobustOperator::Decay, .. }
            );
            let collected_on_right = matches!(right.as_ref(), CompiledExpr::Collected);

            let decay_on_right = matches!(
                right.as_ref(),
                CompiledExpr::DtRobustCall { operator: DtRobustOperator::Decay, .. }
            );
            let collected_on_left = matches!(left.as_ref(), CompiledExpr::Collected);

            if (decay_on_left && collected_on_right) || (decay_on_right && collected_on_left) {
                return Some(ExpressionPattern::DecayAccumulator {
                    has_collected: true,
                });
            }
            None
        }
        _ => None,
    }
}

/// Try to match `decay(prev, halflife)` pattern (without addition).
fn try_match_simple_decay(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    match expr {
        CompiledExpr::DtRobustCall {
            operator: DtRobustOperator::Decay,
            args,
            ..
        } if args.len() >= 2 => {
            if matches!(args[0], CompiledExpr::Prev) {
                return Some(ExpressionPattern::SimpleDecay);
            }
            None
        }
        _ => None,
    }
}

/// Try to match `relax(current, target, tau)` pattern.
fn try_match_relaxation(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    match expr {
        CompiledExpr::DtRobustCall {
            operator: DtRobustOperator::Relax | DtRobustOperator::Smooth,
            args,
            ..
        } if args.len() >= 2 => {
            if matches!(args[0], CompiledExpr::Prev) {
                return Some(ExpressionPattern::Relaxation);
            }
            None
        }
        _ => None,
    }
}

/// Try to match `integrate(prev, rate)` pattern.
fn try_match_integration(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    match expr {
        CompiledExpr::DtRobustCall {
            operator: DtRobustOperator::Integrate,
            args,
            ..
        } if args.len() >= 2 => {
            if matches!(args[0], CompiledExpr::Prev) {
                return Some(ExpressionPattern::Integration);
            }
            None
        }
        _ => None,
    }
}

/// Try to match `a * prev + b * collected + c` linear transform pattern.
fn try_match_linear_transform(expr: &CompiledExpr) -> Option<ExpressionPattern> {
    // Check for expressions of the form:
    // - (a * prev) + (b * collected)
    // - (a * prev) + c
    // - prev * a + collected * b
    // etc.

    fn is_scaled_prev(e: &CompiledExpr) -> bool {
        matches!(
            e,
            CompiledExpr::Binary {
                op: BinaryOpIr::Mul,
                left,
                right
            } if matches!(left.as_ref(), CompiledExpr::Prev | CompiledExpr::Literal(_))
                && matches!(right.as_ref(), CompiledExpr::Prev | CompiledExpr::Literal(_))
        )
    }

    fn is_scaled_collected(e: &CompiledExpr) -> bool {
        matches!(
            e,
            CompiledExpr::Binary {
                op: BinaryOpIr::Mul,
                left,
                right
            } if matches!(left.as_ref(), CompiledExpr::Collected | CompiledExpr::Literal(_))
                && matches!(right.as_ref(), CompiledExpr::Collected | CompiledExpr::Literal(_))
        )
    }

    match expr {
        CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left,
            right,
        } => {
            let left_is_linear = is_scaled_prev(left) || is_scaled_collected(left);
            let right_is_linear =
                is_scaled_prev(right) || is_scaled_collected(right) || matches!(right.as_ref(), CompiledExpr::Literal(_));

            if left_is_linear && right_is_linear {
                return Some(ExpressionPattern::LinearTransform);
            }

            // Check for nested add: (a * prev + b * collected) + c
            if let CompiledExpr::Binary {
                op: BinaryOpIr::Add,
                ..
            } = left.as_ref()
            {
                if try_match_linear_transform(left).is_some()
                    && matches!(right.as_ref(), CompiledExpr::Literal(_))
                {
                    return Some(ExpressionPattern::LinearTransform);
                }
            }

            None
        }
        _ => None,
    }
}

/// Try to match `prev` passthrough pattern.
fn try_match_passthrough(expr: &CompiledExpr) -> bool {
    matches!(expr, CompiledExpr::Prev)
}

/// Try to match constant (no prev dependency) pattern.
fn try_match_constant(expr: &CompiledExpr) -> bool {
    !expr_uses_prev(expr) && !expr_uses_collected(expr)
}

/// Check if an expression references `prev`.
fn expr_uses_prev(expr: &CompiledExpr) -> bool {
    match expr {
        CompiledExpr::Prev => true,
        CompiledExpr::Binary { left, right, .. } => {
            expr_uses_prev(left) || expr_uses_prev(right)
        }
        CompiledExpr::Unary { operand, .. } => expr_uses_prev(operand),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_uses_prev(condition)
                || expr_uses_prev(then_branch)
                || expr_uses_prev(else_branch)
        }
        CompiledExpr::Let { value, body, .. } => expr_uses_prev(value) || expr_uses_prev(body),
        CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
            args.iter().any(expr_uses_prev)
        }
        CompiledExpr::DtRobustCall { args, .. } => args.iter().any(expr_uses_prev),
        CompiledExpr::FieldAccess { object, .. } => expr_uses_prev(object),
        CompiledExpr::Aggregate { body, .. } => expr_uses_prev(body),
        CompiledExpr::Filter { predicate, body, .. } => {
            expr_uses_prev(predicate) || expr_uses_prev(body)
        }
        CompiledExpr::Within {
            position,
            radius,
            body,
            ..
        } => expr_uses_prev(position) || expr_uses_prev(radius) || expr_uses_prev(body),
        CompiledExpr::Other { body, .. } | CompiledExpr::Pairs { body, .. } => expr_uses_prev(body),
        CompiledExpr::First { predicate, .. } => expr_uses_prev(predicate),
        CompiledExpr::Nearest { position, .. } => expr_uses_prev(position),
        _ => false,
    }
}

/// Check if an expression references `collected`.
fn expr_uses_collected(expr: &CompiledExpr) -> bool {
    match expr {
        CompiledExpr::Collected => true,
        CompiledExpr::Binary { left, right, .. } => {
            expr_uses_collected(left) || expr_uses_collected(right)
        }
        CompiledExpr::Unary { operand, .. } => expr_uses_collected(operand),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_uses_collected(condition)
                || expr_uses_collected(then_branch)
                || expr_uses_collected(else_branch)
        }
        CompiledExpr::Let { value, body, .. } => {
            expr_uses_collected(value) || expr_uses_collected(body)
        }
        CompiledExpr::Call { args, .. } | CompiledExpr::KernelCall { args, .. } => {
            args.iter().any(expr_uses_collected)
        }
        CompiledExpr::DtRobustCall { args, .. } => args.iter().any(expr_uses_collected),
        CompiledExpr::FieldAccess { object, .. } => expr_uses_collected(object),
        _ => false,
    }
}

/// Compute a structural hash of an expression for unique pattern grouping.
///
/// This hash captures the shape of the expression tree, not the specific
/// values. Expressions with the same structure but different literals
/// will have the same hash.
fn compute_expr_hash(expr: &CompiledExpr) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    hash_expr_structure(expr, &mut hasher);
    hasher.finish()
}

/// Hash the structure of an expression (not values).
fn hash_expr_structure<H: Hasher>(expr: &CompiledExpr, hasher: &mut H) {
    // Hash the discriminant to distinguish variants
    std::mem::discriminant(expr).hash(hasher);

    match expr {
        CompiledExpr::Literal(_) => {
            // Don't hash the value, just the fact that it's a literal
            "literal".hash(hasher);
        }
        CompiledExpr::Prev => "prev".hash(hasher),
        CompiledExpr::DtRaw => "dt_raw".hash(hasher),
        CompiledExpr::Collected => "collected".hash(hasher),
        CompiledExpr::Signal(id) => {
            "signal".hash(hasher);
            id.0.hash(hasher);
        }
        CompiledExpr::Const(name) => {
            "const".hash(hasher);
            name.hash(hasher);
        }
        CompiledExpr::Config(name) => {
            "config".hash(hasher);
            name.hash(hasher);
        }
        CompiledExpr::Binary { op, left, right } => {
            op.hash(hasher);
            hash_expr_structure(left, hasher);
            hash_expr_structure(right, hasher);
        }
        CompiledExpr::Unary { op, operand } => {
            op.hash(hasher);
            hash_expr_structure(operand, hasher);
        }
        CompiledExpr::Call { function, args } => {
            function.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                hash_expr_structure(arg, hasher);
            }
        }
        CompiledExpr::KernelCall { function, args } => {
            "kernel".hash(hasher);
            function.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                hash_expr_structure(arg, hasher);
            }
        }
        CompiledExpr::DtRobustCall {
            operator,
            args,
            method,
        } => {
            "dt_robust".hash(hasher);
            operator.hash(hasher);
            method.hash(hasher);
            args.len().hash(hasher);
            for arg in args {
                hash_expr_structure(arg, hasher);
            }
        }
        CompiledExpr::FieldAccess { object, field } => {
            "field_access".hash(hasher);
            field.hash(hasher);
            hash_expr_structure(object, hasher);
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            "if".hash(hasher);
            hash_expr_structure(condition, hasher);
            hash_expr_structure(then_branch, hasher);
            hash_expr_structure(else_branch, hasher);
        }
        CompiledExpr::Let { name, value, body } => {
            "let".hash(hasher);
            name.hash(hasher);
            hash_expr_structure(value, hasher);
            hash_expr_structure(body, hasher);
        }
        CompiledExpr::Local(name) => {
            "local".hash(hasher);
            name.hash(hasher);
        }
        CompiledExpr::SelfField(field) => {
            "self_field".hash(hasher);
            field.hash(hasher);
        }
        CompiledExpr::EntityAccess {
            entity,
            instance,
            field,
        } => {
            "entity_access".hash(hasher);
            entity.0.hash(hasher);
            instance.0.hash(hasher);
            field.hash(hasher);
        }
        CompiledExpr::Aggregate { op, entity, body } => {
            "aggregate".hash(hasher);
            op.hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(body, hasher);
        }
        CompiledExpr::Other { entity, body } => {
            "other".hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(body, hasher);
        }
        CompiledExpr::Pairs { entity, body } => {
            "pairs".hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(body, hasher);
        }
        CompiledExpr::Filter {
            entity,
            predicate,
            body,
        } => {
            "filter".hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(predicate, hasher);
            hash_expr_structure(body, hasher);
        }
        CompiledExpr::First { entity, predicate } => {
            "first".hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(predicate, hasher);
        }
        CompiledExpr::Nearest { entity, position } => {
            "nearest".hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(position, hasher);
        }
        CompiledExpr::Within {
            entity,
            position,
            radius,
            body,
        } => {
            "within".hash(hasher);
            entity.0.hash(hasher);
            hash_expr_structure(position, hasher);
            hash_expr_structure(radius, hasher);
            hash_expr_structure(body, hasher);
        }
    }
}

/// Group signals by their expression patterns.
///
/// This function analyzes all signals in a compiled world and groups them
/// into batches that can be executed together. Signals are grouped by:
/// - Same stratum
/// - Same expression pattern
/// - Same value type category
///
/// Returns batches sorted by stratum ID for deterministic scheduling.
pub fn group_signals_by_pattern(world: &CompiledWorld) -> Vec<SignalBatch> {
    let mut groups: IndexMap<BatchKey, Vec<SignalId>> = IndexMap::new();

    for (signal_id, signal) in &world.signals {
        // Skip signals without resolve expressions
        let Some(resolve) = &signal.resolve else {
            continue;
        };

        let pattern = extract_pattern(resolve);
        let value_type = ValueTypeCategory::from(&signal.value_type);

        let key = BatchKey {
            stratum: signal.stratum.clone(),
            pattern: pattern.clone(),
            value_type,
        };

        groups.entry(key).or_default().push(signal_id.clone());
    }

    // Convert to SignalBatch vec, sorting signals within each batch by ID
    let mut batches: Vec<SignalBatch> = groups
        .into_iter()
        .map(|(key, mut signal_ids)| {
            // Sort signals by ID for deterministic ordering
            signal_ids.sort_by(|a, b| a.0.cmp(&b.0));

            SignalBatch {
                pattern: key.pattern,
                stratum: key.stratum,
                value_type: key.value_type,
                signal_ids,
            }
        })
        .collect();

    // Sort batches by stratum for deterministic scheduling
    batches.sort_by(|a, b| a.stratum.0.cmp(&b.stratum.0));

    batches
}

/// Pattern coverage metrics for a compiled world.
#[derive(Debug, Clone, Default)]
pub struct PatternCoverage {
    /// Total number of signals analyzed.
    pub total_signals: usize,
    /// Signals matching each pattern type.
    pub pattern_counts: IndexMap<String, usize>,
    /// Number of signals in vectorizable batches (size >= MIN_BATCH_SIZE).
    pub vectorizable_signals: usize,
    /// Number of batches meeting the vectorization threshold.
    pub vectorizable_batches: usize,
    /// Total number of batches.
    pub total_batches: usize,
}

impl PatternCoverage {
    /// Calculate coverage percentage for vectorizable signals.
    pub fn vectorization_coverage(&self) -> f64 {
        if self.total_signals == 0 {
            0.0
        } else {
            (self.vectorizable_signals as f64 / self.total_signals as f64) * 100.0
        }
    }
}

/// Analyze pattern coverage for a compiled world.
///
/// Returns metrics about pattern distribution and vectorization potential.
pub fn analyze_pattern_coverage(world: &CompiledWorld) -> PatternCoverage {
    let batches = group_signals_by_pattern(world);

    let mut coverage = PatternCoverage::default();
    coverage.total_batches = batches.len();

    for batch in &batches {
        let pattern_name = format!("{:?}", batch.pattern);
        *coverage.pattern_counts.entry(pattern_name).or_insert(0) += batch.signal_ids.len();
        coverage.total_signals += batch.signal_ids.len();

        if batch.is_vectorizable() {
            coverage.vectorizable_signals += batch.signal_ids.len();
            coverage.vectorizable_batches += 1;
        }
    }

    coverage
}

// ============================================================================
// L2 Kernel Generation
// ============================================================================

/// Default population threshold above which L2 vectorized execution is preferred.
///
/// This value is tuned based on typical SIMD break-even points:
/// - Below this threshold, the overhead of L2 setup outweighs benefits
/// - Above this threshold, vectorized execution provides speedups
pub const L2_POPULATION_THRESHOLD: usize = 50_000;

/// Minimum population for L2 to be considered at all.
///
/// Very small populations should always use L1 for lower overhead.
pub const L2_MINIMUM_POPULATION: usize = 1000;

/// Determines if L2 vectorized execution should be used for a given pattern and population.
///
/// # Arguments
///
/// * `pattern` - The expression pattern extracted from the resolve expression
/// * `population` - Expected population size for the entity
///
/// # Returns
///
/// `true` if L2 execution is recommended, `false` for L1 (instance-parallel).
///
/// # Heuristics
///
/// L2 is preferred when:
/// 1. Population exceeds `L2_POPULATION_THRESHOLD` (50k)
/// 2. Pattern supports vectorization (not Custom)
/// 3. Pattern has High or Medium vectorization benefit
///
/// L2 is forced even for smaller populations (above L2_MINIMUM_POPULATION) when:
/// - Pattern has High vectorization benefit
pub fn should_use_l2(pattern: &ExpressionPattern, population: usize) -> bool {
    // Below minimum, always use L1
    if population < L2_MINIMUM_POPULATION {
        return false;
    }

    // Custom patterns don't benefit from vectorization
    if !pattern.supports_batching() {
        return false;
    }

    match pattern.vectorization_benefit() {
        VectorizationBenefit::High => {
            // High benefit: use L2 for populations above minimum threshold
            population >= L2_MINIMUM_POPULATION
        }
        VectorizationBenefit::Medium => {
            // Medium benefit: only use L2 for large populations
            population >= L2_POPULATION_THRESHOLD
        }
        VectorizationBenefit::None => {
            // No benefit from vectorization
            false
        }
    }
}

/// Generates an L2 vectorized kernel for a member signal's resolve expression.
///
/// This function:
/// 1. Lowers the expression to SSA IR
/// 2. Wraps it in a ScalarL2Kernel that implements LaneKernel
///
/// # Arguments
///
/// * `entity_id` - The entity this member belongs to
/// * `signal_name` - The name of the member signal
/// * `resolve_expr` - The resolve expression to compile
/// * `population_hint` - Expected population for capacity pre-allocation
///
/// # Returns
///
/// A `ScalarL2Kernel` that can be registered with the `LaneKernelRegistry`.
///
/// # Example
///
/// ```ignore
/// let kernel = generate_l2_kernel(
///     &entity_id,
///     "temperature",
///     &resolve_expr,
///     10000,
/// );
/// registry.register(kernel);
/// ```
pub fn generate_l2_kernel(
    entity_id: &EntityId,
    signal_name: &str,
    resolve_expr: &CompiledExpr,
    population_hint: usize,
) -> ScalarL2Kernel {
    // Lower the expression to SSA IR
    let ssa = lower_to_ssa(resolve_expr);

    // Create the member signal ID
    let member_signal_id = MemberSignalId::new(entity_id.clone(), signal_name.to_string());

    // Create the L2 kernel
    ScalarL2Kernel::new(
        member_signal_id,
        signal_name.to_string(),
        Arc::new(ssa),
        population_hint,
    )
}

/// Result of analyzing a member signal for L2 execution suitability.
#[derive(Debug, Clone)]
pub struct L2AnalysisResult {
    /// The expression pattern detected.
    pub pattern: ExpressionPattern,
    /// Whether L2 execution is recommended.
    pub use_l2: bool,
    /// Vectorization benefit level.
    pub benefit: VectorizationBenefit,
}

/// Analyzes a member signal's resolve expression for L2 execution suitability.
///
/// # Arguments
///
/// * `resolve_expr` - The resolve expression to analyze
/// * `population_hint` - Expected population size
///
/// # Returns
///
/// Analysis result containing pattern, L2 recommendation, and benefit level.
pub fn analyze_for_l2(resolve_expr: &CompiledExpr, population_hint: usize) -> L2AnalysisResult {
    let pattern = extract_pattern(resolve_expr);
    let benefit = pattern.vectorization_benefit();
    let use_l2 = should_use_l2(&pattern, population_hint);

    L2AnalysisResult {
        pattern,
        use_l2,
        benefit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_accumulator() {
        // prev + collected
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Collected),
        };

        assert_eq!(extract_pattern(&expr), ExpressionPattern::SimpleAccumulator);

        // collected + prev (order shouldn't matter)
        let expr2 = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Collected),
            right: Box::new(CompiledExpr::Prev),
        };

        assert_eq!(
            extract_pattern(&expr2),
            ExpressionPattern::SimpleAccumulator
        );
    }

    #[test]
    fn test_extract_clamped_accumulator() {
        // clamp(prev + collected, 0.0, 100.0)
        let expr = CompiledExpr::KernelCall {
            function: "clamp".to_string(),
            args: vec![
                CompiledExpr::Binary {
                    op: BinaryOpIr::Add,
                    left: Box::new(CompiledExpr::Prev),
                    right: Box::new(CompiledExpr::Collected),
                },
                CompiledExpr::Literal(0.0),
                CompiledExpr::Literal(100.0),
            ],
        };

        assert_eq!(
            extract_pattern(&expr),
            ExpressionPattern::ClampedAccumulator {
                has_min: true,
                has_max: true
            }
        );
    }

    #[test]
    fn test_extract_decay_accumulator() {
        // decay(prev, 1000.0) + collected
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::DtRobustCall {
                operator: DtRobustOperator::Decay,
                args: vec![CompiledExpr::Prev, CompiledExpr::Literal(1000.0)],
                method: crate::IntegrationMethod::Euler,
            }),
            right: Box::new(CompiledExpr::Collected),
        };

        assert_eq!(
            extract_pattern(&expr),
            ExpressionPattern::DecayAccumulator {
                has_collected: true
            }
        );
    }

    #[test]
    fn test_extract_simple_decay() {
        // decay(prev, 1000.0)
        let expr = CompiledExpr::DtRobustCall {
            operator: DtRobustOperator::Decay,
            args: vec![CompiledExpr::Prev, CompiledExpr::Literal(1000.0)],
            method: crate::IntegrationMethod::Euler,
        };

        assert_eq!(extract_pattern(&expr), ExpressionPattern::SimpleDecay);
    }

    #[test]
    fn test_extract_integration() {
        // integrate(prev, rate)
        let expr = CompiledExpr::DtRobustCall {
            operator: DtRobustOperator::Integrate,
            args: vec![
                CompiledExpr::Prev,
                CompiledExpr::Signal(SignalId::from("rate")),
            ],
            method: crate::IntegrationMethod::Euler,
        };

        assert_eq!(extract_pattern(&expr), ExpressionPattern::Integration);
    }

    #[test]
    fn test_extract_passthrough() {
        let expr = CompiledExpr::Prev;
        assert_eq!(extract_pattern(&expr), ExpressionPattern::Passthrough);
    }

    #[test]
    fn test_extract_constant() {
        let expr = CompiledExpr::Literal(42.0);
        assert_eq!(extract_pattern(&expr), ExpressionPattern::Constant);

        // Binary op on constants
        let expr2 = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Literal(1.0)),
            right: Box::new(CompiledExpr::Literal(2.0)),
        };
        assert_eq!(extract_pattern(&expr2), ExpressionPattern::Constant);
    }

    #[test]
    fn test_extract_custom_pattern() {
        // Some complex expression that doesn't match known patterns
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Mul,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Signal(SignalId::from("multiplier"))),
        };

        let pattern = extract_pattern(&expr);
        assert!(matches!(pattern, ExpressionPattern::Custom(_)));
    }

    #[test]
    fn test_pattern_supports_batching() {
        assert!(ExpressionPattern::SimpleAccumulator.supports_batching());
        assert!(ExpressionPattern::ClampedAccumulator {
            has_min: true,
            has_max: true
        }
        .supports_batching());
        assert!(ExpressionPattern::Passthrough.supports_batching());
        assert!(!ExpressionPattern::Custom(12345).supports_batching());
    }

    #[test]
    fn test_expr_hash_same_structure() {
        // Two expressions with same structure but different values should have same hash
        let expr1 = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        };

        let expr2 = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(999.0)),
        };

        assert_eq!(compute_expr_hash(&expr1), compute_expr_hash(&expr2));
    }

    #[test]
    fn test_expr_hash_different_structure() {
        let expr1 = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        };

        let expr2 = CompiledExpr::Binary {
            op: BinaryOpIr::Mul, // Different op
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Literal(1.0)),
        };

        assert_ne!(compute_expr_hash(&expr1), compute_expr_hash(&expr2));
    }

    // ========================================================================
    // L2 Kernel Generation Tests
    // ========================================================================

    #[test]
    fn test_should_use_l2_below_minimum() {
        // Below L2_MINIMUM_POPULATION (1000), should always return false
        let pattern = ExpressionPattern::SimpleAccumulator;
        assert!(!should_use_l2(&pattern, 0));
        assert!(!should_use_l2(&pattern, 500));
        assert!(!should_use_l2(&pattern, 999));
    }

    #[test]
    fn test_should_use_l2_high_benefit() {
        // High benefit patterns should use L2 above minimum threshold
        let pattern = ExpressionPattern::SimpleAccumulator;
        assert_eq!(pattern.vectorization_benefit(), VectorizationBenefit::High);

        // At minimum threshold
        assert!(should_use_l2(&pattern, L2_MINIMUM_POPULATION));

        // Above minimum
        assert!(should_use_l2(&pattern, 5000));
        assert!(should_use_l2(&pattern, L2_POPULATION_THRESHOLD));
        assert!(should_use_l2(&pattern, 100_000));
    }

    #[test]
    fn test_should_use_l2_medium_benefit() {
        // Medium benefit patterns need larger population
        // DecayAccumulator is a Medium benefit pattern
        let pattern = ExpressionPattern::DecayAccumulator { has_collected: true };
        assert_eq!(pattern.vectorization_benefit(), VectorizationBenefit::Medium);

        // Below threshold: should not use L2
        assert!(!should_use_l2(&pattern, L2_MINIMUM_POPULATION));
        assert!(!should_use_l2(&pattern, 10_000));
        assert!(!should_use_l2(&pattern, 49_999));

        // At and above threshold: should use L2
        assert!(should_use_l2(&pattern, L2_POPULATION_THRESHOLD));
        assert!(should_use_l2(&pattern, 100_000));
    }

    #[test]
    fn test_should_use_l2_custom_pattern() {
        // Custom patterns don't support batching
        let pattern = ExpressionPattern::Custom(12345);
        assert!(!pattern.supports_batching());
        assert_eq!(pattern.vectorization_benefit(), VectorizationBenefit::None);

        // Should never use L2 regardless of population
        assert!(!should_use_l2(&pattern, L2_MINIMUM_POPULATION));
        assert!(!should_use_l2(&pattern, L2_POPULATION_THRESHOLD));
        assert!(!should_use_l2(&pattern, 1_000_000));
    }

    #[test]
    fn test_should_use_l2_constant_pattern() {
        // Constant patterns have High benefit (trivial copy operations)
        let pattern = ExpressionPattern::Constant;
        assert_eq!(pattern.vectorization_benefit(), VectorizationBenefit::High);

        // Above minimum threshold: should use L2
        assert!(should_use_l2(&pattern, L2_MINIMUM_POPULATION));
        assert!(should_use_l2(&pattern, L2_POPULATION_THRESHOLD));
        assert!(should_use_l2(&pattern, 1_000_000));
    }

    #[test]
    fn test_analyze_for_l2_simple_accumulator() {
        // Simple accumulator with large population should recommend L2
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Collected),
        };

        let result = analyze_for_l2(&expr, 100_000);

        assert_eq!(result.pattern, ExpressionPattern::SimpleAccumulator);
        assert_eq!(result.benefit, VectorizationBenefit::High);
        assert!(result.use_l2);
    }

    #[test]
    fn test_analyze_for_l2_small_population() {
        // Same pattern but small population should not recommend L2
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Collected),
        };

        let result = analyze_for_l2(&expr, 500);

        assert_eq!(result.pattern, ExpressionPattern::SimpleAccumulator);
        assert_eq!(result.benefit, VectorizationBenefit::High);
        assert!(!result.use_l2); // Population too small
    }

    #[test]
    fn test_analyze_for_l2_decay() {
        // Decay pattern with collected has Medium benefit (involves transcendentals)
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::DtRobustCall {
                operator: DtRobustOperator::Decay,
                args: vec![CompiledExpr::Prev, CompiledExpr::Literal(1000.0)],
                method: crate::IntegrationMethod::Euler,
            }),
            right: Box::new(CompiledExpr::Collected),
        };

        let result = analyze_for_l2(&expr, L2_POPULATION_THRESHOLD);

        assert_eq!(
            result.pattern,
            ExpressionPattern::DecayAccumulator { has_collected: true }
        );
        assert_eq!(result.benefit, VectorizationBenefit::Medium);
        assert!(result.use_l2); // Large population enables L2 for medium benefit
    }

    #[test]
    fn test_generate_l2_kernel() {
        use continuum_runtime::LaneKernel;

        // Test that we can generate an L2 kernel from an expression
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Collected),
        };

        let entity_id = EntityId("test_entity".to_string());
        let kernel = generate_l2_kernel(&entity_id, "temperature", &expr, 10_000);

        // Verify kernel properties via LaneKernel trait
        assert_eq!(kernel.member_signal_id().signal_name, "temperature");
        assert_eq!(kernel.population_hint(), 10_000);
    }
}
