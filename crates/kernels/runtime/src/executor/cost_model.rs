//! Cost model for lowering strategy selection.
//!
//! This module implements compile-time analysis of resolver bytecode to estimate
//! execution cost and select optimal lowering strategies (L1, L2, L3).
//!
//! # Cost Model Inputs
//!
//! ## Compile-Time Factors (from bytecode analysis)
//! - Number of arithmetic ops
//! - Number of function calls (with cost by function type)
//! - Presence of branches/conditionals
//! - Number of signal reads (cross-member lookups)
//! - Number of local variables
//!
//! ## Runtime Factors
//! - Population size
//! - Backend availability (GPU, SIMD)
//!
//! # Decision Matrix
//!
//! | Population | Complexity | Strategy |
//! |------------|------------|----------|
//! | >= 50k     | Low        | L2       |
//! | >= 50k     | High       | L1       |
//! | 2k - 50k   | Any        | L1       |
//! | <= 2k      | Low        | L1       |
//! | <= 2k      | High       | L3       |

use std::collections::HashMap;

use continuum_vm::{BytecodeChunk, Op};
use serde::{Deserialize, Serialize};

use super::LoweringStrategy;

// ============================================================================
// Complexity Score
// ============================================================================

/// Result of analyzing bytecode complexity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplexityScore {
    /// Total operation count
    pub op_count: usize,

    /// Number of arithmetic operations (add, sub, mul, div, pow, neg)
    pub arithmetic_ops: usize,

    /// Number of comparison operations
    pub comparison_ops: usize,

    /// Number of logical operations (and, or, not)
    pub logical_ops: usize,

    /// Number of branch points (if/else)
    pub branch_count: usize,

    /// Number of function calls
    pub call_count: usize,

    /// Number of signal reads
    pub signal_reads: usize,

    /// Number of local variables used
    pub local_count: usize,

    /// Estimated cost of function calls (expensive functions add more)
    pub call_cost: f64,

    /// Overall complexity score (computed from above)
    pub total_score: f64,
}

impl ComplexityScore {
    /// Create a new complexity score from bytecode analysis.
    pub fn from_bytecode(chunk: &BytecodeChunk, weights: &CostWeights) -> Self {
        let mut score = Self {
            local_count: chunk.local_count as usize,
            ..Default::default()
        };

        // Analyze each instruction
        for op in &chunk.ops {
            score.op_count += 1;

            match op {
                // Arithmetic
                Op::Add | Op::Sub | Op::Mul | Op::Div => {
                    score.arithmetic_ops += 1;
                }
                Op::Pow => {
                    score.arithmetic_ops += 1;
                    // Pow is more expensive
                    score.call_cost += weights.pow_penalty;
                }
                Op::Neg => {
                    score.arithmetic_ops += 1;
                }

                // Comparisons
                Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                    score.comparison_ops += 1;
                }

                // Logical
                Op::And | Op::Or | Op::Not => {
                    score.logical_ops += 1;
                }

                // Branches
                Op::JumpIfZero(_) => {
                    score.branch_count += 1;
                }
                Op::Jump(_) => {
                    // Unconditional jump doesn't add branch penalty
                }

                // Function calls
                Op::Call { kernel, arity: _ } => {
                    score.call_count += 1;
                    // Look up kernel cost
                    let kernel_name = chunk.kernels.get(*kernel as usize);
                    let cost = kernel_name
                        .map(|name| weights.kernel_cost(name))
                        .unwrap_or(weights.default_kernel_cost);
                    score.call_cost += cost;
                }

                // Signal reads
                Op::LoadSignal(_) | Op::LoadSignalComponent(_, _) => {
                    score.signal_reads += 1;
                }

                // Simple loads (low cost)
                Op::Const(_)
                | Op::LoadPrev
                | Op::LoadDt
                | Op::LoadInputs
                | Op::LoadConst(_)
                | Op::LoadConfig(_)
                | Op::LoadLocal(_) => {}

                // Stack operations
                Op::StoreLocal(_) | Op::Dup | Op::Pop => {}
            }
        }

        // Compute total score
        score.total_score = (score.arithmetic_ops as f64) * weights.arithmetic_weight
            + (score.comparison_ops as f64) * weights.comparison_weight
            + (score.logical_ops as f64) * weights.logical_weight
            + (score.branch_count as f64) * weights.branch_penalty
            + (score.signal_reads as f64) * weights.signal_read_penalty
            + score.call_cost;

        score
    }

    /// Check if this represents a "simple" expression suitable for L2 vectorization.
    ///
    /// Simple expressions have:
    /// - Few branches (< threshold)
    /// - Low call cost (no expensive trig functions)
    /// - Limited signal reads
    pub fn is_simple(&self, thresholds: &ComplexityThresholds) -> bool {
        self.branch_count <= thresholds.max_branches_for_simple
            && self.call_cost <= thresholds.max_call_cost_for_simple
            && self.signal_reads <= thresholds.max_signal_reads_for_simple
    }

    /// Check if this represents a "heavy" expression that benefits from L3.
    pub fn is_heavy(&self, thresholds: &ComplexityThresholds) -> bool {
        self.total_score >= thresholds.heavy_score_threshold
    }
}

// ============================================================================
// Cost Weights
// ============================================================================

/// Weights for computing complexity scores.
///
/// These can be tuned based on profiling results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostWeights {
    /// Weight for arithmetic operations
    pub arithmetic_weight: f64,

    /// Weight for comparison operations
    pub comparison_weight: f64,

    /// Weight for logical operations
    pub logical_weight: f64,

    /// Penalty for each branch point
    pub branch_penalty: f64,

    /// Penalty for each signal read (cross-member access)
    pub signal_read_penalty: f64,

    /// Extra cost for pow operations
    pub pow_penalty: f64,

    /// Default cost for unknown kernel functions
    pub default_kernel_cost: f64,

    /// Cost multipliers for specific kernel functions
    pub kernel_costs: HashMap<String, f64>,
}

impl Default for CostWeights {
    fn default() -> Self {
        let mut kernel_costs = HashMap::new();

        // Cheap functions (single instruction or small)
        kernel_costs.insert("abs".to_string(), 1.0);
        kernel_costs.insert("min".to_string(), 1.0);
        kernel_costs.insert("max".to_string(), 1.0);
        kernel_costs.insert("sign".to_string(), 1.0);
        kernel_costs.insert("floor".to_string(), 1.0);
        kernel_costs.insert("ceil".to_string(), 1.0);
        kernel_costs.insert("round".to_string(), 1.0);
        kernel_costs.insert("clamp".to_string(), 2.0);

        // Medium cost (standard library)
        kernel_costs.insert("sqrt".to_string(), 3.0);
        kernel_costs.insert("exp".to_string(), 5.0);
        kernel_costs.insert("ln".to_string(), 5.0);
        kernel_costs.insert("log".to_string(), 5.0);

        // Expensive functions (trig)
        kernel_costs.insert("sin".to_string(), 10.0);
        kernel_costs.insert("cos".to_string(), 10.0);
        kernel_costs.insert("tan".to_string(), 15.0);
        kernel_costs.insert("asin".to_string(), 15.0);
        kernel_costs.insert("acos".to_string(), 15.0);
        kernel_costs.insert("atan".to_string(), 12.0);
        kernel_costs.insert("atan2".to_string(), 15.0);

        // Very expensive (special functions, entity operations)
        kernel_costs.insert("other".to_string(), 50.0);
        kernel_costs.insert("pairs".to_string(), 100.0);
        kernel_costs.insert("sum_entities".to_string(), 20.0);
        kernel_costs.insert("avg_entities".to_string(), 25.0);

        Self {
            arithmetic_weight: 1.0,
            comparison_weight: 1.0,
            logical_weight: 0.5,
            branch_penalty: 5.0,
            signal_read_penalty: 3.0,
            pow_penalty: 8.0,
            default_kernel_cost: 5.0,
            kernel_costs,
        }
    }
}

impl CostWeights {
    /// Get the cost for a kernel function.
    pub fn kernel_cost(&self, name: &str) -> f64 {
        self.kernel_costs
            .get(name)
            .copied()
            .unwrap_or(self.default_kernel_cost)
    }
}

// ============================================================================
// Complexity Thresholds
// ============================================================================

/// Thresholds for classifying complexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityThresholds {
    /// Maximum branches for an expression to be considered "simple"
    pub max_branches_for_simple: usize,

    /// Maximum call cost for an expression to be considered "simple"
    pub max_call_cost_for_simple: f64,

    /// Maximum signal reads for an expression to be considered "simple"
    pub max_signal_reads_for_simple: usize,

    /// Minimum score to be considered "heavy"
    pub heavy_score_threshold: f64,
}

impl Default for ComplexityThresholds {
    fn default() -> Self {
        Self {
            // Any branch disqualifies "simple" - branches prevent vectorization
            max_branches_for_simple: 0,
            // Single trig function is OK, but multiple or expensive chains aren't
            max_call_cost_for_simple: 15.0,
            max_signal_reads_for_simple: 5,
            heavy_score_threshold: 100.0,
        }
    }
}

// ============================================================================
// Cost Model
// ============================================================================

/// Cost model for selecting lowering strategies.
///
/// Combines compile-time complexity analysis with runtime population size
/// to select the optimal lowering strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Weights for computing complexity scores
    pub weights: CostWeights,

    /// Thresholds for classifying complexity
    pub thresholds: ComplexityThresholds,

    /// Population threshold for L2 (vector kernel)
    pub l2_population_threshold: usize,

    /// Population threshold for L3 (sub-DAG)
    pub l3_population_threshold: usize,

    /// Whether L2 backend is available
    pub l2_available: bool,

    /// Whether L3 backend is available
    pub l3_available: bool,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            weights: CostWeights::default(),
            thresholds: ComplexityThresholds::default(),
            l2_population_threshold: 50_000,
            l3_population_threshold: 2_000,
            l2_available: true,
            l3_available: true,
        }
    }
}

impl CostModel {
    /// Create a new cost model with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cost model with custom population thresholds.
    pub fn with_thresholds(
        l2_threshold: usize,
        l3_threshold: usize,
        l2_available: bool,
        l3_available: bool,
    ) -> Self {
        Self {
            l2_population_threshold: l2_threshold,
            l3_population_threshold: l3_threshold,
            l2_available,
            l3_available,
            ..Default::default()
        }
    }

    /// Analyze bytecode and compute complexity score.
    pub fn analyze(&self, bytecode: &BytecodeChunk) -> ComplexityScore {
        ComplexityScore::from_bytecode(bytecode, &self.weights)
    }

    /// Select the optimal lowering strategy based on complexity and population.
    ///
    /// The decision follows this matrix:
    ///
    /// | Population    | Complexity | Strategy |
    /// |---------------|------------|----------|
    /// | >= L2_thresh  | Simple     | L2       |
    /// | >= L2_thresh  | Complex    | L1       |
    /// | mid-range     | Any        | L1       |
    /// | <= L3_thresh  | Heavy      | L3       |
    /// | <= L3_thresh  | Light      | L1       |
    pub fn select_strategy(
        &self,
        complexity: &ComplexityScore,
        population: usize,
    ) -> LoweringStrategy {
        // Large population: prefer L2 for simple expressions
        if population >= self.l2_population_threshold {
            if self.l2_available && complexity.is_simple(&self.thresholds) {
                return LoweringStrategy::VectorKernel;
            }
            // Complex expressions fall through to L1
            return LoweringStrategy::InstanceParallel;
        }

        // Small population: consider L3 for heavy expressions
        if population <= self.l3_population_threshold {
            if self.l3_available && complexity.is_heavy(&self.thresholds) {
                return LoweringStrategy::SubDag;
            }
        }

        // Default: L1 (instance-parallel) works well for mid-range
        LoweringStrategy::InstanceParallel
    }

    /// Convenience method: analyze bytecode and select strategy in one call.
    pub fn select_for_bytecode(
        &self,
        bytecode: &BytecodeChunk,
        population: usize,
    ) -> (ComplexityScore, LoweringStrategy) {
        let complexity = self.analyze(bytecode);
        let strategy = self.select_strategy(&complexity, population);
        (complexity, strategy)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_vm::compiler::{compile_expr, BinaryOp, Expr};

    fn compile_simple_expr() -> BytecodeChunk {
        // prev + dt
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Prev),
            right: Box::new(Expr::DtRaw),
        };
        compile_expr(&expr)
    }

    fn compile_complex_expr() -> BytecodeChunk {
        // if prev > 0 then sin(prev) * 2 else -prev
        let expr = Expr::If {
            condition: Box::new(Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Prev),
                right: Box::new(Expr::Literal(0.0)),
            }),
            then_branch: Box::new(Expr::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expr::Call {
                    function: "sin".to_string(),
                    args: vec![Expr::Prev],
                }),
                right: Box::new(Expr::Literal(2.0)),
            }),
            else_branch: Box::new(Expr::Unary {
                op: continuum_vm::compiler::UnaryOp::Neg,
                operand: Box::new(Expr::Prev),
            }),
        };
        compile_expr(&expr)
    }

    #[test]
    fn test_complexity_simple() {
        let bytecode = compile_simple_expr();
        let weights = CostWeights::default();
        let score = ComplexityScore::from_bytecode(&bytecode, &weights);

        assert_eq!(score.arithmetic_ops, 1); // Add
        assert_eq!(score.branch_count, 0);
        assert_eq!(score.call_count, 0);
        assert!(score.total_score < 10.0);
    }

    #[test]
    fn test_complexity_complex() {
        let bytecode = compile_complex_expr();
        let weights = CostWeights::default();
        let score = ComplexityScore::from_bytecode(&bytecode, &weights);

        assert!(score.branch_count >= 1); // if/else
        assert_eq!(score.call_count, 1); // sin
        assert!(score.total_score > 10.0);
    }

    #[test]
    fn test_is_simple() {
        let bytecode = compile_simple_expr();
        let weights = CostWeights::default();
        let thresholds = ComplexityThresholds::default();
        let score = ComplexityScore::from_bytecode(&bytecode, &weights);

        assert!(score.is_simple(&thresholds));
    }

    #[test]
    fn test_is_not_simple_with_branches() {
        let bytecode = compile_complex_expr();
        let weights = CostWeights::default();
        let thresholds = ComplexityThresholds::default();
        let score = ComplexityScore::from_bytecode(&bytecode, &weights);

        // Has branches and expensive call, not simple
        assert!(!score.is_simple(&thresholds));
    }

    #[test]
    fn test_cost_model_l2_for_large_simple() {
        let model = CostModel::default();
        let bytecode = compile_simple_expr();
        let score = model.analyze(&bytecode);

        // Large population + simple expr = L2
        let strategy = model.select_strategy(&score, 100_000);
        assert_eq!(strategy, LoweringStrategy::VectorKernel);
    }

    #[test]
    fn test_cost_model_l1_for_large_complex() {
        let model = CostModel::default();
        let bytecode = compile_complex_expr();
        let score = model.analyze(&bytecode);

        // Large population + complex expr = L1
        let strategy = model.select_strategy(&score, 100_000);
        assert_eq!(strategy, LoweringStrategy::InstanceParallel);
    }

    #[test]
    fn test_cost_model_l1_for_midrange() {
        let model = CostModel::default();
        let bytecode = compile_simple_expr();
        let score = model.analyze(&bytecode);

        // Mid-range population = L1
        let strategy = model.select_strategy(&score, 10_000);
        assert_eq!(strategy, LoweringStrategy::InstanceParallel);
    }

    #[test]
    fn test_cost_model_l3_for_small_heavy() {
        // Use default model with L3 enabled and low heavy threshold
        let model = CostModel {
            thresholds: ComplexityThresholds {
                heavy_score_threshold: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let bytecode = compile_complex_expr();
        let score = ComplexityScore::from_bytecode(&bytecode, &model.weights);

        // Complex expression should be heavy with low threshold
        assert!(score.is_heavy(&model.thresholds));

        // Small population + heavy expr + L3 available = L3
        let strategy = model.select_strategy(&score, 500);
        assert_eq!(strategy, LoweringStrategy::SubDag);
    }

    #[test]
    fn test_kernel_cost_lookup() {
        let weights = CostWeights::default();

        assert_eq!(weights.kernel_cost("abs"), 1.0);
        assert_eq!(weights.kernel_cost("sin"), 10.0);
        assert_eq!(weights.kernel_cost("pairs"), 100.0);
        assert_eq!(weights.kernel_cost("unknown"), 5.0); // default
    }

    #[test]
    fn test_select_for_bytecode() {
        let model = CostModel::default();
        let bytecode = compile_simple_expr();

        let (score, strategy) = model.select_for_bytecode(&bytecode, 100_000);

        assert!(score.total_score > 0.0);
        assert_eq!(strategy, LoweringStrategy::VectorKernel);
    }
}
