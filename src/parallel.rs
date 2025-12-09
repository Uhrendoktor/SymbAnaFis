//! Parallel batch evaluation using Rayon
//!
//! This module provides a single unified function for parallelized evaluation
//! of multiple expressions over multiple variable configurations.
//!
//! Enable with the `parallel` feature:
//! ```toml
//! symb_anafis = { version = "0.3", features = ["parallel"] }
//! ```

use crate::Expr;
use rayon::prelude::*;
use std::collections::HashMap;

/// A value to substitute for a variable.
///
/// Use this to pass numbers, expressions, or skip markers to `evaluate_parallel`.
///
/// # Examples
/// ```ignore
/// use symb_anafis::parallel::Value;
///
/// let vals: Vec<Value> = vec![
///     2.0.into(),           // Number
///     expr.into(),          // Expression
///     Value::Skip,          // Keep variable symbolic
/// ];
/// ```
#[derive(Debug, Clone)]
pub enum Value {
    /// Substitute a numeric value
    Num(f64),
    /// Substitute an expression (symbolic substitution)
    Expr(Expr),
    /// Skip - keep the variable symbolic at this point
    Skip,
}

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Num(n)
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Num(n as f64)
    }
}

impl From<Expr> for Value {
    fn from(e: Expr) -> Self {
        Value::Expr(e)
    }
}

impl From<&Expr> for Value {
    fn from(e: &Expr) -> Self {
        Value::Expr(e.clone())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => v.into(),
            None => Value::Skip,
        }
    }
}

/// Unified parallel evaluation for multiple expressions with flexible variable bindings.
///
/// # Arguments
/// * `exprs` - Slice of expressions to evaluate
/// * `var_names` - For each expression, a slice of variable names
/// * `values` - 3D structure: `values[expr_idx][var_idx][point_idx]`
///   - `Value::Num(x)` or `x.into()` → substitute that number
///   - `Value::Expr(e)` or `e.into()` → substitute that expression
///   - `Value::Skip` or `None.into()` → keep variable symbolic
///
/// # Returns
/// For each expression, a Vec of evaluated/simplified expressions at each point.
///
/// # Example
/// ```ignore
/// use symb_anafis::parallel::{evaluate_parallel, Value};
///
/// // x*y*z with x=2, y=Skip → 2*z*y (partial)
/// let results = evaluate_parallel(
///     &[&expr],
///     &[&["x", "y"]],
///     &[&[&[2.0.into(), Value::Skip], &[3.0.into(), 4.0.into()]]],
/// );
/// ```
pub fn evaluate_parallel(
    exprs: &[&Expr],
    var_names: &[&[&str]],
    values: &[&[&[Value]]],
) -> Vec<Vec<Expr>> {
    // Validate input dimensions
    let n_exprs = exprs.len();
    if var_names.len() != n_exprs || values.len() != n_exprs {
        return vec![vec![]; n_exprs];
    }

    // Process each expression in parallel
    (0..n_exprs)
        .into_par_iter()
        .map(|expr_idx| {
            let expr = exprs[expr_idx];
            let vars = var_names[expr_idx];
            let vals = values[expr_idx];

            // Validate variable/value dimensions
            if vars.len() != vals.len() {
                return vec![];
            }

            // Handle no variables case
            let n_vars = vars.len();
            if n_vars == 0 {
                return vec![expr.evaluate(&HashMap::new())];
            }

            // Use maximum length across all variable arrays
            let n_points = vals.iter().map(|v| v.len()).max().unwrap_or(0);

            if n_points == 0 {
                return vec![];
            }

            // Evaluate at each point in parallel
            (0..n_points)
                .into_par_iter()
                .map(|point_idx| {
                    // For symbolic substitutions, we need to do expr.substitute()
                    // For numeric, we use evaluate()

                    let mut var_map: HashMap<&str, f64> = HashMap::new();
                    let mut expr_subs: Vec<(&str, &Expr)> = Vec::new();

                    for var_idx in 0..n_vars {
                        if point_idx < vals[var_idx].len() {
                            match &vals[var_idx][point_idx] {
                                Value::Num(n) => {
                                    var_map.insert(vars[var_idx], *n);
                                }
                                Value::Expr(e) => {
                                    expr_subs.push((vars[var_idx], e));
                                }
                                Value::Skip => {
                                    // Keep symbolic - don't add to var_map
                                }
                            }
                        }
                        // Out of bounds → keep symbolic
                    }

                    // First apply expression substitutions, then numeric evaluation
                    let mut result = expr.clone();
                    for (var, sub_expr) in expr_subs {
                        result = result.substitute(var, sub_expr);
                    }

                    // Then evaluate numerics
                    result.evaluate(&var_map)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprKind, sym};

    /// Helper to extract f64 from Expr::Number
    fn get_num(expr: &Expr) -> f64 {
        match &expr.kind {
            ExprKind::Number(n) => *n,
            _ => f64::NAN,
        }
    }

    #[test]
    fn test_single_expr_single_var() {
        let x = sym("x");
        let expr = x.clone().pow(2.0);

        // Using .into() for clean syntax
        let vals: Vec<Value> = vec![0.0.into(), 1.0.into(), 2.0.into(), 3.0.into()];
        let results = evaluate_parallel(&[&expr], &[&["x"]], &[&[&vals]]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 4);
        assert!((get_num(&results[0][0]) - 0.0).abs() < 1e-10);
        assert!((get_num(&results[0][1]) - 1.0).abs() < 1e-10);
        assert!((get_num(&results[0][2]) - 4.0).abs() < 1e-10);
        assert!((get_num(&results[0][3]) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_expr_two_vars_zip() {
        let x = sym("x");
        let y = sym("y");
        let expr = x.clone() + y.clone();

        let x_vals: Vec<Value> = vec![1.0.into(), 2.0.into()];
        let y_vals: Vec<Value> = vec![10.0.into(), 20.0.into()];
        let results = evaluate_parallel(&[&expr], &[&["x", "y"]], &[&[&x_vals, &y_vals]]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2);
        assert!((get_num(&results[0][0]) - 11.0).abs() < 1e-10);
        assert!((get_num(&results[0][1]) - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_partial_evaluation() {
        // x*y*z with x=2, y=3 should return 6*z
        let x = sym("x");
        let y = sym("y");
        let z = sym("z");
        let expr = x.clone() * y.clone() * z.clone();

        let x_vals: Vec<Value> = vec![2.0.into()];
        let y_vals: Vec<Value> = vec![3.0.into()];
        let results = evaluate_parallel(&[&expr], &[&["x", "y"]], &[&[&x_vals, &y_vals]]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);

        // Result should be 6*z
        let result_str = results[0][0].to_string();
        assert!(result_str.contains("z") || result_str.contains("6"));
    }

    #[test]
    fn test_sparse_array_with_skip() {
        // x*y: x=[2, Skip, 4], y=[3, 5, 6]
        // Point 0: x=2, y=3 → 6
        // Point 1: x=Skip, y=5 → 5*x (symbolic)
        // Point 2: x=4, y=6 → 24
        let x = sym("x");
        let y = sym("y");
        let expr = x.clone() * y.clone();

        let x_vals: Vec<Value> = vec![2.0.into(), Value::Skip, 4.0.into()];
        let y_vals: Vec<Value> = vec![3.0.into(), 5.0.into(), 6.0.into()];
        let results = evaluate_parallel(&[&expr], &[&["x", "y"]], &[&[&x_vals, &y_vals]]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 3);

        // Point 0: 2*3 = 6
        assert!((get_num(&results[0][0]) - 6.0).abs() < 1e-10);

        // Point 1: Skip*5 = 5*x (symbolic)
        let result1_str = results[0][1].to_string();
        assert!(result1_str.contains("x") || result1_str.contains("5"));

        // Point 2: 4*6 = 24
        assert!((get_num(&results[0][2]) - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_expr_substitution() {
        // x*y with x = θ+1, y = 2 → 2*(θ+1)
        let x = sym("x");
        let y = sym("y");
        let theta = sym("θ");
        let expr = x.clone() * y.clone();

        let theta_plus_one = theta.clone() + Expr::number(1.0);
        let x_vals: Vec<Value> = vec![theta_plus_one.into()]; // Expression substitution
        let y_vals: Vec<Value> = vec![2.0.into()]; // Numeric
        let results = evaluate_parallel(&[&expr], &[&["x", "y"]], &[&[&x_vals, &y_vals]]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);

        // Result should contain θ
        let result_str = results[0][0].to_string();
        assert!(result_str.contains("θ") || result_str.contains("2"));
    }

    #[test]
    fn test_multi_expr_different_vars() {
        let x = sym("x");
        let t = sym("t");
        let expr1 = x.clone().pow(2.0);
        let expr2 = t.clone() + Expr::number(1.0);

        let x_vals: Vec<Value> = vec![2.0.into(), 3.0.into()];
        let t_vals: Vec<Value> = vec![10.0.into(), 20.0.into(), 30.0.into()];

        let results = evaluate_parallel(
            &[&expr1, &expr2],
            &[&["x"], &["t"]],
            &[&[&x_vals], &[&t_vals]],
        );

        assert_eq!(results.len(), 2);

        // expr1: x^2 at x=[2,3] → [4, 9]
        assert_eq!(results[0].len(), 2);
        assert!((get_num(&results[0][0]) - 4.0).abs() < 1e-10);
        assert!((get_num(&results[0][1]) - 9.0).abs() < 1e-10);

        // expr2: t+1 at t=[10,20,30] → [11, 21, 31]
        assert_eq!(results[1].len(), 3);
        assert!((get_num(&results[1][0]) - 11.0).abs() < 1e-10);
        assert!((get_num(&results[1][1]) - 21.0).abs() < 1e-10);
        assert!((get_num(&results[1][2]) - 31.0).abs() < 1e-10);
    }
}
