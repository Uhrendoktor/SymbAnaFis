//! Simplification framework - reduces expressions
pub(crate) mod engine;
pub(crate) mod helpers;
mod patterns;
mod rules;

use crate::{Expr, ExprKind};

use std::collections::HashSet;

/// Simplify an expression with user-specified fixed variables
/// Fixed variables are treated as constants (e.g., "e" as a variable, not Euler's constant)
pub fn simplify_expr(expr: Expr, fixed_vars: HashSet<String>) -> Expr {
    let mut current = expr;

    // Use the new rule-based simplification engine with fixed vars
    current = engine::simplify_expr_with_fixed_vars(current, fixed_vars);

    // Prettify roots (x^0.5 -> sqrt(x)) for display
    // This must be done AFTER simplification to avoid fighting with normalize_roots
    current = helpers::prettify_roots(current);

    // Final step: Evaluate numeric functions like sqrt(4) -> 2
    // This happens at the very end so algebraic simplification works on powers
    current = evaluate_numeric_functions(current);

    current
}

/// Simplify an expression with domain safety and user-specified fixed variables
/// Fixed variables are treated as constants (e.g., "e" as a variable, not Euler's constant)
pub(crate) fn simplify_domain_safe(expr: Expr, fixed_vars: HashSet<String>) -> Expr {
    let mut current = expr;

    let mut simplifier = engine::Simplifier::new()
        .with_domain_safe(true)
        .with_fixed_vars(fixed_vars);
    current = simplifier.simplify(current);

    current = helpers::prettify_roots(current);
    current = evaluate_numeric_functions(current);
    current
}

/// Evaluate numeric functions like sqrt(4) -> 2, cbrt(27) -> 3
/// This runs at the very end after prettification
fn evaluate_numeric_functions(expr: Expr) -> Expr {
    match expr.kind {
        // Recursively process subexpressions first
        ExprKind::Add(u, v) => Expr::add_expr(
            evaluate_numeric_functions(u.as_ref().clone()),
            evaluate_numeric_functions(v.as_ref().clone()),
        ),
        ExprKind::Sub(u, v) => Expr::sub_expr(
            evaluate_numeric_functions(u.as_ref().clone()),
            evaluate_numeric_functions(v.as_ref().clone()),
        ),
        ExprKind::Mul(u, v) => {
            let u = evaluate_numeric_functions(u.as_ref().clone());
            let v = evaluate_numeric_functions(v.as_ref().clone());

            // Canonical form: 0.5 * expr -> expr / 2 (for fractional coefficients)
            // This makes log2(x^0.5) -> log2(x)/2 instead of 0.5*log2(x)
            if let ExprKind::Number(n) = &u.kind
                && *n == 0.5
            {
                return Expr::div_expr(v, Expr::number(2.0));
            }
            if let ExprKind::Number(n) = &v.kind
                && *n == 0.5
            {
                return Expr::div_expr(u, Expr::number(2.0));
            }

            Expr::mul_expr(u, v)
        }
        ExprKind::Div(u, v) => {
            let u = evaluate_numeric_functions(u.as_ref().clone());
            let v = evaluate_numeric_functions(v.as_ref().clone());

            if let (ExprKind::Number(n1), ExprKind::Number(n2)) = (&u.kind, &v.kind)
                && *n2 != 0.0
            {
                let result = n1 / n2;
                if (result - result.round()).abs() < 1e-10 {
                    return Expr::number(result.round());
                }
            }

            Expr::div_expr(u, v)
        }
        ExprKind::Pow(u, v) => {
            let u = evaluate_numeric_functions(u.as_ref().clone());
            let v = evaluate_numeric_functions(v.as_ref().clone());

            // Evaluate Number^Number if result is clean
            if let (ExprKind::Number(base), ExprKind::Number(exp)) = (&u.kind, &v.kind) {
                let result = base.powf(*exp);
                if (result - result.round()).abs() < 1e-10 {
                    return Expr::number(result.round());
                }
            }

            Expr::pow(u, v)
        }
        ExprKind::FunctionCall { name, args } => {
            let args: Vec<Expr> = args.into_iter().map(evaluate_numeric_functions).collect();

            // Evaluate sqrt(n) if n is a perfect square
            if name == "sqrt"
                && args.len() == 1
                && let ExprKind::Number(n) = &args[0].kind
            {
                let sqrt_n = n.sqrt();
                if (sqrt_n - sqrt_n.round()).abs() < 1e-10 {
                    return Expr::number(sqrt_n.round());
                }
            }

            // Evaluate cbrt(n) if n is a perfect cube
            if name == "cbrt"
                && args.len() == 1
                && let ExprKind::Number(n) = &args[0].kind
            {
                let cbrt_n = n.cbrt();
                if (cbrt_n - cbrt_n.round()).abs() < 1e-10 {
                    return Expr::number(cbrt_n.round());
                }
            }

            Expr::func_multi(name, args)
        }
        _ => Expr::new(expr.kind),
    }
}
