use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::sync::Arc; // May still be needed if not fully removed by helpers

rule!(
    SqrtPowerRule,
    "sqrt_power",
    85,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sqrt"
            && args.len() == 1
            && let AstKind::Pow(base, exp) = &args[0].kind
        {
            // Special case: sqrt(x^2) should always return abs(x)
            if let AstKind::Number(n) = &exp.kind
                && *n == 2.0
            {
                // sqrt(x^2) = |x|
                return Some(Expr::func("abs", base.as_ref().clone()));
            }

            // Create new exponent: exp / 2
            let new_exp = Expr::div_expr(exp.as_ref().clone(), Expr::number(2.0));

            // Simplify the division immediately
            let simplified_exp = match &new_exp.kind {
                AstKind::Div(u, v) => {
                    if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind) {
                        if *b != 0.0 {
                            let result = a / b;
                            if (result - result.round()).abs() < 1e-10 {
                                Expr::number(result.round())
                            } else {
                                new_exp
                            }
                        } else {
                            new_exp
                        }
                    } else {
                        new_exp
                    }
                }
                _ => new_exp,
            };

            // If exponent simplified to 1, return base directly
            if matches!(&simplified_exp.kind, AstKind::Number(n) if *n == 1.0) {
                return Some(base.as_ref().clone());
            }

            let result = Expr::pow(base.as_ref().clone(), simplified_exp.clone());

            return Some(result);
        }
        None
    }
);

rule!(
    CbrtPowerRule,
    "cbrt_power",
    85,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cbrt"
            && args.len() == 1
            && let AstKind::Pow(base, exp) = &args[0].kind
        {
            // Create new exponent: exp / 3
            let new_exp = Expr::div_expr(exp.as_ref().clone(), Expr::number(3.0));

            // Simplify the division immediately
            let simplified_exp = match &new_exp.kind {
                AstKind::Div(u, v) => {
                    if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind) {
                        if *b != 0.0 {
                            let result = a / b;
                            if (result - result.round()).abs() < 1e-10 {
                                Expr::number(result.round())
                            } else {
                                new_exp
                            }
                        } else {
                            new_exp
                        }
                    } else {
                        new_exp
                    }
                }
                _ => new_exp,
            };

            // If exponent simplified to 1, return base directly
            if matches!(&simplified_exp.kind, AstKind::Number(n) if *n == 1.0) {
                return Some(base.as_ref().clone());
            }

            return Some(Expr::pow(base.as_ref().clone(), simplified_exp));
        }
        None
    }
);

rule!(SqrtMulRule, "sqrt_mul", 56, Root, &[ExprKind::Mul], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let AstKind::Mul(u, v) = &expr.kind {
        // Check for sqrt(a) * sqrt(b)
        if let (
            AstKind::FunctionCall {
                name: u_name,
                args: u_args,
            },
            AstKind::FunctionCall {
                name: v_name,
                args: v_args,
            },
        ) = (&u.kind, &v.kind)
            && u_name == "sqrt"
            && v_name == "sqrt"
            && u_args.len() == 1
            && v_args.len() == 1
        {
            return Some(Expr::func(
                "sqrt",
                Expr::mul_expr(
                    u_args[0].clone(),
                    v_args[0].clone(),
                ),
            ));
        }
    }
    None
});

rule!(SqrtDivRule, "sqrt_div", 56, Root, &[ExprKind::Div], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let AstKind::Div(u, v) = &expr.kind {
        // Check for sqrt(a) / sqrt(b)
        if let (
            AstKind::FunctionCall {
                name: u_name,
                args: u_args,
            },
            AstKind::FunctionCall {
                name: v_name,
                args: v_args,
            },
        ) = (&u.kind, &v.kind)
            && u_name == "sqrt"
            && v_name == "sqrt"
            && u_args.len() == 1
            && v_args.len() == 1
        {
            return Some(Expr::func(
                "sqrt",
                Expr::div_expr(
                    u_args[0].clone(),
                    v_args[0].clone(),
                ),
            ));
        }
    }
    None
});

rule!(
    NormalizeRootsRule,
    "normalize_roots",
    50,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && args.len() == 1
        {
            match name.as_str() {
                "sqrt" => {
                    return Some(Expr::pow(
                        args[0].clone(),
                        Expr::div_expr(Expr::number(1.0), Expr::number(2.0)),
                    ));
                }
                "cbrt" => {
                    return Some(Expr::pow(
                        args[0].clone(),
                        Expr::div_expr(Expr::number(1.0), Expr::number(3.0)),
                    ));
                }
                _ => {}
            }
        }
        None
    }
);

rule!(
    SqrtExtractSquareRule,
    "sqrt_extract_square",
    84,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        // sqrt(a * x^2) → |x| * sqrt(a)
        // sqrt(x^2 * a) → |x| * sqrt(a)
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sqrt"
            && args.len() == 1
            && let AstKind::Mul(u, v) = &args[0].kind
        {
            // Check if either factor is a square (x^2)
            let (square_base, other) = if let AstKind::Pow(base, exp) = &u.kind
                && let AstKind::Number(n) = &exp.kind
                && *n == 2.0
            {
                (Some(base), v)
            } else if let AstKind::Pow(base, exp) = &v.kind
                && let AstKind::Number(n) = &exp.kind
                && *n == 2.0
            {
                (Some(base), u)
            } else {
                (None, u)
            };

            if let Some(base) = square_base {
                // sqrt(other * base^2) = |base| * sqrt(other)
                let abs_base = Expr::func("abs", base.as_ref().clone());
                let sqrt_other = Expr::func("sqrt", other.as_ref().clone());
                return Some(Expr::mul_expr(abs_base, sqrt_other));
            }
        }
        None
    }
);

/// Get all root simplification rules in priority order
pub(crate) fn get_root_rules() -> Vec<Arc<dyn Rule + Send + Sync>> {
    vec![
        Arc::new(SqrtPowerRule),
        Arc::new(SqrtExtractSquareRule),
        Arc::new(CbrtPowerRule),
        Arc::new(SqrtMulRule),
        Arc::new(SqrtDivRule),
        Arc::new(NormalizeRootsRule),
    ]
}
