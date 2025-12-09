use super::helpers::*;
use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    SinhFromExpRule,
    "sinh_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(numerator, denominator) = &expr.kind {
            if let AstKind::Number(d) = &denominator.kind
                && *d == 2.0
            {
                if let AstKind::Sub(u, v) = &numerator.kind
                    && let Some(x) = match_sinh_pattern_sub(u, v)
                {
                    return Some(Expr::func("sinh", x));
                }
                if let AstKind::Add(u, v) = &numerator.kind {
                    if let Some(neg_inner) = extract_negated_term(v)
                        && let Some(x) = match_sinh_pattern_sub(u, &neg_inner)
                    {
                        return Some(Expr::func("sinh", x));
                    }
                    if let Some(neg_inner) = extract_negated_term(u)
                        && let Some(x) = match_sinh_pattern_sub(v, &neg_inner)
                    {
                        return Some(Expr::func("sinh", x));
                    }
                }
            }

            if let Some(x) = match_alt_sinh_pattern(numerator, denominator) {
                return Some(Expr::func("sinh", x));
            }
        }
        None
    }
);

rule!(
    CoshFromExpRule,
    "cosh_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(numerator, denominator) = &expr.kind {
            if let AstKind::Number(d) = &denominator.kind
                && *d == 2.0
                && let AstKind::Add(u, v) = &numerator.kind
                && let Some(x) = match_cosh_pattern(u, v)
            {
                return Some(Expr::func("cosh", x));
            }

            if let Some(x) = match_alt_cosh_pattern(numerator, denominator) {
                return Some(Expr::func("cosh", x));
            }
        }
        None
    }
);

rule!(
    TanhFromExpRule,
    "tanh_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(numerator, denominator) = &expr.kind {
            let num_arg = if let AstKind::Sub(u, v) = &numerator.kind {
                match_sinh_pattern_sub(u, v)
            } else {
                None
            };

            let den_arg = if let AstKind::Add(u, v) = &denominator.kind {
                match_cosh_pattern(u, v)
            } else {
                None
            };

            if let (Some(n_arg), Some(d_arg)) = (num_arg, den_arg)
                && n_arg == d_arg
            {
                return Some(Expr::func("tanh", n_arg));
            }

            if let Some(x_num) = match_e2x_minus_1_factored(numerator)
                && let Some(x_den) = match_e2x_plus_1(denominator)
                && x_num == x_den
            {
                return Some(Expr::func("tanh", x_num));
            }

            if let Some(x_num) = match_e2x_minus_1_direct(numerator)
                && let Some(x_den) = match_e2x_plus_1(denominator)
                && x_num == x_den
            {
                return Some(Expr::func("tanh", x_num));
            }
        }
        None
    }
);

rule!(
    SechFromExpRule,
    "sech_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(numerator, denominator) = &expr.kind {
            if let AstKind::Number(n) = &numerator.kind
                && *n == 2.0
                && let AstKind::Add(u, v) = &denominator.kind
                && let Some(x) = match_cosh_pattern(u, v)
            {
                return Some(Expr::func("sech", x));
            }

            if let Some(x) = match_alt_sech_pattern(numerator, denominator) {
                return Some(Expr::func("sech", x));
            }
        }
        None
    }
);

rule!(
    CschFromExpRule,
    "csch_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(numerator, denominator) = &expr.kind {
            if let AstKind::Number(n) = &numerator.kind
                && *n == 2.0
                && let AstKind::Sub(u, v) = &denominator.kind
                && let Some(x) = match_sinh_pattern_sub(u, v)
            {
                return Some(Expr::func("csch", x));
            }

            if let AstKind::Mul(a, b) = &numerator.kind {
                let (coeff, exp_term) = if let AstKind::Number(n) = &a.kind {
                    (*n, b)
                } else if let AstKind::Number(n) = &b.kind {
                    (*n, a)
                } else {
                    return None;
                };

                if coeff == 2.0
                    && let Some(x) = ExpTerm::get_direct_exp_arg(exp_term)
                    && let AstKind::Sub(u, v) = &denominator.kind
                    && let AstKind::Number(n) = &v.kind
                    && *n == 1.0
                    && let Some(denom_arg) = ExpTerm::get_direct_exp_arg(u)
                    && is_double_of(&denom_arg, &x)
                {
                    return Some(Expr::func("csch", x));
                }
            }
        }
        None
    }
);

rule!(
    CothFromExpRule,
    "coth_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(numerator, denominator) = &expr.kind {
            let num_arg = if let AstKind::Add(u, v) = &numerator.kind {
                match_cosh_pattern(u, v)
            } else {
                None
            };

            let den_arg = if let AstKind::Sub(u, v) = &denominator.kind {
                match_sinh_pattern_sub(u, v)
            } else {
                None
            };

            if let (Some(n_arg), Some(d_arg)) = (num_arg, den_arg)
                && n_arg == d_arg
            {
                return Some(Expr::func("coth", n_arg));
            }

            if let Some(x_num) = match_e2x_plus_1(numerator)
                && let Some(x_den) = match_e2x_minus_1_factored(denominator)
                && x_num == x_den
            {
                return Some(Expr::func("coth", x_num));
            }

            if let Some(x_num) = match_e2x_plus_1(numerator)
                && let Some(x_den) = match_e2x_minus_1_direct(denominator)
                && x_num == x_den
            {
                return Some(Expr::func("coth", x_num));
            }
        }
        None
    }
);
