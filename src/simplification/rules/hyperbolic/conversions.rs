use super::helpers::*;
use crate::ast::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    SinhFromExpRule,
    "sinh_from_exp",
    80,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Div(numerator, denominator) = expr {
            if let Expr::Number(d) = &**denominator
                && *d == 2.0
            {
                if let Expr::Sub(u, v) = &**numerator
                    && let Some(x) = match_sinh_pattern_sub(u, v)
                {
                    return Some(Expr::FunctionCall {
                        name: "sinh".to_string(),
                        args: vec![x],
                    });
                }
                if let Expr::Add(u, v) = &**numerator {
                    if let Some(neg_inner) = extract_negated_term(v)
                        && let Some(x) = match_sinh_pattern_sub(u, neg_inner)
                    {
                        return Some(Expr::FunctionCall {
                            name: "sinh".to_string(),
                            args: vec![x],
                        });
                    }
                    if let Some(neg_inner) = extract_negated_term(u)
                        && let Some(x) = match_sinh_pattern_sub(v, neg_inner)
                    {
                        return Some(Expr::FunctionCall {
                            name: "sinh".to_string(),
                            args: vec![x],
                        });
                    }
                }
            }

            if let Some(x) = match_alt_sinh_pattern(numerator, denominator) {
                return Some(Expr::FunctionCall {
                    name: "sinh".to_string(),
                    args: vec![x],
                });
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
        if let Expr::Div(numerator, denominator) = expr {
            if let Expr::Number(d) = &**denominator
                && *d == 2.0
                && let Expr::Add(u, v) = &**numerator
                && let Some(x) = match_cosh_pattern(u, v)
            {
                return Some(Expr::FunctionCall {
                    name: "cosh".to_string(),
                    args: vec![x],
                });
            }

            if let Some(x) = match_alt_cosh_pattern(numerator, denominator) {
                return Some(Expr::FunctionCall {
                    name: "cosh".to_string(),
                    args: vec![x],
                });
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
        if let Expr::Div(numerator, denominator) = expr {
            let num_arg = if let Expr::Sub(u, v) = &**numerator {
                match_sinh_pattern_sub(u, v)
            } else {
                None
            };

            let den_arg = if let Expr::Add(u, v) = &**denominator {
                match_cosh_pattern(u, v)
            } else {
                None
            };

            if let (Some(n_arg), Some(d_arg)) = (num_arg, den_arg)
                && n_arg == d_arg
            {
                return Some(Expr::FunctionCall {
                    name: "tanh".to_string(),
                    args: vec![n_arg],
                });
            }

            if let Some(x_num) = match_e2x_minus_1_factored(numerator)
                && let Some(x_den) = match_e2x_plus_1(denominator)
                && x_num == x_den
            {
                return Some(Expr::FunctionCall {
                    name: "tanh".to_string(),
                    args: vec![x_num],
                });
            }

            if let Some(x_num) = match_e2x_minus_1_direct(numerator)
                && let Some(x_den) = match_e2x_plus_1(denominator)
                && x_num == x_den
            {
                return Some(Expr::FunctionCall {
                    name: "tanh".to_string(),
                    args: vec![x_num],
                });
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
        if let Expr::Div(numerator, denominator) = expr {
            if let Expr::Number(n) = &**numerator
                && *n == 2.0
                && let Expr::Add(u, v) = &**denominator
                && let Some(x) = match_cosh_pattern(u, v)
            {
                return Some(Expr::FunctionCall {
                    name: "sech".to_string(),
                    args: vec![x],
                });
            }

            if let Some(x) = match_alt_sech_pattern(numerator, denominator) {
                return Some(Expr::FunctionCall {
                    name: "sech".to_string(),
                    args: vec![x],
                });
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
        if let Expr::Div(numerator, denominator) = expr {
            if let Expr::Number(n) = &**numerator
                && *n == 2.0
                && let Expr::Sub(u, v) = &**denominator
                && let Some(x) = match_sinh_pattern_sub(u, v)
            {
                return Some(Expr::FunctionCall {
                    name: "csch".to_string(),
                    args: vec![x],
                });
            }

            if let Expr::Mul(a, b) = &**numerator {
                let (coeff, exp_term) = if let Expr::Number(n) = &**a {
                    (*n, &**b)
                } else if let Expr::Number(n) = &**b {
                    (*n, &**a)
                } else {
                    return None;
                };

                if coeff == 2.0
                    && let Some(x) = ExpTerm::get_direct_exp_arg(exp_term)
                    && let Expr::Sub(u, v) = &**denominator
                    && let Expr::Number(n) = &**v
                    && *n == 1.0
                    && let Some(denom_arg) = ExpTerm::get_direct_exp_arg(u)
                    && is_double_of(&denom_arg, &x)
                {
                    return Some(Expr::FunctionCall {
                        name: "csch".to_string(),
                        args: vec![x],
                    });
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
        if let Expr::Div(numerator, denominator) = expr {
            let num_arg = if let Expr::Add(u, v) = &**numerator {
                match_cosh_pattern(u, v)
            } else {
                None
            };

            let den_arg = if let Expr::Sub(u, v) = &**denominator {
                match_sinh_pattern_sub(u, v)
            } else {
                None
            };

            if let (Some(n_arg), Some(d_arg)) = (num_arg, den_arg)
                && n_arg == d_arg
            {
                return Some(Expr::FunctionCall {
                    name: "coth".to_string(),
                    args: vec![n_arg],
                });
            }

            if let Some(x_num) = match_e2x_plus_1(numerator)
                && let Some(x_den) = match_e2x_minus_1_factored(denominator)
                && x_num == x_den
            {
                return Some(Expr::FunctionCall {
                    name: "coth".to_string(),
                    args: vec![x_num],
                });
            }

            if let Some(x_num) = match_e2x_plus_1(numerator)
                && let Some(x_den) = match_e2x_minus_1_direct(denominator)
                && x_num == x_den
            {
                return Some(Expr::FunctionCall {
                    name: "coth".to_string(),
                    args: vec![x_num],
                });
            }
        }
        None
    }
);
