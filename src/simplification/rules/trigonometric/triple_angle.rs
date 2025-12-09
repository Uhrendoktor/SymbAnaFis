use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

fn check_sin_triple(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let AstKind::Mul(c1, s1) = &u.kind
        && matches!(&c1.kind, AstKind::Number(n) if *n == 3.0 || (*n - 3.0).abs() < eps)
        && let AstKind::FunctionCall { name, args } = &s1.kind
        && name == "sin"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, _is_neg)) = extract_sin_cubed(v, x, eps)
            && (coeff == 4.0 || (coeff - 4.0).abs() < eps)
        {
            return Some(Expr::func(
                "sin",
                Expr::mul_expr(Expr::number(3.0), x.clone()),
            ));
        }
    }
    None
}

fn check_sin_triple_add(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let AstKind::Mul(c1, s1) = &u.kind
        && matches!(&c1.kind, AstKind::Number(n) if (*n - 3.0).abs() < eps)
        && let AstKind::FunctionCall { name, args } = &s1.kind
        && name == "sin"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_sin_cubed(v, x, eps)
            && is_neg
            && (coeff - 4.0).abs() < eps
        {
            return Some(Expr::func(
                "sin",
                Expr::mul_expr(Expr::number(3.0), x.clone()),
            ));
        }
    }
    if let AstKind::Mul(c1, s1) = &v.kind
        && matches!(&c1.kind, AstKind::Number(n) if (*n - 3.0).abs() < eps)
        && let AstKind::FunctionCall { name, args } = &s1.kind
        && name == "sin"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_sin_cubed(u, x, eps)
            && is_neg
            && (coeff - 4.0).abs() < eps
        {
            return Some(Expr::func(
                "sin",
                Expr::mul_expr(Expr::number(3.0), x.clone()),
            ));
        }
    }
    None
}

fn check_cos_triple(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let AstKind::Mul(c1, c3) = &u.kind
        && matches!(&c1.kind, AstKind::Number(n) if *n == 4.0 || (*n - 4.0).abs() < eps)
        && let AstKind::Pow(base, exp) = &c3.kind
        && matches!(&exp.kind, AstKind::Number(n) if *n == 3.0)
        && let AstKind::FunctionCall { name, args } = &base.kind
        && name == "cos"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, _is_neg)) = extract_cos(v, x, eps)
            && (coeff == 3.0 || (coeff - 3.0).abs() < eps)
        {
            return Some(Expr::func(
                "cos",
                Expr::mul_expr(Expr::number(3.0), x.clone()),
            ));
        }
    }
    None
}

fn check_cos_triple_add(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let AstKind::Mul(c1, c3) = &u.kind
        && matches!(&c1.kind, AstKind::Number(n) if (*n - 4.0).abs() < eps)
        && let AstKind::Pow(base, exp) = &c3.kind
        && matches!(&exp.kind, AstKind::Number(n) if *n == 3.0)
        && let AstKind::FunctionCall { name, args } = &base.kind
        && name == "cos"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_cos(v, x, eps)
            && is_neg
            && (coeff - 3.0).abs() < eps
        {
            return Some(Expr::func(
                "cos",
                Expr::mul_expr(Expr::number(3.0), x.clone()),
            ));
        }
    }
    if let AstKind::Mul(c1, c3) = &v.kind
        && matches!(&c1.kind, AstKind::Number(n) if (*n - 4.0).abs() < eps)
        && let AstKind::Pow(base, exp) = &c3.kind
        && matches!(&exp.kind, AstKind::Number(n) if *n == 3.0)
        && let AstKind::FunctionCall { name, args } = &base.kind
        && name == "cos"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_cos(u, x, eps)
            && is_neg
            && (coeff - 3.0).abs() < eps
        {
            return Some(Expr::func(
                "cos",
                Expr::mul_expr(Expr::number(3.0), x.clone()),
            ));
        }
    }
    None
}

fn extract_sin_cubed(expr: &Expr, x: &Expr, _eps: f64) -> Option<(f64, bool)> {
    if let AstKind::Mul(c, s3) = &expr.kind
        && let AstKind::Pow(base, exp) = &s3.kind
        && matches!(&exp.kind, AstKind::Number(n) if *n == 3.0)
        && let AstKind::FunctionCall { name, args } = &base.kind
        && name == "sin"
        && args.len() == 1
        && args[0] == *x
        && let AstKind::Number(n) = &c.kind
    {
        return Some((n.abs(), *n < 0.0));
    }
    None
}

fn extract_cos(expr: &Expr, x: &Expr, _eps: f64) -> Option<(f64, bool)> {
    if let AstKind::Mul(c, c1) = &expr.kind
        && let AstKind::FunctionCall { name, args } = &c1.kind
        && name == "cos"
        && args.len() == 1
        && args[0] == *x
        && let AstKind::Number(n) = &c.kind
    {
        return Some((n.abs(), *n < 0.0));
    }
    None
}

rule!(
    TrigTripleAngleRule,
    "trig_triple_angle",
    70,
    Trigonometric,
    &[ExprKind::Add, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        let eps = 1e-10;
        match &expr.kind {
            AstKind::Sub(u, v) => {
                if let Some(result) = check_sin_triple(u, v, eps) {
                    return Some(result);
                }
                if let Some(result) = check_cos_triple(u, v, eps) {
                    return Some(result);
                }
            }
            AstKind::Add(u, v) => {
                if let Some(result) = check_sin_triple_add(u, v, eps) {
                    return Some(result);
                }
                if let Some(result) = check_cos_triple_add(u, v, eps) {
                    return Some(result);
                }
            }
            _ => {}
        }
        None
    }
);
