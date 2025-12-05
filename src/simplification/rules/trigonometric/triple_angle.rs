use crate::ast::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

fn check_sin_triple(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let Expr::Mul(c1, s1) = u
        && matches!(**c1, Expr::Number(n) if n == 3.0 || (n - 3.0).abs() < eps)
        && let Expr::FunctionCall { name, args } = &**s1
        && name == "sin"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, _is_neg)) = extract_sin_cubed(v, x, eps)
            && (coeff == 4.0 || (coeff - 4.0).abs() < eps)
        {
            return Some(Expr::FunctionCall {
                name: "sin".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(3.0)), Rc::new(x.clone()))],
            });
        }
    }
    None
}

fn check_sin_triple_add(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let Expr::Mul(c1, s1) = u
        && matches!(**c1, Expr::Number(n) if (n - 3.0).abs() < eps)
        && let Expr::FunctionCall { name, args } = &**s1
        && name == "sin"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_sin_cubed(v, x, eps)
            && is_neg
            && (coeff - 4.0).abs() < eps
        {
            return Some(Expr::FunctionCall {
                name: "sin".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(3.0)), Rc::new(x.clone()))],
            });
        }
    }
    if let Expr::Mul(c1, s1) = v
        && matches!(**c1, Expr::Number(n) if (n - 3.0).abs() < eps)
        && let Expr::FunctionCall { name, args } = &**s1
        && name == "sin"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_sin_cubed(u, x, eps)
            && is_neg
            && (coeff - 4.0).abs() < eps
        {
            return Some(Expr::FunctionCall {
                name: "sin".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(3.0)), Rc::new(x.clone()))],
            });
        }
    }
    None
}

fn check_cos_triple(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let Expr::Mul(c1, c3) = u
        && matches!(**c1, Expr::Number(n) if n == 4.0 || (n - 4.0).abs() < eps)
        && let Expr::Pow(base, exp) = &**c3
        && matches!(**exp, Expr::Number(n) if n == 3.0)
        && let Expr::FunctionCall { name, args } = &**base
        && name == "cos"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, _is_neg)) = extract_cos(v, x, eps)
            && (coeff == 3.0 || (coeff - 3.0).abs() < eps)
        {
            return Some(Expr::FunctionCall {
                name: "cos".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(3.0)), Rc::new(x.clone()))],
            });
        }
    }
    None
}

fn check_cos_triple_add(u: &Expr, v: &Expr, eps: f64) -> Option<Expr> {
    if let Expr::Mul(c1, c3) = u
        && matches!(**c1, Expr::Number(n) if (n - 4.0).abs() < eps)
        && let Expr::Pow(base, exp) = &**c3
        && matches!(**exp, Expr::Number(n) if n == 3.0)
        && let Expr::FunctionCall { name, args } = &**base
        && name == "cos"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_cos(v, x, eps)
            && is_neg
            && (coeff - 3.0).abs() < eps
        {
            return Some(Expr::FunctionCall {
                name: "cos".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(3.0)), Rc::new(x.clone()))],
            });
        }
    }
    if let Expr::Mul(c1, c3) = v
        && matches!(**c1, Expr::Number(n) if (n - 4.0).abs() < eps)
        && let Expr::Pow(base, exp) = &**c3
        && matches!(**exp, Expr::Number(n) if n == 3.0)
        && let Expr::FunctionCall { name, args } = &**base
        && name == "cos"
        && args.len() == 1
    {
        let x = &args[0];
        if let Some((coeff, is_neg)) = extract_cos(u, x, eps)
            && is_neg
            && (coeff - 3.0).abs() < eps
        {
            return Some(Expr::FunctionCall {
                name: "cos".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(3.0)), Rc::new(x.clone()))],
            });
        }
    }
    None
}

fn extract_sin_cubed(expr: &Expr, x: &Expr, _eps: f64) -> Option<(f64, bool)> {
    if let Expr::Mul(c, s3) = expr
        && let Expr::Pow(base, exp) = &**s3
        && matches!(**exp, Expr::Number(n) if n == 3.0)
        && let Expr::FunctionCall { name, args } = &**base
        && name == "sin"
        && args.len() == 1
        && args[0] == *x
        && let Expr::Number(n) = **c
    {
        return Some((n.abs(), n < 0.0));
    }
    None
}

fn extract_cos(expr: &Expr, x: &Expr, _eps: f64) -> Option<(f64, bool)> {
    if let Expr::Mul(c, c1) = expr
        && let Expr::FunctionCall { name, args } = &**c1
        && name == "cos"
        && args.len() == 1
        && args[0] == *x
        && let Expr::Number(n) = **c
    {
        return Some((n.abs(), n < 0.0));
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
        match expr {
            Expr::Sub(u, v) => {
                if let Some(result) = check_sin_triple(u, v, eps) {
                    return Some(result);
                }
                if let Some(result) = check_cos_triple(u, v, eps) {
                    return Some(result);
                }
            }
            Expr::Add(u, v) => {
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
