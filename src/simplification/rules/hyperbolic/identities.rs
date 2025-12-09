use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
// Still potentially needed for internal use, though Expr helpers usually abstract it.

rule!(
    SinhZeroRule,
    "sinh_zero",
    95,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sinh"
            && args.len() == 1
            && matches!(args[0].kind, AstKind::Number(n) if n == 0.0)
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    CoshZeroRule,
    "cosh_zero",
    95,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cosh"
            && args.len() == 1
            && matches!(args[0].kind, AstKind::Number(n) if n == 0.0)
        {
            return Some(Expr::number(1.0));
        }
        None
    }
);

rule!(
    SinhNegationRule,
    "sinh_negation",
    90,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sinh"
            && args.len() == 1
            && let AstKind::Mul(lhs, rhs) = &args[0].kind
        {
            if let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
            {
                return Some(Expr::mul_expr(
                    Expr::number(-1.0),
                    Expr::func("sinh", (**rhs).clone()),
                ));
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == -1.0
            {
                return Some(Expr::mul_expr(
                    Expr::number(-1.0),
                    Expr::func("sinh", (**lhs).clone()),
                ));
            }
        }
        None
    }
);

rule!(
    CoshNegationRule,
    "cosh_negation",
    90,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cosh"
            && args.len() == 1
            && let AstKind::Mul(lhs, rhs) = &args[0].kind
        {
            if let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
            {
                return Some(Expr::func("cosh", (**rhs).clone()));
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == -1.0
            {
                return Some(Expr::func("cosh", (**lhs).clone()));
            }
        }
        None
    }
);

rule!(
    TanhNegationRule,
    "tanh_negation",
    90,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "tanh"
            && args.len() == 1
            && let AstKind::Mul(lhs, rhs) = &args[0].kind
        {
            if let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
            {
                return Some(Expr::mul_expr(
                    Expr::number(-1.0),
                    Expr::func("tanh", (**rhs).clone()),
                ));
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == -1.0
            {
                return Some(Expr::mul_expr(
                    Expr::number(-1.0),
                    Expr::func("tanh", (**lhs).clone()),
                ));
            }
        }
        None
    }
);

rule!(
    SinhAsinhIdentityRule,
    "sinh_asinh_identity",
    95,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sinh"
            && args.len() == 1
            && let AstKind::FunctionCall {
                name: inner_name,
                args: inner_args,
            } = &args[0].kind
            && inner_name == "asinh"
            && inner_args.len() == 1
        {
            return Some(inner_args[0].clone());
        }
        None
    }
);

rule!(
    CoshAcoshIdentityRule,
    "cosh_acosh_identity",
    95,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cosh"
            && args.len() == 1
            && let AstKind::FunctionCall {
                name: inner_name,
                args: inner_args,
            } = &args[0].kind
            && inner_name == "acosh"
            && inner_args.len() == 1
        {
            return Some(inner_args[0].clone());
        }
        None
    }
);

rule!(
    TanhAtanhIdentityRule,
    "tanh_atanh_identity",
    95,
    Hyperbolic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "tanh"
            && args.len() == 1
            && let AstKind::FunctionCall {
                name: inner_name,
                args: inner_args,
            } = &args[0].kind
            && inner_name == "atanh"
            && inner_args.len() == 1
        {
            return Some(inner_args[0].clone());
        }
        None
    }
);

// Hyperbolic identity: cosh^2(x) - sinh^2(x) = 1 and related
rule!(
    HyperbolicIdentityRule,
    "hyperbolic_identity",
    95,
    Hyperbolic,
    &[ExprKind::Add, ExprKind::Sub, ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        // cosh^2(x) - sinh^2(x) = 1
        if let AstKind::Sub(u, v) = &expr.kind
            && let (Some((name1, arg1)), Some((name2, arg2))) =
                (get_hyperbolic_power(u, 2.0), get_hyperbolic_power(v, 2.0))
            && arg1 == arg2
            && name1 == "cosh"
            && name2 == "sinh"
        {
            return Some(Expr::number(1.0));
        }

        // 1 - tanh^2(x) = sech^2(x)
        if let AstKind::Sub(u, v) = &expr.kind
            && let AstKind::Number(n) = &u.kind
            && *n == 1.0
            && let Some((name, arg)) = get_hyperbolic_power(v, 2.0)
            && name == "tanh"
        {
            return Some(Expr::pow(Expr::func("sech", arg), Expr::number(2.0)));
        }

        // coth^2(x) - 1 = csch^2(x)
        if let AstKind::Sub(u, v) = &expr.kind
            && let AstKind::Number(n) = &v.kind
            && *n == 1.0
            && let Some((name, arg)) = get_hyperbolic_power(u, 2.0)
            && name == "coth"
        {
            return Some(Expr::pow(Expr::func("csch", arg), Expr::number(2.0)));
        }

        // (cosh(x) - sinh(x)) * (cosh(x) + sinh(x)) = 1
        if let AstKind::Mul(u, v) = &expr.kind {
            if let (Some(arg1), Some(arg2)) =
                (is_cosh_minus_sinh_term(u), is_cosh_plus_sinh_term(v))
                && arg1 == arg2
            {
                return Some(Expr::number(1.0));
            }
            if let (Some(arg1), Some(arg2)) =
                (is_cosh_minus_sinh_term(v), is_cosh_plus_sinh_term(u))
                && arg1 == arg2
            {
                return Some(Expr::number(1.0));
            }
        }

        // Check Add forms for normalized expressions
        if let AstKind::Add(u, v) = &expr.kind {
            // cosh^2(x) + (-1 * sinh^2(x)) = 1
            if let Some((name1, arg1)) = get_hyperbolic_power(u, 2.0)
                && name1 == "cosh"
                && let AstKind::Mul(lhs, rhs) = &v.kind
                && let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
                && let Some((name2, arg2)) = get_hyperbolic_power(rhs, 2.0)
                && name2 == "sinh"
                && arg1 == arg2
            {
                return Some(Expr::number(1.0));
            }

            // 1 + (-1 * tanh^2(x)) = sech^2(x)
            if let AstKind::Number(n) = &u.kind
                && *n == 1.0
                && let AstKind::Mul(lhs, rhs) = &v.kind
                && let AstKind::Number(nn) = &lhs.kind
                && *nn == -1.0
                && let Some((name, arg)) = get_hyperbolic_power(rhs, 2.0)
                && name == "tanh"
            {
                return Some(Expr::pow(Expr::func("sech", arg), Expr::number(2.0)));
            }
        }

        None
    }
);

fn get_hyperbolic_power(expr: &Expr, power: f64) -> Option<(&str, Expr)> {
    if let AstKind::Pow(base, exp) = &expr.kind
        && let AstKind::Number(p) = &exp.kind
        && *p == power
        && let AstKind::FunctionCall { name, args } = &base.kind
        && args.len() == 1
        && (name == "sinh" || name == "cosh" || name == "tanh" || name == "coth")
    {
        return Some((name, args[0].clone()));
    }
    None
}

fn is_cosh_minus_sinh_term(expr: &Expr) -> Option<Expr> {
    if let AstKind::Sub(u, v) = &expr.kind
        && let AstKind::FunctionCall { name: n1, args: a1 } = &u.kind
        && n1 == "cosh"
        && a1.len() == 1
        && let AstKind::FunctionCall { name: n2, args: a2 } = &v.kind
        && n2 == "sinh"
        && a2.len() == 1
        && a1[0] == a2[0]
    {
        return Some(a1[0].clone());
    }
    None
}

fn is_cosh_plus_sinh_term(expr: &Expr) -> Option<Expr> {
    if let AstKind::Add(u, v) = &expr.kind
        && let AstKind::FunctionCall { name: n1, args: a1 } = &u.kind
        && n1 == "cosh"
        && a1.len() == 1
        && let AstKind::FunctionCall { name: n2, args: a2 } = &v.kind
        && n2 == "sinh"
        && a2.len() == 1
        && a1[0] == a2[0]
    {
        return Some(a1[0].clone());
    }
    None
}

rule!(
    HyperbolicTripleAngleRule,
    "hyperbolic_triple_angle",
    70,
    Hyperbolic,
    &[ExprKind::Add, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        match &expr.kind {
            AstKind::Add(u, v) => {
                // 4*sinh(x)^3 + 3*sinh(x) -> sinh(3x)
                if let (Some((c1, arg1, p1)), Some((c2, arg2, p2))) =
                    (parse_fn_term(u, "sinh"), parse_fn_term(v, "sinh"))
                    && arg1 == arg2
                {
                    let eps = 1e-10;
                    if ((c1 == 4.0 || (c1 - 4.0).abs() < eps) && p1 == 3.0)
                        && (c2 == 3.0 && p2 == 1.0)
                        || ((c2 == 4.0 || (c2 - 4.0).abs() < eps) && p2 == 3.0)
                            && (c1 == 3.0 && p1 == 1.0)
                    {
                        return Some(Expr::func("sinh", Expr::mul_expr(Expr::number(3.0), arg1)));
                    }
                }
            }
            AstKind::Sub(u, v) => {
                // 4*cosh(x)^3 - 3*cosh(x) -> cosh(3x)
                if let (Some((c1, arg1, p1)), Some((c2, arg2, p2))) =
                    (parse_fn_term(u, "cosh"), parse_fn_term(v, "cosh"))
                    && arg1 == arg2
                {
                    let eps = 1e-10;
                    if (c1 == 4.0 || (c1 - 4.0).abs() < eps) && p1 == 3.0 && c2 == 3.0 && p2 == 1.0
                    {
                        return Some(Expr::func("cosh", Expr::mul_expr(Expr::number(3.0), arg1)));
                    }
                }
            }
            _ => {}
        }
        None
    }
);

fn parse_fn_term(expr: &Expr, func_name: &str) -> Option<(f64, Expr, f64)> {
    if let AstKind::FunctionCall { name, args } = &expr.kind
        && name == func_name
        && args.len() == 1
    {
        return Some((1.0, args[0].clone(), 1.0));
    }
    if let AstKind::Mul(lhs, rhs) = &expr.kind
        && let AstKind::Number(c) = &lhs.kind
        && let AstKind::FunctionCall { name, args } = &rhs.kind
        && name == func_name
        && args.len() == 1
    {
        return Some((*c, args[0].clone(), 1.0));
    }
    if let AstKind::Pow(base, exp) = &expr.kind
        && let AstKind::Number(p) = &exp.kind
        && let AstKind::FunctionCall { name, args } = &base.kind
        && name == func_name
        && args.len() == 1
    {
        return Some((1.0, args[0].clone(), *p));
    }
    if let AstKind::Mul(lhs, rhs) = &expr.kind
        && let AstKind::Number(c) = &lhs.kind
        && let AstKind::Pow(base, exp) = &rhs.kind
        && let AstKind::Number(p) = &exp.kind
        && let AstKind::FunctionCall { name, args } = &base.kind
        && name == func_name
        && args.len() == 1
    {
        return Some((*c, args[0].clone(), *p));
    }
    None
}
