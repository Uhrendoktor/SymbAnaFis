use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::helpers;
use crate::simplification::patterns::common::extract_coefficient;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    TrigDoubleAngleRule,
    "trig_double_angle",
    85,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sin"
            && args.len() == 1
        {
            let (coeff, rest) = extract_coefficient(&args[0]);
            if coeff == 2.0 {
                return Some(Expr::mul_expr(
                    Expr::number(2.0),
                    Expr::mul_expr(Expr::func("sin", rest.clone()), Expr::func("cos", rest)),
                ));
            }
        }
        None
    }
);

rule!(
    CosDoubleAngleDifferenceRule,
    "cos_double_angle_difference",
    85,
    Trigonometric,
    &[ExprKind::Add, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        let (pos, neg) = match &expr.kind {
            AstKind::Sub(u, v) => (u, v),
            AstKind::Add(u, v) => {
                if let AstKind::Mul(c, inner) = &u.kind {
                    if matches!(&c.kind, AstKind::Number(n) if *n == -1.0) {
                        (v, inner)
                    } else if let AstKind::Mul(c, inner) = &v.kind {
                        if matches!(&c.kind, AstKind::Number(n) if *n == -1.0) {
                            (u, inner)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else if let AstKind::Mul(c, inner) = &v.kind {
                    if matches!(&c.kind, AstKind::Number(n) if *n == -1.0) {
                        (u, inner)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        if let Some(("cos", arg1)) = helpers::get_fn_pow_named(pos, 2.0)
            && let Some(("sin", arg2)) = helpers::get_fn_pow_named(neg, 2.0)
            && arg1 == arg2
        {
            return Some(Expr::func("cos", Expr::mul_expr(Expr::number(2.0), arg1)));
        }

        if let Some(("sin", arg1)) = helpers::get_fn_pow_named(pos, 2.0)
            && let Some(("cos", arg2)) = helpers::get_fn_pow_named(neg, 2.0)
            && arg1 == arg2
        {
            return Some(Expr::mul_expr(
                Expr::number(-1.0),
                Expr::func("cos", Expr::mul_expr(Expr::number(2.0), arg1)),
            ));
        }

        None
    }
);

rule!(
    TrigSumDifferenceRule,
    "trig_sum_difference",
    70,
    Trigonometric,
    &[ExprKind::Add, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        match &expr.kind {
            AstKind::Add(u, v) => {
                if let Some((x, y)) = helpers::get_product_fn_args(u, "sin", "cos")
                    .and_then(|(s1, c1)| {
                        helpers::get_product_fn_args(v, "sin", "cos")
                            .map(|(s2, c2)| (s1, c1, s2, c2))
                    })
                    .and_then(|(s1, c1, s2, c2)| {
                        if s1 == c2 && c1 == s2 {
                            Some((s1, c1))
                        } else {
                            None
                        }
                    })
                {
                    return Some(Expr::func("sin", Expr::add_expr(x, y)));
                }
            }
            AstKind::Sub(u, v) => {
                if let Some((x, y)) = helpers::get_product_fn_args(u, "sin", "cos")
                    .and_then(|(s1, c1)| {
                        helpers::get_product_fn_args(v, "cos", "sin")
                            .map(|(c2, s2)| (s1, c1, c2, s2))
                    })
                    .and_then(|(s1, c1, c2, s2)| {
                        if s1 == c2 && c1 == s2 {
                            Some((s1, c1))
                        } else {
                            None
                        }
                    })
                {
                    return Some(Expr::func("sin", Expr::sub_expr(x, y)));
                }
                if let Some((cx, cy)) = helpers::get_product_fn_args(u, "cos", "cos")
                    && let Some((sx, sy)) = helpers::get_product_fn_args(v, "sin", "sin")
                    && ((cx == sx && cy == sy) || (cx == sy && cy == sx))
                {
                    return Some(Expr::func("cos", Expr::add_expr(cx, cy)));
                }
            }
            _ => {}
        }
        None
    }
);

fn is_cos_minus_sin(expr: &Expr) -> bool {
    if let AstKind::Sub(a, b) = &expr.kind {
        is_cos(a) && is_sin(b)
    } else if let AstKind::Add(a, b) = &expr.kind {
        if let AstKind::Mul(left, right) = &b.kind {
            if matches!(&left.kind, AstKind::Number(n) if *n == -1.0) {
                is_cos(a) && is_sin(right)
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    }
}

fn is_cos_plus_sin(expr: &Expr) -> bool {
    if let AstKind::Add(a, b) = &expr.kind {
        is_cos(a) && is_sin(b)
    } else {
        false
    }
}

fn is_cos(expr: &Expr) -> bool {
    matches!(&expr.kind, AstKind::FunctionCall { name, args } if name == "cos" && args.len() == 1)
}

fn is_sin(expr: &Expr) -> bool {
    matches!(&expr.kind, AstKind::FunctionCall { name, args } if name == "sin" && args.len() == 1)
}

fn get_cos_arg(expr: &Expr) -> Option<Expr> {
    if let AstKind::FunctionCall { name, args } = &expr.kind {
        if name == "cos" && args.len() == 1 {
            Some(args[0].clone())
        } else {
            None
        }
    } else if let AstKind::Add(a, _) = &expr.kind {
        get_cos_arg(a)
    } else if let AstKind::Sub(a, _) = &expr.kind {
        get_cos_arg(a)
    } else {
        None
    }
}

fn get_sin_arg(expr: &Expr) -> Option<Expr> {
    if let AstKind::FunctionCall { name, args } = &expr.kind {
        if name == "sin" && args.len() == 1 {
            Some(args[0].clone())
        } else {
            None
        }
    } else if let AstKind::Add(_, b) = &expr.kind {
        if let AstKind::Mul(left, right) = &b.kind {
            if matches!(&left.kind, AstKind::Number(n) if *n == -1.0) {
                get_sin_arg(right)
            } else {
                None
            }
        } else {
            get_sin_arg(b)
        }
    } else if let AstKind::Sub(_, b) = &expr.kind {
        get_sin_arg(b)
    } else {
        None
    }
}

rule!(
    TrigProductToDoubleAngleRule,
    "trig_product_to_double_angle",
    90,
    Trigonometric,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(a, b) = &expr.kind {
            let (cos_minus_sin, cos_plus_sin) = if is_cos_minus_sin(a) && is_cos_plus_sin(b) {
                (a, b)
            } else if is_cos_minus_sin(b) && is_cos_plus_sin(a) {
                (b, a)
            } else {
                return None;
            };

            if let Some(arg) = get_cos_arg(cos_minus_sin)
                && get_cos_arg(cos_plus_sin) == Some(arg.clone())
                && get_sin_arg(cos_minus_sin) == Some(arg.clone())
                && get_sin_arg(cos_plus_sin) == Some(arg.clone())
            {
                return Some(Expr::func("cos", Expr::mul_expr(Expr::number(2.0), arg)));
            }
        }
        None
    }
);
