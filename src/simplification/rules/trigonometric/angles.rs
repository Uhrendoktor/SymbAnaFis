use crate::ast::Expr;
use crate::simplification::helpers;
use crate::simplification::patterns::common::extract_coefficient;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    TrigDoubleAngleRule,
    "trig_double_angle",
    85,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "sin"
            && args.len() == 1
        {
            let (coeff, rest) = extract_coefficient(&args[0]);
            if coeff == 2.0 {
                return Some(Expr::Mul(
                    Rc::new(Expr::Number(2.0)),
                    Rc::new(Expr::Mul(
                        Rc::new(Expr::FunctionCall {
                            name: "sin".to_string(),
                            args: vec![rest.clone()],
                        }),
                        Rc::new(Expr::FunctionCall {
                            name: "cos".to_string(),
                            args: vec![rest],
                        }),
                    )),
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
        let (pos, neg) = match expr {
            Expr::Sub(u, v) => (u, v),
            Expr::Add(u, v) => {
                if let Expr::Mul(c, inner) = &**u {
                    if matches!(**c, Expr::Number(n) if n == -1.0) {
                        (v, inner)
                    } else if let Expr::Mul(c, inner) = &**v {
                        if matches!(**c, Expr::Number(n) if n == -1.0) {
                            (u, inner)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else if let Expr::Mul(c, inner) = &**v {
                    if matches!(**c, Expr::Number(n) if n == -1.0) {
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
            return Some(Expr::FunctionCall {
                name: "cos".to_string(),
                args: vec![Expr::Mul(Rc::new(Expr::Number(2.0)), Rc::new(arg1))],
            });
        }

        if let Some(("sin", arg1)) = helpers::get_fn_pow_named(pos, 2.0)
            && let Some(("cos", arg2)) = helpers::get_fn_pow_named(neg, 2.0)
            && arg1 == arg2
        {
            return Some(Expr::Mul(
                Rc::new(Expr::Number(-1.0)),
                Rc::new(Expr::FunctionCall {
                    name: "cos".to_string(),
                    args: vec![Expr::Mul(Rc::new(Expr::Number(2.0)), Rc::new(arg1))],
                }),
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
        match expr {
            Expr::Add(u, v) => {
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
                    return Some(Expr::FunctionCall {
                        name: "sin".to_string(),
                        args: vec![Expr::Add(Rc::new(x), Rc::new(y))],
                    });
                }
            }
            Expr::Sub(u, v) => {
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
                    return Some(Expr::FunctionCall {
                        name: "sin".to_string(),
                        args: vec![Expr::Sub(Rc::new(x), Rc::new(y))],
                    });
                }
                if let Some((cx, cy)) = helpers::get_product_fn_args(u, "cos", "cos")
                    && let Some((sx, sy)) = helpers::get_product_fn_args(v, "sin", "sin")
                    && ((cx == sx && cy == sy) || (cx == sy && cy == sx))
                {
                    return Some(Expr::FunctionCall {
                        name: "cos".to_string(),
                        args: vec![Expr::Add(Rc::new(cx), Rc::new(cy))],
                    });
                }
            }
            _ => {}
        }
        None
    }
);

fn is_cos_minus_sin(expr: &Expr) -> bool {
    if let Expr::Sub(a, b) = expr {
        is_cos(a) && is_sin(b)
    } else if let Expr::Add(a, b) = expr {
        if let Expr::Mul(left, right) = &**b {
            if matches!(**left, Expr::Number(n) if n == -1.0) {
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
    if let Expr::Add(a, b) = expr {
        is_cos(a) && is_sin(b)
    } else {
        false
    }
}

fn is_cos(expr: &Expr) -> bool {
    matches!(expr, Expr::FunctionCall { name, args } if name == "cos" && args.len() == 1)
}

fn is_sin(expr: &Expr) -> bool {
    matches!(expr, Expr::FunctionCall { name, args } if name == "sin" && args.len() == 1)
}

fn get_cos_arg(expr: &Expr) -> Option<Expr> {
    if let Expr::FunctionCall { name, args } = expr {
        if name == "cos" && args.len() == 1 {
            Some(args[0].clone())
        } else {
            None
        }
    } else if let Expr::Add(a, _) = expr {
        get_cos_arg(a)
    } else if let Expr::Sub(a, _) = expr {
        get_cos_arg(a)
    } else {
        None
    }
}

fn get_sin_arg(expr: &Expr) -> Option<Expr> {
    if let Expr::FunctionCall { name, args } = expr {
        if name == "sin" && args.len() == 1 {
            Some(args[0].clone())
        } else {
            None
        }
    } else if let Expr::Add(_, b) = expr {
        if let Expr::Mul(left, right) = &**b {
            if matches!(**left, Expr::Number(n) if n == -1.0) {
                get_sin_arg(right)
            } else {
                None
            }
        } else {
            get_sin_arg(b)
        }
    } else if let Expr::Sub(_, b) = expr {
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
        if let Expr::Mul(a, b) = expr {
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
                return Some(Expr::FunctionCall {
                    name: "cos".to_string(),
                    args: vec![Expr::Mul(Rc::new(Expr::Number(2.0)), Rc::new(arg))],
                });
            }
        }
        None
    }
);
