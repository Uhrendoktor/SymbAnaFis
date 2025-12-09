use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::helpers;
use crate::simplification::patterns::trigonometric::get_trig_function;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    CofunctionIdentityRule,
    "cofunction_identity",
    85,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && args.len() == 1
        {
            if let AstKind::Sub(lhs, rhs) = &args[0].kind
                && ((if let AstKind::Div(num, den) = &lhs.kind {
                    helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 2.0)
                } else {
                    false
                }) || helpers::approx_eq(
                    helpers::get_numeric_value(lhs),
                    std::f64::consts::PI / 2.0,
                ))
            {
                match name.as_str() {
                    "sin" => {
                        return Some(Expr::func("cos", rhs.as_ref().clone()));
                    }
                    "cos" => {
                        return Some(Expr::func("sin", rhs.as_ref().clone()));
                    }
                    "tan" => {
                        return Some(Expr::func("cot", rhs.as_ref().clone()));
                    }
                    "cot" => {
                        return Some(Expr::func("tan", rhs.as_ref().clone()));
                    }
                    "sec" => {
                        return Some(Expr::func("csc", rhs.as_ref().clone()));
                    }
                    "csc" => {
                        return Some(Expr::func("sec", rhs.as_ref().clone()));
                    }
                    _ => {}
                }
            }

            if let AstKind::Add(u, v) = &args[0].kind {
                let (_angle, other) = if helpers::approx_eq(
                    helpers::get_numeric_value(u),
                    std::f64::consts::PI / 2.0,
                ) {
                    (u, v)
                } else if helpers::approx_eq(
                    helpers::get_numeric_value(v),
                    std::f64::consts::PI / 2.0,
                ) {
                    (v, u)
                } else {
                    let is_pi_div_2 = |e: &Expr| {
                        if let AstKind::Div(num, den) = &e.kind {
                            helpers::is_pi(num)
                                && matches!(&den.kind, AstKind::Number(n) if *n == 2.0)
                        } else {
                            false
                        }
                    };

                    if is_pi_div_2(u) {
                        (u, v)
                    } else if is_pi_div_2(v) {
                        (v, u)
                    } else {
                        return None;
                    }
                };

                if let AstKind::Mul(c, x) = &other.kind
                    && matches!(&c.kind, AstKind::Number(n) if *n == -1.0)
                {
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::func("cos", x.as_ref().clone()));
                        }
                        "cos" => {
                            return Some(Expr::func("sin", x.as_ref().clone()));
                        }
                        "tan" => {
                            return Some(Expr::func("cot", x.as_ref().clone()));
                        }
                        "cot" => {
                            return Some(Expr::func("tan", x.as_ref().clone()));
                        }
                        "sec" => {
                            return Some(Expr::func("csc", x.as_ref().clone()));
                        }
                        "csc" => {
                            return Some(Expr::func("sec", x.as_ref().clone()));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

rule!(
    TrigPeriodicityRule,
    "trig_periodicity",
    85,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && (name == "sin" || name == "cos")
            && args.len() == 1
        {
            if let AstKind::Add(lhs, rhs) = &args[0].kind {
                if helpers::is_multiple_of_two_pi(rhs) {
                    return Some(Expr::func(name.clone(), lhs.as_ref().clone()));
                }
                if helpers::is_multiple_of_two_pi(lhs) {
                    return Some(Expr::func(name.clone(), rhs.as_ref().clone()));
                }
            }
            if let AstKind::Sub(lhs, rhs) = &args[0].kind {
                if helpers::is_multiple_of_two_pi(rhs) {
                    return Some(Expr::func(name.clone(), lhs.as_ref().clone()));
                }
                if helpers::is_multiple_of_two_pi(lhs) {
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::mul_expr(
                                Expr::number(-1.0),
                                Expr::func("sin", rhs.as_ref().clone()),
                            ));
                        }
                        "cos" => {
                            return Some(Expr::func("cos", rhs.as_ref().clone()));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

rule!(
    TrigReflectionRule,
    "trig_reflection",
    80,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && (name == "sin" || name == "cos")
            && args.len() == 1
        {
            if let AstKind::Sub(lhs, rhs) = &args[0].kind
                && helpers::is_pi(lhs)
            {
                match name.as_str() {
                    "sin" => {
                        return Some(Expr::func("sin", rhs.as_ref().clone()));
                    }
                    "cos" => {
                        return Some(Expr::mul_expr(
                            Expr::number(-1.0),
                            Expr::func("cos", rhs.as_ref().clone()),
                        ));
                    }
                    _ => {}
                }
            }
            if let AstKind::Add(u, v) = &args[0].kind {
                let (_, other_term) = if helpers::is_pi(u) {
                    (u, v)
                } else if helpers::is_pi(v) {
                    (v, u)
                } else {
                    return None;
                };

                let mut is_neg_x = false;
                if let AstKind::Mul(c, x) = &other_term.kind
                    && matches!(&c.kind, AstKind::Number(n) if *n == -1.0)
                {
                    is_neg_x = true;
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::func("sin", x.as_ref().clone()));
                        }
                        "cos" => {
                            return Some(Expr::mul_expr(
                                Expr::number(-1.0),
                                Expr::func("cos", x.as_ref().clone()),
                            ));
                        }
                        _ => {}
                    }
                }

                if !is_neg_x {
                    match name.as_str() {
                        "sin" | "cos" => {
                            return Some(Expr::mul_expr(
                                Expr::number(-1.0),
                                Expr::func(name.clone(), other_term.as_ref().clone()),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

rule!(
    TrigThreePiOverTwoRule,
    "trig_three_pi_over_two",
    80,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && (name == "sin" || name == "cos")
            && args.len() == 1
        {
            if let AstKind::Sub(lhs, rhs) = &args[0].kind
                && helpers::is_three_pi_over_two(lhs)
            {
                match name.as_str() {
                    "sin" => {
                        return Some(Expr::mul_expr(
                            Expr::number(-1.0),
                            Expr::func("cos", rhs.as_ref().clone()),
                        ));
                    }
                    "cos" => {
                        return Some(Expr::mul_expr(
                            Expr::number(-1.0),
                            Expr::func("sin", rhs.as_ref().clone()),
                        ));
                    }
                    _ => {}
                }
            }

            if let AstKind::Add(u, v) = &args[0].kind {
                let (_angle, other) = if helpers::is_three_pi_over_two(u) {
                    (u, v)
                } else if helpers::is_three_pi_over_two(v) {
                    (v, u)
                } else {
                    return None;
                };

                if let AstKind::Mul(c, x) = &other.kind
                    && matches!(&c.kind, AstKind::Number(n) if *n == -1.0)
                {
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::mul_expr(
                                Expr::number(-1.0),
                                Expr::func("cos", x.as_ref().clone()),
                            ));
                        }
                        "cos" => {
                            return Some(Expr::mul_expr(
                                Expr::number(-1.0),
                                Expr::func("sin", x.as_ref().clone()),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

rule!(
    TrigNegArgRule,
    "trig_neg_arg",
    90,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Some((name, arg)) = get_trig_function(expr)
            && let AstKind::Mul(coeff, inner) = &arg.kind
            && let AstKind::Number(n) = &coeff.kind
            && *n == -1.0
        {
            match name {
                "sin" | "tan" => {
                    return Some(Expr::mul_expr(
                        Expr::number(-1.0),
                        Expr::func(name.to_string(), inner.as_ref().clone()),
                    ));
                }
                "cos" | "sec" => {
                    return Some(Expr::func(name.to_string(), inner.as_ref().clone()));
                }
                _ => {}
            }
        }
        None
    }
);
