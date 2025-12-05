use crate::ast::Expr;
use crate::simplification::helpers;
use crate::simplification::patterns::trigonometric::get_trig_function;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    CofunctionIdentityRule,
    "cofunction_identity",
    85,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && args.len() == 1
        {
            if let Expr::Sub(lhs, rhs) = &args[0]
                && ((if let Expr::Div(num, den) = &**lhs {
                    helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 2.0)
                } else {
                    false
                }) || helpers::approx_eq(
                    helpers::get_numeric_value(lhs),
                    std::f64::consts::PI / 2.0,
                ))
            {
                match name.as_str() {
                    "sin" => {
                        return Some(Expr::FunctionCall {
                            name: "cos".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    "cos" => {
                        return Some(Expr::FunctionCall {
                            name: "sin".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    "tan" => {
                        return Some(Expr::FunctionCall {
                            name: "cot".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    "cot" => {
                        return Some(Expr::FunctionCall {
                            name: "tan".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    "sec" => {
                        return Some(Expr::FunctionCall {
                            name: "csc".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    "csc" => {
                        return Some(Expr::FunctionCall {
                            name: "sec".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    _ => {}
                }
            }

            if let Expr::Add(u, v) = &args[0] {
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
                        if let Expr::Div(num, den) = e {
                            helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 2.0)
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

                if let Expr::Mul(c, x) = &**other
                    && matches!(**c, Expr::Number(n) if n == -1.0)
                {
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::FunctionCall {
                                name: "cos".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
                        }
                        "cos" => {
                            return Some(Expr::FunctionCall {
                                name: "sin".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
                        }
                        "tan" => {
                            return Some(Expr::FunctionCall {
                                name: "cot".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
                        }
                        "cot" => {
                            return Some(Expr::FunctionCall {
                                name: "tan".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
                        }
                        "sec" => {
                            return Some(Expr::FunctionCall {
                                name: "csc".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
                        }
                        "csc" => {
                            return Some(Expr::FunctionCall {
                                name: "sec".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
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
        if let Expr::FunctionCall { name, args } = expr
            && (name == "sin" || name == "cos")
            && args.len() == 1
        {
            if let Expr::Add(lhs, rhs) = &args[0] {
                if helpers::is_multiple_of_two_pi(rhs) {
                    return Some(Expr::FunctionCall {
                        name: name.clone(),
                        args: vec![lhs.as_ref().clone()],
                    });
                }
                if helpers::is_multiple_of_two_pi(lhs) {
                    return Some(Expr::FunctionCall {
                        name: name.clone(),
                        args: vec![rhs.as_ref().clone()],
                    });
                }
            }
            if let Expr::Sub(lhs, rhs) = &args[0] {
                if helpers::is_multiple_of_two_pi(rhs) {
                    return Some(Expr::FunctionCall {
                        name: name.clone(),
                        args: vec![lhs.as_ref().clone()],
                    });
                }
                if helpers::is_multiple_of_two_pi(lhs) {
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::Mul(
                                Rc::new(Expr::Number(-1.0)),
                                Rc::new(Expr::FunctionCall {
                                    name: "sin".to_string(),
                                    args: vec![rhs.as_ref().clone()],
                                }),
                            ));
                        }
                        "cos" => {
                            return Some(Expr::FunctionCall {
                                name: "cos".to_string(),
                                args: vec![rhs.as_ref().clone()],
                            });
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
        if let Expr::FunctionCall { name, args } = expr
            && (name == "sin" || name == "cos")
            && args.len() == 1
        {
            if let Expr::Sub(lhs, rhs) = &args[0]
                && helpers::is_pi(lhs)
            {
                match name.as_str() {
                    "sin" => {
                        return Some(Expr::FunctionCall {
                            name: "sin".to_string(),
                            args: vec![rhs.as_ref().clone()],
                        });
                    }
                    "cos" => {
                        return Some(Expr::Mul(
                            Rc::new(Expr::Number(-1.0)),
                            Rc::new(Expr::FunctionCall {
                                name: "cos".to_string(),
                                args: vec![rhs.as_ref().clone()],
                            }),
                        ));
                    }
                    _ => {}
                }
            }
            if let Expr::Add(u, v) = &args[0] {
                let (_, other_term) = if helpers::is_pi(u) {
                    (u, v)
                } else if helpers::is_pi(v) {
                    (v, u)
                } else {
                    return None;
                };

                let mut is_neg_x = false;
                if let Expr::Mul(c, x) = &**other_term
                    && matches!(**c, Expr::Number(n) if n == -1.0)
                {
                    is_neg_x = true;
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::FunctionCall {
                                name: "sin".to_string(),
                                args: vec![x.as_ref().clone()],
                            });
                        }
                        "cos" => {
                            return Some(Expr::Mul(
                                Rc::new(Expr::Number(-1.0)),
                                Rc::new(Expr::FunctionCall {
                                    name: "cos".to_string(),
                                    args: vec![x.as_ref().clone()],
                                }),
                            ));
                        }
                        _ => {}
                    }
                }

                if !is_neg_x {
                    match name.as_str() {
                        "sin" | "cos" => {
                            return Some(Expr::Mul(
                                Rc::new(Expr::Number(-1.0)),
                                Rc::new(Expr::FunctionCall {
                                    name: name.clone(),
                                    args: vec![other_term.as_ref().clone()],
                                }),
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
        if let Expr::FunctionCall { name, args } = expr
            && (name == "sin" || name == "cos")
            && args.len() == 1
        {
            if let Expr::Sub(lhs, rhs) = &args[0]
                && helpers::is_three_pi_over_two(lhs)
            {
                match name.as_str() {
                    "sin" => {
                        return Some(Expr::Mul(
                            Rc::new(Expr::Number(-1.0)),
                            Rc::new(Expr::FunctionCall {
                                name: "cos".to_string(),
                                args: vec![rhs.as_ref().clone()],
                            }),
                        ));
                    }
                    "cos" => {
                        return Some(Expr::Mul(
                            Rc::new(Expr::Number(-1.0)),
                            Rc::new(Expr::FunctionCall {
                                name: "sin".to_string(),
                                args: vec![rhs.as_ref().clone()],
                            }),
                        ));
                    }
                    _ => {}
                }
            }

            if let Expr::Add(u, v) = &args[0] {
                let (_angle, other) = if helpers::is_three_pi_over_two(u) {
                    (u, v)
                } else if helpers::is_three_pi_over_two(v) {
                    (v, u)
                } else {
                    return None;
                };

                if let Expr::Mul(c, x) = &**other
                    && matches!(**c, Expr::Number(n) if n == -1.0)
                {
                    match name.as_str() {
                        "sin" => {
                            return Some(Expr::Mul(
                                Rc::new(Expr::Number(-1.0)),
                                Rc::new(Expr::FunctionCall {
                                    name: "cos".to_string(),
                                    args: vec![x.as_ref().clone()],
                                }),
                            ));
                        }
                        "cos" => {
                            return Some(Expr::Mul(
                                Rc::new(Expr::Number(-1.0)),
                                Rc::new(Expr::FunctionCall {
                                    name: "sin".to_string(),
                                    args: vec![x.as_ref().clone()],
                                }),
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
            && let Expr::Mul(coeff, inner) = &arg
            && let Expr::Number(n) = **coeff
            && n == -1.0
        {
            match name {
                "sin" | "tan" => {
                    return Some(Expr::Mul(
                        Rc::new(Expr::Number(-1.0)),
                        Rc::new(Expr::FunctionCall {
                            name: name.to_string(),
                            args: vec![inner.as_ref().clone()],
                        }),
                    ));
                }
                "cos" | "sec" => {
                    return Some(Expr::FunctionCall {
                        name: name.to_string(),
                        args: vec![inner.as_ref().clone()],
                    });
                }
                _ => {}
            }
        }
        None
    }
);
