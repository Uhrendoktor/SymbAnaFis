use crate::ast::Expr;
use crate::simplification::helpers;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    PythagoreanIdentityRule,
    "pythagorean_identity",
    80,
    Trigonometric,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Add(u, v) = expr {
            if let (Expr::Pow(sin_base, sin_exp), Expr::Pow(cos_base, cos_exp)) = (&**u, &**v)
                && matches!(**sin_exp, Expr::Number(n) if n == 2.0)
                && matches!(**cos_exp, Expr::Number(n) if n == 2.0)
                && let (
                    Expr::FunctionCall {
                        name: sin_name,
                        args: sin_args,
                    },
                    Expr::FunctionCall {
                        name: cos_name,
                        args: cos_args,
                    },
                ) = (&**sin_base, &**cos_base)
                && sin_name == "sin"
                && cos_name == "cos"
                && sin_args.len() == 1
                && cos_args.len() == 1
                && sin_args[0] == cos_args[0]
            {
                return Some(Expr::Number(1.0));
            }
            if let (Expr::Pow(cos_base, cos_exp), Expr::Pow(sin_base, sin_exp)) = (&**u, &**v)
                && matches!(**cos_exp, Expr::Number(n) if n == 2.0)
                && matches!(**sin_exp, Expr::Number(n) if n == 2.0)
                && let (
                    Expr::FunctionCall {
                        name: cos_name,
                        args: cos_args,
                    },
                    Expr::FunctionCall {
                        name: sin_name,
                        args: sin_args,
                    },
                ) = (&**cos_base, &**sin_base)
                && cos_name == "cos"
                && sin_name == "sin"
                && cos_args.len() == 1
                && sin_args.len() == 1
                && cos_args[0] == sin_args[0]
            {
                return Some(Expr::Number(1.0));
            }
        }
        None
    }
);

rule!(
    PythagoreanComplementsRule,
    "pythagorean_complements",
    70,
    Trigonometric,
    &[ExprKind::Add, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Sub(lhs, rhs) = expr
            && matches!(**lhs, Expr::Number(n) if n == 1.0)
        {
            if let Some(("cos", arg)) = helpers::get_fn_pow_named(rhs, 2.0) {
                return Some(Expr::Pow(
                    Rc::new(Expr::FunctionCall {
                        name: "sin".to_string(),
                        args: vec![arg],
                    }),
                    Rc::new(Expr::Number(2.0)),
                ));
            }
            if let Some(("sin", arg)) = helpers::get_fn_pow_named(rhs, 2.0) {
                return Some(Expr::Pow(
                    Rc::new(Expr::FunctionCall {
                        name: "cos".to_string(),
                        args: vec![arg],
                    }),
                    Rc::new(Expr::Number(2.0)),
                ));
            }
        }

        if let Expr::Add(lhs, rhs) = expr {
            if matches!(**rhs, Expr::Number(n) if n == 1.0)
                && let Expr::Mul(coef, rest) = &**lhs
                && matches!(**coef, Expr::Number(n) if n == -1.0)
            {
                if let Some(("cos", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::Pow(
                        Rc::new(Expr::FunctionCall {
                            name: "sin".to_string(),
                            args: vec![arg],
                        }),
                        Rc::new(Expr::Number(2.0)),
                    ));
                }
                if let Some(("sin", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::Pow(
                        Rc::new(Expr::FunctionCall {
                            name: "cos".to_string(),
                            args: vec![arg],
                        }),
                        Rc::new(Expr::Number(2.0)),
                    ));
                }
            }
            if matches!(**lhs, Expr::Number(n) if n == 1.0)
                && let Expr::Mul(coef, rest) = &**rhs
                && matches!(**coef, Expr::Number(n) if n == -1.0)
            {
                if let Some(("cos", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::Pow(
                        Rc::new(Expr::FunctionCall {
                            name: "sin".to_string(),
                            args: vec![arg],
                        }),
                        Rc::new(Expr::Number(2.0)),
                    ));
                }
                if let Some(("sin", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::Pow(
                        Rc::new(Expr::FunctionCall {
                            name: "cos".to_string(),
                            args: vec![arg],
                        }),
                        Rc::new(Expr::Number(2.0)),
                    ));
                }
            }
        }

        None
    }
);

rule!(
    PythagoreanTangentRule,
    "pythagorean_tangent",
    70,
    Trigonometric,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Add(lhs, rhs) = expr {
            if let Some(("tan", arg)) = helpers::get_fn_pow_named(lhs, 2.0)
                && matches!(**rhs, Expr::Number(n) if n == 1.0)
            {
                return Some(Expr::Pow(
                    Rc::new(Expr::FunctionCall {
                        name: "sec".to_string(),
                        args: vec![arg],
                    }),
                    Rc::new(Expr::Number(2.0)),
                ));
            }
            if matches!(**lhs, Expr::Number(n) if n == 1.0)
                && let Some(("tan", arg)) = helpers::get_fn_pow_named(rhs, 2.0)
            {
                return Some(Expr::Pow(
                    Rc::new(Expr::FunctionCall {
                        name: "sec".to_string(),
                        args: vec![arg],
                    }),
                    Rc::new(Expr::Number(2.0)),
                ));
            }
            if let Some(("cot", arg)) = helpers::get_fn_pow_named(lhs, 2.0)
                && matches!(**rhs, Expr::Number(n) if n == 1.0)
            {
                return Some(Expr::Pow(
                    Rc::new(Expr::FunctionCall {
                        name: "csc".to_string(),
                        args: vec![arg],
                    }),
                    Rc::new(Expr::Number(2.0)),
                ));
            }
            if matches!(**lhs, Expr::Number(n) if n == 1.0)
                && let Some(("cot", arg)) = helpers::get_fn_pow_named(rhs, 2.0)
            {
                return Some(Expr::Pow(
                    Rc::new(Expr::FunctionCall {
                        name: "csc".to_string(),
                        args: vec![arg],
                    }),
                    Rc::new(Expr::Number(2.0)),
                ));
            }
        }
        None
    }
);
