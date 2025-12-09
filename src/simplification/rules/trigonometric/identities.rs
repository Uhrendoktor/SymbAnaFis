use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::helpers;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    PythagoreanIdentityRule,
    "pythagorean_identity",
    80,
    Trigonometric,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Add(u, v) = &expr.kind {
            if let (AstKind::Pow(sin_base, sin_exp), AstKind::Pow(cos_base, cos_exp)) =
                (&u.kind, &v.kind)
                && matches!(&sin_exp.kind, AstKind::Number(n) if *n == 2.0)
                && matches!(&cos_exp.kind, AstKind::Number(n) if *n == 2.0)
                && let (
                    AstKind::FunctionCall {
                        name: sin_name,
                        args: sin_args,
                    },
                    AstKind::FunctionCall {
                        name: cos_name,
                        args: cos_args,
                    },
                ) = (&sin_base.kind, &cos_base.kind)
                && sin_name == "sin"
                && cos_name == "cos"
                && sin_args.len() == 1
                && cos_args.len() == 1
                && sin_args[0] == cos_args[0]
            {
                return Some(Expr::number(1.0));
            }
            if let (AstKind::Pow(cos_base, cos_exp), AstKind::Pow(sin_base, sin_exp)) =
                (&u.kind, &v.kind)
                && matches!(&cos_exp.kind, AstKind::Number(n) if *n == 2.0)
                && matches!(&sin_exp.kind, AstKind::Number(n) if *n == 2.0)
                && let (
                    AstKind::FunctionCall {
                        name: cos_name,
                        args: cos_args,
                    },
                    AstKind::FunctionCall {
                        name: sin_name,
                        args: sin_args,
                    },
                ) = (&cos_base.kind, &sin_base.kind)
                && cos_name == "cos"
                && sin_name == "sin"
                && cos_args.len() == 1
                && sin_args.len() == 1
                && cos_args[0] == sin_args[0]
            {
                return Some(Expr::number(1.0));
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
        if let AstKind::Sub(lhs, rhs) = &expr.kind
            && matches!(&lhs.kind, AstKind::Number(n) if *n == 1.0)
        {
            if let Some(("cos", arg)) = helpers::get_fn_pow_named(rhs, 2.0) {
                return Some(Expr::pow(Expr::func("sin", arg), Expr::number(2.0)));
            }
            if let Some(("sin", arg)) = helpers::get_fn_pow_named(rhs, 2.0) {
                return Some(Expr::pow(Expr::func("cos", arg), Expr::number(2.0)));
            }
        }

        if let AstKind::Add(lhs, rhs) = &expr.kind {
            if matches!(&rhs.kind, AstKind::Number(n) if *n == 1.0)
                && let AstKind::Mul(coef, rest) = &lhs.kind
                && matches!(&coef.kind, AstKind::Number(n) if *n == -1.0)
            {
                if let Some(("cos", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::pow(Expr::func("sin", arg), Expr::number(2.0)));
                }
                if let Some(("sin", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::pow(Expr::func("cos", arg), Expr::number(2.0)));
                }
            }
            if matches!(&lhs.kind, AstKind::Number(n) if *n == 1.0)
                && let AstKind::Mul(coef, rest) = &rhs.kind
                && matches!(&coef.kind, AstKind::Number(n) if *n == -1.0)
            {
                if let Some(("cos", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::pow(Expr::func("sin", arg), Expr::number(2.0)));
                }
                if let Some(("sin", arg)) = helpers::get_fn_pow_named(rest, 2.0) {
                    return Some(Expr::pow(Expr::func("cos", arg), Expr::number(2.0)));
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
        if let AstKind::Add(lhs, rhs) = &expr.kind {
            if let Some(("tan", arg)) = helpers::get_fn_pow_named(lhs, 2.0)
                && matches!(&rhs.kind, AstKind::Number(n) if *n == 1.0)
            {
                return Some(Expr::pow(Expr::func("sec", arg), Expr::number(2.0)));
            }
            if matches!(&lhs.kind, AstKind::Number(n) if *n == 1.0)
                && let Some(("tan", arg)) = helpers::get_fn_pow_named(rhs, 2.0)
            {
                return Some(Expr::pow(Expr::func("sec", arg), Expr::number(2.0)));
            }
            if let Some(("cot", arg)) = helpers::get_fn_pow_named(lhs, 2.0)
                && matches!(&rhs.kind, AstKind::Number(n) if *n == 1.0)
            {
                return Some(Expr::pow(Expr::func("csc", arg), Expr::number(2.0)));
            }
            if matches!(&lhs.kind, AstKind::Number(n) if *n == 1.0)
                && let Some(("cot", arg)) = helpers::get_fn_pow_named(rhs, 2.0)
            {
                return Some(Expr::pow(Expr::func("csc", arg), Expr::number(2.0)));
            }
        }
        None
    }
);
