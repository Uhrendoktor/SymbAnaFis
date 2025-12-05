use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(ExpLnRule, "exp_ln", 80, Algebraic, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::FunctionCall { name, args } = expr
        && name == "exp"
        && args.len() == 1
        && let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0]
        && inner_name == "ln"
        && inner_args.len() == 1
    {
        return Some(inner_args[0].clone());
    }
    None
});

rule!(LnExpRule, "ln_exp", 80, Algebraic, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::FunctionCall { name, args } = expr
        && name == "ln"
        && args.len() == 1
        && let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0]
        && inner_name == "exp"
        && inner_args.len() == 1
    {
        return Some(inner_args[0].clone());
    }
    None
});

rule!(ExpMulLnRule, "exp_mul_ln", 80, Algebraic, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::FunctionCall { name, args } = expr
        && name == "exp"
        && args.len() == 1
        && let Expr::Mul(a, b) = &args[0]
    {
        // Check if b is ln(x)
        if let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &**b
            && inner_name == "ln"
            && inner_args.len() == 1
        {
            return Some(Expr::Pow(Rc::new(inner_args[0].clone()), a.clone()));
        }
        // Check if a is ln(x) (commutative)
        if let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &**a
            && inner_name == "ln"
            && inner_args.len() == 1
        {
            return Some(Expr::Pow(Rc::new(inner_args[0].clone()), b.clone()));
        }
    }
    None
});

rule!(EPowLnRule, "e_pow_ln", 85, Algebraic, &[ExprKind::Pow], alters_domain: true, |expr: &Expr, context: &RuleContext| {
    if let Expr::Pow(base, exp) = expr {
        // Check if base is Symbol("e") AND "e" is not a user-specified fixed variable
        if let Expr::Symbol(ref s) = **base
            && s == "e"
            && !context.fixed_vars.contains("e")
        {
            // Check if exponent is ln(x)
            if let Expr::FunctionCall { name, args } = &**exp
                && name == "ln"
                && args.len() == 1
            {
                return Some(args[0].clone());
            }
        }
    }
    None
});

rule!(EPowMulLnRule, "e_pow_mul_ln", 85, Algebraic, &[ExprKind::Pow], alters_domain: true, |expr: &Expr, context: &RuleContext| {
    if let Expr::Pow(base, exp) = expr {
        // Check if base is Symbol("e") AND "e" is not a user-specified fixed variable
        if let Expr::Symbol(ref s) = **base
            && s == "e"
            && !context.fixed_vars.contains("e")
        {
            // Check if exponent is a*ln(b) or ln(b)*a
            if let Expr::Mul(a, b) = &**exp {
                // Check if b is ln(x)
                if let Expr::FunctionCall {
                    name: inner_name,
                    args: inner_args,
                } = &**b
                    && inner_name == "ln"
                    && inner_args.len() == 1
                {
                    return Some(Expr::Pow(Rc::new(inner_args[0].clone()), a.clone()));
                }
                // Check if a is ln(x) (commutative)
                if let Expr::FunctionCall {
                    name: inner_name,
                    args: inner_args,
                } = &**a
                    && inner_name == "ln"
                    && inner_args.len() == 1
                {
                    return Some(Expr::Pow(Rc::new(inner_args[0].clone()), b.clone()));
                }
            }
        }
    }
    None
});
