use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};

rule!(ExpLnRule, "exp_ln", 80, Algebraic, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let AstKind::FunctionCall { name, args } = &expr.kind
        && name == "exp"
        && args.len() == 1
        && let AstKind::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0].kind
        && inner_name == "ln"
        && inner_args.len() == 1
    {
        return Some(inner_args[0].clone());
    }
    None
});

rule!(LnExpRule, "ln_exp", 80, Algebraic, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let AstKind::FunctionCall { name, args } = &expr.kind
        && name == "ln"
        && args.len() == 1
        && let AstKind::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0].kind
        && inner_name == "exp"
        && inner_args.len() == 1
    {
        return Some(inner_args[0].clone());
    }
    None
});

rule!(ExpMulLnRule, "exp_mul_ln", 80, Algebraic, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let AstKind::FunctionCall { name, args } = &expr.kind
        && name == "exp"
        && args.len() == 1
        && let AstKind::Mul(a, b) = &args[0].kind
    {
        // Check if b is ln(x)
        if let AstKind::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &b.kind
            && inner_name == "ln"
            && inner_args.len() == 1
        {
            return Some(Expr::pow(inner_args[0].clone(), a.as_ref().clone()));
        }
        // Check if a is ln(x) (commutative)
        if let AstKind::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &a.kind
            && inner_name == "ln"
            && inner_args.len() == 1
        {
            return Some(Expr::pow(inner_args[0].clone(), b.as_ref().clone()));
        }
    }
    None
});

rule!(EPowLnRule, "e_pow_ln", 85, Algebraic, &[ExprKind::Pow], alters_domain: true, |expr: &Expr, context: &RuleContext| {
    if let AstKind::Pow(base, exp) = &expr.kind {
        // Check if base is Symbol("e") AND "e" is not a user-specified fixed variable
        if let AstKind::Symbol(s) = &base.kind
            && s == "e"
            && !context.fixed_vars.contains("e")
        {
            // Check if exponent is ln(x)
            if let AstKind::FunctionCall { name, args } = &exp.kind
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
    if let AstKind::Pow(base, exp) = &expr.kind {
        // Check if base is Symbol("e") AND "e" is not a user-specified fixed variable
        if let AstKind::Symbol(s) = &base.kind
            && s == "e"
            && !context.fixed_vars.contains("e")
        {
            // Check if exponent is a*ln(b) or ln(b)*a
            if let AstKind::Mul(a, b) = &exp.kind {
                // Check if b is ln(x)
                if let AstKind::FunctionCall {
                    name: inner_name,
                    args: inner_args,
                } = &b.kind
                    && inner_name == "ln"
                    && inner_args.len() == 1
                {
                    return Some(Expr::pow(inner_args[0].clone(), a.as_ref().clone()));
                }
                // Check if a is ln(x) (commutative)
                if let AstKind::FunctionCall {
                    name: inner_name,
                    args: inner_args,
                } = &a.kind
                    && inner_name == "ln"
                    && inner_args.len() == 1
                {
                    return Some(Expr::pow(inner_args[0].clone(), b.as_ref().clone()));
                }
            }
        }
    }
    None
});
