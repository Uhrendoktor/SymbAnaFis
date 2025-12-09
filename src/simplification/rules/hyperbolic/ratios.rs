use crate::ast::{Expr, ExprKind as AstKind};
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    SinhCoshToTanhRule,
    "sinh_cosh_to_tanh",
    85,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::FunctionCall {
                name: num_name,
                args: num_args,
            } = &num.kind
            && let AstKind::FunctionCall {
                name: den_name,
                args: den_args,
            } = &den.kind
            && num_name == "sinh"
            && den_name == "cosh"
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Expr::func("tanh", num_args[0].clone()));
        }
        None
    }
);

rule!(
    CoshSinhToCothRule,
    "cosh_sinh_to_coth",
    85,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::FunctionCall {
                name: num_name,
                args: num_args,
            } = &num.kind
            && let AstKind::FunctionCall {
                name: den_name,
                args: den_args,
            } = &den.kind
            && num_name == "cosh"
            && den_name == "sinh"
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Expr::func("coth", num_args[0].clone()));
        }
        None
    }
);

rule!(
    OneCoshToSechRule,
    "one_cosh_to_sech",
    85,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::Number(n) = &num.kind
            && (*n - 1.0).abs() < 1e-10
            && let AstKind::FunctionCall { name, args } = &den.kind
            && name == "cosh"
            && args.len() == 1
        {
            return Some(Expr::func_multi("sech", args.clone()));
        }
        None
    }
);

rule!(
    OneSinhToCschRule,
    "one_sinh_to_csch",
    85,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::Number(n) = &num.kind
            && (*n - 1.0).abs() < 1e-10
            && let AstKind::FunctionCall { name, args } = &den.kind
            && name == "sinh"
            && args.len() == 1
        {
            return Some(Expr::func_multi("csch", args.clone()));
        }
        None
    }
);

rule!(
    OneTanhToCothRule,
    "one_tanh_to_coth",
    85,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::Number(n) = &num.kind
            && (*n - 1.0).abs() < 1e-10
            && let AstKind::FunctionCall { name, args } = &den.kind
            && name == "tanh"
            && args.len() == 1
        {
            return Some(Expr::func_multi("coth", args.clone()));
        }
        None
    }
);
