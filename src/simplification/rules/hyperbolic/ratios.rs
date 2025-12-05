use crate::ast::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(
    SinhCoshToTanhRule,
    "sinh_cosh_to_tanh",
    85,
    Hyperbolic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Div(num, den) = expr
            && let Expr::FunctionCall {
                name: num_name,
                args: num_args,
            } = &**num
            && let Expr::FunctionCall {
                name: den_name,
                args: den_args,
            } = &**den
            && num_name == "sinh"
            && den_name == "cosh"
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Expr::FunctionCall {
                name: "tanh".to_string(),
                args: vec![num_args[0].clone()],
            });
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
        if let Expr::Div(num, den) = expr
            && let Expr::FunctionCall {
                name: num_name,
                args: num_args,
            } = &**num
            && let Expr::FunctionCall {
                name: den_name,
                args: den_args,
            } = &**den
            && num_name == "cosh"
            && den_name == "sinh"
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Expr::FunctionCall {
                name: "coth".to_string(),
                args: vec![num_args[0].clone()],
            });
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
        if let Expr::Div(num, den) = expr
            && let Expr::Number(n) = &**num
            && (*n - 1.0).abs() < 1e-10
            && let Expr::FunctionCall { name, args } = &**den
            && name == "cosh"
            && args.len() == 1
        {
            return Some(Expr::FunctionCall {
                name: "sech".to_string(),
                args: args.clone(),
            });
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
        if let Expr::Div(num, den) = expr
            && let Expr::Number(n) = &**num
            && (*n - 1.0).abs() < 1e-10
            && let Expr::FunctionCall { name, args } = &**den
            && name == "sinh"
            && args.len() == 1
        {
            return Some(Expr::FunctionCall {
                name: "csch".to_string(),
                args: args.clone(),
            });
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
        if let Expr::Div(num, den) = expr
            && let Expr::Number(n) = &**num
            && (*n - 1.0).abs() < 1e-10
            && let Expr::FunctionCall { name, args } = &**den
            && name == "tanh"
            && args.len() == 1
        {
            return Some(Expr::FunctionCall {
                name: "coth".to_string(),
                args: args.clone(),
            });
        }
        None
    }
);
