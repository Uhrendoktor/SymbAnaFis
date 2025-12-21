use crate::core::expr::{Expr, ExprKind as AstKind};
use crate::simplification::helpers;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::f64::consts::PI;

rule!(
    SinZeroRule,
    "sin_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sin"
            && args.len() == 1
            && matches!(&args[0].kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    CosZeroRule,
    "cos_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cos"
            && args.len() == 1
            && matches!(&args[0].kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Expr::number(1.0));
        }
        None
    }
);

rule!(
    TanZeroRule,
    "tan_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "tan"
            && args.len() == 1
            && matches!(&args[0].kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    SinPiRule,
    "sin_pi",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sin"
            && args.len() == 1
            && helpers::is_pi(&args[0])
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    CosPiRule,
    "cos_pi",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cos"
            && args.len() == 1
            && helpers::is_pi(&args[0])
        {
            return Some(Expr::number(-1.0));
        }
        None
    }
);

rule!(
    SinPiOverTwoRule,
    "sin_pi_over_two",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "sin"
            && args.len() == 1
            && let AstKind::Div(num, den) = &args[0].kind
            && helpers::is_pi(num)
            && matches!(&den.kind, AstKind::Number(n) if *n == 2.0)
        {
            return Some(Expr::number(1.0));
        }
        None
    }
);

rule!(
    CosPiOverTwoRule,
    "cos_pi_over_two",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name == "cos"
            && args.len() == 1
            && let AstKind::Div(num, den) = &args[0].kind
            && helpers::is_pi(num)
            && matches!(&den.kind, AstKind::Number(n) if *n == 2.0)
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    TrigExactValuesRule,
    "trig_exact_values",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && args.len() == 1
        {
            let arg = &args[0];
            let arg_val = helpers::get_numeric_value(arg);
            let is_numeric_input = matches!(arg.kind, AstKind::Number(_));

            match name.as_str() {
                "sin" => {
                    if helpers::approx_eq(arg_val, PI / 6.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 6.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::number(0.5))
                        } else {
                            Some(Expr::div_expr(Expr::number(1.0), Expr::number(2.0)))
                        };
                    }
                    if helpers::approx_eq(arg_val, PI / 4.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 4.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::number((2.0f64).sqrt() / 2.0))
                        } else {
                            Some(Expr::div_expr(
                                Expr::func("sqrt", Expr::number(2.0)),
                                Expr::number(2.0),
                            ))
                        };
                    }
                }
                "cos" => {
                    if helpers::approx_eq(arg_val, PI / 3.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 3.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::number(0.5))
                        } else {
                            Some(Expr::div_expr(Expr::number(1.0), Expr::number(2.0)))
                        };
                    }
                    if helpers::approx_eq(arg_val, PI / 4.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 4.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::number((2.0f64).sqrt() / 2.0))
                        } else {
                            Some(Expr::div_expr(
                                Expr::func("sqrt", Expr::number(2.0)),
                                Expr::number(2.0),
                            ))
                        };
                    }
                }
                "tan" => {
                    if helpers::approx_eq(arg_val, PI / 4.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 4.0)))
                    {
                        return Some(Expr::number(1.0));
                    }
                    if helpers::approx_eq(arg_val, PI / 3.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 3.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::number((3.0f64).sqrt()))
                        } else {
                            Some(Expr::func("sqrt", Expr::number(3.0)))
                        };
                    }
                    if helpers::approx_eq(arg_val, PI / 6.0)
                        || (matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 6.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::number(1.0 / (3.0f64).sqrt()))
                        } else {
                            Some(Expr::div_expr(
                                Expr::func("sqrt", Expr::number(3.0)),
                                Expr::number(3.0),
                            ))
                        };
                    }
                }
                _ => {}
            }
        }
        None
    }
);

// Ratio rules: convert 1/trig to reciprocal functions
rule!(
    OneCosToSecRule,
    "one_cos_to_sec",
    85,
    Trigonometric,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::Number(n) = &num.kind
            && (*n - 1.0).abs() < 1e-10
        {
            // 1/cos(x) → sec(x)
            if let AstKind::FunctionCall { name, args } = &den.kind
                && name == "cos"
                && args.len() == 1
            {
                return Some(Expr::func("sec", (*args[0]).clone()));
            }
            // 1/cos(x)^n → sec(x)^n
            if let AstKind::Pow(base, exp) = &den.kind
                && let AstKind::FunctionCall { name, args } = &base.kind
                && name == "cos"
                && args.len() == 1
            {
                return Some(Expr::pow(
                    Expr::func("sec", (*args[0]).clone()),
                    (**exp).clone(),
                ));
            }
        }
        None
    }
);

rule!(
    OneSinToCscRule,
    "one_sin_to_csc",
    85,
    Trigonometric,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::Number(n) = &num.kind
            && (*n - 1.0).abs() < 1e-10
        {
            // 1/sin(x) → csc(x)
            if let AstKind::FunctionCall { name, args } = &den.kind
                && name == "sin"
                && args.len() == 1
            {
                return Some(Expr::func("csc", (*args[0]).clone()));
            }
            // 1/sin(x)^n → csc(x)^n
            if let AstKind::Pow(base, exp) = &den.kind
                && let AstKind::FunctionCall { name, args } = &base.kind
                && name == "sin"
                && args.len() == 1
            {
                return Some(Expr::pow(
                    Expr::func("csc", (*args[0]).clone()),
                    (**exp).clone(),
                ));
            }
        }
        None
    }
);

rule!(
    SinCosToTanRule,
    "sin_cos_to_tan",
    85,
    Trigonometric,
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
            && num_name == "sin"
            && den_name == "cos"
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Expr::func("tan", (*num_args[0]).clone()));
        }
        None
    }
);

rule!(
    CosSinToCotRule,
    "cos_sin_to_cot",
    85,
    Trigonometric,
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
            && num_name == "cos"
            && den_name == "sin"
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Expr::func("cot", (*num_args[0]).clone()));
        }
        None
    }
);
