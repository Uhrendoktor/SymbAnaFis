use crate::ast::Expr;
use crate::simplification::helpers;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::f64::consts::PI;
use std::rc::Rc;

rule!(
    SinZeroRule,
    "sin_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "sin"
            && args.len() == 1
            && matches!(args[0], Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(0.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && name == "cos"
            && args.len() == 1
            && matches!(args[0], Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(1.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && name == "tan"
            && args.len() == 1
            && matches!(args[0], Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(0.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && name == "sin"
            && args.len() == 1
            && helpers::is_pi(&args[0])
        {
            return Some(Expr::Number(0.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && name == "cos"
            && args.len() == 1
            && helpers::is_pi(&args[0])
        {
            return Some(Expr::Number(-1.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && name == "sin"
            && args.len() == 1
            && let Expr::Div(num, den) = &args[0]
            && helpers::is_pi(num)
            && matches!(**den, Expr::Number(n) if n == 2.0)
        {
            return Some(Expr::Number(1.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && name == "cos"
            && args.len() == 1
            && let Expr::Div(num, den) = &args[0]
            && helpers::is_pi(num)
            && matches!(**den, Expr::Number(n) if n == 2.0)
        {
            return Some(Expr::Number(0.0));
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
        if let Expr::FunctionCall { name, args } = expr
            && args.len() == 1
        {
            let arg = &args[0];
            let arg_val = helpers::get_numeric_value(arg);
            let is_numeric_input = matches!(arg, Expr::Number(_));

            match name.as_str() {
                "sin" => {
                    if helpers::approx_eq(arg_val, PI / 6.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 6.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::Number(0.5))
                        } else {
                            Some(Expr::Div(
                                Rc::new(Expr::Number(1.0)),
                                Rc::new(Expr::Number(2.0)),
                            ))
                        };
                    }
                    if helpers::approx_eq(arg_val, PI / 4.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 4.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::Number((2.0f64).sqrt() / 2.0))
                        } else {
                            Some(Expr::Div(
                                Rc::new(Expr::FunctionCall {
                                    name: "sqrt".to_string(),
                                    args: vec![Expr::Number(2.0)],
                                }),
                                Rc::new(Expr::Number(2.0)),
                            ))
                        };
                    }
                }
                "cos" => {
                    if helpers::approx_eq(arg_val, PI / 3.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 3.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::Number(0.5))
                        } else {
                            Some(Expr::Div(
                                Rc::new(Expr::Number(1.0)),
                                Rc::new(Expr::Number(2.0)),
                            ))
                        };
                    }
                    if helpers::approx_eq(arg_val, PI / 4.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 4.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::Number((2.0f64).sqrt() / 2.0))
                        } else {
                            Some(Expr::Div(
                                Rc::new(Expr::FunctionCall {
                                    name: "sqrt".to_string(),
                                    args: vec![Expr::Number(2.0)],
                                }),
                                Rc::new(Expr::Number(2.0)),
                            ))
                        };
                    }
                }
                "tan" => {
                    if helpers::approx_eq(arg_val, PI / 4.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 4.0)))
                    {
                        return Some(Expr::Number(1.0));
                    }
                    if helpers::approx_eq(arg_val, PI / 3.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 3.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::Number((3.0f64).sqrt()))
                        } else {
                            Some(Expr::FunctionCall {
                                name: "sqrt".to_string(),
                                args: vec![Expr::Number(3.0)],
                            })
                        };
                    }
                    if helpers::approx_eq(arg_val, PI / 6.0)
                        || (matches!(arg, Expr::Div(num, den) if helpers::is_pi(num) && matches!(**den, Expr::Number(n) if n == 6.0)))
                    {
                        return if is_numeric_input {
                            Some(Expr::Number(1.0 / (3.0f64).sqrt()))
                        } else {
                            Some(Expr::Div(
                                Rc::new(Expr::FunctionCall {
                                    name: "sqrt".to_string(),
                                    args: vec![Expr::Number(3.0)],
                                }),
                                Rc::new(Expr::Number(3.0)),
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
