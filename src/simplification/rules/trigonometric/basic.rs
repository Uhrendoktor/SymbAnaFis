use crate::core::expr::{Expr, ExprKind as AstKind};
use crate::core::known_symbols::{COS, COT, CSC, SEC, SIN, SQRT, TAN, get_symbol};
use crate::simplification::helpers;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::f64::consts::PI;
use std::sync::Arc;

rule_arc!(
    SinZeroRule,
    "sin_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["sin"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *SIN
            && args.len() == 1
            && {
                // Exact check for sin(0.0)
                #[allow(clippy::float_cmp)] // Comparing against exact constant 0.0
                let is_zero = matches!(&args[0].kind, AstKind::Number(n) if *n == 0.0);
                is_zero
            }
        {
            return Some(Arc::new(Expr::number(0.0)));
        }
        None
    }
);

rule_arc!(
    CosZeroRule,
    "cos_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["cos"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *COS
            && args.len() == 1
            && matches!(&args[0].kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Arc::new(Expr::number(1.0)));
        }
        None
    }
);

rule_arc!(
    TanZeroRule,
    "tan_zero",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["tan"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *TAN
            && args.len() == 1
            && matches!(&args[0].kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Arc::new(Expr::number(0.0)));
        }
        None
    }
);

rule_arc!(
    SinPiRule,
    "sin_pi",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["sin"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *SIN
            && args.len() == 1
            && helpers::is_pi(&args[0])
        {
            return Some(Arc::new(Expr::number(0.0)));
        }
        None
    }
);

rule_arc!(
    CosPiRule,
    "cos_pi",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["cos"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *COS
            && args.len() == 1
            && helpers::is_pi(&args[0])
        {
            return Some(Arc::new(Expr::number(-1.0)));
        }
        None
    }
);

rule_arc!(
    SinPiOverTwoRule,
    "sin_pi_over_two",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["sin"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *SIN
            && args.len() == 1
            && let AstKind::Div(num, den) = &args[0].kind
            && helpers::is_pi(num)
        {
            // Exact check for pi/2 denominator
            #[allow(clippy::float_cmp)] // Comparing against exact constant 2.0
            let is_two = matches!(&den.kind, AstKind::Number(n) if *n == 2.0);
            if is_two {
                return Some(Arc::new(Expr::number(1.0)));
            }
        }
        None
    }
);

rule_arc!(
    CosPiOverTwoRule,
    "cos_pi_over_two",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["cos"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && name.id() == *COS
            && args.len() == 1
            && let AstKind::Div(num, den) = &args[0].kind
            && helpers::is_pi(num)
        {
            // Exact check for pi/2 denominator
            #[allow(clippy::float_cmp)] // Comparing against exact constant 2.0
            let is_two = matches!(&den.kind, AstKind::Number(n) if *n == 2.0);
            if is_two {
                return Some(Arc::new(Expr::number(0.0)));
            }
        }
        None
    }
);

rule_arc!(
    TrigExactValuesRule,
    "trig_exact_values",
    95,
    Trigonometric,
    &[ExprKind::Function],
    targets: &["sin", "cos", "tan"],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::FunctionCall { name, args } = &expr.kind
            && args.len() == 1
        {
            let arg = &args[0];
            let arg_val = helpers::get_numeric_value(arg).unwrap_or(f64::NAN);
            let is_numeric_input = matches!(arg.kind, AstKind::Number(_));

            match name {
                n if n.id() == *SIN => {
                    let matches_pi_six = {
                        // Exact check for denominator 6.0 (PI/6)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 6.0
                        let is_six = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 6.0));
                        is_six
                    };
                    if helpers::approx_eq(arg_val, PI / 6.0) || matches_pi_six
                    {
                        return if is_numeric_input {
                            Some(Arc::new(Expr::number(0.5)))
                        } else {
                            Some(Arc::new(Expr::div_from_arcs(Arc::new(Expr::number(1.0)), Arc::new(Expr::number(2.0)))))
                        };
                    }
                    let matches_pi_four = {
                        // Exact check for denominator 4.0 (PI/4)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 4.0
                        let is_four = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 4.0));
                        is_four
                    };
                    if helpers::approx_eq(arg_val, PI / 4.0) || matches_pi_four
                    {
                        return if is_numeric_input {
                            Some(Arc::new(Expr::number((2.0_f64).sqrt() / 2.0)))
                        } else {
                            Some(Arc::new(Expr::div_from_arcs(
                                Arc::new(Expr::func_symbol(get_symbol(&SQRT), Expr::number(2.0))),
                                Arc::new(Expr::number(2.0)),
                            )))
                        };
                    }
                }
                n if n.id() == *COS => {
                    let matches_pi_three = {
                        // Exact check for denominator 3.0 (PI/3)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 3.0
                        let is_three = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 3.0));
                        is_three
                    };
                    if helpers::approx_eq(arg_val, PI / 3.0) || matches_pi_three
                    {
                        return if is_numeric_input {
                            Some(Arc::new(Expr::number(0.5)))
                        } else {
                            Some(Arc::new(Expr::div_from_arcs(Arc::new(Expr::number(1.0)), Arc::new(Expr::number(2.0)))))
                        };
                    }
                    let matches_pi_four = {
                        // Exact check for denominator 4.0 (PI/4)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 4.0
                        let is_four = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 4.0));
                        is_four
                    };
                    if helpers::approx_eq(arg_val, PI / 4.0) || matches_pi_four
                    {
                        return if is_numeric_input {
                            Some(Arc::new(Expr::number((2.0_f64).sqrt() / 2.0)))
                        } else {
                            Some(Arc::new(Expr::div_from_arcs(
                                Arc::new(Expr::func_symbol(get_symbol(&SQRT), Expr::number(2.0))),
                                Arc::new(Expr::number(2.0)),
                            )))
                        };
                    }
                }
                n if n.id() == *TAN => {
                    let matches_pi_four = {
                        // Exact check for denominator 4.0 (PI/4)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 4.0
                        let is_four = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 4.0));
                        is_four
                    };
                    if helpers::approx_eq(arg_val, PI / 4.0) || matches_pi_four
                    {
                        return Some(Arc::new(Expr::number(1.0)));
                    }
                    let matches_pi_three = {
                        // Exact check for denominator 3.0 (PI/3)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 3.0
                        let is_three = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 3.0));
                        is_three
                    };
                    if helpers::approx_eq(arg_val, PI / 3.0) || matches_pi_three
                    {
                        return if is_numeric_input {
                            Some(Arc::new(Expr::number((3.0_f64).sqrt())))
                        } else {
                            Some(Arc::new(Expr::func_symbol(get_symbol(&SQRT), Expr::number(3.0))))
                        };
                    }
                    let matches_pi_six = {
                        // Exact check for denominator 6.0 (PI/6)
                        #[allow(clippy::float_cmp)] // Comparing against exact constant 6.0
                        let is_six = matches!(&arg.kind, AstKind::Div(num, den) if helpers::is_pi(num) && matches!(&den.kind, AstKind::Number(n) if *n == 6.0));
                        is_six
                    };
                    if helpers::approx_eq(arg_val, PI / 6.0) || matches_pi_six
                    {
                        return if is_numeric_input {
                            Some(Arc::new(Expr::number(1.0 / (3.0_f64).sqrt())))
                        } else {
                            Some(Arc::new(Expr::div_from_arcs(
                                Arc::new(Expr::func_symbol(get_symbol(&SQRT), Expr::number(3.0))),
                                Arc::new(Expr::number(3.0)),
                            )))
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
rule_arc!(
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
                && name.id() == *COS
                && args.len() == 1
            {
                // Efficiently constructing using Arc<Expr>
                return Some(Arc::new(Expr::func_multi_from_arcs_symbol(
                    get_symbol(&SEC),
                    vec![Arc::clone(&args[0])],
                )));
            }
            // 1/cos(x)^n → sec(x)^n
            if let AstKind::Pow(base, exp) = &den.kind
                && let AstKind::FunctionCall { name, args } = &base.kind
                && name.id() == *COS
                && args.len() == 1
            {
                let sec_expr =
                    Expr::func_multi_from_arcs_symbol(get_symbol(&SEC), vec![Arc::clone(&args[0])]);
                return Some(Arc::new(Expr::pow_from_arcs(
                    Arc::new(sec_expr),
                    Arc::clone(exp),
                )));
            }
        }
        None
    }
);

rule_arc!(
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
                && name.id() == *SIN
                && args.len() == 1
            {
                return Some(Arc::new(Expr::func_multi_from_arcs_symbol(
                    get_symbol(&CSC),
                    vec![Arc::clone(&args[0])],
                )));
            }
            // 1/sin(x)^n → csc(x)^n
            if let AstKind::Pow(base, exp) = &den.kind
                && let AstKind::FunctionCall { name, args } = &base.kind
                && name.id() == *SIN
                && args.len() == 1
            {
                let csc_expr =
                    Expr::func_multi_from_arcs_symbol(get_symbol(&CSC), vec![Arc::clone(&args[0])]);
                return Some(Arc::new(Expr::pow_from_arcs(
                    Arc::new(csc_expr),
                    Arc::clone(exp),
                )));
            }
        }
        None
    }
);

rule_arc!(
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
            && num_name.id() == *SIN
            && den_name.id() == *COS
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Arc::new(Expr::func_multi_from_arcs_symbol(
                get_symbol(&TAN),
                vec![Arc::clone(&num_args[0])],
            )));
        }
        None
    }
);

rule_arc!(
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
            && num_name.id() == *COS
            && den_name.id() == *SIN
            && num_args.len() == 1
            && den_args.len() == 1
            && num_args[0] == den_args[0]
        {
            return Some(Arc::new(Expr::func_multi_from_arcs_symbol(
                get_symbol(&COT),
                vec![Arc::clone(&num_args[0])],
            )));
        }
        None
    }
);
