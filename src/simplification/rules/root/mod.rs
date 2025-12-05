use crate::ast::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    SqrtPowerRule,
    "sqrt_power",
    85,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "sqrt"
            && args.len() == 1
            && let Expr::Pow(base, exp) = &args[0]
        {
            // Special case: sqrt(x^2) should always return abs(x)
            if let Expr::Number(n) = &**exp
                && *n == 2.0
            {
                // sqrt(x^2) = |x|
                return Some(Expr::FunctionCall {
                    name: "abs".to_string(),
                    args: vec![(**base).clone()],
                });
            }

            // Create new exponent: exp / 2
            let new_exp = Expr::Div(exp.clone(), Rc::new(Expr::Number(2.0)));

            // Simplify the division immediately
            let simplified_exp = match &new_exp {
                Expr::Div(u, v) => {
                    if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v) {
                        if *b != 0.0 {
                            let result = a / b;
                            if (result - result.round()).abs() < 1e-10 {
                                Expr::Number(result.round())
                            } else {
                                new_exp
                            }
                        } else {
                            new_exp
                        }
                    } else {
                        new_exp
                    }
                }
                _ => new_exp,
            };

            // If exponent simplified to 1, return base directly
            if matches!(simplified_exp, Expr::Number(n) if n == 1.0) {
                return Some((**base).clone());
            }

            let result = Expr::Pow(base.clone(), Rc::new(simplified_exp.clone()));

            return Some(result);
        }
        None
    }
);

rule!(
    CbrtPowerRule,
    "cbrt_power",
    85,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "cbrt"
            && args.len() == 1
            && let Expr::Pow(base, exp) = &args[0]
        {
            // Create new exponent: exp / 3
            let new_exp = Expr::Div(exp.clone(), Rc::new(Expr::Number(3.0)));

            // Simplify the division immediately
            let simplified_exp = match &new_exp {
                Expr::Div(u, v) => {
                    if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v) {
                        if *b != 0.0 {
                            let result = a / b;
                            if (result - result.round()).abs() < 1e-10 {
                                Expr::Number(result.round())
                            } else {
                                new_exp
                            }
                        } else {
                            new_exp
                        }
                    } else {
                        new_exp
                    }
                }
                _ => new_exp,
            };

            // If exponent simplified to 1, return base directly
            if matches!(simplified_exp, Expr::Number(n) if n == 1.0) {
                return Some((**base).clone());
            }

            return Some(Expr::Pow(base.clone(), Rc::new(simplified_exp)));
        }
        None
    }
);

rule!(SqrtMulRule, "sqrt_mul", 56, Root, &[ExprKind::Mul], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::Mul(u, v) = expr {
        // Check for sqrt(a) * sqrt(b)
        if let (
            Expr::FunctionCall {
                name: u_name,
                args: u_args,
            },
            Expr::FunctionCall {
                name: v_name,
                args: v_args,
            },
        ) = (&**u, &**v)
            && u_name == "sqrt"
            && v_name == "sqrt"
            && u_args.len() == 1
            && v_args.len() == 1
        {
            return Some(Expr::FunctionCall {
                name: "sqrt".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(u_args[0].clone()),
                    Rc::new(v_args[0].clone()),
                )],
            });
        }
    }
    None
});

rule!(SqrtDivRule, "sqrt_div", 56, Root, &[ExprKind::Div], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::Div(u, v) = expr {
        // Check for sqrt(a) / sqrt(b)
        if let (
            Expr::FunctionCall {
                name: u_name,
                args: u_args,
            },
            Expr::FunctionCall {
                name: v_name,
                args: v_args,
            },
        ) = (&**u, &**v)
            && u_name == "sqrt"
            && v_name == "sqrt"
            && u_args.len() == 1
            && v_args.len() == 1
        {
            return Some(Expr::FunctionCall {
                name: "sqrt".to_string(),
                args: vec![Expr::Div(
                    Rc::new(u_args[0].clone()),
                    Rc::new(v_args[0].clone()),
                )],
            });
        }
    }
    None
});

rule!(
    NormalizeRootsRule,
    "normalize_roots",
    50,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && args.len() == 1
        {
            match name.as_str() {
                "sqrt" => {
                    return Some(Expr::Pow(
                        Rc::new(args[0].clone()),
                        Rc::new(Expr::Div(
                            Rc::new(Expr::Number(1.0)),
                            Rc::new(Expr::Number(2.0)),
                        )),
                    ));
                }
                "cbrt" => {
                    return Some(Expr::Pow(
                        Rc::new(args[0].clone()),
                        Rc::new(Expr::Div(
                            Rc::new(Expr::Number(1.0)),
                            Rc::new(Expr::Number(3.0)),
                        )),
                    ));
                }
                _ => {}
            }
        }
        None
    }
);

rule!(
    SqrtExtractSquareRule,
    "sqrt_extract_square",
    84,
    Root,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        // sqrt(a * x^2) → |x| * sqrt(a)
        // sqrt(x^2 * a) → |x| * sqrt(a)
        if let Expr::FunctionCall { name, args } = expr
            && name == "sqrt"
            && args.len() == 1
            && let Expr::Mul(u, v) = &args[0]
        {
            // Check if either factor is a square (x^2)
            let (square_base, other) = if let Expr::Pow(base, exp) = &**u
                && let Expr::Number(n) = &**exp
                && *n == 2.0
            {
                (Some(base), v)
            } else if let Expr::Pow(base, exp) = &**v
                && let Expr::Number(n) = &**exp
                && *n == 2.0
            {
                (Some(base), u)
            } else {
                (None, u)
            };

            if let Some(base) = square_base {
                // sqrt(other * base^2) = |base| * sqrt(other)
                let abs_base = Expr::FunctionCall {
                    name: "abs".to_string(),
                    args: vec![(**base).clone()],
                };
                let sqrt_other = Expr::FunctionCall {
                    name: "sqrt".to_string(),
                    args: vec![(**other).clone()],
                };
                return Some(Expr::Mul(Rc::new(abs_base), Rc::new(sqrt_other)));
            }
        }
        None
    }
);

/// Get all root simplification rules in priority order
pub(crate) fn get_root_rules() -> Vec<Rc<dyn Rule>> {
    vec![
        Rc::new(SqrtPowerRule),
        Rc::new(SqrtExtractSquareRule),
        Rc::new(CbrtPowerRule),
        Rc::new(SqrtMulRule),
        Rc::new(SqrtDivRule),
        Rc::new(NormalizeRootsRule),
    ]
}
