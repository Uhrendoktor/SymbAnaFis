use crate::ast::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    LnOneRule,
    "ln_one",
    95,
    Exponential,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "ln"
            && args.len() == 1
            && matches!(args[0], Expr::Number(n) if n == 1.0)
        {
            return Some(Expr::Number(0.0));
        }
        None
    }
);

rule!(
    LnERule,
    "ln_e",
    95,
    Exponential,
    &[ExprKind::Function],
    |expr: &Expr, context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "ln"
            && args.len() == 1
        {
            // Check for ln(exp(1))
            if matches!(&args[0], Expr::FunctionCall { name: exp_name, args: exp_args }
                       if exp_name == "exp" && exp_args.len() == 1 && matches!(exp_args[0], Expr::Number(n) if n == 1.0))
            {
                return Some(Expr::Number(1.0));
            }
            // Check for ln(e) where e is a symbol (and not a user-defined variable)
            if let Expr::Symbol(s) = &args[0]
                && s == "e"
                && !context.fixed_vars.contains("e")
            {
                return Some(Expr::Number(1.0));
            }
        }
        None
    }
);

rule!(
    ExpZeroRule,
    "exp_zero",
    95,
    Exponential,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "exp"
            && args.len() == 1
            && matches!(args[0], Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(1.0));
        }
        None
    }
);

rule!(ExpLnIdentityRule, "exp_ln_identity", 90, Exponential, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
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

rule!(LnExpIdentityRule, "ln_exp_identity", 90, Exponential, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::FunctionCall { name, args } = expr
        && name == "ln"
        && args.len() == 1
    {
        // Check for ln(exp(x))
        if let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0]
            && inner_name == "exp"
            && inner_args.len() == 1
        {
            return Some(inner_args[0].clone());
        }
        // Check for ln(e^x)
        if let Expr::Pow(base, exp) = &args[0]
            && let Expr::Symbol(b) = &**base
            && b == "e"
        {
            return Some(exp.as_ref().clone());
        }
    }
    None
});

rule!(
    LogPowerRule,
    "log_power",
    90,
    Exponential,
    &[ExprKind::Function],
    |expr: &Expr, context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && args.len() == 1
            && (name == "ln" || name == "log10" || name == "log2")
        {
            let content = &args[0];
            // log(x^n) = n * log(x)
            if let Expr::Pow(base, exp) = content {
                // Check if exponent is an even integer
                if let Expr::Number(n) = &**exp {
                    let is_even_int = n.fract() == 0.0 && (*n as i64) % 2 == 0;
                    let is_odd_int = n.fract() == 0.0 && (*n as i64) % 2 != 0;

                    if is_even_int && *n != 0.0 {
                        // ln(x^2) = 2*ln(|x|) - always correct for x ≠ 0
                        return Some(Expr::Mul(
                            exp.clone(),
                            Rc::new(Expr::FunctionCall {
                                name: name.clone(),
                                args: vec![Expr::FunctionCall {
                                    name: "abs".to_string(),
                                    args: vec![(**base).clone()],
                                }],
                            }),
                        ));
                    } else if is_odd_int {
                        // ln(x^3) = 3*ln(x) - only valid for x > 0
                        // In domain-safe mode, skip unless base is known positive
                        if context.domain_safe {
                            // Check if base is known to be positive (exp, cosh, etc.)
                            let is_positive = matches!(&**base,
                                Expr::FunctionCall { name: fn_name, .. }
                                if fn_name == "exp" || fn_name == "cosh"
                            );
                            if !is_positive {
                                return None;
                            }
                        }
                        return Some(Expr::Mul(
                            exp.clone(),
                            Rc::new(Expr::FunctionCall {
                                name: name.clone(),
                                args: vec![(**base).clone()],
                            }),
                        ));
                    }
                }

                // For non-integer or symbolic exponents, only apply in aggressive mode
                // Don't use abs(x) - for odd exponents it would expand the domain
                // For symbolic n, we can't know if it's even/odd, so we assume x > 0
                if context.domain_safe {
                    return None;
                }

                return Some(Expr::Mul(
                    exp.clone(),
                    Rc::new(Expr::FunctionCall {
                        name: name.clone(),
                        args: vec![(**base).clone()],
                    }),
                ));
            }
            // log(sqrt(x)) = 0.5 * log(x) - only for x > 0
            if let Expr::FunctionCall {
                name: inner_name,
                args: inner_args,
            } = content
            {
                if inner_name == "sqrt" && inner_args.len() == 1 {
                    // In domain-safe mode, skip unless we know x > 0
                    if context.domain_safe {
                        return None;
                    }
                    return Some(Expr::Mul(
                        Rc::new(Expr::Number(0.5)),
                        Rc::new(Expr::FunctionCall {
                            name: name.clone(),
                            args: vec![inner_args[0].clone()],
                        }),
                    ));
                }
                // log(cbrt(x)) = (1/3) * log(x)
                if inner_name == "cbrt" && inner_args.len() == 1 {
                    // cbrt is defined for all reals, but log needs positive argument
                    if context.domain_safe {
                        return None;
                    }
                    return Some(Expr::Mul(
                        Rc::new(Expr::Div(
                            Rc::new(Expr::Number(1.0)),
                            Rc::new(Expr::Number(3.0)),
                        )),
                        Rc::new(Expr::FunctionCall {
                            name: name.clone(),
                            args: vec![inner_args[0].clone()],
                        }),
                    ));
                }
            }
        }
        None
    }
);

rule!(
    LogBaseRules,
    "log_base_values",
    95,
    Exponential,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && args.len() == 1
        {
            if name == "log10" {
                if matches!(args[0], Expr::Number(n) if n == 1.0) {
                    return Some(Expr::Number(0.0));
                }
                if matches!(args[0], Expr::Number(n) if n == 10.0) {
                    return Some(Expr::Number(1.0));
                }
            } else if name == "log2" {
                if matches!(args[0], Expr::Number(n) if n == 1.0) {
                    return Some(Expr::Number(0.0));
                }
                if matches!(args[0], Expr::Number(n) if n == 2.0) {
                    return Some(Expr::Number(1.0));
                }
            }
        }
        None
    }
);

rule!(
    ExpToEPowRule,
    "exp_to_e_pow",
    95,
    Exponential,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && name == "exp"
            && args.len() == 1
        {
            return Some(Expr::Pow(
                Rc::new(Expr::Symbol("e".to_string())),
                Rc::new(args[0].clone()),
            ));
        }
        None
    }
);

/// Helper function to extract ln argument
fn get_ln_arg(expr: &Expr) -> Option<Expr> {
    if let Expr::FunctionCall { name, args } = expr
        && name == "ln"
        && args.len() == 1
    {
        return Some(args[0].clone());
    }
    None
}

rule_with_helpers!(LogCombinationRule, "log_combination", 85, Exponential, &[ExprKind::Add, ExprKind::Sub],
    helpers: {
        // Helper defined above
    },
    |expr: &Expr, _context: &RuleContext| {
        match expr {
            Expr::Add(u, v) => {
                // ln(a) + ln(b) = ln(a * b)
                if let (Some(arg1), Some(arg2)) = (get_ln_arg(u), get_ln_arg(v)) {
                    return Some(Expr::FunctionCall {
                        name: "ln".to_string(),
                        args: vec![Expr::Mul(Rc::new(arg1), Rc::new(arg2))],
                    });
                }
            }
            Expr::Sub(u, v) => {
                // ln(a) - ln(b) = ln(a / b)
                if let (Some(arg1), Some(arg2)) = (get_ln_arg(u), get_ln_arg(v)) {
                    return Some(Expr::FunctionCall {
                        name: "ln".to_string(),
                        args: vec![Expr::Div(Rc::new(arg1), Rc::new(arg2))],
                    });
                }
            }
            _ => {}
        }
        None
    }
);

/// Get all exponential/logarithmic rules in priority order
pub(crate) fn get_exponential_rules() -> Vec<Rc<dyn Rule>> {
    vec![
        Rc::new(LnOneRule),
        Rc::new(LnERule),
        Rc::new(ExpZeroRule),
        Rc::new(ExpToEPowRule),
        Rc::new(ExpLnIdentityRule),
        Rc::new(LnExpIdentityRule),
        Rc::new(LogPowerRule),
        Rc::new(LogBaseRules),
        Rc::new(LogCombinationRule),
    ]
}
