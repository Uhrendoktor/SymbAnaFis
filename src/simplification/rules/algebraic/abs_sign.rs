use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    AbsNumericRule,
    "abs_numeric",
    95,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "abs" || name == "Abs")
            && args.len() == 1
            && let Expr::Number(n) = &args[0]
        {
            return Some(Expr::Number(n.abs()));
        }
        None
    }
);

rule!(
    AbsAbsRule,
    "abs_abs",
    90,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "abs" || name == "Abs")
            && args.len() == 1
            && let Expr::FunctionCall {
                name: inner_name,
                args: inner_args,
            } = &args[0]
            && (inner_name == "abs" || inner_name == "Abs")
            && inner_args.len() == 1
        {
            return Some(args[0].clone());
        }
        None
    }
);

rule!(
    AbsNegRule,
    "abs_neg",
    90,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "abs" || name == "Abs")
            && args.len() == 1
        {
            // Check for -x (represented as Mul(-1, x))
            if let Expr::Mul(a, b) = &args[0]
                && let Expr::Number(n) = &**a
                && *n == -1.0
            {
                return Some(Expr::FunctionCall {
                    name: "abs".to_string(),
                    args: vec![(**b).clone()],
                });
            }
        }
        None
    }
);

rule!(
    AbsSquareRule,
    "abs_square",
    85,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "abs" || name == "Abs")
            && args.len() == 1
        {
            // Check for x^(even number)
            if let Expr::Pow(_, exp) = &args[0]
                && let Expr::Number(n) = &**exp
            {
                // Check if exponent is a positive even integer
                if *n > 0.0 && n.fract() == 0.0 && (*n as i64) % 2 == 0 {
                    return Some(args[0].clone());
                }
            }
        }
        None
    }
);

rule!(
    AbsPowEvenRule,
    "abs_pow_even",
    85,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        // abs(x)^n where n is positive even integer -> x^n
        if let Expr::Pow(base, exp) = expr
            && let Expr::FunctionCall { name, args } = &**base
            && (name == "abs" || name == "Abs")
            && args.len() == 1
            && let Expr::Number(n) = &**exp
        {
            // Check if exponent is a positive even integer
            if *n > 0.0 && n.fract() == 0.0 && (*n as i64) % 2 == 0 {
                return Some(Expr::Pow(Rc::new(args[0].clone()), exp.clone()));
            }
        }
        None
    }
);

rule!(
    SignNumericRule,
    "sign_numeric",
    95,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "sign" || name == "sgn")
            && args.len() == 1
            && let Expr::Number(n) = &args[0]
        {
            if *n > 0.0 {
                return Some(Expr::Number(1.0));
            } else if *n < 0.0 {
                return Some(Expr::Number(-1.0));
            } else {
                return Some(Expr::Number(0.0));
            }
        }
        None
    }
);

rule!(
    SignSignRule,
    "sign_sign",
    90,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "sign" || name == "sgn")
            && args.len() == 1
            && let Expr::FunctionCall {
                name: inner_name, ..
            } = &args[0]
            && (inner_name == "sign" || inner_name == "sgn")
        {
            return Some(args[0].clone());
        }
        None
    }
);

rule!(
    SignAbsRule,
    "sign_abs",
    85,
    Algebraic,
    &[ExprKind::Function],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::FunctionCall { name, args } = expr
            && (name == "sign" || name == "sgn")
            && args.len() == 1
            && let Expr::FunctionCall {
                name: inner_name, ..
            } = &args[0]
            && (inner_name == "abs" || inner_name == "Abs")
        {
            // sign(abs(x)) = 1 for x != 0
            return Some(Expr::Number(1.0));
        }
        None
    }
);

rule!(
    AbsSignMulRule,
    "abs_sign_mul",
    85,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Mul(a, b) = expr {
            // Check for abs(x) * sign(x) or sign(x) * abs(x)
            let check_pair = |left: &Expr, right: &Expr| -> Option<Expr> {
                if let (
                    Expr::FunctionCall {
                        name: name1,
                        args: args1,
                    },
                    Expr::FunctionCall {
                        name: name2,
                        args: args2,
                    },
                ) = (left, right)
                    && (name1 == "abs" || name1 == "Abs")
                    && (name2 == "sign" || name2 == "sgn")
                    && args1.len() == 1
                    && args2.len() == 1
                    && args1[0] == args2[0]
                {
                    return Some(args1[0].clone());
                }
                None
            };

            if let Some(result) = check_pair(a, b) {
                return Some(result);
            }
            if let Some(result) = check_pair(b, a) {
                return Some(result);
            }
        }
        None
    }
);
