use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    PowerZeroRule,
    "power_zero",
    80,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Pow(_u, _v) = expr
            && matches!(*_v.as_ref(), Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(1.0));
        }
        None
    }
);

rule!(
    PowerOneRule,
    "power_one",
    80,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Pow(u, v) = expr
            && matches!(*v.as_ref(), Expr::Number(n) if n == 1.0)
        {
            return Some((**u).clone());
        }
        None
    }
);

rule!(
    PowerPowerRule,
    "power_power",
    75,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Pow(u, v) = expr
            && let Expr::Pow(base, exp_inner) = &**u
        {
            // Check for special case: (x^even)^(1/even) where result would be x^1
            // This should become abs(x), not x
            if let Expr::Number(inner_n) = &**exp_inner {
                // Check if inner exponent is a positive even integer
                let inner_is_even =
                    *inner_n > 0.0 && inner_n.fract() == 0.0 && (*inner_n as i64) % 2 == 0;

                if inner_is_even {
                    // Check if outer exponent is 1/inner_n (so result would be x^1)
                    if let Expr::Div(num, den) = &**v
                        && let (Expr::Number(num_val), Expr::Number(den_val)) = (&**num, &**den)
                        && *num_val == 1.0
                        && (*den_val - *inner_n).abs() < 1e-10
                    {
                        // (x^even)^(1/even) = abs(x)
                        return Some(Expr::FunctionCall {
                            name: "abs".to_string(),
                            args: vec![(**base).clone()],
                        });
                    }
                    // Also check for cases like (x^4)^(1/2) = x^2 -> should remain as is
                    // since x^2 is always non-negative
                    // Check for numeric outer exponent that would result in x^1
                    if let Expr::Number(outer_n) = &**v {
                        let product = inner_n * outer_n;
                        if (product - 1.0).abs() < 1e-10 {
                            // (x^even)^(something) = x^1 should be abs(x)
                            return Some(Expr::FunctionCall {
                                name: "abs".to_string(),
                                args: vec![(**base).clone()],
                            });
                        }
                    }
                }
            }

            // Create new exponent: exp_inner * v
            // Let the ConstantFoldRule handle numeric simplification on next pass
            let new_exp = Expr::Mul(exp_inner.clone(), v.clone());

            return Some(Expr::Pow(base.clone(), Rc::new(new_exp)));
        }
        None
    }
);

rule!(
    PowerMulRule,
    "power_mul",
    75,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Mul(u, v) = expr {
            // Check if both terms are powers with the same base
            if let (Expr::Pow(base_u, exp_u), Expr::Pow(base_v, exp_v)) = (&**u, &**v)
                && base_u == base_v
            {
                return Some(Expr::Pow(
                    base_u.clone(),
                    Rc::new(Expr::Add(exp_u.clone(), exp_v.clone())),
                ));
            }
            // Check if one is a power and the other is the same base
            if let Expr::Pow(base_u, exp_u) = &**u
                && base_u == v
            {
                return Some(Expr::Pow(
                    base_u.clone(),
                    Rc::new(Expr::Add(exp_u.clone(), Rc::new(Expr::Number(1.0)))),
                ));
            }
            if let Expr::Pow(base_v, exp_v) = &**v
                && base_v == u
            {
                return Some(Expr::Pow(
                    base_v.clone(),
                    Rc::new(Expr::Add(Rc::new(Expr::Number(1.0)), exp_v.clone())),
                ));
            }
        }
        None
    }
);

rule!(
    PowerDivRule,
    "power_div",
    75,
    Algebraic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Div(u, v) = expr {
            // Check if both numerator and denominator are powers with the same base
            if let (Expr::Pow(base_u, exp_u), Expr::Pow(base_v, exp_v)) = (&**u, &**v)
                && base_u == base_v
            {
                return Some(Expr::Pow(
                    base_u.clone(),
                    Rc::new(Expr::Sub(exp_u.clone(), exp_v.clone())),
                ));
            }
            // Check if numerator is a power and denominator is the same base
            if let Expr::Pow(base_u, exp_u) = &**u
                && base_u == v
            {
                return Some(Expr::Pow(
                    base_u.clone(),
                    Rc::new(Expr::Sub(exp_u.clone(), Rc::new(Expr::Number(1.0)))),
                ));
            }
            // Check if denominator is a power and numerator is the same base
            if let Expr::Pow(base_v, exp_v) = &**v
                && base_v == u
            {
                return Some(Expr::Pow(
                    base_v.clone(),
                    Rc::new(Expr::Sub(Rc::new(Expr::Number(1.0)), exp_v.clone())),
                ));
            }
        }
        None
    }
);

rule!(
    PowerCollectionRule,
    "power_collection",
    60,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Mul(_, _) = expr {
            let factors = crate::simplification::helpers::flatten_mul(expr);

            // Group by base
            use std::collections::HashMap;
            let mut base_to_exponents: HashMap<Expr, Vec<Expr>> = HashMap::new();

            for factor in factors {
                if let Expr::Pow(base, exp) = factor {
                    base_to_exponents
                        .entry(base.as_ref().clone())
                        .or_default()
                        .push(exp.as_ref().clone());
                } else {
                    // Non-power factor, treat as base^1
                    base_to_exponents
                        .entry(factor)
                        .or_default()
                        .push(Expr::Number(1.0));
                }
            }

            // Combine exponents for each base
            let mut result_factors = Vec::new();
            for (base, exponents) in base_to_exponents {
                if exponents.len() == 1 {
                    if exponents[0] == Expr::Number(1.0) {
                        result_factors.push(base);
                    } else {
                        result_factors
                            .push(Expr::Pow(Rc::new(base), Rc::new(exponents[0].clone())));
                    }
                } else {
                    // Sum all exponents
                    let mut sum = exponents[0].clone();
                    for exp in &exponents[1..] {
                        sum = Expr::Add(Rc::new(sum), Rc::new(exp.clone()));
                    }
                    result_factors.push(Expr::Pow(Rc::new(base), Rc::new(sum)));
                }
            }

            // Rebuild the expression
            if result_factors.len() == 1 {
                Some(result_factors[0].clone())
            } else {
                let mut result = result_factors[0].clone();
                for factor in &result_factors[1..] {
                    result = Expr::Mul(Rc::new(result), Rc::new(factor.clone()));
                }
                Some(result)
            }
        } else {
            None
        }
    }
);

rule!(
    CommonExponentDivRule,
    "common_exponent_div",
    55,
    Algebraic,
    &[ExprKind::Div],
    |expr: &Expr, context: &RuleContext| {
        if let Expr::Div(num, den) = expr
            && let (Expr::Pow(base_num, exp_num), Expr::Pow(base_den, exp_den)) = (&**num, &**den)
            && exp_num == exp_den
        {
            // Check if this is a fractional root exponent (like 1/2)
            // If so, in domain-safe mode, we need both bases to be non-negative
            if context.domain_safe
                && crate::simplification::helpers::is_fractional_root_exponent(exp_num)
            {
                let num_non_neg = crate::simplification::helpers::is_known_non_negative(base_num);
                let den_non_neg = crate::simplification::helpers::is_known_non_negative(base_den);
                if !(num_non_neg && den_non_neg) {
                    return None;
                }
            }

            return Some(Expr::Pow(
                Rc::new(Expr::Div(base_num.clone(), base_den.clone())),
                exp_num.clone(),
            ));
        }
        None
    }
);

rule!(
    CommonExponentMulRule,
    "common_exponent_mul",
    55,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, context: &RuleContext| {
        if let Expr::Mul(left, right) = expr
            && let (Expr::Pow(base_left, exp_left), Expr::Pow(base_right, exp_right)) =
                (&**left, &**right)
            && exp_left == exp_right
        {
            // Check if this is a fractional root exponent (like 1/2)
            // If so, in domain-safe mode, we need both bases to be non-negative
            if context.domain_safe
                && crate::simplification::helpers::is_fractional_root_exponent(exp_left)
            {
                let left_non_neg = crate::simplification::helpers::is_known_non_negative(base_left);
                let right_non_neg =
                    crate::simplification::helpers::is_known_non_negative(base_right);
                if !(left_non_neg && right_non_neg) {
                    return None;
                }
            }

            return Some(Expr::Pow(
                Rc::new(Expr::Mul(base_left.clone(), base_right.clone())),
                exp_left.clone(),
            ));
        }
        None
    }
);

rule!(
    NegativeExponentToFractionRule,
    "negative_exponent_to_fraction",
    90,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Pow(base, exp) = expr {
            // Handle negative number exponent: x^-n -> 1/x^n
            if let Expr::Number(n) = **exp
                && n < 0.0
            {
                let positive_exp = Expr::Number(-n);
                let denominator = Expr::Pow(base.clone(), Rc::new(positive_exp));
                return Some(Expr::Div(Rc::new(Expr::Number(1.0)), Rc::new(denominator)));
            }
            // Handle negative fraction exponent: x^(-a/b) -> 1/x^(a/b)
            if let Expr::Div(num, den) = &**exp
                && let Expr::Number(n) = &**num
                && *n < 0.0
            {
                let positive_num = Expr::Number(-n);
                let positive_exp = Expr::Div(Rc::new(positive_num), den.clone());
                let denominator = Expr::Pow(base.clone(), Rc::new(positive_exp));
                return Some(Expr::Div(Rc::new(Expr::Number(1.0)), Rc::new(denominator)));
            }
            // Handle Mul(-1, exp): x^(-1 * a) -> 1/x^a
            if let Expr::Mul(left, right) = &**exp
                && let Expr::Number(n) = &**left
                && *n == -1.0
            {
                let denominator = Expr::Pow(base.clone(), right.clone());
                return Some(Expr::Div(Rc::new(Expr::Number(1.0)), Rc::new(denominator)));
            }
        }
        None
    }
);

rule!(
    PowerOfQuotientRule,
    "power_of_quotient",
    88,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Pow(base, exp) = expr
            && let Expr::Div(num, den) = &**base
        {
            // Only expand (a/b)^n -> a^n / b^n when:
            // 1. The exponent is a fractional root (like 1/2, 1/3) - this enables sqrt simplifications
            // 2. The denominator is a power that can be simplified with the exponent

            let is_root_exponent = match &**exp {
                Expr::Div(n, d) => {
                    matches!((&**n, &**d), (Expr::Number(num_val), Expr::Number(den_val))
                    if *num_val == 1.0 && *den_val >= 2.0)
                }
                Expr::Number(n) => *n > 0.0 && *n < 1.0, // e.g., 0.5
                _ => false,
            };

            // Check if denominator is a power that would simplify nicely
            let den_would_simplify = match &**den {
                Expr::Pow(_, inner_exp) => {
                    // If den = x^m and exp = 1/n, then den^exp = x^(m/n)
                    // This simplifies if m/n is an integer
                    if let (Expr::Number(m), Expr::Div(one, n_rc)) = (&**inner_exp, &**exp) {
                        if let (Expr::Number(one_val), Expr::Number(n_val)) = (&**one, &**n_rc) {
                            *one_val == 1.0 && (m / n_val).fract().abs() < 1e-10
                        } else {
                            false
                        }
                    } else if let (Expr::Number(m), Expr::Number(exp_val)) = (&**inner_exp, &**exp)
                    {
                        // den = x^m, exp = n (numeric), check if m*n is simpler
                        (m * exp_val).fract().abs() < 1e-10
                    } else {
                        false
                    }
                }
                // Also expand if denominator is a symbol and exponent is 1/2 (to get sqrt(c^2) = c)
                Expr::Symbol(_) => is_root_exponent,
                Expr::Number(_) => is_root_exponent, // sqrt(4) = 2
                _ => false,
            };

            if is_root_exponent || den_would_simplify {
                // (a/b)^n -> a^n / b^n
                let num_pow = Expr::Pow(num.clone(), exp.clone());
                let den_pow = Expr::Pow(den.clone(), exp.clone());
                return Some(Expr::Div(Rc::new(num_pow), Rc::new(den_pow)));
            }
        }
        None
    }
);
