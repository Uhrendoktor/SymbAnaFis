use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};

rule!(
    PowerZeroRule,
    "power_zero",
    80,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(_u, v) = &expr.kind
            && matches!(v.kind, AstKind::Number(n) if n == 0.0)
        {
            return Some(Expr::number(1.0));
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
        if let AstKind::Pow(u, v) = &expr.kind
            && matches!(v.kind, AstKind::Number(n) if n == 1.0)
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
        if let AstKind::Pow(u, v) = &expr.kind
            && let AstKind::Pow(base, exp_inner) = &u.kind
        {
            // Check for special case: (x^even)^(1/even) where result would be x^1
            // This should become abs(x), not x
            if let AstKind::Number(inner_n) = &exp_inner.kind {
                // Check if inner exponent is a positive even integer
                let inner_is_even =
                    *inner_n > 0.0 && inner_n.fract() == 0.0 && (*inner_n as i64) % 2 == 0;

                if inner_is_even {
                    // Check if outer exponent is 1/inner_n (so result would be x^1)
                    if let AstKind::Div(num, den) = &v.kind
                        && let (AstKind::Number(num_val), AstKind::Number(den_val)) =
                            (&num.kind, &den.kind)
                        && *num_val == 1.0
                        && (*den_val - *inner_n).abs() < 1e-10
                    {
                        // (x^even)^(1/even) = abs(x)
                        return Some(Expr::func_multi("abs".to_string(), vec![(**base).clone()]));
                    }
                    // Also check for cases like (x^4)^(1/2) = x^2 -> should remain as is
                    // since x^2 is always non-negative
                    // Check for numeric outer exponent that would result in x^1
                    if let AstKind::Number(outer_n) = &v.kind {
                        let product = inner_n * outer_n;
                        if (product - 1.0).abs() < 1e-10 {
                            // (x^even)^(something) = x^1 should be abs(x)
                            return Some(Expr::func_multi(
                                "abs".to_string(),
                                vec![(**base).clone()],
                            ));
                        }
                    }
                }
            }

            // Create new exponent: exp_inner * v
            // Let the ConstantFoldRule handle numeric simplification on next pass
            let new_exp = Expr::mul_expr((**exp_inner).clone(), (**v).clone());

            return Some(Expr::pow((**base).clone(), new_exp));
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
        if let AstKind::Mul(u, v) = &expr.kind {
            // Check if both terms are powers with the same base
            if let (AstKind::Pow(base_u, exp_u), AstKind::Pow(base_v, exp_v)) = (&u.kind, &v.kind)
                && base_u == base_v
            {
                return Some(Expr::pow(
                    (**base_u).clone(),
                    Expr::add_expr((**exp_u).clone(), (**exp_v).clone()),
                ));
            }
            // Check if one is a power and the other is the same base
            if let AstKind::Pow(base_u, exp_u) = &u.kind
                && base_u == v
            {
                return Some(Expr::pow(
                    (**base_u).clone(),
                    Expr::add_expr((**exp_u).clone(), Expr::number(1.0)),
                ));
            }
            if let AstKind::Pow(base_v, exp_v) = &v.kind
                && base_v == u
            {
                return Some(Expr::pow(
                    (**base_v).clone(),
                    Expr::add_expr(Expr::number(1.0), (**exp_v).clone()),
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
        if let AstKind::Div(u, v) = &expr.kind {
            // Check if both numerator and denominator are powers with the same base
            if let (AstKind::Pow(base_u, exp_u), AstKind::Pow(base_v, exp_v)) = (&u.kind, &v.kind)
                && base_u == base_v
            {
                return Some(Expr::pow(
                    (**base_u).clone(),
                    Expr::sub_expr((**exp_u).clone(), (**exp_v).clone()),
                ));
            }
            // Check if numerator is a power and denominator is the same base
            if let AstKind::Pow(base_u, exp_u) = &u.kind
                && base_u == v
            {
                return Some(Expr::pow(
                    (**base_u).clone(),
                    Expr::sub_expr((**exp_u).clone(), Expr::number(1.0)),
                ));
            }
            // Check if denominator is a power and numerator is the same base
            if let AstKind::Pow(base_v, exp_v) = &v.kind
                && base_v == u
            {
                return Some(Expr::pow(
                    (**base_v).clone(),
                    Expr::sub_expr(Expr::number(1.0), (**exp_v).clone()),
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
        if let AstKind::Mul(_, _) = &expr.kind {
            let factors = crate::simplification::helpers::flatten_mul(expr);

            // Group by base
            use std::collections::HashMap;
            let mut base_to_exponents: HashMap<Expr, Vec<Expr>> = HashMap::new();

            for factor in factors {
                if let AstKind::Pow(base, exp) = &factor.kind {
                    base_to_exponents
                        .entry((**base).clone())
                        .or_default()
                        .push((**exp).clone());
                } else {
                    // Non-power factor, treat as base^1
                    base_to_exponents
                        .entry(factor)
                        .or_default()
                        .push(Expr::number(1.0));
                }
            }

            // Combine exponents for each base
            let mut result_factors = Vec::new();
            for (base, exponents) in base_to_exponents {
                if exponents.len() == 1 {
                    if exponents[0] == Expr::number(1.0) {
                        result_factors.push(base);
                    } else {
                        result_factors.push(Expr::pow(base, exponents[0].clone()));
                    }
                } else {
                    // Sum all exponents
                    let mut sum = exponents[0].clone();
                    for exp in &exponents[1..] {
                        sum = Expr::add_expr(sum, exp.clone());
                    }
                    result_factors.push(Expr::pow(base, sum));
                }
            }

            // Rebuild the expression
            if result_factors.len() == 1 {
                Some(result_factors[0].clone())
            } else {
                let mut result = result_factors[0].clone();
                for factor in &result_factors[1..] {
                    result = Expr::mul_expr(result, factor.clone());
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
        if let AstKind::Div(num, den) = &expr.kind
            && let (AstKind::Pow(base_num, exp_num), AstKind::Pow(base_den, exp_den)) =
                (&num.kind, &den.kind)
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

            return Some(Expr::pow(
                Expr::div_expr((**base_num).clone(), (**base_den).clone()),
                (**exp_num).clone(),
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
        if let AstKind::Mul(left, right) = &expr.kind
            && let (AstKind::Pow(base_left, exp_left), AstKind::Pow(base_right, exp_right)) =
                (&left.kind, &right.kind)
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

            return Some(Expr::pow(
                Expr::mul_expr((**base_left).clone(), (**base_right).clone()),
                (**exp_left).clone(),
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
        if let AstKind::Pow(base, exp) = &expr.kind {
            // Handle negative number exponent: x^-n -> 1/x^n
            if let AstKind::Number(n) = exp.kind
                && n < 0.0
            {
                let positive_exp = Expr::number(-n);
                let denominator = Expr::pow((**base).clone(), positive_exp);
                return Some(Expr::div_expr(Expr::number(1.0), denominator));
            }
            // Handle negative fraction exponent: x^(-a/b) -> 1/x^(a/b)
            if let AstKind::Div(num, den) = &exp.kind
                && let AstKind::Number(n) = num.kind
                && n < 0.0
            {
                let positive_num = Expr::number(-n);
                let positive_exp = Expr::div_expr(positive_num, (**den).clone());
                let denominator = Expr::pow((**base).clone(), positive_exp);
                return Some(Expr::div_expr(Expr::number(1.0), denominator));
            }
            // Handle Mul(-1, exp): x^(-1 * a) -> 1/x^a
            if let AstKind::Mul(left, right) = &exp.kind
                && let AstKind::Number(n) = left.kind
                && n == -1.0
            {
                let denominator = Expr::pow((**base).clone(), (**right).clone());
                return Some(Expr::div_expr(Expr::number(1.0), denominator));
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
        if let AstKind::Pow(base, exp) = &expr.kind
            && let AstKind::Div(num, den) = &base.kind
        {
            // Only expand (a/b)^n -> a^n / b^n when:
            // 1. The exponent is a fractional root (like 1/2, 1/3) - this enables sqrt simplifications
            // 2. The denominator is a power that can be simplified with the exponent

            let is_root_exponent = match &exp.kind {
                AstKind::Div(n, d) => {
                    matches!((&n.kind, &d.kind), (AstKind::Number(num_val), AstKind::Number(den_val))
                    if *num_val == 1.0 && *den_val >= 2.0)
                }
                AstKind::Number(n) => *n > 0.0 && *n < 1.0, // e.g., 0.5
                _ => false,
            };

            // Check if denominator is a power that would simplify nicely
            let den_would_simplify = match &den.kind {
                AstKind::Pow(_, inner_exp) => {
                    // If den = x^m and exp = 1/n, then den^exp = x^(m/n)
                    // This simplifies if m/n is an integer
                    if let (AstKind::Number(m), AstKind::Div(one, n_rc)) =
                        (&inner_exp.kind, &exp.kind)
                    {
                        if let (AstKind::Number(one_val), AstKind::Number(n_val)) =
                            (&one.kind, &n_rc.kind)
                        {
                            *one_val == 1.0 && (m / n_val).fract().abs() < 1e-10
                        } else {
                            false
                        }
                    } else if let (AstKind::Number(m), AstKind::Number(exp_val)) =
                        (&inner_exp.kind, &exp.kind)
                    {
                        // den = x^m, exp = n (numeric), check if m*n is simpler
                        (m * exp_val).fract().abs() < 1e-10
                    } else {
                        false
                    }
                }
                // Also expand if denominator is a symbol and exponent is 1/2 (to get sqrt(c^2) = c)
                AstKind::Symbol(_) => is_root_exponent,
                AstKind::Number(_) => is_root_exponent, // sqrt(4) = 2
                _ => false,
            };

            if is_root_exponent || den_would_simplify {
                // (a/b)^n -> a^n / b^n
                let num_pow = Expr::pow((**num).clone(), (**exp).clone());
                let den_pow = Expr::pow((**den).clone(), (**exp).clone());
                return Some(Expr::div_expr(num_pow, den_pow));
            }
        }
        None
    }
);
