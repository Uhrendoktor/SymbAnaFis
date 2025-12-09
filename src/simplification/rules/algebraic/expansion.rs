use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};

rule!(
    ExpandPowerForCancellationRule,
    "expand_power_for_cancellation",
    92,
    Algebraic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind {
            // Helper to check if a factor is present in an expression
            let contains_factor = |expr: &Expr, factor: &Expr| -> bool {
                match &expr.kind {
                    AstKind::Mul(_, _) => {
                        let factors = crate::simplification::helpers::flatten_mul(expr);
                        factors.contains(factor)
                    }
                    _ => expr == factor,
                }
            };

            // Helper to check if expansion is useful
            let check_and_expand = |target: &Expr, other: &Expr| -> Option<Expr> {
                if let AstKind::Pow(base, exp) = &target.kind
                    && let AstKind::Mul(_, _) = &base.kind
                {
                    let base_factors = crate::simplification::helpers::flatten_mul(base);
                    // Check if any base factor is present in 'other'
                    let mut useful = false;
                    for factor in &base_factors {
                        if contains_factor(other, factor) {
                            useful = true;
                            break;
                        }
                    }

                    if useful {
                        let mut pow_factors: Vec<Expr> = Vec::new();
                        for factor in base_factors.into_iter() {
                            pow_factors.push(Expr::pow(factor, (**exp).clone()));
                        }
                        return Some(crate::simplification::helpers::rebuild_mul(pow_factors));
                    }
                }
                None
            };

            // Try expanding powers in numerator
            if let Some(expanded) = check_and_expand(num, den) {
                return Some(Expr::div_expr(expanded, (**den).clone()));
            }

            // Try expanding powers in denominator
            if let Some(expanded) = check_and_expand(den, num) {
                return Some(Expr::div_expr((**num).clone(), expanded));
            }
        }
        None
    }
);

rule!(
    PowerExpansionRule,
    "power_expansion",
    86,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(base, exp) = &expr.kind {
            // Expand (a*b)^n -> a^n * b^n ONLY if expansion enables simplification
            // This avoids oscillation with common_exponent_mul while still allowing
            // cases like (2*sqrt(x))^2 -> 4*x
            if let AstKind::Mul(_a, _b) = &base.kind
                && let AstKind::Number(n) = &exp.kind
                && *n > 1.0
                && n.fract() == 0.0
                && (*n as i64) < 10
            {
                let base_factors = crate::simplification::helpers::flatten_mul(base);

                // Check if expansion would enable simplification:
                // - Contains a power that would simplify (e.g., sqrt(x)^2 -> x)
                // - Contains a number (coefficient that would be raised to power)
                let has_simplifiable = base_factors.iter().any(|f| {
                    match &f.kind {
                        // sqrt(x)^2 -> x, x^(1/2)^2 -> x, etc.
                        AstKind::Pow(_, inner_exp) => {
                            if let AstKind::Number(inner_n) = &inner_exp.kind {
                                // Fractional exponent that would become integer
                                (inner_n * n).fract().abs() < 1e-10
                            } else if let AstKind::Div(num, den) = &inner_exp.kind {
                                // x^(a/b) raised to n - check if simplifies
                                if let (AstKind::Number(a), AstKind::Number(b)) =
                                    (&num.kind, &den.kind)
                                {
                                    ((a * n) / b).fract().abs() < 1e-10
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        // FunctionCall like sqrt, cbrt
                        AstKind::FunctionCall { name, .. } => {
                            matches!(name.as_str(), "sqrt" | "cbrt") && *n >= 2.0
                        }
                        // Numeric coefficient
                        AstKind::Number(_) => true,
                        _ => false,
                    }
                });

                if has_simplifiable {
                    let mut factors = Vec::new();
                    for factor in base_factors {
                        factors.push(Expr::pow(factor, (**exp).clone()));
                    }
                    return Some(crate::simplification::helpers::rebuild_mul(factors));
                }
            }

            // Expand (a/b)^n -> a^n / b^n ONLY if expansion enables simplification
            // This avoids oscillation with common_exponent_div
            if let AstKind::Div(a, b) = &base.kind
                && let AstKind::Number(n) = &exp.kind
                && *n > 1.0
                && n.fract() == 0.0
                && (*n as i64) < 10
            {
                // Helper to check if a term would simplify when raised to power n
                let would_simplify = |term: &Expr| -> bool {
                    match &term.kind {
                        AstKind::Pow(_, inner_exp) => {
                            if let AstKind::Number(inner_n) = &inner_exp.kind {
                                (inner_n * n).fract().abs() < 1e-10
                            } else if let AstKind::Div(num, den) = &inner_exp.kind {
                                if let (AstKind::Number(a_val), AstKind::Number(b_val)) =
                                    (&num.kind, &den.kind)
                                {
                                    ((a_val * n) / b_val).fract().abs() < 1e-10
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        AstKind::FunctionCall { name, .. } => {
                            matches!(name.as_str(), "sqrt" | "cbrt") && *n >= 2.0
                        }
                        AstKind::Number(_) => true,
                        AstKind::Mul(_, _) => {
                            // Check factors of multiplication
                            let factors = crate::simplification::helpers::flatten_mul(term);
                            factors.iter().any(|f| match &f.kind {
                                AstKind::Number(_) => true,
                                AstKind::FunctionCall { name, .. } => {
                                    matches!(name.as_str(), "sqrt" | "cbrt")
                                }
                                AstKind::Pow(_, inner_exp) => {
                                    if let AstKind::Number(inner_n) = &inner_exp.kind {
                                        (inner_n * n).fract().abs() < 1e-10
                                    } else {
                                        false
                                    }
                                }
                                _ => false,
                            })
                        }
                        _ => false,
                    }
                };

                // Only expand if numerator or denominator would simplify
                if would_simplify(a) || would_simplify(b) {
                    let a_pow = Expr::pow((**a).clone(), (**exp).clone());
                    let b_pow = Expr::pow((**b).clone(), (**exp).clone());
                    return Some(Expr::div_expr(a_pow, b_pow));
                }
            }
        }
        None
    }
);

rule!(
    PolynomialExpansionRule,
    "polynomial_expansion",
    89,
    Algebraic,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(base, exp) = &expr.kind {
            // Expand (a + b)^n for small integer n
            // BUT only when expansion is likely to help with simplification
            if let AstKind::Add(a, b) = &base.kind
                && let AstKind::Number(n) = &exp.kind
                && *n >= 2.0
                && *n <= 4.0
                && n.fract() == 0.0
            {
                // CONSERVATIVE: Only expand if both terms are pure numbers (will fold to a constant)
                // Otherwise, keep the factored form which is generally more useful
                fn is_number(e: &Expr) -> bool {
                    matches!(e.kind, AstKind::Number(_))
                }

                // Only expand (num + num)^n since that will simplify to a single number
                if !(is_number(a) && is_number(b)) {
                    return None;
                }
                let n_int = *n as i64;
                match n_int {
                    2 => {
                        // (a + b)^2 = a^2 + 2*a*b + b^2
                        let a2 = Expr::pow((**a).clone(), Expr::number(2.0));
                        let b2 = Expr::pow((**b).clone(), Expr::number(2.0));
                        let ab2 = Expr::mul_expr(
                            Expr::number(2.0),
                            Expr::mul_expr((**a).clone(), (**b).clone()),
                        );
                        return Some(Expr::add_expr(Expr::add_expr(a2, ab2), b2));
                    }
                    3 => {
                        // (a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3
                        let a3 = Expr::pow((**a).clone(), Expr::number(3.0));
                        let b3 = Expr::pow((**b).clone(), Expr::number(3.0));
                        let a2b = Expr::mul_expr(
                            Expr::number(3.0),
                            Expr::mul_expr(
                                Expr::pow((**a).clone(), Expr::number(2.0)),
                                (**b).clone(),
                            ),
                        );
                        let ab2 = Expr::mul_expr(
                            Expr::number(3.0),
                            Expr::mul_expr(
                                (**a).clone(),
                                Expr::pow((**b).clone(), Expr::number(2.0)),
                            ),
                        );
                        return Some(Expr::add_expr(
                            Expr::add_expr(a3, a2b),
                            Expr::add_expr(ab2, b3),
                        ));
                    }
                    4 => {
                        // (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4
                        let a4 = Expr::pow((**a).clone(), Expr::number(4.0));
                        let b4 = Expr::pow((**b).clone(), Expr::number(4.0));
                        let a3b = Expr::mul_expr(
                            Expr::number(4.0),
                            Expr::mul_expr(
                                Expr::pow((**a).clone(), Expr::number(3.0)),
                                (**b).clone(),
                            ),
                        );
                        let a2b2 = Expr::mul_expr(
                            Expr::number(6.0),
                            Expr::mul_expr(
                                Expr::pow((**a).clone(), Expr::number(2.0)),
                                Expr::pow((**b).clone(), Expr::number(2.0)),
                            ),
                        );
                        let ab3 = Expr::mul_expr(
                            Expr::number(4.0),
                            Expr::mul_expr(
                                (**a).clone(),
                                Expr::pow((**b).clone(), Expr::number(3.0)),
                            ),
                        );
                        return Some(Expr::add_expr(
                            Expr::add_expr(a4, a3b),
                            Expr::add_expr(a2b2, Expr::add_expr(ab3, b4)),
                        ));
                    }
                    _ => {}
                }
            }
        }
        None
    }
);

rule!(
    ExpandDifferenceOfSquaresProductRule,
    "expand_difference_of_squares_product",
    85,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(a, b) = &expr.kind {
            // Check for (x + y) * (x - y) pattern
            let check_difference_of_squares = |left: &Expr, right: &Expr| -> Option<Expr> {
                if let (AstKind::Add(a1, a2), AstKind::Sub(s1, s2)) = (&left.kind, &right.kind)
                    && a1 == s1
                    && a2 == s2
                {
                    // (a + b)(a - b) = a^2 - b^2
                    let a_squared = Expr::pow((**a1).clone(), Expr::number(2.0));
                    let b_squared = Expr::pow((**a2).clone(), Expr::number(2.0));
                    return Some(Expr::sub_expr(a_squared, b_squared));
                }
                None
            };

            if let Some(result) = check_difference_of_squares(a, b) {
                return Some(result);
            }
            if let Some(result) = check_difference_of_squares(b, a) {
                return Some(result);
            }
        }
        None
    }
);
