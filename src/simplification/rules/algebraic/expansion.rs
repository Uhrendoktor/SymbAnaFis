use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    ExpandPowerForCancellationRule,
    "expand_power_for_cancellation",
    92,
    Algebraic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Div(num, den) = expr {
            // Helper to check if a factor is present in an expression
            let contains_factor = |expr: &Expr, factor: &Expr| -> bool {
                match expr {
                    Expr::Mul(_, _) => {
                        let factors = crate::simplification::helpers::flatten_mul(expr);
                        factors.contains(factor)
                    }
                    _ => expr == factor,
                }
            };

            // Helper to check if expansion is useful
            let check_and_expand = |target: &Expr, other: &Expr| -> Option<Expr> {
                if let Expr::Pow(base, exp) = target
                    && let Expr::Mul(_, _) = &**base
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
                            pow_factors.push(Expr::Pow(Rc::new(factor), exp.clone()));
                        }
                        return Some(crate::simplification::helpers::rebuild_mul(pow_factors));
                    }
                }
                None
            };

            // Try expanding powers in numerator
            if let Some(expanded) = check_and_expand(num, den) {
                return Some(Expr::Div(Rc::new(expanded), den.clone()));
            }

            // Try expanding powers in denominator
            if let Some(expanded) = check_and_expand(den, num) {
                return Some(Expr::Div(num.clone(), Rc::new(expanded)));
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
        if let Expr::Pow(base, exp) = expr {
            // Expand (a*b)^n -> a^n * b^n ONLY if expansion enables simplification
            // This avoids oscillation with common_exponent_mul while still allowing
            // cases like (2*sqrt(x))^2 -> 4*x
            if let Expr::Mul(_a, _b) = &**base
                && let Expr::Number(n) = &**exp
                && *n > 1.0
                && n.fract() == 0.0
                && (*n as i64) < 10
            {
                let base_factors = crate::simplification::helpers::flatten_mul(base);

                // Check if expansion would enable simplification:
                // - Contains a power that would simplify (e.g., sqrt(x)^2 -> x)
                // - Contains a number (coefficient that would be raised to power)
                let has_simplifiable = base_factors.iter().any(|f| {
                    match f {
                        // sqrt(x)^2 -> x, x^(1/2)^2 -> x, etc.
                        Expr::Pow(_, inner_exp) => {
                            if let Expr::Number(inner_n) = &**inner_exp {
                                // Fractional exponent that would become integer
                                (inner_n * n).fract().abs() < 1e-10
                            } else if let Expr::Div(num, den) = &**inner_exp {
                                // x^(a/b) raised to n - check if simplifies
                                if let (Expr::Number(a), Expr::Number(b)) = (&**num, &**den) {
                                    ((a * n) / b).fract().abs() < 1e-10
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        // FunctionCall like sqrt, cbrt
                        Expr::FunctionCall { name, .. } => {
                            matches!(name.as_str(), "sqrt" | "cbrt") && *n >= 2.0
                        }
                        // Numeric coefficient
                        Expr::Number(_) => true,
                        _ => false,
                    }
                });

                if has_simplifiable {
                    let mut factors = Vec::new();
                    for factor in base_factors {
                        factors.push(Expr::Pow(Rc::new(factor), exp.clone()));
                    }
                    return Some(crate::simplification::helpers::rebuild_mul(factors));
                }
            }

            // Expand (a/b)^n -> a^n / b^n ONLY if expansion enables simplification
            // This avoids oscillation with common_exponent_div
            if let Expr::Div(a, b) = &**base
                && let Expr::Number(n) = &**exp
                && *n > 1.0
                && n.fract() == 0.0
                && (*n as i64) < 10
            {
                // Helper to check if a term would simplify when raised to power n
                let would_simplify = |term: &Expr| -> bool {
                    match term {
                        Expr::Pow(_, inner_exp) => {
                            if let Expr::Number(inner_n) = &**inner_exp {
                                (inner_n * n).fract().abs() < 1e-10
                            } else if let Expr::Div(num, den) = &**inner_exp {
                                if let (Expr::Number(a_val), Expr::Number(b_val)) = (&**num, &**den)
                                {
                                    ((a_val * n) / b_val).fract().abs() < 1e-10
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                        Expr::FunctionCall { name, .. } => {
                            matches!(name.as_str(), "sqrt" | "cbrt") && *n >= 2.0
                        }
                        Expr::Number(_) => true,
                        Expr::Mul(_, _) => {
                            // Check factors of multiplication
                            let factors = crate::simplification::helpers::flatten_mul(term);
                            factors.iter().any(|f| match f {
                                Expr::Number(_) => true,
                                Expr::FunctionCall { name, .. } => {
                                    matches!(name.as_str(), "sqrt" | "cbrt")
                                }
                                Expr::Pow(_, inner_exp) => {
                                    if let Expr::Number(inner_n) = &**inner_exp {
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
                    let a_pow = Expr::Pow(a.clone(), exp.clone());
                    let b_pow = Expr::Pow(b.clone(), exp.clone());
                    return Some(Expr::Div(Rc::new(a_pow), Rc::new(b_pow)));
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
        if let Expr::Pow(base, exp) = expr {
            // Expand (a + b)^n for small integer n
            // BUT only when expansion is likely to help with simplification
            if let Expr::Add(a, b) = &**base
                && let Expr::Number(n) = &**exp
                && *n >= 2.0
                && *n <= 4.0
                && n.fract() == 0.0
            {
                // CONSERVATIVE: Only expand if both terms are pure numbers (will fold to a constant)
                // Otherwise, keep the factored form which is generally more useful
                fn is_number(e: &Expr) -> bool {
                    matches!(e, Expr::Number(_))
                }

                // Only expand (num + num)^n since that will simplify to a single number
                if !(is_number(a) && is_number(b)) {
                    return None;
                }
                let n_int = *n as i64;
                match n_int {
                    2 => {
                        // (a + b)^2 = a^2 + 2*a*b + b^2
                        let a2 = Expr::Pow(a.clone(), Rc::new(Expr::Number(2.0)));
                        let b2 = Expr::Pow(b.clone(), Rc::new(Expr::Number(2.0)));
                        let ab2 = Expr::Mul(
                            Rc::new(Expr::Number(2.0)),
                            Rc::new(Expr::Mul(a.clone(), b.clone())),
                        );
                        return Some(Expr::Add(
                            Rc::new(Expr::Add(Rc::new(a2), Rc::new(ab2))),
                            Rc::new(b2),
                        ));
                    }
                    3 => {
                        // (a + b)^3 = a^3 + 3*a^2*b + 3*a*b^2 + b^3
                        let a3 = Expr::Pow(a.clone(), Rc::new(Expr::Number(3.0)));
                        let b3 = Expr::Pow(b.clone(), Rc::new(Expr::Number(3.0)));
                        let a2b = Expr::Mul(
                            Rc::new(Expr::Number(3.0)),
                            Rc::new(Expr::Mul(
                                Rc::new(Expr::Pow(a.clone(), Rc::new(Expr::Number(2.0)))),
                                b.clone(),
                            )),
                        );
                        let ab2 = Expr::Mul(
                            Rc::new(Expr::Number(3.0)),
                            Rc::new(Expr::Mul(
                                a.clone(),
                                Rc::new(Expr::Pow(b.clone(), Rc::new(Expr::Number(2.0)))),
                            )),
                        );
                        return Some(Expr::Add(
                            Rc::new(Expr::Add(Rc::new(a3), Rc::new(a2b))),
                            Rc::new(Expr::Add(Rc::new(ab2), Rc::new(b3))),
                        ));
                    }
                    4 => {
                        // (a + b)^4 = a^4 + 4*a^3*b + 6*a^2*b^2 + 4*a*b^3 + b^4
                        let a4 = Expr::Pow(a.clone(), Rc::new(Expr::Number(4.0)));
                        let b4 = Expr::Pow(b.clone(), Rc::new(Expr::Number(4.0)));
                        let a3b = Expr::Mul(
                            Rc::new(Expr::Number(4.0)),
                            Rc::new(Expr::Mul(
                                Rc::new(Expr::Pow(a.clone(), Rc::new(Expr::Number(3.0)))),
                                b.clone(),
                            )),
                        );
                        let a2b2 = Expr::Mul(
                            Rc::new(Expr::Number(6.0)),
                            Rc::new(Expr::Mul(
                                Rc::new(Expr::Pow(a.clone(), Rc::new(Expr::Number(2.0)))),
                                Rc::new(Expr::Pow(b.clone(), Rc::new(Expr::Number(2.0)))),
                            )),
                        );
                        let ab3 = Expr::Mul(
                            Rc::new(Expr::Number(4.0)),
                            Rc::new(Expr::Mul(
                                a.clone(),
                                Rc::new(Expr::Pow(b.clone(), Rc::new(Expr::Number(3.0)))),
                            )),
                        );
                        return Some(Expr::Add(
                            Rc::new(Expr::Add(Rc::new(a4), Rc::new(a3b))),
                            Rc::new(Expr::Add(
                                Rc::new(a2b2),
                                Rc::new(Expr::Add(Rc::new(ab3), Rc::new(b4))),
                            )),
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
        if let Expr::Mul(a, b) = expr {
            // Check for (x + y) * (x - y) pattern
            let check_difference_of_squares = |left: &Expr, right: &Expr| -> Option<Expr> {
                if let (Expr::Add(a1, a2), Expr::Sub(s1, s2)) = (left, right)
                    && a1 == s1
                    && a2 == s2
                {
                    // (a + b)(a - b) = a^2 - b^2
                    let a_squared = Expr::Pow(a1.clone(), Rc::new(Expr::Number(2.0)));
                    let b_squared = Expr::Pow(a2.clone(), Rc::new(Expr::Number(2.0)));
                    return Some(Expr::Sub(Rc::new(a_squared), Rc::new(b_squared)));
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
