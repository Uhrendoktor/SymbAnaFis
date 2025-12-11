use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};
use std::sync::Arc;

/// Rule for cancelling common terms in fractions: (a*b)/(a*c) -> b/c
/// Also handles powers: x^a / x^b -> x^(a-b)
///
/// In domain-safe mode:
/// - Numeric coefficient simplification is always applied (safe: nonzero constants)
/// - Symbolic factor cancellation only applies to nonzero numeric constants
///
/// In normal mode:
/// - All cancellations are applied (may alter domain by removing x≠0 constraints)
pub(crate) struct FractionCancellationRule;

impl Rule for FractionCancellationRule {
    fn name(&self) -> &'static str {
        "fraction_cancellation"
    }

    fn priority(&self) -> i32 {
        76
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Algebraic
    }

    fn applies_to(&self) -> &'static [ExprKind] {
        // Include Mul because sometimes Div is nested inside Mul and needs to be found
        &[ExprKind::Div, ExprKind::Mul]
    }

    // Note: We don't set alters_domain to true because the rule handles
    // domain safety internally - it always applies safe numeric simplifications
    // and only applies symbolic cancellation when not in domain-safe mode.

    fn apply(&self, expr: &Arc<Expr>, context: &RuleContext) -> Option<Arc<Expr>> {
        // For Mul expressions, check if there's a Div nested inside that we can simplify
        if let AstKind::Mul(_, _) = &expr.kind {
            // Extract all factors including any divisions
            fn find_div_in_mul(e: &Expr) -> Option<(Vec<Expr>, Expr, Expr)> {
                match &e.kind {
                    AstKind::Mul(a, b) => {
                        if let Some((mut factors, num, den)) = find_div_in_mul(a) {
                            factors.push((**b).clone());
                            return Some((factors, num, den));
                        }
                        if let Some((mut factors, num, den)) = find_div_in_mul(b) {
                            factors.push((**a).clone());
                            return Some((factors, num, den));
                        }
                        None
                    }
                    AstKind::Div(num, den) => {
                        Some((vec![], num.as_ref().clone(), den.as_ref().clone()))
                    }
                    _ => None,
                }
            }

            if let Some((extra_factors, num, den)) = find_div_in_mul(expr) {
                // Combine extra factors with numerator
                let mut all_num_factors = crate::simplification::helpers::flatten_mul(&num);
                all_num_factors.extend(extra_factors);
                let combined_num = crate::simplification::helpers::rebuild_mul(all_num_factors);
                let new_div = Expr::div_expr(combined_num, den);
                // Let the Div case below handle the cancellation
                return self.apply(&Arc::new(new_div), context);
            }
            return None;
        }

        if let AstKind::Div(u, v) = &expr.kind {
            let num_factors = crate::simplification::helpers::flatten_mul(u);
            let den_factors = crate::simplification::helpers::flatten_mul(v);

            // 1. Handle numeric coefficients (always safe - nonzero constants)
            let mut num_coeff = 1.0;
            let mut den_coeff = 1.0;
            let mut new_num_factors = Vec::new();
            let mut new_den_factors = Vec::new();

            for f in num_factors {
                if let AstKind::Number(n) = &f.kind {
                    num_coeff *= n;
                } else {
                    new_num_factors.push(f);
                }
            }

            for f in den_factors {
                if let AstKind::Number(n) = &f.kind {
                    den_coeff *= n;
                } else {
                    new_den_factors.push(f);
                }
            }

            // Simplify coefficients (e.g. 2/4 -> 1/2) - always safe
            let ratio = num_coeff / den_coeff;
            if ratio.abs() < 1e-10 {
                return Some(Arc::new(Expr::number(0.0)));
            }

            // Check if ratio or 1/ratio is integer-ish
            // Always keep negative sign in numerator, not denominator
            if (ratio - ratio.round()).abs() < 1e-10 {
                num_coeff = ratio.round();
                den_coeff = 1.0;
            } else if (1.0 / ratio - (1.0 / ratio).round()).abs() < 1e-10 {
                // 1/ratio is an integer, so ratio = 1/n for some integer n
                // Keep sign in numerator: -1/2 should become -1/2, not 1/-2
                let inv = (1.0 / ratio).round();
                if inv < 0.0 {
                    // negative, put -1 in numerator and positive in denominator
                    num_coeff = -1.0;
                    den_coeff = -inv;
                } else {
                    num_coeff = 1.0;
                    den_coeff = inv;
                }
            }
            // Else keep original coefficients

            // Helper to get base and exponent
            fn get_base_exp(e: &Expr) -> (Expr, Expr) {
                match &e.kind {
                    AstKind::Pow(b, exp) => (b.as_ref().clone(), exp.as_ref().clone()),
                    AstKind::FunctionCall { name, args } if args.len() == 1 => {
                        if name == "sqrt" {
                            (args[0].clone(), Expr::number(0.5))
                        } else if name == "cbrt" {
                            (
                                args[0].clone(),
                                Expr::div_expr(Expr::number(1.0), Expr::number(3.0)),
                            )
                        } else {
                            (e.clone(), Expr::number(1.0))
                        }
                    }
                    _ => (e.clone(), Expr::number(1.0)),
                }
            }

            // Helper to check if a base is a nonzero numeric constant (safe to cancel)
            fn is_safe_to_cancel(base: &Expr) -> bool {
                match &base.kind {
                    AstKind::Number(n) => n.abs() > 1e-10, // nonzero number
                    _ => false,
                }
            }

            // 2. Symbolic cancellation
            // In domain-safe mode, only cancel factors that are nonzero constants
            // In normal mode, cancel all matching factors
            let mut i = 0;
            while i < new_num_factors.len() {
                let (base_i, exp_i) = get_base_exp(&new_num_factors[i]);
                let mut matched = false;

                for j in 0..new_den_factors.len() {
                    let (base_j, exp_j) = get_base_exp(&new_den_factors[j]);

                    // Use semantic equivalence check instead of structural equality
                    // This handles cases like Sub(x,1) vs Add(x, Mul(-1, 1))
                    if crate::simplification::helpers::exprs_equivalent(&base_i, &base_j) {
                        // In domain-safe mode, skip cancellation of symbolic factors
                        // (only allow nonzero numeric constants)
                        if context.domain_safe && !is_safe_to_cancel(&base_i) {
                            // Skip this cancellation - it would alter the domain
                            break;
                        }

                        // Found same base, subtract exponents: new_exp = exp_i - exp_j
                        let new_exp = Expr::sub_expr(exp_i.clone(), exp_j.clone());

                        // Simplify exponent
                        let simplified_exp = if let (AstKind::Number(n1), AstKind::Number(n2)) =
                            (&exp_i.kind, &exp_j.kind)
                        {
                            Expr::number(n1 - n2)
                        } else {
                            new_exp
                        };

                        if let AstKind::Number(n) = &simplified_exp.kind {
                            let n = *n;
                            if n == 0.0 {
                                // Cancel completely
                                new_num_factors.remove(i);
                                new_den_factors.remove(j);
                                matched = true;
                                break;
                            } else if n > 0.0 {
                                // Remains in numerator
                                if n == 1.0 {
                                    new_num_factors[i] = base_i.clone();
                                } else {
                                    new_num_factors[i] = Expr::pow(base_i.clone(), Expr::number(n));
                                }
                                new_den_factors.remove(j);
                                matched = true;
                                break;
                            } else {
                                // Moves to denominator (n < 0)
                                new_num_factors.remove(i);
                                let pos_n = -n;
                                if pos_n == 1.0 {
                                    new_den_factors[j] = base_i.clone();
                                } else {
                                    new_den_factors[j] =
                                        Expr::pow(base_i.clone(), Expr::number(pos_n));
                                }
                                matched = true;
                                break;
                            }
                        } else {
                            // Symbolic exponent subtraction
                            new_num_factors[i] = Expr::pow(base_i.clone(), simplified_exp);
                            new_den_factors.remove(j);
                            matched = true;
                            break;
                        }
                    }
                }

                if !matched {
                    i += 1;
                }
            }

            // Add coefficients back
            if num_coeff != 1.0 {
                new_num_factors.insert(0, Expr::number(num_coeff));
            }
            if den_coeff != 1.0 {
                new_den_factors.insert(0, Expr::number(den_coeff));
            }

            // Rebuild numerator
            let new_num = if new_num_factors.is_empty() {
                Expr::number(1.0)
            } else {
                crate::simplification::helpers::rebuild_mul(new_num_factors)
            };

            // Rebuild denominator
            let new_den = if new_den_factors.is_empty() {
                Expr::number(1.0)
            } else {
                crate::simplification::helpers::rebuild_mul(new_den_factors)
            };

            // If denominator is 1, return numerator
            if let AstKind::Number(n) = &new_den.kind
                && *n == 1.0
            {
                return Some(Arc::new(new_num));
            }

            let res = Expr::div_expr(new_num, new_den);
            if res.id != expr.id && res != **expr {
                return Some(Arc::new(res));
            }
        }
        None
    }
}

/// Rule for perfect squares: a^2 + 2ab + b^2 -> (a+b)^2
pub(crate) struct PerfectSquareRule;

impl Rule for PerfectSquareRule {
    fn name(&self) -> &'static str {
        "perfect_square"
    }

    fn priority(&self) -> i32 {
        100 // Highest priority to catch perfect squares first
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Algebraic
    }

    fn applies_to(&self) -> &'static [ExprKind] {
        &[ExprKind::Add]
    }

    fn apply(&self, expr: &Arc<Expr>, _context: &RuleContext) -> Option<Arc<Expr>> {
        // Wrapper: delegate to inner logic and wrap result in Rc
        fn apply_inner(expr: &Expr) -> Option<Expr> {
            if let AstKind::Add(_, _) = &expr.kind {
                let terms = crate::simplification::helpers::flatten_add(expr);

                if terms.len() == 3 {
                    // Try to match pattern: c1*a^2 + c2*a*b + c3*b^2
                    let mut square_terms: Vec<(f64, Expr)> = Vec::new(); // (coefficient, base)
                    let mut linear_terms: Vec<(f64, Expr, Expr)> = Vec::new(); // (coefficient, base1, base2)
                    let mut constants = Vec::new();

                    // Helper to extract coefficient and variables from any multiplication structure
                    fn extract_coeff_and_factors(term: &Expr) -> (f64, Vec<Expr>) {
                        let factors = crate::simplification::helpers::flatten_mul(term);
                        let mut coeff = 1.0;
                        let mut non_numeric: Vec<Expr> = Vec::new();

                        for f in factors {
                            if let AstKind::Number(n) = &f.kind {
                                coeff *= n;
                            } else {
                                non_numeric.push(f);
                            }
                        }
                        (coeff, non_numeric)
                    }

                    for term in &terms {
                        match &term.kind {
                            AstKind::Pow(base, exp) => {
                                if let AstKind::Number(n) = &exp.kind
                                    && (*n - 2.0).abs() < 1e-10
                                {
                                    // x^2 (no coefficient)
                                    square_terms.push((1.0, (**base).clone()));
                                    continue;
                                }
                                // Not a square, treat as other
                                linear_terms.push((1.0, term.clone(), Expr::number(1.0)));
                            }
                            AstKind::Number(n) => {
                                constants.push(*n);
                            }
                            AstKind::Mul(_, _) => {
                                let (coeff, factors) = extract_coeff_and_factors(term);

                                // Check what kind of term this is
                                if factors.len() == 1 {
                                    // c * something
                                    if let AstKind::Pow(base, exp) = &factors[0].kind
                                        && let AstKind::Number(n) = &exp.kind
                                        && (*n - 2.0).abs() < 1e-10
                                    {
                                        // c * x^2
                                        square_terms.push((coeff, (**base).clone()));
                                        continue;
                                    }
                                    // c * x -> linear term with implicit 1
                                    linear_terms.push((
                                        coeff,
                                        factors[0].clone(),
                                        Expr::number(1.0),
                                    ));
                                } else if factors.len() == 2 {
                                    // c * a * b
                                    linear_terms.push((
                                        coeff,
                                        factors[0].clone(),
                                        factors[1].clone(),
                                    ));
                                } else {
                                    // More complex case - skip for now
                                }
                            }
                            _ => {
                                // Treat as 1 * other * 1
                                linear_terms.push((1.0, term.clone(), Expr::number(1.0)));
                            }
                        }
                    }

                    // Case 1: Standard perfect square a^2 + 2*a*b + b^2
                    if square_terms.len() == 2 && linear_terms.len() == 1 {
                        let (c1, a) = &square_terms[0];
                        let (c2, b) = &square_terms[1];
                        let (cross_coeff, cross_a, cross_b) = &linear_terms[0];

                        // Check if c1 and c2 have integer square roots
                        let sqrt_c1 = c1.sqrt();
                        let sqrt_c2 = c2.sqrt();

                        if (sqrt_c1 - sqrt_c1.round()).abs() < 1e-10
                            && (sqrt_c2 - sqrt_c2.round()).abs() < 1e-10
                        {
                            // Check if cross_coeff = +/- 2 * sqrt(c1) * sqrt(c2)
                            let expected_cross_abs = (2.0 * sqrt_c1 * sqrt_c2).abs();
                            let cross_coeff_abs = cross_coeff.abs();

                            if (expected_cross_abs - cross_coeff_abs).abs() < 1e-10 {
                                // Check if the variables match
                                if (a == cross_a && b == cross_b) || (a == cross_b && b == cross_a)
                                {
                                    let sign = cross_coeff.signum();

                                    // Build (sqrt(c1)*a + sign * sqrt(c2)*b)
                                    let term_a = if (sqrt_c1 - 1.0).abs() < 1e-10 {
                                        a.clone()
                                    } else {
                                        Expr::mul_expr(Expr::number(sqrt_c1.round()), a.clone())
                                    };

                                    let term_b = if (sqrt_c2 - 1.0).abs() < 1e-10 {
                                        b.clone()
                                    } else {
                                        Expr::mul_expr(Expr::number(sqrt_c2.round()), b.clone())
                                    };

                                    let inner = if sign > 0.0 {
                                        Expr::add_expr(term_a, term_b)
                                    } else {
                                        // term_a - term_b
                                        Expr::add_expr(
                                            term_a,
                                            Expr::mul_expr(Expr::number(-1.0), term_b),
                                        )
                                    };

                                    return Some(Expr::pow(inner, Expr::number(2.0)));
                                }
                            }
                        }
                    }

                    // Case 2: One square + constant + linear: c1*a^2 + c2*a + c3
                    if square_terms.len() == 1 && linear_terms.len() == 1 && constants.len() == 1 {
                        let (c1, a) = &square_terms[0];
                        let (c2, cross_a, cross_b) = &linear_terms[0];
                        let c3 = constants[0];

                        // Check if c1 and c3 have integer square roots
                        let sqrt_c1 = c1.sqrt();
                        let sqrt_c3 = c3.sqrt();

                        if (sqrt_c1 - sqrt_c1.round()).abs() < 1e-10
                            && (sqrt_c3 - sqrt_c3.round()).abs() < 1e-10
                        {
                            // Check if c2 = +/- 2 * sqrt(c1) * sqrt(c3)
                            let expected_cross_abs = (2.0 * sqrt_c1 * sqrt_c3).abs();
                            let cross_coeff_abs = c2.abs();

                            if (expected_cross_abs - cross_coeff_abs).abs() < 1e-10 {
                                // Check if linear term matches
                                if (a == cross_a
                                    && matches!(&cross_b.kind, AstKind::Number(n) if *n == 1.0))
                                    || (a == cross_b
                                        && matches!(&cross_a.kind, AstKind::Number(n) if *n == 1.0))
                                {
                                    let sign = c2.signum();

                                    let term_a = if (sqrt_c1 - 1.0).abs() < 1e-10 {
                                        a.clone()
                                    } else {
                                        Expr::mul_expr(Expr::number(sqrt_c1.round()), a.clone())
                                    };

                                    let term_b = Expr::number(sqrt_c3.round());

                                    let inner = if sign > 0.0 {
                                        Expr::add_expr(term_a, term_b)
                                    } else {
                                        // term_a - term_b
                                        Expr::add_expr(
                                            term_a,
                                            Expr::mul_expr(Expr::number(-1.0), term_b),
                                        )
                                    };

                                    return Some(Expr::pow(inner, Expr::number(2.0)));
                                }
                            }
                        }
                    }
                }
            }

            // Case 3: Handle post-GCD form: c * (a^2 + a) + d
            // This happens when 4*x^2 + 4*x + 1 gets transformed to 4*(x^2 + x) + 1
            // We need to detect (sqrt(c)*a + sqrt(d))^2 when c = sqrt(c)^2 and d = sqrt(d)^2
            // and 2*sqrt(c)*sqrt(d) = c (the coefficient on x inside the factored part)
            if let AstKind::Add(left, right) = &expr.kind {
                // Try both orderings: c*(inner) + d and d + c*(inner)
                let orderings: Vec<(&Expr, &Expr)> = vec![(left, right), (right, left)];

                for (mul_side, const_side) in orderings {
                    // const_side should be a number
                    if let AstKind::Number(d) = &const_side.kind {
                        if *d <= 0.0 {
                            continue;
                        }
                        let sqrt_d = d.sqrt();
                        if (sqrt_d - sqrt_d.round()).abs() >= 1e-10 {
                            continue;
                        }

                        // mul_side should be c * (a^2 + a) or c * (a^2 + k*a)
                        if let AstKind::Mul(factor1, factor2) = &mul_side.kind {
                            // Try both orderings of the multiplication
                            let mul_orderings: Vec<(&Expr, &Expr)> =
                                vec![(&**factor1, &**factor2), (&**factor2, &**factor1)];

                            for (coeff_expr, inner_expr) in mul_orderings {
                                if let AstKind::Number(c) = &coeff_expr.kind {
                                    if *c <= 0.0 {
                                        continue;
                                    }
                                    let sqrt_c = c.sqrt();
                                    if (sqrt_c - sqrt_c.round()).abs() >= 1e-10 {
                                        continue;
                                    }

                                    // Check if 2 * sqrt(c) * sqrt(d) = c
                                    // For 4*x^2 + 4*x + 1: c=4, d=1, sqrt(c)=2, sqrt(d)=1
                                    // 2 * 2 * 1 = 4 = c ✓
                                    let expected_c = 2.0 * sqrt_c * sqrt_d;
                                    if (expected_c - *c).abs() >= 1e-10 {
                                        continue;
                                    }

                                    // inner_expr should be a^2 + a or a^2 + k*a (where k matches our expected coefficient ratio)
                                    if let AstKind::Add(inner1, inner2) = &inner_expr.kind {
                                        // Try both orderings
                                        let inner_orderings: Vec<(&Expr, &Expr)> =
                                            vec![(&**inner1, &**inner2), (&**inner2, &**inner1)];

                                        for (square_part, linear_part) in inner_orderings {
                                            // square_part should be a^2
                                            if let AstKind::Pow(base, exp) = &square_part.kind
                                                && let AstKind::Number(n) = &exp.kind
                                            {
                                                if (*n - 2.0).abs() >= 1e-10 {
                                                    continue;
                                                }
                                                let a = (**base).clone();

                                                // linear_part should be a (coefficient 1) or k*a
                                                // For perfect square, coefficient should be 1 in the factored form
                                                // (we already accounted for sqrt(c)*sqrt(d) relationship above)
                                                let linear_matches = match &linear_part.kind {
                                                    _ if linear_part == &a => true,
                                                    AstKind::Mul(m1, m2) => {
                                                        // Check for 1*a or a*1
                                                        (matches!(&m1.kind, AstKind::Number(k) if (*k - 1.0).abs() < 1e-10)
                                                            && **m2 == a)
                                                            || (matches!(&m2.kind, AstKind::Number(k) if (*k - 1.0).abs() < 1e-10)
                                                                && **m1 == a)
                                                    }
                                                    _ => false,
                                                };

                                                if linear_matches {
                                                    // Build (sqrt(c)*a + sqrt(d))^2
                                                    let term_a = if (sqrt_c - 1.0).abs() < 1e-10 {
                                                        a.clone()
                                                    } else {
                                                        Expr::mul_expr(
                                                            Expr::number(sqrt_c.round()),
                                                            a.clone(),
                                                        )
                                                    };

                                                    let term_b = Expr::number(sqrt_d.round());

                                                    let inner = Expr::add_expr(term_a, term_b);

                                                    return Some(Expr::pow(
                                                        inner,
                                                        Expr::number(2.0),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Case 4: Handle generalized form: c * a * (a + k) + d -> (sqrt(c)*a + sqrt(d))^2
            // For this to be a perfect square: c*k = 2*sqrt(c)*sqrt(d)
            // Example: 4x(x+1) + 1 -> c=4, k=1, d=1, check: 4*1 = 2*2*1 = 4 ✓ -> (2x + 1)^2
            // Example: x*(x+2) + 1 -> c=1, k=2, d=1, check: 1*2 = 2*1*1 = 2 ✓ -> (x + 1)^2
            if let AstKind::Add(left, right) = &expr.kind {
                // Try both orderings
                let orderings: Vec<(&Expr, &Expr)> = vec![(left, right), (right, left)];

                for (mul_side, const_side) in orderings {
                    // const_side should be a positive number with integer square root
                    if let AstKind::Number(d) = &const_side.kind {
                        if *d <= 0.0 {
                            continue;
                        }
                        let sqrt_d = d.sqrt();
                        if (sqrt_d - sqrt_d.round()).abs() >= 1e-10 {
                            continue;
                        }

                        // mul_side should be c * a * (a + k) - extract coefficient and factors
                        let mul_factors = crate::simplification::helpers::flatten_mul(mul_side);
                        let mut c = 1.0;
                        let non_numeric: Vec<Expr> = mul_factors
                            .into_iter()
                            .filter(|f| {
                                if let AstKind::Number(n) = &f.kind {
                                    c *= n;
                                    false
                                } else {
                                    true
                                }
                            })
                            .collect();

                        // c must have integer square root
                        if c <= 0.0 {
                            continue;
                        }
                        let sqrt_c = c.sqrt();
                        if (sqrt_c - sqrt_c.round()).abs() >= 1e-10 {
                            continue;
                        }

                        // non_numeric should have exactly 2 factors: a and (a + k)
                        if non_numeric.len() != 2 {
                            continue;
                        }

                        // Try both orderings: (a, (a+k)) and ((a+k), a)
                        let factor_orderings = vec![
                            (non_numeric[0].clone(), non_numeric[1].clone()),
                            (non_numeric[1].clone(), non_numeric[0].clone()),
                        ];

                        for (var_factor, add_factor) in factor_orderings {
                            // add_factor should be (a + k) or (k + a)
                            if let AstKind::Add(add1, add2) = &add_factor.kind {
                                let add_orderings: Vec<(&Expr, &Expr)> =
                                    vec![(&**add1, &**add2), (&**add2, &**add1)];

                                for (var_in_add, k_expr) in add_orderings {
                                    // k_expr should be a number
                                    if let AstKind::Number(k) = &k_expr.kind {
                                        // var_factor and var_in_add should be the same
                                        if &var_factor != var_in_add {
                                            continue;
                                        }

                                        // Check perfect square condition: c*k = 2*sqrt(c)*sqrt(d)
                                        let lhs = c * k;
                                        let rhs = 2.0 * sqrt_c * sqrt_d;

                                        if (lhs - rhs).abs() >= 1e-10 {
                                            continue;
                                        }

                                        // Build (sqrt(c)*a + sqrt(d))^2
                                        let term_a = if (sqrt_c - 1.0).abs() < 1e-10 {
                                            var_factor.clone()
                                        } else {
                                            Expr::mul_expr(
                                                Expr::number(sqrt_c.round()),
                                                var_factor.clone(),
                                            )
                                        };

                                        let term_b = Expr::number(sqrt_d.round());

                                        let inner = Expr::add_expr(term_a, term_b);
                                        return Some(Expr::pow(inner, Expr::number(2.0)));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            None
        }
        // Call inner function and wrap result in Rc
        apply_inner(expr.as_ref()).map(Arc::new)
    }
}

rule!(
    FactorDifferenceOfSquaresRule,
    "factor_difference_of_squares",
    46,
    Algebraic,
    &[ExprKind::Sub, ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        // Look for a^2 - b^2 pattern (either as Sub or Add with negative term)

        // Helper to extract square root of power: x^4 -> (x^2, true), x^2 -> (x, true), x^3 -> (x, false)
        fn get_square_root_form(e: &Expr) -> Option<Expr> {
            if let AstKind::Pow(base, exp) = &e.kind
                && let AstKind::Number(n) = &exp.kind
            {
                if (*n - 2.0).abs() < 1e-10 {
                    // x^2 -> x
                    return Some((**base).clone());
                } else if n.fract() == 0.0 && *n > 2.0 && (n / 2.0).fract() == 0.0 {
                    // x^(2k) where k > 1 -> x^k
                    let half_exp = n / 2.0;
                    return Some(Expr::pow((**base).clone(), Expr::number(half_exp)));
                }
            }
            None
        }

        // Direct Sub case: a^2 - b^2 (or a^(2n) - b^(2m))
        if let AstKind::Sub(a, b) = &expr.kind {
            // Try standard Pow^2 - Pow^2 pattern
            if let (Some(sqrt_a), Some(sqrt_b)) = (get_square_root_form(a), get_square_root_form(b))
            {
                return Some(Expr::mul_expr(
                    Expr::add_expr(sqrt_a.clone(), sqrt_b.clone()),
                    Expr::sub_expr(sqrt_a, sqrt_b),
                ));
            }

            // Handle x^(2n) - 1 (where b is a number that's a perfect square)
            if let Some(sqrt_a) = get_square_root_form(a)
                && let AstKind::Number(n) = &b.kind
            {
                let sqrt_n = n.sqrt();
                if (sqrt_n - sqrt_n.round()).abs() < 1e-10 && *n > 0.0 {
                    let sqrt_val = sqrt_n.round();
                    // x^2 - n => (x - sqrt(n))(x + sqrt(n))
                    return Some(Expr::mul_expr(
                        Expr::add_expr(sqrt_a.clone(), Expr::number(sqrt_val)),
                        Expr::sub_expr(sqrt_a, Expr::number(sqrt_val)),
                    ));
                }
            }

            // Handle 1 - x^(2n) (where a is a number that's a perfect square)
            if let Some(sqrt_b) = get_square_root_form(b)
                && let AstKind::Number(n) = &a.kind
            {
                let sqrt_n = n.sqrt();
                if (sqrt_n - sqrt_n.round()).abs() < 1e-10 && *n > 0.0 {
                    let sqrt_val = sqrt_n.round();
                    // n - x^2 => (sqrt(n) - x)(sqrt(n) + x)
                    return Some(Expr::mul_expr(
                        Expr::add_expr(Expr::number(sqrt_val), sqrt_b.clone()),
                        Expr::sub_expr(Expr::number(sqrt_val), sqrt_b),
                    ));
                }
            }
        }

        // Add case: handle -1 + x^2 or x^2 + (-1) forms (also x^(2n) variants)
        if let AstKind::Add(_, _) = &expr.kind {
            let terms = crate::simplification::helpers::flatten_add(expr);
            if terms.len() == 2 {
                // Look for one even-powered term and one negative number
                let mut squared_term: Option<Expr> = None; // This will be the sqrt of the power
                let mut constant: Option<f64> = None;

                for term in &terms {
                    match &term.kind {
                        AstKind::Pow(base, exp) => {
                            if let AstKind::Number(n) = &exp.kind {
                                if (*n - 2.0).abs() < 1e-10 {
                                    // x^2 -> x
                                    squared_term = Some((**base).clone());
                                } else if n.fract() == 0.0 && *n > 2.0 && (n / 2.0).fract() == 0.0 {
                                    // x^(2k) -> x^k
                                    let half_exp = n / 2.0;
                                    squared_term =
                                        Some(Expr::pow((**base).clone(), Expr::number(half_exp)));
                                }
                            }
                        }
                        AstKind::Number(n) => {
                            constant = Some(*n);
                        }
                        AstKind::Mul(coeff, inner) => {
                            // Handle -1 * x^2 case
                            if let AstKind::Number(c) = &coeff.kind
                                && (*c + 1.0).abs() < 1e-10
                            {
                                // -1 * something
                                if let AstKind::Pow(base, exp) = &inner.kind
                                    && let AstKind::Number(n) = &exp.kind
                                {
                                    if (*n - 2.0).abs() < 1e-10 {
                                        squared_term = Some((**base).clone());
                                    } else if n.fract() == 0.0
                                        && *n > 2.0
                                        && (n / 2.0).fract() == 0.0
                                    {
                                        let half_exp = n / 2.0;
                                        squared_term = Some(Expr::pow(
                                            (**base).clone(),
                                            Expr::number(half_exp),
                                        ));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Check for pattern: x^2 + (-c) where c is a perfect square
                if let (Some(base), Some(c)) = (squared_term, constant)
                    && c < 0.0
                {
                    // x^2 + (-c) = x^2 - c
                    let pos_c = -c;
                    let sqrt_c = pos_c.sqrt();
                    if (sqrt_c - sqrt_c.round()).abs() < 1e-10 {
                        let sqrt_val = sqrt_c.round();
                        // x^2 - c => (x - sqrt(c))(x + sqrt(c))
                        return Some(Expr::mul_expr(
                            Expr::add_expr(base.clone(), Expr::number(sqrt_val)),
                            Expr::sub_expr(base, Expr::number(sqrt_val)),
                        ));
                    }
                }
            }
        }

        None
    }
);

rule!(
    PerfectCubeRule,
    "perfect_cube",
    40,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        // Look for a^3 + 3a^2b + 3ab^2 + b^3 pattern
        // This is more complex, so we'll look for simpler patterns
        if let AstKind::Add(a, _b) = &expr.kind
            && let AstKind::Add(a3, rest1) = &a.kind
            && let AstKind::Add(_a2b3, rest2) = &rest1.kind
            && let AstKind::Add(_ab23, b3) = &rest2.kind
        {
            // Check if this matches a^3 + 3a^2b + 3ab^2 + b^3
            // This is a simplified check - full implementation would be more complex
            if let AstKind::Pow(base1, exp1) = &a3.kind
                && let AstKind::Pow(base4, exp4) = &b3.kind
                && base1 == base4
                && matches!(&exp1.kind, AstKind::Number(n) if (n - 3.0).abs() < 1e-10)
                && matches!(&exp4.kind, AstKind::Number(n) if (n - 3.0).abs() < 1e-10)
            {
                return Some(Expr::pow(
                    Expr::add_expr((**base1).clone(), Expr::number(1.0)),
                    Expr::number(3.0),
                ));
            }
        }
        None
    }
);

rule!(
    NumericGcdFactoringRule,
    "numeric_gcd_factoring",
    42,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        // Factor out greatest common divisor from numeric coefficients
        let terms = crate::simplification::helpers::flatten_add(expr);

        // Extract coefficients and variables
        let mut coeffs_and_terms = Vec::new();
        for term in terms {
            match &term.kind {
                AstKind::Mul(coeff, var) => {
                    if let AstKind::Number(n) = &coeff.kind {
                        coeffs_and_terms.push((*n, (**var).clone()));
                    } else {
                        coeffs_and_terms.push((
                            1.0,
                            Expr::mul_expr(coeff.as_ref().clone(), var.as_ref().clone()),
                        ));
                    }
                }
                AstKind::Number(n) => {
                    coeffs_and_terms.push((*n, Expr::number(1.0)));
                }
                _ => {
                    coeffs_and_terms.push((1.0, term));
                }
            }
        }

        // Find GCD of coefficients
        let coeffs: Vec<i64> = coeffs_and_terms
            .iter()
            .map(|(c, _)| *c as i64)
            .filter(|&c| c != 0)
            .collect();

        if coeffs.len() <= 1 {
            return None;
        }

        let gcd = coeffs
            .iter()
            .fold(coeffs[0], |a, &b| crate::simplification::helpers::gcd(a, b));

        if gcd <= 1 {
            return None;
        }

        // Factor out the GCD
        let gcd_expr = Expr::number(gcd as f64);
        let mut new_terms = Vec::new();

        for (coeff, term) in coeffs_and_terms {
            let new_coeff = coeff / (gcd as f64);
            if (new_coeff - 1.0).abs() < 1e-10 {
                new_terms.push(term);
            } else if (new_coeff - (-1.0)).abs() < 1e-10 {
                new_terms.push(Expr::mul_expr(Expr::number(-1.0), term));
            } else {
                new_terms.push(Expr::mul_expr(Expr::number(new_coeff), term));
            }
        }

        let factored_terms = crate::simplification::helpers::rebuild_add(new_terms);
        Some(Expr::mul_expr(gcd_expr, factored_terms))
    }
);

rule!(
    CommonTermFactoringRule,
    "common_term_factoring",
    40,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        let terms = crate::simplification::helpers::flatten_add(expr);

        if terms.len() < 2 {
            return None;
        }

        // Find common factors across all terms
        let mut common_factors = Vec::new();

        // Start with factors from first term
        if let AstKind::Mul(_, _) = &terms[0].kind {
            let first_factors = crate::simplification::helpers::flatten_mul(&terms[0]);
            for factor in first_factors {
                // Check if this factor appears in all other terms
                let mut all_have_factor = true;
                for term in &terms[1..] {
                    if !crate::simplification::helpers::contains_factor(term, &factor) {
                        all_have_factor = false;
                        break;
                    }
                }
                if all_have_factor {
                    common_factors.push(factor);
                }
            }
        }

        if common_factors.is_empty() {
            return None;
        }

        // Factor out common factors
        let common_part = if common_factors.len() == 1 {
            common_factors[0].clone()
        } else {
            crate::simplification::helpers::rebuild_mul(common_factors)
        };

        let mut remaining_terms = Vec::new();
        for term in terms {
            remaining_terms.push(crate::simplification::helpers::remove_factors(
                &term,
                &common_part,
            ));
        }

        let remaining_sum = crate::simplification::helpers::rebuild_add(remaining_terms);
        Some(Expr::mul_expr(common_part, remaining_sum))
    }
);

rule!(
    CommonPowerFactoringRule,
    "common_power_factoring",
    43,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        let terms = crate::simplification::helpers::flatten_add(expr);

        if terms.len() < 2 {
            return None;
        }

        // Look for common power patterns like x^3 + x^2 -> x^2(x + 1)
        // We need to find a common base and factor out the minimum exponent

        // Collect all (base, exponent) pairs from power terms
        let mut base_exponents: std::collections::HashMap<String, Vec<(f64, Expr)>> =
            std::collections::HashMap::new();

        for term in &terms {
            let (_, base_expr) = crate::simplification::helpers::extract_coeff(term);

            if let AstKind::Pow(base, exp) = &base_expr.kind {
                if let AstKind::Number(exp_val) = &exp.kind
                    && *exp_val > 0.0
                    && exp_val.fract() == 0.0
                {
                    // Integer positive exponent
                    // Use standard signature for grouping equivalent bases
                    let base_key = crate::simplification::helpers::get_term_signature(base);
                    base_exponents
                        .entry(base_key)
                        .or_default()
                        .push((*exp_val, term.clone()));
                }
            } else if let AstKind::Symbol(s) = &base_expr.kind {
                // x is treated as x^1
                // Manually construct signature to match get_term_signature output for Symbol
                let base_key = format!("symbol:{}", s);
                base_exponents
                    .entry(base_key)
                    .or_default()
                    .push((1.0, term.clone()));
            }
        }

        // Find a base that appears in ALL terms with different exponents
        for exp_terms in base_exponents.values() {
            if exp_terms.len() == terms.len() && exp_terms.len() >= 2 {
                // This base appears in all terms
                let exponents: Vec<f64> = exp_terms.iter().map(|(e, _)| *e).collect();
                let min_exp = exponents.iter().cloned().fold(f64::INFINITY, f64::min);

                // Check if we have different exponents (otherwise no factoring needed)
                if exponents.iter().all(|e| (*e - min_exp).abs() < 1e-10) {
                    continue; // All same exponent, skip
                }

                if min_exp >= 1.0 {
                    // We can factor out base^min_exp
                    // Need to reconstruct the base from the base_key
                    let sample_term = &exp_terms[0].1;
                    let (_, sample_base) =
                        crate::simplification::helpers::extract_coeff(sample_term);

                    let base = if let AstKind::Pow(b, _) = &sample_base.kind {
                        (**b).clone()
                    } else {
                        sample_base.clone()
                    };

                    let common_factor = if (min_exp - 1.0).abs() < 1e-10 {
                        base.clone()
                    } else {
                        Expr::pow(base.clone(), Expr::number(min_exp))
                    };

                    // Build remaining terms after factoring out
                    let mut remaining_terms = Vec::new();

                    for term in &terms {
                        let (coeff, base_expr) =
                            crate::simplification::helpers::extract_coeff(term);

                        let new_exp = if let AstKind::Pow(_, exp) = &base_expr.kind {
                            if let AstKind::Number(e) = &exp.kind {
                                *e - min_exp
                            } else {
                                continue;
                            }
                        } else {
                            // Symbol case: x -> x^1, so new_exp = 1 - min_exp
                            1.0 - min_exp
                        };

                        let remaining = if new_exp.abs() < 1e-10 {
                            // x^0 = 1
                            Expr::number(coeff)
                        } else if (new_exp - 1.0).abs() < 1e-10 {
                            // x^1 = x
                            if (coeff - 1.0).abs() < 1e-10 {
                                base.clone()
                            } else {
                                Expr::mul_expr(Expr::number(coeff), base.clone())
                            }
                        } else {
                            let power = Expr::pow(base.clone(), Expr::number(new_exp));
                            if (coeff - 1.0).abs() < 1e-10 {
                                power
                            } else {
                                Expr::mul_expr(Expr::number(coeff), power)
                            }
                        };

                        remaining_terms.push(remaining);
                    }

                    let remaining_sum =
                        crate::simplification::helpers::rebuild_add(remaining_terms);
                    return Some(Expr::mul_expr(common_factor, remaining_sum));
                }
            }
        }

        None
    }
);
