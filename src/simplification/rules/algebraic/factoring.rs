use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

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

    fn apply(&self, expr: &Expr, context: &RuleContext) -> Option<Expr> {
        // For Mul expressions, check if there's a Div nested inside that we can simplify
        if let Expr::Mul(_, _) = expr {
            // Extract all factors including any divisions
            fn find_div_in_mul(e: &Expr) -> Option<(Vec<Expr>, Expr, Expr)> {
                match e {
                    Expr::Mul(a, b) => {
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
                    Expr::Div(num, den) => Some((vec![], (**num).clone(), (**den).clone())),
                    _ => None,
                }
            }

            if let Some((extra_factors, num, den)) = find_div_in_mul(expr) {
                // Combine extra factors with numerator
                let mut all_num_factors = crate::simplification::helpers::flatten_mul(&num);
                all_num_factors.extend(extra_factors);
                let combined_num = crate::simplification::helpers::rebuild_mul(all_num_factors);
                let new_div = Expr::Div(Rc::new(combined_num), Rc::new(den));
                // Let the Div case below handle the cancellation
                return self.apply(&new_div, context);
            }
            return None;
        }

        if let Expr::Div(u, v) = expr {
            let num_factors = crate::simplification::helpers::flatten_mul(u);
            let den_factors = crate::simplification::helpers::flatten_mul(v);

            // 1. Handle numeric coefficients (always safe - nonzero constants)
            let mut num_coeff = 1.0;
            let mut den_coeff = 1.0;
            let mut new_num_factors = Vec::new();
            let mut new_den_factors = Vec::new();

            for f in num_factors {
                if let Expr::Number(n) = f {
                    num_coeff *= n;
                } else {
                    new_num_factors.push(f);
                }
            }

            for f in den_factors {
                if let Expr::Number(n) = f {
                    den_coeff *= n;
                } else {
                    new_den_factors.push(f);
                }
            }

            // Simplify coefficients (e.g. 2/4 -> 1/2) - always safe
            let ratio = num_coeff / den_coeff;
            if ratio.abs() < 1e-10 {
                return Some(Expr::Number(0.0));
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
                match e {
                    Expr::Pow(b, exp) => (b.as_ref().clone(), exp.as_ref().clone()),
                    Expr::FunctionCall { name, args } if args.len() == 1 => {
                        if name == "sqrt" {
                            (args[0].clone(), Expr::Number(0.5))
                        } else if name == "cbrt" {
                            (
                                args[0].clone(),
                                Expr::Div(Rc::new(Expr::Number(1.0)), Rc::new(Expr::Number(3.0))),
                            )
                        } else {
                            (e.clone(), Expr::Number(1.0))
                        }
                    }
                    _ => (e.clone(), Expr::Number(1.0)),
                }
            }

            // Helper to check if a base is a nonzero numeric constant (safe to cancel)
            fn is_safe_to_cancel(base: &Expr) -> bool {
                match base {
                    Expr::Number(n) => n.abs() > 1e-10, // nonzero number
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
                        let new_exp = Expr::Sub(Rc::new(exp_i.clone()), Rc::new(exp_j.clone()));

                        // Simplify exponent
                        let simplified_exp =
                            if let (Expr::Number(n1), Expr::Number(n2)) = (&exp_i, &exp_j) {
                                Expr::Number(n1 - n2)
                            } else {
                                new_exp
                            };

                        if let Expr::Number(n) = simplified_exp {
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
                                    new_num_factors[i] = Expr::Pow(
                                        Rc::new(base_i.clone()),
                                        Rc::new(Expr::Number(n)),
                                    );
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
                                    new_den_factors[j] = Expr::Pow(
                                        Rc::new(base_i.clone()),
                                        Rc::new(Expr::Number(pos_n)),
                                    );
                                }
                                matched = true;
                                break;
                            }
                        } else {
                            // Symbolic exponent subtraction
                            new_num_factors[i] =
                                Expr::Pow(Rc::new(base_i.clone()), Rc::new(simplified_exp));
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
                new_num_factors.insert(0, Expr::Number(num_coeff));
            }
            if den_coeff != 1.0 {
                new_den_factors.insert(0, Expr::Number(den_coeff));
            }

            // Rebuild numerator
            let new_num = if new_num_factors.is_empty() {
                Expr::Number(1.0)
            } else {
                crate::simplification::helpers::rebuild_mul(new_num_factors)
            };

            // Rebuild denominator
            let new_den = if new_den_factors.is_empty() {
                Expr::Number(1.0)
            } else {
                crate::simplification::helpers::rebuild_mul(new_den_factors)
            };

            // If denominator is 1, return numerator
            if let Expr::Number(n) = new_den
                && n == 1.0
            {
                return Some(new_num);
            }

            let res = Expr::Div(Rc::new(new_num), Rc::new(new_den));
            if res != *expr {
                return Some(res);
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
        48 // Higher priority than CommonTermFactoringRule (40) to catch perfect squares first
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Algebraic
    }

    fn applies_to(&self) -> &'static [ExprKind] {
        &[ExprKind::Add]
    }

    fn apply(&self, expr: &Expr, _context: &RuleContext) -> Option<Expr> {
        if let Expr::Add(_, _) = expr {
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
                        if let Expr::Number(n) = &f {
                            coeff *= n;
                        } else {
                            non_numeric.push(f);
                        }
                    }
                    (coeff, non_numeric)
                }

                for term in &terms {
                    match term {
                        Expr::Pow(base, exp) => {
                            if let Expr::Number(n) = &**exp
                                && (*n - 2.0).abs() < 1e-10
                            {
                                // x^2 (no coefficient)
                                square_terms.push((1.0, (**base).clone()));
                                continue;
                            }
                            // Not a square, treat as other
                            linear_terms.push((1.0, term.clone(), Expr::Number(1.0)));
                        }
                        Expr::Number(n) => {
                            constants.push(*n);
                        }
                        Expr::Mul(_, _) => {
                            let (coeff, factors) = extract_coeff_and_factors(term);

                            // Check what kind of term this is
                            if factors.len() == 1 {
                                // c * something
                                if let Expr::Pow(base, exp) = &factors[0]
                                    && let Expr::Number(n) = &**exp
                                    && (*n - 2.0).abs() < 1e-10
                                {
                                    // c * x^2
                                    square_terms.push((coeff, (**base).clone()));
                                    continue;
                                }
                                // c * x -> linear term with implicit 1
                                linear_terms.push((coeff, factors[0].clone(), Expr::Number(1.0)));
                            } else if factors.len() == 2 {
                                // c * a * b
                                linear_terms.push((coeff, factors[0].clone(), factors[1].clone()));
                            } else {
                                // More complex case - skip for now
                            }
                        }
                        other => {
                            // Treat as 1 * other * 1
                            linear_terms.push((1.0, other.clone(), Expr::Number(1.0)));
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
                            if (a == cross_a && b == cross_b) || (a == cross_b && b == cross_a) {
                                let sign = cross_coeff.signum();

                                // Build (sqrt(c1)*a + sign * sqrt(c2)*b)
                                let term_a = if (sqrt_c1 - 1.0).abs() < 1e-10 {
                                    a.clone()
                                } else {
                                    Expr::Mul(
                                        Rc::new(Expr::Number(sqrt_c1.round())),
                                        Rc::new(a.clone()),
                                    )
                                };

                                let term_b = if (sqrt_c2 - 1.0).abs() < 1e-10 {
                                    b.clone()
                                } else {
                                    Expr::Mul(
                                        Rc::new(Expr::Number(sqrt_c2.round())),
                                        Rc::new(b.clone()),
                                    )
                                };

                                let inner = if sign > 0.0 {
                                    Expr::Add(Rc::new(term_a), Rc::new(term_b))
                                } else {
                                    // term_a - term_b
                                    Expr::Add(
                                        Rc::new(term_a),
                                        Rc::new(Expr::Mul(
                                            Rc::new(Expr::Number(-1.0)),
                                            Rc::new(term_b),
                                        )),
                                    )
                                };

                                return Some(Expr::Pow(Rc::new(inner), Rc::new(Expr::Number(2.0))));
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
                            if (a == cross_a && matches!(cross_b, Expr::Number(n) if *n == 1.0))
                                || (a == cross_b && matches!(cross_a, Expr::Number(n) if *n == 1.0))
                            {
                                let sign = c2.signum();

                                let term_a = if (sqrt_c1 - 1.0).abs() < 1e-10 {
                                    a.clone()
                                } else {
                                    Expr::Mul(
                                        Rc::new(Expr::Number(sqrt_c1.round())),
                                        Rc::new(a.clone()),
                                    )
                                };

                                let term_b = Expr::Number(sqrt_c3.round());

                                let inner = if sign > 0.0 {
                                    Expr::Add(Rc::new(term_a), Rc::new(term_b))
                                } else {
                                    // term_a - term_b
                                    Expr::Add(
                                        Rc::new(term_a),
                                        Rc::new(Expr::Mul(
                                            Rc::new(Expr::Number(-1.0)),
                                            Rc::new(term_b),
                                        )),
                                    )
                                };

                                return Some(Expr::Pow(Rc::new(inner), Rc::new(Expr::Number(2.0))));
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
        if let Expr::Add(left, right) = expr {
            // Try both orderings: c*(inner) + d and d + c*(inner)
            let orderings: Vec<(&Expr, &Expr)> = vec![(left, right), (right, left)];

            for (mul_side, const_side) in orderings {
                // const_side should be a number
                if let Expr::Number(d) = const_side {
                    if *d <= 0.0 {
                        continue;
                    }
                    let sqrt_d = d.sqrt();
                    if (sqrt_d - sqrt_d.round()).abs() >= 1e-10 {
                        continue;
                    }

                    // mul_side should be c * (a^2 + a) or c * (a^2 + k*a)
                    if let Expr::Mul(factor1, factor2) = mul_side {
                        // Try both orderings of the multiplication
                        let mul_orderings: Vec<(&Expr, &Expr)> =
                            vec![(&**factor1, &**factor2), (&**factor2, &**factor1)];

                        for (coeff_expr, inner_expr) in mul_orderings {
                            if let Expr::Number(c) = coeff_expr {
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
                                if let Expr::Add(inner1, inner2) = inner_expr {
                                    // Try both orderings
                                    let inner_orderings: Vec<(&Expr, &Expr)> =
                                        vec![(&**inner1, &**inner2), (&**inner2, &**inner1)];

                                    for (square_part, linear_part) in inner_orderings {
                                        // square_part should be a^2
                                        if let Expr::Pow(base, exp) = square_part
                                            && let Expr::Number(n) = &**exp
                                        {
                                            if (*n - 2.0).abs() >= 1e-10 {
                                                continue;
                                            }
                                            let a = (**base).clone();

                                            // linear_part should be a (coefficient 1) or k*a
                                            // For perfect square, coefficient should be 1 in the factored form
                                            // (we already accounted for sqrt(c)*sqrt(d) relationship above)
                                            let linear_matches = match linear_part {
                                                e if *e == a => true,
                                                Expr::Mul(m1, m2) => {
                                                    // Check for 1*a or a*1
                                                    (matches!(&**m1, Expr::Number(k) if (*k - 1.0).abs() < 1e-10)
                                                        && **m2 == a)
                                                        || (matches!(&**m2, Expr::Number(k) if (*k - 1.0).abs() < 1e-10)
                                                            && **m1 == a)
                                                }
                                                _ => false,
                                            };

                                            if linear_matches {
                                                // Build (sqrt(c)*a + sqrt(d))^2
                                                let term_a = if (sqrt_c - 1.0).abs() < 1e-10 {
                                                    a.clone()
                                                } else {
                                                    Expr::Mul(
                                                        Rc::new(Expr::Number(sqrt_c.round())),
                                                        Rc::new(a.clone()),
                                                    )
                                                };

                                                let term_b = Expr::Number(sqrt_d.round());

                                                let inner =
                                                    Expr::Add(Rc::new(term_a), Rc::new(term_b));

                                                return Some(Expr::Pow(
                                                    Rc::new(inner),
                                                    Rc::new(Expr::Number(2.0)),
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

        // Case 4: Handle form: a * (a + 2) + 1 -> (a + 1)^2
        // This is what x^2 + 2*x + 1 becomes after CommonPowerFactoringRule transforms it to x*(x+2) + 1
        if let Expr::Add(left, right) = expr {
            // Try both orderings
            let orderings: Vec<(&Expr, &Expr)> = vec![(left, right), (right, left)];

            for (mul_side, const_side) in orderings {
                // const_side should be 1
                if let Expr::Number(d) = const_side {
                    if (*d - 1.0).abs() >= 1e-10 {
                        continue;
                    }

                    // mul_side should be a * (a + 2)
                    if let Expr::Mul(factor1, factor2) = mul_side {
                        // Try both orderings
                        let mul_orderings: Vec<(&Expr, &Expr)> =
                            vec![(&**factor1, &**factor2), (&**factor2, &**factor1)];

                        for (var_side, add_side) in mul_orderings {
                            // add_side should be (a + 2) or (2 + a)
                            if let Expr::Add(add1, add2) = add_side {
                                let add_orderings: Vec<(&Expr, &Expr)> =
                                    vec![(&**add1, &**add2), (&**add2, &**add1)];

                                for (var_in_add, num_in_add) in add_orderings {
                                    // num_in_add should be 2
                                    if let Expr::Number(n) = num_in_add {
                                        if (*n - 2.0).abs() >= 1e-10 {
                                            continue;
                                        }

                                        // var_side and var_in_add should be the same variable
                                        if var_side == var_in_add {
                                            // Build (a + 1)^2
                                            let inner = Expr::Add(
                                                Rc::new(var_side.clone()),
                                                Rc::new(Expr::Number(1.0)),
                                            );
                                            return Some(Expr::Pow(
                                                Rc::new(inner),
                                                Rc::new(Expr::Number(2.0)),
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

        None
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
            if let Expr::Pow(base, exp) = e
                && let Expr::Number(n) = &**exp
            {
                if (*n - 2.0).abs() < 1e-10 {
                    // x^2 -> x
                    return Some((**base).clone());
                } else if n.fract() == 0.0 && *n > 2.0 && (n / 2.0).fract() == 0.0 {
                    // x^(2k) where k > 1 -> x^k
                    let half_exp = n / 2.0;
                    return Some(Expr::Pow(base.clone(), Rc::new(Expr::Number(half_exp))));
                }
            }
            None
        }

        // Direct Sub case: a^2 - b^2 (or a^(2n) - b^(2m))
        if let Expr::Sub(a, b) = expr {
            // Try standard Pow^2 - Pow^2 pattern
            if let (Some(sqrt_a), Some(sqrt_b)) = (get_square_root_form(a), get_square_root_form(b))
            {
                return Some(Expr::Mul(
                    Rc::new(Expr::Add(Rc::new(sqrt_a.clone()), Rc::new(sqrt_b.clone()))),
                    Rc::new(Expr::Sub(Rc::new(sqrt_a), Rc::new(sqrt_b))),
                ));
            }

            // Handle x^(2n) - 1 (where b is a number that's a perfect square)
            if let Some(sqrt_a) = get_square_root_form(a)
                && let Expr::Number(n) = &**b
            {
                let sqrt_n = n.sqrt();
                if (sqrt_n - sqrt_n.round()).abs() < 1e-10 && *n > 0.0 {
                    let sqrt_val = sqrt_n.round();
                    // x^2 - n => (x - sqrt(n))(x + sqrt(n))
                    return Some(Expr::Mul(
                        Rc::new(Expr::Add(
                            Rc::new(sqrt_a.clone()),
                            Rc::new(Expr::Number(sqrt_val)),
                        )),
                        Rc::new(Expr::Sub(Rc::new(sqrt_a), Rc::new(Expr::Number(sqrt_val)))),
                    ));
                }
            }

            // Handle 1 - x^(2n) (where a is a number that's a perfect square)
            if let Some(sqrt_b) = get_square_root_form(b)
                && let Expr::Number(n) = &**a
            {
                let sqrt_n = n.sqrt();
                if (sqrt_n - sqrt_n.round()).abs() < 1e-10 && *n > 0.0 {
                    let sqrt_val = sqrt_n.round();
                    // n - x^2 => (sqrt(n) - x)(sqrt(n) + x)
                    return Some(Expr::Mul(
                        Rc::new(Expr::Add(
                            Rc::new(Expr::Number(sqrt_val)),
                            Rc::new(sqrt_b.clone()),
                        )),
                        Rc::new(Expr::Sub(Rc::new(Expr::Number(sqrt_val)), Rc::new(sqrt_b))),
                    ));
                }
            }
        }

        // Add case: handle -1 + x^2 or x^2 + (-1) forms (also x^(2n) variants)
        if let Expr::Add(_, _) = expr {
            let terms = crate::simplification::helpers::flatten_add(expr);
            if terms.len() == 2 {
                // Look for one even-powered term and one negative number
                let mut squared_term: Option<Expr> = None; // This will be the sqrt of the power
                let mut constant: Option<f64> = None;

                for term in &terms {
                    match term {
                        Expr::Pow(base, exp) => {
                            if let Expr::Number(n) = &**exp {
                                if (*n - 2.0).abs() < 1e-10 {
                                    // x^2 -> x
                                    squared_term = Some((**base).clone());
                                } else if n.fract() == 0.0 && *n > 2.0 && (n / 2.0).fract() == 0.0 {
                                    // x^(2k) -> x^k
                                    let half_exp = n / 2.0;
                                    squared_term = Some(Expr::Pow(
                                        base.clone(),
                                        Rc::new(Expr::Number(half_exp)),
                                    ));
                                }
                            }
                        }
                        Expr::Number(n) => {
                            constant = Some(*n);
                        }
                        Expr::Mul(coeff, inner) => {
                            // Handle -1 * x^2 case
                            if let Expr::Number(c) = &**coeff
                                && (*c + 1.0).abs() < 1e-10
                            {
                                // -1 * something
                                if let Expr::Pow(base, exp) = &**inner
                                    && let Expr::Number(n) = &**exp
                                {
                                    if (*n - 2.0).abs() < 1e-10 {
                                        squared_term = Some((**base).clone());
                                    } else if n.fract() == 0.0
                                        && *n > 2.0
                                        && (n / 2.0).fract() == 0.0
                                    {
                                        let half_exp = n / 2.0;
                                        squared_term = Some(Expr::Pow(
                                            base.clone(),
                                            Rc::new(Expr::Number(half_exp)),
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
                        return Some(Expr::Mul(
                            Rc::new(Expr::Add(
                                Rc::new(base.clone()),
                                Rc::new(Expr::Number(sqrt_val)),
                            )),
                            Rc::new(Expr::Sub(Rc::new(base), Rc::new(Expr::Number(sqrt_val)))),
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
        if let Expr::Add(a, _b) = expr
            && let Expr::Add(a3, rest1) = &**a
            && let Expr::Add(_a2b3, rest2) = &**rest1
            && let Expr::Add(_ab23, b3) = &**rest2
        {
            // Check if this matches a^3 + 3a^2b + 3ab^2 + b^3
            // This is a simplified check - full implementation would be more complex
            if let Expr::Pow(base1, exp1) = &**a3
                && let Expr::Pow(base4, exp4) = &**b3
                && base1 == base4
                && matches!(&**exp1, Expr::Number(n) if (n - 3.0).abs() < 1e-10)
                && matches!(&**exp4, Expr::Number(n) if (n - 3.0).abs() < 1e-10)
            {
                return Some(Expr::Pow(
                    Rc::new(Expr::Add(base1.clone(), Rc::new(Expr::Number(1.0)))),
                    Rc::new(Expr::Number(3.0)),
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
            match term {
                Expr::Mul(coeff, var) => {
                    if let Expr::Number(n) = *coeff {
                        coeffs_and_terms.push((n, (*var).clone()));
                    } else {
                        coeffs_and_terms.push((1.0, Expr::Mul(coeff, var)));
                    }
                }
                Expr::Number(n) => {
                    coeffs_and_terms.push((n, Expr::Number(1.0)));
                }
                other => {
                    coeffs_and_terms.push((1.0, other));
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
        let gcd_expr = Expr::Number(gcd as f64);
        let mut new_terms = Vec::new();

        for (coeff, term) in coeffs_and_terms {
            let new_coeff = coeff / (gcd as f64);
            if (new_coeff - 1.0).abs() < 1e-10 {
                new_terms.push(term);
            } else if (new_coeff - (-1.0)).abs() < 1e-10 {
                new_terms.push(Expr::Mul(Rc::new(Expr::Number(-1.0)), Rc::new(term)));
            } else {
                new_terms.push(Expr::Mul(Rc::new(Expr::Number(new_coeff)), Rc::new(term)));
            }
        }

        let factored_terms = crate::simplification::helpers::rebuild_add(new_terms);
        Some(Expr::Mul(Rc::new(gcd_expr), Rc::new(factored_terms)))
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
        if let Expr::Mul(_, _) = &terms[0] {
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
        Some(Expr::Mul(Rc::new(common_part), Rc::new(remaining_sum)))
    }
);

rule!(
    CommonPowerFactoringRule,
    "common_power_factoring",
    39,
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

            if let Expr::Pow(base, exp) = &base_expr {
                if let Expr::Number(exp_val) = &**exp
                    && *exp_val > 0.0
                    && exp_val.fract() == 0.0
                {
                    // Integer positive exponent
                    let base_key = format!("{:?}", base);
                    base_exponents
                        .entry(base_key)
                        .or_default()
                        .push((*exp_val, term.clone()));
                }
            } else if let Expr::Symbol(s) = &base_expr {
                // x is treated as x^1
                let base_key = format!("Symbol(\"{}\")", s);
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

                    let base = if let Expr::Pow(b, _) = &sample_base {
                        (**b).clone()
                    } else {
                        sample_base.clone()
                    };

                    let common_factor = if (min_exp - 1.0).abs() < 1e-10 {
                        base.clone()
                    } else {
                        Expr::Pow(Rc::new(base.clone()), Rc::new(Expr::Number(min_exp)))
                    };

                    // Build remaining terms after factoring out
                    let mut remaining_terms = Vec::new();

                    for term in &terms {
                        let (coeff, base_expr) =
                            crate::simplification::helpers::extract_coeff(term);

                        let new_exp = if let Expr::Pow(_, exp) = &base_expr {
                            if let Expr::Number(e) = &**exp {
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
                            Expr::Number(coeff)
                        } else if (new_exp - 1.0).abs() < 1e-10 {
                            // x^1 = x
                            if (coeff - 1.0).abs() < 1e-10 {
                                base.clone()
                            } else {
                                Expr::Mul(Rc::new(Expr::Number(coeff)), Rc::new(base.clone()))
                            }
                        } else {
                            let power =
                                Expr::Pow(Rc::new(base.clone()), Rc::new(Expr::Number(new_exp)));
                            if (coeff - 1.0).abs() < 1e-10 {
                                power
                            } else {
                                Expr::Mul(Rc::new(Expr::Number(coeff)), Rc::new(power))
                            }
                        };

                        remaining_terms.push(remaining);
                    }

                    let remaining_sum =
                        crate::simplification::helpers::rebuild_add(remaining_terms);
                    return Some(Expr::Mul(Rc::new(common_factor), Rc::new(remaining_sum)));
                }
            }
        }

        None
    }
);
