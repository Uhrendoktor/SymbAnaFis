use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(DivSelfRule, "div_self", 78, Algebraic, &[ExprKind::Div], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::Div(u, v) = expr
        && u == v
    {
        return Some(Expr::Number(1.0));
    }
    None
});

rule!(
    DivDivRule,
    "div_div_flatten",
    92,
    Algebraic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Div(num, den) = expr {
            // Case 1: (a/b)/(c/d) -> (a*d)/(b*c)
            if let (Expr::Div(a, b), Expr::Div(c, d)) = (&**num, &**den) {
                return Some(Expr::Div(
                    Rc::new(Expr::Mul(a.clone(), d.clone())),
                    Rc::new(Expr::Mul(b.clone(), c.clone())),
                ));
            }
            // Case 2: x/(c/d) -> (x*d)/c
            if let Expr::Div(c, d) = &**den {
                return Some(Expr::Div(
                    Rc::new(Expr::Mul(num.clone(), d.clone())),
                    c.clone(),
                ));
            }
            // Case 3: (a/b)/y -> a/(b*y)
            if let Expr::Div(a, b) = &**num {
                return Some(Expr::Div(
                    a.clone(),
                    Rc::new(Expr::Mul(b.clone(), den.clone())),
                ));
            }
        }
        None
    }
);

rule!(
    CombineNestedFractionRule,
    "combine_nested_fraction",
    91,
    Algebraic,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Div(num, outer_den) = expr {
            // Case 1: (a + b/c) / d -> (a*c + b) / (c*d)
            if let Expr::Add(a, v) = &**num
                && let Expr::Div(b, c) = &**v
            {
                // (a*c + b) / (c*d)
                let new_num = Expr::Add(Rc::new(Expr::Mul(a.clone(), c.clone())), b.clone());
                let new_den = Expr::Mul(c.clone(), outer_den.clone());
                return Some(Expr::Div(Rc::new(new_num), Rc::new(new_den)));
            }
            // Case 2: (b/c + a) / d -> (b + a*c) / (c*d)
            if let Expr::Add(u, a) = &**num
                && let Expr::Div(b, c) = &**u
            {
                let new_num = Expr::Add(b.clone(), Rc::new(Expr::Mul(a.clone(), c.clone())));
                let new_den = Expr::Mul(c.clone(), outer_den.clone());
                return Some(Expr::Div(Rc::new(new_num), Rc::new(new_den)));
            }
            // Case 3: (a - b/c) / d -> (a*c - b) / (c*d)
            if let Expr::Sub(a, v) = &**num
                && let Expr::Div(b, c) = &**v
            {
                let new_num = Expr::Sub(Rc::new(Expr::Mul(a.clone(), c.clone())), b.clone());
                let new_den = Expr::Mul(c.clone(), outer_den.clone());
                return Some(Expr::Div(Rc::new(new_num), Rc::new(new_den)));
            }
            // Case 4: (b/c - a) / d -> (b - a*c) / (c*d)
            if let Expr::Sub(u, a) = &**num
                && let Expr::Div(b, c) = &**u
            {
                let new_num = Expr::Sub(b.clone(), Rc::new(Expr::Mul(a.clone(), c.clone())));
                let new_den = Expr::Mul(c.clone(), outer_den.clone());
                return Some(Expr::Div(Rc::new(new_num), Rc::new(new_den)));
            }
        }
        None
    }
);

rule!(
    AddFractionRule,
    "add_fraction",
    45,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Add(u, v) = expr {
            // Case 1: a/b + c/d
            if let (Expr::Div(n1, d1), Expr::Div(n2, d2)) = (&**u, &**v) {
                // Check for common denominator
                if d1 == d2 {
                    return Some(Expr::Div(
                        Rc::new(Expr::Add(n1.clone(), n2.clone())),
                        d1.clone(),
                    ));
                }
                // (n1*d2 + n2*d1) / (d1*d2)
                let new_num = Expr::Add(
                    Rc::new(Expr::Mul(n1.clone(), d2.clone())),
                    Rc::new(Expr::Mul(n2.clone(), d1.clone())),
                );
                let new_den = Expr::Mul(d1.clone(), d2.clone());
                return Some(Expr::Div(Rc::new(new_num), Rc::new(new_den)));
            }

            // Case 2: a + b/c
            if let Expr::Div(n, d) = &**v {
                // (u*d + n) / d, but if u is 1, just use d
                let u_times_d = if matches!(&**u, Expr::Number(x) if (*x - 1.0).abs() < 1e-10) {
                    (**d).clone()
                } else {
                    Expr::Mul(u.clone(), d.clone())
                };
                let new_num = Expr::Add(Rc::new(u_times_d), n.clone());
                return Some(Expr::Div(Rc::new(new_num), d.clone()));
            }

            // Case 3: a/b + c
            if let Expr::Div(n, d) = &**u {
                // (n + v*d) / d, but if v is 1, just use d
                let v_times_d = if matches!(&**v, Expr::Number(x) if (*x - 1.0).abs() < 1e-10) {
                    (**d).clone()
                } else {
                    Expr::Mul(v.clone(), d.clone())
                };
                let new_num = Expr::Add(n.clone(), Rc::new(v_times_d));
                return Some(Expr::Div(Rc::new(new_num), d.clone()));
            }
        }
        None
    }
);

rule_with_helpers!(FractionToEndRule, "fraction_to_end", 50, Algebraic, &[ExprKind::Div, ExprKind::Mul],
    helpers: {
        // Helper to check if expression contains any Div inside Mul
        fn mul_contains_div(e: &Expr) -> bool {
            match e {
                Expr::Div(_, _) => true,
                Expr::Mul(a, b) => mul_contains_div(a) || mul_contains_div(b),
                _ => false,
            }
        }

        // Helper to extract all factors from a multiplication, separating numerators and denominators
        fn extract_factors(e: &Expr, numerators: &mut Vec<Expr>, denominators: &mut Vec<Expr>) {
            match e {
                Expr::Mul(a, b) => {
                    extract_factors(a, numerators, denominators);
                    extract_factors(b, numerators, denominators);
                }
                Expr::Div(num, den) => {
                    extract_factors(num, numerators, denominators);
                    denominators.push((**den).clone());
                }
                other => {
                    numerators.push(other.clone());
                }
            }
        }
    },
    |expr: &Expr, _context: &RuleContext| {
        // Case 1: Div where numerator is a Mul containing Divs
        // e.g., ((1/a) * b * (1/c)) / d -> b / (a * c * d)
        if let Expr::Div(num, den) = expr
            && mul_contains_div(num) {
                let mut numerators = Vec::new();
                let mut denominators = Vec::new();
                extract_factors(num, &mut numerators, &mut denominators);

                // Add the outer denominator
                denominators.push((**den).clone());

                // Filter out 1s from numerators (they're identity elements)
                let filtered_nums: Vec<Expr> = numerators
                    .into_iter()
                    .filter(|e| !matches!(e, Expr::Number(n) if (*n - 1.0).abs() < 1e-10))
                    .collect();

                let num_expr = if filtered_nums.is_empty() {
                    Expr::Number(1.0)
                } else {
                    crate::simplification::helpers::rebuild_mul(filtered_nums)
                };

                let den_expr = crate::simplification::helpers::rebuild_mul(denominators);

                let result = Expr::Div(Rc::new(num_expr), Rc::new(den_expr));

                if result != *expr {
                    return Some(result);
                }
            }

        // Case 2: Mul containing at least one Div
        if let Expr::Mul(_, _) = expr {
            if !mul_contains_div(expr) {
                return None;
            }

            let mut numerators = Vec::new();
            let mut denominators = Vec::new();
            extract_factors(expr, &mut numerators, &mut denominators);

            // Only transform if we have denominators
            if denominators.is_empty() {
                return None;
            }

            // Filter out 1s from numerators (they're identity elements)
            let filtered_nums: Vec<Expr> = numerators
                .into_iter()
                .filter(|e| !matches!(e, Expr::Number(n) if (*n - 1.0).abs() < 1e-10))
                .collect();

            // Build the result: (num1 * num2 * ...) / (den1 * den2 * ...)
            let num_expr = if filtered_nums.is_empty() {
                Expr::Number(1.0)
            } else {
                crate::simplification::helpers::rebuild_mul(filtered_nums)
            };

            let den_expr = crate::simplification::helpers::rebuild_mul(denominators);

            let result = Expr::Div(Rc::new(num_expr), Rc::new(den_expr));

            // Only return if we actually changed something
            if result != *expr {
                return Some(result);
            }
        }

        None
    }
);
