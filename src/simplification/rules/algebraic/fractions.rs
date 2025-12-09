use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};

rule!(DivSelfRule, "div_self", 78, Algebraic, &[ExprKind::Div], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let AstKind::Div(u, v) = &expr.kind
        && u == v
    {
        return Some(Expr::number(1.0));
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
        if let AstKind::Div(num, den) = &expr.kind {
            // Case 1: (a/b)/(c/d) -> (a*d)/(b*c)
            if let (AstKind::Div(a, b), AstKind::Div(c, d)) = (&num.kind, &den.kind) {
                return Some(Expr::div_expr(
                    Expr::mul_expr((**a).clone(), (**d).clone()),
                    Expr::mul_expr((**b).clone(), (**c).clone()),
                ));
            }
            // Case 2: x/(c/d) -> (x*d)/c
            if let AstKind::Div(c, d) = &den.kind {
                return Some(Expr::div_expr(
                    Expr::mul_expr((**num).clone(), (**d).clone()),
                    (**c).clone(),
                ));
            }
            // Case 3: (a/b)/y -> a/(b*y)
            if let AstKind::Div(a, b) = &num.kind {
                return Some(Expr::div_expr(
                    (**a).clone(),
                    Expr::mul_expr((**b).clone(), (**den).clone()),
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
        if let AstKind::Div(num, outer_den) = &expr.kind {
            // Case 1: (a + b/c) / d -> (a*c + b) / (c*d)
            if let AstKind::Add(a, v) = &num.kind
                && let AstKind::Div(b, c) = &v.kind
            {
                // (a*c + b) / (c*d)
                let new_num =
                    Expr::add_expr(Expr::mul_expr((**a).clone(), (**c).clone()), (**b).clone());
                let new_den = Expr::mul_expr((**c).clone(), (**outer_den).clone());
                return Some(Expr::div_expr(new_num, new_den));
            }
            // Case 2: (b/c + a) / d -> (b + a*c) / (c*d)
            if let AstKind::Add(u, a) = &num.kind
                && let AstKind::Div(b, c) = &u.kind
            {
                let new_num =
                    Expr::add_expr((**b).clone(), Expr::mul_expr((**a).clone(), (**c).clone()));
                let new_den = Expr::mul_expr((**c).clone(), (**outer_den).clone());
                return Some(Expr::div_expr(new_num, new_den));
            }
            // Case 3: (a - b/c) / d -> (a*c - b) / (c*d)
            if let AstKind::Sub(a, v) = &num.kind
                && let AstKind::Div(b, c) = &v.kind
            {
                let new_num =
                    Expr::sub_expr(Expr::mul_expr((**a).clone(), (**c).clone()), (**b).clone());
                let new_den = Expr::mul_expr((**c).clone(), (**outer_den).clone());
                return Some(Expr::div_expr(new_num, new_den));
            }
            // Case 4: (b/c - a) / d -> (b - a*c) / (c*d)
            if let AstKind::Sub(u, a) = &num.kind
                && let AstKind::Div(b, c) = &u.kind
            {
                let new_num =
                    Expr::sub_expr((**b).clone(), Expr::mul_expr((**a).clone(), (**c).clone()));
                let new_den = Expr::mul_expr((**c).clone(), (**outer_den).clone());
                return Some(Expr::div_expr(new_num, new_den));
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
        if let AstKind::Add(u, v) = &expr.kind {
            // Case 1: a/b + c/d
            if let (AstKind::Div(n1, d1), AstKind::Div(n2, d2)) = (&u.kind, &v.kind) {
                // Check for common denominator
                if d1 == d2 {
                    return Some(Expr::div_expr(
                        Expr::add_expr((**n1).clone(), (**n2).clone()),
                        (**d1).clone(),
                    ));
                }
                // (n1*d2 + n2*d1) / (d1*d2)
                let new_num = Expr::add_expr(
                    Expr::mul_expr((**n1).clone(), (**d2).clone()),
                    Expr::mul_expr((**n2).clone(), (**d1).clone()),
                );
                let new_den = Expr::mul_expr((**d1).clone(), (**d2).clone());
                return Some(Expr::div_expr(new_num, new_den));
            }

            // Case 2: a + b/c
            if let AstKind::Div(n, d) = &v.kind {
                // (u*d + n) / d, but if u is 1, just use d
                let u_times_d = if matches!(&u.kind, AstKind::Number(x) if (*x - 1.0).abs() < 1e-10)
                {
                    (**d).clone()
                } else {
                    Expr::mul_expr((**u).clone(), (**d).clone())
                };
                let new_num = Expr::add_expr(u_times_d, (**n).clone());
                return Some(Expr::div_expr(new_num, (**d).clone()));
            }

            // Case 3: a/b + c
            if let AstKind::Div(n, d) = &u.kind {
                // (n + v*d) / d, but if v is 1, just use d
                let v_times_d = if matches!(&v.kind, AstKind::Number(x) if (*x - 1.0).abs() < 1e-10)
                {
                    (**d).clone()
                } else {
                    Expr::mul_expr((**v).clone(), (**d).clone())
                };
                let new_num = Expr::add_expr((**n).clone(), v_times_d);
                return Some(Expr::div_expr(new_num, (**d).clone()));
            }
        }
        None
    }
);

rule_with_helpers!(FractionToEndRule, "fraction_to_end", 50, Algebraic, &[ExprKind::Div, ExprKind::Mul],
    helpers: {
        // Helper to check if expression contains any Div inside Mul
        fn mul_contains_div(e: &Expr) -> bool {
            match &e.kind {
                AstKind::Div(_, _) => true,
                AstKind::Mul(a, b) => mul_contains_div(a) || mul_contains_div(b),
                _ => false,
            }
        }

        // Helper to extract all factors from a multiplication, separating numerators and denominators
        fn extract_factors(e: &Expr, numerators: &mut Vec<Expr>, denominators: &mut Vec<Expr>) {
            match &e.kind {
                AstKind::Mul(a, b) => {
                    extract_factors(a, numerators, denominators);
                    extract_factors(b, numerators, denominators);
                }
                AstKind::Div(num, den) => {
                    extract_factors(num, numerators, denominators);
                    denominators.push((**den).clone());
                }
                _ => {
                    numerators.push(e.clone());
                }
            }
        }
    },
    |expr: &Expr, _context: &RuleContext| {
        // Case 1: Div where numerator is a Mul containing Divs
        // e.g., ((1/a) * b * (1/c)) / d -> b / (a * c * d)
        if let AstKind::Div(num, den) = &expr.kind
            && mul_contains_div(num) {
                let mut numerators = Vec::new();
                let mut denominators = Vec::new();
                extract_factors(num, &mut numerators, &mut denominators);

                // Add the outer denominator
                denominators.push((**den).clone());

                // Filter out 1s from numerators (they're identity elements)
                let filtered_nums: Vec<Expr> = numerators
                    .into_iter()
                    .filter(|e| !matches!(e.kind, AstKind::Number(n) if (n - 1.0).abs() < 1e-10))
                    .collect();

                let num_expr = if filtered_nums.is_empty() {
                    Expr::number(1.0)
                } else {
                    crate::simplification::helpers::rebuild_mul(filtered_nums)
                };

                let den_expr = crate::simplification::helpers::rebuild_mul(denominators);

                let result = Expr::div_expr(num_expr, den_expr);

                if result != *expr {
                    return Some(result);
                }
            }

        // Case 2: Mul containing at least one Div
        if let AstKind::Mul(_, _) = &expr.kind {
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
                .filter(|e| !matches!(e.kind, AstKind::Number(n) if (n - 1.0).abs() < 1e-10))
                .collect();

            // Build the result: (num1 * num2 * ...) / (den1 * den2 * ...)
            let num_expr = if filtered_nums.is_empty() {
                Expr::number(1.0)
            } else {
                crate::simplification::helpers::rebuild_mul(filtered_nums)
            };

            let den_expr = crate::simplification::helpers::rebuild_mul(denominators);

            let result = Expr::div_expr(num_expr, den_expr);

            // Only return if we actually changed something
            if result != *expr {
                return Some(result);
            }
        }

        None
    }
);
