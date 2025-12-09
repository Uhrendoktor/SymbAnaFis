use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};

rule!(
    MulDivCombinationRule,
    "mul_div_combination",
    85,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(a, b) = &expr.kind {
            // Case 1: a * (b / c) -> (a * b) / c
            if let AstKind::Div(num, den) = &b.kind {
                return Some(Expr::div_expr(
                    Expr::mul_expr(a.as_ref().clone(), num.as_ref().clone()),
                    den.as_ref().clone(),
                ));
            }
            // Case 2: (a/b) * c -> (a*c)/b
            if let AstKind::Div(num, den) = &a.kind {
                return Some(Expr::div_expr(
                    Expr::mul_expr(num.as_ref().clone(), b.as_ref().clone()),
                    den.as_ref().clone(),
                ));
            }
        }
        None
    }
);

rule!(
    CombineTermsRule,
    "combine_terms",
    50,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        // Only handle Add - don't touch Sub (preserve subtraction form)
        let terms = match &expr.kind {
            AstKind::Add(_, _) => crate::simplification::helpers::flatten_add(expr),
            _ => return None,
        };

        if terms.len() < 2 {
            return None;
        }

        // Group terms by their base to find combinable terms
        let mut term_groups: std::collections::HashMap<Expr, f64> =
            std::collections::HashMap::new();
        for term in &terms {
            let (coeff, base) = crate::simplification::helpers::extract_coeff(term);
            *term_groups.entry(base).or_insert(0.0) += coeff;
        }

        // If no terms were actually combined, don't change anything
        if term_groups.len() == terms.len() {
            return None;
        }

        // Build result from combined terms
        let mut result = Vec::new();
        for (base, coeff) in term_groups {
            if coeff == 0.0 {
                // Drop zero terms
            } else if coeff == 1.0 {
                result.push(base);
            } else if let AstKind::Number(n) = &base.kind {
                if *n == 1.0 {
                    result.push(Expr::number(coeff));
                } else {
                    result.push(Expr::mul_expr(Expr::number(coeff), base));
                }
            } else {
                result.push(Expr::mul_expr(Expr::number(coeff), base));
            }
        }

        if result.is_empty() {
            return Some(Expr::number(0.0));
        }

        // Sort terms for canonical ordering
        result.sort_by(crate::simplification::helpers::compare_expr);

        let new_expr = crate::simplification::helpers::rebuild_add(result);
        if new_expr.id != expr.id && new_expr != *expr {
            return Some(new_expr);
        }
        None
    }
);

rule!(
    CombineFactorsRule,
    "combine_factors",
    58,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        let factors = crate::simplification::helpers::flatten_mul(expr);

        if factors.len() < 2 {
            return None;
        }

        // Group factors by base and combine exponents
        let mut factor_groups: std::collections::HashMap<Expr, Vec<Expr>> =
            std::collections::HashMap::new();

        for factor in &factors {
            match &factor.kind {
                AstKind::Pow(base, exp) => {
                    factor_groups
                        .entry(base.as_ref().clone())
                        .or_default()
                        .push(exp.as_ref().clone());
                }
                _ => {
                    factor_groups
                        .entry(factor.clone())
                        .or_default()
                        .push(Expr::number(1.0));
                }
            }
        }

        // Combine exponents for each base
        let mut combined_factors = Vec::new();
        for (base, exponents) in factor_groups {
            if exponents.len() == 1 {
                if exponents[0] == Expr::number(1.0) {
                    combined_factors.push(base);
                } else {
                    combined_factors.push(Expr::pow(base, exponents[0].clone()));
                }
            } else {
                // Sum all exponents
                let mut total_exp = exponents[0].clone();
                for exp in &exponents[1..] {
                    total_exp = Expr::add_expr(total_exp, exp.clone());
                }
                combined_factors.push(Expr::pow(base, total_exp));
            }
        }

        if combined_factors.len() != factors.len() {
            Some(crate::simplification::helpers::rebuild_mul(
                combined_factors,
            ))
        } else {
            None
        }
    }
);

// Helper: Extract factor and addends from a * (b + c + ...) or (b + c + ...) * a
fn extract_product_with_sum(expr: &Expr) -> Option<(Expr, Vec<Expr>)> {
    if let AstKind::Mul(a, b) = &expr.kind {
        // Check if b is an Add
        if matches!(b.kind, AstKind::Add(_, _)) {
            let addends = crate::simplification::helpers::flatten_add(b);
            return Some((a.as_ref().clone(), addends));
        }
        // Check if a is an Add
        if matches!(a.kind, AstKind::Add(_, _)) {
            let addends = crate::simplification::helpers::flatten_add(a);
            return Some((b.as_ref().clone(), addends));
        }
        // Check nested: (a * b) * (c + d) or a * (b * (c + d))
        if let Some((inner_factor, addends)) = extract_product_with_sum(b) {
            let combined_factor = Expr::mul_expr(a.as_ref().clone(), inner_factor);
            return Some((combined_factor, addends));
        }
        if let Some((inner_factor, addends)) = extract_product_with_sum(a) {
            let combined_factor = Expr::mul_expr(inner_factor, b.as_ref().clone());
            return Some((combined_factor, addends));
        }
    }
    None
}

// Helper: Check if expression contains a variable (not just numbers)
// Helper: Check if expression contains a variable (not just numbers)
fn contains_variable(expr: &Expr) -> bool {
    match &expr.kind {
        AstKind::Symbol(_) => true,
        AstKind::Number(_) => false,
        AstKind::Mul(a, b)
        | AstKind::Add(a, b)
        | AstKind::Sub(a, b)
        | AstKind::Div(a, b)
        | AstKind::Pow(a, b) => contains_variable(a) || contains_variable(b),
        AstKind::FunctionCall { args, .. } => args.iter().any(contains_variable),
        AstKind::Derivative { inner, .. } => contains_variable(inner),
    }
}

// Helper: Extract base and numeric exponent from an expression
// x -> (x, 1.0), x^n -> (x, n)
fn extract_base_and_exp(expr: &Expr) -> Option<(Expr, f64)> {
    match &expr.kind {
        AstKind::Symbol(_) => Some((expr.clone(), 1.0)),
        AstKind::Pow(base, exp) => {
            if let AstKind::Number(n) = &exp.kind {
                Some((base.as_ref().clone(), *n))
            } else {
                None // Non-numeric exponent, can't combine
            }
        }
        _ => None,
    }
}

// Helper: Combine variable expressions, handling x*x*x...->x^n
fn combine_var_parts(a: Expr, b: Expr) -> Expr {
    // Try to extract base and exponent from both
    if let (Some((base_a, exp_a)), Some((base_b, exp_b))) =
        (extract_base_and_exp(&a), extract_base_and_exp(&b))
    {
        // If same base, combine exponents
        if base_a == base_b {
            let total_exp = exp_a + exp_b;
            if (total_exp - 1.0).abs() < 1e-10 {
                return base_a;
            }
            return Expr::pow(base_a, Expr::number(total_exp));
        }
    }

    // Default: just multiply
    Expr::mul_expr(a, b)
}

// Helper: Distribute a factor over addends and build canonical terms
fn distribute_factor(factor: &Expr, addends: &[Expr]) -> Vec<Expr> {
    addends
        .iter()
        .map(|addend| {
            // Extract coefficients from both factor and addend
            let (factor_coeff, factor_var) = crate::simplification::helpers::extract_coeff(factor);
            let (addend_coeff, addend_var) = crate::simplification::helpers::extract_coeff(addend);

            let total_coeff = factor_coeff * addend_coeff;

            // Combine variable parts, converting x*x to x^2 etc.
            let combined_var = if matches!(factor_var.kind, AstKind::Number(n) if n == 1.0) {
                addend_var
            } else if matches!(addend_var.kind, AstKind::Number(n) if n == 1.0) {
                factor_var
            } else {
                combine_var_parts(factor_var, addend_var)
            };

            // Build canonical result (coefficient first)
            if (total_coeff - 1.0).abs() < 1e-10 {
                combined_var
            } else {
                Expr::mul_expr(Expr::number(total_coeff), combined_var)
            }
        })
        .collect()
}

rule!(
    CombineLikeTermsInAdditionRule,
    "combine_like_terms_in_addition",
    52,
    Algebraic,
    &[ExprKind::Add, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        let terms = crate::simplification::helpers::flatten_add(expr);

        if terms.len() < 2 {
            return None;
        }

        // First pass: collect signatures of non-product-with-sum terms
        let mut existing_signatures: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for term in &terms {
            if extract_product_with_sum(term).is_none() {
                let (_, base) = crate::simplification::helpers::extract_coeff(term);
                existing_signatures
                    .insert(crate::simplification::helpers::get_term_signature(&base));
            }
        }

        // Second pass: expand products that would create combinable terms
        let mut expanded_terms: Vec<Expr> = Vec::new();
        let mut did_expand = false;

        for term in &terms {
            if let Some((factor, addends)) = extract_product_with_sum(term) {
                // Only expand if factor contains a variable
                if contains_variable(&factor) {
                    let distributed = distribute_factor(&factor, &addends);

                    // Check if any distributed term would combine with existing terms
                    let would_combine = distributed.iter().any(|dt| {
                        let (_, base) = crate::simplification::helpers::extract_coeff(dt);
                        let sig = crate::simplification::helpers::get_term_signature(&base);
                        existing_signatures.contains(&sig)
                    });

                    if would_combine {
                        expanded_terms.extend(distributed);
                        did_expand = true;
                        continue;
                    }
                }
            }
            expanded_terms.push(term.clone());
        }

        // If we expanded anything, return the expanded form (will be re-simplified)
        if did_expand {
            return Some(crate::simplification::helpers::rebuild_add(expanded_terms));
        }

        // Original like-term combination logic
        // Group terms by their structure (ignoring coefficients)
        let mut like_terms: std::collections::HashMap<String, Vec<Expr>> =
            std::collections::HashMap::new();

        for term in &terms {
            let key = crate::simplification::helpers::get_term_signature(term);
            like_terms.entry(key).or_default().push(term.clone());
        }

        // Combine like terms
        let mut combined_terms = Vec::new();
        for (_signature, group_terms) in like_terms {
            if group_terms.len() == 1 {
                combined_terms.push(group_terms[0].clone());
            } else {
                // Sum the coefficients of like terms
                let mut total_coeff = 0.0;
                let mut base_term = None;

                for term in group_terms {
                    let (coeff, var_part) = crate::simplification::helpers::extract_coeff(&term);
                    total_coeff += coeff;
                    if base_term.is_none() {
                        base_term = Some(var_part);
                    }
                }

                if (total_coeff - 0.0).abs() < 1e-10 {
                    // Terms cancel out
                    continue;
                }

                let base_term = base_term.unwrap();
                if (total_coeff - 1.0).abs() < 1e-10 {
                    combined_terms.push(base_term);
                } else {
                    combined_terms.push(Expr::mul_expr(Expr::number(total_coeff), base_term));
                }
            }
        }

        if combined_terms.len() != terms.len() {
            Some(crate::simplification::helpers::rebuild_add(combined_terms))
        } else {
            None
        }
    }
);

rule!(
    DistributeNegationRule,
    "distribute_negation",
    50,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(a, b) = &expr.kind {
            // Check if multiplying by -1
            let is_neg_one_a = matches!(a.kind, AstKind::Number(n) if (n + 1.0).abs() < 1e-10);
            let is_neg_one_b = matches!(b.kind, AstKind::Number(n) if (n + 1.0).abs() < 1e-10);

            let (neg_pos, inner) = if is_neg_one_a {
                (true, &**b)
            } else if is_neg_one_b {
                (true, &**a)
            } else {
                (false, expr)
            };

            if neg_pos {
                // -1 * (A - B) -> B - A
                if let AstKind::Sub(x, y) = &inner.kind {
                    return Some(Expr::sub_expr(y.as_ref().clone(), x.as_ref().clone()));
                }
                // -1 * (A + B) -> -A - B = -A + (-B)
                if let AstKind::Add(x, y) = &inner.kind {
                    return Some(Expr::sub_expr(
                        Expr::mul_expr(Expr::number(-1.0), x.as_ref().clone()),
                        y.as_ref().clone(),
                    ));
                }
            }
        }
        None
    }
);

// DistributeMulInNumeratorRule removed - it conflicts with MulDivCombinationRule causing oscillation
// We want (a*b)/c form (compressed) unless there are cancellations possible
