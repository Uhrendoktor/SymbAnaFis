use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use crate::{Expr, ExprKind as AstKind};

rule!(
    CanonicalizeRule,
    "canonicalize",
    15,
    Algebraic,
    &[ExprKind::Add, ExprKind::Mul, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        match &expr.kind {
            AstKind::Add(_, _) => {
                let terms = crate::simplification::helpers::flatten_add(expr);
                if terms.len() > 1 {
                    Some(crate::simplification::helpers::rebuild_add(terms))
                } else {
                    None
                }
            }
            AstKind::Mul(_, _) => {
                let factors = crate::simplification::helpers::flatten_mul(expr);
                if factors.len() > 1 {
                    Some(crate::simplification::helpers::rebuild_mul(factors))
                } else {
                    None
                }
            }
            AstKind::Sub(a, b) => {
                // Convert a - b to a + (-b)
                Some(Expr::add_expr(
                    a.as_ref().clone(),
                    Expr::mul_expr(Expr::number(-1.0), b.as_ref().clone()),
                ))
            }
            _ => None,
        }
    }
);

rule!(
    CanonicalizeMultiplicationRule,
    "canonicalize_multiplication",
    15,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        let factors = crate::simplification::helpers::flatten_mul(expr);

        if factors.len() <= 1 {
            return None;
        }

        // Sort factors for canonical ordering (numbers first, then symbols, etc.)
        let mut sorted_factors = factors.clone();
        sorted_factors.sort_by(crate::simplification::helpers::compare_mul_factors);

        // Check if order changed
        if sorted_factors != factors {
            Some(crate::simplification::helpers::rebuild_mul(sorted_factors))
        } else {
            None
        }
    }
);

rule!(
    CanonicalizeAdditionRule,
    "canonicalize_addition",
    15,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        let terms = crate::simplification::helpers::flatten_add(expr);

        if terms.len() <= 1 {
            return None;
        }

        // Sort terms for canonical ordering
        let mut sorted_terms = terms.clone();
        sorted_terms.sort_by(crate::simplification::helpers::compare_expr);

        // Check if order changed
        if sorted_terms != terms {
            Some(crate::simplification::helpers::rebuild_add(sorted_terms))
        } else {
            None
        }
    }
);

rule!(
    CanonicalizeSubtractionRule,
    "canonicalize_subtraction",
    15,
    Algebraic,
    &[ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        // Convert subtraction to addition with negative
        if let AstKind::Sub(a, b) = &expr.kind {
            Some(Expr::add_expr(
                a.as_ref().clone(),
                Expr::mul_expr(Expr::number(-1.0), b.as_ref().clone()),
            ))
        } else {
            None
        }
    }
);

rule!(
    NormalizeAddNegationRule,
    "normalize_add_negation",
    5,
    Algebraic,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        // Convert additions with negative terms to subtraction form for cleaner display
        if let AstKind::Add(a, b) = &expr.kind {
            // Check if first term (a) is -1 * something: (-x) + y -> y - x
            if let AstKind::Mul(coeff, inner) = &a.kind
                && let AstKind::Number(n) = &coeff.kind
                && (*n + 1.0).abs() < 1e-10
            {
                // Convert Add(Mul(-1, inner), b) to Sub(b, inner)
                return Some(Expr::sub_expr(b.as_ref().clone(), inner.as_ref().clone()));
            }
            // Check if second term (b) is -1 * something: x + (-y) -> x - y
            if let AstKind::Mul(coeff, inner) = &b.kind
                && let AstKind::Number(n) = &coeff.kind
                && (*n + 1.0).abs() < 1e-10
            {
                // Convert Add(a, Mul(-1, inner)) to Sub(a, inner)
                return Some(Expr::sub_expr(a.as_ref().clone(), inner.as_ref().clone()));
            }
            // Check if first term is a negative number: (-n) + x -> x - n
            if let AstKind::Number(n) = &a.kind
                && *n < 0.0
            {
                return Some(Expr::sub_expr(b.as_ref().clone(), Expr::number(-*n)));
            }
            // Check if second term is a negative number: x + (-n) -> x - n
            if let AstKind::Number(n) = &b.kind
                && *n < 0.0
            {
                return Some(Expr::sub_expr(a.as_ref().clone(), Expr::number(-*n)));
            }
        }
        None
    }
);

rule!(
    SimplifyNegativeOneRule,
    "simplify_negative_one",
    80,
    Algebraic,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(a, b) = &expr.kind {
            // Case: (-1) * 1 = -1
            if let (AstKind::Number(n1), AstKind::Number(n2)) = (&a.kind, &b.kind) {
                if (*n1 + 1.0).abs() < 1e-10 && (*n2 - 1.0).abs() < 1e-10 {
                    return Some(Expr::number(-1.0));
                }
                if (*n2 + 1.0).abs() < 1e-10 && (*n1 - 1.0).abs() < 1e-10 {
                    return Some(Expr::number(-1.0));
                }
            }
            // Case: (-1) * (-1) = 1
            if let (AstKind::Number(n1), AstKind::Number(n2)) = (&a.kind, &b.kind)
                && (*n1 + 1.0).abs() < 1e-10
                && (*n2 + 1.0).abs() < 1e-10
            {
                return Some(Expr::number(1.0));
            }
        }
        None
    }
);
