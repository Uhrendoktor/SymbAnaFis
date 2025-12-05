use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

rule!(
    CanonicalizeRule,
    "canonicalize",
    15,
    Algebraic,
    &[ExprKind::Add, ExprKind::Mul, ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        match expr {
            Expr::Add(_, _) => {
                let terms = crate::simplification::helpers::flatten_add(expr);
                if terms.len() > 1 {
                    Some(crate::simplification::helpers::rebuild_add(terms))
                } else {
                    None
                }
            }
            Expr::Mul(_, _) => {
                let factors = crate::simplification::helpers::flatten_mul(expr);
                if factors.len() > 1 {
                    Some(crate::simplification::helpers::rebuild_mul(factors))
                } else {
                    None
                }
            }
            Expr::Sub(a, b) => {
                // Convert a - b to a + (-b)
                Some(Expr::Add(
                    a.clone(),
                    Rc::new(Expr::Mul(Rc::new(Expr::Number(-1.0)), b.clone())),
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
        if let Expr::Sub(a, b) = expr {
            Some(Expr::Add(
                a.clone(),
                Rc::new(Expr::Mul(Rc::new(Expr::Number(-1.0)), b.clone())),
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
        if let Expr::Add(a, b) = expr {
            // Check if first term (a) is -1 * something: (-x) + y -> y - x
            if let Expr::Mul(coeff, inner) = &**a
                && let Expr::Number(n) = &**coeff
                && (*n + 1.0).abs() < 1e-10
            {
                // Convert Add(Mul(-1, inner), b) to Sub(b, inner)
                return Some(Expr::Sub(b.clone(), inner.clone()));
            }
            // Check if second term (b) is -1 * something: x + (-y) -> x - y
            if let Expr::Mul(coeff, inner) = &**b
                && let Expr::Number(n) = &**coeff
                && (*n + 1.0).abs() < 1e-10
            {
                // Convert Add(a, Mul(-1, inner)) to Sub(a, inner)
                return Some(Expr::Sub(a.clone(), inner.clone()));
            }
            // Check if first term is a negative number: (-n) + x -> x - n
            if let Expr::Number(n) = &**a
                && *n < 0.0
            {
                return Some(Expr::Sub(b.clone(), Rc::new(Expr::Number(-*n))));
            }
            // Check if second term is a negative number: x + (-n) -> x - n
            if let Expr::Number(n) = &**b
                && *n < 0.0
            {
                return Some(Expr::Sub(a.clone(), Rc::new(Expr::Number(-*n))));
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
        if let Expr::Mul(a, b) = expr {
            // Case: (-1) * 1 = -1
            if let (Expr::Number(n1), Expr::Number(n2)) = (&**a, &**b) {
                if (*n1 + 1.0).abs() < 1e-10 && (*n2 - 1.0).abs() < 1e-10 {
                    return Some(Expr::Number(-1.0));
                }
                if (*n2 + 1.0).abs() < 1e-10 && (*n1 - 1.0).abs() < 1e-10 {
                    return Some(Expr::Number(-1.0));
                }
            }
            // Case: (-1) * (-1) = 1
            if let (Expr::Number(n1), Expr::Number(n2)) = (&**a, &**b)
                && (*n1 + 1.0).abs() < 1e-10
                && (*n2 + 1.0).abs() < 1e-10
            {
                return Some(Expr::Number(1.0));
            }
        }
        None
    }
);
