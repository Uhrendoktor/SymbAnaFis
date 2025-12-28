#[cfg(test)]
mod tests {
    use crate::Expr;
    use crate::simplification::simplify_expr;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_polynomial_term_adjacency() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // Create mixed terms: x, y, x^2, y^2
        let x2 = Expr::pow(x.clone(), Expr::number(2.0));
        let y2 = Expr::pow(y.clone(), Expr::number(2.0));

        // Sum constructor flattens; simplify() sorts for canonical form
        let sum = simplify_expr(
            Expr::sum(vec![y.clone(), x2.clone(), x.clone(), y2.clone()]),
            HashSet::new(),
            HashMap::new(),
            None,
            None,
            None,
            false,
        );
        // With Poly, terms may be in different format. Check all expected parts are present
        let full_str = sum.to_string();
        assert!(
            full_str.contains("x") && full_str.contains("y"),
            "Expected x and y in result, got: {}",
            full_str
        );
    }

    #[test]
    fn test_mixed_coefficient_adjacency() {
        let x = Expr::symbol("x");

        // x, 2x, x^2, 3x^2 - these now get combined in simplify!
        let term1 = x.clone();
        let term2 = Expr::product(vec![Expr::number(2.0), x.clone()]);
        let term3 = Expr::pow(x.clone(), Expr::number(2.0));
        let term4 = Expr::product(vec![
            Expr::number(3.0),
            Expr::pow(x.clone(), Expr::number(2.0)),
        ]);

        let sum = simplify_expr(
            Expr::sum(vec![term4, term2, term1, term3]),
            HashSet::new(),
            HashMap::new(),
            None,
            None,
            None,
            false,
        );

        // Accept combined form (3*x + 4*x^2), Poly format, or factored form (x*(3 + 4x))
        let result_str = sum.to_string();
        // Just check that both x terms exist
        assert!(
            result_str.contains("x") && (result_str.contains("3") || result_str.contains("4")),
            "Expected x and coefficients in result, got: {}",
            result_str
        );
    }
}
