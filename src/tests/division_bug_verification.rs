#[cfg(test)]
mod division_bug_tests {
    use crate::{Expr, simplification::simplify_expr};
    use std::collections::HashSet;
    use std::rc::Rc;
    #[test]
    fn verify_display_parens() {
        // Test 1: A / (C * R^2) - should have parentheses
        let expr = Expr::Div(
            Rc::new(Expr::Symbol("A".to_string())),
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("C".to_string())),
                Rc::new(Expr::Pow(
                    Rc::new(Expr::Symbol("R".to_string())),
                    Rc::new(Expr::Number(2.0)),
                )),
            )),
        );
        let display = format!("{}", expr);
        println!("Display: {}", display);
        assert_eq!(
            display, "A/(C*R^2)",
            "Display should be 'A/(C*R^2)' not '{}'",
            display
        );
    }

    #[test]
    fn verify_simplification_cancellation() {
        // Test 2: (-C * R * V0) / (C * R^2) should simplify to -V0 / R
        let expr = Expr::Div(
            Rc::new(Expr::Mul(
                Rc::new(Expr::Mul(
                    Rc::new(Expr::Mul(
                        Rc::new(Expr::Number(-1.0)),
                        Rc::new(Expr::Symbol("C".to_string())),
                    )),
                    Rc::new(Expr::Symbol("R".to_string())),
                )),
                Rc::new(Expr::Symbol("V0".to_string())),
            )),
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("C".to_string())),
                Rc::new(Expr::Pow(
                    Rc::new(Expr::Symbol("R".to_string())),
                    Rc::new(Expr::Number(2.0)),
                )),
            )),
        );

        println!("Original:   {}", expr);
        let simplified = simplify_expr(expr, HashSet::new());
        println!("Simplified: {}", simplified);

        // Expected: -V0/R
        let display = format!("{}", simplified);
        assert_eq!(
            display, "-V0/R",
            "Simplification should be '-V0/R' not '{}'",
            display
        );
    }
}
