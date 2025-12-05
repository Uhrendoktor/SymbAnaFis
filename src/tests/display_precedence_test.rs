#[cfg(test)]
mod tests {
    use crate::Expr;
    use std::collections::HashSet;
    use std::rc::Rc;
    #[test]
    fn test_division_precedence_mul() {
        // a / (b * c) should be displayed as "a / (b * c)"
        // If displayed as "a / b * c", it means (a / b) * c which is wrong.
        let expr = Expr::Div(
            Rc::new(Expr::Symbol("a".to_string())),
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("b".to_string())),
                Rc::new(Expr::Symbol("c".to_string())),
            )),
        );
        let display = format!("{}", expr);
        println!("Display: {}", display);
        assert_eq!(display, "a/(b*c)");
    }

    #[test]
    fn test_division_precedence_div() {
        // a / (b / c) should be displayed as "a / (b / c)"
        let expr = Expr::Div(
            Rc::new(Expr::Symbol("a".to_string())),
            Rc::new(Expr::Div(
                Rc::new(Expr::Symbol("b".to_string())),
                Rc::new(Expr::Symbol("c".to_string())),
            )),
        );
        let display = format!("{}", expr);
        println!("Display: {}", display);
        assert_eq!(display, "a/(b/c)");
    }

    #[test]
    fn test_rc_circuit_derivative() {
        use crate::simplification::simplify_expr;
        // V0 * exp(-t / (R * C))
        // Derivative should be: -V0 * exp(-t / (R * C)) / (R * C)
        // Or with the bug: ... / C * R^2 ??

        // Let's construct the expression that might be causing issues
        // Div(Mul(-C*R*V0, exp), Mul(C, R^2))
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

        println!("Original: {}", expr);
        let simplified = simplify_expr(expr.clone(), HashSet::new());
        println!("Simplified: {}", simplified);

        // Check if simplified result has correct structure
        // Should be -V0 / R (ignoring exp for now as it's just a factor)
        // Actually the test case above doesn't include exp, let's add it to be precise

        let exp_term = Expr::FunctionCall {
            name: "exp".to_string(),
            args: vec![Expr::Symbol("t".to_string())], // Simplified arg
        };

        let expr_full = Expr::Div(
            Rc::new(Expr::Mul(
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
                Rc::new(exp_term.clone()),
            )),
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("C".to_string())),
                Rc::new(Expr::Pow(
                    Rc::new(Expr::Symbol("R".to_string())),
                    Rc::new(Expr::Number(2.0)),
                )),
            )),
        );

        println!("Full Original: {}", expr_full);
        let simplified_full = simplify_expr(expr_full, HashSet::new());
        println!("Full Simplified: {}", simplified_full);

        // Expected display: -V0 * exp(t) / R
        // Bad display: -V0 * exp(t) / C * R^2 (or similar)
        let display = format!("{}", simplified_full);
        assert!(
            !display.contains("/ C *"),
            "Display contains unparenthesized denominator multiplication: {}",
            display
        );
    }

    #[test]
    fn test_display_rc_denominator() {
        // A / (C * R^2)
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
        assert_eq!(display, "A/(C*R^2)");
    }
}
