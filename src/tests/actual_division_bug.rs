#[cfg(test)]

mod actual_bug_test {
    use crate::{Expr, simplification::simplify_expr};
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_rc_derivative_actual_bug() {
        // The actual expression from the user:
        // -C * R * V0 * exp(-t / (C * R)) / (C * R^2)
        // Should simplify to: -V0 * exp(-t / (C * R)) / R

        let exp_term = Expr::FunctionCall {
            name: "exp".to_string(),
            args: vec![Expr::Div(
                Rc::new(Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("t".to_string())),
                )),
                Rc::new(Expr::Mul(
                    Rc::new(Expr::Symbol("C".to_string())),
                    Rc::new(Expr::Symbol("R".to_string())),
                )),
            )],
        };

        // Numerator: -C * R * V0 * exp(...)
        // Which is: -1 * C * R * V0 * exp(...)
        let numerator = Expr::Mul(
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
            Rc::new(exp_term),
        );

        // Denominator: C * R^2
        let denominator = Expr::Mul(
            Rc::new(Expr::Symbol("C".to_string())),
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("R".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
        );

        let expr = Expr::Div(Rc::new(numerator), Rc::new(denominator));

        println!("\nOriginal expression:");
        println!("{}", expr);

        let simplified = simplify_expr(expr, HashSet::new());

        println!("\nSimplified expression:");
        println!("{}", simplified);

        println!("\nExpected:");
        println!("-V0*exp(-t/(C*R))/R");
        // The simplified form should have cancelled C and reduced R^2 to R
        // Check that the denominator is just R, not (C * R^2) or similar
        let display = format!("{}", simplified);
        assert!(
            display.ends_with("/R"),
            "Expression should end with '/R', got: {}",
            display
        );
    }

    #[test]
    fn test_simple_constant_division() {
        // Even simpler: (-1 * C * R) / (C * R^2)
        // Should simplify to: -1 / R

        let numerator = Expr::Mul(
            Rc::new(Expr::Mul(
                Rc::new(Expr::Number(-1.0)),
                Rc::new(Expr::Symbol("C".to_string())),
            )),
            Rc::new(Expr::Symbol("R".to_string())),
        );

        let denominator = Expr::Mul(
            Rc::new(Expr::Symbol("C".to_string())),
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("R".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
        );

        let expr = Expr::Div(Rc::new(numerator), Rc::new(denominator));

        println!("\nSimple test:");
        println!("Original:   {}", expr);
        let simplified = simplify_expr(expr, HashSet::new());
        println!("Simplified: {}", simplified);
        println!("Expected:   -1/R");

        let display = format!("{}", simplified);
        assert_eq!(display, "-1/R", "Expected '-1/R', got '{}'", display);
    }
}
