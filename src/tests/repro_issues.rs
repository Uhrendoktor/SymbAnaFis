#[cfg(test)]
mod tests {
    use crate::Expr;
    use crate::simplify;
    use std::rc::Rc;
    #[test]
    fn test_fraction_cancellation_product_base() {
        // (C * R) / (C * R)^2 -> 1 / (C * R)
        let expr = "C * R / (C * R)^2";
        let simplified = simplify(expr.to_string(), None, None).unwrap();
        println!("{} -> {}", expr, simplified);
        assert_eq!(simplified.to_string(), "1/(C*R)");
    }

    #[test]
    fn test_sqrt_div_self() {
        // sqrt(x) / x -> 1 / sqrt(x)
        let expr = "sqrt(x) / x";
        let simplified = simplify(expr.to_string(), None, None).unwrap();
        println!("{} -> {}", expr, simplified);
        assert_eq!(simplified.to_string(), "1/sqrt(x)");
    }

    #[test]
    fn test_numeric_fraction_simplification() {
        // 2 * x / 4 -> x / 2
        let expr = "2 * x / 4";
        let simplified = simplify(expr.to_string(), None, None).unwrap();
        println!("{} -> {}", expr, simplified);
        assert_eq!(simplified.to_string(), "x/2");
    }

    #[test]
    fn test_sqrt_product_div_product() {
        let a = Expr::Symbol("a".to_string());
        let b = Expr::Symbol("b".to_string());
        let expr = Expr::Div(
            Rc::new(Expr::FunctionCall {
                name: "sqrt".to_string(),
                args: vec![Expr::Mul(Rc::new(a.clone()), Rc::new(b.clone()))],
            }),
            Rc::new(Expr::Mul(Rc::new(a.clone()), Rc::new(b.clone()))),
        );

        let simplified =
            crate::simplification::simplify_expr(expr, std::collections::HashSet::new());
        let result = format!("{}", simplified);
        println!("sqrt(a * b) / (a * b) -> {}", result);

        assert!(
            result.contains("1/sqrt(a*b)") || result.contains("1/(sqrt(a)*sqrt(b))"),
            "Expected 1/sqrt(a*b) or equivalent, got {}",
            result
        );
    }

    #[test]
    fn test_heat_flux_simplification() {
        // sqrt(alpha) * sqrt(t) / (alpha * t * sqrt(pi))
        let alpha = Expr::Symbol("alpha".to_string());
        let t = Expr::Symbol("t".to_string());
        let pi = Expr::Symbol("pi".to_string());

        let num = Expr::Mul(
            Rc::new(Expr::FunctionCall {
                name: "sqrt".to_string(),
                args: vec![alpha.clone()],
            }),
            Rc::new(Expr::FunctionCall {
                name: "sqrt".to_string(),
                args: vec![t.clone()],
            }),
        );
        let den = Expr::Mul(
            Rc::new(alpha.clone()),
            Rc::new(Expr::Mul(
                Rc::new(t.clone()),
                Rc::new(Expr::FunctionCall {
                    name: "sqrt".to_string(),
                    args: vec![pi.clone()],
                }),
            )),
        );
        let expr = Expr::Div(Rc::new(num), Rc::new(den));

        let simplified =
            crate::simplification::simplify_expr(expr, std::collections::HashSet::new());
        let result = format!("{}", simplified);
        println!("Heat flux term -> {}", result);

        // Should simplify to 1 / (sqrt(alpha) * sqrt(t) * sqrt(pi))
        // or 1 / sqrt(alpha * t * pi)
        assert!(
            !result.contains("sqrt(alpha) /"),
            "Should not contain sqrt(alpha) in numerator, got {}",
            result
        );
        assert!(
            !result.contains("sqrt(t) /"),
            "Should not contain sqrt(t) in numerator, got {}",
            result
        );
    }

    #[test]
    fn test_power_div_cancellation() {
        // (x - 1)^2 / (x - 1) should simplify to (x - 1)
        let expr = "(x - 1)^2 / (x - 1)";
        let result = simplify(expr.to_string(), None, None).unwrap();
        println!("(x-1)^2 / (x-1) = {}", result);
        // Should be x - 1
        assert_eq!(result.to_string(), "x - 1");
    }

    #[test]
    fn test_full_fraction_cancellation() {
        // (x - 1)^2 * (x + 1) / (x^2 - 1) should simplify to (x - 1)
        let expr = "(x - 1)^2 * (x + 1) / (x^2 - 1)";
        let result = simplify(expr.to_string(), None, None).unwrap();
        println!("(x-1)^2 * (x+1) / (x^2-1) = {}", result);
        // Should be x - 1
        assert_eq!(result.to_string(), "x - 1");
    }
}
