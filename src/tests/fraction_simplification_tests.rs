#[cfg(test)]
mod tests {
    use crate::Expr;
    use crate::simplification::simplify_expr;
    use std::collections::HashSet;
    use std::rc::Rc;
    #[test]
    fn test_nested_fraction_div_div() {
        // (x/y) / (z/a) -> (x*a) / (y*z)
        let expr = Expr::Div(
            Rc::new(Expr::Div(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Symbol("y".to_string())),
            )),
            Rc::new(Expr::Div(
                Rc::new(Expr::Symbol("z".to_string())),
                Rc::new(Expr::Symbol("a".to_string())),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        // Expected: (x*a) / (y*z)
        // Note: ordering of multiplication might vary, so we check structure
        if let Expr::Div(num, den) = simplified {
            // Check numerator: x*a or a*x
            if let Expr::Mul(n1, n2) = num.as_ref() {
                let s1 = format!("{}", n1);
                let s2 = format!("{}", n2);
                assert!((s1 == "x" && s2 == "a") || (s1 == "a" && s2 == "x"));
            } else {
                panic!("Expected numerator to be multiplication");
            }

            // Check denominator: y*z or z*y
            if let Expr::Mul(d1, d2) = den.as_ref() {
                let s1 = format!("{}", d1);
                let s2 = format!("{}", d2);
                assert!((s1 == "y" && s2 == "z") || (s1 == "z" && s2 == "y"));
            } else {
                panic!("Expected denominator to be multiplication");
            }
        } else {
            panic!("Expected division");
        }
    }

    #[test]
    fn test_nested_fraction_val_div() {
        // x / (y/z) -> (x*z) / y
        let expr = Expr::Div(
            Rc::new(Expr::Symbol("x".to_string())),
            Rc::new(Expr::Div(
                Rc::new(Expr::Symbol("y".to_string())),
                Rc::new(Expr::Symbol("z".to_string())),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        // Expected: (x*z) / y
        if let Expr::Div(num, den) = simplified {
            if let Expr::Mul(n1, n2) = num.as_ref() {
                let s1 = format!("{}", n1);
                let s2 = format!("{}", n2);
                assert!((s1 == "x" && s2 == "z") || (s1 == "z" && s2 == "x"));
            } else {
                panic!("Expected numerator to be multiplication");
            }

            if let Expr::Symbol(s) = den.as_ref() {
                assert_eq!(s, "y");
            } else {
                panic!("Expected denominator y");
            }
        } else {
            panic!("Expected division");
        }
    }

    #[test]
    fn test_nested_fraction_div_val() {
        // (x/y) / z -> x / (y*z)
        let expr = Expr::Div(
            Rc::new(Expr::Div(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Symbol("y".to_string())),
            )),
            Rc::new(Expr::Symbol("z".to_string())),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        // Expected: x / (y*z)
        if let Expr::Div(num, den) = simplified {
            if let Expr::Symbol(s) = num.as_ref() {
                assert_eq!(s, "x");
            } else {
                panic!("Expected numerator to be x");
            }

            if let Expr::Mul(d1, d2) = den.as_ref() {
                let s1 = format!("{}", d1);
                let s2 = format!("{}", d2);
                assert!((s1 == "y" && s2 == "z") || (s1 == "z" && s2 == "y"));
            } else {
                panic!("Expected denominator to be multiplication");
            }
        } else {
            panic!("Expected division");
        }
    }

    #[test]
    fn test_nested_fraction_numbers() {
        // (1/2) / (1/3) -> (1*3) / (2*1) -> 3/2 -> 1.5
        let expr = Expr::Div(
            Rc::new(Expr::Div(
                Rc::new(Expr::Number(1.0)),
                Rc::new(Expr::Number(2.0)),
            )),
            Rc::new(Expr::Div(
                Rc::new(Expr::Number(1.0)),
                Rc::new(Expr::Number(3.0)),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        if let Expr::Div(num, den) = simplified {
            if let (Expr::Number(n), Expr::Number(d)) = (num.as_ref().clone(), den.as_ref().clone())
            {
                assert_eq!(n, 3.0);
                assert_eq!(d, 2.0);
            } else {
                panic!("Expected numerator and denominator to be numbers");
            }
        } else {
            panic!("Expected division 3/2, got {:?}", simplified);
        }
    }

    #[test]
    fn test_fraction_cancellation_products() {
        // (C * R) / (C * R^2) -> 1 / R
        let expr = Expr::Div(
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("C".to_string())),
                Rc::new(Expr::Symbol("R".to_string())),
            )),
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("C".to_string())),
                Rc::new(Expr::Pow(
                    Rc::new(Expr::Symbol("R".to_string())),
                    Rc::new(Expr::Number(2.0)),
                )),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        // Expected: 1 / R
        if let Expr::Div(num, den) = simplified {
            assert_eq!(*num, Expr::Number(1.0));
            if let Expr::Symbol(s) = den.as_ref() {
                assert_eq!(s, "R");
            } else {
                panic!("Expected denominator R, got {:?}", den);
            }
        } else if let Expr::Pow(base, exp) = simplified {
            // R^-1
            if let Expr::Symbol(s) = base.as_ref() {
                assert_eq!(s, "R");
                assert_eq!(*exp, Expr::Number(-1.0));
            } else {
                panic!("Expected R^-1");
            }
        } else {
            panic!("Expected 1/R or R^-1, got {:?}", simplified);
        }
    }
}
