#[cfg(test)]
mod tests {
    use crate::Expr;
    use crate::simplification::simplify_expr;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_power_collection_mul() {
        // x^2 * y^2 -> (x*y)^2
        let expr = Expr::Mul(
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("y".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        // Expected: (x*y)^2
        if let Expr::Pow(base, exp) = simplified {
            if let Expr::Number(n) = *exp {
                assert_eq!(n, 2.0);
            } else {
                panic!("Expected exponent 2.0");
            }

            if let Expr::Mul(a, b) = base.as_ref() {
                let s1 = format!("{}", a);
                let s2 = format!("{}", b);
                assert!((s1 == "x" && s2 == "y") || (s1 == "y" && s2 == "x"));
            } else {
                panic!("Expected base to be multiplication");
            }
        } else {
            panic!("Expected power expression");
        }
    }

    #[test]
    fn test_power_collection_div() {
        // x^2 / y^2 -> (x/y)^2
        let expr = Expr::Div(
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("y".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        // Expected: (x/y)^2
        if let Expr::Pow(base, exp) = simplified {
            if let Expr::Number(n) = *exp {
                assert_eq!(n, 2.0);
            } else {
                panic!("Expected exponent 2.0");
            }

            if let Expr::Div(num, den) = base.as_ref() {
                if let Expr::Symbol(s) = num.as_ref() {
                    assert_eq!(s, "x");
                } else {
                    panic!("Expected numerator x");
                }
                if let Expr::Symbol(s) = den.as_ref() {
                    assert_eq!(s, "y");
                } else {
                    panic!("Expected denominator y");
                }
            } else {
                panic!("Expected base to be division");
            }
        } else {
            panic!("Expected power expression");
        }
    }

    #[test]
    fn test_root_conversion_sqrt() {
        // x^(1/2) -> sqrt(x)
        let expr = Expr::Pow(
            Rc::new(Expr::Symbol("x".to_string())),
            Rc::new(Expr::Div(
                Rc::new(Expr::Number(1.0)),
                Rc::new(Expr::Number(2.0)),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        if let Expr::FunctionCall { name, args } = simplified {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            if let Expr::Symbol(s) = &args[0] {
                assert_eq!(s, "x");
            } else {
                panic!("Expected argument x");
            }
        } else {
            panic!("Expected sqrt function call");
        }
    }

    #[test]
    fn test_root_conversion_sqrt_decimal() {
        // x^0.5 -> sqrt(x)
        let expr = Expr::Pow(
            Rc::new(Expr::Symbol("x".to_string())),
            Rc::new(Expr::Number(0.5)),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        if let Expr::FunctionCall { name, args } = simplified {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            if let Expr::Symbol(s) = &args[0] {
                assert_eq!(s, "x");
            } else {
                panic!("Expected argument x");
            }
        } else {
            panic!("Expected sqrt function call");
        }
    }

    #[test]
    fn test_root_conversion_cbrt() {
        // x^(1/3) -> cbrt(x)
        let expr = Expr::Pow(
            Rc::new(Expr::Symbol("x".to_string())),
            Rc::new(Expr::Div(
                Rc::new(Expr::Number(1.0)),
                Rc::new(Expr::Number(3.0)),
            )),
        );
        let simplified = simplify_expr(expr, HashSet::new());

        if let Expr::FunctionCall { name, args } = simplified {
            assert_eq!(name, "cbrt");
            assert_eq!(args.len(), 1);
            if let Expr::Symbol(s) = &args[0] {
                assert_eq!(s, "x");
            } else {
                panic!("Expected argument x");
            }
        } else {
            panic!("Expected cbrt function call");
        }
    }
}
