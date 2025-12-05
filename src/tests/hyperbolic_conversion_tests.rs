use crate::Expr;
use crate::simplification::simplify_expr;
use std::collections::HashSet;
use std::rc::Rc;

#[test]
fn test_simplify_to_sinh() {
    // (exp(x) - exp(-x)) / 2 -> sinh(x)
    let expr = Expr::Div(
        Rc::new(Expr::Sub(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
        Rc::new(Expr::Number(2.0)),
    );

    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "sinh");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected sinh(x), got {:?}", simplified);
    }
}

#[test]
fn test_simplify_to_cosh() {
    // (exp(x) + exp(-x)) / 2 -> cosh(x)
    let expr = Expr::Div(
        Rc::new(Expr::Add(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
        Rc::new(Expr::Number(2.0)),
    );

    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "cosh");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected cosh(x), got {:?}", simplified);
    }
}

#[test]
fn test_simplify_to_tanh() {
    // (exp(x) - exp(-x)) / (exp(x) + exp(-x)) -> tanh(x)
    let expr = Expr::Div(
        Rc::new(Expr::Sub(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
        Rc::new(Expr::Add(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
    );

    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "tanh");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected tanh(x), got {:?}", simplified);
    }
}

#[test]
fn test_simplify_to_coth() {
    // (exp(x) + exp(-x)) / (exp(x) - exp(-x)) -> coth(x)
    let expr = Expr::Div(
        Rc::new(Expr::Add(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
        Rc::new(Expr::Sub(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
    );

    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "coth");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected coth(x), got {:?}", simplified);
    }
}

#[test]
fn test_simplify_to_sech() {
    // 2 / (exp(x) + exp(-x)) -> sech(x)
    let expr = Expr::Div(
        Rc::new(Expr::Number(2.0)),
        Rc::new(Expr::Add(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
    );

    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "sech");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected sech(x), got {:?}", simplified);
    }
}

#[test]
fn test_simplify_to_csch() {
    // 2 / (exp(x) - exp(-x)) -> csch(x)
    let expr = Expr::Div(
        Rc::new(Expr::Number(2.0)),
        Rc::new(Expr::Sub(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )],
            }),
        )),
    );

    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "csch");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected csch(x)");
    }
}

#[test]
fn test_hyperbolic_identities() {
    // sinh(-x) = -sinh(x)
    let expr = Expr::FunctionCall {
        name: "sinh".to_string(),
        args: vec![Expr::Mul(
            Rc::new(Expr::Number(-1.0)),
            Rc::new(Expr::Symbol("x".to_string())),
        )],
    };
    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::Mul(a, b) = simplified {
        assert_eq!(*a, Expr::Number(-1.0));
        if let Expr::FunctionCall { name, args } = b.as_ref() {
            assert_eq!(name, "sinh");
            assert_eq!(args[0], Expr::Symbol("x".to_string()));
        } else {
            panic!("Expected sinh(x)");
        }
    } else {
        panic!("Expected -sinh(x)");
    }

    // cosh(-x) = cosh(x)
    let expr = Expr::FunctionCall {
        name: "cosh".to_string(),
        args: vec![Expr::Mul(
            Rc::new(Expr::Number(-1.0)),
            Rc::new(Expr::Symbol("x".to_string())),
        )],
    };
    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::FunctionCall { name, args } = simplified {
        assert_eq!(name, "cosh");
        assert_eq!(args[0], Expr::Symbol("x".to_string()));
    } else {
        panic!("Expected cosh(x)");
    }

    // cosh^2(x) - sinh^2(x) = 1
    let expr = Expr::Sub(
        Rc::new(Expr::Pow(
            Rc::new(Expr::FunctionCall {
                name: "cosh".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::Number(2.0)),
        )),
        Rc::new(Expr::Pow(
            Rc::new(Expr::FunctionCall {
                name: "sinh".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::Number(2.0)),
        )),
    );
    assert_eq!(simplify_expr(expr, HashSet::new()), Expr::Number(1.0));

    // 1 - tanh^2(x) = sech^2(x)
    let expr = Expr::Sub(
        Rc::new(Expr::Number(1.0)),
        Rc::new(Expr::Pow(
            Rc::new(Expr::FunctionCall {
                name: "tanh".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::Number(2.0)),
        )),
    );
    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::Pow(base, exp) = simplified {
        assert_eq!(*exp, Expr::Number(2.0));
        if let Expr::FunctionCall { name, args } = base.as_ref() {
            assert_eq!(name, "sech");
            assert_eq!(args[0], Expr::Symbol("x".to_string()));
        } else {
            panic!("Expected sech(x)");
        }
    } else {
        panic!("Expected sech^2(x)");
    }

    // coth^2(x) - 1 = csch^2(x)
    let expr = Expr::Sub(
        Rc::new(Expr::Pow(
            Rc::new(Expr::FunctionCall {
                name: "coth".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
            Rc::new(Expr::Number(2.0)),
        )),
        Rc::new(Expr::Number(1.0)),
    );
    let simplified = simplify_expr(expr, HashSet::new());
    if let Expr::Pow(base, exp) = simplified {
        assert_eq!(*exp, Expr::Number(2.0));
        if let Expr::FunctionCall { name, args } = base.as_ref() {
            assert_eq!(name, "csch");
            assert_eq!(args[0], Expr::Symbol("x".to_string()));
        } else {
            panic!("Expected csch(x)");
        }
    } else {
        panic!("Expected csch^2(x)");
    }
}
