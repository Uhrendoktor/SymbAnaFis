#[test]
fn debug_rc_derivative() {
    use crate::{Expr, simplification::simplify_expr};
    use std::collections::HashSet;
    use std::rc::Rc;
    // Simplified RC test
    let rc = Expr::Mul(
        Rc::new(Expr::Symbol("V0".to_string())),
        Rc::new(Expr::FunctionCall {
            name: "exp".to_string(),
            args: vec![Expr::Div(
                Rc::new(Expr::Mul(
                    Rc::new(Expr::Number(-1.0)),
                    Rc::new(Expr::Symbol("t".to_string())),
                )),
                Rc::new(Expr::Mul(
                    Rc::new(Expr::Symbol("R".to_string())),
                    Rc::new(Expr::Symbol("C".to_string())),
                )),
            )],
        }),
    );

    eprintln!("===== RC CIRCUIT DERIVATIVE TEST =====");
    eprintln!("Original: {}", rc);

    let mut fixed = HashSet::new();
    fixed.insert("R".to_string());
    fixed.insert("C".to_string());
    fixed.insert("V0".to_string());

    let deriv = rc.derive("t", &fixed);
    eprintln!("Raw derivative: {}", deriv);

    let simplified = simplify_expr(deriv, fixed);
    eprintln!("Simplified: {}", simplified);
    eprintln!("Simplified Debug: {:#?}", simplified);
    eprintln!("Expected: -V0 * exp(-t / (R * C)) / (R * C)");

    let s = format!("{}", simplified);
    assert!(!s.contains("/ C * R"), "Bug found: {}", s);
}
