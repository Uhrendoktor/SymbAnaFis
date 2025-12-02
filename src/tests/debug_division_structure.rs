#[test]
fn test_division_structure() {
    use crate::Expr;
    use std::rc::Rc;
    // Debug: What structure does the derivative create?
    // Manual construction of: something / (C * R^2)
    let proper = Expr::Div(
        Rc::new(Expr::Symbol("X".to_string())),
        Rc::new(Expr::Mul(
            Rc::new(Expr::Symbol("C".to_string())),
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("R".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
        )),
    );
    eprintln!("Proper structure: {}", proper);
    eprintln!("  Debug: {:?}", proper);

    // What if it's: (something / C) * R^2 ?
    let wrong = Expr::Mul(
        Rc::new(Expr::Div(
            Rc::new(Expr::Symbol("X".to_string())),
            Rc::new(Expr::Symbol("C".to_string())),
        )),
        Rc::new(Expr::Pow(
            Rc::new(Expr::Symbol("R".to_string())),
            Rc::new(Expr::Number(2.0)),
        )),
    );
    eprintln!("Wrong structure: {}", wrong);
    eprintln!("  Debug: {:?}", wrong);

    // What about: something / R * C^2 (parsed as (something/R)*C^2)?
    let ambiguous = Expr::Mul(
        Rc::new(Expr::Div(
            Rc::new(Expr::Symbol("X".to_string())),
            Rc::new(Expr::Symbol("R".to_string())),
        )),
        Rc::new(Expr::Pow(
            Rc::new(Expr::Symbol("C".to_string())),
            Rc::new(Expr::Number(2.0)),
        )),
    );
    eprintln!("Ambiguous structure: {}", ambiguous);
    eprintln!("  Debug: {:?}", ambiguous);
}
