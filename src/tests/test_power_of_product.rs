#[test]
fn test_power_of_product() {
    use crate::{Expr, simplification::simplify_expr};
    use std::collections::HashSet;
    use std::rc::Rc;
    // Test: (R * C)^2
    let product = Expr::Mul(
        Rc::new(Expr::Symbol("R".to_string())),
        Rc::new(Expr::Symbol("C".to_string())),
    );
    let squared = Expr::Pow(Rc::new(product), Rc::new(Expr::Number(2.0)));

    eprintln!("(R * C)^2 displays as: {}", squared);
    eprintln!("Simplified: {}", simplify_expr(squared, HashSet::new()));
    eprintln!("Expected: R^2 * C^2 or (R * C)^2");

    // Test: Something / (R * C)^2
    let div = Expr::Div(
        Rc::new(Expr::Symbol("X".to_string())),
        Rc::new(Expr::Pow(
            Rc::new(Expr::Mul(
                Rc::new(Expr::Symbol("R".to_string())),
                Rc::new(Expr::Symbol("C".to_string())),
            )),
            Rc::new(Expr::Number(2.0)),
        )),
    );

    eprintln!("\nX / (R * C)^2 displays as: {}", div);
    eprintln!("Simplified: {}", simplify_expr(div, HashSet::new()));
    eprintln!("Expected: X / (R^2 * C^2) or X / (R * C)^2");
}
