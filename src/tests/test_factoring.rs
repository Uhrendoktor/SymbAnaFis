#[cfg(test)]
mod tests {
    use crate::{Expr, simplification::simplify_expr};
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_perfect_square_factoring() {
        // x^2 + 2x + 1 -> (x + 1)^2
        let expr = Expr::Add(
            Rc::new(Expr::Add(
                Rc::new(Expr::Pow(
                    Rc::new(Expr::Symbol("x".to_string())),
                    Rc::new(Expr::Number(2.0)),
                )),
                Rc::new(Expr::Mul(
                    Rc::new(Expr::Number(2.0)),
                    Rc::new(Expr::Symbol("x".to_string())),
                )),
            )),
            Rc::new(Expr::Number(1.0)),
        );
        let simplified = simplify_expr(expr, HashSet::new());
        println!("Simplified: {:?}", simplified);

        // Expected: (x + 1)^2
        let expected = Expr::Pow(
            Rc::new(Expr::Add(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Number(1.0)),
            )),
            Rc::new(Expr::Number(2.0)),
        );

        assert_eq!(simplified, expected);
    }
}
