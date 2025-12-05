#[cfg(test)]
mod tests {
    use crate::{Expr, simplification::simplify_expr};
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_numeric_gcd() {
        // 2/4 -> 1/2
        let expr = Expr::Div(Rc::new(Expr::Number(2.0)), Rc::new(Expr::Number(4.0)));
        let simplified = simplify_expr(expr, HashSet::new());
        if let Expr::Div(num, den) = simplified {
            assert_eq!(num, Rc::new(Expr::Number(1.0)));
            assert_eq!(den, Rc::new(Expr::Number(2.0)));
        } else {
            panic!("Expected Div, got {:?}", simplified);
        }

        // 10/2 -> 5
        let expr = Expr::Div(Rc::new(Expr::Number(10.0)), Rc::new(Expr::Number(2.0)));
        let simplified = simplify_expr(expr, HashSet::new());
        assert_eq!(simplified, Expr::Number(5.0));

        // 6/9 -> 2/3
        let expr = Expr::Div(Rc::new(Expr::Number(6.0)), Rc::new(Expr::Number(9.0)));
        let simplified = simplify_expr(expr, HashSet::new());
        if let Expr::Div(num, den) = simplified {
            assert_eq!(num, Rc::new(Expr::Number(2.0)));
            assert_eq!(den, Rc::new(Expr::Number(3.0)));
        } else {
            panic!("Expected Div, got {:?}", simplified);
        }
    }
}
