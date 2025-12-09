#[cfg(test)]
mod tests {
    use crate::{Expr, ExprKind, simplification::helpers};

    #[test]
    fn debug_perfect_square_logic() {
        // Construct 4*x^2 + 4*x + 1
        let x = Expr::symbol("x");
        let term1 = Expr::mul_expr(Expr::number(4.0), Expr::pow(x.clone(), Expr::number(2.0)));
        let term2 = Expr::mul_expr(Expr::number(4.0), x.clone());
        let term3 = Expr::number(1.0);

        let expr = Expr::add_expr(Expr::add_expr(term1.clone(), term2.clone()), term3.clone());

        println!("Expr: {}", expr);

        // Manual flatten
        let terms = helpers::flatten_add(&expr);
        println!("Terms count: {}", terms.len());
        for (i, term) in terms.iter().enumerate() {
            println!("Term {}: {:?}", i, term);

            // Debug flattening mul
            let factors = helpers::flatten_mul(term);
            println!("  Factors: {:?}", factors);

            // Debug extract coeff
            let (coeff, base) = helpers::extract_coeff(term);
            println!("  Coeff: {}, Base: {}", coeff, base);
        }

        // Verify PerfectSquareRule logic manually
        let mut square_terms = Vec::new();
        let mut linear_terms = Vec::new();
        let mut constants = Vec::new();

        for term in &terms {
            match &term.kind {
                ExprKind::Pow(base, exp) => {
                    if let ExprKind::Number(n) = &exp.kind
                        && (*n - 2.0).abs() < 1e-10
                    {
                        square_terms.push((1.0, base.as_ref().clone()));
                        continue;
                    }
                    linear_terms.push((1.0, term.clone(), Expr::number(1.0)));
                }
                ExprKind::Number(n) => {
                    constants.push(*n);
                }
                ExprKind::Mul(_, _) => {
                    let (coeff, factors) = extract_coeff_and_factors_debug(term);
                    println!("  Mul term decomp: coeff={}, factors={:?}", coeff, factors);

                    if factors.len() == 1 {
                        if let ExprKind::Pow(base, exp) = &factors[0].kind
                            && let ExprKind::Number(n) = &exp.kind
                            && (*n - 2.0).abs() < 1e-10
                        {
                            square_terms.push((coeff, base.as_ref().clone()));
                            continue;
                        }
                        linear_terms.push((coeff, factors[0].clone(), Expr::number(1.0)));
                    } else if factors.len() == 2 {
                        linear_terms.push((coeff, factors[0].clone(), factors[1].clone()));
                    }
                }
                _ => {
                    linear_terms.push((1.0, term.clone(), Expr::number(1.0)));
                }
            }
        }

        println!("Squares: {:?}", square_terms);
        println!("Linears: {:?}", linear_terms);
        println!("Constants: {:?}", constants);

        assert_eq!(square_terms.len(), 1, "Expected 1 square term");
        assert_eq!(linear_terms.len(), 1, "Expected 1 linear term");
        assert_eq!(constants.len(), 1, "Expected 1 constant");
    }

    fn extract_coeff_and_factors_debug(term: &Expr) -> (f64, Vec<Expr>) {
        let factors = helpers::flatten_mul(term);
        let mut coeff = 1.0;
        let mut non_numeric: Vec<Expr> = Vec::new();

        for f in factors {
            if let ExprKind::Number(n) = &f.kind {
                coeff *= n;
            } else {
                non_numeric.push(f);
            }
        }
        (coeff, non_numeric)
    }
}
