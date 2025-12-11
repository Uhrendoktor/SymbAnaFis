//! Pattern matching utilities for simplification rules
//!
//! Provides extractors for common expression patterns used by rule implementations.

use crate::{Expr, ExprKind};

/// Common pattern matching utilities for simplification rules
pub(crate) mod common {
    use super::*;

    /// Extract coefficient and base from a multiplication term
    /// Returns (coefficient, base) where base is normalized
    pub fn extract_coefficient(expr: &Expr) -> (f64, Expr) {
        match &expr.kind {
            ExprKind::Number(n) => (*n, Expr::number(1.0)),
            ExprKind::Mul(coeff, base) => {
                if let ExprKind::Number(n) = &coeff.kind {
                    (*n, base.as_ref().clone())
                } else {
                    (1.0, expr.clone())
                }
            }
            _ => (1.0, expr.clone()),
        }
    }
}

/// Trigonometric pattern matching utilities
pub(crate) mod trigonometric {
    use super::*;

    /// Extract function name and argument if expression is a trig function
    pub fn get_trig_function(expr: &Expr) -> Option<(&str, Expr)> {
        if let ExprKind::FunctionCall { name, args } = &expr.kind {
            if args.len() == 1 {
                match name.as_str() {
                    "sin" | "cos" | "tan" | "cot" | "sec" | "csc" => {
                        Some((name.as_str(), args[0].clone()))
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}
