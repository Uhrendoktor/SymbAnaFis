//! Type-safe Symbol and operator overloading for ergonomic expression building
//!
//! # Example
//! ```ignore
//! use symb_anafis::{sym, Expr};
//!
//! let x = sym("x");
//! let expr = x.clone().pow(2.0) + x.sin();  // x² + sin(x)
//! ```

use crate::Expr;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Type-safe symbol for building expressions ergonomically
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(String);

impl Symbol {
    /// Create a new symbol with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Symbol(name.into())
    }

    /// Get the name of the symbol
    pub fn name(&self) -> &str {
        &self.0
    }

    /// Convert to an Expr
    pub fn to_expr(&self) -> Expr {
        Expr::symbol(&self.0)
    }

    /// Raise to a power
    pub fn pow(self, exp: impl Into<Expr>) -> Expr {
        Expr::pow(self.to_expr(), exp.into())
    }

    // === Parametric special functions ===

    /// Polygamma function: ψ^(n)(x)
    /// `x.polygamma(n)` → `polygamma(n, x)`
    pub fn polygamma(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("polygamma", vec![n.into(), self.to_expr()])
    }

    /// Beta function: B(a, b)
    pub fn beta(self, other: impl Into<Expr>) -> Expr {
        Expr::func_multi("beta", vec![self.to_expr(), other.into()])
    }

    /// Bessel function of the first kind: J_n(x)
    pub fn besselj(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("besselj", vec![n.into(), self.to_expr()])
    }

    /// Bessel function of the second kind: Y_n(x)
    pub fn bessely(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("bessely", vec![n.into(), self.to_expr()])
    }

    /// Modified Bessel function of the first kind: I_n(x)
    pub fn besseli(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("besseli", vec![n.into(), self.to_expr()])
    }

    /// Modified Bessel function of the second kind: K_n(x)
    pub fn besselk(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("besselk", vec![n.into(), self.to_expr()])
    }
}

// Allow Symbol to be used where &str is expected
impl AsRef<str> for Symbol {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ===== Unified Macro for generating math function methods =====
// Single macro handles both Symbol and Expr with a converter expression

/// Generate math function methods for a type
/// - For Symbol: converts to Expr via to_expr() before wrapping in function
/// - For Expr: uses self directly
macro_rules! impl_math_functions {
    ($type:ty, $converter:expr, $($fn_name:ident => $func_str:literal),* $(,)?) => {
        impl $type {
            $(
                pub fn $fn_name(self) -> Expr {
                    Expr::func($func_str, $converter(self))
                }
            )*
        }
    };
}

// Define the function list once
macro_rules! math_function_list {
    ($macro_name:ident, $type:ty, $converter:expr) => {
        $macro_name!($type, $converter,
            // Trigonometric functions
            sin => "sin", cos => "cos", tan => "tan",
            cot => "cot", sec => "sec", csc => "csc",
            // Inverse trigonometric functions
            asin => "asin", acos => "acos", atan => "atan",
            acot => "acot", asec => "asec", acsc => "acsc",
            // Hyperbolic functions
            sinh => "sinh", cosh => "cosh", tanh => "tanh",
            coth => "coth", sech => "sech", csch => "csch",
            // Inverse hyperbolic functions
            asinh => "asinh", acosh => "acosh", atanh => "atanh",
            acoth => "acoth", asech => "asech", acsch => "acsch",
            // Exponential and logarithmic functions
            exp => "exp", ln => "ln", log => "log",
            log10 => "log10", log2 => "log2",
            // Root functions
            sqrt => "sqrt", cbrt => "cbrt",
            // Special functions (single-argument only)
            abs => "abs", sign => "sign", sinc => "sinc",
            erf => "erf", erfc => "erfc", gamma => "gamma",
            digamma => "digamma", trigamma => "trigamma",
            zeta => "zeta", lambertw => "lambertw",
            // Note: polygamma(n,x), beta(a,b), and bessel functions are 2-arg
            // Use Expr::func_multi() for those instead
        );
    };
}

// Apply to Symbol (convert via to_expr())
math_function_list!(impl_math_functions, Symbol, |s: Symbol| s.to_expr());

// Apply to Expr (use directly)
math_function_list!(impl_math_functions, Expr, |e: Expr| e);

// Convert Symbol to Expr
impl From<Symbol> for Expr {
    fn from(s: Symbol) -> Self {
        s.to_expr()
    }
}

// Convert f64 to Expr
impl From<f64> for Expr {
    fn from(n: f64) -> Self {
        Expr::number(n)
    }
}

// Convert i32 to Expr
impl From<i32> for Expr {
    fn from(n: i32) -> Self {
        Expr::number(n as f64)
    }
}

// ===== Macro for generating operator implementations =====
// This macro generates arithmetic operator impls for combinations of types

macro_rules! impl_binary_ops {
    // For types that convert to Expr via .to_expr()
    (symbol_ops $lhs:ty, $rhs:ty, $to_lhs:expr, $to_rhs:expr) => {
        impl Add<$rhs> for $lhs {
            type Output = Expr;
            fn add(self, rhs: $rhs) -> Expr {
                Expr::add_expr($to_lhs(self), $to_rhs(rhs))
            }
        }
        impl Sub<$rhs> for $lhs {
            type Output = Expr;
            fn sub(self, rhs: $rhs) -> Expr {
                Expr::sub_expr($to_lhs(self), $to_rhs(rhs))
            }
        }
        impl Mul<$rhs> for $lhs {
            type Output = Expr;
            fn mul(self, rhs: $rhs) -> Expr {
                Expr::mul_expr($to_lhs(self), $to_rhs(rhs))
            }
        }
        impl Div<$rhs> for $lhs {
            type Output = Expr;
            fn div(self, rhs: $rhs) -> Expr {
                Expr::div_expr($to_lhs(self), $to_rhs(rhs))
            }
        }
    };
}

// Symbol operations
impl_binary_ops!(symbol_ops Symbol, Symbol, |s: Symbol| s.to_expr(), |r: Symbol| r.to_expr());
impl_binary_ops!(symbol_ops Symbol, Expr, |s: Symbol| s.to_expr(), |r: Expr| r);
impl_binary_ops!(symbol_ops Symbol, f64, |s: Symbol| s.to_expr(), |r: f64| Expr::number(r));

// Expr operations
impl_binary_ops!(symbol_ops Expr, Expr, |s: Expr| s, |r: Expr| r);
impl_binary_ops!(symbol_ops Expr, Symbol, |s: Expr| s, |r: Symbol| r.to_expr());
impl_binary_ops!(symbol_ops Expr, f64, |s: Expr| s, |r: f64| Expr::number(r));

// f64 on left side (only for + and *)
impl Add<Expr> for f64 {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::add_expr(Expr::number(self), rhs)
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::mul_expr(Expr::number(self), rhs)
    }
}

impl Sub<Expr> for f64 {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::sub_expr(Expr::number(self), rhs)
    }
}

impl Sub<Symbol> for f64 {
    type Output = Expr;
    fn sub(self, rhs: Symbol) -> Expr {
        Expr::sub_expr(Expr::number(self), rhs.to_expr())
    }
}

impl Mul<Symbol> for f64 {
    type Output = Expr;
    fn mul(self, rhs: Symbol) -> Expr {
        Expr::mul_expr(Expr::number(self), rhs.to_expr())
    }
}

// Negation
impl Neg for Symbol {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::mul_expr(Expr::number(-1.0), self.to_expr())
    }
}

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::mul_expr(Expr::number(-1.0), self)
    }
}

// ===== Expr Methods =====

impl Expr {
    /// Raise to a power (since Rust ^ is XOR, not power)
    /// Use this method on Expr instances: `expr.pow(2.0)` or `expr.pow_of(2.0)`
    ///
    /// Note: This is an instance method that consumes self.
    /// For constructing from two expressions, use `Expr::pow(base, exp)` static method.
    #[inline]
    pub fn pow_expr(self, exp: impl Into<Expr>) -> Expr {
        Expr::pow(self, exp.into())
    }

    /// Alias for pow_expr for backward compatibility
    #[inline]
    pub fn pow_of(self, exp: impl Into<Expr>) -> Expr {
        self.pow_expr(exp)
    }

    // === Parametric special functions ===
    // These are 2-arg functions where the first arg is typically an order/parameter
    // and self is the main argument.

    /// Polygamma function: ψ^(n)(x) - nth derivative of digamma
    /// `x.polygamma(n)` → `polygamma(n, x)`
    pub fn polygamma(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("polygamma", vec![n.into(), self])
    }

    /// Beta function: B(a, b)
    /// `x.beta(y)` → `beta(x, y)`
    pub fn beta(self, other: impl Into<Expr>) -> Expr {
        Expr::func_multi("beta", vec![self, other.into()])
    }

    /// Bessel function of the first kind: J_n(x)
    /// `x.besselj(n)` → `besselj(n, x)`
    pub fn besselj(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("besselj", vec![n.into(), self])
    }

    /// Bessel function of the second kind: Y_n(x)
    /// `x.bessely(n)` → `bessely(n, x)`
    pub fn bessely(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("bessely", vec![n.into(), self])
    }

    /// Modified Bessel function of the first kind: I_n(x)
    /// `x.besseli(n)` → `besseli(n, x)`
    pub fn besseli(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("besseli", vec![n.into(), self])
    }

    /// Modified Bessel function of the second kind: K_n(x)
    /// `x.besselk(n)` → `besselk(n, x)`
    pub fn besselk(self, n: impl Into<Expr>) -> Expr {
        Expr::func_multi("besselk", vec![n.into(), self])
    }

    // Note: For custom n-arg functions, use Expr::call() - see ast.rs
}

/// Convenience function to create a Symbol
pub fn sym(name: &str) -> Symbol {
    Symbol::new(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_basic() {
        let x = sym("x");
        assert_eq!(x.name(), "x");
        assert_eq!(format!("{}", x.to_expr()), "x");
    }

    #[test]
    fn test_symbol_arithmetic() {
        let x = sym("x");
        let y = sym("y");

        let sum = x.clone() + y.clone();
        assert_eq!(format!("{}", sum), "x + y");

        let product = x.clone() * y.clone();
        // Note: display uses explicit * for symbol * symbol
        assert!(format!("{}", product).contains("x") && format!("{}", product).contains("y"));

        let scaled = 2.0 * x.clone();
        assert_eq!(format!("{}", scaled), "2x");
    }

    #[test]
    fn test_symbol_power() {
        let x = sym("x");
        let squared = x.pow(2.0);
        assert_eq!(format!("{}", squared), "x^2");
    }

    #[test]
    fn test_symbol_functions() {
        let x = sym("x");
        assert_eq!(format!("{}", x.clone().sin()), "sin(x)");
        assert_eq!(format!("{}", x.clone().cos()), "cos(x)");
        assert_eq!(format!("{}", x.clone().exp()), "exp(x)");
        assert_eq!(format!("{}", x.ln()), "ln(x)");
    }

    #[test]
    fn test_expr_arithmetic() {
        let x = sym("x");
        let expr = x.clone().pow(2.0) + 2.0 * x.clone() + 1.0;
        assert!(format!("{}", expr).contains("x^2"));
    }
}
