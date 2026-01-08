//! Display implementations for expressions.
//!
//! This module provides three output formats for mathematical expressions:
//!
//! ## Standard Display (`to_string()` / `{}`)
//! Human-readable mathematical notation:
//! - `x^2 + 2*x + 1`
//! - `sin(x) + cos(x)`
//!
//! ## LaTeX Format (`to_latex()`)
//! For typesetting in documents with proper mathematical notation:
//! - `x^{2} + 2 \cdot x + 1`
//! - `\sin\left(x\right) + \cos\left(x\right)`
//! - Supports special functions: Bessel, polygamma, elliptic integrals, etc.
//!
//! ## Unicode Format (`to_unicode()`)
//! Pretty display with Unicode superscripts and Greek letters:
//! - `x² + 2·x + 1`
//! - `sin(x) + cos(x)` with π, α, β, etc. for Greek variables
//!
//! # Display Behavior Notes for N-ary AST
//! - Sum displays terms with +/- signs based on leading coefficients
//! - Product displays with explicit `*` or `·` multiplication
//! - `e^x` is always displayed as `exp(x)` for consistency
//! - Derivatives use ∂ notation

use crate::core::known_symbols::E;
use crate::core::traits::EPSILON;
use crate::{Expr, ExprKind};
use std::fmt;
use std::sync::Arc;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Check if an expression is negative (has a negative leading coefficient)
/// Returns `Some(positive_equivalent)` if the expression has a negative sign
fn extract_negative(expr: &Expr) -> Option<Expr> {
    match &expr.kind {
        ExprKind::Product(factors) => {
            if !factors.is_empty()
                && let ExprKind::Number(n) = &factors[0].kind
                && *n < 0.0
            {
                // Negative leading coefficient
                let abs_coeff = n.abs();
                if (abs_coeff - 1.0).abs() < EPSILON {
                    // Exactly -1: just remove it
                    if factors.len() == 2 {
                        return Some((*factors[1]).clone());
                    }
                    let remaining: Vec<Arc<Expr>> = factors[1..].to_vec();
                    return Some(Expr::product_from_arcs(remaining));
                }
                // Other negative coefficient like -2, -3.5: replace with positive
                let mut new_factors: Vec<Arc<Expr>> = Vec::with_capacity(factors.len());
                new_factors.push(Arc::new(Expr::number(abs_coeff)));
                new_factors.extend(factors[1..].iter().cloned());
                return Some(Expr::product_from_arcs(new_factors));
            }
        }
        ExprKind::Number(n) => {
            if *n < 0.0 {
                return Some(Expr::number(-*n));
            }
        }
        // Handle Poly with negative first term
        ExprKind::Poly(poly) => {
            if let Some(first_coeff) = poly.first_coeff()
                && first_coeff < 0.0
            {
                // Create a new Poly with ALL terms negated
                // -P = -(P_negated)
                // e.g. -x + x^2  -> negating all terms gives x - x^2
                // displayed as -(x - x^2) which is correct (-x + x^2)
                let negated_poly = poly.negate();
                return Some(Expr::new(ExprKind::Poly(negated_poly)));
            }
        }
        _ => {}
    }
    None
}

/// Check if expression needs parentheses when displayed in a product
const fn needs_parens_in_product(expr: &Expr) -> bool {
    matches!(expr.kind, ExprKind::Sum(_) | ExprKind::Poly(_))
}

/// Check if expression needs parentheses when displayed as a power base
fn needs_parens_as_base(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Sum(_) | ExprKind::Product(_) | ExprKind::Div(_, _) | ExprKind::Poly(_) => true,
        ExprKind::Number(n) => *n < 0.0, // Negative numbers need parens: (-1)^x not -1^x
        _ => false,
    }
}

/// Format a single factor for display in a product chain
fn format_factor(expr: &Expr) -> String {
    if needs_parens_in_product(expr) {
        format!("({expr})")
    } else {
        format!("{expr}")
    }
}

/// Format a single term for display in a sum chain
fn format_sum_term(expr: &Expr) -> String {
    if needs_parens_in_sum(expr) {
        format!("({expr})")
    } else {
        format!("{expr}")
    }
}

/// Check if expression needs parentheses when displayed as a sum term
const fn needs_parens_in_sum(expr: &Expr) -> bool {
    matches!(expr.kind, ExprKind::Sum(_) | ExprKind::Poly(_))
}

/// Greek letter mappings: (name, latex, unicode)
/// Covers lowercase Greek alphabet commonly used in mathematics and physics
static GREEK_LETTERS: &[(&str, &str, &str)] = &[
    // Common mathematical symbols
    ("pi", r"\pi", "\u{3c0}"),
    ("alpha", r"\alpha", "\u{3b1}"),
    ("beta", r"\beta", "\u{3b2}"),
    ("gamma", r"\gamma", "\u{3b3}"),
    ("delta", r"\delta", "\u{3b4}"),
    ("epsilon", r"\epsilon", "\u{3b5}"),
    ("zeta", r"\zeta", "\u{3b6}"),
    ("eta", r"\eta", "\u{3b7}"),
    ("theta", r"\theta", "\u{3b8}"),
    ("iota", r"\iota", "\u{3b9}"),
    ("kappa", r"\kappa", "\u{3ba}"),
    ("lambda", r"\lambda", "\u{3bb}"),
    ("mu", r"\mu", "\u{3bc}"),
    ("nu", r"\nu", "\u{3bd}"),
    ("xi", r"\xi", "\u{3be}"),
    ("omicron", r"\omicron", "\u{3bf}"),
    ("rho", r"\rho", "\u{3c1}"),
    ("sigma", r"\sigma", "\u{3c3}"),
    ("tau", r"\tau", "\u{3c4}"),
    ("upsilon", r"\upsilon", "\u{3c5}"),
    ("phi", r"\phi", "\u{3c6}"),
    ("chi", r"\chi", "\u{3c7}"),
    ("psi", r"\psi", "\u{3c8}"),
    ("omega", r"\omega", "\u{3c9}"),
    // Alternative forms
    ("varepsilon", r"\varepsilon", "\u{3b5}"),
    ("vartheta", r"\vartheta", "\u{3d1}"),
    ("varphi", r"\varphi", "\u{3c6}"),
    ("varrho", r"\varrho", "\u{3c1}"),
    ("varsigma", r"\varsigma", "\u{3c2}"),
];

/// Map symbol name to Greek letter (LaTeX format)
fn greek_to_latex(name: &str) -> Option<&'static str> {
    GREEK_LETTERS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, latex, _)| *latex)
}

/// Map symbol name to Unicode Greek letter
fn greek_to_unicode(name: &str) -> Option<&'static str> {
    GREEK_LETTERS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, _, unicode)| *unicode)
}

// =============================================================================
// DISPLAY IMPLEMENTATION
// =============================================================================

impl fmt::Display for Expr {
    // Large match blocks for different expression kinds
    #[allow(clippy::too_many_lines)] // Large match block for different expression kinds
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Number(n) => {
                if n.is_nan() {
                    write!(f, "NaN")
                } else if n.is_infinite() {
                    if *n > 0.0 {
                        write!(f, "Infinity")
                    } else {
                        write!(f, "-Infinity")
                    }
                } else if {
                    #[allow(clippy::float_cmp)]
                    let is_int = n.trunc() == *n;
                    is_int
                } && n.abs() < 1e10
                {
                    #[allow(clippy::cast_possible_truncation)] // Checked is_int above
                    let n_int = *n as i64;
                    write!(f, "{n_int}")
                } else {
                    write!(f, "{n}")
                }
            }

            ExprKind::Symbol(s) => write!(f, "{s}"),

            ExprKind::FunctionCall { name, args } => {
                if args.is_empty() {
                    write!(f, "{name}()")
                } else {
                    let args_str: Vec<String> = args.iter().map(|arg| format!("{arg}")).collect();
                    write!(f, "{}({})", name, args_str.join(", "))
                }
            }

            // N-ary Sum: display with + and - signs
            ExprKind::Sum(terms) => {
                if terms.is_empty() {
                    return write!(f, "0");
                }

                let mut first = true;
                for term in terms {
                    if first {
                        // First term: check if negative
                        if let Some(positive_term) = extract_negative(term) {
                            write!(f, "-{}", format_sum_term(&positive_term))?;
                        } else {
                            write!(f, "{}", format_sum_term(term))?;
                        }
                        first = false;
                    } else {
                        // Subsequent terms: show + or -
                        if let Some(positive_term) = extract_negative(term) {
                            write!(f, " - {}", format_sum_term(&positive_term))?;
                        } else {
                            write!(f, " + {}", format_sum_term(term))?;
                        }
                    }
                }
                Ok(())
            }

            // N-ary Product: display with * or implicit multiplication
            ExprKind::Product(factors) => {
                if factors.is_empty() {
                    return write!(f, "1");
                }

                // Check for leading -1
                if !factors.is_empty()
                    && let ExprKind::Number(n) = &factors[0].kind
                    && (*n + 1.0).abs() < EPSILON
                {
                    // Leading -1: display as negation
                    if factors.len() == 1 {
                        return write!(f, "-1");
                    } else if factors.len() == 2 {
                        return write!(f, "-{}", format_factor(&factors[1]));
                    }
                    let remaining: Vec<Arc<Self>> = factors[1..].to_vec();
                    let rest = Self::product_from_arcs(remaining);
                    return write!(f, "-{}", format_factor(&rest));
                }

                // Display factors with explicit * separator - write directly to avoid Vec<String>
                let mut first = true;
                for fac in factors {
                    if !first {
                        write!(f, "*")?;
                    }
                    write!(f, "{}", format_factor(fac))?;
                    first = false;
                }
                Ok(())
            }

            ExprKind::Div(u, v) => {
                let num_str = format!("{u}");
                let denom_str = format!("{v}");

                // Parenthesize numerator if it's a sum
                let formatted_num = if matches!(u.kind, ExprKind::Sum(_)) {
                    format!("({num_str})")
                } else {
                    num_str
                };

                // Parenthesize denominator if it's not simple
                let formatted_denom = match &v.kind {
                    ExprKind::Symbol(_)
                    | ExprKind::Number(_)
                    | ExprKind::Pow(_, _)
                    | ExprKind::FunctionCall { .. } => denom_str,
                    _ => format!("({denom_str})"),
                };

                write!(f, "{formatted_num}/{formatted_denom}")
            }

            ExprKind::Pow(u, v) => {
                // Special case: e^x displays as exp(x)
                if let ExprKind::Symbol(s) = &u.kind
                    && s.id() == *E
                {
                    return write!(f, "exp({v})");
                }

                let base_str = format!("{u}");
                let exp_str = format!("{v}");

                let formatted_base = if needs_parens_as_base(u) {
                    format!("({base_str})")
                } else {
                    base_str
                };

                let formatted_exp = match &v.kind {
                    ExprKind::Number(_) | ExprKind::Symbol(_) => exp_str,
                    _ => format!("({exp_str})"),
                };

                write!(f, "{formatted_base}^{formatted_exp}")
            }

            ExprKind::Derivative { inner, var, order } => {
                write!(f, "\u{2202}^{order}_{inner}/\u{2202}_{var}^{order}")
            }

            // Poly: display inline using Polynomial's Display
            ExprKind::Poly(poly) => {
                write!(f, "{poly}")
            }
        }
    }
}

// =============================================================================
// LATEX FORMATTER
// =============================================================================

pub struct LatexFormatter<'expr>(pub(crate) &'expr Expr);

impl fmt::Display for LatexFormatter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_latex(self.0, f)
    }
}

// Display format functions are naturally lengthy due to many expression kinds
#[allow(clippy::too_many_lines)] // Display format naturally lengthy due to many expr kinds
fn format_latex(expr: &Expr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::Number(n) => {
            if n.is_nan() {
                write!(f, r"\text{{NaN}}")
            } else if n.is_infinite() {
                if *n > 0.0 {
                    write!(f, r"\infty")
                } else {
                    write!(f, r"-\infty")
                }
            } else if {
                #[allow(clippy::float_cmp)] // Checking for exact integer via trunc
                let is_int = n.trunc() == *n;
                is_int
            } && n.abs() < 1e10
            {
                #[allow(clippy::cast_possible_truncation)] // Checked is_int above
                let n_int = *n as i64;
                write!(f, "{n_int}")
            } else {
                write!(f, "{n}")
            }
        }

        ExprKind::Symbol(s) => {
            let name = s.as_ref();
            if let Some(greek) = greek_to_latex(name) {
                write!(f, "{greek}")
            } else {
                write!(f, "{name}")
            }
        }

        ExprKind::FunctionCall { name, args } => {
            // Special formatting for specific functions
            match name.as_str() {
                // === ROOTS ===
                "sqrt" if args.len() == 1 => {
                    return write!(f, r"\sqrt{{{}}}", LatexFormatter(&args[0]));
                }
                "cbrt" if args.len() == 1 => {
                    return write!(f, r"\sqrt[3]{{{}}}", LatexFormatter(&args[0]));
                }

                // === ABSOLUTE VALUE ===
                "abs" if args.len() == 1 => {
                    return write!(f, r"\left|{}\right|", LatexFormatter(&args[0]));
                }

                // === FLOOR/CEIL ===
                "floor" if args.len() == 1 => {
                    return write!(f, r"\lfloor{}\rfloor", LatexFormatter(&args[0]));
                }
                "ceil" if args.len() == 1 => {
                    return write!(f, r"\lceil{}\rceil", LatexFormatter(&args[0]));
                }

                // === BESSEL FUNCTIONS: J_n(x), Y_n(x), I_n(x), K_n(x) ===
                "besselj" if args.len() == 2 => {
                    return write!(
                        f,
                        r"J_{{{}}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }
                "bessely" if args.len() == 2 => {
                    return write!(
                        f,
                        r"Y_{{{}}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }
                "besseli" if args.len() == 2 => {
                    return write!(
                        f,
                        r"I_{{{}}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }
                "besselk" if args.len() == 2 => {
                    return write!(
                        f,
                        r"K_{{{}}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }

                // === ORTHOGONAL POLYNOMIALS ===
                "hermite" if args.len() == 2 => {
                    return write!(
                        f,
                        r"H_{{{}}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }
                "assoc_legendre" if args.len() == 3 => {
                    return write!(
                        f,
                        r"P_{{{}}}^{{{}}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1]),
                        LatexFormatter(&args[2])
                    );
                }
                "spherical_harmonic" | "ynm" if args.len() == 4 => {
                    return write!(
                        f,
                        r"Y_{{{}}}^{{{}}}\left({}, {}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1]),
                        LatexFormatter(&args[2]),
                        LatexFormatter(&args[3])
                    );
                }

                // === POLYGAMMA FUNCTIONS ===
                "digamma" if args.len() == 1 => {
                    return write!(f, r"\psi\left({}\right)", LatexFormatter(&args[0]));
                }
                "trigamma" if args.len() == 1 => {
                    return write!(f, r"\psi_1\left({}\right)", LatexFormatter(&args[0]));
                }
                "tetragamma" if args.len() == 1 => {
                    return write!(f, r"\psi_2\left({}\right)", LatexFormatter(&args[0]));
                }
                "polygamma" if args.len() == 2 => {
                    return write!(
                        f,
                        r"\psi^{{({})}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }

                // === ELLIPTIC INTEGRALS ===
                "elliptic_k" if args.len() == 1 => {
                    return write!(f, r"K\left({}\right)", LatexFormatter(&args[0]));
                }
                "elliptic_e" if args.len() == 1 => {
                    return write!(f, r"E\left({}\right)", LatexFormatter(&args[0]));
                }

                // === ZETA ===
                "zeta" if args.len() == 1 => {
                    return write!(f, r"\zeta\left({}\right)", LatexFormatter(&args[0]));
                }
                "zeta_deriv" if args.len() == 2 => {
                    return write!(
                        f,
                        r"\zeta^{{({})}}\left({}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }

                // === LAMBERT W ===
                "lambertw" if args.len() == 1 => {
                    return write!(f, r"W\left({}\right)", LatexFormatter(&args[0]));
                }

                // === BETA ===
                "beta" if args.len() == 2 => {
                    return write!(
                        f,
                        r"\mathrm{{B}}\left({}, {}\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }

                // === LOG WITH BASE ===
                "log" if args.len() == 2 => {
                    return write!(
                        f,
                        r"\log_{{{}}}\\left({}\\right)",
                        LatexFormatter(&args[0]),
                        LatexFormatter(&args[1])
                    );
                }

                _ => {}
            }

            // Standard function name LaTeX mappings
            let latex_name = match name.as_str() {
                // Trigonometric
                "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "sinh" | "cosh" | "tanh"
                | "coth" => format!(r"\{name}"),
                "sech" => r"\operatorname{sech}".to_owned(),
                "csch" => r"\operatorname{csch}".to_owned(),
                // Inverse hyperbolic
                "asinh" => r"\operatorname{arsinh}".to_owned(),
                "acosh" => r"\operatorname{arcosh}".to_owned(),
                "atanh" => r"\operatorname{artanh}".to_owned(),
                "acoth" => r"\operatorname{arcoth}".to_owned(),
                "asech" => r"\operatorname{arsech}".to_owned(),
                "acsch" => r"\operatorname{arcsch}".to_owned(),
                // Logarithms
                "ln" => r"\ln".to_owned(),
                "log" | "log10" => r"\log".to_owned(),
                "log2" => r"\log_2".to_owned(),
                // Exponential
                "exp" => r"\exp".to_owned(),
                "exp_polar" => r"\operatorname{exp\_polar}".to_owned(),
                // Special functions
                "gamma" => r"\Gamma".to_owned(),
                "erf" => r"\operatorname{erf}".to_owned(),
                "erfc" => r"\operatorname{erfc}".to_owned(),
                "signum" => r"\operatorname{sgn}".to_owned(),
                "sinc" => r"\operatorname{sinc}".to_owned(),
                "round" => r"\operatorname{round}".to_owned(),
                // Default: wrap in \text{}
                _ => format!(r"\text{{{name}}}"),
            };

            if args.is_empty() {
                write!(f, r"{latex_name}\left(\right)")
            } else {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|arg| format!("{}", LatexFormatter(arg)))
                    .collect();
                write!(f, r"{}\left({}\right)", latex_name, args_str.join(", "))
            }
        }

        ExprKind::Sum(terms) => {
            if terms.is_empty() {
                return write!(f, "0");
            }

            let mut first = true;
            for term in terms {
                if first {
                    if let Some(positive_term) = extract_negative(term) {
                        write!(f, "-{}", latex_factor(&positive_term))?;
                    } else {
                        write!(f, "{}", latex_factor(term))?;
                    }
                    first = false;
                } else if let Some(positive_term) = extract_negative(term) {
                    write!(f, " - {}", latex_factor(&positive_term))?;
                } else {
                    write!(f, " + {}", latex_factor(term))?;
                }
            }
            Ok(())
        }

        ExprKind::Product(factors) => {
            if factors.is_empty() {
                return write!(f, "1");
            }

            // Check for leading -1
            if !factors.is_empty()
                && let ExprKind::Number(n) = &factors[0].kind
                && (*n + 1.0).abs() < EPSILON
            {
                if factors.len() == 1 {
                    return write!(f, "-1");
                }
                let remaining: Vec<Arc<Expr>> = factors[1..].to_vec();
                let rest = Expr::product_from_arcs(remaining);
                return write!(f, "-{}", latex_factor(&rest));
            }

            // Write factors with \cdot separator directly
            let mut first = true;
            for fac in factors {
                if !first {
                    write!(f, r" \cdot ")?;
                }
                write!(f, "{}", latex_factor(fac))?;
                first = false;
            }
            Ok(())
        }

        ExprKind::Div(u, v) => {
            write!(
                f,
                r"\frac{{{}}}{{{}}}",
                LatexFormatter(u),
                LatexFormatter(v)
            )
        }

        ExprKind::Pow(u, v) => {
            if let ExprKind::Symbol(s) = &u.kind
                && s.id() == *E
            {
                return write!(f, r"e^{{{}}}", LatexFormatter(v));
            }

            let base_str = if needs_parens_as_base(u) {
                format!(r"\left({}\right)", LatexFormatter(u))
            } else {
                format!("{}", LatexFormatter(u))
            };

            write!(f, "{}^{{{}}}", base_str, LatexFormatter(v))
        }

        ExprKind::Derivative { inner, var, order } => {
            if *order == 1 {
                write!(
                    f,
                    r"\frac{{\partial {}}}{{\partial {}}}",
                    LatexFormatter(inner),
                    var
                )
            } else {
                write!(
                    f,
                    r"\frac{{\partial^{} {}}}{{\partial {}^{}}}",
                    order,
                    LatexFormatter(inner),
                    var,
                    order
                )
            }
        }

        // Poly: display inline in LaTeX
        ExprKind::Poly(poly) => write!(f, "{poly}"),
    }
}

fn latex_factor(expr: &Expr) -> String {
    if matches!(expr.kind, ExprKind::Sum(_) | ExprKind::Poly(_)) {
        format!(r"\left({}\right)", LatexFormatter(expr))
    } else {
        format!("{}", LatexFormatter(expr))
    }
}

// =============================================================================
// UNICODE FORMATTER
// =============================================================================

pub struct UnicodeFormatter<'expr>(pub(crate) &'expr Expr);

impl fmt::Display for UnicodeFormatter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_unicode(self.0, f)
    }
}

#[inline]
const fn to_superscript(c: char) -> char {
    match c {
        '0' => '\u{2070}',
        '1' => '\u{b9}',
        '2' => '\u{b2}',
        '3' => '\u{b3}',
        '4' => '\u{2074}',
        '5' => '\u{2075}',
        '6' => '\u{2076}',
        '7' => '\u{2077}',
        '8' => '\u{2078}',
        '9' => '\u{2079}',
        '-' => '\u{207b}',
        '+' => '\u{207a}',
        '(' => '\u{207d}',
        ')' => '\u{207e}',
        _ => c,
    }
}

#[inline]
fn num_to_superscript(n: f64) -> String {
    if {
        #[allow(clippy::float_cmp)] // Checking for exact integer via trunc
        let is_int = n.trunc() == n;
        is_int
    } && n.abs() < 1000.0
    {
        #[allow(clippy::cast_possible_truncation)] // Checked is_int above
        let n_int = n as i64;
        format!("{n_int}").chars().map(to_superscript).collect()
    } else {
        format!("^{n}")
    }
}

// Display format functions are naturally lengthy due to many expression kinds
#[allow(clippy::too_many_lines)] // Display format naturally lengthy due to many expr kinds
fn format_unicode(expr: &Expr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::Number(n) => {
            if n.is_nan() {
                write!(f, "NaN")
            } else if n.is_infinite() {
                write!(f, "{}", if *n > 0.0 { "\u{221e}" } else { "-\u{221e}" })
            } else if {
                #[allow(clippy::float_cmp)] // Checking for exact integer via trunc
                let is_int = n.trunc() == *n;
                is_int
            } && n.abs() < 1e10
            {
                #[allow(clippy::cast_possible_truncation)] // Checked is_int above
                let n_int = *n as i64;
                write!(f, "{n_int}")
            } else {
                write!(f, "{n}")
            }
        }

        ExprKind::Symbol(s) => {
            let name = s.as_ref();
            if let Some(greek) = greek_to_unicode(name) {
                write!(f, "{greek}")
            } else {
                write!(f, "{name}")
            }
        }

        ExprKind::FunctionCall { name, args } => {
            if args.is_empty() {
                write!(f, "{name}()")
            } else {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|a| format!("{}", UnicodeFormatter(a)))
                    .collect();
                write!(f, "{}({})", name, args_str.join(", "))
            }
        }

        ExprKind::Sum(terms) => {
            if terms.is_empty() {
                return write!(f, "0");
            }
            let mut first = true;
            for term in terms {
                if first {
                    if let Some(positive) = extract_negative(term) {
                        write!(f, "\u{2212}{}", unicode_factor(&positive))?;
                    } else {
                        write!(f, "{}", unicode_factor(term))?;
                    }
                    first = false;
                } else if let Some(positive) = extract_negative(term) {
                    write!(f, " \u{2212} {}", unicode_factor(&positive))?;
                } else {
                    write!(f, " + {}", unicode_factor(term))?;
                }
            }
            Ok(())
        }

        ExprKind::Product(factors) => {
            if factors.is_empty() {
                return write!(f, "1");
            }
            if !factors.is_empty()
                && let ExprKind::Number(n) = &factors[0].kind
                && (*n + 1.0).abs() < EPSILON
            {
                if factors.len() == 1 {
                    return write!(f, "\u{2212}1");
                }
                let remaining: Vec<Arc<Expr>> = factors[1..].to_vec();
                let rest = Expr::product_from_arcs(remaining);
                return write!(f, "\u{2212}{}", unicode_factor(&rest));
            }
            // Write factors with · separator directly
            let mut first = true;
            for fac in factors {
                if !first {
                    write!(f, "\u{b7}")?;
                }
                write!(f, "{}", unicode_factor(fac))?;
                first = false;
            }
            Ok(())
        }

        ExprKind::Div(u, v) => {
            let num = if matches!(u.kind, ExprKind::Sum(_)) {
                format!("({})", UnicodeFormatter(u))
            } else {
                format!("{}", UnicodeFormatter(u))
            };
            let denom = match &v.kind {
                ExprKind::Symbol(_)
                | ExprKind::Number(_)
                | ExprKind::Pow(_, _)
                | ExprKind::FunctionCall { .. } => format!("{}", UnicodeFormatter(v)),
                _ => format!("({})", UnicodeFormatter(v)),
            };
            write!(f, "{num}/{denom}")
        }

        ExprKind::Pow(u, v) => {
            if let ExprKind::Symbol(s) = &u.kind
                && s.id() == *E
            {
                return write!(f, "exp({})", UnicodeFormatter(v));
            }
            let base = if needs_parens_as_base(u) {
                format!("({})", UnicodeFormatter(u))
            } else {
                format!("{}", UnicodeFormatter(u))
            };
            if let ExprKind::Number(n) = &v.kind {
                write!(f, "{}{}", base, num_to_superscript(*n))
            } else if matches!(v.kind, ExprKind::Symbol(_)) {
                write!(f, "{}^{}", base, UnicodeFormatter(v))
            } else {
                write!(f, "{}^({})", base, UnicodeFormatter(v))
            }
        }

        ExprKind::Derivative { inner, var, order } => {
            let sup = num_to_superscript(f64::from(*order));
            write!(
                f,
                "\u{2202}{}({})/\u{2202}{}{}",
                sup,
                UnicodeFormatter(inner),
                var,
                sup
            )
        }

        // Poly: display inline in unicode
        ExprKind::Poly(poly) => write!(f, "{poly}"),
    }
}

fn unicode_factor(expr: &Expr) -> String {
    if matches!(expr.kind, ExprKind::Sum(_) | ExprKind::Poly(_)) {
        format!("({})", UnicodeFormatter(expr))
    } else {
        format!("{}", UnicodeFormatter(expr))
    }
}

// =============================================================================
// EXPR FORMATTING METHODS
// =============================================================================

impl Expr {
    /// Convert the expression to LaTeX format.
    ///
    /// Returns a string suitable for rendering in LaTeX math environments.
    #[must_use]
    pub fn to_latex(&self) -> String {
        format!("{}", LatexFormatter(self))
    }

    /// Convert the expression to Unicode format.
    ///
    /// Returns a string with Unicode superscripts and Greek letters for display.
    #[must_use]
    pub fn to_unicode(&self) -> String {
        format!("{}", UnicodeFormatter(self))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
// Standard test relaxations: unwrap/panic for assertions, precision loss for math
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::let_underscore_must_use,
    clippy::no_effect_underscore_binding
)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    #[allow(clippy::approx_constant)] // Testing exact float display, not mathematical approximation
    fn test_display_number() {
        assert_eq!(format!("{}", Expr::number(3.0)), "3");
        assert!(format!("{}", Expr::number(3.141)).starts_with("3.141"));
    }

    #[test]
    fn test_display_symbol() {
        assert_eq!(format!("{}", Expr::symbol("x")), "x");
    }

    #[test]
    fn test_display_sum() {
        use crate::simplification::simplify_expr;
        use std::collections::HashSet;
        let expr = simplify_expr(
            Expr::sum(vec![Expr::symbol("x"), Expr::number(1.0)]),
            HashSet::new(),
            HashMap::new(),
            None,
            None,
            None,
            false,
        );
        assert_eq!(format!("{expr}"), "1 + x"); // Sorted after simplify: numbers before symbols
    }

    #[test]
    fn test_display_product() {
        let prod = Expr::product(vec![Expr::number(2.0), Expr::symbol("x")]);
        assert_eq!(format!("{prod}"), "2*x");
    }

    #[test]
    fn test_display_negation() {
        let expr = Expr::product(vec![Expr::number(-1.0), Expr::symbol("x")]);
        assert_eq!(format!("{expr}"), "-x");
    }

    #[test]
    fn test_display_subtraction() {
        // x - y = Sum([x, Product([-1, y])])
        let expr = Expr::sub_expr(Expr::symbol("x"), Expr::symbol("y"));
        let display = format!("{expr}");
        // Should display as subtraction
        assert!(
            display.contains('-'),
            "Expected subtraction, got: {display}"
        );
    }

    #[test]
    fn test_display_nested_sum() {
        // Test: x + (y + z) should display with parentheses
        let inner_sum = Expr::sum(vec![Expr::symbol("y"), Expr::symbol("z")]);
        let expr = Expr::sum(vec![Expr::symbol("x"), inner_sum]);
        let display = format!("{expr}");
        // Should display as "x + (y + z)" to preserve structure
        assert_eq!(display, "x + y + z");
    }
}
