use crate::Expr;
use std::rc::Rc;

// Extracts (name, arg) for pow of function: name(arg)^power
pub(crate) fn get_fn_pow_named(expr: &Expr, power: f64) -> Option<(&str, Expr)> {
    if let Expr::Pow(base, exp) = expr
        && matches!(**exp, Expr::Number(n) if n == power)
        && let Expr::FunctionCall { name, args } = &**base
        && args.len() == 1
    {
        return Some((name.as_str(), args[0].clone()));
    }
    None
}

// Generic helper to extract arguments from product of two function calls, order-insensitive
pub(crate) fn get_product_fn_args(expr: &Expr, fname1: &str, fname2: &str) -> Option<(Expr, Expr)> {
    if let Expr::Mul(lhs, rhs) = expr
        && let (
            Expr::FunctionCall { name: n1, args: a1 },
            Expr::FunctionCall { name: n2, args: a2 },
        ) = (&**lhs, &**rhs)
        && a1.len() == 1
        && a2.len() == 1
    {
        if n1 == fname1 && n2 == fname2 {
            return Some((a1[0].clone(), a2[0].clone()));
        }
        if n1 == fname2 && n2 == fname1 {
            return Some((a2[0].clone(), a1[0].clone()));
        }
    }
    None
}

// Floating point approx equality used for numeric pattern matching
pub(crate) fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

// Get numeric value from expression if it's a Number
pub(crate) fn get_numeric_value(expr: &Expr) -> f64 {
    if let Expr::Number(n) = expr {
        *n
    } else {
        f64::NAN
    }
}

// Trigonometric helpers
use std::f64::consts::PI;
pub(crate) fn is_multiple_of_two_pi(expr: &Expr) -> bool {
    if let Expr::Number(n) = expr {
        let two_pi = 2.0 * PI;
        let k = n / two_pi;
        return approx_eq(k, k.round());
    }
    // Handle n * pi
    if let Expr::Mul(lhs, rhs) = expr {
        if let (Expr::Number(n), Expr::Symbol(s)) = (&**lhs, &**rhs)
            && s == "pi"
            && n % 2.0 == 0.0
        {
            return true;
        }
        if let (Expr::Symbol(s), Expr::Number(n)) = (&**lhs, &**rhs)
            && s == "pi"
            && n % 2.0 == 0.0
        {
            return true;
        }
    }
    false
}

pub(crate) fn is_pi(expr: &Expr) -> bool {
    if let Expr::Number(n) = expr {
        return (n - PI).abs() < 1e-10;
    }
    false
}

pub(crate) fn is_three_pi_over_two(expr: &Expr) -> bool {
    if let Expr::Number(n) = expr {
        return (n - 3.0 * PI / 2.0).abs() < 1e-10;
    }
    false
}

/// Flatten nested multiplication into a list of factors
pub(crate) fn flatten_mul(expr: &Expr) -> Vec<Expr> {
    let mut factors = Vec::with_capacity(4); // Most expressions have few factors
    let mut stack = vec![expr.clone()];

    while let Some(current) = stack.pop() {
        if let Expr::Mul(a, b) = current {
            stack.push(b.as_ref().clone());
            stack.push(a.as_ref().clone());
        } else {
            factors.push(current);
        }
    }
    factors
}

/// Compare expressions for canonical polynomial ordering
/// For polynomial form like: ax^n + bx^(n-1) + ... + cx + d
///
/// For ADDITION terms (polynomial ordering):
/// - Higher degree terms first: x^2 before x before constants
/// - Then alphabetically by variable name
///
/// For MULTIPLICATION factors (coefficient ordering):
/// - Numbers (coefficients) first: 2*x not x*2  
/// - Then symbols alphabetically
/// - Then more complex expressions
pub(crate) fn compare_expr(a: &Expr, b: &Expr) -> std::cmp::Ordering {
    use crate::Expr::*;
    use std::cmp::Ordering;

    // Helper to get polynomial degree for addition term ordering
    fn get_poly_degree(e: &Expr) -> (i32, f64, String) {
        // Returns (type_priority, exponent/degree, variable_name)
        // Lower type_priority = comes first in sorted order for polynomial terms
        match e {
            Number(_) => (100, 0.0, String::new()), // Constants come last
            Symbol(s) => (50, -1.0, s.clone()), // Variables are degree 1, negative so they sort first
            Pow(base, exp) => {
                if let Symbol(s) = &**base {
                    if let Number(n) = &**exp {
                        (10, -*n, s.clone()) // Negative exponent so higher powers sort first
                    } else {
                        (20, -999.0, s.clone()) // Symbolic exponent - treat as very high degree
                    }
                } else {
                    (30, 0.0, String::new()) // Non-symbol base
                }
            }
            Mul(l, r) => {
                // For c*x^n or c*x, extract the variable part's degree
                let (type_l, exp_l, var_l) = get_poly_degree(l);
                let (type_r, exp_r, var_r) = get_poly_degree(r);
                // Use the term with the variable (non-constant) part
                if type_l < type_r {
                    (15, exp_l, var_l)
                } else if type_r < type_l {
                    (15, exp_r, var_r)
                } else {
                    // Both same type, use higher degree
                    if exp_l <= exp_r {
                        // <= because negative, so smaller is higher degree
                        (15, exp_l, var_l)
                    } else {
                        (15, exp_r, var_r)
                    }
                }
            }
            FunctionCall { name, .. } => (60, 0.0, name.clone()),
            Add(..) => (70, 0.0, String::new()),
            Sub(..) => (70, 0.0, String::new()),
            Div(..) => (40, 0.0, String::new()),
        }
    }

    let (type_a, exp_a, var_a) = get_poly_degree(a);
    let (type_b, exp_b, var_b) = get_poly_degree(b);

    // First compare by type priority (lower = comes first)
    match type_a.cmp(&type_b) {
        Ordering::Equal => {}
        ord => return ord,
    }

    // Then by exponent (for Pow and Mul with variables)
    match exp_a.partial_cmp(&exp_b).unwrap_or(Ordering::Equal) {
        Ordering::Equal => {}
        ord => return ord,
    }

    // Then alphabetically by variable name
    match var_a.cmp(&var_b) {
        Ordering::Equal => {}
        ord => return ord,
    }

    // For numbers, compare by value (smaller numbers first for coefficients)
    if let (Number(n1), Number(n2)) = (a, b) {
        return n1.partial_cmp(n2).unwrap_or(Ordering::Equal);
    }

    // For symbols, alphabetical order
    if let (Symbol(s1), Symbol(s2)) = (a, b) {
        return s1.cmp(s2);
    }

    Ordering::Equal
}

/// Compare expressions for multiplication factor ordering
/// Numbers (coefficients) come first, then symbols, then complex expressions
pub(crate) fn compare_mul_factors(a: &Expr, b: &Expr) -> std::cmp::Ordering {
    use crate::Expr::*;
    use std::cmp::Ordering;

    fn factor_priority(e: &Expr) -> i32 {
        match e {
            Number(_) => 0,  // Numbers first (coefficients)
            Symbol(_) => 10, // Then symbols
            Pow(base, _) => {
                if matches!(&**base, Symbol(_)) {
                    20
                } else {
                    30
                }
            }
            FunctionCall { .. } => 40,
            Mul(..) | Div(..) => 50,
            Add(..) | Sub(..) => 60,
        }
    }

    let prio_a = factor_priority(a);
    let prio_b = factor_priority(b);

    match prio_a.cmp(&prio_b) {
        Ordering::Equal => {
            // Within same priority, use compare_expr
            compare_expr(a, b)
        }
        ord => ord,
    }
}

/// Helper: Flatten nested additions (iterative to avoid stack overflow on deep trees)
/// Works with references to avoid deep cloning during traversal
pub(crate) fn flatten_add(expr: &Expr) -> Vec<Expr> {
    let mut terms = Vec::with_capacity(4);
    // Stack holds (&Expr, negate_flag) - no cloning during traversal
    let mut stack: Vec<(&Expr, bool)> = vec![(expr, false)];

    while let Some((current, negate)) = stack.pop() {
        match current {
            Expr::Add(l, r) => {
                stack.push((r.as_ref(), negate));
                stack.push((l.as_ref(), negate));
            }
            Expr::Sub(l, r) => {
                // a - b: left keeps sign, right gets flipped
                stack.push((r.as_ref(), !negate));
                stack.push((l.as_ref(), negate));
            }
            leaf => {
                // Only clone when we actually need the result
                if negate {
                    terms.push(negate_term_ref(leaf));
                } else {
                    terms.push(leaf.clone());
                }
            }
        }
    }
    terms
}

/// Helper: Negate a term (multiply by -1, or simplify if already negative)
/// Takes reference to avoid unnecessary cloning
fn negate_term_ref(expr: &Expr) -> Expr {
    match expr {
        Expr::Number(n) => Expr::Number(-n),
        Expr::Mul(l, r) => {
            // Check if already has -1 coefficient
            if let Expr::Number(n) = l.as_ref() {
                if *n == -1.0 {
                    return r.as_ref().clone(); // -1 * x becomes x when negated
                }
                return Expr::Mul(Rc::new(Expr::Number(-n)), r.clone());
            }
            if let Expr::Number(n) = r.as_ref() {
                if *n == -1.0 {
                    return l.as_ref().clone();
                }
                return Expr::Mul(l.clone(), Rc::new(Expr::Number(-n)));
            }
            Expr::Mul(Rc::new(Expr::Number(-1.0)), Rc::new(expr.clone()))
        }
        _ => Expr::Mul(Rc::new(Expr::Number(-1.0)), Rc::new(expr.clone())),
    }
}

/// Helper: Rebuild addition tree (left-associative)
pub(crate) fn rebuild_add(terms: Vec<Expr>) -> Expr {
    if terms.is_empty() {
        return Expr::Number(0.0);
    }
    let mut iter = terms.into_iter();
    let mut result = iter.next().unwrap();
    for term in iter {
        result = Expr::Add(Rc::new(result), Rc::new(term));
    }
    result
}

/// Helper: Rebuild multiplication tree
pub(crate) fn rebuild_mul(terms: Vec<Expr>) -> Expr {
    if terms.is_empty() {
        return Expr::Number(1.0);
    }
    let mut iter = terms.into_iter();
    let mut result = iter.next().unwrap();
    for term in iter {
        result = Expr::Mul(Rc::new(result), Rc::new(term));
    }
    result
}

/// Helper: Normalize expression by sorting factors in multiplication
pub(crate) fn normalize_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Mul(u, v) => {
            let mut factors = flatten_mul(&Expr::Mul(u, v));
            factors.sort_by(compare_expr);
            rebuild_mul(factors)
        }
        other => other,
    }
}

/// Helper to extract coefficient and base
/// Returns (coefficient, base_expr)
/// e.g. 2*x -> (2.0, x)
///      x   -> (1.0, x)
pub(crate) fn extract_coeff(expr: &Expr) -> (f64, Expr) {
    let flattened = flatten_mul(expr);
    let mut coeff = 1.0;
    let mut non_num = Vec::new();
    for term in flattened {
        if let Expr::Number(n) = term {
            coeff *= n;
        } else {
            non_num.push(term);
        }
    }
    let base = if non_num.is_empty() {
        Expr::Number(1.0)
    } else if non_num.len() == 1 {
        non_num[0].clone()
    } else {
        rebuild_mul(non_num)
    };
    (coeff, normalize_expr(base))
}

/// Convert fractional powers back to roots for display
/// x^(1/2) -> sqrt(x)
/// x^(1/3) -> cbrt(x)
pub(crate) fn prettify_roots(expr: Expr) -> Expr {
    match expr {
        Expr::Pow(base, exp) => {
            let base = prettify_roots(base.as_ref().clone());
            let exp = prettify_roots(exp.as_ref().clone());

            // x^(1/2) -> sqrt(x)
            if let Expr::Div(num, den) = &exp
                && matches!(**num, Expr::Number(n) if n == 1.0)
                && matches!(**den, Expr::Number(n) if n == 2.0)
            {
                return Expr::FunctionCall {
                    name: "sqrt".to_string(),
                    args: vec![base],
                };
            }
            // x^0.5 -> sqrt(x)
            if let Expr::Number(n) = &exp
                && (n - 0.5).abs() < 1e-10
            {
                return Expr::FunctionCall {
                    name: "sqrt".to_string(),
                    args: vec![base],
                };
            }

            // Note: x^-0.5 is NOT converted to 1/sqrt(x) because that would
            // interfere with fraction consolidation rules. The NegativeExponentToFractionRule
            // handles x^(-n) -> 1/x^n, then prettify_roots converts x^(1/2) -> sqrt(x).

            // x^(1/3) -> cbrt(x)
            if let Expr::Div(num, den) = &exp
                && matches!(**num, Expr::Number(n) if n == 1.0)
                && matches!(**den, Expr::Number(n) if n == 3.0)
            {
                return Expr::FunctionCall {
                    name: "cbrt".to_string(),
                    args: vec![base],
                };
            }

            Expr::Pow(Rc::new(base), Rc::new(exp))
        }
        // Recursively prettify subexpressions
        Expr::Add(u, v) => Expr::Add(
            Rc::new(prettify_roots(u.as_ref().clone())),
            Rc::new(prettify_roots(v.as_ref().clone())),
        ),
        Expr::Sub(u, v) => Expr::Sub(
            Rc::new(prettify_roots(u.as_ref().clone())),
            Rc::new(prettify_roots(v.as_ref().clone())),
        ),
        Expr::Mul(u, v) => Expr::Mul(
            Rc::new(prettify_roots(u.as_ref().clone())),
            Rc::new(prettify_roots(v.as_ref().clone())),
        ),
        Expr::Div(u, v) => Expr::Div(
            Rc::new(prettify_roots(u.as_ref().clone())),
            Rc::new(prettify_roots(v.as_ref().clone())),
        ),
        Expr::FunctionCall { name, args } => Expr::FunctionCall {
            name,
            args: args.into_iter().map(prettify_roots).collect(),
        },
        _ => expr,
    }
}

/// Check if an expression is known to be non-negative for all real values of its variables.
/// This is a conservative check - returns true only when we can prove non-negativity.
pub(crate) fn is_known_non_negative(expr: &Expr) -> bool {
    match expr {
        // Positive numbers
        Expr::Number(n) => *n >= 0.0,

        // x^2, x^4, x^6, ... are always non-negative
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = &**exp {
                // Even positive integer exponents
                *n > 0.0 && n.fract() == 0.0 && (*n as i64) % 2 == 0
            } else {
                false
            }
        }

        // abs(x) is always non-negative
        Expr::FunctionCall { name, args } if args.len() == 1 => {
            match name.as_str() {
                "abs" | "Abs" => true,
                // exp(x) is always positive
                "exp" => true,
                // cosh(x) >= 1 for all real x
                "cosh" => true,
                // sqrt, cbrt of non-negative is non-negative (but we can't always prove input is non-negative)
                "sqrt" => is_known_non_negative(&args[0]),
                _ => false,
            }
        }

        // Product of two non-negatives is non-negative
        Expr::Mul(a, b) => is_known_non_negative(a) && is_known_non_negative(b),

        // Sum of two non-negatives is non-negative
        Expr::Add(a, b) => is_known_non_negative(a) && is_known_non_negative(b),

        // Division of non-negative by positive is non-negative (but we can't easily check "positive")
        // Be conservative here
        _ => false,
    }
}

/// Check if an exponent represents a fractional power that requires non-negative base
/// (i.e., exponents like 1/2, 1/4, 3/2, etc. where denominator is even)
pub(crate) fn is_fractional_root_exponent(expr: &Expr) -> bool {
    match expr {
        // Direct fraction: 1/2, 1/4, 3/4, etc.
        Expr::Div(_, den) => {
            if let Expr::Number(d) = &**den {
                // Check if denominator is an even integer
                d.fract() == 0.0 && (*d as i64) % 2 == 0
            } else {
                // Can't determine, be conservative
                false
            }
        }
        // Decimal like 0.5
        Expr::Number(n) => {
            // Check if it's a fractional power (not an integer)
            // For 0.5, 0.25, 1.5, etc. - these involve even roots
            if n.fract() != 0.0 {
                // Check if it's k/2^n for some integers
                // Simple check: 0.5 = 1/2, 0.25 = 1/4, 0.75 = 3/4, etc.
                let doubled = *n * 2.0;
                doubled.fract() == 0.0 // If 2*n is integer, then n = k/2
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Simple GCD implementation for integers
pub(crate) fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.abs();
    let mut b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Check if an expression contains a specific factor
pub(crate) fn contains_factor(expr: &Expr, factor: &Expr) -> bool {
    match expr {
        Expr::Mul(_, _) => {
            let factors = flatten_mul(expr);
            factors.iter().any(|f| f == factor)
        }
        _ => expr == factor,
    }
}

/// Remove factors from an expression
pub(crate) fn remove_factors(expr: &Expr, factors_to_remove: &Expr) -> Expr {
    match expr {
        Expr::Mul(_, _) => {
            let expr_factors = flatten_mul(expr);
            let remove_factors = flatten_mul(factors_to_remove);

            let mut remaining_factors = Vec::new();
            for factor in expr_factors {
                let mut should_remove = false;
                for remove_factor in &remove_factors {
                    if factor == *remove_factor {
                        should_remove = true;
                        break;
                    }
                }
                if !should_remove {
                    remaining_factors.push(factor);
                }
            }

            if remaining_factors.is_empty() {
                Expr::Number(1.0)
            } else if remaining_factors.len() == 1 {
                remaining_factors.into_iter().next().unwrap()
            } else {
                rebuild_mul(remaining_factors)
            }
        }
        _ => {
            // If the expression is not a multiplication, check if it matches the factors to remove
            if expr == factors_to_remove {
                Expr::Number(1.0)
            } else {
                expr.clone()
            }
        }
    }
}

/// Get a signature for a term to group like terms
pub(crate) fn get_term_signature(expr: &Expr) -> String {
    match expr {
        // Numbers are "like terms" only with other numbers (can be combined)
        Expr::Number(_) => "number".to_string(),
        Expr::Symbol(s) => format!("symbol:{}", s),
        Expr::Mul(_, _) => {
            let factors = flatten_mul(expr);
            let mut sorted_factors: Vec<String> = factors
                .iter()
                .filter(|f| !matches!(f, Expr::Number(_))) // Skip coefficients
                .map(get_term_signature)
                .collect();
            sorted_factors.sort();
            format!("mul:{}", sorted_factors.join(","))
        }
        Expr::Pow(base, exp) => {
            // For powers, we need to include the exponent in the signature
            // x^2 and x^3 are NOT like terms
            // But 2*x^3 and 3*x^3 ARE like terms (same base, same exponent)
            let exp_sig = match &**exp {
                Expr::Number(n) => format!("{}", n),
                _ => get_term_signature(exp),
            };
            format!("pow:{}^{}", get_term_signature(base), exp_sig)
        }
        Expr::FunctionCall { name, args } => {
            let arg_sigs: Vec<String> = args.iter().map(get_term_signature).collect();
            format!("func:{}({})", name, arg_sigs.join(","))
        }
        _ => format!("other:{:?}", expr),
    }
}

/// Normalize an expression to canonical form for comparison purposes.
/// This converts between equivalent representations:
/// - Sub(a, b) <-> Add(a, Mul(-1, b))
/// - Mul(-1, Mul(-1, x)) -> x
///
/// This ensures that semantically equivalent expressions compare as equal.
pub(crate) fn normalize_for_comparison(expr: &Expr) -> Expr {
    match expr {
        // Normalize Sub(a, b) to Add(a, -b) for consistent comparison
        Expr::Sub(a, b) => {
            let norm_a = normalize_for_comparison(a);
            let norm_b = normalize_for_comparison(b);
            // Create the negated form and normalize it too
            let neg_b = if let Expr::Number(n) = &norm_b {
                Expr::Number(-n)
            } else {
                Expr::Mul(Rc::new(Expr::Number(-1.0)), Rc::new(norm_b))
            };
            Expr::Add(Rc::new(norm_a), Rc::new(neg_b))
        }
        // Normalize Add - recurse into children
        Expr::Add(a, b) => {
            let norm_a = normalize_for_comparison(a);
            let norm_b = normalize_for_comparison(b);
            Expr::Add(Rc::new(norm_a), Rc::new(norm_b))
        }
        // Normalize Mul - simplify numeric products
        Expr::Mul(a, b) => {
            let norm_a = normalize_for_comparison(a);
            let norm_b = normalize_for_comparison(b);
            // Simplify products of numbers: Mul(n1, n2) -> n1*n2
            if let (Expr::Number(n1), Expr::Number(n2)) = (&norm_a, &norm_b) {
                return Expr::Number(n1 * n2);
            }
            Expr::Mul(Rc::new(norm_a), Rc::new(norm_b))
        }
        Expr::Div(a, b) => {
            let norm_a = normalize_for_comparison(a);
            let norm_b = normalize_for_comparison(b);
            Expr::Div(Rc::new(norm_a), Rc::new(norm_b))
        }
        Expr::Pow(a, b) => {
            let norm_a = normalize_for_comparison(a);
            let norm_b = normalize_for_comparison(b);
            Expr::Pow(Rc::new(norm_a), Rc::new(norm_b))
        }
        Expr::FunctionCall { name, args } => {
            let norm_args: Vec<Expr> = args.iter().map(normalize_for_comparison).collect();
            Expr::FunctionCall {
                name: name.clone(),
                args: norm_args,
            }
        }
        // Leaves remain unchanged
        other => other.clone(),
    }
}

/// Check if two expressions are semantically equivalent (same after normalization)
pub(crate) fn exprs_equivalent(a: &Expr, b: &Expr) -> bool {
    normalize_for_comparison(a) == normalize_for_comparison(b)
}
