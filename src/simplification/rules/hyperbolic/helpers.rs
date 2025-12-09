use crate::ast::{Expr, ExprKind as AstKind};

// ============================================================================
// SHARED HELPER FUNCTIONS FOR HYPERBOLIC PATTERN MATCHING
// ============================================================================

/// Represents an exponential term with its argument
/// e^x has argument x, e^(-x) has argument -x, 1/e^x has argument -x
#[derive(Debug, Clone)]
pub(crate) struct ExpTerm {
    pub arg: Expr,
}

impl ExpTerm {
    /// Try to extract an exponential term from various forms:
    /// - e^x -> ExpTerm { arg: x }
    /// - exp(x) -> ExpTerm { arg: x }  
    /// - 1/e^x -> ExpTerm { arg: -x }
    /// - 1/exp(x) -> ExpTerm { arg: -x }
    pub fn from_expr(expr: &Expr) -> Option<Self> {
        // Direct form: e^x or exp(x)
        if let Some(arg) = Self::get_direct_exp_arg(expr) {
            return Some(ExpTerm { arg });
        }

        // Reciprocal form: 1/e^x or 1/exp(x)
        if let AstKind::Div(num, den) = &expr.kind
            && let AstKind::Number(n) = &num.kind
            && *n == 1.0
            && let Some(arg) = Self::get_direct_exp_arg(den)
        {
            // 1/e^x = e^(-x)
            return Some(ExpTerm {
                arg: Self::negate(&arg),
            });
        }

        None
    }

    /// Get the argument from e^x or exp(x) directly (not handling 1/e^x)
    pub fn get_direct_exp_arg(expr: &Expr) -> Option<Expr> {
        match &expr.kind {
            AstKind::Pow(base, exp) => {
                if let AstKind::Symbol(b) = &base.kind
                    && b == "e"
                {
                    return Some((**exp).clone());
                }
                None
            }
            AstKind::FunctionCall { name, args } => {
                if name == "exp" && args.len() == 1 {
                    return Some(args[0].clone());
                }
                None
            }
            _ => None,
        }
    }

    /// Check if this argument is the negation of another
    pub fn is_negation_of(&self, other: &Expr) -> bool {
        Self::args_are_negations(&self.arg, other)
    }

    /// Check if two arguments are negations of each other: arg1 = -arg2
    pub fn args_are_negations(arg1: &Expr, arg2: &Expr) -> bool {
        // Check if arg1 = -1 * arg2
        if let AstKind::Mul(lhs, rhs) = &arg1.kind {
            if let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
                && **rhs == *arg2
            {
                return true;
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == -1.0
                && **lhs == *arg2
            {
                return true;
            }
        }
        // Check if arg2 = -1 * arg1
        if let AstKind::Mul(lhs, rhs) = &arg2.kind {
            if let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
                && **rhs == *arg1
            {
                return true;
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == -1.0
                && **lhs == *arg1
            {
                return true;
            }
        }
        false
    }

    /// Create the negation of an expression: x -> -1 * x
    pub fn negate(expr: &Expr) -> Expr {
        // If it's already a negation, return the inner part
        if let AstKind::Mul(lhs, rhs) = &expr.kind {
            if let AstKind::Number(n) = &lhs.kind
                && *n == -1.0
            {
                return (**rhs).clone();
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == -1.0
            {
                return (**lhs).clone();
            }
        }
        Expr::mul_expr(Expr::number(-1.0), expr.clone())
    }
}

/// Try to match the pattern (e^x + e^(-x)) for cosh detection
/// Returns Some(x) if pattern matches (always returns the positive argument)
pub(crate) fn match_cosh_pattern(u: &Expr, v: &Expr) -> Option<Expr> {
    let exp1 = ExpTerm::from_expr(u)?;
    let exp2 = ExpTerm::from_expr(v)?;

    // Check if exp2.arg = -exp1.arg, return the positive one
    if exp2.is_negation_of(&exp1.arg) {
        // exp1.arg is positive if exp2.arg is its negation
        return Some(get_positive_form(&exp1.arg));
    }
    // Check reverse: exp1.arg = -exp2.arg
    if exp1.is_negation_of(&exp2.arg) {
        return Some(get_positive_form(&exp2.arg));
    }
    None
}

/// Try to match the pattern (e^x - e^(-x)) for sinh detection
/// Returns Some(x) if pattern matches (always returns the positive argument)
pub(crate) fn match_sinh_pattern_sub(u: &Expr, v: &Expr) -> Option<Expr> {
    // u should be e^x, v should be e^(-x)
    let exp1 = ExpTerm::from_expr(u)?;
    let exp2 = ExpTerm::from_expr(v)?;

    // Check if exp2.arg = -exp1.arg (so u = e^x, v = e^(-x))
    if exp2.is_negation_of(&exp1.arg) {
        return Some(get_positive_form(&exp1.arg));
    }
    None
}

/// Get the positive form of an expression
/// If expr is -x (i.e., Mul(-1, x)), return x
/// Otherwise return expr as-is
pub(crate) fn get_positive_form(expr: &Expr) -> Expr {
    if let AstKind::Mul(lhs, rhs) = &expr.kind {
        if let AstKind::Number(n) = &lhs.kind
            && *n == -1.0
        {
            return (**rhs).clone();
        }
        if let AstKind::Number(n) = &rhs.kind
            && *n == -1.0
        {
            return (**lhs).clone();
        }
    }
    expr.clone()
}

/// Try to match alternative cosh pattern: (e^(2x) + 1) / (2 * e^x) = cosh(x)
/// Returns Some(x) if pattern matches
pub(crate) fn match_alt_cosh_pattern(numerator: &Expr, denominator: &Expr) -> Option<Expr> {
    // Denominator must be 2 * e^x
    let x = match_two_times_exp(denominator)?;

    // Numerator must be e^(2x) + 1
    if let AstKind::Add(u, v) = &numerator.kind {
        // Check for e^(2x) + 1 or 1 + e^(2x)
        let (exp_term, _) = if matches!(&v.kind, AstKind::Number(n) if *n == 1.0) {
            (u, v)
        } else if matches!(&u.kind, AstKind::Number(n) if *n == 1.0) {
            (v, u)
        } else {
            return None;
        };

        // Check exp_term = e^(2x)
        if let Some(exp_arg) = ExpTerm::get_direct_exp_arg(exp_term)
            && is_double_of(&exp_arg, &x)
        {
            return Some(x);
        }
    }
    None
}

/// Try to match alternative sinh pattern: (e^(2x) - 1) / (2 * e^x) = sinh(x)
/// Returns Some(x) if pattern matches
pub(crate) fn match_alt_sinh_pattern(numerator: &Expr, denominator: &Expr) -> Option<Expr> {
    // Denominator must be 2 * e^x
    let x = match_two_times_exp(denominator)?;

    // Numerator must be e^(2x) - 1
    if let AstKind::Sub(u, v) = &numerator.kind {
        // Check for e^(2x) - 1
        if matches!(&v.kind, AstKind::Number(n) if *n == 1.0)
            && let Some(exp_arg) = ExpTerm::get_direct_exp_arg(u)
            && is_double_of(&exp_arg, &x)
        {
            return Some(x);
        }
    }
    None
}

/// Match pattern: 2 * e^x or e^x * 2
/// Returns the argument x if pattern matches
pub(crate) fn match_two_times_exp(expr: &Expr) -> Option<Expr> {
    if let AstKind::Mul(lhs, rhs) = &expr.kind {
        // 2 * e^x
        if let AstKind::Number(n) = &lhs.kind
            && *n == 2.0
        {
            return ExpTerm::get_direct_exp_arg(rhs);
        }
        // e^x * 2
        if let AstKind::Number(n) = &rhs.kind
            && *n == 2.0
        {
            return ExpTerm::get_direct_exp_arg(lhs);
        }
    }
    None
}

/// Check if expr = 2 * other (i.e., expr is double of other)
pub(crate) fn is_double_of(expr: &Expr, other: &Expr) -> bool {
    if let AstKind::Mul(lhs, rhs) = &expr.kind {
        if let AstKind::Number(n) = &lhs.kind
            && *n == 2.0
            && **rhs == *other
        {
            return true;
        }
        if let AstKind::Number(n) = &rhs.kind
            && *n == 2.0
            && **lhs == *other
        {
            return true;
        }
    }
    false
}

/// Try to match alternative sech pattern: (2 * e^x) / (e^(2x) + 1) = sech(x)
/// This is the reciprocal of the alt_cosh form
/// Returns Some(x) if pattern matches
pub(crate) fn match_alt_sech_pattern(numerator: &Expr, denominator: &Expr) -> Option<Expr> {
    // Numerator must be 2 * e^x
    let x = match_two_times_exp(numerator)?;

    // Denominator must be e^(2x) + 1
    if let AstKind::Add(u, v) = &denominator.kind {
        // Check for e^(2x) + 1 or 1 + e^(2x)
        let exp_term = if matches!(&v.kind, AstKind::Number(n) if *n == 1.0) {
            u
        } else if matches!(&u.kind, AstKind::Number(n) if *n == 1.0) {
            v
        } else {
            return None;
        };

        // Check exp_term = e^(2x)
        if let Some(exp_arg) = ExpTerm::get_direct_exp_arg(exp_term)
            && is_double_of(&exp_arg, &x)
        {
            return Some(x);
        }
    }
    None
}

/// Try to match pattern: (e^x - 1/e^x) * e^x = e^(2x) - 1 (for sinh numerator in tanh)
/// Returns Some(x) if pattern matches
pub(crate) fn match_e2x_minus_1_factored(expr: &Expr) -> Option<Expr> {
    // Pattern: (e^x - 1/e^x) * e^x or e^x * (e^x - 1/e^x)
    if let AstKind::Mul(lhs, rhs) = &expr.kind {
        // Check both orderings
        if let Some(x) = try_match_factored_sinh_times_exp(lhs, rhs) {
            return Some(x);
        }
        if let Some(x) = try_match_factored_sinh_times_exp(rhs, lhs) {
            return Some(x);
        }
    }
    None
}

/// Helper: try to match (e^x - 1/e^x) * e^x
fn try_match_factored_sinh_times_exp(factor: &Expr, exp_part: &Expr) -> Option<Expr> {
    // exp_part should be e^x
    let x = ExpTerm::get_direct_exp_arg(exp_part)?;

    // factor should be (e^x - 1/e^x)
    if let AstKind::Sub(u, v) = &factor.kind {
        // u = e^x
        if let Some(arg_u) = ExpTerm::get_direct_exp_arg(u)
            && arg_u == x
        {
            // v = 1/e^x
            if let AstKind::Div(num, den) = &v.kind
                && matches!(&num.kind, AstKind::Number(n) if *n == 1.0)
                && let Some(arg_v) = ExpTerm::get_direct_exp_arg(den)
                && arg_v == x
            {
                return Some(x);
            }
        }
    }
    None
}

/// Match pattern: e^(2x) + 1 directly
/// Returns Some(x) if pattern matches
pub(crate) fn match_e2x_plus_1(expr: &Expr) -> Option<Expr> {
    if let AstKind::Add(u, v) = &expr.kind {
        // Check for e^(2x) + 1 or 1 + e^(2x)
        let (exp_term, _const_term) = if matches!(&v.kind, AstKind::Number(n) if *n == 1.0) {
            (u, v)
        } else if matches!(&u.kind, AstKind::Number(n) if *n == 1.0) {
            (v, u)
        } else {
            return None;
        };

        // Check exp_term = e^(2x)
        if let Some(exp_arg) = ExpTerm::get_direct_exp_arg(exp_term) {
            // exp_arg should be 2*x
            if let AstKind::Mul(lhs, rhs) = &exp_arg.kind
                && let AstKind::Number(n) = &lhs.kind
                && *n == 2.0
            {
                return Some((**rhs).clone());
            }
            if let AstKind::Mul(lhs, rhs) = &exp_arg.kind
                && let AstKind::Number(n) = &rhs.kind
                && *n == 2.0
            {
                return Some((**lhs).clone());
            }
        }
    }
    None
}

/// Match pattern: e^(2x) - 1 directly (not factored form)
/// Returns Some(x) if pattern matches
pub(crate) fn match_e2x_minus_1_direct(expr: &Expr) -> Option<Expr> {
    if let AstKind::Sub(u, v) = &expr.kind {
        // Check for e^(2x) - 1
        if matches!(&v.kind, AstKind::Number(n) if *n == 1.0) {
            // Check u = e^(2x)
            if let Some(exp_arg) = ExpTerm::get_direct_exp_arg(u) {
                // exp_arg should be 2*x
                if let AstKind::Mul(lhs, rhs) = &exp_arg.kind
                    && let AstKind::Number(n) = &lhs.kind
                    && *n == 2.0
                {
                    return Some((**rhs).clone());
                }
                if let AstKind::Mul(lhs, rhs) = &exp_arg.kind
                    && let AstKind::Number(n) = &rhs.kind
                    && *n == 2.0
                {
                    return Some((**lhs).clone());
                }
            }
        }
    }
    // Also check Add form where second term is -1: e^(2x) + (-1)
    if let AstKind::Add(u, v) = &expr.kind {
        // Check for e^(2x) + (-1) (i.e., -1 as a number)
        if matches!(&v.kind, AstKind::Number(n) if *n == -1.0)
            && let Some(exp_arg) = ExpTerm::get_direct_exp_arg(u)
            && let AstKind::Mul(lhs, rhs) = &exp_arg.kind
        {
            if let AstKind::Number(n) = &lhs.kind
                && *n == 2.0
            {
                return Some((**rhs).clone());
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == 2.0
            {
                return Some((**lhs).clone());
            }
        }
        // Check (-1) + e^(2x)
        if matches!(&u.kind, AstKind::Number(n) if *n == -1.0)
            && let Some(exp_arg) = ExpTerm::get_direct_exp_arg(v)
            && let AstKind::Mul(lhs, rhs) = &exp_arg.kind
        {
            if let AstKind::Number(n) = &lhs.kind
                && *n == 2.0
            {
                return Some((**rhs).clone());
            }
            if let AstKind::Number(n) = &rhs.kind
                && *n == 2.0
            {
                return Some((**lhs).clone());
            }
        }
    }
    None
}

/// Try to extract the inner expression from -1 * expr
pub(crate) fn extract_negated_term(expr: &Expr) -> Option<Expr> {
    if let AstKind::Mul(lhs, rhs) = &expr.kind {
        if let AstKind::Number(n) = &lhs.kind
            && *n == -1.0
        {
            return Some((**rhs).clone());
        }
        if let AstKind::Number(n) = &rhs.kind
            && *n == -1.0
        {
            return Some((**lhs).clone());
        }
    }
    None
}
