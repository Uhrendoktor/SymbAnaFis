use crate::Expr;
use crate::ExprKind as AstKind;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::sync::Arc;

// ===== Identity Rules (Priority 100) =====

rule!(
    AddZeroRule,
    "add_zero",
    100,
    Numeric,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Add(u, v) = &expr.kind {
            if matches!(&u.kind, AstKind::Number(n) if *n == 0.0) {
                return Some((**v).clone());
            }
            if matches!(&v.kind, AstKind::Number(n) if *n == 0.0) {
                return Some((**u).clone());
            }
        }
        None
    }
);

rule!(
    SubZeroRule,
    "sub_zero",
    100,
    Numeric,
    &[ExprKind::Sub],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Sub(u, v) = &expr.kind
            && matches!(&v.kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some((**u).clone());
        }
        None
    }
);

rule!(
    MulZeroRule,
    "mul_zero",
    100,
    Numeric,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(u, v) = &expr.kind {
            if matches!(&u.kind, AstKind::Number(n) if *n == 0.0) {
                return Some(Expr::number(0.0));
            }
            if matches!(&v.kind, AstKind::Number(n) if *n == 0.0) {
                return Some(Expr::number(0.0));
            }
        }
        None
    }
);

rule!(
    MulOneRule,
    "mul_one",
    100,
    Numeric,
    &[ExprKind::Mul],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Mul(u, v) = &expr.kind {
            if matches!(&u.kind, AstKind::Number(n) if *n == 1.0) {
                return Some((**v).clone());
            }
            if matches!(&v.kind, AstKind::Number(n) if *n == 1.0) {
                return Some((**u).clone());
            }
        }
        None
    }
);

rule!(
    DivOneRule,
    "div_one",
    100,
    Numeric,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(u, v) = &expr.kind
            && matches!(&v.kind, AstKind::Number(n) if *n == 1.0)
        {
            return Some((**u).clone());
        }
        None
    }
);

rule!(
    ZeroDivRule,
    "zero_div",
    100,
    Numeric,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(u, _v) = &expr.kind
            && matches!(&u.kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    PowZeroRule,
    "pow_zero",
    100,
    Numeric,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(_u, v) = &expr.kind
            && matches!(&v.kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Expr::number(1.0));
        }
        None
    }
);

rule!(
    PowOneRule,
    "pow_one",
    100,
    Numeric,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(u, v) = &expr.kind
            && matches!(&v.kind, AstKind::Number(n) if *n == 1.0)
        {
            return Some((**u).clone());
        }
        None
    }
);

rule!(
    ZeroPowRule,
    "zero_pow",
    100,
    Numeric,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(u, _v) = &expr.kind
            && matches!(&u.kind, AstKind::Number(n) if *n == 0.0)
        {
            return Some(Expr::number(0.0));
        }
        None
    }
);

rule!(
    OnePowRule,
    "one_pow",
    100,
    Numeric,
    &[ExprKind::Pow],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Pow(u, _v) = &expr.kind
            && matches!(&u.kind, AstKind::Number(n) if *n == 1.0)
        {
            return Some(Expr::number(1.0));
        }
        None
    }
);

// ===== Normalization Rule (Priority 95) =====

rule!(
    NormalizeSignDivRule,
    "normalize_sign_div",
    95,
    Numeric,
    &[ExprKind::Div],
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(num, den) = &expr.kind {
            // Check if denominator is negative number
            if let AstKind::Number(d) = &den.kind
                && *d < 0.0
            {
                // x / -y -> -x / y
                return Some(Expr::div_expr(
                    Expr::mul_expr(Expr::number(-1.0), num.as_ref().clone()),
                    Expr::number(-*d),
                ));
            }

            // Check if denominator is (-1 * something)
            if let AstKind::Mul(c, rest) = &den.kind
                && matches!(&c.kind, AstKind::Number(n) if *n == -1.0)
            {
                // x / (-1 * y) -> -x / y
                return Some(Expr::div_expr(
                    Expr::mul_expr(Expr::number(-1.0), num.as_ref().clone()),
                    rest.as_ref().clone(),
                ));
            }

            // Check if denominator is (something * -1)
            if let AstKind::Mul(rest, c) = &den.kind
                && matches!(&c.kind, AstKind::Number(n) if *n == -1.0)
            {
                // x / (y * -1) -> -x / y
                return Some(Expr::div_expr(
                    Expr::mul_expr(Expr::number(-1.0), num.as_ref().clone()),
                    rest.as_ref().clone(),
                ));
            }
        }
        None
    }
);

// ===== Compaction Rules (Priority 90, 80) =====

rule!(
    ConstantFoldRule,
    "constant_fold",
    90,
    Numeric,
    &[
        ExprKind::Add,
        ExprKind::Sub,
        ExprKind::Mul,
        ExprKind::Div,
        ExprKind::Pow
    ],
    |expr: &Expr, _context: &RuleContext| {
        match &expr.kind {
            AstKind::Add(u, v) => {
                if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind) {
                    let result = a + b;
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::number(result));
                    }
                }
                // Handle Div(Number, Number) + Number
                if let (AstKind::Div(num, den), AstKind::Number(b)) = (&u.kind, &v.kind)
                    && let (AstKind::Number(num_val), AstKind::Number(den_val)) =
                        (&num.kind, &den.kind)
                    && *den_val != 0.0
                {
                    let new_num = num_val + b * den_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                    return Some(Expr::div_expr(
                        Expr::number(new_num),
                        Expr::number(*den_val),
                    ));
                }
                // Handle Number + Div(Number, Number)
                if let (AstKind::Number(a), AstKind::Div(num, den)) = (&u.kind, &v.kind)
                    && let (AstKind::Number(num_val), AstKind::Number(den_val)) =
                        (&num.kind, &den.kind)
                    && *den_val != 0.0
                {
                    let new_num = a * den_val + num_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                    return Some(Expr::div_expr(
                        Expr::number(new_num),
                        Expr::number(*den_val),
                    ));
                }
            }
            AstKind::Sub(u, v) => {
                if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind) {
                    let result = a - b;
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::number(result));
                    }
                }
                // Handle Div(Number, Number) - Number
                if let (AstKind::Div(num, den), AstKind::Number(b)) = (&u.kind, &v.kind)
                    && let (AstKind::Number(num_val), AstKind::Number(den_val)) =
                        (&num.kind, &den.kind)
                    && *den_val != 0.0
                {
                    let new_num = num_val - b * den_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                    return Some(Expr::div_expr(
                        Expr::number(new_num),
                        Expr::number(*den_val),
                    ));
                }
                // Handle Number - Div(Number, Number)
                if let (AstKind::Number(a), AstKind::Div(num, den)) = (&u.kind, &v.kind)
                    && let (AstKind::Number(num_val), AstKind::Number(den_val)) =
                        (&num.kind, &den.kind)
                    && *den_val != 0.0
                {
                    let new_num = a * den_val - num_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                    return Some(Expr::div_expr(
                        Expr::number(new_num),
                        Expr::number(*den_val),
                    ));
                }
            }
            AstKind::Mul(u, v) => {
                if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind) {
                    let result = a * b;
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::number(result));
                    }
                }
                // Flatten and combine multiple numbers
                let factors = crate::simplification::helpers::flatten_mul(expr);
                let mut numbers: Vec<f64> = Vec::new();
                let mut non_numbers: Vec<Expr> = Vec::new();

                for factor in &factors {
                    if let AstKind::Number(n) = &factor.kind {
                        numbers.push(*n);
                    } else {
                        non_numbers.push(factor.clone());
                    }
                }

                if numbers.len() >= 2 {
                    let combined: f64 = numbers.iter().product();
                    if !combined.is_nan() && !combined.is_infinite() {
                        let mut result_factors = vec![Expr::number(combined)];
                        result_factors.extend(non_numbers);
                        return Some(crate::simplification::helpers::rebuild_mul(result_factors));
                    }
                }

                // Mul(Number, Div(Number, Number))
                if let (AstKind::Number(a), AstKind::Div(b, c)) = (&u.kind, &v.kind)
                    && let (AstKind::Number(b_val), AstKind::Number(c_val)) = (&b.kind, &c.kind)
                    && *c_val != 0.0
                {
                    let result = (a * b_val) / c_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                    return Some(Expr::div_expr(
                        Expr::number(a * b_val),
                        Expr::number(*c_val),
                    ));
                }
                // Mul(Div(Number, Number), Number)
                if let (AstKind::Div(b, c), AstKind::Number(a)) = (&u.kind, &v.kind)
                    && let (AstKind::Number(b_val), AstKind::Number(c_val)) = (&b.kind, &c.kind)
                    && *c_val != 0.0
                {
                    let result = (a * b_val) / c_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                    return Some(Expr::div_expr(
                        Expr::number(a * b_val),
                        Expr::number(*c_val),
                    ));
                }
            }
            AstKind::Div(u, v) => {
                if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind)
                    && *b != 0.0
                {
                    let result = a / b;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::number(result.round()));
                    }
                }
            }
            AstKind::Pow(u, v) => {
                if let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind) {
                    let result = a.powf(*b);
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::number(result));
                    }
                }
            }
            _ => {}
        }
        None
    }
);

rule_with_helpers!(FractionSimplifyRule, "fraction_simplify", 80, Numeric, &[ExprKind::Div],
    helpers: {
        fn gcd(mut a: i64, mut b: i64) -> i64 {
            while b != 0 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a
        }
    },
    |expr: &Expr, _context: &RuleContext| {
        if let AstKind::Div(u, v) = &expr.kind
            && let (AstKind::Number(a), AstKind::Number(b)) = (&u.kind, &v.kind)
            && *b != 0.0
        {
            let is_int_a = a.fract() == 0.0;
            let is_int_b = b.fract() == 0.0;

            if is_int_a && is_int_b {
                if a % b == 0.0 {
                    return Some(Expr::number(a / b));
                } else {
                    let a_int = *a as i64;
                    let b_int = *b as i64;
                    let common = gcd(a_int.abs(), b_int.abs());

                    if common > 1 {
                        return Some(Expr::div_expr(
                            Expr::number((a_int / common) as f64),
                            Expr::number((b_int / common) as f64),
                        ));
                    }
                }
            }
        }
        None
    }
);

/// Get all numeric rules in priority order
pub(crate) fn get_numeric_rules() -> Vec<Arc<dyn Rule + Send + Sync>> {
    vec![
        Arc::new(AddZeroRule),
        Arc::new(SubZeroRule),
        Arc::new(MulZeroRule),
        Arc::new(MulOneRule),
        Arc::new(DivOneRule),
        Arc::new(ZeroDivRule),
        Arc::new(PowZeroRule),
        Arc::new(PowOneRule),
        Arc::new(ZeroPowRule),
        Arc::new(OnePowRule),
        Arc::new(NormalizeSignDivRule),
        Arc::new(ConstantFoldRule),
        Arc::new(FractionSimplifyRule),
    ]
}
