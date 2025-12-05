use crate::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};
use std::rc::Rc;

// ===== Identity Rules (Priority 100) =====

rule!(
    AddZeroRule,
    "add_zero",
    100,
    Numeric,
    &[ExprKind::Add],
    |expr: &Expr, _context: &RuleContext| {
        if let Expr::Add(u, v) = expr {
            if matches!(**u, Expr::Number(n) if n == 0.0) {
                return Some((**v).clone());
            }
            if matches!(**v, Expr::Number(n) if n == 0.0) {
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
        if let Expr::Sub(u, v) = expr
            && matches!(**v, Expr::Number(n) if n == 0.0)
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
        if let Expr::Mul(u, v) = expr {
            if matches!(**u, Expr::Number(n) if n == 0.0) {
                return Some(Expr::Number(0.0));
            }
            if matches!(**v, Expr::Number(n) if n == 0.0) {
                return Some(Expr::Number(0.0));
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
        if let Expr::Mul(u, v) = expr {
            if matches!(**u, Expr::Number(n) if n == 1.0) {
                return Some((**v).clone());
            }
            if matches!(**v, Expr::Number(n) if n == 1.0) {
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
        if let Expr::Div(u, v) = expr
            && matches!(**v, Expr::Number(n) if n == 1.0)
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
        if let Expr::Div(_u, _v) = expr
            && matches!(**_u, Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(0.0));
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
        if let Expr::Pow(_u, v) = expr
            && matches!(**v, Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(1.0));
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
        if let Expr::Pow(u, v) = expr
            && matches!(**v, Expr::Number(n) if n == 1.0)
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
        if let Expr::Pow(_u, _v) = expr
            && matches!(**_u, Expr::Number(n) if n == 0.0)
        {
            return Some(Expr::Number(0.0));
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
        if let Expr::Pow(_u, _v) = expr
            && matches!(**_u, Expr::Number(n) if n == 1.0)
        {
            return Some(Expr::Number(1.0));
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
        if let Expr::Div(num, den) = expr {
            // Check if denominator is negative number
            if let Expr::Number(d) = **den
                && d < 0.0
            {
                // x / -y -> -x / y
                return Some(Expr::Div(
                    Rc::new(Expr::Mul(Rc::new(Expr::Number(-1.0)), num.clone())),
                    Rc::new(Expr::Number(-d)),
                ));
            }

            // Check if denominator is (-1 * something)
            if let Expr::Mul(c, rest) = &**den
                && matches!(**c, Expr::Number(n) if n == -1.0)
            {
                // x / (-1 * y) -> -x / y
                return Some(Expr::Div(
                    Rc::new(Expr::Mul(Rc::new(Expr::Number(-1.0)), num.clone())),
                    rest.clone(),
                ));
            }

            // Check if denominator is (something * -1)
            if let Expr::Mul(rest, c) = &**den
                && matches!(**c, Expr::Number(n) if n == -1.0)
            {
                // x / (y * -1) -> -x / y
                return Some(Expr::Div(
                    Rc::new(Expr::Mul(Rc::new(Expr::Number(-1.0)), num.clone())),
                    rest.clone(),
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
        match expr {
            Expr::Add(u, v) => {
                if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v) {
                    let result = a + b;
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::Number(result));
                    }
                }
                // Handle Div(Number, Number) + Number
                if let (Expr::Div(num, den), Expr::Number(b)) = (&**u, &**v)
                    && let (Expr::Number(num_val), Expr::Number(den_val)) = (&**num, &**den)
                    && *den_val != 0.0
                {
                    let new_num = num_val + b * den_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                    return Some(Expr::Div(
                        Rc::new(Expr::Number(new_num)),
                        Rc::new(Expr::Number(*den_val)),
                    ));
                }
                // Handle Number + Div(Number, Number)
                if let (Expr::Number(a), Expr::Div(num, den)) = (&**u, &**v)
                    && let (Expr::Number(num_val), Expr::Number(den_val)) = (&**num, &**den)
                    && *den_val != 0.0
                {
                    let new_num = a * den_val + num_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                    return Some(Expr::Div(
                        Rc::new(Expr::Number(new_num)),
                        Rc::new(Expr::Number(*den_val)),
                    ));
                }
            }
            Expr::Sub(u, v) => {
                if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v) {
                    let result = a - b;
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::Number(result));
                    }
                }
                // Handle Div(Number, Number) - Number
                if let (Expr::Div(num, den), Expr::Number(b)) = (&**u, &**v)
                    && let (Expr::Number(num_val), Expr::Number(den_val)) = (&**num, &**den)
                    && *den_val != 0.0
                {
                    let new_num = num_val - b * den_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                    return Some(Expr::Div(
                        Rc::new(Expr::Number(new_num)),
                        Rc::new(Expr::Number(*den_val)),
                    ));
                }
                // Handle Number - Div(Number, Number)
                if let (Expr::Number(a), Expr::Div(num, den)) = (&**u, &**v)
                    && let (Expr::Number(num_val), Expr::Number(den_val)) = (&**num, &**den)
                    && *den_val != 0.0
                {
                    let new_num = a * den_val - num_val;
                    let result = new_num / den_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                    return Some(Expr::Div(
                        Rc::new(Expr::Number(new_num)),
                        Rc::new(Expr::Number(*den_val)),
                    ));
                }
            }
            Expr::Mul(u, v) => {
                if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v) {
                    let result = a * b;
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::Number(result));
                    }
                }
                // Flatten and combine multiple numbers
                let factors = crate::simplification::helpers::flatten_mul(expr);
                let mut numbers: Vec<f64> = Vec::new();
                let mut non_numbers: Vec<Expr> = Vec::new();

                for factor in &factors {
                    if let Expr::Number(n) = factor {
                        numbers.push(*n);
                    } else {
                        non_numbers.push(factor.clone());
                    }
                }

                if numbers.len() >= 2 {
                    let combined: f64 = numbers.iter().product();
                    if !combined.is_nan() && !combined.is_infinite() {
                        let mut result_factors = vec![Expr::Number(combined)];
                        result_factors.extend(non_numbers);
                        return Some(crate::simplification::helpers::rebuild_mul(result_factors));
                    }
                }

                // Mul(Number, Div(Number, Number))
                if let (Expr::Number(a), Expr::Div(b, c)) = (&**u, &**v)
                    && let (Expr::Number(b_val), Expr::Number(c_val)) = (&**b, &**c)
                    && *c_val != 0.0
                {
                    let result = (a * b_val) / c_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                    return Some(Expr::Div(
                        Rc::new(Expr::Number(a * b_val)),
                        Rc::new(Expr::Number(*c_val)),
                    ));
                }
                // Mul(Div(Number, Number), Number)
                if let (Expr::Div(b, c), Expr::Number(a)) = (&**u, &**v)
                    && let (Expr::Number(b_val), Expr::Number(c_val)) = (&**b, &**c)
                    && *c_val != 0.0
                {
                    let result = (a * b_val) / c_val;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                    return Some(Expr::Div(
                        Rc::new(Expr::Number(a * b_val)),
                        Rc::new(Expr::Number(*c_val)),
                    ));
                }
            }
            Expr::Div(u, v) => {
                if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v)
                    && *b != 0.0
                {
                    let result = a / b;
                    if (result - result.round()).abs() < 1e-10 {
                        return Some(Expr::Number(result.round()));
                    }
                }
            }
            Expr::Pow(u, v) => {
                if let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v) {
                    let result = a.powf(*b);
                    if !result.is_nan() && !result.is_infinite() {
                        return Some(Expr::Number(result));
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
        if let Expr::Div(u, v) = expr
            && let (Expr::Number(a), Expr::Number(b)) = (&**u, &**v)
            && *b != 0.0
        {
            let is_int_a = a.fract() == 0.0;
            let is_int_b = b.fract() == 0.0;

            if is_int_a && is_int_b {
                if a % b == 0.0 {
                    return Some(Expr::Number(a / b));
                } else {
                    let a_int = *a as i64;
                    let b_int = *b as i64;
                    let common = gcd(a_int.abs(), b_int.abs());

                    if common > 1 {
                        return Some(Expr::Div(
                            Rc::new(Expr::Number((a_int / common) as f64)),
                            Rc::new(Expr::Number((b_int / common) as f64)),
                        ));
                    }
                }
            }
        }
        None
    }
);

/// Get all numeric rules in priority order
pub(crate) fn get_numeric_rules() -> Vec<Rc<dyn Rule>> {
    vec![
        Rc::new(AddZeroRule),
        Rc::new(SubZeroRule),
        Rc::new(MulZeroRule),
        Rc::new(MulOneRule),
        Rc::new(DivOneRule),
        Rc::new(ZeroDivRule),
        Rc::new(PowZeroRule),
        Rc::new(PowOneRule),
        Rc::new(ZeroPowRule),
        Rc::new(OnePowRule),
        Rc::new(NormalizeSignDivRule),
        Rc::new(ConstantFoldRule),
        Rc::new(FractionSimplifyRule),
    ]
}
