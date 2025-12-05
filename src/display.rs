// Display formatting for AST
use crate::Expr;
use std::fmt;
use std::rc::Rc;

/// Check if an expression is negative (has a leading -1 coefficient)
/// Returns Some(inner) if the expression is -1 * inner, None otherwise
fn extract_negative(expr: &Expr) -> Option<Expr> {
    if let Expr::Mul(left, right) = expr {
        // Direct -1 * x pattern
        if let Expr::Number(n) = &**left
            && *n == -1.0
        {
            return Some((**right).clone());
        }
        // Nested: (-1 * a) * b = -(a * b)
        if let Some(inner_left) = extract_negative(left) {
            return Some(Expr::Mul(Rc::new(inner_left), right.clone()));
        }
    }
    None
}

/// Check if an expression starts with a function call or exp() (for nested muls)
fn starts_with_function_or_exp(expr: &Expr) -> bool {
    match expr {
        Expr::FunctionCall { .. } => true,
        // e^x displays as exp(x), so treat it like a function
        Expr::Pow(base, _) => {
            if let Expr::Symbol(s) = &**base {
                s == "e"
            } else {
                false
            }
        }
        Expr::Mul(left, _) => starts_with_function_or_exp(left),
        _ => false,
    }
}

/// Check if we need explicit * between two expressions
fn needs_explicit_mul(left: &Expr, right: &Expr) -> bool {
    // Number * Number always needs explicit *
    if matches!(left, Expr::Number(_)) && matches!(right, Expr::Number(_)) {
        return true;
    }
    // Number * (something that starts with number) needs *
    if matches!(left, Expr::Number(_)) {
        if let Expr::Mul(inner_left, _) = right
            && matches!(**inner_left, Expr::Number(_))
        {
            return true;
        }
        if matches!(right, Expr::Div(_, _)) {
            return true;
        }
    }
    // Symbol * Symbol needs explicit * for readability: alpha*t not alphat
    if matches!(left, Expr::Symbol(_)) && matches!(right, Expr::Symbol(_)) {
        return true;
    }
    // Symbol * Number needs explicit *: x*2 not x2
    if matches!(left, Expr::Symbol(_)) && matches!(right, Expr::Number(_)) {
        return true;
    }
    // Symbol * Pow needs explicit *: pi*sigma^2 not pisigma^2
    if matches!(left, Expr::Symbol(_)) && matches!(right, Expr::Pow(_, _)) {
        return true;
    }
    // Symbol/Pow * FunctionCall or exp() (or mul starting with function/exp) needs explicit *
    if matches!(left, Expr::Symbol(_) | Expr::Pow(_, _))
        && (matches!(right, Expr::FunctionCall { .. }) || starts_with_function_or_exp(right))
    {
        return true;
    }
    // Pow * Symbol needs explicit *: x^2*y not x^2y
    if matches!(left, Expr::Pow(_, _)) && matches!(right, Expr::Symbol(_)) {
        return true;
    }
    // FunctionCall * anything needs explicit *
    if matches!(left, Expr::FunctionCall { .. }) {
        return true;
    }
    // Nested mul ending in symbol/pow * symbol, pow, or function call or exp
    if let Expr::Mul(_, inner_right) = left {
        if matches!(**inner_right, Expr::Symbol(_) | Expr::Pow(_, _))
            && (matches!(
                right,
                Expr::Symbol(_) | Expr::Pow(_, _) | Expr::FunctionCall { .. }
            ) || starts_with_function_or_exp(right))
        {
            return true;
        }
        if matches!(**inner_right, Expr::FunctionCall { .. }) {
            return true;
        }
    }
    false
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => {
                if n.is_nan() {
                    write!(f, "NaN")
                } else if n.is_infinite() {
                    if *n > 0.0 {
                        write!(f, "Infinity")
                    } else {
                        write!(f, "-Infinity")
                    }
                } else if n.fract() == 0.0 && n.abs() < 1e10 {
                    // Display as integer if no fractional part
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }

            Expr::Symbol(s) => write!(f, "{}", s),

            Expr::FunctionCall { name, args } => {
                if args.is_empty() {
                    write!(f, "{}()", name)
                } else {
                    let args_str: Vec<String> = args.iter().map(|arg| format!("{}", arg)).collect();
                    write!(f, "{}({})", name, args_str.join(", "))
                }
            }

            Expr::Add(u, v) => {
                // Check if v is a negative term to display as subtraction
                if let Some(positive_v) = extract_negative(v) {
                    let inner_str = format_mul_operand(&positive_v);
                    write!(f, "{} - {}", u, inner_str)
                } else {
                    write!(f, "{} + {}", u, v)
                }
            }

            Expr::Sub(u, v) => {
                // Parenthesize RHS when it's an addition or subtraction to preserve
                // the intended grouping: `a - (b + c)` instead of `a - b + c`.
                let right_str = match &**v {
                    Expr::Add(_, _) | Expr::Sub(_, _) => format!("({})", v),
                    _ => format!("{}", v),
                };
                write!(f, "{} - {}", u, right_str)
            }

            Expr::Mul(u, v) => {
                if let Expr::Number(n) = **u {
                    if n == -1.0 {
                        write!(f, "-{}", format_mul_operand(v))
                    } else if needs_explicit_mul(u, v) {
                        // Need explicit * between numbers or number and division
                        write!(f, "{}*{}", format_mul_operand(u), format_mul_operand(v))
                    } else {
                        // Compact form: 2x, 3sin(x), etc.
                        write!(f, "{}{}", format_mul_operand(u), format_mul_operand(v))
                    }
                } else if needs_explicit_mul(u, v) {
                    write!(f, "{}*{}", format_mul_operand(u), format_mul_operand(v))
                } else {
                    // symbol * symbol or similar - use compact form
                    write!(f, "{}{}", format_mul_operand(u), format_mul_operand(v))
                }
            }

            Expr::Div(u, v) => {
                let num_str = format!("{}", u);
                let denom_str = format!("{}", v);
                // Add parentheses around numerator if it's addition or subtraction
                let formatted_num = match **u {
                    Expr::Add(_, _) | Expr::Sub(_, _) => format!("({})", num_str),
                    _ => num_str,
                };
                // Add parentheses around denominator if it's not a simple identifier, number, power, or function
                let formatted_denom = match **v {
                    Expr::Symbol(_)
                    | Expr::Number(_)
                    | Expr::Pow(_, _)
                    | Expr::FunctionCall { .. } => denom_str,
                    _ => format!("({})", denom_str),
                };
                write!(f, "{}/{}", formatted_num, formatted_denom)
            }

            Expr::Pow(u, v) => {
                // Special case: e^x displays as exp(x)
                if let Expr::Symbol(s) = &**u
                    && s == "e"
                {
                    write!(f, "exp({})", v)
                } else {
                    let base_str = format!("{}", u);
                    let exp_str = format!("{}", v);

                    // Add parentheses around base if it's not a simple expression
                    // CRITICAL: Mul and Div MUST be parenthesized to avoid ambiguity
                    // (C * R)^2 should display as "(C * R)^2", not "C * R^2"
                    let formatted_base = match **u {
                        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _) => {
                            format!("({})", base_str)
                        }
                        _ => base_str,
                    };

                    // Add parentheses around exponent if it's not a simple number or symbol
                    let formatted_exp = match **v {
                        Expr::Number(_) | Expr::Symbol(_) => exp_str,
                        _ => format!("({})", exp_str),
                    };

                    write!(f, "{}^{}", formatted_base, formatted_exp)
                }
            }
        }
    }
}

/// Format operand for multiplication to minimize parentheses
fn format_mul_operand(expr: &Expr) -> String {
    match expr {
        Expr::Add(_, _) | Expr::Sub(_, _) => format!("({})", expr),
        _ => format!("{}", expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn test_display_number() {
        let expr = Expr::Number(3.0);
        assert_eq!(format!("{}", expr), "3");

        let expr = Expr::Number(314.0 / 100.0);
        // Formatting may use full precision; ensure it starts with the expected prefix
        assert!(format!("{}", expr).starts_with("3.14"));
    }

    #[test]
    fn test_display_symbol() {
        let expr = Expr::Symbol("x".to_string());
        assert_eq!(format!("{}", expr), "x");
    }

    #[test]
    fn test_display_addition() {
        let expr = Expr::Add(
            Rc::new(Expr::Symbol("x".to_string())),
            Rc::new(Expr::Number(1.0)),
        );
        assert_eq!(format!("{}", expr), "x + 1");
    }

    #[test]
    fn test_display_multiplication() {
        let expr = Expr::Mul(
            Rc::new(Expr::Symbol("x".to_string())),
            Rc::new(Expr::Number(2.0)),
        );
        assert_eq!(format!("{}", expr), "x*2");
    }

    #[test]
    fn test_display_function() {
        let expr = Expr::FunctionCall {
            name: "sin".to_string(),
            args: vec![Expr::Symbol("x".to_string())],
        };
        assert_eq!(format!("{}", expr), "sin(x)");
    }

    #[test]
    fn test_display_negative_term() {
        let expr = Expr::Mul(
            Rc::new(Expr::Number(-1.0)),
            Rc::new(Expr::Symbol("x".to_string())),
        );
        assert_eq!(format!("{}", expr), "-x");

        let expr2 = Expr::Mul(
            Rc::new(Expr::Number(-1.0)),
            Rc::new(Expr::FunctionCall {
                name: "sin".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
        );
        assert_eq!(format!("{}", expr2), "-sin(x)");
    }

    #[test]
    fn test_display_fraction_parens() {
        // 1 / x -> 1/x
        let expr = Expr::Div(
            Rc::new(Expr::Number(1.0)),
            Rc::new(Expr::Symbol("x".to_string())),
        );
        assert_eq!(format!("{}", expr), "1/x");

        // 1 / x^2 -> 1/x^2
        let expr = Expr::Div(
            Rc::new(Expr::Number(1.0)),
            Rc::new(Expr::Pow(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Number(2.0)),
            )),
        );
        assert_eq!(format!("{}", expr), "1/x^2");

        // 1 / sin(x) -> 1/sin(x)
        let expr = Expr::Div(
            Rc::new(Expr::Number(1.0)),
            Rc::new(Expr::FunctionCall {
                name: "sin".to_string(),
                args: vec![Expr::Symbol("x".to_string())],
            }),
        );
        assert_eq!(format!("{}", expr), "1/sin(x)");

        // 1 / (2 * x) -> 1/(2x)
        let expr = Expr::Div(
            Rc::new(Expr::Number(1.0)),
            Rc::new(Expr::Mul(
                Rc::new(Expr::Number(2.0)),
                Rc::new(Expr::Symbol("x".to_string())),
            )),
        );
        assert_eq!(format!("{}", expr), "1/(2x)");

        // 1 / (x + 1) -> 1/(x + 1)
        let expr = Expr::Div(
            Rc::new(Expr::Number(1.0)),
            Rc::new(Expr::Add(
                Rc::new(Expr::Symbol("x".to_string())),
                Rc::new(Expr::Number(1.0)),
            )),
        );
        assert_eq!(format!("{}", expr), "1/(x + 1)");
    }

    #[test]
    fn test_display_neg_x_exp() {
        // -1 * (x * exp(y)) should display as -x*exp(y)
        let exp_y = Expr::FunctionCall {
            name: "exp".to_string(),
            args: vec![Expr::Symbol("y".to_string())],
        };
        let x_times_exp = Expr::Mul(Rc::new(Expr::Symbol("x".to_string())), Rc::new(exp_y));
        let neg_x_times_exp = Expr::Mul(Rc::new(Expr::Number(-1.0)), Rc::new(x_times_exp));

        eprintln!("AST: {:?}", neg_x_times_exp);
        eprintln!("Display: {}", neg_x_times_exp);
        assert_eq!(format!("{}", neg_x_times_exp), "-x*exp(y)");

        // Alternative structure: (-1 * x) * exp(y)
        let neg_x = Expr::Mul(
            Rc::new(Expr::Number(-1.0)),
            Rc::new(Expr::Symbol("x".to_string())),
        );
        let neg_x_times_exp2 = Expr::Mul(
            Rc::new(neg_x),
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("y".to_string())],
            }),
        );

        eprintln!("AST2: {:?}", neg_x_times_exp2);
        eprintln!("Display2: {}", neg_x_times_exp2);
        assert_eq!(format!("{}", neg_x_times_exp2), "-x*exp(y)");

        // Test: -x * (exp(y) * z) - the exp is nested in a Mul on the right
        let exp_y_times_z = Expr::Mul(
            Rc::new(Expr::FunctionCall {
                name: "exp".to_string(),
                args: vec![Expr::Symbol("y".to_string())],
            }),
            Rc::new(Expr::Symbol("z".to_string())),
        );
        let neg_x_times_exp_z = Expr::Mul(
            Rc::new(Expr::Mul(
                Rc::new(Expr::Number(-1.0)),
                Rc::new(Expr::Symbol("x".to_string())),
            )),
            Rc::new(exp_y_times_z),
        );

        eprintln!("AST3: {:?}", neg_x_times_exp_z);
        eprintln!("Display3: {}", neg_x_times_exp_z);
    }

    #[test]
    fn test_display_quantum_derivative() {
        // Test actual quantum derivative: -xexp(-x^2/(2sigma^2))/(sigma^3*sqrt(pi))
        // The full expression from the example
        use crate::parser;
        use crate::simplification;
        use std::collections::HashSet;

        let fixed_set: HashSet<String> = ["sigma".to_string()].iter().cloned().collect();
        let custom_funcs: HashSet<String> = HashSet::new();

        let ast = parser::parse(
            "(exp(-x^2 / (4 * sigma^2)) / sqrt(sigma * sqrt(pi)))^2",
            &fixed_set,
            &custom_funcs,
        )
        .unwrap();

        let derivative = ast.derive("x", &fixed_set);
        let simplified = simplification::simplify_expr(derivative, fixed_set);

        eprintln!("Simplified AST: {:?}", simplified);
        eprintln!("Quantum derivative: {}", simplified);

        // Should contain x*exp not xexp
        let display = format!("{}", simplified);
        assert!(
            display.contains("x*exp") || display.contains("x * exp"),
            "Expected 'x*exp' but got: {}",
            display
        );
    }
}
