//! Comparison Example: Diff Output with/without Simplification
//!
//! This example compares the output of differentiation between:
//! 1. SymbAnaFis with simplification (default)
//! 2. SymbAnaFis without simplification (raw derivative)
//! 3. Symbolica (for reference)
//!
//! Run with: cargo run --example diff_comparison

use symb_anafis::Diff;
use symbolica::{atom::AtomCore, parse, symbol};

fn main() {
    // Load symbolica license from .env if present
    let _ = dotenvy::dotenv();

    println!("=== Derivative Output Comparison ===\n");

    // Test expressions of varying complexity
    let expressions = [
        ("x^2", "Simple polynomial"),
        ("sin(x)*cos(x)", "Trig product"),
        ("exp(x^2)", "Nested exponential"),
        ("ln(x^2 + 1)", "Log of sum"),
        ("x^3 + 2*x^2 - x + 1", "Cubic polynomial"),
        ("sin(x)/cos(x)", "Trig fraction (tan)"),
        ("(x^2 + 1)/(x - 1)", "Rational function"),
    ];

    let x_sym = symbol!("x");

    for (expr, description) in expressions {
        println!("────────────────────────────────────────");
        println!("Expression: {} ({})", expr, description);
        println!("────────────────────────────────────────");

        // SymbAnaFis WITHOUT simplification (raw derivative)
        let raw = Diff::new()
            .skip_simplification(true)
            .diff_str(expr, "x")
            .unwrap();
        println!("Raw (no simplify):  {}", raw);

        // SymbAnaFis WITH simplification (default)
        let simplified = Diff::new().diff_str(expr, "x").unwrap();
        println!("Simplified:         {}", simplified);

        // Symbolica
        let atom = parse!(expr);
        let symbolica_result = atom.derivative(x_sym);
        println!("Symbolica:          {}", symbolica_result);

        // Show size comparison
        println!(
            "Size reduction:     {} chars -> {} chars ({:.1}%)",
            raw.len(),
            simplified.len(),
            100.0 * (1.0 - simplified.len() as f64 / raw.len() as f64)
        );
        println!();
    }

    // Large expression example
    println!("=== Large Expression (10 terms) ===\n");
    let large_expr = "x^10 + sin(x)*cos(x) + exp(x)/x + ln(x^2) + x^5 - 3*x^3 + tan(x) + sqrt(x^2+1) + x/(x+1) + sin(x^2)";

    println!("Input: {}\n", large_expr);

    let raw = Diff::new()
        .skip_simplification(true)
        .diff_str(large_expr, "x")
        .unwrap();

    let simplified = Diff::new().diff_str(large_expr, "x").unwrap();

    let atom = parse!(large_expr);
    let symbolica_result = atom.derivative(x_sym);

    println!("Raw derivative ({} chars):\n{}\n", raw.len(), raw);
    println!("Simplified ({} chars):\n{}\n", simplified.len(), simplified);
    println!(
        "Symbolica ({} chars):\n{}",
        format!("{}", symbolica_result).len(),
        symbolica_result
    );
}
