/// API Showcase: Complete SymbAnaFis Feature Demonstration
///
/// This example demonstrates ALL the capabilities of SymbAnaFis:
/// - String-based and Type-Safe APIs
/// - Differentiation (single variable, multi-variable)
/// - Simplification
/// - Numerical Evaluation
/// - Gradient, Hessian, and Jacobian
/// - All supported mathematical functions
/// - Custom derivatives
/// - Safety features and configuration
///
/// Run with: cargo run --example api_showcase
use std::collections::HashMap;
#[allow(unused_imports)]
use symb_anafis::{
    Diff, Expr, ExprKind, Simplify, Symbol, diff, evaluate_str, gradient, gradient_str, hessian,
    hessian_str, jacobian, jacobian_str, simplify, symb,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          SYMB ANAFIS: COMPLETE API SHOWCASE                      ║");
    println!("║          Symbolic Differentiation Library for Rust               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    part1_string_api();
    part2_type_safe_api();
    part3_numerical_evaluation();
    part4_multi_variable_calculus();
    part5_all_functions();
    part6_custom_derivatives();
    part7_safety_features();

    println!("\n✅ Showcase Complete!");
}

// =============================================================================
// PART 1: STRING-BASED API
// =============================================================================
fn part1_string_api() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📦 PART 1: STRING-BASED API");
    println!("   Best for parsing user input, configuration files, web APIs");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 1.1 Simple diff() function
    println!("  1.1 Simple Differentiation: diff()");
    let formula = "x^3 + 2*x^2 - 5*x + 1";
    println!("      Formula: {}", formula);
    let result = diff(formula, "x", None, None).unwrap();
    println!("      d/dx:    {}\n", result);

    // 1.2 Simple simplify() function
    println!("  1.2 Simplification: simplify()");
    let ugly = "x + x + x + 0*y + 1*z";
    println!("      Before: {}", ugly);
    let clean = simplify(ugly, None, None).unwrap();
    println!("      After:  {}\n", clean);

    // 1.3 Diff builder with options
    println!("  1.3 Diff Builder with Options");
    let a_sym = symb("a");
    let b_sym = symb("b");
    let result = Diff::new()
        .domain_safe(true) // Enable domain safety checks
        .fixed_var(&a_sym) // Treat 'a' as constant
        .fixed_var(&b_sym) // Treat 'b' as constant
        .diff_str("a*x^2 + b*x + c", "x")
        .unwrap();
    println!(
        "      d/dx [ax² + bx + c] with a,b as constants: {}\n",
        result
    );

    // 1.4 Simplify builder with options
    println!("  1.4 Simplify Builder with Options");
    let k_sym = symb("k");
    let result = Simplify::new()
        .domain_safe(true) // Safe simplifications only
        .fixed_var(&k_sym) // k is a constant
        .simplify_str("k*x + k*y")
        .unwrap();
    println!("      Simplified k*x + k*y: {}\n", result);
}

// =============================================================================
// PART 2: TYPE-SAFE RUSTY API
// =============================================================================
fn part2_type_safe_api() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🦀 PART 2: TYPE-SAFE RUSTY API");
    println!("   Best for building expressions programmatically in Rust");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 2.1 Creating symbols
    println!("  2.1 Creating Symbols: symb()");
    let x: Symbol = symb("x");
    let y: Symbol = symb("y");
    println!("      Created x and y symbols\n");

    // 2.2 Building expressions with operators
    println!("  2.2 Building Expressions with Operators (+, -, *, /, ^)");
    let expr1: Expr = x.clone() + y.clone();
    let expr2: Expr = x.clone() * y.clone();
    let expr3: Expr = x.clone().pow(2.0) + y.clone().pow(2.0);
    println!("      x + y  = {}", expr1);
    println!("      x * y  = {}", expr2);
    println!("      x² + y² = {}\n", expr3);

    // 2.3 Building expressions with functions
    println!("  2.3 Building Expressions with Functions");
    let expr_sin = x.clone().sin();
    let expr_exp = x.clone().exp();
    let expr_ln = x.clone().ln();
    let expr_sqrt = x.clone().sqrt();
    println!("      sin(x) = {}", expr_sin);
    println!("      exp(x) = {}", expr_exp);
    println!("      ln(x)  = {}", expr_ln);
    println!("      √x     = {}\n", expr_sqrt);

    // 2.4 Differentiation with Expr
    println!("  2.4 Differentiating Expression Objects");
    let f: Expr = x.clone().pow(3.0) + (2.0 * x.clone()).sin();
    println!("      f(x) = {}", f);
    let df = Diff::new().differentiate(f, &x).unwrap();
    println!("      f'(x) = {}\n", df);

    // 2.5 Expr utility methods
    println!("  2.5 Expression Utility Methods");
    let complex: Expr = x.clone().pow(2.0) + y.clone().sin();
    println!("      Expression: {}", complex);
    println!("      Node count: {}", complex.node_count());
    println!("      Max depth:  {}", complex.max_depth());
    println!();
}

// =============================================================================
// PART 3: NUMERICAL EVALUATION
// =============================================================================
fn part3_numerical_evaluation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🔢 PART 3: NUMERICAL EVALUATION");
    println!("   Evaluate expressions with specific variable values");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 3.1 Evaluate Expr with HashMap
    println!("  3.1 Evaluate Expression Objects");
    let x = symb("x");
    let y = symb("y");
    let expr: Expr = x.clone().pow(2.0) + y.clone().pow(2.0);

    let mut vars: HashMap<&str, f64> = HashMap::new();
    vars.insert("x", 3.0);
    vars.insert("y", 4.0);

    let result = expr.evaluate(&vars);
    println!("      x² + y² at (x=3, y=4)");
    if let ExprKind::Number(n) = result.kind {
        println!("      Result: {} (expected: 25)\n", n);
    }

    // 3.2 Evaluate using evaluate_str
    println!("  3.2 Evaluate from String: evaluate_str()");
    let result = evaluate_str("sin(pi/6)^2 + cos(pi/6)^2", &[]).unwrap();
    println!("      sin²(π/6) + cos²(π/6) = {} (expected: 1)\n", result);

    // 3.3 Partial evaluation (mixed symbolic/numeric)
    println!("  3.3 Partial Evaluation");
    let result = evaluate_str("a*x + b", &[("a", 2.0), ("b", 5.0)]).unwrap();
    println!("      a*x + b with a=2, b=5 → {}\n", result);
}

// =============================================================================
// PART 4: MULTI-VARIABLE CALCULUS
// =============================================================================
fn part4_multi_variable_calculus() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📐 PART 4: MULTI-VARIABLE CALCULUS");
    println!("   Gradient, Hessian, and Jacobian computations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 4.1 Gradient (Expr-based)
    println!("  4.1 Gradient: ∇f = [∂f/∂x, ∂f/∂y, ...]");
    let x = symb("x");
    let y = symb("y");
    let f: Expr = x.clone().pow(2.0) * y.clone() + y.clone().pow(3.0);
    println!("      f(x,y) = x²y + y³");

    let grad = gradient(&f, &[&x, &y]);
    println!("      ∂f/∂x = {}", grad[0]);
    println!("      ∂f/∂y = {}\n", grad[1]);

    // 4.2 Gradient (String-based)
    println!("  4.2 Gradient from String: gradient_str()");
    let grad = gradient_str("sin(x)*cos(y)", &["x", "y"]).unwrap();
    println!("      f = sin(x)cos(y)");
    println!("      ∂f/∂x = {}", grad[0]);
    println!("      ∂f/∂y = {}\n", grad[1]);

    // 4.3 Hessian
    println!("  4.3 Hessian Matrix: H[i][j] = ∂²f/∂xᵢ∂xⱼ");
    let hess = hessian_str("x^2*y + y^3", &["x", "y"]).unwrap();
    println!("      f = x²y + y³");
    println!("      H = | {} {} |", hess[0][0], hess[0][1]);
    println!("          | {} {} |\n", hess[1][0], hess[1][1]);

    // 4.4 Jacobian
    println!("  4.4 Jacobian Matrix: J[i][j] = ∂fᵢ/∂xⱼ");
    let jac = jacobian_str(&["x^2 + y", "x*y"], &["x", "y"]).unwrap();
    println!("      f₁ = x² + y, f₂ = xy");
    println!("      J = | {} {} |", jac[0][0], jac[0][1]);
    println!("          | {} {} |\n", jac[1][0], jac[1][1]);
}

// =============================================================================
// PART 5: ALL SUPPORTED FUNCTIONS
// =============================================================================
fn part5_all_functions() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📚 PART 5: ALL SUPPORTED MATHEMATICAL FUNCTIONS");
    println!("   60+ functions with symbolic differentiation and numeric eval");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  TRIGONOMETRIC:");
    println!("    sin(x), cos(x), tan(x), cot(x), sec(x), csc(x)\n");

    println!("  INVERSE TRIG:");
    println!("    asin(x), acos(x), atan(x), acot(x), asec(x), acsc(x)\n");

    println!("  HYPERBOLIC:");
    println!("    sinh(x), cosh(x), tanh(x), coth(x), sech(x), csch(x)\n");

    println!("  INVERSE HYPERBOLIC:");
    println!("    asinh(x), acosh(x), atanh(x), acoth(x), asech(x), acsch(x)\n");

    println!("  EXPONENTIAL & LOGARITHMIC:");
    println!("    exp(x), ln(x), log(x), log10(x), log2(x)\n");

    println!("  POWERS & ROOTS:");
    println!("    sqrt(x), cbrt(x), x^n, x^(1/n)\n");

    println!("  SPECIAL FUNCTIONS:");
    println!("    gamma(x), digamma(x), trigamma(x), polygamma(n,x)");
    println!("    erf(x), erfc(x), zeta(x), beta(a,b)\n");

    println!("  BESSEL FUNCTIONS:");
    println!("    besselj(n,x), bessely(n,x), besseli(n,x), besselk(n,x)\n");

    println!("  OTHER:");
    println!("    abs(x), sign(x), floor(x), ceil(x), round(x)");
    println!("    sinc(x), lambertw(x)\n");

    // Demonstrate some derivatives
    println!("  Example Derivatives:");
    let examples = [
        ("gamma(x)", "x"),
        ("erf(x)", "x"),
        ("besselj(0, x)", "x"),
        ("lambertw(x)", "x"),
    ];
    for (expr, var) in &examples {
        let result = diff(expr, var, None, None).unwrap();
        println!("    d/d{} [{}] = {}", var, expr, result);
    }
    println!();
}

// =============================================================================
// PART 6: CUSTOM FUNCTIONS (DERIVATIVES & EVALUATION)
// =============================================================================
fn part6_custom_derivatives() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✨ PART 6: CUSTOM FUNCTIONS");
    println!("   Define derivative rules AND evaluation for user-defined functions");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let x = symb("x");

    // 6.1 Custom Derivative Only
    println!("  6.1 Custom Derivative Rule: custom_derivative()");
    println!("      Define: my_func(u) with derivative: d/dx[my_func(u)] = 3u² · u'");

    let custom_diff = Diff::new().custom_derivative("my_func", |inner_u, _var, u_prime| {
        // d/dx[my_func(u)] = 3 * u^2 * u'
        Expr::number(3.0) * inner_u.clone().pow_of(Expr::number(2.0)) * u_prime.clone()
    });

    let my_expr = Expr::func("my_func", x.clone().pow(2.0));
    println!("      Expression: {}", my_expr);

    let result = custom_diff.differentiate(my_expr, &x).unwrap();
    println!("      d/dx: {}", result);

    let simplified = Simplify::new().simplify(result).unwrap();
    println!("      Simplified: {}\n", simplified);

    // 6.2 Custom Evaluation (NEW!)
    println!("  6.2 Custom Evaluation: custom_eval()");
    println!("      Define: f(x) = x² + 1 for numerical evaluation");

    use std::sync::Arc;

    // Create f(x)
    let f_of_x = Expr::func("f", x.to_expr());
    println!("      Expression: {}", f_of_x);

    // Without custom evaluator: f(3) stays as f(3)
    let mut vars: HashMap<&str, f64> = HashMap::new();
    vars.insert("x", 3.0);
    let result_no_eval = f_of_x.evaluate(&vars);
    println!("      Without custom_eval: f(3) → {}", result_no_eval);

    // With custom evaluator: f(3) computes to 10
    type CustomEval = Arc<dyn Fn(&[f64]) -> Option<f64> + Send + Sync>;
    let mut custom_evals: HashMap<String, CustomEval> = HashMap::new();
    custom_evals.insert(
        "f".to_string(),
        Arc::new(|args: &[f64]| Some(args[0].powi(2) + 1.0)), // f(x) = x² + 1
    );
    let result_with_eval = f_of_x.evaluate_with_custom(&vars, &custom_evals);
    println!(
        "      With custom_eval:    f(3) → {} (3² + 1 = 10)\n",
        result_with_eval
    );

    // 6.3 Combined: Derivative AND Evaluation
    println!("  6.3 Complete Custom Function: Derivative + Evaluation");
    println!("      Define g(x) = sin(x)² with known derivative: 2sin(x)cos(x)·x'");

    // Build: g(x²)
    let g_of_xsq = Expr::func("g", x.clone().pow(2.0));
    println!("      Expression: {}", g_of_xsq);

    // Setup differentiation with custom derivative
    let diff_builder = Diff::new().custom_derivative("g", |inner, _var, inner_prime| {
        // d/dx[g(u)] = 2·sin(u)·cos(u)·u'
        Expr::number(2.0) * inner.clone().sin() * inner.clone().cos() * inner_prime.clone()
    });

    let derivative = diff_builder.differentiate(g_of_xsq.clone(), &x).unwrap();
    println!("      d/dx[g(x²)] = {}", derivative);

    // Setup evaluation: g(x) = sin²(x)
    let mut g_eval: HashMap<String, CustomEval> = HashMap::new();
    g_eval.insert(
        "g".to_string(),
        Arc::new(|args: &[f64]| Some(args[0].sin().powi(2))), // g(x) = sin²(x)
    );

    // Evaluate g(x²) at x = π/4 → g(π²/16) = sin²(π²/16)
    let mut vars2: HashMap<&str, f64> = HashMap::new();
    let pi_over_4 = std::f64::consts::FRAC_PI_4;
    vars2.insert("x", pi_over_4);
    let evaluated = g_of_xsq.evaluate_with_custom(&vars2, &g_eval);
    println!("      g(x²) at x=π/4 = {}", evaluated);
    println!();
}

// =============================================================================
// PART 7: SAFETY FEATURES
// =============================================================================
fn part7_safety_features() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🛡️  PART 7: SAFETY FEATURES & CONFIGURATION");
    println!("   Prevent resource exhaustion and handle edge cases");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // 7.1 Max Depth Limit
    println!("  7.1 Maximum Expression Depth");
    let x = symb("x");
    let mut deep: Expr = x.clone().into();
    for _ in 0..60 {
        deep = deep.sin();
    }
    println!("      Created expression with depth: {}", deep.max_depth());

    let safe_diff = Diff::new().max_depth(25);
    match safe_diff.differentiate(deep, &x) {
        Ok(_) => println!("      Differentiation succeeded"),
        Err(e) => println!("      ✅ Prevented: {:?}", e),
    }
    println!();

    // 7.2 Max Node Count Limit
    println!("  7.2 Maximum Node Count");
    let x2 = symb("x");
    let mut broad: Expr = x2.clone().into();
    for _ in 0..12 {
        broad = broad.clone() + broad.clone();
    }
    println!("      Created expression with {} nodes", broad.node_count());

    let safe_diff = Diff::new().max_nodes(500);
    match safe_diff.differentiate(broad, &x2) {
        Ok(_) => println!("      Differentiation succeeded"),
        Err(e) => println!("      ✅ Prevented: {:?}", e),
    }
    println!();

    // 7.3 Domain Safety
    println!("  7.3 Domain Safety Mode");
    println!("      Diff::new().domain_safe(true)");
    println!("      Prevents simplifications that could introduce undefined values");
    println!("      Example: √(x²) = |x| (not x, which fails for x < 0)\n");

    // 7.4 Fixed Variables
    println!("  7.4 Fixed Variables (Constants)");
    let a_const = symb("a");
    let b_const = symb("b");
    let result = Diff::new()
        .fixed_var(&a_const)
        .fixed_var(&b_const)
        .diff_str("a*x^2 + b*x + c", "x")
        .unwrap();
    println!("      With a, b as constants:");
    println!("      d/dx [ax² + bx + c] = {}", result);
    println!();
}
