import timeit
import sympy
from sympy import symbols, sympify, diff, simplify, sin, cos, tan, exp, log

def run_bench(name, stmt, setup, number=10000):
    t = timeit.Timer(stmt, setup)
    # Autorange determination or fixed number? 
    # Rust criterion uses statistical analysis. 
    # We'll stick to a reasonable fixed number for quick comparison or use repeat.
    # To match criterion default roughly, we might want faster runs.
    # Let's do 1000 iters for heavier tasks.
    
    try:
        time_taken = t.timeit(number=number)
        per_iter = (time_taken / number) * 1e6 # microseconds
        print(f"{name:<50} {per_iter:>10.2f} us/iter")
    except Exception as e:
        print(f"{name:<50} FAILED: {e}")

x, y = symbols('x y')

# ==============================================================================
# 1. Parsing Benchmarks (String -> Expression)
# ==============================================================================
print(f"{'='*30} PARSING {'='*30}")
run_bench("parse_poly_x^3+2x^2+x", 
          "sympify('x**3 + 2*x**2 + x')", 
          "from sympy import sympify", number=1000)

run_bench("parse_trig_sin(x)*cos(x)", 
          "sympify('sin(x) * cos(x)')", 
          "from sympy import sympify", number=1000)

run_bench("parse_complex_x^2*sin(x)*exp(x)", 
          "sympify('x**2 * sin(x) * exp(x)')", 
          "from sympy import sympify", number=1000)

run_bench("parse_nested_sin(cos(tan(x)))", 
          "sympify('sin(cos(tan(x)))')", 
          "from sympy import sympify", number=1000)


# ==============================================================================
# 2. Differentiation (Expression -> Expression)
# Equivalent to 'bench_diff_ast' (Raw AST derivative)
# ==============================================================================
print(f"\n{'='*30} DIFFERENTIATION (AST) {'='*30}")

setup_ast = """
from sympy import symbols, sin, cos, tan, exp, diff
x, y = symbols('x y')
poly = x**3 + 2*x**2 + x
trig = sin(x) * cos(x)
complex_expr = x**2 * sin(x) * exp(x)
nested = sin(cos(tan(x)))
"""

run_bench("diff_ast_poly", "diff(poly, x)", setup_ast, number=1000)
run_bench("diff_ast_trig", "diff(trig, x)", setup_ast, number=1000)
run_bench("diff_ast_complex", "diff(complex_expr, x)", setup_ast, number=1000)
run_bench("diff_ast_nested", "diff(nested, x)", setup_ast, number=1000)


# ==============================================================================
# 3. Differentiation (String -> Expression)
# Equivalent to 'bench_differentiation' (Parse + Diff)
# ==============================================================================
print(f"\n{'='*30} DIFFERENTIATION (FULL) {'='*30}")

setup_full = "from sympy import sympify, diff, Symbol; x = Symbol('x')"

run_bench("poly_x^3+2x^2+x", 
          "diff(sympify('x**3 + 2*x**2 + x'), x)", 
          setup_full, number=1000)

run_bench("trig_sin(x)*cos(x)", 
          "diff(sympify('sin(x) * cos(x)'), x)", 
          setup_full, number=1000)

run_bench("chain_sin(x^2)", 
          "diff(sympify('sin(x**2)'), x)", 
          setup_full, number=1000)

run_bench("exp_e^(x^2)", 
          "diff(sympify('exp(x**2)'), x)", 
          setup_full, number=1000)

run_bench("complex_x^2*sin(x)*exp(x)", 
          "diff(sympify('x**2 * sin(x) * exp(x)'), x)", 
          setup_full, number=1000)

run_bench("quotient_(x^2+1)/(x-1)", 
          "diff(sympify('(x**2 + 1) / (x - 1)'), x)", 
          setup_full, number=1000)

run_bench("nested_sin(cos(tan(x)))", 
          "diff(sympify('sin(cos(tan(x)))'), x)", 
          setup_full, number=1000)

run_bench("power_x^x", 
          "diff(sympify('x**x'), x)", 
          setup_full, number=1000)


# ==============================================================================
# 4. Simplification (Expression -> Expression)
# Equivalent to 'bench_simplify_ast'
# ==============================================================================
print(f"\n{'='*30} SIMPLIFICATION (AST) {'='*30}")

setup_simp_ast = """
from sympy import symbols, sin, cos, exp, simplify
x, y = symbols('x y')
pythag = sin(x)**2 + cos(x)**2
perfect = x**2 + 2*x + 1
frac = (x + 1)**2 / (x + 1)
exp_comb = exp(x) * exp(y)
"""

run_bench("simplify_ast_pythagorean", "simplify(pythag)", setup_simp_ast, number=100)
run_bench("simplify_ast_perfect_square", "simplify(perfect)", setup_simp_ast, number=100)
run_bench("simplify_ast_fraction", "simplify(frac)", setup_simp_ast, number=100)
run_bench("simplify_ast_exp_combine", "simplify(exp_comb)", setup_simp_ast, number=100)


# ==============================================================================
# 5. Simplification (String -> Expression)
# Equivalent to 'bench_simplification'
# ==============================================================================
print(f"\n{'='*30} SIMPLIFICATION (FULL) {'='*30}")

setup_simp_full = "from sympy import sympify, simplify"

run_bench("pythagorean_sin^2+cos^2", 
          "simplify(sympify('sin(x)**2 + cos(x)**2'))", 
          setup_simp_full, number=100)

run_bench("perfect_square_x^2+2x+1", 
          "simplify(sympify('x**2 + 2*x + 1'))", 
          setup_simp_full, number=100)

run_bench("fraction_(x+1)^2/(x+1)", 
          "simplify(sympify('(x + 1)**2 / (x + 1)'))", 
          setup_simp_full, number=100)

run_bench("exp_combine_e^x*e^y", 
          "simplify(sympify('exp(x) * exp(y)'))", 
          setup_simp_full, number=100)

run_bench("like_terms_2x+3x+x", 
          "simplify(sympify('2*x + 3*x + x'))", 
          setup_simp_full, number=100)

run_bench("frac_add_(x^2+1)/(x^2-1)+1/(x+1)", 
          "simplify(sympify('(x**2 + 1)/(x**2 - 1) + 1/(x + 1)'))", 
          setup_simp_full, number=100)

run_bench("hyp_sinh_(e^x-e^-x)/2", 
          "simplify(sympify('(exp(x) - exp(-x)) / 2'))", 
          setup_simp_full, number=100)

run_bench("power_x^2*x^3/x", 
          "simplify(sympify('x**2 * x**3 / x'))", 
          setup_simp_full, number=100)


# ==============================================================================
# 6. Combined (Diff + Simplify)
# ==============================================================================
print(f"\n{'='*30} COMBINED (DIFF + SIMPLIFY) {'='*30}")

setup_combined = "from sympy import sympify, diff, simplify, Symbol; x = Symbol('x')"

# d/dx[sin(x)^2]_simplified
run_bench("d/dx[sin(x)^2]_simplified", 
          "simplify(diff(sympify('sin(x)**2'), x))", 
          setup_combined, number=100)

# d/dx[(x^2+1)/(x-1)]_simplified
run_bench("d/dx[(x^2+1)/(x-1)]_simplified", 
          "simplify(diff(sympify('(x**2 + 1) / (x - 1)'), x))", 
          setup_combined, number=100)
