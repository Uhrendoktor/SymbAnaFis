#!/usr/bin/env python3
"""
Direct comparison of symb_anafis vs SymPy on Normal Distribution PDF
"""

import time
import sympy as sp
import symb_anafis as sa

print("=" * 70)
print("COMPARISON: SymbAnaFis vs SymPy")
print("Expression: Normal Distribution PDF derivative")
print("=" * 70)

# Expression: d/dx[exp(-(x - mu)^2 / (2 * sigma^2)) / sqrt(2 * pi * sigma^2)]
expr_str = "exp(-(x - mu)^2 / (2 * sigma^2)) / sqrt(2 * pi * sigma^2)"

print(f"\nExpression: {expr_str}\n")

# ============================================================================
# SYMPY
# ============================================================================
print("=" * 70)
print("SYMPY")
print("=" * 70)

# Define symbols
x, mu, sigma = sp.symbols('x mu sigma', real=True)

# Parse and differentiate
expr_sympy = sp.exp(-(x - mu)**2 / (2 * sigma**2)) / sp.sqrt(2 * sp.pi * sigma**2)

start = time.perf_counter()
derivative_sympy = sp.diff(expr_sympy, x)
sympy_diff_time = (time.perf_counter() - start) * 1e6

start = time.perf_counter()
simplified_sympy = sp.simplify(derivative_sympy)
sympy_simplify_time = (time.perf_counter() - start) * 1e6

total_sympy = sympy_diff_time + sympy_simplify_time

print(f"Differentiation time:  {sympy_diff_time:.2f} µs")
print(f"Simplification time:   {sympy_simplify_time:.2f} µs")
print(f"Total time:            {total_sympy:.2f} µs")
print(f"\nDerivative:\n{derivative_sympy}")
print(f"\nSimplified:\n{simplified_sympy}")

# ============================================================================
# SYMB_ANAFIS
# ============================================================================
print("\n" + "=" * 70)
print("SYMB_ANAFIS")
print("=" * 70)

# Parse expression
start = time.perf_counter()
expr_sa = sa.parse(expr_str)
sa_parse_time = (time.perf_counter() - start) * 1e6

# Differentiate
start = time.perf_counter()
derivative_sa = sa.diff(expr_str, 'x')
sa_diff_time = (time.perf_counter() - start) * 1e6

# Simplify
start = time.perf_counter()
simplified_sa = sa.simplify(derivative_sa)
sa_simplify_time = (time.perf_counter() - start) * 1e6

total_sa = sa_parse_time + sa_diff_time + sa_simplify_time

print(f"Parse time:            {sa_parse_time:.2f} µs")
print(f"Differentiation time:  {sa_diff_time:.2f} µs")
print(f"Simplification time:   {sa_simplify_time:.2f} µs")
print(f"Total time:            {total_sa:.2f} µs")
print(f"\nDerivative:\n{derivative_sa}")
print(f"\nSimplified:\n{simplified_sa}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

speedup = total_sympy / total_sa

print(f"\nSymPy total:           {total_sympy:.2f} µs")
print(f"SymbAnaFis total:      {total_sa:.2f} µs")
print(f"Speedup:               {speedup:.2f}x")

if speedup > 1:
    print(f"\n✓ SymbAnaFis is {speedup:.2f}x FASTER than SymPy")
else:
    print(f"\n✗ SymPy is {1/speedup:.2f}x faster than SymbAnaFis")

print("\n" + "=" * 70)
