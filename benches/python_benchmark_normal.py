#!/usr/bin/env python3
"""
Benchmark the Normal Distribution PDF derivative through Python bindings.
Tests both raw differentiation and diff+simplify pipeline.
"""

import time
import symb_anafis

# Normal PDF: f(x) = exp(-(x - μ)² / (2σ²)) / √(2πσ²)
normal_pdf = "exp(-(x - mu)^2 / (2 * sigma^2)) / sqrt(2 * pi * sigma^2)"

print("=" * 70)
print("SymbAnaFis Python Binding Benchmark: Normal Distribution PDF")
print("=" * 70)
print(f"\nExpression: f(x) = {normal_pdf}")
print("\nμ (mu): mean parameter")
print("σ (sigma): standard deviation parameter")
print("x: variable to differentiate with respect to")

print("\n" + "=" * 70)
print("1. PARSING")
print("=" * 70)

# Benchmark parsing
parse_times = []
for _ in range(1000):
    start = time.perf_counter()
    symb_anafis.parse(normal_pdf)
    parse_times.append(time.perf_counter() - start)

avg_parse = sum(parse_times) / len(parse_times)
print(f"Average parse time: {avg_parse * 1e6:.2f} µs")
print(f"Min: {min(parse_times) * 1e6:.2f} µs, Max: {max(parse_times) * 1e6:.2f} µs")

print("\n" + "=" * 70)
print("2. RAW DIFFERENTIATION (no simplification)")
print("=" * 70)

diff_times = []
for _ in range(100):
    start = time.perf_counter()
    result_raw = symb_anafis.diff(normal_pdf, "x")
    diff_times.append(time.perf_counter() - start)

avg_diff = sum(diff_times) / len(diff_times)
print(f"Average differentiation time: {avg_diff * 1e6:.2f} µs")
print(f"Min: {min(diff_times) * 1e6:.2f} µs, Max: {max(diff_times) * 1e6:.2f} µs")
print(f"\nRaw output length: {len(result_raw)} chars")
print(f"Raw output:\n{result_raw}\n")

print("\n" + "=" * 70)
print("3. DIFFERENTIATION + SIMPLIFICATION")
print("=" * 70)

diff_simplify_times = []
for _ in range(100):
    start = time.perf_counter()
    # First differentiate, then simplify
    raw = symb_anafis.diff(normal_pdf, "x")
    result_simplified = symb_anafis.simplify(raw)
    diff_simplify_times.append(time.perf_counter() - start)

avg_diff_simplify = sum(diff_simplify_times) / len(diff_simplify_times)
print(f"Average diff+simplify time: {avg_diff_simplify * 1e6:.2f} µs")
print(f"Min: {min(diff_simplify_times) * 1e6:.2f} µs, Max: {max(diff_simplify_times) * 1e6:.2f} µs")
print(f"\nSimplified output length: {len(result_simplified)} chars")
print(f"Simplified output:\n{result_simplified}\n")

print("\n" + "=" * 70)
print("4. PERFORMANCE SUMMARY")
print("=" * 70)

print(f"\nParsing:                    {avg_parse * 1e6:8.2f} µs")
print(f"Differentiation:            {avg_diff * 1e6:8.2f} µs")
print(f"Diff + Simplify:            {avg_diff_simplify * 1e6:8.2f} µs")

speedup = avg_diff_simplify / avg_diff
print(f"\nSimplification cost factor:  {speedup:.2f}x")
print(f"Simplification time alone:   {(avg_diff_simplify - avg_diff) * 1e6:8.2f} µs")

print("\n" + "=" * 70)
print("5. SIMPLIFICATION RESULTS")
print("=" * 70)

# Also test simplifying various intermediate forms
print("\n5a. Direct simplification of common sub-expressions:")

test_exprs = [
    ("sin(x)^2 + cos(x)^2", "Pythagorean identity"),
    ("exp(x) * exp(y)", "Exponential combination"),
    ("x * x * x / x", "Power cancellation"),
]

for expr, description in test_exprs:
    start = time.perf_counter()
    result = symb_anafis.simplify(expr)
    elapsed = time.perf_counter() - start
    print(f"  • {description}")
    print(f"    Input:  {expr}")
    print(f"    Output: {result}")
    print(f"    Time:   {elapsed * 1e6:.2f} µs\n")

print("=" * 70)
