# SymbAnaFis vs SymPy Benchmark Comparison

**Date:** December 5, 2025 (Updated)
**SymbAnaFis Version:** 0.2.5  
**SymPy Version:** Latest (Python 3)  
**System:** Linux  
**Criterion Version:** 0.8.0

## Summary

SymbAnaFis is a symbolic mathematics library written in Rust, designed for parsing, differentiation, and simplification of mathematical expressions. This document compares its performance against SymPy, the industry-standard Python symbolic mathematics library.

### Key Findings

| Category | Winner | Speedup Range |
|----------|--------|---------------|
| **Normal PDF derivative** | SymbAnaFis | **51.53x faster** |
| **Differentiation + Simplify** | SymbAnaFis | 19x - 73x faster |
| **Simplification Only** | SymbAnaFis | 4x - 69x faster |
| **Combined Diff + Simplify** | SymbAnaFis | 19x - 26x faster |

---

## Detailed Results

### Differentiation (with Simplification)

Both libraries perform differentiation followed by simplification for fair comparison.

| Expression | SymbAnaFis | SymPy | Ratio | Change |
|------------|------------|-------|-------|--------|
| `x^3 + 2x^2 + x` | 81.07 µs | 2322.03 µs | SymbAnaFis **28.6x faster** | 
| `sin(x) * cos(x)` | 123.63 µs | 9143.11 µs | SymbAnaFis **73.9x faster** |
| `sin(x^2)` (chain rule) | 63.87 µs | 3256.28 µs | SymbAnaFis **51.0x faster** | 
| `e^(x^2)` | 58.96 µs | 1324.07 µs | SymbAnaFis **22.5x faster** | 
| `x^2 * sin(x) * exp(x)` | 325.88 µs | 18994.27 µs | SymbAnaFis **58.3x faster** |
| `(x^2 + 1) / (x - 1)` (quotient rule) | 250.73 µs | 6596.52 µs | SymbAnaFis **26.3x faster** | 
| `sin(cos(tan(x)))` | 202.12 µs | 13599.54 µs | SymbAnaFis **67.3x faster** | 
| `x^x` | 67.54 µs | 2095.25 µs | SymbAnaFis **31.0x faster** | 

### Differentiation (AST Only - No Simplification)

| Expression | SymbAnaFis | Change |
|------------|------------|----|
| Polynomial | 147.25 ns | 
| Trigonometric | 141.61 ns | 
| Complex | 246.28 ns | 
| Nested | 364.75 ns | 

> These sub-microsecond times show the raw differentiation engine is extremely fast.

---

### Simplification

| Expression | SymbAnaFis | SymPy | Ratio | Change |
|------------|------------|-------|-------|--------|
| `sin²(x) + cos²(x)` → `1` | 57.15 µs | 4023.68 µs | SymbAnaFis **70x faster** | 
| `x² + 2x + 1` → `(x+1)²` | 52.85 µs | 237.83 µs | SymbAnaFis **4.5x faster** | 
| `(x+1)² / (x+1)` → `x+1` | 41.39 µs | 985.60 µs | SymbAnaFis **23.8x faster** | 
| `e^x * e^y` → `e^(x+y)` | 56.63 µs | 1430.39 µs | SymbAnaFis **25.3x faster** | 
| `2x + 3x + x` → `6x` | 46.09 µs | 487.14 µs | SymbAnaFis **10.6x faster** | 
| `(x²+1)/(x²-1) + 1/(x+1)` | 216.64 µs | 3466.14 µs | SymbAnaFis **16x faster** | 
| `(e^x - e^-x)/2` → `sinh(x)` | 71.28 µs | 3527.34 µs | SymbAnaFis **49.5x faster** | 
| `x² * x³ / x` → `x⁴` | 47.03 µs | 498.26 µs | SymbAnaFis **10.6x faster** | 

---

### Combined: Differentiation + Simplification

| Expression | SymbAnaFis | SymPy | Ratio | Change |
|------------|------------|-------|-------|--------|
| `d/dx[sin(x)²]` simplified | 110.85 µs | 2946.35 µs | SymbAnaFis **26.6x faster** | 
| `d/dx[(x²+1)/(x-1)]` simplified | 354.54 µs | 6727.66 µs | SymbAnaFis **19.0x faster** | 

---

### Real-World Example: Normal Distribution PDF

A complex real-world expression from statistics:

```
f(x) = exp(-(x - μ)² / (2σ²)) / √(2πσ²)
```

#### Raw Differentiation (No Simplification)

| Metric | SymbAnaFis | SymPy | Ratio |
|--------|------------|-------|-------|
| **Time** | **1.80 µs** | 4443.48 µs | SymbAnaFis **2469x faster** |
| **Output length** | 153 chars | 120 chars | SymbAnaFis more verbose |
| **Characters × Time** | 275 | 534,418 | SymbAnaFis 1944x lower |

**SymbAnaFis raw output:** 
```
(exp(-(x - mu)^2/(2sigma^2))*-2(x - mu)^12sigma^2/(2sigma^2)^2sqrt(2pi*sigma^2) - exp(-(x - mu)^2/(2sigma^2))*0/(2sqrt(2pi*sigma^2)))/sqrt(2pi*sigma^2)^2
```

**SymPy raw output:**
```
-sqrt(2)*(-2*mu + 2*x)*exp(-(-mu + x)**2/(2*sigma**2))/(4*sqrt(pi)*sigma**2*Abs(sigma))
```


#### With Simplification

| Metric | SymbAnaFis | SymPy | Ratio |
|--------|------------|-------|-------|
| **Parse time** | 52.99 µs | N/A | N/A |
| **Differentiation time** | 1253.13 µs | 4395.92 µs | SymbAnaFis **3.5x faster** |
| **Simplification time** | 369.98 µs | 81980.44 µs | SymbAnaFis **221x faster** |
| **Total time** | **1676.11 µs** | **86376.37 µs** | SymbAnaFis **51.53x faster** |
| **Output length** | 67 chars | 94 chars | Difference of Spaces and Power notation |

**SymbAnaFis simplified:**
```
-exp(-(x - mu)^2/(2sigma^2))(x - mu)/(sigma^2*sqrt(2pi)*abs(sigma))
```

**SymPy simplified:**
```
sqrt(2)*(mu - x)*exp(-(mu - x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**2*Abs(sigma))
```

#### Output Equivalence Verification

Both outputs are mathematically equivalent:

- **SymbAnaFis**: `-(x - mu) / (sigma^2 * sqrt(2pi*sigma^2))` = `-(x - mu) / (2*pi*sigma^3)`
- **SymPy**: `sqrt(2)*(mu - x) / (2*sqrt(pi)*sigma^2*abs(sigma))` = `(mu - x) / (2*pi*sigma^3)` (for σ > 0)

The exponential term is identical: `exp(-(x - mu)^2 / (2*sigma^2))`

Both represent the correct derivative of the normal PDF (up to sign and algebraic form).

---

### Parsing Only

| Expression | SymbAnaFis |
|------------|------------||
| `x^3 + 2x^2 + x` | 607.68 ns |
| `sin(x) * cos(x)` | 539.51 ns |
| `x^2 * sin(x) * exp(x)` | 734.66 ns |
| `sin(cos(tan(x)))` | 542.54 ns |

> Parsing is sub-microsecond for all tested expressions.

---

## Analysis

### Why SymbAnaFis is Faster

1. **Rule-based engine with ExprKind filtering**: O(1) rule lookup instead of O(n) scanning
2. **No Python overhead**: Pure Rust with zero-cost abstractions
3. **Pattern matching optimization**: Rules only run on applicable expression types
4. **Efficient AST representation**: Using Rust's `Rc` for shared expression nodes
5. **Compiled native code**: No interpreter overhead

### Performance Summary

SymbAnaFis consistently outperforms SymPy across all benchmarks when comparing equivalent operations (differentiation + simplification):

- **Differentiation + Simplify**: 8x - 31x faster
- **Simplification only**: 1.8x - 31x faster
- **Parsing**: Sub-microsecond (500-800 ns)

### Real-World Implications

For scientific computing, physics simulations, and engineering applications where you need both differentiation AND simplification:
- SymbAnaFis provides **significant performance benefits**
- Typical speedups of **12-20x** for common expressions
- Up to **31x faster** for trigonometric expressions

---

## Running the Benchmarks

### SymbAnaFis (Rust)
```bash
cargo bench
```

### SymPy (Python)
```bash
python3 benches/sympy_benchmark.py
```

---

## Hardware

Benchmarks were run on a single machine to ensure fair comparison:
- All tests use the same expressions
- Criterion uses statistical sampling (100 samples per benchmark)
- SymPy benchmark uses `timeit` with 1000 iterations

---

## Future Optimizations

Potential improvements for SymbAnaFis:
- [ ] SIMD-accelerated pattern matching
- [ ] Parallel rule application for independent sub-expressions
- [ ] Caching of common sub-expression simplifications
- [ ] JIT compilation of hot paths

---

*Generated with Criterion 0.8.0 and Python timeit*
