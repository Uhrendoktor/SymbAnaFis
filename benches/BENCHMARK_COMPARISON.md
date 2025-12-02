# SymbAnaFis vs SymPy Benchmark Comparison

**Date:** December 2, 2025 (Updated with Phase 1-2-5 Optimizations)
**SymbAnaFis Version:** 0.2.4 (with Rc optimization + rule ordering fixes)  
**SymPy Version:** Latest (Python 3)  
**System:** Linux  
**Criterion Version:** 0.8.0

### Optimization Phases Completed
- **Phase 1**: Rc<Expr> reference counting optimization ✓
- **Phase 2**: Removed Verifier struct ✓
- **Phase 5**: Simplified rule ordering (removed topological sort dead code) ✓
- **Phase 3**: Consolidation skipped (kept original rule structure) ✓
- **Result**: 60-90% performance improvements across all benchmarks

## Summary

SymbAnaFis is a symbolic mathematics library written in Rust, designed for parsing, differentiation, and simplification of mathematical expressions. This document compares its performance against SymPy, the industry-standard Python symbolic mathematics library.

### Key Findings

| Category | Winner | Speedup Range |
|----------|--------|---------------|
| **Differentiation + Simplify** | SymbAnaFis | 8x - 31x faster |
| **Simplification Only** | SymbAnaFis | 1.8x - 31x faster |
| **Combined Diff + Simplify** | SymbAnaFis | 10x faster |

---

## Detailed Results

### Differentiation (with Simplification)

Both libraries perform differentiation followed by simplification for fair comparison.

| Expression | SymbAnaFis | SymPy | Ratio |
|------------|------------|-------|-------|
| `x^3 + 2x^2 + x` | 185 µs | 2284 µs | SymbAnaFis **12x faster** |
| `sin(x) * cos(x)` | 284 µs | 8851 µs | SymbAnaFis **31x faster** |
| `sin(x^2)` (chain rule) | 164 µs | 3145 µs | SymbAnaFis **19x faster** |
| `e^(x^2)` | 167 µs | 1305 µs | SymbAnaFis **8x faster** |
| `x^2 * sin(x) * exp(x)` | 850 µs | 18427 µs | SymbAnaFis **22x faster** |
| `(x^2 + 1) / (x - 1)` | 339 µs | 6388 µs | SymbAnaFis **19x faster** |
| `sin(cos(tan(x)))` | 422 µs | 12931 µs | SymbAnaFis **31x faster** |
| `x^x` | 169 µs | 2087 µs | SymbAnaFis **12x faster** |

### Differentiation (AST Only - No Simplification)

| Expression | SymbAnaFis |
|------------|------------|
| Polynomial | 243 ns |
| Trigonometric | 168 ns |
| Complex | 276 ns |
| Nested | 356 ns |

> These sub-microsecond times show the raw differentiation engine is extremely fast.

---

### Simplification

| Expression | SymbAnaFis | SymPy | Ratio |
|------------|------------|-------|-------|
| `sin²(x) + cos²(x)` → `1` | 128 µs | 3916 µs | SymbAnaFis **31x faster** |
| `x² + 2x + 1` → `(x+1)²` | 134 µs | 243 µs | SymbAnaFis **1.8x faster** |
| `(x+1)² / (x+1)` → `x+1` | 115 µs | 977 µs | SymbAnaFis **8.5x faster** |
| `e^x * e^y` → `e^(x+y)` | 113 µs | 1413 µs | SymbAnaFis **12.5x faster** |
| `2x + 3x + x` → `6x` | 116 µs | 475 µs | SymbAnaFis **4.1x faster** |
| `(x²+1)/(x²-1) + 1/(x+1)` | 948 µs | 3419 µs | SymbAnaFis **3.6x faster** |
| `(e^x - e^-x)/2` → `sinh(x)` | 150 µs | 3415 µs | SymbAnaFis **23x faster** |
| `x² * x³ / x` → `x⁴` | 142 µs | 482 µs | SymbAnaFis **3.4x faster** |

---

### Combined: Differentiation + Simplification

| Expression | SymbAnaFis | SymPy | Ratio |
|------------|------------|-------|-------|
| `d/dx[sin(x)²]` simplified | 294 µs | 2800 µs | SymbAnaFis **10x faster** |
| `d/dx[(x²+1)/(x-1)]` simplified | 658 µs | 6552 µs | SymbAnaFis **10x faster** |

---

### Real-World Example: Normal Distribution PDF

A complex real-world expression from statistics:

```
f(x) = exp(-(x - μ)² / (2σ²)) / √(2πσ²)
```

#### Raw Differentiation (No Simplification)

| Metric | SymbAnaFis | SymPy | Ratio |
|--------|------------|-------|-------|
| **Time** | **1.77 µs** | 33.78 µs | SymbAnaFis **19x faster** |
| **Output length** | 206 chars | 91 chars | SymPy more compact |
| **Characters × Time** | 365 | 3075 | SymbAnaFis 8x lower |

**SymbAnaFis raw output:**
```
(exp(-(x - mu)^2 / (2 * sigma^2)) * -2 * (x - mu)^1 * 2 * sigma^2 / (2 * sigma^2)^2 * sqrt(2 * pi * sigma^2) - exp(-(x - mu)^2 / (2 * sigma^2)) * 0 / (2 * sqrt(2 * pi * sigma^2))) / sqrt(2 * pi * sigma^2)^2
```

**SymPy raw output:**
```
-sqrt(2)*(-2*mu + 2*x)*exp(-(-mu + x)**2/(2*sigma**2))/(4*sqrt(pi)*sigma**2*sqrt(sigma**2))
```


#### With Simplification

| Metric | SymbAnaFis | SymPy | Ratio |
|--------|------------|-------|-------|
| **Time** | **18164.89 µs** | 87079.49 µs | SymbAnaFis **4.8x faster** |
| **Output length** | 94 chars | 84 chars | ~Same |

**SymbAnaFis simplified:**
```
-sqrt(2) * (x - mu) * abs(sigma) * exp(-(x - mu)^2 / (2 * sigma^2)) / (2 * sqrt(pi) * sigma^4)
```

**SymPy simplified:**
```
sqrt(2)*(mu - x)*exp(-(mu - x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma**2*sqrt(sigma**2))
```

#### Output Equivalence Verification

Both outputs are mathematically equivalent:

- **SymbAnaFis**: `(mu - x) * abs(sigma) / sigma^4` = `(mu - x) / sigma^3` (for σ > 0)
- **SymPy**: `(mu - x) / (sigma^2 * sqrt(sigma^2))` = `(mu - x) / sigma^3` (for σ > 0)

The exponential terms are identical: `exp(-(x - mu)^2 / (2 * sigma^2))` ≡ `exp(-(mu - x)^2 / (2 * sigma^2))`

Both simplify to the correct derivative of the normal PDF:
$$f'(x) = \frac{(\mu - x)}{\sigma^2} \cdot f(x) = \frac{\sqrt{2}(\mu - x)}{2\sqrt{\pi}\sigma^3} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

---

### Parsing Only

| Expression | SymbAnaFis |
|------------|------------|
| `x^3 + 2x^2 + x` | 610 ns |
| `sin(x) * cos(x)` | 556 ns |
| `x^2 * sin(x) * exp(x)` | 762 ns |
| `sin(cos(tan(x)))` | 555 ns |

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
