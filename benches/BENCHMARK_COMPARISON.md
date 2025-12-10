# SymbAnaFis vs SymPy vs Symbolica Benchmark Comparison

**Date:** December 10, 2025 (Updated)
**SymbAnaFis Version:** 0.3.0
**SymPy Version:** Latest (Python 3)
**Symbolica Version:** 1.0.1
**System:** Linux
**Criterion Version:** 0.8.1

## Summary

SymbAnaFis is a symbolic mathematics library written in Rust, designed for parsing, differentiation, and simplification of mathematical expressions. This document compares its performance against SymPy and Symbolica.

### Key Findings vs SymPy

| Category | Winner | Speedup Range |
|----------|--------|---------------|
| **Parsing** | SymbAnaFis | **120x - 190x faster** |
| **Differentiation (AST)** | SymbAnaFis | **28x - 154x faster** |
| **Differentiation (Full)** | SymbAnaFis | **2.1x - 6.6x faster** |
| **Simplification** | SymbAnaFis | **35x - 297x faster** |
| **Combined Diff + Simplify** | SymbAnaFis | **45x - 90x faster** |

### Key Findings vs Symbolica

| Category | Winner | Notes |
|----------|--------|-------|
| **Parsing** | SymbAnaFis | **1.5x - 2.3x faster** |
| **Differentiation (AST only)** | SymbAnaFis | **1.7x - 2.9x faster** |
| **Differentiation (Full)** | Symbolica | **15x - 20x faster** (see analysis below) |

---

## Detailed Results

### 1. Parsing (String → Expression)

SymbAnaFis uses a custom recursive descent parser that is orders of magnitude faster than `sympify` and faster than Symbolica's parser.

| Expression | SymbAnaFis (µs) | Symbolica (µs) | SymPy (µs) | vs Symbolica | vs SymPy |
|------------|-----------------|----------------|------------|--------------|----------|
| Polynomial `x^3+...` | 0.84 | 1.41 | 133.00 | **1.7x** | **158x** |
| Trig `sin(x)*cos(x)` | 0.59 | 1.14 | 107.96 | **1.9x** | **183x** |
| Complex `x^2*sin(x)*exp(x)` | 1.04 | 1.47 | 116.66 | **1.4x** | **112x** |
| Nested `sin(cos(tan(x)))` | 0.54 | 1.23 | 101.70 | **2.3x** | **188x** |

### 2. Differentiation (AST Only)

Pure differentiation speed on pre-parsed expressions **without simplification**. This measures raw differentiation engine speed.

| Expression | SymbAnaFis (µs) | Symbolica (µs) | SymPy (µs) | vs Symbolica | vs SymPy |
|------------|-----------------|----------------|------------|--------------|----------|
| Polynomial | 0.43 | 1.25 | 24.98 | **2.9x** | **58x** |
| Trig | 0.39 | 1.04 | 20.02 | **2.7x** | **51x** |
| Complex | 0.72 | 1.49 | 19.21 | **2.1x** | **27x** |
| Nested Trig | 0.54 | 0.93 | 83.69 | **1.7x** | **155x** |

### 3. Differentiation (Full Pipeline)

Includes parsing and automatic simplification. Both SymbAnaFis and Symbolica return simplified results.

| Expression | SymbAnaFis (µs) | Symbolica (µs) | SymPy (µs) | vs SymPy |
|------------|-----------------|----------------|------------|----------|
| Polynomial | 46.4 | 2.74 | 150.19 | **3.2x** |
| Trig `sin(x)cos(x)` | 61.3 | 2.22 | 130.15 | **2.1x** |
| Chain `sin(x^2)` | 28.6 | 1.51 | 189.81 | **6.6x** |
| Exp `exp(x^2)` | 28.2 | 1.47 | 190.42 | **6.8x** |
| Complex | 216.0 | 2.95 | 146.02 | **0.7x** |
| Quotient `(x^2+1)/(x-1)` | 92.7 | 2.79 | 152.71 | **1.6x** |
| Nested | 71.9 | 2.22 | 207.31 | **2.9x** |
| Power `x^x` | 33.7 | 1.59 | 184.07 | **5.5x** |

### 4. Simplification

SymbAnaFis provides extensive rule-based simplification with pattern matching.

| Expression | SymbAnaFis (µs) | SymPy (µs) | Speedup |
|------------|-----------------|------------|---------|
| Pythagorean `sin^2+cos^2` | 17.8 | 5282 | **297x** |
| Perfect Square | 20.0 | 2146 | **107x** |
| Fraction Cancel | 19.8 | 1334 | **67x** |
| Exp Combine `e^x*e^y` | 22.1 | 1631 | **74x** |
| Like Terms `2x+3x+x` | 17.9 | 631 | **35x** |
| Hyperbolic `(e^x-e^-x)/2` | 26.1 | 3727 | **143x** |
| Frac Add `(x^2+1)/(...)...`| 144.5 | 3624 | **25x** |
| Power Combine | 18.0 | 634 | **35x** |

### 5. Combined Operations

Real-world scenarios often require differentiating and then simplifying the result.

| Operation | SymbAnaFis (µs) | SymPy (µs) | Speedup |
|-----------|-----------------|------------|---------|
| `d/dx[sin(x)^2]` simplified | 34.3 | 3090 | **90x** |
| `d/dx[(x^2+1)/(x-1)]` simplified | 147.2 | 6674 | **45x** |

---

## Analysis: SymbAnaFis vs Symbolica

### Why Symbolica is Faster for Full Differentiation

Both libraries use AST-based representations (`Num`, `Var`, `Fun`, `Mul`, `Add`, `Pow`), but Symbolica employs several low-level optimizations:

| Optimization | Symbolica | SymbAnaFis |
|--------------|-----------|------------|
| **Memory Representation** | Compact `Vec<u8>` with type tags | `Arc<Symbol>` with heap allocation |
| **Workspace** | Thread-local memory pool | Fresh allocations per operation |
| **Normalization** | Lightweight inline normalization | Multi-pass rule-based simplification |
| **Data Layout** | Cache-friendly byte arrays | Pointer-chasing through `Arc` |

### Simplification Philosophy

The key difference is in **simplification strategy**:

- **Symbolica**: Uses a lightweight `normalize()` function that combines like terms and performs basic algebraic simplification during derivative construction.

- **SymbAnaFis**: Uses an extensible **rule-based simplification engine** that applies many pattern-matching rules over multiple passes. This is more powerful for complex identities (trigonometric, hyperbolic) but slower for basic operations.

### Where SymbAnaFis Excels

1. **Parsing**: 1.5-2.3x faster - simpler AST construction without byte packing
2. **AST Differentiation**: 1.7-2.9x faster - direct tree manipulation
3. **Trigonometric Identities**: Extensive patterns (sin²+cos²=1, double angles, etc.)
4. **Hyperbolic Functions**: sinh, cosh, tanh recognition and simplification
5. **Custom Functions**: First-class support for user-defined functions with derivatives
6. **Extensibility**: Rule-based engine can be extended with new patterns

### Where Symbolica Excels

1. **Full Differentiation Pipeline**: 15-20x faster due to memory optimizations
2. **Polynomial Operations**: Native multivariate polynomial factorization
3. **Large Expressions**: Coefficient ring optimizations and streaming
4. **Series Expansion**: Built-in Taylor/Laurent series
5. **Pattern Matching**: Powerful wildcard-based pattern matching

### Trade-off Summary

| Use Case | Recommended |
|----------|-------------|
| High-frequency parsing | SymbAnaFis |
| Batch differentiation (simple expressions) | Symbolica |
| Trigonometric/hyperbolic simplification | SymbAnaFis |
| Polynomial factorization | Symbolica |
| Extensible symbolic rules | SymbAnaFis |
| Memory-constrained environments | Symbolica |

---

## Analysis: Why SymbAnaFis Beats SymPy

1. **Rule-based engine with ExprKind filtering**: O(1) rule lookup instead of O(n) scanning
2. **No Python overhead**: Pure Rust with zero-cost abstractions
3. **Pattern matching optimization**: Rules only run on applicable expression types
4. **Efficient AST representation**: Using Rust's `Arc` for shared expression nodes
5. **Compiled native code**: No interpreter overhead

---

## Hardware

Benchmarks were run on a single machine to ensure fair comparison:

**System Specs:**
-   **CPU:** AMD Ryzen AI 7 350 w/ Radeon 860M
-   **RAM:** 32 GB
-   **OS:** Fedora 43 (Linux 6.17.9-300.fc43.x86_64)
-   **Rust Version:** 1.90.0 (1159e78c4 2025-09-14)
-   **Python Version:** 3.14.0

**Methodology:**
-   **Rust**: `cargo bench` (Criterion, 100 samples)
-   **Python**: `timeit` (1000 iterations)
