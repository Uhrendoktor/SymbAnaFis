# Benchmark Results

**SymbAnaFis Version:** 0.4.0  
**Date:** 2025-12-27  
**System:** Linux (Plotters Backend)

## Test Expressions

| Name | Expression | Nodes | Domain |
|------|------------|-------|--------|
| Normal PDF | `exp(-(x-μ)²/(2σ²))/√(2πσ²)` | ~30 | Statistics |
| Gaussian 2D | `exp(-((x-x₀)²+(y-y₀)²)/(2s²))/(2πs²)` | ~40 | ML/Physics |
| Maxwell-Boltzmann | `4π(m/(2πkT))^(3/2) v² exp(-mv²/(2kT))` | ~50 | Physics |
| Lorentz Factor | `1/√(1-v²/c²)` | ~15 | Relativity |
| Lennard-Jones | `4ε((σ/r)¹² - (σ/r)⁶)` | ~25 | Chemistry |
| Logistic Sigmoid | `1/(1+exp(-k(x-x₀)))` | ~15 | ML |
| Damped Oscillator | `A·exp(-γt)·cos(ωt+φ)` | ~25 | Physics |
| Planck Blackbody | `2hν³/c² · 1/(exp(hν/(kT))-1)` | ~35 | Physics |

---

## 1. Parsing (String → AST)

| Expression | SymbAnaFis (SA) | Symbolica (SY) | Speedup (SA vs SY) |
|------------|-----------------|----------------|--------------------| 
| Normal PDF | 2.68 µs | 4.38 µs | **1.63x** |
| Gaussian 2D | 3.67 µs | 6.26 µs | **1.71x** |
| Maxwell-Boltzmann | 4.22 µs | 6.03 µs | **1.43x** |
| Lorentz Factor | 1.39 µs | 2.30 µs | **1.65x** |
| Lennard-Jones | 2.23 µs | 3.55 µs | **1.59x** |
| Logistic Sigmoid | 1.82 µs | 2.13 µs | **1.17x** |
| Damped Oscillator | 2.13 µs | 2.46 µs | **1.15x** |
| Planck Blackbody | 2.96 µs | 4.04 µs | **1.36x** |
| Bessel Wave | 1.95 µs | 2.24 µs | **1.15x** |

> **Result:** SymbAnaFis parses **1.2x - 1.8x** faster than Symbolica.

---

## 2. Differentiation (Raw - No Simplification)

| Expression | SymbAnaFis (SA) | Symbolica (SY) | Speedup (SA vs SY) |
|------------|-----------------|----------------|--------------------| 
| Normal PDF | 1.09 µs | 1.60 µs | **1.47x** |
| Gaussian 2D | 0.91 µs | 2.15 µs | **2.36x** |
| Maxwell-Boltzmann | 1.65 µs | 3.17 µs | **1.92x** |
| Lorentz Factor | 1.16 µs | 1.81 µs | **1.56x** |
| Lennard-Jones | 1.20 µs | 1.83 µs | **1.53x** |
| Logistic Sigmoid | 0.68 µs | 1.10 µs | **1.62x** |
| Damped Oscillator | 1.05 µs | 1.59 µs | **1.51x** |
| Planck Blackbody | 1.35 µs | 2.98 µs | **2.21x** |
| Bessel Wave | 1.47 µs | 1.62 µs | **1.10x** |

> **Result:** SymbAnaFis raw differentiation is **1.1x - 2.0x** faster.

---

## 3. Differentiation (Fair Comparison)

> **Methodology:** Both libraries tested with equivalent "light" simplification (term collection only, no deep restructuring).

| Expression | SA (diff_only) | Symbolica (diff) | SA Speedup |
|------------|----------------|------------------|------------|
| Normal PDF | 1.07 µs | 1.60 µs | **1.50x** |
| Gaussian 2D | 0.90 µs | 2.15 µs | **2.39x** |
| Maxwell-Boltzmann | 1.63 µs | 3.18 µs | **1.95x** |
| Lorentz Factor | 1.17 µs | 1.80 µs | **1.54x** |
| Lennard-Jones | 1.28 µs | 1.83 µs | **1.43x** |
| Logistic Sigmoid | 0.68 µs | 1.10 µs | **1.62x** |
| Damped Oscillator | 1.04 µs | 1.59 µs | **1.53x** |
| Planck Blackbody | 1.33 µs | 2.95 µs | **2.22x** |
| Bessel Wave | 1.43 µs | 1.61 µs | **1.13x** |

### SymbAnaFis Full Simplification Cost

| Expression | SA diff_only | SA diff+simplify | Simplify Overhead |
|------------|--------------|------------------|-------------------|
| Normal PDF | 1.07 µs | 103 µs | **96x** |
| Gaussian 2D | 0.90 µs | 87.8 µs | **98x** |
| Maxwell-Boltzmann | 1.63 µs | 164 µs | **101x** |
| Lorentz Factor | 1.17 µs | 176 µs | **150x** |
| Lennard-Jones | 1.28 µs | 17.7 µs | **14x** |
| Logistic Sigmoid | 0.68 µs | 79.3 µs | **117x** |
| Damped Oscillator | 1.04 µs | 115 µs | **111x** |
| Planck Blackbody | 1.33 µs | 192 µs | **144x** |
| Bessel Wave | 1.43 µs | 98.9 µs | **69x** |

> **Note:** SymbAnaFis full simplification performs deep AST restructuring (trig identities, algebraic transformations). Symbolica only performs light term collection.

---

## 4. Simplification Only (SymbAnaFis)

| Expression | Time |
|------------|------|
| Normal PDF | 100 µs |
| Gaussian 2D | 85.7 µs |
| Maxwell-Boltzmann | 164 µs |
| Lorentz Factor | 170 µs |
| Lennard-Jones | 15.9 µs |
| Logistic Sigmoid | 78.6 µs |
| Damped Oscillator | 114 µs |
| Planck Blackbody | 188 µs |
| Bessel Wave | 96.9 µs |

---

## 5. Compilation (AST → Bytecode/Evaluator)

> **Note:** Times shown are for compiling the **simplified** expression (post-differentiation).

| Expression | SymbAnaFis (SA) | Symbolica (SY) | Speedup (SA vs SY) |
|------------|-----------------|----------------|--------------------| 
| Normal PDF | 0.69 µs | 8.78 µs | **12.7x** |
| Gaussian 2D | 0.89 µs | 16.2 µs | **18.2x** |
| Maxwell-Boltzmann | 0.94 µs | 8.32 µs | **8.9x** |
| Lorentz Factor | 0.55 µs | 4.79 µs | **8.7x** |
| Lennard-Jones | 0.55 µs | 12.8 µs | **23.3x** |
| Logistic Sigmoid | 0.69 µs | 4.92 µs | **7.1x** |
| Damped Oscillator | 0.83 µs | 7.46 µs | **9.0x** |
| Planck Blackbody | 1.52 µs | 4.90 µs | **3.2x** |
| Bessel Wave | 0.82 µs | *(skipped)* | — |

> **Result:** SymbAnaFis compilation is **3.6x - 25x** faster than Symbolica's evaluator creation.

---

## 6. Evaluation (Compiled, 1000 points)

| Expression | SymbAnaFis (Simpl) | Symbolica (SY) | SA vs SY |
|------------|--------------------|----------------|--------------------| 
| Normal PDF | 53.7 µs | 33.0 µs | 0.61x |
| Gaussian 2D | 57.6 µs | 34.4 µs | 0.60x |
| Maxwell-Boltzmann | 60.7 µs | 42.1 µs | 0.69x |
| Lorentz Factor | 41.6 µs | 32.3 µs | 0.78x |
| Lennard-Jones | 46.3 µs | 34.6 µs | **0.75x** |
| Logistic Sigmoid | 75.7 µs | 29.9 µs | 0.39x |
| Damped Oscillator | 43.6 µs | 32.9 µs | 0.75x |
| Planck Blackbody | 69.9 µs | 32.0 µs | 0.46x |
| Bessel Wave | 79.1 µs | *(skipped)* | — |

> **Result:** With SIMD (f64x4), SymbAnaFis closed the gap to **0.4x - 0.8x** of Symbolica's speed. The **Lorentz Factor** benchmark now runs at **80%** of Symbolica's performance!

---

## 7. Full Pipeline (Parse → Diff → Simplify → Compile → Eval 1000 pts)

| Expression | SymbAnaFis (SA) | Symbolica (SY) | Speedup (SY vs SA) |
|------------|-----------------|----------------|--------------------| 
| Normal PDF | 167 µs | 53.0 µs | **3.2x** |
| Gaussian 2D | 151 µs | 70.4 µs | **2.1x** |
| Maxwell-Boltzmann | 247 µs | 113 µs | **2.2x** |
| Lorentz Factor | 223 µs | 57.8 µs | **3.9x** |
| Lennard-Jones | 63.5 µs | 61.7 µs | **1.03x** |
| Logistic Sigmoid | 147 µs | 47.8 µs | **3.1x** |
| Damped Oscillator | 164 µs | 78.9 µs | **2.1x** |
| Planck Blackbody | 293 µs | 95.5 µs | **3.1x** |
| Bessel Wave | 184 µs | *(skipped)* | — |

> **Result:** Symbolica is **1.9x - 4.8x** faster in the full pipeline, mainly due to:
> 1. Lighter simplification (only term collection vs full restructuring)
> 2. Faster evaluation engine

---

## 8. Large Expressions (100-300 terms)

> **Note:** Large expressions with mixed terms (polynomials, trig, exp, log, fractions).

### 100 Terms

| Operation | SymbAnaFis | Symbolica | Speedup |
|-----------|------------|-----------|---------|
| Parse | 75.4 µs | 107 µs | **SA 1.4x** |
| Diff (no simplify) | 47.6 µs | 113 µs | **SA 2.4x** |
| Compile (simplified) | 14.6 µs | 1,030 µs | **SA 71x** |
| Eval 1000pts (simplified) | 1,540 µs | 1,480 µs | **SY 1.04x** |

### 300 Terms

| Operation | SymbAnaFis | Symbolica | Speedup |
|-----------|------------|-----------|---------|
| Parse | 231 µs | 342 µs | **SA 1.5x** |
| Diff (no simplify) | 143 µs | 338 µs | **SA 2.4x** |
| Compile (simplified) | 44.8 µs | 12,600 µs | **SA 281x** |
| Eval 1000pts (simplified) | 4,920 µs | 4,130 µs | **SY 1.2x** |

> **Key Insight:** After SymbAnaFis's full simplification, compilation is dramatically faster (71x-281x). Symbolica's evaluator is still faster for large expressions (1.04x-1.2x), but the gap narrowed significantly.

---

## 9. Tree-Walk vs Compiled Evaluation

> **Note:** Compares generalized `evaluate()` (HashMap-based tree-walk) vs compiled bytecode evaluation.

| Expression | Tree-Walk (1000 pts) | Compiled (1000 pts) | Speedup |
|------------|----------------------|---------------------|---------|
| Normal PDF | 514 µs | 51 µs | **10.0x** |
| Gaussian 2D | 1,006 µs | 55 µs | **18.3x** |
| Maxwell-Boltzmann | 597 µs | 67 µs | **8.9x** |
| Lorentz Factor | 397 µs | 39 µs | **10.2x** |
| Lennard-Jones | 319 µs | 45 µs | **7.1x** |
| Logistic Sigmoid | 511 µs | 73 µs | **7.0x** |
| Damped Oscillator | 461 µs | 39 µs | **11.8x** |
| Planck Blackbody | 914 µs | 81 µs | **11.3x** |
| Bessel Wave | 574 µs | 80 µs | **7.2x** |

> **Result:** Compiled evaluation is **7x - 18x faster** than tree-walk evaluation. Use `CompiledEvaluator` for repeated evaluation of the same expression.

---

## 10. Batch Evaluation Performance (SIMD-optimized)

> **Note:** `eval_batch` now uses f64x4 SIMD to process 4 values simultaneously.

| Points | loop_evaluate | eval_batch (SIMD) | Speedup |
|--------|---------------|-------------------|---------|
| 100 | 3.51 µs | 1.20 µs | **2.9x** |
| 1,000 | 35.1 µs | 12.2 µs | **2.9x** |
| 10,000 | 351 µs | 122 µs | **2.9x** |
| 100,000 | 3.52 ms | 1.22 ms | **2.9x** |

> **Result:** SIMD-optimized `eval_batch` is now **~2.9x faster** than loop evaluation by processing 4 f64 values per instruction using f64x4 vectors.

---

## 11. Multi-Expression Batch Evaluation

> **Note:** Evaluates 3 different expressions (Lorentz, Quadratic, Trig) × 1000 points each.

| Method | Time | vs Sequential |
|--------|------|---------------|
| **eval_batch_per_expr (SIMD)** | **22.6 µs** | **58% faster** |
| eval_f64_per_expr (SIMD+parallel) | 35.1 µs | 35% faster |
| sequential_loops | 54.2 µs | baseline |

> **Result:** SIMD-optimized `eval_batch` is **~2.4x faster** than sequential evaluation loops when processing multiple expressions.

---

## 12. eval_f64 vs evaluate_parallel APIs

> **Note:** Compares the two high-level parallel evaluation APIs.

### `eval_f64` vs `evaluate_parallel` (High Load - 10,000 points)

| API | Time | Notes |
|-----|------|-------|
| `eval_f64` (SIMD+parallel) | **40 µs** | **5.8x Faster**. Uses f64x4 SIMD + chunked parallelism. |
| `evaluate_parallel` | 231 µs | Slower due to per-point evaluation overhead. |

**Result:** `eval_f64` scales significantly better. For 10,000 points, it is **~5.8x faster** than the general API.
- `eval_f64` uses `&[f64]` (8 bytes/item) -> Cache friendly.
- `evaluate_parallel` uses `Vec<Value>` (24 bytes/item) -> Memory bound.
- Zero-allocation optimization on `evaluate_parallel` showed no gain, confirming the bottleneck is data layout, not allocator contention.

---

## Summary

| Operation | Winner | Speedup |
|-----------|--------|---------|
| **Parsing** | SymbAnaFis | **1.1x - 1.7x** faster |
| **Differentiation** | SymbAnaFis | **1.1x - 2.4x** faster |
| **Compilation** | SymbAnaFis | **3.2x - 280x** faster |
| **Tree-Walk → Compiled** | Compiled | **5x - 13x** faster |
| **eval_batch vs loop** | eval_batch | **~16%** faster |
| **Evaluation** (small expr) | Symbolica | **1.3x - 2.6x** faster |
| **Evaluation** (large expr, simplified) | Symbolica | **1.0x - 1.2x** faster |
| **Full Pipeline** (small) | Symbolica | **1.0x - 3.9x** faster |

### Key Insights

1. **Compile for repeated evaluation:** Compiled bytecode is 5-13x faster than tree-walk evaluation.

2. **Simplification pays off:** For large expressions, SymbAnaFis's full simplification dramatically reduces expression size, leading to much faster compilation and evaluation.

3. **Different strategies:**
   - **Symbolica:** Light term collection (`3x + 2x → 5x`), faster simplification, optimized evaluator
   - **SymbAnaFis:** Deep AST restructuring (trig identities, algebraic normalization), massive compilation speedup

4. **Batch evaluation helps:** Using `eval_batch` provides ~16% speedup over calling `evaluate()` in a loop.

5. **When to use which:**
   - **Small expressions, one-shot evaluation:** Symbolica's faster evaluation wins
   - **Large expressions, repeated evaluation:** SymbAnaFis's simplification + fast compile wins
```
