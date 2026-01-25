# Benchmark Results

**SymbAnaFis Version:** Unrealesed Dev Build  
**Date:** 2026-01-24
---

## System Specifications

- **CPU:** AMD Ryzen AI 7 350 w/ Radeon 860M (8 cores, 16 threads)
- **CPU Max:** 5.09 GHz
- **RAM:** 32 GB (30 GiB total)
- **OS:** Linux 6.17.12 (Fedora 43)
- **Rust:** rustc 1.93.0
- **Backend:** Plotters

## Test Expressions

| Name              | Expression                              | Nodes | Domain     |
| ----------------- | --------------------------------------- | ----- | ---------- |
| Normal PDF        | `exp(-(x-μ)²/(2σ²))/√(2πσ²)`            | ~30   | Statistics |
| Gaussian 2D       | `exp(-((x-x₀)²+(y-y₀)²)/(2s²))/(2πs²)`  | ~40   | ML/Physics |
| Maxwell-Boltzmann | `4π(m/(2πkT))^(3/2) v² exp(-mv²/(2kT))` | ~50   | Physics    |
| Lorentz Factor    | `1/√(1-v²/c²)`                          | ~15   | Relativity |
| Lennard-Jones     | `4ε((σ/r)¹² - (σ/r)⁶)`                  | ~25   | Chemistry  |
| Logistic Sigmoid  | `1/(1+exp(-k(x-x₀)))`                   | ~15   | ML         |
| Damped Oscillator | `A·exp(-γt)·cos(ωt+φ)`                  | ~25   | Physics    |
| Planck Blackbody  | `2hν³/c² · 1/(exp(hν/(kT))-1)`          | ~35   | Physics    |
| Bessel Wave       | `besselj(0,k*r)*cos(ω*t)`               | ~10   | Physics    |

---

## 1. Parsing (String → AST)

| Expression        | SymbAnaFis (SA) | Symbolica (SY) | Speedup (SA vs SY) |
| ----------------- | --------------: | -------------: | -----------------: |
| Normal PDF        |        2.562 µs |       4.330 µs |          **1.69x** |
| Gaussian 2D       |        3.551 µs |       6.116 µs |          **1.72x** |
| Maxwell-Boltzmann |        4.207 µs |       5.840 µs |          **1.39x** |
| Lorentz Factor    |        1.371 µs |       2.268 µs |          **1.65x** |
| Lennard-Jones     |        2.095 µs |       3.511 µs |          **1.67x** |
| Logistic Sigmoid  |        1.642 µs |       2.162 µs |          **1.32x** |
| Damped Oscillator |        1.927 µs |       2.501 µs |          **1.30x** |
| Planck Blackbody  |        2.578 µs |       3.907 µs |          **1.52x** |
| Bessel Wave       |        1.718 µs |       2.237 µs |          **1.30x** |

> **Result:** SymbAnaFis parses **1.2x - 1.7x** faster than Symbolica.

---

## 2. Differentiation (Light)

| Expression        | SA (diff_only) | Symbolica (diff) | SA Speedup |
| ----------------- | -------------: | ---------------: | ---------: |
| Normal PDF        |       0.985 µs |         1.641 µs |  **1.66x** |
| Gaussian 2D       |       0.830 µs |         2.204 µs |  **2.66x** |
| Maxwell-Boltzmann |       1.534 µs |         3.235 µs |  **2.11x** |
| Lorentz Factor    |       0.998 µs |         1.782 µs |  **1.78x** |
| Lennard-Jones     |       1.188 µs |         1.832 µs |  **1.54x** |
| Logistic Sigmoid  |       0.541 µs |         1.100 µs |  **2.03x** |
| Damped Oscillator |       1.001 µs |         1.606 µs |  **1.60x** |
| Planck Blackbody  |       1.302 µs |         3.025 µs |  **2.32x** |
| Bessel Wave       |       1.396 µs |         1.663 µs |  **1.19x** |

### SymbAnaFis Full Simplification Cost

| Expression        | SA diff_only | SA diff+simplify |
| ----------------- | -----------: | ---------------: |
| Normal PDF        |     0.985 µs |        76.502 µs |
| Gaussian 2D       |     0.830 µs |        70.010 µs |
| Maxwell-Boltzmann |     1.534 µs |       173.970 µs |
| Lorentz Factor    |     0.998 µs |       134.650 µs |
| Lennard-Jones     |     1.188 µs |        15.401 µs |
| Logistic Sigmoid  |     0.541 µs |        61.976 µs |
| Damped Oscillator |     1.001 µs |        83.571 µs |
| Planck Blackbody  |     1.302 µs |        181.00 µs |
| Bessel Wave       |     1.396 µs |        69.138 µs |

---

## 4. Simplification Only (SymbAnaFis)

| Expression        | Time (median) |
| ----------------- | ------------: |
| Normal PDF        |     75.623 µs |
| Gaussian 2D       |     69.337 µs |
| Maxwell-Boltzmann |     173.68 µs |
| Lorentz Factor    |     133.28 µs |
| Lennard-Jones     |     14.528 µs |
| Logistic Sigmoid  |     61.403 µs |
| Damped Oscillator |     82.229 µs |
| Planck Blackbody  |     178.49 µs |
| Bessel Wave       |     67.810 µs |

---

## 5. Compilation (simplified) (medians)

| Expression        | SymbAnaFis (SA) | Symbolica (SY) | Speedup (SA vs SY) |
| ----------------- | --------------: | -------------: | -----------------: |
| Normal PDF        |        0.699 µs |       8.777 µs |          **12.6x** |
| Gaussian 2D       |        0.694 µs |       16.32 µs |          **23.5x** |
| Maxwell-Boltzmann |        1.079 µs |       8.308 µs |           **7.7x** |
| Lorentz Factor    |        0.476 µs |       4.819 µs |          **10.1x** |
| Lennard-Jones     |        0.531 µs |       12.93 µs |          **24.4x** |
| Logistic Sigmoid  |        0.456 µs |       4.910 µs |          **10.8x** |
| Damped Oscillator |        0.551 µs |       7.363 µs |          **13.4x** |
| Planck Blackbody  |        1.004 µs |       4.982 µs |           **5.0x** |
| Bessel Wave       |        0.777 µs |    *(skipped)* |                  — |

---

## 6. Evaluation (Compiled, 1000 points) (medians)

| Expression        | SymbAnaFis (Simpl) | Symbolica (SY) | SA vs SY |
| ----------------- | -----------------: | -------------: | -------: |
| Normal PDF        |           41.79 µs |       32.93 µs |    0.80x |
| Gaussian 2D       |           41.56 µs |       33.42 µs |    0.80x |
| Maxwell-Boltzmann |           70.57 µs |       42.20 µs |    0.59x |
| Lorentz Factor    |           33.02 µs |       32.31 µs |    0.98x |
| Lennard-Jones     |           30.47 µs |       34.15 µs |    1.12x |
| Logistic Sigmoid  |           29.86 µs |       29.80 µs |    1.00x |
| Damped Oscillator |           43.15 µs |       32.90 µs |    0.76x |
| Planck Blackbody  |           57.08 µs |       32.02 µs |    0.56x |
| Bessel Wave       |          100.02 µs |    *(skipped)* |        — |

---

## 7. Full Pipeline (Parse → Diff → Simplify → Compile → Eval 1000 pts)

| Expression        | SA (Full Simp) | SA (No Simp) | Symbolica (SY) | SA Full vs SY | SA No-Simp vs SY |
| ----------------- | -------------: | -----------: | -------------: | ------------: | ---------------: |
| Normal PDF        |      120.17 µs |     47.44 µs |       49.03 µs |         0.41x |        **1.03x** |
| Gaussian 2D       |      118.87 µs |     52.62 µs |       65.33 µs |         0.55x |        **1.24x** |
| Maxwell-Boltzmann |      252.66 µs |     86.37 µs |      108.53 µs |         0.43x |        **1.26x** |
| Lorentz Factor    |      165.65 µs |     36.16 µs |       53.52 µs |         0.32x |        **1.48x** |
| Lennard-Jones     |       60.75 µs |     51.24 µs |       56.29 µs |         0.93x |        **1.10x** |
| Logistic Sigmoid  |       95.68 µs |     29.11 µs |       43.09 µs |         0.45x |        **1.48x** |
| Damped Oscillator |      134.46 µs |     52.46 µs |       74.54 µs |         0.55x |        **1.42x** |
| Planck Blackbody  |      248.94 µs |     56.40 µs |       90.78 µs |         0.36x |        **1.61x** |
| Bessel Wave       |      181.60 µs |    114.16 µs |    *(skipped)* |             — |                — |

> **Key Finding:** Without full simplification, SymbAnaFis beats Symbolica on **all 8 expressions** (avg **1.33x faster**).
> The performance gap with full simplification is entirely due to deep algebraic restructuring (60-180µs overhead).

---

## 8. Large Expressions (100 / 300 terms) (medians)

### 100 Terms

| Operation                 | SymbAnaFis | Symbolica |      Speedup |
| ------------------------- | ---------: | --------: | -----------: |
| Parse                     |   72.90 µs | 107.13 µs | **SA 1.47x** |
| Diff (no simplify)        |   47.86 µs | 112.54 µs | **SA 2.35x** |
| Diff+Simplify             |   3.806 ms |         — |            — |
| Compile (raw)             |  102.37 µs |  1.039 ms | **SA 10.1x** |
| Compile (simplified)      |   32.75 µs |  1.039 ms | **SA 31.7x** |
| Eval 1000pts (raw)        |   2.411 ms |  1.875 ms |        0.78x |
| Eval 1000pts (simplified) |   1.543 ms |  1.875 ms | **SA 1.22x** |

### 300 Terms

| Operation                 | SymbAnaFis | Symbolica |      Speedup |
| ------------------------- | ---------: | --------: | -----------: |
| Parse                     |  223.18 µs | 334.21 µs | **SA 1.50x** |
| Diff (no simplify)        |  145.29 µs | 367.81 µs | **SA 2.53x** |
| Diff+Simplify             |   10.84 ms |         — |            — |
| Compile (raw)             |  771.87 µs | 12.498 ms | **SA 16.2x** |
| Compile (simplified)      |  165.25 µs | 12.498 ms | **SA 75.7x** |
| Eval 1000pts (raw)        |   7.123 ms |  5.350 ms |        0.75x |
| Eval 1000pts (simplified) |   4.536 ms |  5.350 ms | **SA 1.18x** |

---

## 9. Tree-Walk vs Compiled Evaluation (medians)

| Expression        | Tree-Walk (1000 pts) | Compiled (1000 pts) |   Speedup |
| ----------------- | -------------------: | ------------------: | --------: |
| Normal PDF        |            491.75 µs |            38.97 µs | **12.6x** |
| Gaussian 2D       |            496.01 µs |            35.79 µs | **13.9x** |
| Maxwell-Boltzmann |            595.78 µs |            59.12 µs | **10.1x** |
| Lorentz Factor    |            381.33 µs |            25.97 µs | **14.7x** |
| Lennard-Jones     |            317.74 µs |            22.80 µs | **13.9x** |
| Logistic Sigmoid  |            503.32 µs |            23.30 µs | **21.6x** |
| Damped Oscillator |            449.75 µs |            36.01 µs | **12.5x** |
| Planck Blackbody  |            913.84 µs |            43.99 µs | **20.8x** |
| Bessel Wave       |            571.58 µs |            96.23 µs |  **5.9x** |

---

## 10. Batch Evaluation Performance (SIMD)

| Points  | loop_evaluate | eval_batch (SIMD) |  Speedup |
| ------- | ------------: | ----------------: | -------: |
| 100     |      2.276 µs |          1.156 µs | **2.0x** |
| 1,000   |      22.40 µs |          11.54 µs | **1.9x** |
| 10,000  |     226.65 µs |         115.42 µs | **2.0x** |
| 100,000 |      2.276 ms |          1.155 ms | **2.0x** |

---

## 11. Multi-Expression Batch Evaluation

| Method                            |     Time | vs Sequential |
| --------------------------------- | -------: | ------------: |
| eval_batch_per_expr (SIMD)        | 21.43 µs |   2.1x faster |
| eval_f64_per_expr (SIMD+parallel) | 39.57 µs |   1.1x faster |
| sequential_loops                  | 41.77 µs |      baseline |

---

## 12. eval_f64 vs evaluate_parallel APIs

| API                        |      Time |
| -------------------------- | --------: |
| `eval_f64` (SIMD+parallel) |  45.21 µs |
| `evaluate_parallel`        | 228.83 µs |

