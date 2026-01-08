# CI Benchmark Results

**SymbAnaFis Version:** 0.5.1  
**Date:** Thu Jan  8 15:12:39 UTC 2026  
**Commit:** `c0149f856c65`  
**Rust:** 1.92.0  

> Auto-generated from Criterion benchmark output

## 1. Parsing (String → AST)

| Expression | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **3.32 µs** | 4.82 µs | **SymbAnaFis** (1.45x) |
| Damped Oscillator | **3.69 µs** | 5.40 µs | **SymbAnaFis** (1.46x) |
| Gaussian 2D | **6.66 µs** | 13.95 µs | **SymbAnaFis** (2.09x) |
| Lennard-Jones | **3.80 µs** | 7.64 µs | **SymbAnaFis** (2.01x) |
| Logistic Sigmoid | **3.22 µs** | 4.65 µs | **SymbAnaFis** (1.45x) |
| Lorentz Factor | **2.59 µs** | 4.92 µs | **SymbAnaFis** (1.90x) |
| Maxwell-Boltzmann | **7.74 µs** | 13.12 µs | **SymbAnaFis** (1.69x) |
| Normal PDF | **5.05 µs** | 10.13 µs | **SymbAnaFis** (2.00x) |
| Planck Blackbody | **4.92 µs** | 8.81 µs | **SymbAnaFis** (1.79x) |

---

## 2. Differentiation

| Expression | SymbAnaFis (Light) | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **2.19 µs** | 3.65 µs | **SymbAnaFis (Light)** (1.67x) |
| Damped Oscillator | **1.55 µs** | 3.54 µs | **SymbAnaFis (Light)** (2.28x) |
| Gaussian 2D | **1.47 µs** | 4.59 µs | **SymbAnaFis (Light)** (3.12x) |
| Lennard-Jones | **1.75 µs** | 4.00 µs | **SymbAnaFis (Light)** (2.28x) |
| Logistic Sigmoid | **854.77 ns** | 2.37 µs | **SymbAnaFis (Light)** (2.78x) |
| Lorentz Factor | **1.60 µs** | 3.77 µs | **SymbAnaFis (Light)** (2.36x) |
| Maxwell-Boltzmann | **2.43 µs** | 6.88 µs | **SymbAnaFis (Light)** (2.83x) |
| Normal PDF | **1.71 µs** | 3.42 µs | **SymbAnaFis (Light)** (2.00x) |
| Planck Blackbody | **2.06 µs** | 6.61 µs | **SymbAnaFis (Light)** (3.21x) |

---

## 3. Differentiation + Simplification

| Expression | SymbAnaFis (Full) | Speedup |
| :--- | :---: | :---: |
| Bessel Wave | 133.33 µs | - |
| Damped Oscillator | 155.97 µs | - |
| Gaussian 2D | 135.75 µs | - |
| Lennard-Jones | 29.77 µs | - |
| Logistic Sigmoid | 120.66 µs | - |
| Lorentz Factor | 265.94 µs | - |
| Maxwell-Boltzmann | 335.05 µs | - |
| Normal PDF | 147.87 µs | - |
| Planck Blackbody | 337.06 µs | - |

---

## 4. Simplification Only

| Expression | SymbAnaFis | Speedup |
| :--- | :---: | :---: |
| Bessel Wave | 129.78 µs | - |
| Damped Oscillator | 152.03 µs | - |
| Gaussian 2D | 132.89 µs | - |
| Lennard-Jones | 26.92 µs | - |
| Logistic Sigmoid | 118.59 µs | - |
| Lorentz Factor | 259.63 µs | - |
| Maxwell-Boltzmann | 331.75 µs | - |
| Normal PDF | 143.91 µs | - |
| Planck Blackbody | 334.17 µs | - |

---

## 5. Compilation

| Expression | SA (Simplified) | Symbolica | SA (Raw) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | **1.85 µs** | - | 2.15 µs | - |
| Damped Oscillator | **1.83 µs** | 15.21 µs | 2.08 µs | **SA (Simplified)** (8.31x) |
| Gaussian 2D | **1.98 µs** | 34.48 µs | 2.15 µs | **SA (Simplified)** (17.42x) |
| Lennard-Jones | 1.24 µs | 27.38 µs | **1.12 µs** | **SA (Simplified)** (22.11x) |
| Logistic Sigmoid | 1.55 µs | 9.98 µs | **1.38 µs** | **SA (Simplified)** (6.45x) |
| Lorentz Factor | **1.24 µs** | 9.66 µs | 1.51 µs | **SA (Simplified)** (7.78x) |
| Maxwell-Boltzmann | **2.08 µs** | 17.81 µs | 2.97 µs | **SA (Simplified)** (8.57x) |
| Normal PDF | **1.56 µs** | 18.45 µs | 1.67 µs | **SA (Simplified)** (11.80x) |
| Planck Blackbody | 2.94 µs | 10.83 µs | **2.91 µs** | **SA (Simplified)** (3.68x) |

---

## 6. Evaluation (1000 points)

| Expression | SA (Simplified) | Symbolica | SA (Raw) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | **197.63 µs** | - | 259.61 µs | - |
| Damped Oscillator | 88.96 µs | **58.05 µs** | 117.15 µs | **Symbolica** (1.53x) |
| Gaussian 2D | 111.37 µs | **64.62 µs** | 145.83 µs | **Symbolica** (1.72x) |
| Lennard-Jones | 93.70 µs | **63.71 µs** | 99.45 µs | **Symbolica** (1.47x) |
| Logistic Sigmoid | 131.55 µs | **51.19 µs** | 104.55 µs | **Symbolica** (2.57x) |
| Lorentz Factor | 72.72 µs | **54.56 µs** | 125.77 µs | **Symbolica** (1.33x) |
| Maxwell-Boltzmann | 135.07 µs | **81.49 µs** | 299.76 µs | **Symbolica** (1.66x) |
| Normal PDF | 98.13 µs | **63.86 µs** | 127.73 µs | **Symbolica** (1.54x) |
| Planck Blackbody | 169.53 µs | **64.22 µs** | 157.14 µs | **Symbolica** (2.64x) |

---

## 7. Full Pipeline

| Expression | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | 341.42 µs | - | - |
| Damped Oscillator | 267.39 µs | **177.88 µs** | **Symbolica** (1.50x) |
| Gaussian 2D | 282.21 µs | **160.04 µs** | **Symbolica** (1.76x) |
| Lennard-Jones | **128.18 µs** | 129.59 µs | **SymbAnaFis** (1.01x) |
| Logistic Sigmoid | 250.50 µs | **94.51 µs** | **Symbolica** (2.65x) |
| Lorentz Factor | 363.09 µs | **119.60 µs** | **Symbolica** (3.04x) |
| Maxwell-Boltzmann | 520.23 µs | **252.65 µs** | **Symbolica** (2.06x) |
| Normal PDF | 281.87 µs | **123.93 µs** | **Symbolica** (2.27x) |
| Planck Blackbody | 566.79 µs | **213.24 µs** | **Symbolica** (2.66x) |

---

## Parallel: Evaluation Methods (1k pts)

| Expression | Compiled Loop | Tree Walk | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **186.55 µs** | 1.24 ms | **Compiled Loop** (6.65x) |
| Damped Oscillator | **83.17 µs** | 1.03 ms | **Compiled Loop** (12.36x) |
| Gaussian 2D | **104.63 µs** | 1.84 ms | **Compiled Loop** (17.63x) |
| Lennard-Jones | **86.37 µs** | 667.77 µs | **Compiled Loop** (7.73x) |
| Logistic Sigmoid | **126.30 µs** | 1.10 ms | **Compiled Loop** (8.69x) |
| Lorentz Factor | **65.91 µs** | 821.82 µs | **Compiled Loop** (12.47x) |
| Maxwell-Boltzmann | **146.57 µs** | 1.30 ms | **Compiled Loop** (8.88x) |
| Normal PDF | **94.23 µs** | 1.08 ms | **Compiled Loop** (11.51x) |
| Planck Blackbody | **206.11 µs** | 2.01 ms | **Compiled Loop** (9.78x) |

---

## Parallel: Scaling (Points)

| Points | Eval Batch (SIMD) | Loop | Speedup |
| :--- | :---: | :---: | :---: |
| 100 | **2.17 µs** | 6.35 µs | **Eval Batch (SIMD)** (2.92x) |
| 1000 | **21.65 µs** | 63.74 µs | **Eval Batch (SIMD)** (2.94x) |
| 10000 | **216.66 µs** | 634.63 µs | **Eval Batch (SIMD)** (2.93x) |
| 100000 | **366.64 µs** | 6.36 ms | **Eval Batch (SIMD)** (17.34x) |

---

## Large Expressions (100 terms)

| Operation | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Parse | **149.58 µs** | 234.36 µs | **SA** (1.57x) |
| Diff (no simplify) | **87.90 µs** | 249.51 µs | **SA** (2.84x) |
| Diff+Simplify | 7.48 ms | — | — |
| Compile (simplified) | **35.92 µs** | 2.10 ms | **SA** (58.54x) |
| Eval 1000pts (simplified) | 5.07 ms | **3.73 ms** | **SY** (1.36x) |

---

## Large Expressions (300 terms)

| Operation | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Parse | **469.98 µs** | 727.41 µs | **SA** (1.55x) |
| Diff (no simplify) | **276.85 µs** | 791.48 µs | **SA** (2.86x) |
| Diff+Simplify | 22.05 ms | — | — |
| Compile (simplified) | **129.01 µs** | 15.20 ms | **SA** (117.80x) |
| Eval 1000pts (simplified) | 14.47 ms | **10.39 ms** | **SY** (1.39x) |

---

