# CI Benchmark Results

**SymbAnaFis Version:** 0.7.0  
**Date:** Fri Jan 23 01:49:00 UTC 2026  
**Commit:** `618253bb6c5c`  
**Rust:** 1.93.0  

> Auto-generated from Criterion benchmark output

## 1. Parsing (String → AST)

| Expression | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **3.30 µs** | 4.96 µs | **SymbAnaFis** (1.50x) |
| Damped Oscillator | **3.74 µs** | 5.51 µs | **SymbAnaFis** (1.47x) |
| Gaussian 2D | **6.48 µs** | 13.87 µs | **SymbAnaFis** (2.14x) |
| Lennard-Jones | **3.87 µs** | 7.60 µs | **SymbAnaFis** (1.96x) |
| Logistic Sigmoid | **2.93 µs** | 4.68 µs | **SymbAnaFis** (1.60x) |
| Lorentz Factor | **2.51 µs** | 4.85 µs | **SymbAnaFis** (1.93x) |
| Maxwell-Boltzmann | **7.89 µs** | 12.96 µs | **SymbAnaFis** (1.64x) |
| Normal PDF | **5.14 µs** | 10.01 µs | **SymbAnaFis** (1.95x) |
| Planck Blackbody | **5.05 µs** | 8.75 µs | **SymbAnaFis** (1.73x) |

---

## 2. Differentiation

| Expression | SymbAnaFis (Light) | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **2.21 µs** | 3.63 µs | **SymbAnaFis (Light)** (1.64x) |
| Damped Oscillator | **1.55 µs** | 3.45 µs | **SymbAnaFis (Light)** (2.22x) |
| Gaussian 2D | **1.43 µs** | 4.46 µs | **SymbAnaFis (Light)** (3.12x) |
| Lennard-Jones | **1.79 µs** | 3.98 µs | **SymbAnaFis (Light)** (2.22x) |
| Logistic Sigmoid | **853.49 ns** | 2.36 µs | **SymbAnaFis (Light)** (2.77x) |
| Lorentz Factor | **1.60 µs** | 3.78 µs | **SymbAnaFis (Light)** (2.36x) |
| Maxwell-Boltzmann | **2.44 µs** | 6.91 µs | **SymbAnaFis (Light)** (2.83x) |
| Normal PDF | **1.68 µs** | 3.44 µs | **SymbAnaFis (Light)** (2.04x) |
| Planck Blackbody | **2.10 µs** | 6.64 µs | **SymbAnaFis (Light)** (3.16x) |

---

## 3. Differentiation + Simplification

| Expression | SymbAnaFis (Full) | Speedup |
| :--- | :---: | :---: |
| Bessel Wave | 134.56 µs | - |
| Damped Oscillator | 160.28 µs | - |
| Gaussian 2D | 136.55 µs | - |
| Lennard-Jones | 29.43 µs | - |
| Logistic Sigmoid | 120.29 µs | - |
| Lorentz Factor | 264.58 µs | - |
| Maxwell-Boltzmann | 335.11 µs | - |
| Normal PDF | 146.86 µs | - |
| Planck Blackbody | 340.64 µs | - |

---

## 4. Simplification Only

| Expression | SymbAnaFis | Speedup |
| :--- | :---: | :---: |
| Bessel Wave | 130.23 µs | - |
| Damped Oscillator | 157.60 µs | - |
| Gaussian 2D | 133.41 µs | - |
| Lennard-Jones | 26.54 µs | - |
| Logistic Sigmoid | 118.53 µs | - |
| Lorentz Factor | 259.39 µs | - |
| Maxwell-Boltzmann | 331.87 µs | - |
| Normal PDF | 143.64 µs | - |
| Planck Blackbody | 336.28 µs | - |

---

## 5. Compilation

| Expression | SA (Simplified) | Symbolica | SA (Raw) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | **1.63 µs** | - | 1.69 µs | - |
| Damped Oscillator | **1.35 µs** | 15.10 µs | 1.51 µs | **SA (Simplified)** (11.16x) |
| Gaussian 2D | **1.98 µs** | 34.67 µs | 2.17 µs | **SA (Simplified)** (17.52x) |
| Lennard-Jones | **1.22 µs** | 28.31 µs | 1.40 µs | **SA (Simplified)** (23.22x) |
| Logistic Sigmoid | **1.02 µs** | 9.90 µs | 1.11 µs | **SA (Simplified)** (9.72x) |
| Lorentz Factor | **1.02 µs** | 9.75 µs | 1.65 µs | **SA (Simplified)** (9.54x) |
| Maxwell-Boltzmann | **2.32 µs** | 18.90 µs | 2.53 µs | **SA (Simplified)** (8.15x) |
| Normal PDF | **1.64 µs** | 18.78 µs | 1.87 µs | **SA (Simplified)** (11.44x) |
| Planck Blackbody | **2.18 µs** | 11.22 µs | 2.29 µs | **SA (Simplified)** (5.15x) |

---

## 6. Evaluation (1000 points)

| Expression | SA (Simplified) | Symbolica | SA (Raw) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | **245.08 µs** | - | 305.97 µs | - |
| Damped Oscillator | 109.65 µs | **57.87 µs** | 130.48 µs | **Symbolica** (1.89x) |
| Gaussian 2D | 121.47 µs | **78.90 µs** | 134.68 µs | **Symbolica** (1.54x) |
| Lennard-Jones | 91.22 µs | **63.08 µs** | 120.20 µs | **Symbolica** (1.45x) |
| Logistic Sigmoid | 80.52 µs | **50.80 µs** | 83.26 µs | **Symbolica** (1.59x) |
| Lorentz Factor | 82.04 µs | **54.60 µs** | 104.45 µs | **Symbolica** (1.50x) |
| Maxwell-Boltzmann | 172.06 µs | **81.69 µs** | 179.82 µs | **Symbolica** (2.11x) |
| Normal PDF | 105.61 µs | **65.22 µs** | 124.55 µs | **Symbolica** (1.62x) |
| Planck Blackbody | 168.27 µs | **62.64 µs** | 162.08 µs | **Symbolica** (2.69x) |

---

## 7. Full Pipeline

| Expression | SymbAnaFis | SA (No Diff Simp) | Symbolica | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | 457.82 µs | **287.84 µs** | - | - |
| Damped Oscillator | 281.51 µs | **121.44 µs** | 165.75 µs | **Symbolica** (1.70x) |
| Gaussian 2D | 266.54 µs | **134.22 µs** | 149.39 µs | **Symbolica** (1.78x) |
| Lennard-Jones | 130.37 µs | **113.55 µs** | 119.62 µs | **Symbolica** (1.09x) |
| Logistic Sigmoid | 206.56 µs | **75.72 µs** | 84.25 µs | **Symbolica** (2.45x) |
| Lorentz Factor | 350.53 µs | **93.90 µs** | 107.76 µs | **Symbolica** (3.25x) |
| Maxwell-Boltzmann | 543.05 µs | **187.54 µs** | 238.37 µs | **Symbolica** (2.28x) |
| Normal PDF | 262.37 µs | 123.03 µs | **114.00 µs** | **Symbolica** (2.30x) |
| Planck Blackbody | 533.63 µs | **159.34 µs** | 204.37 µs | **Symbolica** (2.61x) |

---

## Parallel: Evaluation Methods (1k pts)

| Expression | Compiled Loop | Tree Walk | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **420.26 µs** | 1.24 ms | **Compiled Loop** (2.95x) |
| Damped Oscillator | **83.05 µs** | 1.02 ms | **Compiled Loop** (12.23x) |
| Gaussian 2D | **102.82 µs** | 1.04 ms | **Compiled Loop** (10.09x) |
| Lennard-Jones | **60.72 µs** | 668.84 µs | **Compiled Loop** (11.02x) |
| Logistic Sigmoid | **95.53 µs** | 1.09 ms | **Compiled Loop** (11.36x) |
| Lorentz Factor | **156.01 µs** | 829.94 µs | **Compiled Loop** (5.32x) |
| Maxwell-Boltzmann | **123.28 µs** | 1.30 ms | **Compiled Loop** (10.58x) |
| Normal PDF | **97.50 µs** | 1.04 ms | **Compiled Loop** (10.69x) |
| Planck Blackbody | **124.05 µs** | 1.99 ms | **Compiled Loop** (16.07x) |

---

## Parallel: Scaling (Points)

| Points | Eval Batch (SIMD) | Loop | Speedup |
| :--- | :---: | :---: | :---: |
| 100 | **2.31 µs** | 6.44 µs | **Eval Batch (SIMD)** (2.79x) |
| 1000 | **22.75 µs** | 64.36 µs | **Eval Batch (SIMD)** (2.83x) |
| 10000 | **226.88 µs** | 643.77 µs | **Eval Batch (SIMD)** (2.84x) |
| 100000 | **369.34 µs** | 6.46 ms | **Eval Batch (SIMD)** (17.49x) |

---

## Large Expressions (100 terms)

| Operation | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Parse | **147.40 µs** | 238.25 µs | **SA** (1.62x) |
| Diff (no simplify) | **85.09 µs** | 252.44 µs | **SA** (2.97x) |
| Diff+Simplify | 7.55 ms | — | — |
| Compile (simplified) | **38.56 µs** | 2.20 ms | **SA** (57.05x) |
| Eval 1000pts (simplified) | **3.77 ms** | 3.81 ms | **SA** (1.01x) |

---

## Large Expressions (300 terms)

| Operation | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Parse | **464.56 µs** | 730.00 µs | **SA** (1.57x) |
| Diff (no simplify) | **270.19 µs** | 798.22 µs | **SA** (2.95x) |
| Diff+Simplify | 21.95 ms | — | — |
| Compile (simplified) | **128.48 µs** | 15.09 ms | **SA** (117.42x) |
| Eval 1000pts (simplified) | 10.89 ms | **10.22 ms** | **SY** (1.07x) |

---

