# CI Benchmark Results

**SymbAnaFis Version:** 0.7.0  
**Date:** Sat Jan 24 14:57:28 UTC 2026  
**Commit:** `5165dcc1fbe1`  
**Rust:** 1.93.0  

> Auto-generated from Criterion benchmark output

## 1. Parsing (String → AST)

| Expression | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **3.20 µs** | 4.84 µs | **SymbAnaFis** (1.51x) |
| Damped Oscillator | **3.54 µs** | 5.37 µs | **SymbAnaFis** (1.52x) |
| Gaussian 2D | **6.43 µs** | 13.82 µs | **SymbAnaFis** (2.15x) |
| Lennard-Jones | **3.88 µs** | 7.80 µs | **SymbAnaFis** (2.01x) |
| Logistic Sigmoid | **2.91 µs** | 4.64 µs | **SymbAnaFis** (1.60x) |
| Lorentz Factor | **2.56 µs** | 4.83 µs | **SymbAnaFis** (1.89x) |
| Maxwell-Boltzmann | **8.12 µs** | 12.86 µs | **SymbAnaFis** (1.58x) |
| Normal PDF | **5.05 µs** | 10.06 µs | **SymbAnaFis** (1.99x) |
| Planck Blackbody | **4.86 µs** | 8.81 µs | **SymbAnaFis** (1.81x) |

---

## 2. Differentiation

| Expression | SymbAnaFis (Light) | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **2.14 µs** | 3.51 µs | **SymbAnaFis (Light)** (1.64x) |
| Damped Oscillator | **1.50 µs** | 3.44 µs | **SymbAnaFis (Light)** (2.29x) |
| Gaussian 2D | **1.32 µs** | 4.50 µs | **SymbAnaFis (Light)** (3.41x) |
| Lennard-Jones | **1.74 µs** | 3.86 µs | **SymbAnaFis (Light)** (2.21x) |
| Logistic Sigmoid | **840.36 ns** | 2.34 µs | **SymbAnaFis (Light)** (2.78x) |
| Lorentz Factor | **1.56 µs** | 3.73 µs | **SymbAnaFis (Light)** (2.40x) |
| Maxwell-Boltzmann | **2.32 µs** | 6.80 µs | **SymbAnaFis (Light)** (2.93x) |
| Normal PDF | **1.56 µs** | 3.40 µs | **SymbAnaFis (Light)** (2.18x) |
| Planck Blackbody | **1.98 µs** | 6.49 µs | **SymbAnaFis (Light)** (3.28x) |

---

## 3. Differentiation + Simplification

| Expression | SymbAnaFis (Full) | Speedup |
| :--- | :---: | :---: |
| Bessel Wave | 131.11 µs | - |
| Damped Oscillator | 157.11 µs | - |
| Gaussian 2D | 134.15 µs | - |
| Lennard-Jones | 28.90 µs | - |
| Logistic Sigmoid | 118.42 µs | - |
| Lorentz Factor | 260.41 µs | - |
| Maxwell-Boltzmann | 329.65 µs | - |
| Normal PDF | 145.45 µs | - |
| Planck Blackbody | 333.69 µs | - |

---

## 4. Simplification Only

| Expression | SymbAnaFis | Speedup |
| :--- | :---: | :---: |
| Bessel Wave | 127.52 µs | - |
| Damped Oscillator | 153.12 µs | - |
| Gaussian 2D | 132.00 µs | - |
| Lennard-Jones | 26.63 µs | - |
| Logistic Sigmoid | 116.41 µs | - |
| Lorentz Factor | 255.07 µs | - |
| Maxwell-Boltzmann | 327.65 µs | - |
| Normal PDF | 142.68 µs | - |
| Planck Blackbody | 327.37 µs | - |

---

## 5. Compilation

| Expression | SA (Simplified) | Symbolica | SA (Raw) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | **1.56 µs** | - | 1.63 µs | - |
| Damped Oscillator | **1.14 µs** | 14.87 µs | 1.38 µs | **SA (Simplified)** (12.98x) |
| Gaussian 2D | **1.44 µs** | 34.77 µs | 1.87 µs | **SA (Simplified)** (24.14x) |
| Lennard-Jones | **1.05 µs** | 27.67 µs | 1.09 µs | **SA (Simplified)** (26.26x) |
| Logistic Sigmoid | **948.88 ns** | 9.69 µs | 955.41 ns | **SA (Simplified)** (10.21x) |
| Lorentz Factor | **986.30 ns** | 9.53 µs | 1.55 µs | **SA (Simplified)** (9.66x) |
| Maxwell-Boltzmann | **2.10 µs** | 18.07 µs | 2.65 µs | **SA (Simplified)** (8.61x) |
| Normal PDF | **1.35 µs** | 18.39 µs | 1.68 µs | **SA (Simplified)** (13.61x) |
| Planck Blackbody | **1.94 µs** | 10.78 µs | 2.02 µs | **SA (Simplified)** (5.54x) |

---

## 6. Evaluation (1000 points)

| Expression | SA (Simplified) | Symbolica | SA (Raw) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | **258.17 µs** | - | 313.66 µs | - |
| Damped Oscillator | 95.35 µs | **57.94 µs** | 110.86 µs | **Symbolica** (1.65x) |
| Gaussian 2D | 99.07 µs | **64.52 µs** | 119.60 µs | **Symbolica** (1.54x) |
| Lennard-Jones | 68.82 µs | **63.33 µs** | 105.11 µs | **Symbolica** (1.09x) |
| Logistic Sigmoid | 69.64 µs | **64.87 µs** | 68.96 µs | **Symbolica** (1.07x) |
| Lorentz Factor | 76.20 µs | **54.49 µs** | 87.37 µs | **Symbolica** (1.40x) |
| Maxwell-Boltzmann | 160.60 µs | **81.52 µs** | 161.99 µs | **Symbolica** (1.97x) |
| Normal PDF | 90.50 µs | **63.59 µs** | 108.19 µs | **Symbolica** (1.42x) |
| Planck Blackbody | 135.32 µs | **62.73 µs** | 134.38 µs | **Symbolica** (2.16x) |

---

## 7. Full Pipeline

| Expression | SymbAnaFis | SA (No Diff Simp) | Symbolica | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Bessel Wave | 399.16 µs | **309.90 µs** | - | - |
| Damped Oscillator | 259.29 µs | **102.96 µs** | 165.72 µs | **Symbolica** (1.56x) |
| Gaussian 2D | 250.06 µs | **120.37 µs** | 148.58 µs | **Symbolica** (1.68x) |
| Lennard-Jones | 117.08 µs | **93.88 µs** | 119.22 µs | **SymbAnaFis** (1.02x) |
| Logistic Sigmoid | 193.93 µs | **61.27 µs** | 83.87 µs | **Symbolica** (2.31x) |
| Lorentz Factor | 331.53 µs | **79.13 µs** | 106.69 µs | **Symbolica** (3.11x) |
| Maxwell-Boltzmann | 496.07 µs | **172.24 µs** | 234.96 µs | **Symbolica** (2.11x) |
| Normal PDF | 247.35 µs | **103.41 µs** | 113.49 µs | **Symbolica** (2.18x) |
| Planck Blackbody | 491.87 µs | **128.37 µs** | 201.67 µs | **Symbolica** (2.44x) |

---

## Parallel: Evaluation Methods (1k pts)

| Expression | Compiled Loop | Tree Walk | Speedup |
| :--- | :---: | :---: | :---: |
| Bessel Wave | **224.85 µs** | 1.25 ms | **Compiled Loop** (5.57x) |
| Damped Oscillator | **88.78 µs** | 1.03 ms | **Compiled Loop** (11.62x) |
| Gaussian 2D | **92.12 µs** | 1.05 ms | **Compiled Loop** (11.36x) |
| Lennard-Jones | **64.12 µs** | 677.99 µs | **Compiled Loop** (10.57x) |
| Logistic Sigmoid | **53.55 µs** | 1.09 ms | **Compiled Loop** (20.37x) |
| Lorentz Factor | **65.94 µs** | 825.86 µs | **Compiled Loop** (12.52x) |
| Maxwell-Boltzmann | **143.89 µs** | 1.31 ms | **Compiled Loop** (9.14x) |
| Normal PDF | **82.19 µs** | 1.06 ms | **Compiled Loop** (12.86x) |
| Planck Blackbody | **125.12 µs** | 2.01 ms | **Compiled Loop** (16.09x) |

---

## Parallel: Scaling (Points)

| Points | Eval Batch (SIMD) | Loop | Speedup |
| :--- | :---: | :---: | :---: |
| 100 | **2.26 µs** | 5.83 µs | **Eval Batch (SIMD)** (2.58x) |
| 1000 | **22.65 µs** | 56.33 µs | **Eval Batch (SIMD)** (2.49x) |
| 10000 | **226.43 µs** | 581.85 µs | **Eval Batch (SIMD)** (2.57x) |
| 100000 | **376.49 µs** | 5.83 ms | **Eval Batch (SIMD)** (15.49x) |

---

## Large Expressions (100 terms)

| Operation | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Parse | **140.22 µs** | 236.06 µs | **SA** (1.68x) |
| Diff (no simplify) | **80.17 µs** | 248.04 µs | **SA** (3.09x) |
| Diff+Simplify | 7.43 ms | — | — |
| Compile (simplified) | **47.48 µs** | 2.21 ms | **SA** (46.51x) |
| Eval 1000pts (simplified) | 3.99 ms | **3.71 ms** | **SY** (1.07x) |

---

## Large Expressions (300 terms)

| Operation | SymbAnaFis | Symbolica | Speedup |
| :--- | :---: | :---: | :---: |
| Parse | **446.67 µs** | 728.69 µs | **SA** (1.63x) |
| Diff (no simplify) | **252.18 µs** | 779.13 µs | **SA** (3.09x) |
| Diff+Simplify | 21.56 ms | — | — |
| Compile (simplified) | **200.98 µs** | 15.30 ms | **SA** (76.11x) |
| Eval 1000pts (simplified) | 11.88 ms | **10.43 ms** | **SY** (1.14x) |

---

