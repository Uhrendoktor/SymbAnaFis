# CI Benchmark Results

**SymbAnaFis Version:** 0.5.1  
**Date:** Thu Jan  8 03:09:03 UTC 2026  
**Commit:** `48402221cb1e`  
**Rust:** 1.92.0  

> Auto-generated from Criterion benchmark output

## 1. Parsing (String → AST)

| Expression        | SymbAnaFis  | Symbolica |        Speedup         |
| :---------------- | :---------: | :-------: | :--------------------: |
| Bessel Wave       | **3.43 µs** |  4.82 µs  | **SymbAnaFis** (1.41x) |
| Damped Oscillator | **3.60 µs** |  5.34 µs  | **SymbAnaFis** (1.48x) |
| Gaussian 2D       | **6.55 µs** | 13.56 µs  | **SymbAnaFis** (2.07x) |
| Lennard-Jones     | **3.95 µs** |  7.54 µs  | **SymbAnaFis** (1.91x) |
| Logistic Sigmoid  | **3.00 µs** |  4.64 µs  | **SymbAnaFis** (1.55x) |
| Lorentz Factor    | **2.51 µs** |  4.82 µs  | **SymbAnaFis** (1.92x) |
| Maxwell-Boltzmann | **7.88 µs** | 12.93 µs  | **SymbAnaFis** (1.64x) |
| Normal PDF        | **5.08 µs** |  9.79 µs  | **SymbAnaFis** (1.93x) |
| Planck Blackbody  | **4.92 µs** |  8.72 µs  | **SymbAnaFis** (1.77x) |

---

## 2. Differentiation

| Expression        | SymbAnaFis (Light) | Symbolica |            Speedup             |
| :---------------- | :----------------: | :-------: | :----------------------------: |
| Bessel Wave       |    **2.19 µs**     |  3.57 µs  | **SymbAnaFis (Light)** (1.63x) |
| Damped Oscillator |    **1.52 µs**     |  3.49 µs  | **SymbAnaFis (Light)** (2.30x) |
| Gaussian 2D       |    **1.48 µs**     |  4.61 µs  | **SymbAnaFis (Light)** (3.12x) |
| Lennard-Jones     |    **1.77 µs**     |  3.97 µs  | **SymbAnaFis (Light)** (2.25x) |
| Logistic Sigmoid  |   **855.73 ns**    |  2.42 µs  | **SymbAnaFis (Light)** (2.83x) |
| Lorentz Factor    |    **1.59 µs**     |  3.80 µs  | **SymbAnaFis (Light)** (2.39x) |
| Maxwell-Boltzmann |    **2.43 µs**     |  6.93 µs  | **SymbAnaFis (Light)** (2.86x) |
| Normal PDF        |    **1.66 µs**     |  3.51 µs  | **SymbAnaFis (Light)** (2.11x) |
| Planck Blackbody  |    **2.05 µs**     |  6.61 µs  | **SymbAnaFis (Light)** (3.22x) |

---

## 3. Differentiation + Simplification

| Expression        | SymbAnaFis (Full) | Speedup |
| :---------------- | :---------------: | :-----: |
| Bessel Wave       |     132.84 µs     |    -    |
| Damped Oscillator |     154.08 µs     |    -    |
| Gaussian 2D       |     136.40 µs     |    -    |
| Lennard-Jones     |     29.19 µs      |    -    |
| Logistic Sigmoid  |     119.89 µs     |    -    |
| Lorentz Factor    |     265.54 µs     |    -    |
| Maxwell-Boltzmann |     335.25 µs     |    -    |
| Normal PDF        |     147.57 µs     |    -    |
| Planck Blackbody  |     336.58 µs     |    -    |

---

## 4. Simplification Only

| Expression        | SymbAnaFis | Speedup |
| :---------------- | :--------: | :-----: |
| Bessel Wave       | 129.12 µs  |    -    |
| Damped Oscillator | 150.72 µs  |    -    |
| Gaussian 2D       | 133.79 µs  |    -    |
| Lennard-Jones     |  26.94 µs  |    -    |
| Logistic Sigmoid  | 117.00 µs  |    -    |
| Lorentz Factor    | 262.15 µs  |    -    |
| Maxwell-Boltzmann | 329.76 µs  |    -    |
| Normal PDF        | 144.26 µs  |    -    |
| Planck Blackbody  | 332.42 µs  |    -    |

---

## 5. Compilation

| Expression        | SA (Simplified) | Symbolica |  SA (Raw)   |           Speedup            |
| :---------------- | :-------------: | :-------: | :---------: | :--------------------------: |
| Bessel Wave       |   **1.85 µs**   |     -     |   2.11 µs   |              -               |
| Damped Oscillator |   **1.87 µs**   | 15.11 µs  |   2.12 µs   | **SA (Simplified)** (8.08x)  |
| Gaussian 2D       |   **1.90 µs**   | 34.23 µs  |   2.07 µs   | **SA (Simplified)** (18.03x) |
| Lennard-Jones     |     1.23 µs     | 27.36 µs  | **1.10 µs** | **SA (Simplified)** (22.30x) |
| Logistic Sigmoid  |     1.54 µs     |  9.80 µs  | **1.30 µs** | **SA (Simplified)** (6.36x)  |
| Lorentz Factor    |   **1.31 µs**   |  9.79 µs  |   1.45 µs   | **SA (Simplified)** (7.46x)  |
| Maxwell-Boltzmann |   **2.07 µs**   | 17.96 µs  |   2.99 µs   | **SA (Simplified)** (8.69x)  |
| Normal PDF        |   **1.53 µs**   | 18.51 µs  |   1.66 µs   | **SA (Simplified)** (12.09x) |
| Planck Blackbody  |     3.09 µs     | 10.96 µs  | **2.84 µs** | **SA (Simplified)** (3.55x)  |

---

## 6. Evaluation (1000 points)

| Expression        | SA (Simplified) |  Symbolica   | SA (Raw)  |        Speedup        |
| :---------------- | :-------------: | :----------: | :-------: | :-------------------: |
| Bessel Wave       |  **198.67 µs**  |      -       | 258.42 µs |           -           |
| Damped Oscillator |    89.28 µs     | **57.82 µs** | 122.74 µs | **Symbolica** (1.54x) |
| Gaussian 2D       |    110.39 µs    | **64.46 µs** | 146.88 µs | **Symbolica** (1.71x) |
| Lennard-Jones     |    94.61 µs     | **63.14 µs** | 100.73 µs | **Symbolica** (1.50x) |
| Logistic Sigmoid  |    132.25 µs    | **51.13 µs** | 105.93 µs | **Symbolica** (2.59x) |
| Lorentz Factor    |    73.02 µs     | **54.55 µs** | 125.69 µs | **Symbolica** (1.34x) |
| Maxwell-Boltzmann |    131.89 µs    | **81.11 µs** | 289.46 µs | **Symbolica** (1.63x) |
| Normal PDF        |    99.06 µs     | **63.56 µs** | 128.73 µs | **Symbolica** (1.56x) |
| Planck Blackbody  |    166.18 µs    | **63.68 µs** | 152.32 µs | **Symbolica** (2.61x) |

---

## 7. Full Pipeline

| Expression        |  SymbAnaFis   |   Symbolica   |        Speedup         |
| :---------------- | :-----------: | :-----------: | :--------------------: |
| Bessel Wave       |   344.74 µs   |       -       |           -            |
| Damped Oscillator |   265.80 µs   | **181.85 µs** | **Symbolica** (1.46x)  |
| Gaussian 2D       |   286.09 µs   | **178.35 µs** | **Symbolica** (1.60x)  |
| Lennard-Jones     | **127.27 µs** |   136.39 µs   | **SymbAnaFis** (1.07x) |
| Logistic Sigmoid  |   250.15 µs   | **95.72 µs**  | **Symbolica** (2.61x)  |
| Lorentz Factor    |   366.39 µs   | **118.56 µs** | **Symbolica** (3.09x)  |
| Maxwell-Boltzmann |   521.82 µs   | **251.03 µs** | **Symbolica** (2.08x)  |
| Normal PDF        |   281.64 µs   | **130.00 µs** | **Symbolica** (2.17x)  |
| Planck Blackbody  |   564.74 µs   | **230.52 µs** | **Symbolica** (2.45x)  |

---

## Parallel: Evaluation Methods (1k pts)

| Expression        | Compiled Loop | Tree Walk |          Speedup           |
| :---------------- | :-----------: | :-------: | :------------------------: |
| Bessel Wave       | **185.97 µs** |  1.28 ms  | **Compiled Loop** (6.87x)  |
| Damped Oscillator | **85.61 µs**  |  1.05 ms  | **Compiled Loop** (12.22x) |
| Gaussian 2D       | **106.51 µs** |  1.83 ms  | **Compiled Loop** (17.21x) |
| Lennard-Jones     | **90.27 µs**  | 668.14 µs | **Compiled Loop** (7.40x)  |
| Logistic Sigmoid  | **126.33 µs** |  1.09 ms  | **Compiled Loop** (8.61x)  |
| Lorentz Factor    | **65.81 µs**  | 823.17 µs | **Compiled Loop** (12.51x) |
| Maxwell-Boltzmann | **132.89 µs** |  1.30 ms  | **Compiled Loop** (9.77x)  |
| Normal PDF        | **94.62 µs**  |  1.05 ms  | **Compiled Loop** (11.08x) |
| Planck Blackbody  | **189.89 µs** |  2.02 ms  | **Compiled Loop** (10.61x) |

---

## Parallel: Scaling (Points)

| Points | Eval Batch (SIMD) |   Loop    |            Speedup             |
| :----- | :---------------: | :-------: | :----------------------------: |
| 100    |    **2.20 µs**    |  6.34 µs  | **Eval Batch (SIMD)** (2.89x)  |
| 1000   |   **21.92 µs**    | 63.35 µs  | **Eval Batch (SIMD)** (2.89x)  |
| 10000  |   **219.82 µs**   | 631.80 µs | **Eval Batch (SIMD)** (2.87x)  |
| 100000 |   **369.99 µs**   |  6.32 ms  | **Eval Batch (SIMD)** (17.09x) |

---

## Large Expressions (100 terms)

| Operation                 |  SymbAnaFis   |  Symbolica  |     Speedup     |
| :------------------------ | :-----------: | :---------: | :-------------: |
| Parse                     | **147.82 µs** |  234.62 µs  | **SA** (1.59x)  |
| Diff (no simplify)        | **84.73 µs**  |  247.97 µs  | **SA** (2.93x)  |
| Diff+Simplify             |    7.51 ms    |      —      |        —        |
| Compile (simplified)      | **36.81 µs**  |   2.12 ms   | **SA** (57.54x) |
| Eval 1000pts (simplified) |    4.76 ms    | **3.65 ms** | **SY** (1.30x)  |

---

## Large Expressions (300 terms)

| Operation                 |  SymbAnaFis   |  Symbolica   |     Speedup      |
| :------------------------ | :-----------: | :----------: | :--------------: |
| Parse                     | **468.10 µs** |  724.02 µs   |  **SA** (1.55x)  |
| Diff (no simplify)        | **269.33 µs** |  792.04 µs   |  **SA** (2.94x)  |
| Diff+Simplify             |   22.29 ms    |      —       |        —         |
| Compile (simplified)      | **122.44 µs** |   14.79 ms   | **SA** (120.76x) |
| Eval 1000pts (simplified) |   14.29 ms    | **10.62 ms** |  **SY** (1.34x)  |

---

