# SymbAnaFis Roadmap

## v0.7.0 - The "Symbolic Solver" Update

**Focus**: Equation solving and domain analysis for complete singularity handling

### Symbolic Solver (Core)
- [ ] **Linear Equation Solver**: Gaussian elimination on symbolic matrix.
- [ ] **Polynomial Root Finding**: Analytical solutions for degrees ≤4.
- [ ] **Variable Isolation**: Basic `solve(y = f(x), x)` functionality.

### Domain Analysis & Singularities
- [ ] **Full Domain Analysis**: Detect where expressions are undefined at compile time.
- [ ] **Conditional Bytecode**: Generate branches for singularity handling (L'Hôpital fallbacks).
- [ ] **Series Expansion**: Taylor/Laurent series for limit computation (`series(sin(x)/x, x, 0)`).

### JIT Compilation (Optional Feature)
- [ ] **Cranelift Backend**: `features = ["jit"]` - translate `Expr` to native machine code (x86/ARM).
    - *Goal*: Surpass Stack VM for expressions evaluated >1M times.

---

## v0.8.0 - The "Extended Capabilities" Update

**Focus**: Rounding out the symbolic manipulation toolkit

### Extended Bytecode
- [ ] **Special Functions in VM**: Native OpCodes for:
    - [ ] Factorial, DoubleFactorial
    - [ ] Exponential integrals (Ei, Li)
    - [ ] Trigonometric integrals (Si, Ci)

### Input/Output
- [ ] **LaTeX Parsing**: Parse LaTeX strings into expressions (`parse(r"\frac{1}{2}x^2")`).
- [ ] **Pretty Printing**: Improved display formatting options.

### Advanced (Stretch Goals)
- [ ] **Indefinite Integration**: Heuristic approach for common patterns.
- [ ] **Tensor/Matrix Expressions**: First-class Matrix * Vector symbolic operations.

---

## Documentation & Ecosystem (Ongoing)

**Focus**: Fixing the "Palace Entry" problem.

- [ ] **Cookbook / Examples**:
    - [ ] "Discovering Physical Laws from Data" (Symbolic Regression demo).
    - [ ] "Neural ODE Training with SymbAnaFis".
    - [ ] "Solving Heat Equation via JIT Compilation".
- [ ] **Interactive Web Demo**: WASM compilation for "Try it now" page.

---

## Ideas / Backlog (Long Term)

- [ ] **GPU Acceleration**: OpenCL/CUDA backends for `eval_batch` on massive datasets (>100M points).
- [ ] **Complex Number Support**: First-class complex arithmetic.
- [ ] **Interval Arithmetic**: Rigorous bounds computation.

---

## AnaFis Ecosystem - Companion Crates

**Focus**: Domain-specific libraries built on `symb_anafis` core.

### `opt-anafis` - Optimization
- [ ] Gradient descent (SGD, momentum, Adam)
- [ ] Newton's method (using symbolic Hessian)
- [ ] Line search utilities
- [ ] L-BFGS for large-scale problems

### `fit-anafis` - Fitting & Regression
- [ ] Nonlinear Least Squares (Levenberg-Marquardt)
- [ ] Orthogonal Distance Regression (ODR)
- [ ] Weighted Least Squares
- [ ] Model builders and residual generators

### `ml-anafis` - Machine Learning
- [ ] Symbolic KAN (Kolmogorov-Arnold Networks)
- [ ] Symbolic Regression (genetic programming operators)
- [ ] PINN templates (Physics-Informed Neural Networks)
- [ ] Loss function builders

### `phys-anafis` - Physics & Scientific
- [ ] ODE integrators (RK4, adaptive step)
- [ ] Lagrangian/Hamiltonian mechanics helpers
- [ ] Equation of motion derivation
- [ ] Sensitivity analysis utilities

### `geo-anafis` - Geometry & Graphics
- [ ] Implicit surface utilities (normals, ray-marching)
- [ ] Parametric curve/surface tools
- [ ] Curve fitting with symbolic gradients

---

## Contributing

Contributions welcome! Priority areas:
1.  **Beta Testers**: Users applying the library to ML/Physics problems to report edge cases.
2.  **Special Functions**: Implementation of numeric traits for obscure physics functions.
3.  **Docs**: Writing "How-to" guides for beginners.
