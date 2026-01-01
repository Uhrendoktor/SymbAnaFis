# SymbAnaFis Roadmap

## v0.6.0 - The "Stability & Architecture" Update

**Focus**: Solidifying the core architecture, handling breaking changes early

### ⚠ Breaking Changes
- [x] **Zero-Copy NumPy Integration**: Update `eval_f64` and `CompiledEvaluator` to return `numpy.ndarray` instead of `list` (Python API).
- [ ] **Result Type Standardization**: Ensure consistent error handling across all new FFI boundaries.

### Core Architecture
- [ ] **Expression Interning (Hash Consing)**: Implement global caching using Weak references to deduplicate common sub-expressions (e.g., `sin(x+y)` is stored once in memory).
    - *Benefit*: O(1) equality checks and massive RAM savings.
- [ ] **Removable Singularity Handling**: Implement compile-time detection for 0/0 cases (like `sinc(x)` or `sin(x)/x`).
    - *Implementation*: Auto-generate conditional bytecode branches using Taylor Series expansion for small arguments (x→0).

### Equation Solving & Algebra
- [ ] Linear equation solver (Gaussian elimination on symbolic matrix).
- [ ] Polynomial root finding (for degrees ≤4 analytically).
- [ ] Basic Substitution and Variable Isolation (`solve(y = x + a, x)`).

---

## v0.7.0 - The "Speed Demon" Update (JIT & HPC)

**Focus**: Bridging the gap between "Fast Scripting" and "Native Performance" for long-running simulations.

### JIT Compilation (Optional Feature)
- [ ] **Cranelift Backend**: Implement an optional JIT compiler that translates `Expr` directly to machine code (x86/ARM).
    - *Goal*: Surpass the Stack VM performance for static expressions evaluated >1M times.

### Extended Bytecode Support
- [ ] **Special Functions in VM**: Add native OpCodes for:
    - [ ] Factorial, DoubleFactorial
    - [ ] Exponential integrals (Ei, Li)
    - [ ] Trigonometric integrals (Si, Ci)

---

## v0.8.0 - The "Symbolic AI" Update

**Focus**: Providing the tooling required for Physics-Informed Machine Learning and Symbolic Regression.

### Symbolic Regression Helpers
- [ ] **Genetic Programming Utilities**:
    - [ ] Mutation operators (e.g., `random_subtree_change`).
    - [ ] Crossover operators (e.g., `swap_subtrees`).
- [ ] **Loss Function Generators**: Auto-generate Mean Squared Error (MSE) expressions between symbolic formulas and data points.

### Neural ODE Support
- [ ] **Standalone Differentiable Physics**: Implement native backward-pass gradients to support Neural ODE training without external frameworks (like PyTorch/JAX).

---

## Documentation & Ecosystem (Ongoing)

**Focus**: Fixing the "Palace Entry" problem.

- [ ] **Cookbook / Examples**:
    - [ ] "Discovering Physical Laws from Data" (Symbolic Regression demo).
    - [ ] "Neural ODE Training with SymbAnaFis".
    - [ ] "Solving Heat Equation via JIT Compilation".
- [ ] **Automated Benchmarks**: Add a CI step that generates the comparison graphs (vs Symbolica/SymPy) automatically.
- [ ] **Interactive Web Demo**: Compile the library to WASM and create a simple "Try it now" page.

---

## Ideas / Backlog (Long Term)

- [ ] **Tensor/Matrix Expressions**: First-class support for Matrix * Vector symbolic operations (Symbolic Linear Algebra).
- [ ] **Series Expansion**: Full Taylor/Laurent series generation (`series(sin(x), x, 0, 5)`).
- [ ] **GPU Acceleration**: OpenCL/CUDA backends for `eval_batch` on massive datasets (>100M points).
- [ ] **Integration**: Risch algorithm (or heuristic approach) for indefinite integration.
- [ ] **LaTeX Parsing**: Ability to parse LaTeX strings into expressions (e.g., `parse(r"\frac{1}{2}x^2")`).

---

## Contributing

Contributions welcome! Priority areas:
1.  **Beta Testers**: Users applying the library to ML/Physics problems to report edge cases.
2.  **Special Functions**: Implementation of numeric traits for obscure physics functions.
3.  **Docs**: Writing "How-to" guides for beginners.
