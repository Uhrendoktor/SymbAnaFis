# Examples

This directory contains examples demonstrating SymbAnaFis capabilities.

## Quick Reference

| Example | Description | Run With |
|---------|-------------|----------|
| **quickstart** | Minimal 25-line demo | `cargo run --example quickstart` |
| **api_showcase** | Complete API tour (10 parts) | `cargo run --example api_showcase` |
| **dual_autodiff** | Automatic differentiation with dual numbers | `cargo run --example dual_autodiff` |
| **applications** | Physics & engineering | `cargo run --example applications` |
| **simplification_comparison** | Compare against Symbolica CAS | `cargo run --example simplification_comparison` |

## quickstart.rs - Get Started Fast

Core features in 25 lines: differentiation, simplification, Symbol Copy, LaTeX/Unicode output.

```bash
cargo run --example quickstart
```

## api_showcase.rs - Complete API Tour

Comprehensive demo of ALL features:

| Part | Feature |
|------|---------|
| 1 | String-based API (`diff`, `simplify`) |
| 2 | Type-safe API + Symbol Copy |
| 3 | Numerical evaluation |
| 4 | Multi-variable calculus (gradient, hessian, jacobian) |
| 5 | All 60+ supported functions |
| 6 | Custom derivatives & evaluation |
| 7 | Safety features (max depth, domain safety) |
| 8 | Expression output (LaTeX, Unicode) |
| 9 | Uncertainty propagation |
| 10 | Parallel evaluation (requires `parallel` feature) |

```bash
cargo run --example api_showcase
cargo run --example api_showcase --features parallel  # Include Part 10
```

## dual_autodiff.rs - Automatic Differentiation

Demonstrates SymbAnaFis's `Dual<T>` number implementation for exact derivative computation through algebraic manipulation.

**Features shown:**
- Basic dual number arithmetic
- Transcendental functions (sin, exp)
- Higher-order derivatives
- Chain rule application
- Comparison with symbolic differentiation

**Sample Output:**
```text
=== Quadratic Function: f(x) = x² + 3x + 1 ===
f(2) = 11
f'(2) = 7
Expected: f(2) = 11, f'(2) = 7

=== Transcendental Function: f(x) = sin(x) * exp(x) ===
f(1) = 2.287355
f'(1) = 5.435620
Expected f'(1) = 5.435620
Error: 0.00e0
```

```bash
cargo run --example dual_autodiff
```

## applications.rs - Real-World Physics

20 physics & engineering applications including:
- Kinematics, thermodynamics, quantum mechanics
- Fluid dynamics, optics, control systems
- Electromagnetism, relativity, astrophysics

```bash
cargo run --example applications
```

## simplification_comparison.rs - Benchmarking Accuracy

Compares SymbAnaFis differentiation quality against the commercial Symbolica engine across complex physics expressions (Maxwell-Boltzmann, Planck Blackbody, etc.).

**Sample Output:**
```text
--- Logistic Sigmoid (d/dx) ---
Input: 1/(1+exp(-k*(x-x0)))
SymbAnaFis (Simplified): exp(-k*(x - x0))*k/(1 + exp(-k*(x - x0)))^2
Symbolica:               k*(exp(-k*(x-x0))+1)^-2*exp(-k*(x-x0))
```