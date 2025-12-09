//! Python bindings for symb_anafis using PyO3

use crate::{Expr as RustExpr, builder, sym};
use pyo3::prelude::*;
use std::collections::HashSet;

/// Wrapper for Rust Expr to expose to Python
#[pyclass(unsendable, name = "Expr")]
#[derive(Clone)]
struct PyExpr(RustExpr);

#[pymethods]
impl PyExpr {
    /// Create a symbolic expression from a string
    #[new]
    fn new(name: &str) -> Self {
        PyExpr(sym(name).into())
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Expr({})", self.0)
    }

    // Arithmetic operators
    fn __add__(&self, other: &PyExpr) -> PyExpr {
        PyExpr(self.0.clone() + other.0.clone())
    }

    fn __sub__(&self, other: &PyExpr) -> PyExpr {
        PyExpr(self.0.clone() - other.0.clone())
    }

    fn __mul__(&self, other: &PyExpr) -> PyExpr {
        PyExpr(self.0.clone() * other.0.clone())
    }

    fn __truediv__(&self, other: &PyExpr) -> PyExpr {
        PyExpr(self.0.clone() / other.0.clone())
    }

    fn __pow__(&self, other: &PyExpr, _modulo: Option<Py<PyAny>>) -> PyExpr {
        PyExpr(self.0.clone().pow_of(other.0.clone()))
    }

    // Functions
    fn sin(&self) -> PyExpr {
        PyExpr(self.0.clone().sin())
    }
    fn cos(&self) -> PyExpr {
        PyExpr(self.0.clone().cos())
    }
    fn tan(&self) -> PyExpr {
        PyExpr(self.0.clone().tan())
    }
    fn cot(&self) -> PyExpr {
        PyExpr(self.0.clone().cot())
    }
    fn sec(&self) -> PyExpr {
        PyExpr(self.0.clone().sec())
    }
    fn csc(&self) -> PyExpr {
        PyExpr(self.0.clone().csc())
    }

    fn asin(&self) -> PyExpr {
        PyExpr(self.0.clone().asin())
    }
    fn acos(&self) -> PyExpr {
        PyExpr(self.0.clone().acos())
    }
    fn atan(&self) -> PyExpr {
        PyExpr(self.0.clone().atan())
    }
    fn acot(&self) -> PyExpr {
        PyExpr(self.0.clone().acot())
    }
    fn asec(&self) -> PyExpr {
        PyExpr(self.0.clone().asec())
    }
    fn acsc(&self) -> PyExpr {
        PyExpr(self.0.clone().acsc())
    }

    fn sinh(&self) -> PyExpr {
        PyExpr(self.0.clone().sinh())
    }
    fn cosh(&self) -> PyExpr {
        PyExpr(self.0.clone().cosh())
    }
    fn tanh(&self) -> PyExpr {
        PyExpr(self.0.clone().tanh())
    }
    fn coth(&self) -> PyExpr {
        PyExpr(self.0.clone().coth())
    }
    fn sech(&self) -> PyExpr {
        PyExpr(self.0.clone().sech())
    }
    fn csch(&self) -> PyExpr {
        PyExpr(self.0.clone().csch())
    }

    fn asinh(&self) -> PyExpr {
        PyExpr(self.0.clone().asinh())
    }
    fn acosh(&self) -> PyExpr {
        PyExpr(self.0.clone().acosh())
    }
    fn atanh(&self) -> PyExpr {
        PyExpr(self.0.clone().atanh())
    }
    fn acoth(&self) -> PyExpr {
        PyExpr(self.0.clone().acoth())
    }
    fn asech(&self) -> PyExpr {
        PyExpr(self.0.clone().asech())
    }
    fn acsch(&self) -> PyExpr {
        PyExpr(self.0.clone().acsch())
    }

    fn exp(&self) -> PyExpr {
        PyExpr(self.0.clone().exp())
    }
    fn ln(&self) -> PyExpr {
        PyExpr(self.0.clone().ln())
    }
    fn log(&self) -> PyExpr {
        PyExpr(self.0.clone().log())
    }
    fn log10(&self) -> PyExpr {
        PyExpr(self.0.clone().log10())
    }
    fn log2(&self) -> PyExpr {
        PyExpr(self.0.clone().log2())
    }

    fn sqrt(&self) -> PyExpr {
        PyExpr(self.0.clone().sqrt())
    }
    fn cbrt(&self) -> PyExpr {
        PyExpr(self.0.clone().cbrt())
    }

    fn abs(&self) -> PyExpr {
        PyExpr(self.0.clone().abs())
    }
    fn sign(&self) -> PyExpr {
        PyExpr(self.0.clone().sign())
    }
    fn sinc(&self) -> PyExpr {
        PyExpr(self.0.clone().sinc())
    }
    fn erf(&self) -> PyExpr {
        PyExpr(self.0.clone().erf())
    }
    fn erfc(&self) -> PyExpr {
        PyExpr(self.0.clone().erfc())
    }
    fn gamma(&self) -> PyExpr {
        PyExpr(self.0.clone().gamma())
    }
    fn digamma(&self) -> PyExpr {
        PyExpr(self.0.clone().digamma())
    }
    fn trigamma(&self) -> PyExpr {
        PyExpr(self.0.clone().trigamma())
    }
    fn polygamma(&self, n: f64) -> PyExpr {
        PyExpr(self.0.clone().polygamma(n))
    }
    fn beta(&self, other: &PyExpr) -> PyExpr {
        PyExpr(self.0.clone().beta(other.0.clone()))
    }
    fn zeta(&self) -> PyExpr {
        PyExpr(self.0.clone().zeta())
    }
    fn lambertw(&self) -> PyExpr {
        PyExpr(self.0.clone().lambertw())
    }
    fn besselj(&self, n: f64) -> PyExpr {
        PyExpr(self.0.clone().besselj(n))
    }
    fn bessely(&self, n: f64) -> PyExpr {
        PyExpr(self.0.clone().bessely(n))
    }
    fn besseli(&self, n: f64) -> PyExpr {
        PyExpr(self.0.clone().besseli(n))
    }
    fn besselk(&self, n: f64) -> PyExpr {
        PyExpr(self.0.clone().besselk(n))
    }

    fn pow(&self, exp: f64) -> PyExpr {
        PyExpr(self.0.clone().pow_of(exp))
    }
}

/// Builder for differentiation operations
#[pyclass(name = "Diff")]
struct PyDiff {
    inner: builder::Diff,
}

#[pymethods]
impl PyDiff {
    #[new]
    fn new() -> Self {
        PyDiff {
            inner: builder::Diff::new(),
        }
    }

    fn domain_safe(mut self_: PyRefMut<'_, Self>, safe: bool) -> PyRefMut<'_, Self> {
        self_.inner = self_.inner.clone().domain_safe(safe);
        self_
    }

    fn fixed_var(mut self_: PyRefMut<'_, Self>, var: String) -> PyRefMut<'_, Self> {
        let sym = crate::Symbol::new(var);
        self_.inner = self_.inner.clone().fixed_var(&sym);
        self_
    }

    fn max_depth(mut self_: PyRefMut<'_, Self>, depth: usize) -> PyRefMut<'_, Self> {
        self_.inner = self_.inner.clone().max_depth(depth);
        self_
    }

    fn max_nodes(mut self_: PyRefMut<'_, Self>, nodes: usize) -> PyRefMut<'_, Self> {
        self_.inner = self_.inner.clone().max_nodes(nodes);
        self_
    }

    fn custom_derivative(
        mut self_: PyRefMut<'_, Self>,
        name: String,
        callback: Py<PyAny>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        // We need to wrap the Python callback into a Rust CustomDerivativeFn
        // This closure needs to be Send + Sync, so we wrap PyObject appropriately
        // Actually, Python objects are not Send, so we need to be careful.
        // But for single-threaded usage (GIL), we can use Python::with_gil inside the closure?
        // No, `dyn Fn` must be Send+Sync to be stored in `Diff` (because Diff is used in potential threads?).
        // `Diff` implementation uses `Arc<dyn Fn ... + Send + Sync>`.
        // To make Python callback Send+Sync, we can use a wrapper that acquires GIL.

        let callback_fn = move |inner: &RustExpr, var: &str, inner_prime: &RustExpr| -> RustExpr {
            Python::attach(|py| {
                // Convert Rust types to Python objects using into_pyobject
                // Handle conversion errors gracefully instead of panicking
                let py_inner = match PyExpr(inner.clone()).into_pyobject(py) {
                    Ok(obj) => obj,
                    Err(_) => return RustExpr::number(0.0),
                };
                let py_var = match var.into_pyobject(py) {
                    Ok(obj) => obj,
                    Err(_) => return RustExpr::number(0.0),
                };
                let py_inner_prime = match PyExpr(inner_prime.clone()).into_pyobject(py) {
                    Ok(obj) => obj,
                    Err(_) => return RustExpr::number(0.0),
                };

                let result = callback.call1(py, (py_inner, py_var, py_inner_prime));

                match result {
                    Ok(res) => {
                        if let Ok(py_expr) = res.extract::<PyExpr>(py) {
                            py_expr.0
                        } else {
                            // Fallback
                            RustExpr::number(0.0)
                        }
                    }
                    Err(_) => RustExpr::number(0.0), // Ignore python errors inside derivation
                }
            })
        };

        self_.inner = self_.inner.clone().custom_derivative(name, callback_fn);
        Ok(self_)
    }

    fn diff_str(&self, formula: &str, var: &str) -> PyResult<String> {
        self.inner
            .diff_str(formula, var)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }

    fn differentiate(&self, expr: &PyExpr, var: &str) -> PyResult<PyExpr> {
        let sym = crate::Symbol::new(var);
        self.inner
            .differentiate(expr.0.clone(), &sym)
            .map(PyExpr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }
}

/// Builder for simplification operations
#[pyclass(name = "Simplify")]
struct PySimplify {
    inner: builder::Simplify,
}

#[pymethods]
impl PySimplify {
    #[new]
    fn new() -> Self {
        PySimplify {
            inner: builder::Simplify::new(),
        }
    }

    fn domain_safe(mut self_: PyRefMut<'_, Self>, safe: bool) -> PyRefMut<'_, Self> {
        self_.inner = self_.inner.clone().domain_safe(safe);
        self_
    }

    fn fixed_var(mut self_: PyRefMut<'_, Self>, var: String) -> PyRefMut<'_, Self> {
        let sym = crate::Symbol::new(var);
        self_.inner = self_.inner.clone().fixed_var(&sym);
        self_
    }

    fn max_depth(mut self_: PyRefMut<'_, Self>, depth: usize) -> PyRefMut<'_, Self> {
        self_.inner = self_.inner.clone().max_depth(depth);
        self_
    }

    fn max_nodes(mut self_: PyRefMut<'_, Self>, nodes: usize) -> PyRefMut<'_, Self> {
        self_.inner = self_.inner.clone().max_nodes(nodes);
        self_
    }

    fn simplify(&self, expr: &PyExpr) -> PyResult<PyExpr> {
        self.inner
            .simplify(expr.0.clone())
            .map(PyExpr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }

    fn simplify_str(&self, formula: &str) -> PyResult<String> {
        self.inner
            .simplify_str(formula)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }
}

/// Differentiate a mathematical expression symbolically.
#[pyfunction]
#[pyo3(signature = (formula, var, fixed_vars=None, custom_functions=None))]
fn diff(
    formula: &str,
    var: &str,
    fixed_vars: Option<Vec<String>>,
    custom_functions: Option<Vec<String>>,
) -> PyResult<String> {
    let fixed: Option<Vec<String>> = fixed_vars.map(|v| v.iter().map(|s| s.to_string()).collect());
    let custom: Option<Vec<String>> =
        custom_functions.map(|v| v.iter().map(|s| s.to_string()).collect());

    crate::diff(
        formula.to_string(),
        var.to_string(),
        fixed.as_deref(),
        custom.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
}

/// Simplify a mathematical expression.
#[pyfunction]
#[pyo3(signature = (formula, fixed_vars=None, custom_functions=None))]
fn simplify(
    formula: &str,
    fixed_vars: Option<Vec<String>>,
    custom_functions: Option<Vec<String>>,
) -> PyResult<String> {
    let fixed: Option<Vec<String>> = fixed_vars.map(|v| v.iter().map(|s| s.to_string()).collect());
    let custom: Option<Vec<String>> =
        custom_functions.map(|v| v.iter().map(|s| s.to_string()).collect());

    crate::simplify(formula.to_string(), fixed.as_deref(), custom.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
}

/// Parse a mathematical expression and return its string representation.
#[pyfunction]
#[pyo3(signature = (formula, fixed_vars=None, custom_functions=None))]
fn parse(
    formula: &str,
    fixed_vars: Option<Vec<String>>,
    custom_functions: Option<Vec<String>>,
) -> PyResult<String> {
    let fixed: HashSet<String> = fixed_vars
        .map(|v| v.into_iter().collect())
        .unwrap_or_default();
    let custom: HashSet<String> = custom_functions
        .map(|v| v.into_iter().collect())
        .unwrap_or_default();

    crate::parse(formula, &fixed, &custom)
        .map(|expr| expr.to_string())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
}

#[pymodule]
fn symb_anafis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpr>()?;
    m.add_class::<PyDiff>()?;
    m.add_class::<PySimplify>()?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
