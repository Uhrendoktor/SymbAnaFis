use crate::Expr;
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::OnceLock;

/// Definition of a mathematical function including its evaluation and differentiation logic
#[derive(Clone)]
pub(crate) struct FunctionDefinition {
    /// Canonical name of the function (e.g., "sin", "besselj")
    pub name: &'static str,

    /// Acceptable argument count (arity)
    pub arity: RangeInclusive<usize>,

    /// Numerical evaluation function
    pub eval: fn(&[f64]) -> Option<f64>,

    /// Symbolic differentiation function
    /// Arguments: (args of the function call, derivatives of the arguments)
    /// Returns the total derivative dA/dx = sum( (dA/d_arg_i) * (d_arg_i/dx) )
    pub derivative: fn(&[Expr], &[Expr]) -> Expr,
}

impl FunctionDefinition {
    /// Helper to check if argument count is valid
    pub(crate) fn validate_arity(&self, args: usize) -> bool {
        self.arity.contains(&args)
    }
}

/// Static registry storing all function definitions
static REGISTRY: OnceLock<HashMap<&'static str, FunctionDefinition>> = OnceLock::new();

/// Initialize the registry with all function definitions
fn init_registry() -> HashMap<&'static str, FunctionDefinition> {
    let mut map = HashMap::with_capacity(70);

    // Populate from definitions
    for def in crate::functions::definitions::all_definitions() {
        map.insert(def.name, def);
    }

    map
}

/// Central registry for getting function definitions
pub(crate) struct Registry;

impl Registry {
    /// Get a function definition by name - O(1) HashMap lookup
    pub(crate) fn get(name: &str) -> Option<&'static FunctionDefinition> {
        REGISTRY.get_or_init(init_registry).get(name)
    }
}
