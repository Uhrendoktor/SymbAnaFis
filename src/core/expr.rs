//! Abstract Syntax Tree for mathematical expressions
//!
//! N-ary Sum/Product architecture for efficient simplification.
//! Phase-specific epoch tracking for skip-if-already-processed optimization.

use std::cmp::Ordering as CmpOrdering;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::symbol::{InternedSymbol, get_or_intern};

/// Type alias for custom evaluation functions map
pub(crate) type CustomEvalMap =
    std::collections::HashMap<String, std::sync::Arc<dyn Fn(&[f64]) -> Option<f64> + Send + Sync>>;

// =============================================================================
// EXPRESSION ID COUNTER
// =============================================================================

static EXPR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_id() -> u64 {
    EXPR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

// =============================================================================
// EXPR - The main expression type
// =============================================================================

#[derive(Debug, Clone)]
pub struct Expr {
    /// Unique ID for debugging and caching (not used in equality comparisons)
    pub id: u64,
    /// Structural hash for O(1) equality rejection (Phase 7b optimization)
    pub hash: u64,
    /// The kind of expression (structure)
    pub kind: ExprKind,
}

impl Deref for Expr {
    type Target = ExprKind;
    fn deref(&self) -> &Self::Target {
        &self.kind
    }
}

// Structural equality based on KIND only (with hash fast-reject)
impl PartialEq for Expr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Fast reject: different hashes mean definitely not equal
        if self.hash != other.hash {
            return false;
        }
        // Slow path: verify structural equality (handles hash collisions)
        self.kind == other.kind
    }
}

impl Eq for Expr {}

impl std::hash::Hash for Expr {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Use pre-computed hash directly
        self.hash.hash(state);
    }
}

// =============================================================================
// EXPRKIND - N-ary Sum/Product architecture
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// Constant number (e.g., 3.14, 1e10)
    Number(f64),

    /// Variable or constant symbol (e.g., "x", "a", "pi")
    /// Uses InternedSymbol for O(1) equality comparisons
    Symbol(InternedSymbol),

    /// Function call (built-in or custom)
    /// Uses InternedSymbol for O(1) name comparisons
    /// Args use Arc<Expr> for consistency with Sum/Product
    FunctionCall {
        name: InternedSymbol,
        args: Vec<Arc<Expr>>,
    },

    /// N-ary sum: a + b + c + ...
    /// Stored flat and sorted for canonical form.
    /// Subtraction is represented as: a - b = Sum([a, Product([-1, b])])
    Sum(Vec<Arc<Expr>>),

    /// N-ary product: a * b * c * ...
    /// Stored flat and sorted for canonical form.
    Product(Vec<Arc<Expr>>),

    /// Division (binary - not associative)
    Div(Arc<Expr>, Arc<Expr>),

    /// Exponentiation (binary - not associative)
    Pow(Arc<Expr>, Arc<Expr>),

    /// Partial derivative notation: ∂^order/∂var^order of inner expression
    Derivative {
        inner: Arc<Expr>,
        var: String,
        order: u32,
    },

    /// Polynomial in sparse representation (coefficient * powers)
    /// Used for efficient polynomial operations (differentiation, multiplication)
    Poly(super::poly::Polynomial),
}

// =============================================================================
// EXPR CONSTRUCTORS AND METHODS
// =============================================================================

/// Compute structural hash for an ExprKind (Phase 7b optimization).
/// Unlike get_term_hash in helpers.rs (which ignores numeric coefficients for
/// like-term grouping), this hashes ALL content for true structural equality.
fn compute_expr_hash(kind: &ExprKind) -> u64 {
    // FNV-1a constants
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    #[inline(always)]
    fn hash_u64(mut hash: u64, n: u64) -> u64 {
        for byte in n.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    #[inline(always)]
    fn hash_f64(hash: u64, n: f64) -> u64 {
        hash_u64(hash, n.to_bits())
    }

    #[inline(always)]
    fn hash_byte(mut hash: u64, b: u8) -> u64 {
        hash ^= b as u64;
        hash.wrapping_mul(FNV_PRIME)
    }

    fn hash_kind(hash: u64, kind: &ExprKind) -> u64 {
        match kind {
            ExprKind::Number(n) => {
                let h = hash_byte(hash, b'N');
                hash_f64(h, *n)
            }

            ExprKind::Symbol(s) => {
                let h = hash_byte(hash, b'S');
                hash_u64(h, s.id())
            }

            // Sum: Use commutative (order-independent) hashing
            ExprKind::Sum(terms) => {
                let h = hash_byte(hash, b'+');
                // Commutative: sum of individual hashes
                let mut acc: u64 = 0;
                for t in terms {
                    acc = acc.wrapping_add(t.hash);
                }
                hash_u64(h, acc)
            }

            // Product: Use commutative (order-independent) hashing
            ExprKind::Product(factors) => {
                let h = hash_byte(hash, b'*');
                // Commutative: sum of individual hashes
                let mut acc: u64 = 0;
                for f in factors {
                    acc = acc.wrapping_add(f.hash);
                }
                hash_u64(h, acc)
            }

            // Div: Non-commutative, ordered
            ExprKind::Div(num, den) => {
                let h = hash_byte(hash, b'/');
                let h = hash_u64(h, num.hash);
                hash_u64(h, den.hash)
            }

            // Pow: Non-commutative, ordered
            ExprKind::Pow(base, exp) => {
                let h = hash_byte(hash, b'^');
                let h = hash_u64(h, base.hash);
                hash_u64(h, exp.hash)
            }

            // FunctionCall: Name + ordered args
            ExprKind::FunctionCall { name, args } => {
                let h = hash_byte(hash, b'F');
                let h = hash_u64(h, name.id());
                args.iter().fold(h, |acc, arg| hash_u64(acc, arg.hash))
            }

            // Derivative: var + order + inner
            ExprKind::Derivative { inner, var, order } => {
                let h = hash_byte(hash, b'D');
                let h = var.as_bytes().iter().fold(h, |acc, &b| hash_byte(acc, b));
                let h = hash_u64(h, *order as u64);
                hash_u64(h, inner.hash)
            }

            // Polynomial: hash based on terms (order-independent)
            ExprKind::Poly(poly) => {
                let h = hash_byte(hash, b'P');
                // Hash each term: commutative, so sum hashes
                let mut acc: u64 = 0;
                for term in poly.terms() {
                    // Hash coefficient
                    let term_hash = hash_f64(0, term.coeff);
                    // Hash powers
                    let term_hash = term.powers.iter().fold(term_hash, |h, (sym, pow)| {
                        let h = hash_u64(h, sym.id());
                        hash_u64(h, *pow as u64)
                    });
                    acc = acc.wrapping_add(term_hash);
                }
                hash_u64(h, acc)
            }
        }
    }

    hash_kind(FNV_OFFSET, kind)
}

impl Expr {
    /// Create a new expression with fresh ID
    pub fn new(kind: ExprKind) -> Self {
        let hash = compute_expr_hash(&kind);
        Expr {
            id: next_id(),
            hash,
            kind,
        }
    }

    // -------------------------------------------------------------------------
    // Accessor methods
    // -------------------------------------------------------------------------

    /// Check if expression is a constant number and return its value
    pub fn as_number(&self) -> Option<f64> {
        match &self.kind {
            ExprKind::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Check if this expression is the number zero (with tolerance)
    #[inline]
    pub fn is_zero_num(&self) -> bool {
        self.as_number().is_some_and(super::traits::is_zero)
    }

    /// Check if this expression is the number one (with tolerance)
    #[inline]
    pub fn is_one_num(&self) -> bool {
        self.as_number().is_some_and(super::traits::is_one)
    }

    /// Check if this expression is the number negative one (with tolerance)
    #[inline]
    pub fn is_neg_one_num(&self) -> bool {
        self.as_number().is_some_and(super::traits::is_neg_one)
    }

    // -------------------------------------------------------------------------
    // Basic constructors
    // -------------------------------------------------------------------------

    /// Create a number expression
    pub fn number(n: f64) -> Self {
        Expr::new(ExprKind::Number(n))
    }

    /// Create a symbol expression (auto-interned)
    pub fn symbol(s: impl AsRef<str>) -> Self {
        Expr::new(ExprKind::Symbol(get_or_intern(s.as_ref())))
    }

    /// Create from an already-interned symbol
    pub(crate) fn from_interned(interned: InternedSymbol) -> Self {
        Expr::new(ExprKind::Symbol(interned))
    }

    /// Create a polynomial expression directly
    pub fn poly(p: super::poly::Polynomial) -> Self {
        // Empty polynomial is 0
        if p.terms().is_empty() {
            return Expr::number(0.0);
        }
        // Single constant term is just a number
        if p.terms().len() == 1 && p.terms()[0].powers.is_empty() {
            return Expr::number(p.terms()[0].coeff);
        }
        Expr::new(ExprKind::Poly(p))
    }

    // -------------------------------------------------------------------------
    // N-ary Sum constructor (smart - flattens and sorts)
    // -------------------------------------------------------------------------
}

/// Extract the polynomial base hash from a term (x, x^2, 3*x^3 all have base "x")
/// Returns None if term is not polynomial-like (contains functions, etc.)
fn get_poly_base_hash(expr: &Expr) -> Option<u64> {
    match &expr.kind {
        // Symbol x → hash of x
        ExprKind::Symbol(s) => Some(s.id()),
        // x^n where n is a positive integer → hash of x
        ExprKind::Pow(base, exp) => {
            if let ExprKind::Number(n) = &exp.kind
                && *n >= 1.0 && n.fract() == 0.0
                    && let ExprKind::Symbol(s) = &base.kind {
                        return Some(s.id());
                    }
            None
        }
        // c*x^n → hash of x
        ExprKind::Product(factors) => {
            let mut base_hash = None;
            for f in factors.iter() {
                match &f.kind {
                    ExprKind::Number(_) => {} // Skip coefficients
                    ExprKind::Symbol(s) => {
                        if base_hash.is_some() {
                            return None;
                        } // Multiple variables
                        base_hash = Some(s.id());
                    }
                    ExprKind::Pow(b, exp) => {
                        if let ExprKind::Number(n) = &exp.kind
                            && *n >= 1.0 && n.fract() == 0.0
                                && let ExprKind::Symbol(s) = &b.kind {
                                    if base_hash.is_some() {
                                        return None;
                                    }
                                    base_hash = Some(s.id());
                                    continue;
                                }
                        return None;
                    }
                    _ => return None, // Functions or other complex expressions
                }
            }
            base_hash
        }
        ExprKind::Number(_) => Some(0), // Constants have "base" 0 (can combine with any polynomial)
        _ => None,
    }
}

impl Expr {
    /// Create a sum expression from terms.
    /// Flattens nested sums. Sorting and like-term combination is deferred to simplification
    /// for performance (avoids O(N²) cascade during differentiation).
    ///
    /// Auto-optimization: If 3+ terms form a pure polynomial (only numbers, symbols,
    /// products of coeff*symbol^n), converts to Poly for O(N) differentiation.
    pub fn sum(terms: Vec<Expr>) -> Self {
        if terms.is_empty() {
            return Expr::number(0.0);
        }
        if terms.len() == 1 {
            return terms.into_iter().next().unwrap();
        }

        let mut flat: Vec<Arc<Expr>> = Vec::with_capacity(terms.len());
        let mut numeric_sum: f64 = 0.0;

        for t in terms {
            match t.kind {
                ExprKind::Sum(inner) => flat.extend(inner),
                ExprKind::Poly(poly) => {
                    // Flatten Poly into its terms (they'll be merged later)
                    for term_expr in poly.to_expr_terms() {
                        flat.push(Arc::new(term_expr));
                    }
                }
                ExprKind::Number(n) => numeric_sum += n, // Combine numbers immediately
                _ => flat.push(Arc::new(t)),
            }
        }

        // Add accumulated numeric constant at the BEGINNING (canonical order: numbers first)
        if numeric_sum.abs() > 1e-14 {
            flat.insert(0, Arc::new(Expr::number(numeric_sum)));
        }

        if flat.is_empty() {
            return Expr::number(0.0);
        }
        if flat.len() == 1 {
            return Arc::try_unwrap(flat.pop().unwrap()).unwrap_or_else(|arc| (*arc).clone());
        }

        // Streaming incremental Poly building: if term has same base as last, combine into Poly
        // Since terms are canonically sorted, same-base terms are always adjacent
        if flat.len() >= 2 {
            let mut result: Vec<Arc<Expr>> = Vec::new();
            let mut last_base: Option<u64> = None;

            for term in flat.into_iter() {
                let current_base = get_poly_base_hash(&term);

                if current_base.is_some() && current_base == last_base && !result.is_empty() {
                    // Same base as last term - merge into Poly
                    let last = result.pop().unwrap();
                    let last_expr = Arc::try_unwrap(last).unwrap_or_else(|arc| (*arc).clone());
                    let term_expr = Arc::try_unwrap(term).unwrap_or_else(|arc| (*arc).clone());

                    // If last was already a Poly, add to it; else create new Poly from both
                    let merged = if let ExprKind::Poly(poly) = &last_expr.kind {
                        // Convert Poly back to sum, add new term, reconvert
                        let mut terms_vec: Vec<Arc<Expr>> =
                            poly.to_expr_terms().into_iter().map(Arc::new).collect();
                        terms_vec.push(Arc::new(term_expr.clone()));
                        let temp_sum = Expr::new(ExprKind::Sum(terms_vec));
                        if let Some(new_poly) = super::poly::Polynomial::try_from_expr(&temp_sum) {
                            if new_poly.substitutions_count() == 0 {
                                Expr::poly(new_poly)
                            } else {
                                result.push(Arc::new(last_expr));
                                result.push(Arc::new(term_expr));
                                last_base = current_base;
                                continue;
                            }
                        } else {
                            result.push(Arc::new(last_expr));
                            result.push(Arc::new(term_expr));
                            last_base = current_base;
                            continue;
                        }
                    } else {
                        // Create new Poly from both terms
                        let temp_sum = Expr::new(ExprKind::Sum(vec![
                            Arc::new(last_expr),
                            Arc::new(term_expr),
                        ]));
                        if let Some(poly) = super::poly::Polynomial::try_from_expr(&temp_sum) {
                            if poly.substitutions_count() == 0 {
                                Expr::poly(poly)
                            } else {
                                // Can't convert - keep original
                                result.push(Arc::new(temp_sum));
                                last_base = current_base;
                                continue;
                            }
                        } else {
                            result.push(Arc::new(temp_sum));
                            last_base = current_base;
                            continue;
                        }
                    };
                    result.push(Arc::new(merged));
                } else {
                    // Different base - just add
                    result.push(term);
                }
                last_base = current_base;
            }

            if result.len() == 1 {
                return Arc::try_unwrap(result.pop().unwrap()).unwrap_or_else(|arc| (*arc).clone());
            }
            return Expr::new(ExprKind::Sum(result));
        }

        Expr::new(ExprKind::Sum(flat))
    }

    /// Create sum from Arc terms (flattens only, sorting deferred to simplification)
    pub fn sum_from_arcs(terms: Vec<Arc<Expr>>) -> Self {
        if terms.is_empty() {
            return Expr::number(0.0);
        }
        if terms.len() == 1 {
            return Arc::try_unwrap(terms.into_iter().next().unwrap())
                .unwrap_or_else(|arc| (*arc).clone());
        }

        // Flatten nested sums and combine numbers
        let mut flat: Vec<Arc<Expr>> = Vec::with_capacity(terms.len());
        let mut numeric_sum: f64 = 0.0;

        for t in terms {
            match &t.kind {
                ExprKind::Sum(inner) => flat.extend(inner.clone()),
                ExprKind::Number(n) => numeric_sum += n,
                _ => flat.push(t),
            }
        }

        // Add accumulated numeric constant at the BEGINNING (canonical order: numbers first)
        if numeric_sum.abs() > 1e-14 {
            flat.insert(0, Arc::new(Expr::number(numeric_sum)));
        }

        if flat.is_empty() {
            return Expr::number(0.0);
        }
        if flat.len() == 1 {
            return Arc::try_unwrap(flat.pop().unwrap()).unwrap_or_else(|arc| (*arc).clone());
        }

        Expr::new(ExprKind::Sum(flat))
    }

    // -------------------------------------------------------------------------
    // N-ary Product constructor (smart - flattens and sorts)
    // -------------------------------------------------------------------------

    /// Create a product expression from factors.
    /// Flattens nested products. Sorting deferred to simplification.
    pub fn product(factors: Vec<Expr>) -> Self {
        if factors.is_empty() {
            return Expr::number(1.0);
        }
        if factors.len() == 1 {
            return factors.into_iter().next().unwrap();
        }

        let mut flat: Vec<Arc<Expr>> = Vec::with_capacity(factors.len());
        let mut numeric_prod: f64 = 1.0;

        for f in factors {
            match f.kind {
                ExprKind::Product(inner) => flat.extend(inner),
                ExprKind::Number(n) => {
                    if n == 0.0 {
                        return Expr::number(0.0); // Early exit for zero
                    }
                    numeric_prod *= n;
                }
                _ => flat.push(Arc::new(f)),
            }
        }

        // Add numeric coefficient if not 1.0
        if (numeric_prod - 1.0).abs() > 1e-14 {
            flat.insert(0, Arc::new(Expr::number(numeric_prod)));
        }

        if flat.is_empty() {
            return Expr::number(1.0);
        }
        if flat.len() == 1 {
            return Arc::try_unwrap(flat.pop().unwrap()).unwrap_or_else(|arc| (*arc).clone());
        }

        Expr::new(ExprKind::Product(flat))
    }

    /// Create product from Arc factors (flattens and sorts for canonical form)
    pub fn product_from_arcs(factors: Vec<Arc<Expr>>) -> Self {
        if factors.is_empty() {
            return Expr::number(1.0);
        }
        if factors.len() == 1 {
            return Arc::try_unwrap(factors.into_iter().next().unwrap())
                .unwrap_or_else(|arc| (*arc).clone());
        }

        // Flatten nested products
        let mut flat: Vec<Arc<Expr>> = Vec::with_capacity(factors.len());
        for f in factors {
            match &f.kind {
                ExprKind::Product(inner) => flat.extend(inner.clone()),
                _ => flat.push(f),
            }
        }

        if flat.len() == 1 {
            return Arc::try_unwrap(flat.pop().unwrap()).unwrap_or_else(|arc| (*arc).clone());
        }

        // Sort for canonical form
        flat.sort_by(|a, b| expr_cmp(a, b));
        Expr::new(ExprKind::Product(flat))
    }

    // -------------------------------------------------------------------------
    // Binary operation constructors (for legacy compatibility during migration)
    // -------------------------------------------------------------------------

    /// Create addition: a + b → Sum([a, b])
    pub fn add_expr(left: Expr, right: Expr) -> Self {
        Expr::sum(vec![left, right])
    }

    /// Create subtraction: a - b → Sum([a, Product([-1, b])])
    pub fn sub_expr(left: Expr, right: Expr) -> Self {
        let neg_right = Expr::product(vec![Expr::number(-1.0), right]);
        Expr::sum(vec![left, neg_right])
    }

    /// Create multiplication: a * b → Product([a, b])
    ///
    /// Optimization: If both operands are Poly, use fast O(N*M) polynomial multiplication
    pub fn mul_expr(left: Expr, right: Expr) -> Self {
        // Fast path: Poly * Poly uses polynomial multiplication
        if let (ExprKind::Poly(p1), ExprKind::Poly(p2)) = (&left.kind, &right.kind) {
            let result = p1.mul(p2);
            return Expr::poly(result);
        }

        Expr::product(vec![left, right])
    }

    /// Create multiplication from Arc operands (avoids Expr cloning)
    pub fn mul_from_arcs(factors: Vec<Arc<Expr>>) -> Self {
        Expr::product_from_arcs(factors)
    }

    /// Unwrap an Arc<Expr> without cloning if refcount is 1
    #[inline]
    pub fn unwrap_arc(arc: Arc<Expr>) -> Expr {
        Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
    }

    /// Create division
    pub fn div_expr(left: Expr, right: Expr) -> Self {
        Expr::new(ExprKind::Div(Arc::new(left), Arc::new(right)))
    }

    /// Create division from Arc operands (avoids cloning if Arc ref count is 1)
    pub fn div_from_arcs(left: Arc<Expr>, right: Arc<Expr>) -> Self {
        Expr::new(ExprKind::Div(left, right))
    }

    /// Create power expression
    pub fn pow(base: Expr, exponent: Expr) -> Self {
        Expr::new(ExprKind::Pow(Arc::new(base), Arc::new(exponent)))
    }

    /// Create power from Arc operands (avoids cloning if Arc ref count is 1)
    pub fn pow_from_arcs(base: Arc<Expr>, exponent: Arc<Expr>) -> Self {
        Expr::new(ExprKind::Pow(base, exponent))
    }

    /// Create a function call expression (single argument)
    pub fn func(name: impl AsRef<str>, content: Expr) -> Self {
        Expr::new(ExprKind::FunctionCall {
            name: get_or_intern(name.as_ref()),
            args: vec![Arc::new(content)],
        })
    }

    /// Create a multi-argument function call
    pub fn func_multi(name: impl AsRef<str>, args: Vec<Expr>) -> Self {
        Expr::new(ExprKind::FunctionCall {
            name: get_or_intern(name.as_ref()),
            args: args.into_iter().map(Arc::new).collect(),
        })
    }

    /// Create a function call from Arc arguments (avoids cloning)
    pub fn func_multi_from_arcs(name: impl AsRef<str>, args: Vec<Arc<Expr>>) -> Self {
        Expr::new(ExprKind::FunctionCall {
            name: get_or_intern(name.as_ref()),
            args,
        })
    }

    /// Create a function call with explicit arguments using array syntax
    pub fn call<const N: usize>(name: impl AsRef<str>, args: [Expr; N]) -> Self {
        Expr::func_multi(name, args.into())
    }

    /// Create a partial derivative expression
    pub fn derivative(inner: Expr, var: String, order: u32) -> Self {
        Expr::new(ExprKind::Derivative {
            inner: Arc::new(inner),
            var,
            order,
        })
    }

    // -------------------------------------------------------------------------
    // Negation helper
    // -------------------------------------------------------------------------

    /// Negate this expression: -x = Product([-1, x])
    pub fn negate(self) -> Self {
        Expr::product(vec![Expr::number(-1.0), self])
    }

    // -------------------------------------------------------------------------
    // Analysis methods
    // -------------------------------------------------------------------------

    /// Count the total number of nodes in the AST
    pub fn node_count(&self) -> usize {
        match &self.kind {
            ExprKind::Number(_) | ExprKind::Symbol(_) => 1,
            ExprKind::FunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.node_count()).sum::<usize>()
            }
            ExprKind::Sum(terms) => 1 + terms.iter().map(|t| t.node_count()).sum::<usize>(),
            ExprKind::Product(factors) => 1 + factors.iter().map(|f| f.node_count()).sum::<usize>(),
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => 1 + l.node_count() + r.node_count(),
            ExprKind::Derivative { inner, .. } => 1 + inner.node_count(),
            // Poly is counted as 1 node + its expanded form
            ExprKind::Poly(poly) => 1 + poly.terms().len(),
        }
    }

    /// Get the maximum nesting depth of the AST
    pub fn max_depth(&self) -> usize {
        match &self.kind {
            ExprKind::Number(_) | ExprKind::Symbol(_) => 1,
            ExprKind::FunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.max_depth()).max().unwrap_or(0)
            }
            ExprKind::Sum(terms) => 1 + terms.iter().map(|t| t.max_depth()).max().unwrap_or(0),
            ExprKind::Product(factors) => {
                1 + factors.iter().map(|f| f.max_depth()).max().unwrap_or(0)
            }
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => 1 + l.max_depth().max(r.max_depth()),
            ExprKind::Derivative { inner, .. } => 1 + inner.max_depth(),
            ExprKind::Poly(_) => 2, // Poly is shallow: one level for poly, one for terms
        }
    }

    /// Check if the expression contains a specific variable
    pub fn contains_var(&self, var: &str) -> bool {
        match &self.kind {
            ExprKind::Number(_) => false,
            ExprKind::Symbol(s) => s == var,
            ExprKind::FunctionCall { args, .. } => args.iter().any(|a| a.contains_var(var)),
            ExprKind::Sum(terms) => terms.iter().any(|t| t.contains_var(var)),
            ExprKind::Product(factors) => factors.iter().any(|f| f.contains_var(var)),
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => l.contains_var(var) || r.contains_var(var),
            ExprKind::Derivative { inner, var: v, .. } => v == var || inner.contains_var(var),
            ExprKind::Poly(poly) => poly
                .terms()
                .iter()
                .any(|t| t.powers.iter().any(|(s, _)| s == var)),
        }
    }

    /// Check if the expression contains any free variables
    pub fn has_free_variables(&self, excluded: &std::collections::HashSet<String>) -> bool {
        match &self.kind {
            ExprKind::Number(_) => false,
            ExprKind::Symbol(name) => !excluded.contains(name.as_ref()),
            ExprKind::Sum(terms) => terms.iter().any(|t| t.has_free_variables(excluded)),
            ExprKind::Product(factors) => factors.iter().any(|f| f.has_free_variables(excluded)),
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => {
                l.has_free_variables(excluded) || r.has_free_variables(excluded)
            }
            ExprKind::FunctionCall { args, .. } => {
                args.iter().any(|arg| arg.has_free_variables(excluded))
            }
            ExprKind::Derivative { inner, var, .. } => {
                !excluded.contains(var) || inner.has_free_variables(excluded)
            }
            ExprKind::Poly(poly) => poly
                .terms()
                .iter()
                .any(|t| t.powers.iter().any(|(s, _)| !excluded.contains(s.as_ref()))),
        }
    }

    /// Collect all variables in the expression
    pub fn variables(&self) -> std::collections::HashSet<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match &self.kind {
            ExprKind::Symbol(s) => {
                if let Some(name) = s.name() {
                    vars.insert(name.to_string());
                }
            }
            ExprKind::FunctionCall { args, .. } => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
            ExprKind::Sum(terms) => {
                for t in terms {
                    t.collect_variables(vars);
                }
            }
            ExprKind::Product(factors) => {
                for f in factors {
                    f.collect_variables(vars);
                }
            }
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => {
                l.collect_variables(vars);
                r.collect_variables(vars);
            }
            ExprKind::Derivative { inner, var, .. } => {
                vars.insert(var.clone());
                inner.collect_variables(vars);
            }
            ExprKind::Number(_) => {}
            ExprKind::Poly(poly) => {
                for term in poly.terms() {
                    for (sym, _) in &term.powers {
                        if let Some(name) = sym.name() {
                            vars.insert(name.to_string());
                        }
                    }
                }
            }
        }
    }

    /// Create a deep clone (with new IDs)
    pub fn deep_clone(&self) -> Expr {
        match &self.kind {
            ExprKind::Number(n) => Expr::number(*n),
            ExprKind::Symbol(s) => Expr::from_interned(s.clone()),
            ExprKind::FunctionCall { name, args } => Expr::new(ExprKind::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|arg| Arc::new(arg.deep_clone())).collect(),
            }),
            ExprKind::Sum(terms) => {
                let cloned: Vec<Arc<Expr>> = terms
                    .iter()
                    .map(|t| Arc::new(t.as_ref().deep_clone()))
                    .collect();
                Expr::new(ExprKind::Sum(cloned))
            }
            ExprKind::Product(factors) => {
                let cloned: Vec<Arc<Expr>> = factors
                    .iter()
                    .map(|f| Arc::new(f.as_ref().deep_clone()))
                    .collect();
                Expr::new(ExprKind::Product(cloned))
            }
            ExprKind::Div(a, b) => Expr::div_expr(a.as_ref().deep_clone(), b.as_ref().deep_clone()),
            ExprKind::Pow(a, b) => Expr::pow(a.as_ref().deep_clone(), b.as_ref().deep_clone()),
            ExprKind::Derivative { inner, var, order } => {
                Expr::derivative(inner.as_ref().deep_clone(), var.clone(), *order)
            }
            ExprKind::Poly(poly) => {
                // Poly is expensive to deep clone, so we just clone it
                Expr::new(ExprKind::Poly(poly.clone()))
            }
        }
    }

    // -------------------------------------------------------------------------
    // Convenience methods
    // -------------------------------------------------------------------------

    /// Differentiate with respect to a variable
    pub fn diff(&self, var: &str) -> Result<Expr, crate::DiffError> {
        crate::Diff::new().differentiate(self.clone(), &crate::symb(var))
    }

    /// Simplify this expression
    pub fn simplified(&self) -> Result<Expr, crate::DiffError> {
        crate::Simplify::new().simplify(self.clone())
    }

    /// Fold over the expression tree (pre-order)
    pub fn fold<T, F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, &Expr) -> T + Copy,
    {
        let acc = f(init, self);
        match &self.kind {
            ExprKind::Number(_) | ExprKind::Symbol(_) => acc,
            ExprKind::FunctionCall { args, .. } => args.iter().fold(acc, |a, arg| arg.fold(a, f)),
            ExprKind::Sum(terms) => terms.iter().fold(acc, |a, t| t.fold(a, f)),
            ExprKind::Product(factors) => factors.iter().fold(acc, |a, f_| f_.fold(a, f)),
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => {
                let acc = l.fold(acc, f);
                r.fold(acc, f)
            }
            ExprKind::Derivative { inner, .. } => inner.fold(acc, f),
            ExprKind::Poly(_) => acc, // Poly is opaque for folding
        }
    }

    /// Transform the expression tree (post-order)
    pub fn map<F>(&self, f: F) -> Expr
    where
        F: Fn(&Expr) -> Expr + Copy,
    {
        let transformed = match &self.kind {
            ExprKind::Number(_) | ExprKind::Symbol(_) => self.clone(),
            ExprKind::FunctionCall { name, args } => Expr::new(ExprKind::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|arg| Arc::new(arg.map(f))).collect(),
            }),
            ExprKind::Sum(terms) => {
                let mapped: Vec<Arc<Expr>> =
                    terms.iter().map(|t| Arc::new(t.as_ref().map(f))).collect();
                Expr::new(ExprKind::Sum(mapped))
            }
            ExprKind::Product(factors) => {
                let mapped: Vec<Arc<Expr>> = factors
                    .iter()
                    .map(|fac| Arc::new(fac.as_ref().map(f)))
                    .collect();
                Expr::new(ExprKind::Product(mapped))
            }
            ExprKind::Div(a, b) => Expr::div_expr(a.map(f), b.map(f)),
            ExprKind::Pow(a, b) => Expr::pow(a.map(f), b.map(f)),
            ExprKind::Derivative { inner, var, order } => {
                Expr::derivative(inner.map(f), var.clone(), *order)
            }
            ExprKind::Poly(poly) => {
                // Poly is opaque for mapping - just clone
                Expr::new(ExprKind::Poly(poly.clone()))
            }
        };
        f(&transformed)
    }

    /// Substitute a variable with another expression
    pub fn substitute(&self, var: &str, replacement: &Expr) -> Expr {
        self.map(|node| {
            if let ExprKind::Symbol(s) = &node.kind
                && s == var
            {
                return replacement.clone();
            }
            node.clone()
        })
    }

    /// Evaluate expression with given variable values
    pub fn evaluate(&self, vars: &std::collections::HashMap<&str, f64>) -> Expr {
        self.evaluate_with_custom(vars, &std::collections::HashMap::new())
    }

    /// Evaluate with custom function evaluators
    pub fn evaluate_with_custom(
        &self,
        vars: &std::collections::HashMap<&str, f64>,
        custom_evals: &CustomEvalMap,
    ) -> Expr {
        match &self.kind {
            ExprKind::Number(n) => Expr::number(*n),
            ExprKind::Symbol(s) => {
                if let Some(name) = s.name()
                    && let Some(&val) = vars.get(name)
                {
                    return Expr::number(val);
                }
                self.clone()
            }
            ExprKind::FunctionCall { name, args } => {
                let eval_args: Vec<Expr> = args
                    .iter()
                    .map(|a| a.evaluate_with_custom(vars, custom_evals))
                    .collect();

                let numeric_args: Option<Vec<f64>> =
                    eval_args.iter().map(|e| e.as_number()).collect();

                if let Some(args_vec) = numeric_args {
                    if let Some(custom_eval) = custom_evals.get(name.as_str())
                        && let Some(result) = custom_eval(&args_vec)
                    {
                        return Expr::number(result);
                    }
                    if let Some(func_def) = crate::functions::registry::Registry::get(name.as_str())
                        && let Some(result) = (func_def.eval)(&args_vec)
                    {
                        return Expr::number(result);
                    }
                }

                Expr::new(ExprKind::FunctionCall {
                    name: name.clone(),
                    args: eval_args.into_iter().map(Arc::new).collect(),
                })
            }
            ExprKind::Sum(terms) => {
                // Optimized: single-pass accumulation
                let mut num_sum: f64 = 0.0;
                let mut others: Vec<Expr> = Vec::new();

                for t in terms {
                    let eval_t = t.evaluate_with_custom(vars, custom_evals);
                    if let ExprKind::Number(n) = eval_t.kind {
                        num_sum += n;
                    } else {
                        others.push(eval_t);
                    }
                }

                if others.is_empty() {
                    Expr::number(num_sum)
                } else if num_sum != 0.0 {
                    others.push(Expr::number(num_sum));
                    Expr::sum(others)
                } else if others.len() == 1 {
                    others.pop().unwrap()
                } else {
                    Expr::sum(others)
                }
            }
            ExprKind::Product(factors) => {
                // Optimized: single-pass accumulation with early zero exit
                let mut num_prod: f64 = 1.0;
                let mut others: Vec<Expr> = Vec::new();

                for f in factors {
                    let eval_f = f.evaluate_with_custom(vars, custom_evals);
                    if let ExprKind::Number(n) = eval_f.kind {
                        if n == 0.0 {
                            return Expr::number(0.0); // Early exit
                        }
                        num_prod *= n;
                    } else {
                        others.push(eval_f);
                    }
                }

                if others.is_empty() {
                    Expr::number(num_prod)
                } else if num_prod != 1.0 {
                    others.insert(0, Expr::number(num_prod));
                    Expr::product(others)
                } else if others.len() == 1 {
                    others.pop().unwrap()
                } else {
                    Expr::product(others)
                }
            }
            ExprKind::Div(a, b) => {
                let ea = a.evaluate_with_custom(vars, custom_evals);
                let eb = b.evaluate_with_custom(vars, custom_evals);
                match (&ea.kind, &eb.kind) {
                    (ExprKind::Number(x), ExprKind::Number(y)) if *y != 0.0 => Expr::number(x / y),
                    _ => Expr::div_expr(ea, eb),
                }
            }
            ExprKind::Pow(a, b) => {
                let ea = a.evaluate_with_custom(vars, custom_evals);
                let eb = b.evaluate_with_custom(vars, custom_evals);
                match (&ea.kind, &eb.kind) {
                    (ExprKind::Number(x), ExprKind::Number(y)) => Expr::number(x.powf(*y)),
                    _ => Expr::pow(ea, eb),
                }
            }
            ExprKind::Derivative { inner, var, order } => Expr::derivative(
                inner.evaluate_with_custom(vars, custom_evals),
                var.clone(),
                *order,
            ),
            ExprKind::Poly(poly) => {
                // Direct polynomial evaluation: Σ (coeff × ∏ var^pow)
                let mut total = 0.0;
                let mut all_numeric = true;

                'outer: for term in poly.terms() {
                    let mut term_val = term.coeff;
                    for (sym, pow) in &term.powers {
                        if let Some(name) = sym.name() {
                            if let Some(&val) = vars.get(name) {
                                term_val *= val.powi(*pow as i32);
                            } else {
                                all_numeric = false;
                                break 'outer;
                            }
                        } else {
                            all_numeric = false;
                            break 'outer;
                        }
                    }
                    total += term_val;
                }

                if all_numeric {
                    Expr::number(total)
                } else {
                    // Partial evaluation not possible, return Poly as-is
                    self.clone()
                }
            }
        }
    }
}

// =============================================================================
// CANONICAL ORDERING FOR EXPRESSIONS
// =============================================================================

/// Compare expressions for canonical ordering.
/// Order: Numbers < Symbols (alphabetical) < Functions < Sum < Product < Div < Pow
fn expr_cmp(a: &Expr, b: &Expr) -> CmpOrdering {
    use ExprKind::*;

    // 1. Numbers always come first
    if let (Number(x), Number(y)) = (&a.kind, &b.kind) {
        return x.partial_cmp(y).unwrap_or(CmpOrdering::Equal);
    }
    if matches!(a.kind, Number(_)) {
        return CmpOrdering::Less;
    }
    if matches!(b.kind, Number(_)) {
        return CmpOrdering::Greater;
    }

    // Helper: Extract sort key (Base, Exponent, Coefficient)
    // Returns: (Base, Exponent, Coefficient, IsAtomic)
    // Note: Exponent is Option<&Expr> (None means 1), Coefficient is f64
    fn extract_key(e: &Expr) -> (&Expr, Option<&Expr>, f64, bool) {
        match &e.kind {
            // Case: x^2 -> Base x, Exp 2, Coeff 1
            Pow(b, exp) => (b.as_ref(), Some(exp.as_ref()), 1.0, false),

            // Case: 2*x -> Base x, Exp 1, Coeff 2 (Only if Product starts with Number)
            Product(factors) if factors.len() == 2 => {
                if let Number(n) = &factors[0].kind {
                    (&factors[1], None, *n, false)
                } else {
                    (e, None, 1.0, true)
                }
            }
            // Case: x -> Base x, Exp 1, Coeff 1
            _ => (e, None, 1.0, true),
        }
    }

    let (base_a, exp_a, coeff_a, atomic_a) = extract_key(a);
    let (base_b, exp_b, coeff_b, atomic_b) = extract_key(b);

    // 2. If both are atomic (e.g., Symbol vs Symbol), use strict type sorting fallback
    // This prevents infinite recursion (comparing x vs x)
    if atomic_a && atomic_b {
        return expr_cmp_type_strict(a, b);
    }

    // 3. Compare Bases (Recursively)
    // Recursion is safe because at least one is composite (smaller depth)
    let base_cmp = expr_cmp(base_a, base_b);
    if base_cmp != CmpOrdering::Equal {
        return base_cmp;
    }

    // logic: 1 vs 2 -> Less
    // logic: 1 vs 1 -> Equal
    // logic: 2 vs 1 -> Greater

    // If one has explicit exponent and one implied 1:
    // x (1) vs x^2 (2) -> 1 < 2 -> Less
    match (exp_a, exp_b) {
        (Some(e_a), Some(e_b)) => {
            let exp_cmp = expr_cmp(e_a, e_b);
            if exp_cmp != CmpOrdering::Equal {
                return exp_cmp;
            }
        }
        (Some(e_a), None) => {
            // Compare expr e_a vs 1.0
            // Usually e_a > 1 (like 2, 3), but could be 0.5
            // Safer to compare full expr
            let one = Expr::number(1.0);
            let exp_cmp = expr_cmp(e_a, &one);
            if exp_cmp != CmpOrdering::Equal {
                return exp_cmp;
            }
        }
        (None, Some(e_b)) => {
            // Compare 1.0 vs e_b
            let one = Expr::number(1.0);
            let exp_cmp = expr_cmp(&one, e_b);
            if exp_cmp != CmpOrdering::Equal {
                return exp_cmp;
            }
        }
        (None, None) => {} // Both 1
    }

    // 5. Compare Coefficients (1 < 2)
    // x vs 2x -> 1 < 2 -> Less -> x, 2x
    coeff_a.partial_cmp(&coeff_b).unwrap_or(CmpOrdering::Equal)
}

// Fallback: Original strict type comparisons for atomic terms
fn expr_cmp_type_strict(a: &Expr, b: &Expr) -> CmpOrdering {
    use ExprKind::*;
    match (&a.kind, &b.kind) {
        (Symbol(x), Symbol(y)) => x.as_ref().cmp(y.as_ref()),
        (Symbol(_), _) => CmpOrdering::Less,
        (_, Symbol(_)) => CmpOrdering::Greater,

        (FunctionCall { name: n1, args: a1 }, FunctionCall { name: n2, args: a2 }) => {
            n1.cmp(n2).then_with(|| {
                for (x, y) in a1.iter().zip(a2.iter()) {
                    match expr_cmp(x, y) {
                        CmpOrdering::Equal => continue,
                        other => return other,
                    }
                }
                a1.len().cmp(&a2.len())
            })
        }
        (FunctionCall { .. }, _) => CmpOrdering::Less,
        (_, FunctionCall { .. }) => CmpOrdering::Greater,

        (Sum(t1), Sum(t2)) => t1.len().cmp(&t2.len()).then_with(|| {
            for (x, y) in t1.iter().zip(t2.iter()) {
                match expr_cmp(x, y) {
                    CmpOrdering::Equal => continue,
                    other => return other,
                }
            }
            CmpOrdering::Equal
        }),
        (Sum(_), _) => CmpOrdering::Less,
        (_, Sum(_)) => CmpOrdering::Greater,

        // Products are handled as atomics if they don't match the "Coeff * Rest" pattern
        (Product(f1), Product(f2)) => f1.len().cmp(&f2.len()).then_with(|| {
            for (x, y) in f1.iter().zip(f2.iter()) {
                match expr_cmp(x, y) {
                    CmpOrdering::Equal => continue,
                    other => return other,
                }
            }
            CmpOrdering::Equal
        }),
        (Product(_), _) => CmpOrdering::Less,
        (_, Product(_)) => CmpOrdering::Greater,

        (Div(l1, r1), Div(l2, r2)) => expr_cmp(l1, l2).then_with(|| expr_cmp(r1, r2)),
        (Div(_, _), _) => CmpOrdering::Less,
        (_, Div(_, _)) => CmpOrdering::Greater,

        (Pow(b1, e1), Pow(b2, e2)) => expr_cmp(b1, b2).then_with(|| expr_cmp(e1, e2)),
        (Pow(_, _), _) => CmpOrdering::Less,
        (_, Pow(_, _)) => CmpOrdering::Greater,

        (
            Derivative {
                inner: i1,
                var: v1,
                order: o1,
            },
            Derivative {
                inner: i2,
                var: v2,
                order: o2,
            },
        ) => v1
            .cmp(v2)
            .then_with(|| o1.cmp(o2))
            .then_with(|| expr_cmp(i1, i2)),

        _ => CmpOrdering::Equal, // Should be covered by match arms above but safe fallback
    }
}

// =============================================================================
// HASH FOR EXPRKIND
// =============================================================================

impl std::hash::Hash for ExprKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ExprKind::Number(n) => n.to_bits().hash(state),
            ExprKind::Symbol(s) => s.hash(state),
            ExprKind::FunctionCall { name, args } => {
                name.hash(state);
                args.hash(state);
            }
            ExprKind::Sum(terms) => {
                terms.len().hash(state);
                for t in terms {
                    t.hash(state);
                }
            }
            ExprKind::Product(factors) => {
                factors.len().hash(state);
                for f in factors {
                    f.hash(state);
                }
            }
            ExprKind::Div(l, r) | ExprKind::Pow(l, r) => {
                l.hash(state);
                r.hash(state);
            }
            ExprKind::Derivative { inner, var, order } => {
                inner.hash(state);
                var.hash(state);
                order.hash(state);
            }
            ExprKind::Poly(poly) => {
                // Hash polynomial terms
                poly.terms().len().hash(state);
                for term in poly.terms() {
                    term.coeff.to_bits().hash(state);
                    term.powers.len().hash(state);
                    for (sym, pow) in &term.powers {
                        sym.hash(state);
                        pow.hash(state);
                    }
                }
            }
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_flattening() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        let z = Expr::symbol("z");

        // (x + y) + z should flatten to Sum or Poly (3+ terms may become Poly)
        let inner = Expr::sum(vec![x.clone(), y.clone()]);
        let outer = Expr::sum(vec![inner, z.clone()]);

        match &outer.kind {
            ExprKind::Sum(terms) => assert_eq!(terms.len(), 3),
            ExprKind::Poly(poly) => assert_eq!(poly.terms().len(), 3),
            _ => panic!("Expected Sum or Poly"),
        }
    }

    #[test]
    fn test_product_flattening() {
        let a = Expr::symbol("a");
        let b = Expr::symbol("b");
        let c = Expr::symbol("c");

        // (a * b) * c should flatten to Product([a, b, c])
        let inner = Expr::product(vec![a.clone(), b.clone()]);
        let outer = Expr::product(vec![inner, c.clone()]);

        match &outer.kind {
            ExprKind::Product(factors) => assert_eq!(factors.len(), 3),
            _ => panic!("Expected Product"),
        }
    }

    #[test]
    fn test_subtraction_as_sum() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");

        // x - y = Sum([x, Product([-1, y])])
        let result = Expr::sub_expr(x.clone(), y.clone());

        match &result.kind {
            ExprKind::Sum(terms) => {
                assert_eq!(terms.len(), 2);
            }
            _ => panic!("Expected Sum from subtraction"),
        }
    }
}
