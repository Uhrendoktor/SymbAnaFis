use super::rules::{ExprKind, RuleContext, RuleRegistry};
use crate::Expr;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::OnceLock;

/// Check if tracing is enabled via environment variable (cached)
fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var("SYMB_TRACE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    })
}

/// Main simplification engine with rule-based architecture
pub(crate) struct Simplifier {
    registry: RuleRegistry,
    rule_caches: HashMap<String, HashMap<Expr, Option<Expr>>>, // Per-rule memoization with content-based keys
    max_iterations: usize,
    max_depth: usize,
    context: RuleContext,
    domain_safe: bool,
}

impl Default for Simplifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Simplifier {
    pub fn new() -> Self {
        let mut registry = RuleRegistry::new();
        registry.load_all_rules();
        registry.order_by_dependencies();

        Self {
            registry,
            rule_caches: HashMap::new(),
            max_iterations: 1000,
            max_depth: 50,
            context: RuleContext::default(),
            domain_safe: false,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_domain_safe(mut self, domain_safe: bool) -> Self {
        self.domain_safe = domain_safe;
        self
    }

    pub fn with_variables(mut self, variables: HashSet<String>) -> Self {
        self.context = self.context.with_variables(variables);
        self
    }

    pub fn with_fixed_vars(mut self, fixed_vars: HashSet<String>) -> Self {
        self.context = self.context.with_fixed_vars(fixed_vars);
        self
    }

    /// Main simplification entry point
    pub fn simplify(&mut self, expr: Expr) -> Expr {
        let mut current = Rc::new(expr);
        let mut iterations = 0;
        let mut seen_expressions: HashSet<Expr> = HashSet::new();

        loop {
            if iterations >= self.max_iterations {
                eprintln!(
                    "Warning: Simplification exceeded maximum iterations ({})",
                    self.max_iterations
                );
                break;
            }

            let original = current.clone();
            current = self.apply_rules_bottom_up(current, 0);

            if trace_enabled() {
                eprintln!(
                    "[DEBUG] Iteration {}: {} -> {}",
                    iterations, original, current
                );
            }

            // Use structural equality to check if expression changed
            if *current == *original {
                break; // No changes
            }

            // After a full pass of all rules, check if we've seen this result before
            // Use normalized form for proper cycle detection (handles Sub vs Add(-1*x) equivalence)
            let normalized = crate::simplification::helpers::normalize_for_comparison(&current);
            if seen_expressions.contains(&normalized) {
                // We're in a cycle - stop here with the current result
                if trace_enabled() {
                    eprintln!("[DEBUG] Cycle detected, stopping");
                }
                break;
            }
            // Add AFTER checking, so first iteration's result doesn't trigger false positive
            seen_expressions.insert(normalized);

            iterations += 1;
        }

        (*current).clone()
    }

    /// Apply rules bottom-up through the expression tree
    fn apply_rules_bottom_up(&mut self, expr: Rc<Expr>, depth: usize) -> Rc<Expr> {
        if depth > self.max_depth {
            return expr;
        }

        match expr.as_ref() {
            Expr::Add(u, v) => {
                let u_simplified = self.apply_rules_bottom_up(u.clone(), depth + 1);
                let v_simplified = self.apply_rules_bottom_up(v.clone(), depth + 1);

                // Only create new node if children actually changed
                if Rc::ptr_eq(&u_simplified, u) && Rc::ptr_eq(&v_simplified, v) {
                    self.apply_rules_to_node(expr, depth)
                } else {
                    let new_expr = Rc::new(Expr::Add(u_simplified, v_simplified));
                    self.apply_rules_to_node(new_expr, depth)
                }
            }
            Expr::Sub(u, v) => {
                let u_simplified = self.apply_rules_bottom_up(u.clone(), depth + 1);
                let v_simplified = self.apply_rules_bottom_up(v.clone(), depth + 1);

                if Rc::ptr_eq(&u_simplified, u) && Rc::ptr_eq(&v_simplified, v) {
                    self.apply_rules_to_node(expr, depth)
                } else {
                    let new_expr = Rc::new(Expr::Sub(u_simplified, v_simplified));
                    self.apply_rules_to_node(new_expr, depth)
                }
            }
            Expr::Mul(u, v) => {
                let u_simplified = self.apply_rules_bottom_up(u.clone(), depth + 1);
                let v_simplified = self.apply_rules_bottom_up(v.clone(), depth + 1);

                if Rc::ptr_eq(&u_simplified, u) && Rc::ptr_eq(&v_simplified, v) {
                    self.apply_rules_to_node(expr, depth)
                } else {
                    let new_expr = Rc::new(Expr::Mul(u_simplified, v_simplified));
                    self.apply_rules_to_node(new_expr, depth)
                }
            }
            Expr::Div(u, v) => {
                let u_simplified = self.apply_rules_bottom_up(u.clone(), depth + 1);
                let v_simplified = self.apply_rules_bottom_up(v.clone(), depth + 1);

                if Rc::ptr_eq(&u_simplified, u) && Rc::ptr_eq(&v_simplified, v) {
                    self.apply_rules_to_node(expr, depth)
                } else {
                    let new_expr = Rc::new(Expr::Div(u_simplified, v_simplified));
                    self.apply_rules_to_node(new_expr, depth)
                }
            }
            Expr::Pow(u, v) => {
                let u_simplified = self.apply_rules_bottom_up(u.clone(), depth + 1);
                let v_simplified = self.apply_rules_bottom_up(v.clone(), depth + 1);

                if Rc::ptr_eq(&u_simplified, u) && Rc::ptr_eq(&v_simplified, v) {
                    self.apply_rules_to_node(expr, depth)
                } else {
                    let new_expr = Rc::new(Expr::Pow(u_simplified, v_simplified));
                    self.apply_rules_to_node(new_expr, depth)
                }
            }
            Expr::FunctionCall { name, args } => {
                let args_simplified: Vec<Rc<Expr>> = args
                    .iter()
                    .map(|arg| self.apply_rules_bottom_up(Rc::new(arg.clone()), depth + 1))
                    .collect();

                // Check if any arg changed
                let changed = args_simplified
                    .iter()
                    .zip(args.iter())
                    .any(|(new, old)| new.as_ref() != old);

                if !changed {
                    self.apply_rules_to_node(expr, depth)
                } else {
                    let new_expr = Rc::new(Expr::FunctionCall {
                        name: name.clone(),
                        args: args_simplified
                            .into_iter()
                            .map(|rc| (*rc).clone())
                            .collect(),
                    });
                    self.apply_rules_to_node(new_expr, depth)
                }
            }
            _ => self.apply_rules_to_node(expr, depth),
        }
    }

    /// Apply all applicable rules to a single node in dependency order
    fn apply_rules_to_node(&mut self, mut current: Rc<Expr>, depth: usize) -> Rc<Expr> {
        let mut context = self
            .context
            .clone()
            .with_depth(depth)
            .with_domain_safe(self.domain_safe);

        // Get the expression kind once and only check rules that apply to it
        let kind = ExprKind::of(current.as_ref());
        let applicable_rules = self.registry.get_rules_for_kind(kind);

        for rule in applicable_rules {
            if context.domain_safe && rule.alters_domain() {
                continue;
            }

            let rule_name = rule.name();

            // Check per-rule cache using content-based key
            let cache_key = current.as_ref().clone();
            if let Some(cache) = self.rule_caches.get(rule_name)
                && let Some(cached_result) = cache.get(&cache_key)
            {
                if let Some(new_expr) = cached_result {
                    current = Rc::new(new_expr.clone());
                    continue;
                } else {
                    continue; // Cached as "no change"
                }
            }

            // Apply rule
            let original = current.as_ref().clone();
            if let Some(new_expr) = rule.apply(current.as_ref(), &context) {
                if trace_enabled() {
                    eprintln!("[TRACE] {} : {} => {}", rule_name, current, new_expr);
                }
                current = Rc::new(new_expr.clone());

                // Cache the transformation
                self.rule_caches
                    .entry(rule_name.to_string())
                    .or_default()
                    .insert(original, Some(new_expr));

                context = context.with_parent(current.as_ref().clone());
            } else {
                // Cache as "no change"
                self.rule_caches
                    .entry(rule_name.to_string())
                    .or_default()
                    .insert(original, None);
            }
        }

        current
    }
}

/// Convenience function with user-specified fixed variables
pub(crate) fn simplify_expr_with_fixed_vars(expr: Expr, fixed_vars: HashSet<String>) -> Expr {
    let variables = expr.variables();
    // Skip verification for performance - just simplify directly
    let mut simplifier = Simplifier::new()
        .with_max_iterations(1000)
        .with_max_depth(20)
        .with_variables(variables)
        .with_fixed_vars(fixed_vars);
    simplifier.simplify(expr)
}
