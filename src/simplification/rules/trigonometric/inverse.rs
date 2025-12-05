use crate::ast::Expr;
use crate::simplification::rules::{ExprKind, Rule, RuleCategory, RuleContext};

rule!(InverseTrigIdentityRule, "inverse_trig_identity", 90, Trigonometric, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::FunctionCall { name, args } = expr
        && args.len() == 1
        && let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0]
        && inner_args.len() == 1
    {
        let inner_arg = &inner_args[0];
        match (name.as_str(), inner_name.as_str()) {
            ("sin", "asin") | ("cos", "acos") | ("tan", "atan") => {
                return Some(inner_arg.clone());
            }
            _ => {}
        }
    }
    None
});

rule!(InverseTrigCompositionRule, "inverse_trig_composition", 85, Trigonometric, &[ExprKind::Function], alters_domain: true, |expr: &Expr, _context: &RuleContext| {
    if let Expr::FunctionCall { name, args } = expr
        && args.len() == 1
        && let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
        } = &args[0]
        && inner_args.len() == 1
    {
        let inner_arg = &inner_args[0];
        match (name.as_str(), inner_name.as_str()) {
            ("asin", "sin") | ("acos", "cos") | ("atan", "tan") => {
                return Some(inner_arg.clone());
            }
            _ => {}
        }
    }
    None
});
