use crate::simplification::rules::Rule;
use std::rc::Rc;

pub(crate) mod conversions;
mod helpers;
pub(crate) mod identities;
pub(crate) mod ratios;

pub(crate) use conversions::*;
pub(crate) use identities::*;
pub(crate) use ratios::*;

/// Get all hyperbolic rules in priority order
pub(crate) fn get_hyperbolic_rules() -> Vec<Rc<dyn Rule>> {
    vec![
        // High priority rules first
        Rc::new(SinhZeroRule),
        Rc::new(CoshZeroRule),
        Rc::new(SinhAsinhIdentityRule),
        Rc::new(CoshAcoshIdentityRule),
        Rc::new(TanhAtanhIdentityRule),
        Rc::new(SinhNegationRule),
        Rc::new(CoshNegationRule),
        Rc::new(TanhNegationRule),
        // Identity rules
        Rc::new(HyperbolicIdentityRule),
        // Ratio rules - convert to tanh, coth, sech, csch
        Rc::new(SinhCoshToTanhRule),
        Rc::new(CoshSinhToCothRule),
        Rc::new(OneCoshToSechRule),
        Rc::new(OneSinhToCschRule),
        Rc::new(OneTanhToCothRule),
        // Conversion from exponential forms
        Rc::new(SinhFromExpRule),
        Rc::new(CoshFromExpRule),
        Rc::new(TanhFromExpRule),
        Rc::new(SechFromExpRule),
        Rc::new(CschFromExpRule),
        Rc::new(CothFromExpRule),
        Rc::new(HyperbolicTripleAngleRule),
    ]
}
