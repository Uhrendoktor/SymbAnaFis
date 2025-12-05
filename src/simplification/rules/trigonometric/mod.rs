use crate::simplification::rules::Rule;
use std::rc::Rc;

pub(crate) mod angles;
/// Trigonometric simplification rules
pub(crate) mod basic;
pub(crate) mod identities;
pub(crate) mod inverse;
pub(crate) mod transformations;
pub(crate) mod triple_angle;

/// Get all trigonometric rules in priority order
pub(crate) fn get_trigonometric_rules() -> Vec<Rc<dyn Rule>> {
    vec![
        // Basic rules: special values and constants
        Rc::new(basic::SinZeroRule),
        Rc::new(basic::CosZeroRule),
        Rc::new(basic::TanZeroRule),
        Rc::new(basic::SinPiRule),
        Rc::new(basic::CosPiRule),
        Rc::new(basic::SinPiOverTwoRule),
        Rc::new(basic::CosPiOverTwoRule),
        Rc::new(basic::TrigExactValuesRule),
        // Pythagorean and complementary identities
        Rc::new(identities::PythagoreanIdentityRule),
        Rc::new(identities::PythagoreanComplementsRule),
        Rc::new(identities::PythagoreanTangentRule),
        // Inverse trig functions
        Rc::new(inverse::InverseTrigIdentityRule),
        Rc::new(inverse::InverseTrigCompositionRule),
        // Cofunction, periodicity, reflection, and negation
        Rc::new(transformations::CofunctionIdentityRule),
        Rc::new(transformations::TrigPeriodicityRule),
        Rc::new(transformations::TrigReflectionRule),
        Rc::new(transformations::TrigThreePiOverTwoRule),
        Rc::new(transformations::TrigNegArgRule),
        // Angle-based: double angle, sum/difference, product-to-sum
        Rc::new(angles::TrigDoubleAngleRule),
        Rc::new(angles::CosDoubleAngleDifferenceRule),
        Rc::new(angles::TrigSumDifferenceRule),
        Rc::new(angles::TrigProductToDoubleAngleRule),
        // Triple angle formulas
        Rc::new(triple_angle::TrigTripleAngleRule),
    ]
}
