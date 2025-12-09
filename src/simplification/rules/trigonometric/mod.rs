use crate::simplification::rules::Rule;
use std::sync::Arc;

pub(crate) mod angles;
/// Trigonometric simplification rules
pub(crate) mod basic;
pub(crate) mod identities;
pub(crate) mod inverse;
pub(crate) mod transformations;
pub(crate) mod triple_angle;

/// Get all trigonometric rules in priority order
pub(crate) fn get_trigonometric_rules() -> Vec<Arc<dyn Rule + Send + Sync>> {
    vec![
        // Basic rules: special values and constants
        Arc::new(basic::SinZeroRule),
        Arc::new(basic::CosZeroRule),
        Arc::new(basic::TanZeroRule),
        Arc::new(basic::SinPiRule),
        Arc::new(basic::CosPiRule),
        Arc::new(basic::SinPiOverTwoRule),
        Arc::new(basic::CosPiOverTwoRule),
        Arc::new(basic::TrigExactValuesRule),
        // Pythagorean and complementary identities
        Arc::new(identities::PythagoreanIdentityRule),
        Arc::new(identities::PythagoreanComplementsRule),
        Arc::new(identities::PythagoreanTangentRule),
        // Inverse trig functions
        Arc::new(inverse::InverseTrigIdentityRule),
        Arc::new(inverse::InverseTrigCompositionRule),
        // Cofunction, periodicity, reflection, and negation
        Arc::new(transformations::CofunctionIdentityRule),
        Arc::new(transformations::TrigPeriodicityRule),
        Arc::new(transformations::TrigReflectionRule),
        Arc::new(transformations::TrigThreePiOverTwoRule),
        Arc::new(transformations::TrigNegArgRule),
        // Angle-based: double angle, sum/difference, product-to-sum
        Arc::new(angles::TrigDoubleAngleRule),
        Arc::new(angles::CosDoubleAngleDifferenceRule),
        Arc::new(angles::TrigSumDifferenceRule),
        Arc::new(angles::TrigProductToDoubleAngleRule),
        // Triple angle formulas
        Arc::new(triple_angle::TrigTripleAngleRule),
    ]
}
