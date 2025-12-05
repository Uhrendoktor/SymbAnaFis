use crate::simplification::rules::Rule;
use std::rc::Rc;

pub(crate) mod abs_sign;
pub(crate) mod canonicalization;
pub(crate) mod combination;
pub(crate) mod expansion;
pub(crate) mod factoring;
pub(crate) mod fractions;
/// Algebraic simplification rules
pub(crate) mod identities;
pub(crate) mod powers;

/// Get all algebraic rules in priority order
pub(crate) fn get_algebraic_rules() -> Vec<Rc<dyn Rule>> {
    vec![
        // Exponential/logarithmic identities
        Rc::new(identities::ExpLnRule),
        Rc::new(identities::LnExpRule),
        Rc::new(identities::ExpMulLnRule),
        Rc::new(identities::EPowLnRule),
        Rc::new(identities::EPowMulLnRule),
        // Power rules
        Rc::new(powers::PowerZeroRule),
        Rc::new(powers::PowerOneRule),
        Rc::new(powers::PowerPowerRule),
        Rc::new(powers::PowerMulRule),
        Rc::new(powers::PowerDivRule),
        Rc::new(powers::PowerCollectionRule),
        Rc::new(powers::CommonExponentDivRule),
        Rc::new(powers::CommonExponentMulRule),
        Rc::new(powers::NegativeExponentToFractionRule),
        Rc::new(powers::PowerOfQuotientRule), // (a/b)^n -> a^n / b^n
        // Fraction rules
        Rc::new(fractions::DivSelfRule),
        Rc::new(fractions::DivDivRule),
        Rc::new(fractions::CombineNestedFractionRule),
        Rc::new(fractions::AddFractionRule),
        Rc::new(fractions::FractionToEndRule),
        // Absolute value and sign rules
        Rc::new(abs_sign::AbsNumericRule),
        Rc::new(abs_sign::SignNumericRule),
        Rc::new(abs_sign::AbsAbsRule),
        Rc::new(abs_sign::AbsNegRule),
        Rc::new(abs_sign::AbsSquareRule),
        Rc::new(abs_sign::AbsPowEvenRule),
        Rc::new(abs_sign::SignSignRule),
        Rc::new(abs_sign::SignAbsRule),
        Rc::new(abs_sign::AbsSignMulRule),
        // Expansion rules
        Rc::new(expansion::ExpandPowerForCancellationRule),
        Rc::new(expansion::PowerExpansionRule),
        Rc::new(expansion::PolynomialExpansionRule),
        Rc::new(expansion::ExpandDifferenceOfSquaresProductRule),
        // Factoring rules
        Rc::new(factoring::FractionCancellationRule),
        Rc::new(factoring::PerfectSquareRule),
        Rc::new(factoring::FactorDifferenceOfSquaresRule),
        Rc::new(factoring::PerfectCubeRule),
        Rc::new(factoring::NumericGcdFactoringRule),
        Rc::new(factoring::CommonTermFactoringRule),
        Rc::new(factoring::CommonPowerFactoringRule),
        // Canonicalization rules
        Rc::new(canonicalization::CanonicalizeRule),
        Rc::new(canonicalization::CanonicalizeMultiplicationRule),
        Rc::new(canonicalization::CanonicalizeAdditionRule),
        Rc::new(canonicalization::CanonicalizeSubtractionRule),
        Rc::new(canonicalization::NormalizeAddNegationRule),
        Rc::new(canonicalization::SimplifyNegativeOneRule),
        // Combination rules
        Rc::new(combination::MulDivCombinationRule),
        Rc::new(combination::CombineTermsRule),
        Rc::new(combination::CombineFactorsRule),
        Rc::new(combination::CombineLikeTermsInAdditionRule),
        Rc::new(combination::DistributeNegationRule),
        // DistributeMulInNumeratorRule removed - conflicts with MulDivCombinationRule
    ]
}
