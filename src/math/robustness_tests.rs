#[cfg(test)]
mod tests {
    use crate::math::*;

    #[test]
    fn test_zeta_deriv_robustness() {
        // n=2, x=2.0 (series convergence test)
        // Check it doesn't return None or panic
        let deriv2 = eval_zeta_deriv(2, 2.0_f64);
        assert!(deriv2.is_some());

        let val = deriv2.unwrap();
        // ζ''(2) > 0
        assert!(val > 0.0);

        // n=2, x=0.5 (should return None now, avoiding bad numerical diff)
        let deriv2_bad = eval_zeta_deriv(2, 0.5_f64);
        assert!(deriv2_bad.is_none());
    }
}
