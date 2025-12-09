use crate::traits::MathScalar;
use num_traits::{Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Dual<T: MathScalar> {
    pub val: T,
    pub eps: T,
}

impl<T: MathScalar> Dual<T> {
    pub fn new(val: T, eps: T) -> Self {
        Self { val, eps }
    }

    pub fn constant(val: T) -> Self {
        Self {
            val,
            eps: T::zero(),
        }
    }
}

impl<T: MathScalar> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.val, self.eps)
    }
}

// Basic Arithmetic

impl<T: MathScalar> Add for Dual<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.val + rhs.val, self.eps + rhs.eps)
    }
}

impl<T: MathScalar> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.val - rhs.val, self.eps - rhs.eps)
    }
}

impl<T: MathScalar> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // Product rule
        Self::new(self.val * rhs.val, self.val * rhs.eps + self.eps * rhs.val)
    }
}

impl<T: MathScalar> Div for Dual<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // Quotient rule
        let val = self.val / rhs.val;
        let eps = (self.eps * rhs.val - self.val * rhs.eps) / (rhs.val * rhs.val);
        Self::new(val, eps)
    }
}

impl<T: MathScalar> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.val, -self.eps)
    }
}

impl<T: MathScalar> Rem for Dual<T> {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        // IMPORTANT: The remainder operation is NOT differentiable!
        // It's discontinuous at integer multiples of the divisor.
        // The derivative is technically 0 almost everywhere, but undefined at jumps.
        // For AD purposes, we set eps = 0 to indicate non-differentiability.
        Self::new(self.val % rhs.val, T::zero())
    }
}

// Assignments

impl<T: MathScalar> AddAssign for Dual<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
        self.eps += rhs.eps;
    }
}

impl<T: MathScalar> SubAssign for Dual<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.val -= rhs.val;
        self.eps -= rhs.eps;
    }
}

impl<T: MathScalar> MulAssign for Dual<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: MathScalar> DivAssign for Dual<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: MathScalar> RemAssign for Dual<T> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// Traits for MathScalar

impl<T: MathScalar> Zero for Dual<T> {
    fn zero() -> Self {
        Self::constant(T::zero())
    }
    fn is_zero(&self) -> bool {
        self.val.is_zero() && self.eps.is_zero()
    }
}

impl<T: MathScalar> One for Dual<T> {
    fn one() -> Self {
        Self::constant(T::one())
    }
}

impl<T: MathScalar> Num for Dual<T> {
    type FromStrRadixErr = T::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::constant(T::from_str_radix(str, radix)?))
    }
}

impl<T: MathScalar> ToPrimitive for Dual<T> {
    fn to_i64(&self) -> Option<i64> {
        self.val.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.val.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        self.val.to_f64()
    }
}

impl<T: MathScalar> FromPrimitive for Dual<T> {
    fn from_i64(n: i64) -> Option<Self> {
        T::from_i64(n).map(Self::constant)
    }
    fn from_u64(n: u64) -> Option<Self> {
        T::from_u64(n).map(Self::constant)
    }
    fn from_f64(n: f64) -> Option<Self> {
        T::from_f64(n).map(Self::constant)
    }
}

impl<T: MathScalar> NumCast for Dual<T> {
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        T::from(n).map(Self::constant)
    }
}

// Signed trait implementation using the correct abs from Float trait
impl<T: MathScalar> num_traits::Signed for Dual<T> {
    fn abs(&self) -> Self {
        // Use Float::abs which correctly implements d/dx|x| = sign(x)
        Float::abs(*self)
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Float::abs_sub(*self, *other)
    }

    fn signum(&self) -> Self {
        Float::signum(*self)
    }

    fn is_positive(&self) -> bool {
        self.val > T::zero()
    }

    fn is_negative(&self) -> bool {
        self.val < T::zero()
    }
}

impl<T: MathScalar> Bounded for Dual<T> {
    fn min_value() -> Self {
        Self::constant(T::min_value())
    }
    fn max_value() -> Self {
        Self::constant(T::max_value())
    }
}

impl<T: MathScalar> FloatConst for Dual<T> {
    fn E() -> Self {
        Self::constant(T::E())
    }
    fn FRAC_1_PI() -> Self {
        Self::constant(T::FRAC_1_PI())
    }
    fn FRAC_1_SQRT_2() -> Self {
        Self::constant(T::FRAC_1_SQRT_2())
    }
    fn FRAC_2_PI() -> Self {
        Self::constant(T::FRAC_2_PI())
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Self::constant(T::FRAC_2_SQRT_PI())
    }
    fn FRAC_PI_2() -> Self {
        Self::constant(T::FRAC_PI_2())
    }
    fn FRAC_PI_3() -> Self {
        Self::constant(T::FRAC_PI_3())
    }
    fn FRAC_PI_4() -> Self {
        Self::constant(T::FRAC_PI_4())
    }
    fn FRAC_PI_6() -> Self {
        Self::constant(T::FRAC_PI_6())
    }
    fn FRAC_PI_8() -> Self {
        Self::constant(T::FRAC_PI_8())
    }
    fn LN_10() -> Self {
        Self::constant(T::LN_10())
    }
    fn LN_2() -> Self {
        Self::constant(T::LN_2())
    }
    fn LOG10_2() -> Self {
        Self::constant(T::LOG10_2())
    }
    fn LOG10_E() -> Self {
        Self::constant(T::LOG10_E())
    }
    fn LOG2_10() -> Self {
        Self::constant(T::LOG2_10())
    }
    fn LOG2_E() -> Self {
        Self::constant(T::LOG2_E())
    }
    fn PI() -> Self {
        Self::constant(T::PI())
    }
    fn SQRT_2() -> Self {
        Self::constant(T::SQRT_2())
    }
}

impl<T: MathScalar + Float> Float for Dual<T> {
    fn nan() -> Self {
        Self::constant(T::nan())
    }
    fn infinity() -> Self {
        Self::constant(T::infinity())
    }
    fn neg_infinity() -> Self {
        Self::constant(T::neg_infinity())
    }
    fn neg_zero() -> Self {
        Self::constant(T::neg_zero())
    }
    fn min_value() -> Self {
        Self::constant(T::min_value())
    }
    fn max_value() -> Self {
        Self::constant(T::max_value())
    }
    fn min_positive_value() -> Self {
        Self::constant(T::min_positive_value())
    }
    fn is_nan(self) -> bool {
        self.val.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.val.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.val.is_finite()
    }
    fn is_normal(self) -> bool {
        self.val.is_normal()
    }
    fn classify(self) -> std::num::FpCategory {
        self.val.classify()
    }

    fn floor(self) -> Self {
        Self::constant(self.val.floor())
    }
    fn ceil(self) -> Self {
        Self::constant(self.val.ceil())
    }
    fn round(self) -> Self {
        Self::constant(self.val.round())
    }
    fn trunc(self) -> Self {
        Self::constant(self.val.trunc())
    }
    fn fract(self) -> Self {
        Self::new(self.val.fract(), self.eps)
    }

    fn abs(self) -> Self {
        let sign = if self.val >= T::zero() {
            T::one()
        } else {
            -T::one()
        };
        Self::new(self.val.abs(), self.eps * sign)
    }

    fn signum(self) -> Self {
        Self::constant(self.val.signum())
    }

    fn is_sign_positive(self) -> bool {
        self.val.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.val.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        // self * a + b
        self * a + b
    }

    fn recip(self) -> Self {
        Self::one() / self
    }

    fn powi(self, n: i32) -> Self {
        let n_t = T::from(n).unwrap();
        let val_pow = self.val.powi(n);
        let val_pow_minus_1 = self.val.powi(n - 1);
        Self::new(val_pow, self.eps * n_t * val_pow_minus_1)
    }

    fn powf(self, n: Self) -> Self {
        // x^y = exp(y * ln(x))
        (n * self.ln()).exp()
    }

    fn sqrt(self) -> Self {
        let sqrt_val = self.val.sqrt();
        Self::new(sqrt_val, self.eps / (T::from(2.0).unwrap() * sqrt_val))
    }

    fn exp(self) -> Self {
        let exp_val = self.val.exp();
        Self::new(exp_val, self.eps * exp_val)
    }

    fn exp2(self) -> Self {
        let ln2 = T::LN_2();
        let exp2_val = self.val.exp2();
        Self::new(exp2_val, self.eps * exp2_val * ln2)
    }

    fn ln(self) -> Self {
        Self::new(self.val.ln(), self.eps / self.val)
    }

    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }

    fn log2(self) -> Self {
        self.ln() / Self::constant(T::LN_2())
    }

    fn log10(self) -> Self {
        self.ln() / Self::constant(T::LN_10())
    }

    fn max(self, other: Self) -> Self {
        if self.val > other.val { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self.val < other.val { self } else { other }
    }

    fn abs_sub(self, other: Self) -> Self {
        if self.val <= other.val {
            Self::zero()
        } else {
            self - other
        }
    }

    fn cbrt(self) -> Self {
        let val_cbrt = self.val.cbrt();
        let three = T::from(3.0).unwrap();
        // d/dx x^(1/3) = 1/3 x^(-2/3) = 1/(3 * cbrt(x)^2)
        Self::new(val_cbrt, self.eps / (three * val_cbrt * val_cbrt))
    }

    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }

    // Trig
    fn sin(self) -> Self {
        Self::new(self.val.sin(), self.eps * self.val.cos())
    }

    fn cos(self) -> Self {
        Self::new(self.val.cos(), -self.eps * self.val.sin())
    }

    fn tan(self) -> Self {
        let tan_val = self.val.tan();
        let sec2_val = T::one() + tan_val * tan_val;
        Self::new(tan_val, self.eps * sec2_val)
    }

    fn asin(self) -> Self {
        let one = T::one();
        let deriv = one / (one - self.val * self.val).sqrt();
        Self::new(self.val.asin(), self.eps * deriv)
    }

    fn acos(self) -> Self {
        let one = T::one();
        let deriv = -one / (one - self.val * self.val).sqrt();
        Self::new(self.val.acos(), self.eps * deriv)
    }

    fn atan(self) -> Self {
        let one = T::one();
        let deriv = one / (one + self.val * self.val);
        Self::new(self.val.atan(), self.eps * deriv)
    }

    fn atan2(self, other: Self) -> Self {
        // atan(y/x)
        let val = self.val.atan2(other.val);
        let r2 = self.val * self.val + other.val * other.val;
        let eps = (other.val * self.eps - self.val * other.eps) / r2;
        Self::new(val, eps)
    }

    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    fn exp_m1(self) -> Self {
        self.exp() - Self::one()
    }

    fn ln_1p(self) -> Self {
        (self + Self::one()).ln()
    }

    fn sinh(self) -> Self {
        Self::new(self.val.sinh(), self.eps * self.val.cosh())
    }

    fn cosh(self) -> Self {
        Self::new(self.val.cosh(), self.eps * self.val.sinh())
    }

    fn tanh(self) -> Self {
        let tanh_val = self.val.tanh();
        let sech2_val = T::one() - tanh_val * tanh_val;
        Self::new(tanh_val, self.eps * sech2_val)
    }

    fn asinh(self) -> Self {
        let one = T::one();
        let deriv = one / (self.val * self.val + one).sqrt();
        Self::new(self.val.asinh(), self.eps * deriv)
    }

    fn acosh(self) -> Self {
        let one = T::one();
        let deriv = one / (self.val * self.val - one).sqrt();
        Self::new(self.val.acosh(), self.eps * deriv)
    }

    fn atanh(self) -> Self {
        let one = T::one();
        let deriv = one / (one - self.val * self.val);
        Self::new(self.val.atanh(), self.eps * deriv)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.val.integer_decode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_dual_basic_arithmetic() {
        // x = 2, dx = 1 (variable)
        let x = Dual::new(2.0, 1.0);
        // c = 3, dc = 0 (constant)
        let c = Dual::constant(3.0);

        // x + c = 5, derivative = 1
        let sum = x + c;
        assert!(approx_eq(sum.val, 5.0));
        assert!(approx_eq(sum.eps, 1.0));

        // x - c = -1, derivative = 1
        let diff = x - c;
        assert!(approx_eq(diff.val, -1.0));
        assert!(approx_eq(diff.eps, 1.0));

        // x * c = 6, d/dx(3x) = 3
        let prod = x * c;
        assert!(approx_eq(prod.val, 6.0));
        assert!(approx_eq(prod.eps, 3.0));

        // x / c = 2/3, d/dx(x/3) = 1/3
        let quot = x / c;
        assert!(approx_eq(quot.val, 2.0 / 3.0));
        assert!(approx_eq(quot.eps, 1.0 / 3.0));
    }

    #[test]
    fn test_dual_product_rule() {
        // f(x) = x * x = x^2, f'(x) = 2x
        let x = Dual::new(3.0, 1.0);
        let x_squared = x * x;
        assert!(approx_eq(x_squared.val, 9.0));
        assert!(approx_eq(x_squared.eps, 6.0)); // 2 * 3
    }

    #[test]
    fn test_dual_quotient_rule() {
        // f(x) = x / (x + 1), f'(x) = 1/(x+1)^2
        let x = Dual::new(2.0, 1.0);
        let one = Dual::constant(1.0);
        let result = x / (x + one);

        assert!(approx_eq(result.val, 2.0 / 3.0));
        // f'(2) = 1/(3)^2 = 1/9
        assert!(approx_eq(result.eps, 1.0 / 9.0));
    }

    #[test]
    fn test_dual_sin_cos() {
        use std::f64::consts::PI;

        // At x = π/4: sin(π/4) = √2/2, cos(π/4) = √2/2
        let x = Dual::new(PI / 4.0, 1.0);

        let sin_x = x.sin();
        let cos_x = x.cos();

        let sqrt2_2 = 2.0_f64.sqrt() / 2.0;
        assert!(approx_eq(sin_x.val, sqrt2_2));
        assert!(approx_eq(sin_x.eps, sqrt2_2)); // d/dx sin(x) = cos(x)

        assert!(approx_eq(cos_x.val, sqrt2_2));
        assert!(approx_eq(cos_x.eps, -sqrt2_2)); // d/dx cos(x) = -sin(x)
    }

    #[test]
    fn test_dual_exp_ln() {
        // At x = 1: exp(1) = e, d/dx exp(x) = exp(x)
        let x = Dual::new(1.0, 1.0);
        let exp_x = x.exp();

        assert!(approx_eq(exp_x.val, std::f64::consts::E));
        assert!(approx_eq(exp_x.eps, std::f64::consts::E)); // d/dx exp(x) = exp(x)

        // At x = e: ln(e) = 1, d/dx ln(x) = 1/x
        let y = Dual::new(std::f64::consts::E, 1.0);
        let ln_y = y.ln();

        assert!(approx_eq(ln_y.val, 1.0));
        assert!(approx_eq(ln_y.eps, 1.0 / std::f64::consts::E)); // d/dx ln(x) = 1/x
    }

    #[test]
    fn test_dual_sqrt() {
        // At x = 4: sqrt(4) = 2, d/dx sqrt(x) = 1/(2*sqrt(x)) = 1/4
        let x = Dual::new(4.0, 1.0);
        let sqrt_x = x.sqrt();

        assert!(approx_eq(sqrt_x.val, 2.0));
        assert!(approx_eq(sqrt_x.eps, 0.25));
    }

    #[test]
    fn test_dual_powi() {
        // At x = 2: x^3 = 8, d/dx x^3 = 3x^2 = 12
        let x = Dual::new(2.0, 1.0);
        let x_cubed = x.powi(3);

        assert!(approx_eq(x_cubed.val, 8.0));
        assert!(approx_eq(x_cubed.eps, 12.0));
    }

    #[test]
    fn test_dual_chain_rule() {
        // f(x) = sin(x^2), f'(x) = 2x * cos(x^2)
        let x = Dual::new(2.0, 1.0);
        let x_squared = x * x;
        let result = x_squared.sin();

        assert!(approx_eq(result.val, 4.0_f64.sin()));
        // f'(2) = 2*2 * cos(4) = 4 * cos(4)
        assert!(approx_eq(result.eps, 4.0 * 4.0_f64.cos()));
    }

    #[test]
    fn test_dual_tan() {
        use std::f64::consts::PI;

        // At x = π/4: tan(π/4) = 1, d/dx tan(x) = sec^2(x) = 2
        let x = Dual::new(PI / 4.0, 1.0);
        let tan_x = x.tan();

        assert!(approx_eq(tan_x.val, 1.0));
        assert!(approx_eq(tan_x.eps, 2.0)); // sec^2(π/4) = 1/cos^2(π/4) = 2
    }

    #[test]
    fn test_dual_hyperbolic() {
        // At x = 0: sinh(0) = 0, cosh(0) = 1, tanh(0) = 0
        // d/dx sinh(x) = cosh(x) = 1 at x=0
        // d/dx cosh(x) = sinh(x) = 0 at x=0
        // d/dx tanh(x) = sech^2(x) = 1 at x=0
        let x = Dual::new(0.0, 1.0);

        let sinh_x = x.sinh();
        assert!(approx_eq(sinh_x.val, 0.0));
        assert!(approx_eq(sinh_x.eps, 1.0));

        let cosh_x = x.cosh();
        assert!(approx_eq(cosh_x.val, 1.0));
        assert!(approx_eq(cosh_x.eps, 0.0));

        let tanh_x = x.tanh();
        assert!(approx_eq(tanh_x.val, 0.0));
        assert!(approx_eq(tanh_x.eps, 1.0));
    }
}
