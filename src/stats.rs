//! Scalar statistics helpers replacing `scipy.stats.norm` in the reference
//! implementation.
//!
//! The dblock sigma schedules are defined in terms of the standard normal
//! CDF `Phi` and its inverse (the quantile function). These are implemented
//! here with f64 precision:
//!
//! - [`erf`]: Maclaurin series (|x| < 3), accurate to ~1e-14.
//! - [`erfc`]: [`erf`] for |x| < 3, otherwise a truncated asymptotic
//!   expansion `exp(-x^2)/(x*sqrt(pi)) * (1 - 1/(2x^2) + ...)`, accurate to
//!   better than 1e-11 relative down to deep tails.
//! - [`norm_cdf`]: `Phi(x) = erfc(-x/sqrt(2)) / 2`.
//! - [`norm_ppf`]: Acklam's rational approximation refined by Newton steps,
//!   ~1e-15 relative accuracy.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, PI};

/// Error function `erf(x)` with ~1e-14 absolute accuracy.
pub fn erf(x: f64) -> f64 {
    let ax = x.abs();
    if ax >= 3.0 {
        // erf is within 1e-4 of +/-1; defer to erfc for the interesting part.
        return if x >= 0.0 { 1.0 - erfc(ax) } else { erfc(ax) - 1.0 };
    }
    // Maclaurin series: erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1))
    let xx = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n: u32 = 1;
    while n < 60 {
        term *= -xx / n as f64;
        let contrib = term / (2 * n + 1) as f64;
        sum += contrib;
        if contrib.abs() <= sum.abs() * 1e-17 {
            break;
        }
        n += 1;
    }
    sum * FRAC_2_SQRT_PI
}
/// Complementary error function `erfc(x)` with ~1e-15 relative accuracy.
pub fn erfc(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 2.0 {
        return 1.0 - erf(x);
    }
    if ax >= 27.3 {
        // exp(-ax^2) underflows; erfc is 0 or 2.
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    // Continued fraction (Abramowitz & Stegun 7.1.14), evaluated with the
    // modified Lentz algorithm:
    //   erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x+ (1/2)/(x+ (2/2)/(x+ ...)))
    const TINY: f64 = 1e-300;
    let mut f = ax; // b0 = x
    let mut c = f;
    let mut d = 0.0f64;
    let mut k: u32 = 1;
    while k <= 300 {
        let a = 0.5 * k as f64;
        d = ax + a * d;
        if d == 0.0 {
            d = TINY;
        }
        c = ax + a / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= 1e-17 {
            break;
        }
        k += 1;
    }
    let r = (-ax * ax).exp() / (PI.sqrt() * f);
    if x >= 0.0 {
        r
    } else {
        2.0 - r
    }
}

/// Standard normal cumulative distribution function `Phi(x)`.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Inverse of the standard normal CDF (probit function).
///
/// Acklam's rational approximation followed by Newton refinements,
/// giving ~1e-15 relative accuracy.
pub fn norm_ppf(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "norm_ppf requires 0 < p < 1, got {p}");

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let mut x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // Newton refinement: solve Phi(x) = p.
    for _ in 0..2 {
        let e = norm_cdf(x) - p;
        if e == 0.0 {
            break;
        }
        let pdf = (-0.5 * x * x).exp() / SQRT_2PI;
        let dx = -e / pdf;
        x += dx;
        if dx.abs() <= 1e-16 * x.abs() {
            break;
        }
    }
    x
}

const SQRT_2PI: f64 = 2.5066282746310002; // sqrt(2 pi)

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_erf_known_values() {
        // scipy.special.erf references
        assert_relative_eq!(erf(0.0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(erf(0.5), 0.5204998778130465, epsilon = 1e-14);
        assert_relative_eq!(erf(1.0), 0.8427007929497149, epsilon = 1e-14);
        assert_relative_eq!(erf(2.0), 0.9953222650189527, epsilon = 1e-14);
        assert_relative_eq!(erf(2.5), 0.999593047982555, epsilon = 1e-13);
        assert_relative_eq!(erf(-1.5), -0.9661051464753108, epsilon = 1e-14);
    }

    #[test]
    fn test_erfc_known_values() {
        // scipy.special.erfc references
        assert_relative_eq!(erfc(0.0), 1.0, epsilon = 1e-14);
        assert_relative_eq!(erfc(1.0), 0.15729920705028513, epsilon = 1e-14);
        assert_relative_eq!(erfc(2.0), 0.004677734981063127, epsilon = 1e-13);
        assert_relative_eq!(erfc(3.0), 2.209049699858544e-05, max_relative = 1e-11);
        assert_relative_eq!(erfc(4.0), 1.541725790028002e-08, max_relative = 1e-10);
        assert_relative_eq!(erfc(5.0), 1.537459794428035e-12, max_relative = 1e-9);
        assert_relative_eq!(erfc(-2.0), 1.9953222650189537, epsilon = 1e-13);
        assert_relative_eq!(erfc(10.0), 2.088487583762545e-45, max_relative = 1e-8);
    }

    #[test]
    fn test_norm_cdf_known_values() {
        assert_relative_eq!(norm_cdf(0.0), 0.5, epsilon = 1e-15);
        // scipy.stats.norm.cdf(1.96) = 0.9750021048517795
        assert_relative_eq!(norm_cdf(1.96), 0.9750021048517795, epsilon = 1e-14);
        // scipy.stats.norm.cdf(2.5) = 0.9937903346742238
        assert_relative_eq!(norm_cdf(2.5), 0.9937903346742238, epsilon = 1e-14);
        // scipy.stats.norm.cdf(-3.0) = 0.0013498980316301035
        assert_relative_eq!(norm_cdf(-3.0), 0.0013498980316301035, max_relative = 1e-11);
        // scipy.stats.norm.cdf(-4.6517) = 1.6460488695804655e-06 (sigma_max endpoint)
        assert_relative_eq!(norm_cdf(-4.6517), 1.6460488695804655e-06, max_relative = 1e-9);
    }

    #[test]
    fn test_norm_ppf_known_values() {
        assert_relative_eq!(norm_ppf(0.5), 0.0, epsilon = 1e-12);
        // scipy.stats.norm.ppf(0.975) = 1.959963984540054
        assert_relative_eq!(norm_ppf(0.975), 1.959963984540054, max_relative = 1e-13);
        assert_relative_eq!(norm_ppf(0.025), -1.959963984540054, max_relative = 1e-13);
        // scipy.stats.norm.ppf(0.999998) = 4.6113823623082295
        assert_relative_eq!(norm_ppf(0.999998), 4.6113823623082295, max_relative = 1e-11);
        // Deep tail: scipy.stats.norm.ppf(1.457e-5) ~ -4.1787 (used by block sigmas)
        let x = norm_ppf(1.457e-5);
        assert_relative_eq!(norm_cdf(x), 1.457e-5, max_relative = 1e-9);
    }

    #[test]
    fn ppf_roundtrip() {
        for p in [1e-7, 1e-5, 0.001, 0.1, 0.3, 0.5, 0.77, 0.99999, 1.0 - 1e-7] {
            let x = norm_ppf(p);
            let back = norm_cdf(x);
            assert_relative_eq!(back, p, max_relative = 1e-9);
        }
    }
}
