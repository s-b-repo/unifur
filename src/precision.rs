//! Reduced-precision arithmetic for denoising (roadmap 10.6 and 15.1).
//!
//! The `ndarray` CPU backend computes in `f32` only, so a native `bf16` /
//! `f16` tensor type is not available. What *is* available -- and what the
//! roadmap items actually need -- is the ability to answer "how much accuracy
//! does this trajectory lose at reduced precision, and where can it be spent
//! safely?". This module provides that by **emulating** the reduced formats:
//! values are rounded onto the target format's representable grid (correct
//! round-to-nearest-even, correct subnormal and overflow behaviour) while the
//! arithmetic itself stays in `f32`.
//!
//! Emulation is not the same as native low-precision execution: it models the
//! *representation* error exactly but not the accumulation order of a real
//! bf16 kernel, and it is slower rather than faster. It is therefore a
//! numerical-analysis tool, not a speedup -- but it is a faithful one, and
//! [`Precision::round_scalar`] is the single place a native cast would be
//! swapped in once a backend offers one.
//!
//! [`PrecisionPolicy`] expresses the actual Phase-10.6 idea: high-sigma
//! windows tolerate coarse precision (the signal is dominated by noise
//! anyway) while low-sigma windows, where the estimate is nearly converged,
//! keep full precision.

use burn::tensor::{backend::Backend, Tensor, TensorData};

/// A floating-point format, identified by the properties that determine its
/// rounding behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// IEEE binary32: 24 significand bits.
    #[default]
    F32,
    /// bfloat16: 8 significand bits, `f32`'s exponent range.
    Bf16,
    /// IEEE binary16: 11 significand bits, exponent range +/-15.
    F16,
}

impl Precision {
    /// Parse a CLI precision name.
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "f32" | "fp32" => Ok(Self::F32),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            other => anyhow::bail!("unknown precision '{other}' (expected f32|bf16|f16)"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }

    /// Significand bits *including* the implicit leading one.
    pub fn significand_bits(&self) -> u32 {
        match self {
            Self::F32 => 24,
            Self::Bf16 => 8,
            Self::F16 => 11,
        }
    }

    /// Unit roundoff `u = 2^-p`: the tight bound on the relative error of
    /// round-to-nearest for any value in the format's normal range.
    pub fn unit_roundoff(&self) -> f32 {
        // 2^-p, computed exactly via the exponent field.
        (2.0f64).powi(-(self.significand_bits() as i32)) as f32
    }

    /// Smallest positive subnormal.
    pub fn min_subnormal(&self) -> f32 {
        match self {
            Self::F32 => f32::from_bits(1),
            // bf16 shares f32's exponent range; its smallest subnormal is
            // 2^-126 * 2^-7 = 2^-133.
            Self::Bf16 => (2.0f64).powi(-133) as f32,
            Self::F16 => (2.0f64).powi(-24) as f32,
        }
    }

    /// Largest finite magnitude.
    pub fn max_finite(&self) -> f32 {
        match self {
            Self::F32 => f32::MAX,
            // (2 - 2^-7) * 2^127
            Self::Bf16 => ((2.0f64 - (2.0f64).powi(-7)) * (2.0f64).powi(127)) as f32,
            Self::F16 => 65504.0,
        }
    }

    /// Round `x` onto this format's representable grid (round-to-nearest,
    /// ties to even), with correct subnormal flush and overflow to infinity.
    ///
    /// `Precision::F32` is the identity: the values already live in `f32`.
    pub fn round_scalar(&self, x: f32) -> f32 {
        if *self == Self::F32 || !x.is_finite() || x == 0.0 {
            return x;
        }

        let sign = if x.is_sign_negative() { -1.0f32 } else { 1.0 };
        let a = x.abs() as f64;
        let quantum = self.min_subnormal() as f64;
        let p = self.significand_bits();

        // Subnormal range: the grid is a uniform lattice of `quantum`.
        let smallest_normal = quantum * (2.0f64).powi(p as i32 - 1);
        let rounded = if a < smallest_normal {
            round_ties_even(a / quantum) * quantum
        } else {
            // Normal range: keep `p` significand bits, i.e. round to a
            // multiple of 2^(exponent - p + 1).
            let exponent = a.log2().floor() as i32;
            let scale = (2.0f64).powi(exponent - p as i32 + 1);
            let mut r = round_ties_even(a / scale) * scale;
            // Rounding up can carry into the next binade; that value is still
            // representable, so only overflow needs handling.
            if r > self.max_finite() as f64 {
                r = f64::INFINITY;
            }
            r
        };
        sign * rounded as f32
    }

    /// Round every element of a tensor (see [`Self::round_scalar`]).
    ///
    /// Round-trips through host memory because no backend exposes bitwise
    /// float manipulation; `Precision::F32` short-circuits so the common path
    /// costs nothing.
    pub fn round<B: Backend<FloatElem = f32>, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        if *self == Self::F32 {
            return tensor;
        }
        let device = tensor.device();
        let dims = tensor.dims();
        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .map(|v| self.round_scalar(v))
            .collect();
        Tensor::<B, 1>::from_data(TensorData::new(values, [dims.iter().product::<usize>()]), &device)
            .reshape(dims)
    }
}

/// Round half to even, matching IEEE-754's default mode.
fn round_ties_even(v: f64) -> f64 {
    let floor = v.floor();
    let frac = v - floor;
    match frac.partial_cmp(&0.5) {
        Some(std::cmp::Ordering::Less) => floor,
        Some(std::cmp::Ordering::Greater) => floor + 1.0,
        // Exactly halfway: pick the even neighbour.
        _ => {
            if (floor as i64) % 2 == 0 {
                floor
            } else {
                floor + 1.0
            }
        }
    }
}

/// Which precision to use at which point of the trajectory (roadmap 10.6).
///
/// At high sigma the latent is dominated by noise of magnitude `sigma`, so a
/// relative representation error of `2^-8` is far below the noise floor. Near
/// `sigma_min` the estimate is nearly converged and the same relative error is
/// no longer negligible, so the policy switches back to full precision.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionPolicy {
    /// Used while `sigma >= switch_sigma`.
    pub high_noise: Precision,
    /// Used while `sigma < switch_sigma`.
    pub low_noise: Precision,
    pub switch_sigma: f64,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::full(Precision::F32)
    }
}

impl PrecisionPolicy {
    /// One precision everywhere.
    pub fn full(precision: Precision) -> Self {
        Self { high_noise: precision, low_noise: precision, switch_sigma: 0.0 }
    }

    /// `coarse` above `switch_sigma`, `f32` below it.
    pub fn mixed(coarse: Precision, switch_sigma: f64) -> Self {
        Self { high_noise: coarse, low_noise: Precision::F32, switch_sigma }
    }

    /// Precision governing a window at noise level `sigma`.
    pub fn for_sigma(&self, sigma: f64) -> Precision {
        if sigma >= self.switch_sigma {
            self.high_noise
        } else {
            self.low_noise
        }
    }

    /// Whether the policy ever leaves `f32` (lets callers skip the rounding
    /// path entirely).
    pub fn is_full_precision(&self) -> bool {
        self.high_noise == Precision::F32 && self.low_noise == Precision::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_f32_is_the_identity() {
        for x in [0.0f32, 1.0, -3.25, 1e-30, 1e30, f32::MIN_POSITIVE] {
            assert_eq!(Precision::F32.round_scalar(x), x);
        }
    }

    #[test]
    fn test_relative_error_is_bounded_by_the_unit_roundoff() {
        // The defining property of round-to-nearest at p significand bits:
        // |round(x) - x| <= 2^-p |x| for every x in the normal range. This is
        // the bound every downstream error estimate rests on, so it is checked
        // over a wide dynamic range rather than at a few hand-picked values.
        for precision in [Precision::Bf16, Precision::F16] {
            let u = precision.unit_roundoff();
            let max_exp: i32 = match precision {
                Precision::F16 => 14,
                _ => 100,
            };
            let mut checked = 0usize;
            for exp in -max_exp..=max_exp {
                for mantissa in 0..64 {
                    let x = (1.0f64 + mantissa as f64 / 64.0) * (2.0f64).powi(exp);
                    for signed in [x as f32, -(x as f32)] {
                        let r = precision.round_scalar(signed);
                        assert!(r.is_finite(), "{} lost {signed}", precision.name());
                        let rel = ((r - signed) / signed).abs();
                        assert!(
                            rel <= u * (1.0 + 1e-6),
                            "{}: rel error {rel} > u = {u} at x = {signed}",
                            precision.name()
                        );
                        checked += 1;
                    }
                }
            }
            assert!(checked > 1000, "coverage sanity");
        }
    }

    #[test]
    fn test_rounding_is_idempotent_and_monotone() {
        // Idempotence proves the output really is on the target grid;
        // monotonicity proves the rounding never reorders values, which is
        // what lets a gate threshold keep its meaning under emulation.
        for precision in [Precision::Bf16, Precision::F16] {
            let samples: Vec<f32> = (-200..200)
                .map(|i| i as f32 * 0.037)
                .chain((-20..20).map(|i| (i as f32) * 1e-5))
                .collect();
            let mut prev = f32::NEG_INFINITY;
            let mut sorted = samples.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for x in sorted {
                let r = precision.round_scalar(x);
                assert_eq!(precision.round_scalar(r), r, "{} not idempotent at {x}", precision.name());
                assert!(r >= prev, "{} not monotone at {x}", precision.name());
                prev = r;
            }
        }
    }

    #[test]
    fn test_bf16_matches_bit_truncation_with_round_to_nearest_even() {
        // Independent implementation: bf16 is exactly the top 16 bits of an
        // f32 with an RNE bias added, so the analytic path above must agree
        // with the bit-level one.
        fn bf16_by_bits(x: f32) -> f32 {
            let bits = x.to_bits();
            let bias = 0x7fff + ((bits >> 16) & 1);
            f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
        }
        for i in 0..5000 {
            let x = (i as f32 - 2500.0) * 0.0137;
            let a = Precision::Bf16.round_scalar(x);
            let b = bf16_by_bits(x);
            assert_eq!(a, b, "bf16 mismatch at {x}: {a} vs {b}");
        }
    }

    #[test]
    fn test_f16_range_limits() {
        assert_eq!(Precision::F16.round_scalar(65504.0), 65504.0);
        assert!(Precision::F16.round_scalar(70000.0).is_infinite(), "f16 must overflow");
        // Below half the smallest subnormal everything flushes to zero,
        // keeping the sign.
        assert_eq!(Precision::F16.round_scalar(1e-9), 0.0);
        assert_eq!(Precision::F16.round_scalar(-1e-9), -0.0);
        // ...and the smallest subnormal itself survives.
        let tiny = Precision::F16.min_subnormal();
        assert_eq!(Precision::F16.round_scalar(tiny), tiny);
        // bf16 keeps f32's exponent range, so the same value is finite there.
        assert!(Precision::Bf16.round_scalar(70000.0).is_finite());
    }

    #[test]
    fn test_special_values_pass_through() {
        for precision in [Precision::Bf16, Precision::F16] {
            assert!(precision.round_scalar(f32::NAN).is_nan());
            assert_eq!(precision.round_scalar(f32::INFINITY), f32::INFINITY);
            assert_eq!(precision.round_scalar(0.0), 0.0);
        }
    }

    #[test]
    fn test_tensor_rounding_matches_scalar() {
        let device = Default::default();
        let values: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.317).collect();
        let t = Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([4, 6]);
        let rounded: Vec<f32> = Precision::Bf16
            .round(t)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let expected: Vec<f32> = values.iter().map(|&v| Precision::Bf16.round_scalar(v)).collect();
        assert_eq!(rounded, expected);
    }

    #[test]
    fn test_policy_switches_at_the_threshold() {
        let policy = PrecisionPolicy::mixed(Precision::Bf16, 1.0);
        assert_eq!(policy.for_sigma(80.0), Precision::Bf16);
        assert_eq!(policy.for_sigma(1.0), Precision::Bf16, "threshold is inclusive");
        assert_eq!(policy.for_sigma(0.99), Precision::F32);
        assert!(!policy.is_full_precision());
        assert!(PrecisionPolicy::default().is_full_precision());
        assert!(PrecisionPolicy::full(Precision::F32).is_full_precision());
    }

    #[test]
    fn test_precision_parse_roundtrip() {
        for name in ["f32", "bf16", "f16"] {
            assert_eq!(Precision::parse(name).unwrap().name(), name);
        }
        assert_eq!(Precision::parse("fp32").unwrap(), Precision::F32);
        assert!(Precision::parse("int8").is_err());
    }
}
