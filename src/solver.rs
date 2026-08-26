//! Inference solvers for the dblock probability-flow ODE
//! `dz/dsigma = (x0(z, sigma) - z) / sigma` (roadmap Phase 4).
//!
//! Every solver is expressed through an `x0`-prediction oracle (the trained
//! denoiser projected through the label-embedding table), following the
//! k-diffusion conventions:
//!
//! - [`SolverKind::Euler`]: first-order deterministic step (`diffusion_step`
//!   of the reference implementation).
//! - [`SolverKind::Heun`]: second-order predictor-corrector; costs one extra
//!   model call per step.
//! - [`SolverKind::Ddim`]: DDIM update with ancestral-noise strength `eta`;
//!   at `eta == 0` it coincides exactly with Euler in this parameterization.
//! - [`SolverKind::DpmPlusPlus2M`]: second-order linear multistep in
//!   `lambda = log sigma` space (Lu et al., 2022). No extra model calls; it
//!   reuses the previous step's x0 prediction.

use burn::tensor::{Tensor, backend::Backend};
use rand::Rng;

/// Available ODE solvers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SolverKind {
    #[default]
    Euler,
    Heun,
    /// `eta` controls ancestral noise (0 = deterministic = Euler).
    Ddim { eta: f64 },
    DpmPlusPlus2M,
    /// Third-order linear multistep (roadmap 4.5).
    ///
    /// Coefficients are derived by exactly integrating the exponential
    /// kernel `exp(-(h - s))` against the quadratic interpolant through the
    /// three most recent x0 predictions (nodes at backward offsets `g0` and
    /// `g0 + g1` in lambda space), i.e.
    /// `x+ = e^-h x + a J2 + b J1 + c J0` with `Jk = int_0^h s^k e^-(h-s) ds`
    /// closed forms `J0 = 1 - e^-h`, `J1 = h - 1 + e^-h`,
    /// `J2 = h^2 - 2h + 2 - 2 e^-h`, and interpolation coefficients solved
    /// from the two prior gaps.
    DpmPlusPlus3M,
}

impl SolverKind {
    /// Parse a CLI solver name (`euler`, `heun`, `ddim`, `dpmpp2m`).
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "euler" => Ok(Self::Euler),
            "heun" => Ok(Self::Heun),
            "ddim" => Ok(Self::Ddim { eta: 1.0 }),
            "dpmpp2m" => Ok(Self::DpmPlusPlus2M),
            "dpmpp3m" => Ok(Self::DpmPlusPlus3M),
            other => anyhow::bail!("unknown solver '{other}' (expected euler|heun|ddim|dpmpp2m|dpmpp3m)"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Heun => "heun",
            Self::Ddim { .. } => "ddim",
            Self::DpmPlusPlus2M => "dpmpp2m",
            Self::DpmPlusPlus3M => "dpmpp3m",
        }
    }
}

/// Integrate the probability-flow ODE from `schedule[0]` down to
/// `schedule[len-1]` starting from latent `initial`.
///
/// `predictor(sigma, &z)` must return the clean-data estimate at the current
/// latent. The schedule must be strictly descending. Returns the latent at
/// the last schedule point; callers typically apply one final denoise.
pub fn integrate<B, F>(
    initial: Tensor<B, 2>,
    schedule: &[f64],
    mut predictor: F,
    kind: SolverKind,
    _rng: &mut impl Rng,
) -> Tensor<B, 2>
where
    B: Backend,
    F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
{
    assert!(schedule.len() >= 2, "need at least two schedule points");
    assert!(
        schedule.windows(2).all(|w| w[0] > w[1]),
        "schedule must be strictly descending"
    );

    match kind {
        SolverKind::Euler => euler(initial, schedule, &mut predictor, None),
        SolverKind::Ddim { eta } => euler(initial, schedule, &mut predictor, Some(eta)),
        SolverKind::Heun => heun(initial, schedule, &mut predictor),
        SolverKind::DpmPlusPlus2M => dpmpp_2m(initial, schedule, &mut predictor),
        SolverKind::DpmPlusPlus3M => dpmpp_3m(initial, schedule, &mut predictor),
    }
}

fn euler<B, F>(
    mut z: Tensor<B, 2>,
    schedule: &[f64],
    predictor: &mut F,
    ancestral_eta: Option<f64>,
) -> Tensor<B, 2>
where
    B: Backend,
    F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
{
    for window in schedule.windows(2) {
        let (s, s_next) = (window[0], window[1]);
        let x0 = predictor(s, &z);

        // Optional DDIM-style ancestral splitting of the step.
        let (target, noise_std) = match ancestral_eta {
            None => (s_next, 0.0),
            Some(eta) if eta <= 0.0 => (s_next, 0.0),
            Some(eta) => {
                let ratio = s_next / s;
                let var_up = eta * eta * s_next * s_next * (1.0 - ratio * ratio);
                let up = var_up.max(0.0).sqrt();
                let down = (s_next * s_next - up * up).max(0.0).sqrt();
                (down, up)
            }
        };

        if noise_std > 0.0 {
            let device = z.device();
            let dims = z.dims();
            let eps = Tensor::<B, 2>::random(dims, burn::tensor::Distribution::Normal(0.0, 1.0), &device)
                .mul_scalar(noise_std as f32);
            z = z + eps;
        }

        // Euler step of dz/dsigma = (x0 - z)/sigma toward `target`.
        let d = (z.clone() - x0) / s;
        z = z + (target - s) * d;
    }
    z
}

fn heun<B, F>(mut z: Tensor<B, 2>, schedule: &[f64], predictor: &mut F) -> Tensor<B, 2>
where
    B: Backend,
    F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
{
    for window in schedule.windows(2) {
        let (s, s_next) = (window[0], window[1]);
        let d = (z.clone() - predictor(s, &z)) / s;

        // Predictor: provisional Euler step to the next sigma.
        let z_euler = z.clone() + (s_next - s) * d.clone();
        // Corrector: average the drifts at both endpoints.
        let d_next = (z_euler.clone() - predictor(s_next, &z_euler)) / s_next;
        z = z + (s_next - s) * (d + d_next) * 0.5;
    }
    z
}

fn dpmpp_2m<B, F>(mut z: Tensor<B, 2>, schedule: &[f64], predictor: &mut F) -> Tensor<B, 2>
where
    B: Backend,
    F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
{
    let mut old_denoised: Option<Tensor<B, 2>> = None;
    let mut t_prev: f64 = f64::NAN;

    for window in schedule.windows(2) {
        let (s, s_next) = (window[0], window[1]);
        // lambda = -log sigma; strictly increasing along the descending schedule.
        let (t, t_next) = (-s.ln(), -s_next.ln());
        let h = t_next - t;

        let denoised = predictor(s, &z);
        let denoised_d = match &old_denoised {
            None => denoised.clone(),
            Some(old) => {
                let r = (t_prev - t) / h;
                // Multistep extrapolation blending current and previous x0.
                denoised.clone() * (1.0 + 0.5 / r) - old.clone() * (0.5 / r)
            }
        };

        z = z * (s_next / s) as f32 + (-(-h).exp_m1()) * denoised_d;
        t_prev = t;
        old_denoised = Some(denoised);
    }
    z
}

/// Third-order multistep: quadratic (in lambda) extrapolation of the last
/// three x0 predictions, integrated against the exact exponential kernel.
fn dpmpp_3m<B, F>(mut z: Tensor<B, 2>, schedule: &[f64], predictor: &mut F) -> Tensor<B, 2>
where
    B: Backend,
    F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
{
    // Two most recent (lambda_t, x0-prediction) pairs; None during warm-up.
    let mut prev1: Option<(f64, Tensor<B, 2>)> = None;
    let mut prev2: Option<(f64, Tensor<B, 2>)> = None;

    for window in schedule.windows(2) {
        let (s, s_next) = (window[0], window[1]);
        let (t, t_next) = (-s.ln(), -s_next.ln());
        let h = t_next - t;
        let e = (-h).exp();

        let denoised = predictor(s, &z);

        match (&prev1, &prev2) {
            (Some((t1, d1)), Some((t2, d2))) => {
                // Backward interpolation node offsets from the current point.
                let g0 = t - t1;
                let g1 = g0 + (t1 - t2);
                let a_n = d1.clone() - denoised.clone();
                let b_n = d2.clone() - denoised.clone();
                let coef_a =
                    (a_n.clone() * (g1 as f32) - b_n * (g0 as f32)) / ((g0 * g1 * (g0 - g1)) as f32);
                let coef_b = coef_a.clone() * (g0 as f32) - a_n / (g0 as f32);

                let j0 = 1.0 - e;
                let j1 = h - 1.0 + e;
                let j2 = h * h - 2.0 * h + 2.0 - 2.0 * e;

                z = z * (e as f32)
                    + denoised.clone() * (j0 as f32)
                    + coef_b * (j1 as f32)
                    + coef_a * (j2 as f32);
            }
            _ => {
                // Warm-up: exact first-order exponential step.
                z = z * (e as f32) + (1.0 - e) * denoised.clone();
            }
        }

        prev2 = prev1.take();
        prev1 = Some((t, denoised));
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};

    type B = NdArray<f32>;

    const SCHEDULE: [f64; 8] = [80.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.05, 0.002];

    /// For a CONSTANT x0 oracle the ODE dz/ds = (c - z)/s has closed form
    /// z(s) = c + (z0 - c) * s / s0, which every consistent solver must
    /// reproduce to discretization error.
    fn constant_oracle_error(kind: SolverKind) -> f32 {
        let device = Default::default();
        let b = 4usize;
        let h_dim = 8usize;
        let z0 = Tensor::<B, 2>::ones([b, h_dim], &device);
        let c = 0.0f64;

        let mut rng = StdRng::seed_from_u64(0);
        let z_end = integrate(
            z0.clone(),
            &SCHEDULE,
            |_sigma, _z| Tensor::full([b, h_dim], c as f32, &device),
            kind,
            &mut rng,
        );

        let expected = (z0 - c as f32) * (SCHEDULE[SCHEDULE.len() - 1] / SCHEDULE[0]) as f32;
        (z_end - expected).abs().max().into_scalar()
    }

    #[test]
    fn test_solver_parse_roundtrip() {
        for name in ["euler", "heun", "ddim", "dpmpp2m"] {
            assert_eq!(SolverKind::parse(name).unwrap().name(), name);
        }
        assert!(SolverKind::parse("rk45").is_err());
    }

    #[test]
    fn test_schedule_validation() {
        let device = Default::default();
        let z = Tensor::<B, 2>::ones([1, 4], &device);
        let mut rng = StdRng::seed_from_u64(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            integrate(
                z,
                &[1.0, 2.0],
                |_, _: &Tensor<B, 2>| unimplemented!(),
                SolverKind::Euler,
                &mut rng,
            )
        }));
        assert!(result.is_err(), "ascending schedule must be rejected");
    }

    #[test]
    fn test_constant_oracle_convergence() {
        // Deterministic solvers must all track the closed-form solution.
        assert!(constant_oracle_error(SolverKind::Euler) < 1e-4);
        assert!(constant_oracle_error(SolverKind::Heun) < 1e-4);
        assert!(constant_oracle_error(SolverKind::Ddim { eta: 0.0 }) < 1e-4);
        assert!(constant_oracle_error(SolverKind::DpmPlusPlus2M) < 1e-4);
    }

    #[test]
    fn test_ddim_eta_adds_noise() {
        // With eta > 0 the trajectory becomes stochastic; two different seeds
        // must land on measurably different latents.
        let run = |seed: u64| -> Vec<f32> {
            let device = Default::default();
            let mut rng = StdRng::seed_from_u64(seed);
            integrate(
                Tensor::<B, 2>::ones([2, 8], &device),
                &SCHEDULE,
                |_, _| Tensor::zeros([2, 8], &device),
                SolverKind::Ddim { eta: 1.0 },
                &mut rng,
            )
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect()
        };
        let a = run(1);
        let b = run(2);
        assert_ne!(a, b, "eta > 0 should introduce stochasticity");
    }

    /// Smooth nonconstant oracle x0(sigma) = tanh(sigma / 20); the ODE
    /// reference is computed with 20k tiny Euler steps in sigma space.
    fn reference_solution(z0: &[f32]) -> Vec<f32> {
        // Integrate dx/dlambda = -x + D(lambda) with uniform lambda steps
        // (stable all the way down to sigma_min).
        let mut z = z0.to_vec();
        let steps = 50_000usize;
        let lam_hi = SCHEDULE[0].ln();
        let lam_lo = SCHEDULE[SCHEDULE.len() - 1].ln();
        let dl = (lam_hi - lam_lo) / steps as f64; // positive magnitude
        let mut lam = lam_hi;
        for _ in 0..steps {
            let sig = lam.exp();
            let d0 = (sig / 20.0).tanh() as f32;
            let step = dl as f32;
            for v in z.iter_mut() {
                *v += (d0 - *v) * step;
            }
            lam -= dl;
        }
        z
    }

    #[test]
    fn test_dpmpp3m_order_beats_second_order() {
        // Uniform-in-lambda grid: coarse/unequal production schedules cannot
        // exhibit asymptotic order; 16+ uniform points can.
        let t_lo = -SCHEDULE[0].ln();
        let t_hi = -SCHEDULE[SCHEDULE.len() - 1].ln();
        let n_steps = 16usize;
        let mut grid: Vec<f64> = (0..=n_steps)
            .map(|i| (-(t_lo + i as f64 * (t_hi - t_lo) / n_steps as f64)).exp())
            .collect();
        grid.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending sigmas

        let device = Default::default();
        let b = 2usize;
        let dim = 4usize;
        let z0_vec: Vec<f32> = (0..b * dim).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let z0 =
            Tensor::<B, 1>::from_floats(z0_vec.clone().as_slice(), &device).reshape([b, dim]);

        let run = |kind: SolverKind| -> f32 {
            let mut rng = StdRng::seed_from_u64(0);
            let z_end = integrate(
                z0.clone(),
                &grid,
                |sigma, _zz| {
                    let vals: Vec<f32> =
                        vec![(sigma / 20.0).tanh() as f32; b * dim];
                    Tensor::<B, 1>::from_floats(vals.as_slice(), &device).reshape([b, dim])
                },
                kind,
                &mut rng,
            );
            let got: Vec<f32> = z_end.into_data().convert::<f32>().iter::<f32>().collect();
            let reference = reference_solution(&z0_vec);
            got.iter()
                .zip(reference)
                .map(|(g, r)| (g - r).abs())
                .fold(0.0f32, f32::max)
        };

        let err_euler = run(SolverKind::Euler);
        let err_2m = run(SolverKind::DpmPlusPlus2M);
        let err_3m = run(SolverKind::DpmPlusPlus3M);

        assert!(err_2m < err_euler, "2M ({err_2m}) should beat Euler ({err_euler})");
        assert!(
            err_3m < err_2m,
            "third-order 3M ({err_3m}) must beat second-order 2M ({err_2m})"
        );
    }
}
