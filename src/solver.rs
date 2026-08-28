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
//!   `lambda = -log sigma` space (Lu et al., 2022). No extra model calls; it
//!   reuses the previous step's x0 prediction.
//! - [`SolverKind::DpmPlusPlus3M`]: third-order multistep, quadratic
//!   extrapolation of the last three x0 predictions.
//!
//! # Exponential-integrator form
//!
//! Substituting `lambda = -log sigma` turns the ODE into the semilinear form
//! `dz/dlambda = -z + x0(lambda)`, whose exact solution over one step of size
//! `h` is
//!
//! ```text
//! z(lambda + h) = e^-h z(lambda) + int_0^h e^-(h-s) x0(lambda + s) ds.
//! ```
//!
//! The DPM-Solver++ family replaces `x0` inside that integral by a polynomial
//! interpolant through past predictions, so the stiff linear part is always
//! handled exactly and only the source term is approximated.
//!
//! Two variants of that idea appear here, deliberately:
//!
//! - [`SolverKind::DpmPlusPlus2M`] is the *published* update
//!   `z+ = e^-h z + (1 - e^-h) D~` with the standard extrapolated
//!   `D~ = (1 + 1/2r) D_i - (1/2r) D_{i-1}`. Its `(1 - e^-h) h / 2` weight on
//!   the slope differs from the exact `int_0^h s e^-(h-s) ds = h - 1 + e^-h`
//!   at order `h^3`, which is why it is second- and not third-order. Kept
//!   bit-faithful to the reference implementations so published results
//!   reproduce.
//! - [`SolverKind::DpmPlusPlus3M`] integrates the quadratic interpolant
//!   against the exact kernel, so it reproduces *any* `x0` that is quadratic
//!   in `lambda` to machine precision at any step size. That exactness is
//!   asserted as a certificate rather than assumed (see [`crate::verify`]).

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
    /// Parse a CLI solver name
    /// (`euler`, `heun`, `ddim`, `dpmpp2m`, `dpmpp3m`).
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

    /// Every solver, for benchmark/verification sweeps.
    pub fn all() -> [Self; 5] {
        [
            Self::Euler,
            Self::Heun,
            Self::Ddim { eta: 1.0 },
            Self::DpmPlusPlus2M,
            Self::DpmPlusPlus3M,
        ]
    }

    /// Deterministic (noise-free) solvers, i.e. every kind whose trajectory is
    /// a pure ODE integration.
    pub fn deterministic() -> [Self; 5] {
        [
            Self::Euler,
            Self::Heun,
            Self::Ddim { eta: 0.0 },
            Self::DpmPlusPlus2M,
            Self::DpmPlusPlus3M,
        ]
    }

    /// Classical order of accuracy of the discretization.
    pub fn order(&self) -> u32 {
        match self {
            Self::Euler | Self::Ddim { .. } => 1,
            Self::Heun | Self::DpmPlusPlus2M => 2,
            Self::DpmPlusPlus3M => 3,
        }
    }

    /// Model evaluations consumed per integration step.
    pub fn calls_per_step(&self) -> usize {
        match self {
            Self::Heun => 2,
            _ => 1,
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

/// Incremental driver for one solver: feed it a single
/// `(sigma, sigma_next, z, x0)` at a time and it returns the updated latent,
/// carrying whatever multistep history the method needs.
///
/// This is the single implementation of every update rule. [`integrate`] is a
/// thin loop over it, and [`crate::multi_block`] drives it step by step so it
/// can vary the executed layer span per window and apply quality gates
/// between steps -- without either path re-deriving the arithmetic.
pub struct SolverState<B: Backend> {
    kind: SolverKind,
    /// `(lambda, x0)` of the most recent steps, newest first, capped at two
    /// (the deepest history any implemented method needs).
    history: Vec<(f64, Tensor<B, 2>)>,
}

impl<B: Backend> SolverState<B> {
    pub fn new(kind: SolverKind) -> Self {
        Self { kind, history: Vec::with_capacity(2) }
    }

    pub fn kind(&self) -> SolverKind {
        self.kind
    }

    /// Number of past x0 predictions currently retained. Multistep methods
    /// use lower-order updates until this is large enough.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Advance the latent from `s` to `s_next`.
    ///
    /// `x0` is the clean-data estimate already computed at `(s, z)`; the
    /// caller owns that evaluation so it can choose the layer span. Only
    /// [`SolverKind::Heun`] needs the extra `predictor` call (at the step's
    /// endpoint); the other kinds ignore it.
    pub fn step<F>(
        &mut self,
        s: f64,
        s_next: f64,
        z: Tensor<B, 2>,
        x0: &Tensor<B, 2>,
        predictor: &mut F,
        rng: &mut impl Rng,
    ) -> Tensor<B, 2>
    where
        F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
    {
        let out = match self.kind {
            SolverKind::Euler => self.step_euler(s, s_next, z, x0, None, rng),
            SolverKind::Ddim { eta } => self.step_euler(s, s_next, z, x0, Some(eta), rng),
            SolverKind::Heun => {
                // Predictor: provisional Euler step to the next sigma.
                let d = (z.clone() - x0.clone()) / s;
                let z_euler = z.clone() + (s_next - s) * d.clone();
                // Corrector: average the drifts at both endpoints.
                let d_next = (z_euler.clone() - predictor(s_next, &z_euler)) / s_next;
                z + (s_next - s) * (d + d_next) * 0.5
            }
            SolverKind::DpmPlusPlus2M => self.step_dpmpp_2m(s, s_next, z, x0),
            SolverKind::DpmPlusPlus3M => self.step_dpmpp_3m(s, s_next, z, x0),
        };

        self.history.insert(0, (-s.ln(), x0.clone()));
        self.history.truncate(2);
        out
    }

    fn step_euler(
        &self,
        s: f64,
        s_next: f64,
        z: Tensor<B, 2>,
        x0: &Tensor<B, 2>,
        ancestral_eta: Option<f64>,
        rng: &mut impl Rng,
    ) -> Tensor<B, 2> {
        // DDIM/ancestral split of the step (k-diffusion `get_ancestral_step`):
        // deterministically descend to `sigma_down`, then inject `sigma_up`
        // noise so the marginal variance still lands on `s_next`.
        let (sigma_down, sigma_up) = ancestral_split(s, s_next, ancestral_eta);

        // Euler step of dz/dsigma = (z - x0)/sigma toward `sigma_down`.
        let d = (z.clone() - x0.clone()) / s;
        let mut z = z + (sigma_down - s) * d;

        // ...then the ancestral noise, at the *new* sigma level.
        if sigma_up > 0.0 {
            let noise = randn_like(&z, rng).mul_scalar(sigma_up as f32);
            z = z + noise;
        }
        z
    }

    /// Published DPM-Solver++(2M) update (Lu et al., 2022):
    /// `z+ = e^-h z + (1 - e^-h) D~` with the extrapolated
    /// `D~ = (1 + 1/2r) D_i - (1/2r) D_{i-1}`.
    fn step_dpmpp_2m(&self, s: f64, s_next: f64, z: Tensor<B, 2>, x0: &Tensor<B, 2>) -> Tensor<B, 2> {
        // lambda = -log sigma; strictly increasing along the descending schedule.
        let (t, t_next) = (-s.ln(), -s_next.ln());
        let h = t_next - t;

        let denoised_d = match self.history.first() {
            None => x0.clone(),
            Some((t_prev, old)) => {
                // r = h_prev / h_cur, both positive: lambda grows along the
                // descending schedule, so h_prev = t - t_prev > 0.
                let r = (t - t_prev) / h;
                x0.clone() * (1.0 + 0.5 / r) as f32 - old.clone() * (0.5 / r) as f32
            }
        };

        z * (s_next / s) as f32 + denoised_d * (-(-h).exp_m1()) as f32
    }

    /// Third-order multistep: quadratic (in lambda) extrapolation of the last
    /// three x0 predictions, integrated against the exact exponential kernel.
    fn step_dpmpp_3m(&self, s: f64, s_next: f64, z: Tensor<B, 2>, x0: &Tensor<B, 2>) -> Tensor<B, 2> {
        let (t, t_next) = (-s.ln(), -s_next.ln());
        let h = t_next - t;
        let e = (-h).exp();
        let [j0, j1, j2] = kernel_moments(h);

        match (self.history.first(), self.history.get(1)) {
            (Some((t1, d1)), Some((t2, d2))) => {
                // Backward interpolation node offsets from the current point.
                let (g0, g1) = (t - t1, t - t2);
                let a_n = d1.clone() - x0.clone();
                let b_n = d2.clone() - x0.clone();
                let [wa_a, wa_b, wb_a, wb_b] = quadratic_interp_weights(g0, g1);
                let coef_a = a_n.clone() * (wa_a as f32) + b_n.clone() * (wa_b as f32);
                let coef_b = a_n * (wb_a as f32) + b_n * (wb_b as f32);

                z * (e as f32)
                    + x0.clone() * (j0 as f32)
                    + coef_b * (j1 as f32)
                    + coef_a * (j2 as f32)
            }
            (Some((t1, d1)), None) => {
                // One prior node available: linear (in lambda) interpolant
                // x0(lambda + s) = x0 + b*s with b = (x0 - d1)/g0, integrated
                // against the same exponential kernel. This is the
                // exponential-integrator form of the 2M step and keeps the
                // warm-up second order instead of dropping to first.
                let g0 = t - t1;
                let coef_b = (x0.clone() - d1.clone()) / (g0 as f32);
                z * (e as f32) + x0.clone() * (j0 as f32) + coef_b * (j1 as f32)
            }
            _ => {
                // First step: exact first-order exponential step (no history).
                z * (e as f32) + x0.clone() * (j0 as f32)
            }
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
    rng: &mut impl Rng,
) -> Tensor<B, 2>
where
    B: Backend,
    F: FnMut(f64, &Tensor<B, 2>) -> Tensor<B, 2>,
{
    assert_descending(schedule);

    let mut state = SolverState::new(kind);
    let mut z = initial;
    for window in schedule.windows(2) {
        let (s, s_next) = (window[0], window[1]);
        let x0 = predictor(s, &z);
        z = state.step(s, s_next, z, &x0, &mut predictor, rng);
    }
    z
}

/// Reject schedules that are not strictly descending and positive.
///
/// Every update rule divides by `sigma` and takes `log sigma`, so a flat,
/// ascending or non-positive schedule is a programming error rather than a
/// degenerate case to tolerate silently.
pub fn assert_descending(schedule: &[f64]) {
    assert!(schedule.len() >= 2, "need at least two schedule points");
    assert!(
        schedule.windows(2).all(|w| w[0] > w[1]),
        "schedule must be strictly descending"
    );
    assert!(
        schedule.last().is_some_and(|&s| s > 0.0),
        "schedule sigmas must be strictly positive"
    );
}

/// Split one `s -> s_next` step into a deterministic target `sigma_down` and
/// an ancestral noise scale `sigma_up` such that
/// `sigma_down^2 + sigma_up^2 == s_next^2` (variance preserved).
///
/// `eta = 0` (or `None`) gives `(s_next, 0)`, i.e. the plain Euler step.
pub fn ancestral_split(s: f64, s_next: f64, eta: Option<f64>) -> (f64, f64) {
    match eta {
        None => (s_next, 0.0),
        Some(eta) if eta <= 0.0 => (s_next, 0.0),
        Some(eta) => {
            let ratio = s_next / s;
            let var_up = eta * eta * s_next * s_next * (1.0 - ratio * ratio);
            // Clamp so eta > 1 cannot ask for more noise than the target level
            // holds (otherwise `sigma_down` would be imaginary).
            let up = var_up.max(0.0).sqrt().min(s_next);
            let down = (s_next * s_next - up * up).max(0.0).sqrt();
            (down, up)
        }
    }
}

/// Standard-normal tensor shaped like `x`, drawn from the *caller's* RNG so a
/// seeded `rng` fully determines a stochastic trajectory (Box-Muller; `rand`
/// is used without `rand_distr` to keep the dependency set small).
pub(crate) fn randn_like<B: Backend>(x: &Tensor<B, 2>, rng: &mut impl Rng) -> Tensor<B, 2> {
    let dims = x.dims();
    let n = dims[0] * dims[1];
    let mut values = Vec::with_capacity(n);
    while values.len() < n {
        // u1 in (0, 1]: ln(0) must not be reachable.
        let u1: f64 = 1.0 - rng.random::<f64>();
        let u2: f64 = rng.random::<f64>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        values.push((radius * angle.cos()) as f32);
        if values.len() < n {
            values.push((radius * angle.sin()) as f32);
        }
    }
    Tensor::<B, 1>::from_floats(values.as_slice(), &x.device()).reshape(dims)
}

/// Moments of the exponential kernel over one step,
/// `J_k = int_0^h s^k e^-(h-s) ds` for `k = 0, 1, 2`:
///
/// ```text
/// J0 = 1 - e^-h
/// J1 = h - 1 + e^-h
/// J2 = h^2 - 2h + 2 - 2 e^-h
/// ```
///
/// These are the only place the exponential integrator's step size enters, so
/// they are verified directly against numerical quadrature in the tests.
pub fn kernel_moments(h: f64) -> [f64; 3] {
    let e = (-h).exp();
    [1.0 - e, h - 1.0 + e, h * h - 2.0 * h + 2.0 - 2.0 * e]
}

/// Weights of the backward quadratic interpolant used by
/// [`SolverKind::DpmPlusPlus3M`].
///
/// Given the current prediction `d` and two earlier ones at lambda-offsets
/// `-g0` and `-g1` (so `0 < g0 < g1`), expressed through the differences
/// `a_n = d1 - d` and `b_n = d2 - d`, the interpolant
/// `p(s) = d + b s + a s^2` pinned to `p(-g0) = d1`, `p(-g1) = d2` has
///
/// ```text
/// a = w[0] a_n + w[1] b_n,   b = w[2] a_n + w[3] b_n.
/// ```
///
/// Returning weights rather than the coefficients themselves lets the tensor
/// path and the scalar verification share one source of truth.
pub fn quadratic_interp_weights(g0: f64, g1: f64) -> [f64; 4] {
    let det = g0 * g1 * (g0 - g1);
    let (wa_a, wa_b) = (g1 / det, -g0 / det);
    [wa_a, wa_b, g0 * wa_a - 1.0 / g0, g0 * wa_b]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};

    type B = NdArray<f32>;

    const SCHEDULE: [f64; 8] = [80.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.05, 0.002];

    /// Descending sigma grid with `n_steps` intervals uniform in
    /// `lambda = -log sigma`. Asymptotic order is only observable on a
    /// uniform grid; production schedules are deliberately non-uniform.
    fn uniform_lambda_grid(sigma_hi: f64, sigma_lo: f64, n_steps: usize) -> Vec<f64> {
        let t_lo = -sigma_hi.ln();
        let t_hi = -sigma_lo.ln();
        (0..=n_steps)
            .map(|i| (-(t_lo + i as f64 * (t_hi - t_lo) / n_steps as f64)).exp())
            .collect()
    }

    /// For a CONSTANT x0 oracle the ODE dz/ds = (z - c)/s has closed form
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

    /// Max-abs error of `kind` on a state-independent oracle `x0(lambda)`,
    /// measured against a very fine reference integration of the same ODE.
    ///
    /// A state-independent oracle keeps the ODE linear, so the reference can
    /// be made essentially exact and the residual is pure discretization
    /// error of the solver under test.
    fn oracle_error<F>(kind: SolverKind, grid: &[f64], x0_of_lambda: F) -> f64
    where
        F: Fn(f64) -> f64 + Copy,
    {
        let device = Default::default();
        let (b, dim) = (1usize, 1usize);
        let z0 = 1.7f64;

        let mut rng = StdRng::seed_from_u64(0);
        let z_end = integrate(
            Tensor::<B, 2>::full([b, dim], z0 as f32, &device),
            grid,
            |sigma, _z| {
                Tensor::<B, 2>::full([b, dim], x0_of_lambda(-sigma.ln()) as f32, &device)
            },
            kind,
            &mut rng,
        );
        let got: f32 = z_end.into_scalar();
        (got as f64 - reference_linear_ode(z0, grid, x0_of_lambda)).abs()
    }

    /// Reference solution of `dz/dlambda = -z + x0(lambda)` in f64, by
    /// composing the exact one-step update over a very fine uniform grid.
    fn reference_linear_ode<F>(z0: f64, grid: &[f64], x0_of_lambda: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let lam0 = -grid[0].ln();
        let lam1 = -grid[grid.len() - 1].ln();
        let n = 400_000usize;
        let dl = (lam1 - lam0) / n as f64;
        let mut z = z0;
        let mut lam = lam0;
        for _ in 0..n {
            // Midpoint value of the (state-independent) source term makes the
            // composed update second-order exact in dl, i.e. ~1e-11 overall.
            let src = x0_of_lambda(lam + 0.5 * dl);
            let e = (-dl).exp();
            z = e * z + (1.0 - e) * src;
            lam += dl;
        }
        z
    }

    /// Least-squares slope of log2(error) against log2(steps), i.e. the
    /// empirically observed order of convergence.
    fn empirical_order<F>(kind: SolverKind, x0_of_lambda: F, step_counts: &[usize]) -> f64
    where
        F: Fn(f64) -> f64 + Copy,
    {
        let points: Vec<(f64, f64)> = step_counts
            .iter()
            .map(|&n| {
                let grid = uniform_lambda_grid(SCHEDULE[0], SCHEDULE[SCHEDULE.len() - 1], n);
                let err = oracle_error(kind, &grid, x0_of_lambda).max(1e-12);
                ((n as f64).log2(), err.log2())
            })
            .collect();
        let m = points.len() as f64;
        let mx = points.iter().map(|p| p.0).sum::<f64>() / m;
        let my = points.iter().map(|p| p.1).sum::<f64>() / m;
        let num: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let den: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
        -num / den // error shrinks with steps, so negate to get a positive order
    }

    #[test]
    fn test_solver_parse_roundtrip() {
        for name in ["euler", "heun", "ddim", "dpmpp2m", "dpmpp3m"] {
            assert_eq!(SolverKind::parse(name).unwrap().name(), name);
        }
        assert!(SolverKind::parse("rk45").is_err());
        // `all()` must cover every parseable name exactly once.
        let mut names: Vec<&str> = SolverKind::all().iter().map(|k| k.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SolverKind::all().len());
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
        for kind in SolverKind::deterministic() {
            let err = constant_oracle_error(kind);
            assert!(err < 1e-4, "{} drifted off the closed form: {err}", kind.name());
        }
    }

    #[test]
    fn test_ancestral_split_preserves_variance() {
        // sigma_down^2 + sigma_up^2 == sigma_next^2 for every eta, and eta = 0
        // degenerates to the plain Euler target.
        for &(s, s_next) in &[(80.0, 30.0), (1.0, 0.3), (0.05, 0.002)] {
            let (d0, u0) = ancestral_split(s, s_next, Some(0.0));
            assert_eq!((d0, u0), (s_next, 0.0));
            for &eta in &[0.25f64, 0.5, 1.0, 2.0] {
                let (down, up) = ancestral_split(s, s_next, Some(eta));
                let total = down * down + up * up;
                assert!(
                    (total - s_next * s_next).abs() <= 1e-12 * s_next * s_next,
                    "variance not preserved at eta={eta}: {total} vs {}",
                    s_next * s_next
                );
                assert!(up <= s_next + 1e-12, "sigma_up must not exceed sigma_next");
            }
        }
    }

    #[test]
    fn test_ddim_eta_is_reproducible_from_the_caller_seed() {
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
        assert_ne!(run(1), run(2), "eta > 0 should introduce stochasticity");
        assert_eq!(run(1), run(1), "the same seed must replay the same trajectory");
    }

    #[test]
    fn test_kernel_moments_match_quadrature() {
        // J_k = int_0^h s^k e^-(h-s) ds, checked against composite Simpson at
        // 20k panels (error ~ h^5/n^4, far below the 1e-12 tolerance).
        for &h in &[0.05f64, 0.5, 1.0, 2.5, 6.0] {
            let closed = kernel_moments(h);
            for (k, &expected) in closed.iter().enumerate() {
                let f = |s: f64| s.powi(k as i32) * (-(h - s)).exp();
                let n = 20_000usize; // even
                let dx = h / n as f64;
                let mut acc = f(0.0) + f(h);
                for i in 1..n {
                    let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                    acc += w * f(i as f64 * dx);
                }
                let quad = acc * dx / 3.0;
                assert!(
                    (quad - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                    "J{k}({h}) closed form {expected} vs quadrature {quad}"
                );
            }
        }
    }

    #[test]
    fn test_quadratic_interp_weights_reproduce_the_nodes() {
        // The interpolant must pass exactly through both history points; that
        // property alone pins the weights, so verifying it verifies the
        // algebra behind the 3M coefficients.
        for &(g0, g1) in &[(0.5f64, 1.25), (1.0, 3.0), (0.1, 0.15), (2.0, 2.5)] {
            for &(a_n, b_n) in &[(1.0f64, -2.0), (0.0, 1.0), (-0.3, -0.9)] {
                let [wa_a, wa_b, wb_a, wb_b] = quadratic_interp_weights(g0, g1);
                let a = wa_a * a_n + wa_b * b_n;
                let b = wb_a * a_n + wb_b * b_n;
                // p(s) = b s + a s^2 measured relative to the current value.
                let p = |s: f64| b * s + a * s * s;
                assert!(
                    (p(-g0) - a_n).abs() < 1e-9,
                    "node 1 missed: p(-{g0}) = {} != {a_n}",
                    p(-g0)
                );
                assert!(
                    (p(-g1) - b_n).abs() < 1e-9,
                    "node 2 missed: p(-{g1}) = {} != {b_n}",
                    p(-g1)
                );
            }
        }
    }

    #[test]
    fn test_third_order_residual_is_warmup_limited() {
        // 3M integrates the quadratic interpolant against the exact kernel, so
        // once its history is filled it adds no error at all on a quadratic
        // x0(lambda). The only residual left is the first (historyless) step,
        // and the linear part of the ODE damps that transient by
        // e^-(remaining lambda span). Refining the grid therefore shrinks the
        // error far faster than the nominal h^3.
        let quadratic = |lam: f64| 0.05 * lam * lam + 0.3 * lam - 0.7;
        let (hi, lo) = (SCHEDULE[0], SCHEDULE[SCHEDULE.len() - 1]);

        let err_8 = oracle_error(SolverKind::DpmPlusPlus3M, &uniform_lambda_grid(hi, lo, 8), quadratic);
        let err_16 =
            oracle_error(SolverKind::DpmPlusPlus3M, &uniform_lambda_grid(hi, lo, 16), quadratic);

        assert!(err_8 < 1e-4, "3M residual at 8 steps too large: {err_8:e}");
        assert!(
            err_16 < err_8 / 8.0,
            "residual must shrink faster than third order: {err_8:e} -> {err_16:e}"
        );
    }

    #[test]
    fn test_published_2m_is_not_an_exact_exponential_integrator() {
        // The published 2M weights the slope by (1 - e^-h) h / 2 instead of the
        // exact h - 1 + e^-h. Pinning that keeps the deviation intentional and
        // stops anyone "fixing" it into a different algorithm by accident.
        let grid = uniform_lambda_grid(SCHEDULE[0], SCHEDULE[SCHEDULE.len() - 1], 4);
        let affine = |lam: f64| 0.3 * lam - 0.7;
        let err_2m = oracle_error(SolverKind::DpmPlusPlus2M, &grid, affine);
        assert!(err_2m > 1e-4, "2M unexpectedly exact on affine x0: {err_2m:e}");
    }

    #[test]
    fn test_dpmpp2m_matches_published_update() {
        // One multistep update recomputed by hand from Lu et al. (2022):
        //   z+ = (sigma_next / sigma) z + (1 - e^-h) D~,
        //   D~ = (1 + 1/2r) D_i - (1/2r) D_{i-1},  r = h_prev / h.
        let device = Default::default();
        let grid = [10.0f64, 3.0, 1.0];
        let d_of = |sigma: f64| 0.4 * sigma.ln() + 0.1;

        let mut rng = StdRng::seed_from_u64(0);
        let got: f64 = integrate(
            Tensor::<B, 2>::full([1, 1], 1.7f32, &device),
            &grid,
            |sigma, _z| Tensor::<B, 2>::full([1, 1], d_of(sigma) as f32, &device),
            SolverKind::DpmPlusPlus2M,
            &mut rng,
        )
        .into_scalar() as f64;

        // Step 1: warm-up (no history) is the plain first-order update.
        let h0 = -grid[1].ln() + grid[0].ln();
        let mut z = (grid[1] / grid[0]) * 1.7 + (1.0 - (-h0).exp()) * d_of(grid[0]);
        // Step 2: multistep with r = h_prev / h.
        let h1 = -grid[2].ln() + grid[1].ln();
        let r = h0 / h1;
        let d_tilde = (1.0 + 0.5 / r) * d_of(grid[1]) - (0.5 / r) * d_of(grid[0]);
        z = (grid[2] / grid[1]) * z + (1.0 - (-h1).exp()) * d_tilde;

        assert!(
            (got - z).abs() < 1e-5,
            "2M update drifted from the published formula: {got} vs {z}"
        );
    }

    #[test]
    fn test_empirical_orders_match_theory() {
        // Smooth, state-independent oracle; error must decay like steps^-order.
        let oracle = |lam: f64| (0.6 * lam).tanh();
        let counts = [8usize, 16, 32, 64];

        for kind in SolverKind::deterministic() {
            let order = empirical_order(kind, oracle, &counts);
            let expected = kind.order() as f64;
            assert!(
                order >= expected - 0.25,
                "{} observed order {order:.2}, expected ~{expected}",
                kind.name()
            );
        }
    }

    #[test]
    fn test_higher_order_beats_lower_order_at_equal_steps() {
        let oracle = |lam: f64| (0.6 * lam).tanh();
        let grid = uniform_lambda_grid(SCHEDULE[0], SCHEDULE[SCHEDULE.len() - 1], 16);

        let err_euler = oracle_error(SolverKind::Euler, &grid, oracle);
        let err_2m = oracle_error(SolverKind::DpmPlusPlus2M, &grid, oracle);
        let err_3m = oracle_error(SolverKind::DpmPlusPlus3M, &grid, oracle);

        assert!(err_2m < err_euler, "2M ({err_2m:e}) should beat Euler ({err_euler:e})");
        assert!(err_3m < err_2m, "3M ({err_3m:e}) should beat 2M ({err_2m:e})");
    }
}
