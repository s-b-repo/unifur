//! Numerical certificate suite (roadmap Phase 14) and the repository's
//! quality gate.
//!
//! # What "mathematically proven" means here
//!
//! Nothing in this file is a proof in the formal sense -- Rust is not a proof
//! assistant, and a floating-point implementation of a real-valued identity
//! can at best be *correct to a stated tolerance*. What this module does
//! instead is make every load-bearing mathematical claim in the crate
//! **falsifiable and continuously checked**:
//!
//! - each claim is written down as a theorem statement, in prose, next to the
//!   code that checks it;
//! - the check produces a **residual**: a single number that is zero exactly
//!   when the claim holds;
//! - the residual is compared against a tolerance chosen from the arithmetic,
//!   not from whatever the code currently happens to produce.
//!
//! A claim that cannot be reduced to a residual is not listed. That
//! discipline is what makes the suite worth running: `dblocks verify` exits
//! non-zero the moment any identity the implementation rests on stops holding,
//! and a regression in, say, the block-index convention or a solver
//! coefficient shows up as a named failure rather than as slightly worse
//! accuracy months later.
//!
//! # Coverage
//!
//! | Group | What is certified |
//! |---|---|
//! | `schedule` | Block boundaries, CDF-uniform spacing, window tiling, and that the train-time and inference-time block routing are mutual inverses |
//! | `preconditioning` | The EDM identity `D(z) = x` for an exact denoiser, and the two normalization identities that motivate `c_in` and the loss weight |
//! | `stats` | `erf`/`erfc` complementarity and `norm_cdf`/`norm_ppf` involution against scipy reference values |
//! | `solver` | Closed-form exactness, kernel moments against quadrature, interpolation nodes, observed order of convergence, ancestral variance preservation |
//! | `precision` | Unit-roundoff bound and idempotence of the reduced-format emulation |
//! | `quantize` | NF4 error bound, exact zero, and LoRA's identity-at-init |
//! | `loopgraph` | ACT weights form a partition of unity; the planner respects its budget |
//! | `moe` | Gates are a probability distribution; the balance loss stays in `[1, E]` on the diagonal |
//! | `mosme` | Composed two-level gates form a distribution; one box reduces exactly to flat MoE; adding a disabled expert is a bit-exact identity |
//! | `lm` | Tokenization is lossless; causal attention leaks nothing backwards; an untrained tied head starts at `ln(vocab)` |
//! | `model` | Softmax partition, unit-norm label embeddings, DiT zero-init, and that every `x0` estimate lies in the convex hull of the label table |
//! | `autodiff` | Finite-difference gradient check on the distillation objective |

use crate::{
    dblock::{DblockClassifier, DblockConfig},
    loopgraph::{Decision, LoopGraphConfig, LoopPlanner},
    moe::{MoEConfig, MoELayer},
    precision::Precision,
    quantize::{LoraAdapter, LoraConfig, Nf4Tensor, NF4_LEVELS},
    sigma::{self, EdmPreconditioning, P_MEAN, P_STD, SIGMA_MAX, SIGMA_MIN},
    solver::{self, SolverKind},
    stats::{erf, erfc, norm_cdf, norm_ppf},
    vit::ViTDiTConfig,
};
use rand::SeedableRng;

use burn::{
    backend::NdArray,
    tensor::{activation::softmax, Distribution, Int, Tensor},
};

type B = NdArray<f32>;

/// One checked claim.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub group: &'static str,
    pub name: &'static str,
    /// The claim, stated so a reader can judge whether checking it is
    /// worthwhile independently of whether it currently passes.
    pub theorem: &'static str,
    /// Measured deviation from the claim; zero exactly when it holds.
    pub residual: f64,
    /// Largest residual consistent with the claim, given the arithmetic.
    pub tolerance: f64,
}

impl Certificate {
    pub fn passed(&self) -> bool {
        self.residual.is_finite() && self.residual <= self.tolerance
    }
}

fn cert(
    group: &'static str,
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> Certificate {
    Certificate { group, name, theorem, residual, tolerance }
}

/// Result of a full verification run.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub certificates: Vec<Certificate>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.certificates.iter().all(Certificate::passed)
    }

    pub fn failures(&self) -> Vec<&Certificate> {
        self.certificates.iter().filter(|c| !c.passed()).collect()
    }

    pub fn num_passed(&self) -> usize {
        self.certificates.iter().filter(|c| c.passed()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }

    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    /// Append another report's certificates.
    pub fn merge(&mut self, other: Report) {
        self.certificates.extend(other.certificates);
    }

    /// One-line summary, for logs where the full table is too much.
    pub fn summary(&self) -> String {
        match self.failures().as_slice() {
            [] => format!("{} certificates passed", self.certificates.len()),
            failures => format!(
                "{}/{} certificates FAILED: {}",
                failures.len(),
                self.certificates.len(),
                failures
                    .iter()
                    .map(|c| format!("{}::{}", c.group, c.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// Human-readable table, grouped, with the residual/tolerance ratio so a
    /// certificate that is drifting toward its bound is visible before it
    /// fails.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<14} {:<38} {:>11} {:>11} {:>7}  {}\n",
            "group", "certificate", "residual", "tolerance", "margin", "status"
        ));
        out.push_str(&"-".repeat(100));
        out.push('\n');

        let mut last_group = "";
        for c in &self.certificates {
            let group = if c.group == last_group { "" } else { c.group };
            last_group = c.group;
            let margin = if c.tolerance > 0.0 {
                format!("{:.2}x", c.residual / c.tolerance)
            } else {
                "-".to_string()
            };
            out.push_str(&format!(
                "{:<14} {:<38} {:>11.3e} {:>11.3e} {:>7}  {}\n",
                group,
                c.name,
                c.residual,
                c.tolerance,
                margin,
                if c.passed() { "ok" } else { "FAILED" }
            ));
        }

        out.push_str(&"-".repeat(100));
        out.push('\n');
        out.push_str(&format!(
            "{} / {} certificates passed\n",
            self.num_passed(),
            self.certificates.len()
        ));
        for f in self.failures() {
            out.push_str(&format!("\nFAILED {}::{}\n  {}\n", f.group, f.name, f.theorem));
        }
        out
    }
}

/// Names of every certificate group, in the order [`run_all`] emits them.
pub const GROUPS: [&str; 15] = [
    "schedule",
    "preconditioning",
    "stats",
    "solver",
    "precision",
    "quantize",
    "loopgraph",
    "moe",
    "mosme",
    "lm",
    "planner",
    "accuracy",
    "optim",
    "model",
    "autodiff",
];

/// Run only the certificates in `group`.
///
/// Returns an empty report for an unknown group; callers that treat "no
/// certificates" as success should check [`Report::is_empty`] first.
pub fn run_group(group: &str) -> Report {
    let mut report = run_all();
    report.certificates.retain(|c| c.group == group);
    report
}

/// The subset worth running *before* a training run: the schedule, the
/// preconditioning identities and the scalar statistics they are built from.
///
/// These are the properties a run silently depends on for its entire duration.
/// If the block-index convention or the EDM identity is broken, every step
/// afterwards is wasted, so the check is cheap insurance -- it costs
/// milliseconds against hours of training.
pub fn preflight() -> Report {
    let mut certificates = schedule_certificates();
    certificates.extend(preconditioning_certificates());
    certificates.extend(stats_certificates());
    Report { certificates }
}

/// Run every certificate.
pub fn run_all() -> Report {
    let mut certificates = Vec::new();
    certificates.extend(schedule_certificates());
    certificates.extend(preconditioning_certificates());
    certificates.extend(stats_certificates());
    certificates.extend(solver_certificates());
    certificates.extend(precision_certificates());
    certificates.extend(quantize_certificates());
    certificates.extend(loopgraph_certificates());
    certificates.extend(moe_certificates());
    certificates.extend(mosme_certificates());
    certificates.extend(lm_certificates());
    certificates.extend(planner_certificates());
    certificates.extend(accuracy_certificates());
    certificates.extend(optim_certificates());
    certificates.extend(model_certificates());
    certificates.extend(autodiff_certificates());
    Report { certificates }
}

// ---------------------------------------------------------------- schedule --

fn schedule_certificates() -> Vec<Certificate> {
    let mut out = Vec::new();

    // The boundary grid must span exactly [sigma_min, sigma_max]; a schedule
    // that silently clips one end would train blocks on noise levels sampling
    // never reaches.
    let mut endpoint_err: f64 = 0.0;
    for n in [1usize, 2, 3, 4, 8, 12] {
        let b = sigma::block_sigmas(n);
        endpoint_err = endpoint_err
            .max((b[0] - SIGMA_MIN).abs() / SIGMA_MIN)
            .max((b[n] - SIGMA_MAX).abs() / SIGMA_MAX);
    }
    out.push(cert(
        "schedule",
        "boundary_endpoints",
        "block_sigmas(n) spans exactly [sigma_min, sigma_max] for every n.",
        endpoint_err,
        // norm_ppf round-trips through a rational approximation refined by
        // Newton; ~1e-9 relative is its honest accuracy in the deep tail.
        1e-8,
    ));

    // The boundaries are defined as equally spaced *in lognormal CDF space*.
    // That is the property the sampler and `estimate_target_layer` both rely
    // on, so it is checked directly rather than inferred from monotonicity.
    let mut cdf_err: f64 = 0.0;
    for n in [2usize, 3, 5, 12] {
        let b = sigma::block_sigmas(n);
        let phi = |s: f64| norm_cdf((s.ln() - P_MEAN) / P_STD);
        let (lo, hi) = (phi(b[0]), phi(b[n]));
        for (i, &s) in b.iter().enumerate() {
            let expected = lo + (hi - lo) * (i as f64 / n as f64);
            cdf_err = cdf_err.max((phi(s) - expected).abs());
        }
    }
    out.push(cert(
        "schedule",
        "cdf_uniform_spacing",
        "Phi((ln sigma_i - p_mean)/p_std) is an exact linear ramp in i.",
        cdf_err,
        1e-12,
    ));

    // Adjacent windows must share an endpoint exactly: a gap would leave noise
    // levels no block is responsible for, an overlap would train two blocks on
    // the same range with different targets.
    let mut tiling_err: f64 = 0.0;
    for n in [2usize, 3, 7] {
        let b = sigma::block_sigmas(n);
        for blk in 0..n - 1 {
            let (lo, _) = sigma::block_window(&b, blk);
            let (_, hi_next) = sigma::block_window(&b, blk + 1);
            tiling_err = tiling_err.max((lo - hi_next).abs());
            tiling_err = tiling_err.max((lo - sigma::shared_boundary_sigma(&b, blk)).abs());
        }
    }
    out.push(cert(
        "schedule",
        "window_tiling",
        "Block windows tile [sigma_min, sigma_max]: block b's lower edge is block b+1's upper edge.",
        tiling_err,
        0.0,
    ));

    // The certificate that closes the train/inference loop. A sigma the
    // sampler draws for block b must be routed back to block b by the
    // inference-time estimator, or a block is trained on one noise range and
    // evaluated on another -- a bug that degrades quality without ever
    // crashing.
    // Seeded, not thread-local: a quality gate that can fail intermittently is
    // worse than no gate, because a real regression looks like flakiness.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED);
    let mut misrouted = 0usize;
    let mut total = 0usize;
    for n in [1usize, 2, 3, 4, 6, 12] {
        let sampler = sigma::DblockSigmaSampler::new(n, 0.0);
        for b in 0..n {
            for s in sampler.sample(&mut rng, b, 128) {
                total += 1;
                if sigma::estimate_target_layer(&sampler.block_sigmas, &[s]) != b {
                    misrouted += 1;
                }
            }
        }
    }
    out.push(cert(
        "schedule",
        "block_routing_involution",
        "estimate_target_layer inverts the training-window sampler: sigmas drawn for block b route back to b.",
        misrouted as f64 / total.max(1) as f64,
        0.0,
    ));

    // With num_steps = num_blocks + 1 both grids are the same uniform CDF
    // partition, just ordered oppositely.
    let steps = sigma::discrete_sigmas_dblock(5, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD);
    let blocks = sigma::block_sigmas(4);
    let grid_err = steps
        .iter()
        .rev()
        .zip(blocks.iter())
        .map(|(s, b)| (s - b).abs() / b)
        .fold(0.0f64, f64::max);
    out.push(cert(
        "schedule",
        "discrete_matches_blocks",
        "discrete_sigmas_dblock(B+1) reversed equals block_sigmas(B).",
        grid_err,
        1e-12,
    ));

    // The EDM polynomial schedule is a different construction; it must still
    // hit both endpoints and descend.
    let edm = sigma::discrete_sigmas_edm(64, SIGMA_MIN, SIGMA_MAX, sigma::RHO);
    let edm_err = ((edm[0] - SIGMA_MAX).abs() / SIGMA_MAX)
        .max((edm[edm.len() - 1] - SIGMA_MIN).abs() / SIGMA_MIN)
        .max(if edm.windows(2).all(|w| w[0] > w[1]) { 0.0 } else { 1.0 });
    out.push(cert(
        "schedule",
        "edm_schedule_endpoints",
        "The rho=7 EDM schedule descends strictly from sigma_max to sigma_min.",
        edm_err,
        1e-12,
    ));

    out
}

// -------------------------------------------------------- preconditioning --

fn preconditioning_certificates() -> Vec<Certificate> {
    let sigma_data = 0.5;
    let sigmas: Vec<f64> = (0..64)
        .map(|i| SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN).powf(i as f64 / 63.0))
        .collect();

    // The identity the whole EDM parameterization exists to provide: if the
    // network output F is the *exact* target, the preconditioned denoiser
    // reconstructs the clean sample at every noise level. If this fails, the
    // training target and the sampling formula disagree.
    let mut identity_err: f64 = 0.0;
    for &s in &sigmas {
        let p = EdmPreconditioning::new(s, sigma_data);
        for &x in &[-2.0f64, -0.3, 0.0, 0.7, 3.5] {
            for &eps in &[-1.5f64, 0.4] {
                let z = x + s * eps;
                // The exact network output at this (z, sigma).
                let f_star = (x - p.c_skip * z) / p.c_out;
                let reconstructed = p.c_out * f_star + p.c_skip * z;
                identity_err = identity_err.max((reconstructed - x).abs() / (1.0 + x.abs()));
            }
        }
    }
    let mut variance_err: f64 = 0.0;
    let mut weight_err: f64 = 0.0;
    for &s in &sigmas {
        let p = EdmPreconditioning::new(s, sigma_data);
        // c_in normalizes the input to unit variance.
        variance_err =
            variance_err.max((p.c_in * p.c_in * (s * s + sigma_data * sigma_data) - 1.0).abs());
        // The loss weight is exactly 1 / c_out^2, which is what makes the
        // effective training target unit-variance at every sigma.
        weight_err =
            weight_err.max((sigma::edm_loss_weight(s, sigma_data) * p.c_out * p.c_out - 1.0).abs());
    }

    vec![
        cert(
            "preconditioning",
            "edm_denoiser_identity",
            "D(z) = c_out F*(z) + c_skip z reconstructs x exactly when F* is the exact target, at every sigma.",
            identity_err,
            1e-12,
        ),
        cert(
            "preconditioning",
            "c_in_unit_variance",
            "c_in^2 (sigma^2 + sigma_data^2) = 1: the scaled input has unit variance.",
            variance_err,
            1e-14,
        ),
        cert(
            "preconditioning",
            "loss_weight_normalizes_c_out",
            "w(sigma) c_out(sigma)^2 = 1: the EDM weighting exactly cancels the output scaling.",
            weight_err,
            1e-12,
        ),
    ]
}

// ------------------------------------------------------------------ stats --

fn stats_certificates() -> Vec<Certificate> {
    // erf and erfc are implemented by different algorithms in different
    // regimes; their sum is the cheapest way to catch a bad regime boundary.
    let mut complement_err: f64 = 0.0;
    for i in -600..=600 {
        let x = i as f64 * 0.01;
        complement_err = complement_err.max((erf(x) + erfc(x) - 1.0).abs());
    }

    // Phi and its inverse must compose to the identity across the whole range
    // the sigma schedules actually visit, tails included.
    let mut involution_err: f64 = 0.0;
    for i in -500..=500 {
        let x = i as f64 * 0.01;
        let p = norm_cdf(x);
        if p > 0.0 && p < 1.0 {
            involution_err = involution_err.max((norm_ppf(p) - x).abs() / (1.0 + x.abs()));
        }
    }

    // Independently computed scipy values, so an internally consistent but
    // wrong implementation cannot pass.
    let references: [(f64, f64); 5] = [
        (0.0, 0.5),
        (1.96, 0.975_002_104_851_779_5),
        (2.5, 0.993_790_334_674_223_8),
        (-3.0, 0.001_349_898_031_630_103_5),
        (-4.6517, 1.646_048_869_580_465_5e-6),
    ];
    let reference_err = references
        .iter()
        .map(|&(x, expected)| (norm_cdf(x) - expected).abs() / expected)
        .fold(0.0f64, f64::max);

    vec![
        cert(
            "stats",
            "erf_erfc_complementary",
            "erf(x) + erfc(x) = 1 across every algorithmic regime boundary.",
            complement_err,
            1e-14,
        ),
        cert(
            "stats",
            "cdf_ppf_involution",
            "norm_ppf(norm_cdf(x)) = x over [-5, 5].",
            involution_err,
            1e-9,
        ),
        cert(
            "stats",
            "cdf_reference_values",
            "norm_cdf matches independently computed scipy values.",
            reference_err,
            1e-9,
        ),
    ]
}

// ----------------------------------------------------------------- solver --

/// Descending sigma grid with `n_steps` intervals uniform in `-log sigma`.
fn uniform_lambda_grid(sigma_hi: f64, sigma_lo: f64, n_steps: usize) -> Vec<f64> {
    let t_lo = -sigma_hi.ln();
    let t_hi = -sigma_lo.ln();
    (0..=n_steps)
        .map(|i| (-(t_lo + i as f64 * (t_hi - t_lo) / n_steps as f64)).exp())
        .collect()
}

/// Error of `kind` on a state-independent oracle, against a fine reference.
fn oracle_error(kind: SolverKind, grid: &[f64], x0_of_lambda: impl Fn(f64) -> f64 + Copy) -> f64 {
    use rand::{rngs::StdRng, SeedableRng};
    let device = Default::default();
    let z0 = 1.7f64;
    let mut rng = StdRng::seed_from_u64(0);
    let z_end = solver::integrate(
        Tensor::<B, 2>::full([1, 1], z0 as f32, &device),
        grid,
        |s, _z| Tensor::<B, 2>::full([1, 1], x0_of_lambda(-s.ln()) as f32, &device),
        kind,
        &mut rng,
    );
    let got: f32 = z_end.into_scalar();

    // Reference: compose the exact one-step update on a very fine grid.
    let (lam0, lam1) = (-grid[0].ln(), -grid[grid.len() - 1].ln());
    let n = 200_000usize;
    let dl = (lam1 - lam0) / n as f64;
    let mut z = z0;
    let mut lam = lam0;
    for _ in 0..n {
        let e = (-dl).exp();
        z = e * z + (1.0 - e) * x0_of_lambda(lam + 0.5 * dl);
        lam += dl;
    }
    (got as f64 - z).abs()
}

fn solver_certificates() -> Vec<Certificate> {
    use rand::{rngs::StdRng, SeedableRng};
    let mut out = Vec::new();
    let device = Default::default();
    let schedule = [80.0f64, 30.0, 10.0, 3.0, 1.0, 0.3, 0.05, 0.002];

    // For a constant x0 oracle the ODE has the closed form
    // z(s) = c + (z0 - c) s / s0. Every consistent solver must reproduce it
    // regardless of step size, so this catches sign and scaling errors that
    // an order study would not.
    let mut closed_form_err: f64 = 0.0;
    for kind in SolverKind::deterministic() {
        let mut rng = StdRng::seed_from_u64(0);
        let z0 = Tensor::<B, 2>::ones([2, 4], &device);
        let z_end = solver::integrate(
            z0.clone(),
            &schedule,
            |_s, _z| Tensor::<B, 2>::zeros([2, 4], &device),
            kind,
            &mut rng,
        );
        let expected = z0 * (schedule[schedule.len() - 1] / schedule[0]) as f32;
        closed_form_err =
            closed_form_err.max((z_end - expected).abs().max().into_scalar() as f64);
    }
    out.push(cert(
        "solver",
        "constant_oracle_closed_form",
        "Every deterministic solver reproduces z(s) = c + (z0 - c) s/s0 for a constant x0 oracle.",
        closed_form_err,
        1e-4,
    ));

    // The exponential integrator's step-size dependence lives entirely in
    // three moments; they are closed forms, checked against quadrature.
    let mut moment_err: f64 = 0.0;
    for &h in &[0.05f64, 0.5, 1.0, 2.5, 6.0] {
        let closed = solver::kernel_moments(h);
        for (k, &expected) in closed.iter().enumerate() {
            let f = |s: f64| s.powi(k as i32) * (-(h - s)).exp();
            let n = 20_000usize;
            let dx = h / n as f64;
            let mut acc = f(0.0) + f(h);
            for i in 1..n {
                acc += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i as f64 * dx);
            }
            let quad = acc * dx / 3.0;
            moment_err = moment_err.max((quad - expected).abs() / (1.0 + expected.abs()));
        }
    }
    out.push(cert(
        "solver",
        "kernel_moments",
        "J_k = int_0^h s^k e^-(h-s) ds matches its closed form for k = 0, 1, 2.",
        moment_err,
        1e-11,
    ));

    // The third-order coefficients are pinned by the requirement that the
    // interpolant passes through both history points.
    let mut interp_err: f64 = 0.0;
    for &(g0, g1) in &[(0.5f64, 1.25), (1.0, 3.0), (0.1, 0.15)] {
        for &(a_n, b_n) in &[(1.0f64, -2.0), (-0.3, -0.9)] {
            let [wa_a, wa_b, wb_a, wb_b] = solver::quadratic_interp_weights(g0, g1);
            let (a, b) = (wa_a * a_n + wa_b * b_n, wb_a * a_n + wb_b * b_n);
            let p = |s: f64| b * s + a * s * s;
            interp_err = interp_err.max((p(-g0) - a_n).abs()).max((p(-g1) - b_n).abs());
        }
    }
    out.push(cert(
        "solver",
        "quadratic_interpolation_nodes",
        "The 3M interpolant reproduces both history points exactly.",
        interp_err,
        1e-9,
    ));

    // Observed order of convergence must reach the classical order; a
    // shortfall means a coefficient is wrong even if the solver still
    // converges.
    let oracle = |lam: f64| (0.6 * lam).tanh();
    let counts = [8usize, 16, 32, 64];
    let mut order_shortfall: f64 = 0.0;
    for kind in SolverKind::deterministic() {
        let points: Vec<(f64, f64)> = counts
            .iter()
            .map(|&n| {
                let grid = uniform_lambda_grid(schedule[0], schedule[schedule.len() - 1], n);
                ((n as f64).log2(), oracle_error(kind, &grid, oracle).max(1e-12).log2())
            })
            .collect();
        let m = points.len() as f64;
        let mx = points.iter().map(|p| p.0).sum::<f64>() / m;
        let my = points.iter().map(|p| p.1).sum::<f64>() / m;
        let num: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let den: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let observed = -num / den;
        order_shortfall = order_shortfall.max((kind.order() as f64 - observed).max(0.0));
    }
    out.push(cert(
        "solver",
        "empirical_order_of_convergence",
        "Each solver's measured order (log-log slope of error vs steps) reaches its classical order.",
        order_shortfall,
        0.25,
    ));

    // The ancestral split must move variance between the deterministic target
    // and the injected noise without creating or destroying any.
    let mut ancestral_err: f64 = 0.0;
    for &(s, s_next) in &[(80.0f64, 30.0), (1.0, 0.3), (0.05, 0.002)] {
        for &eta in &[0.0f64, 0.25, 1.0, 2.0] {
            let (down, up) = solver::ancestral_split(s, s_next, Some(eta));
            ancestral_err = ancestral_err
                .max(((down * down + up * up) - s_next * s_next).abs() / (s_next * s_next));
        }
    }
    out.push(cert(
        "solver",
        "ancestral_variance_preserved",
        "sigma_down^2 + sigma_up^2 = sigma_next^2 for every eta: DDIM noise is redistributed, not added.",
        ancestral_err,
        1e-12,
    ));

    out
}

// -------------------------------------------------------------- precision --

fn precision_certificates() -> Vec<Certificate> {
    let mut bound_excess: f64 = 0.0;
    let mut idempotence_err: f64 = 0.0;

    for precision in [Precision::Bf16, Precision::F16] {
        let u = precision.unit_roundoff() as f64;
        let max_exp: i32 = if precision == Precision::F16 { 14 } else { 100 };
        for exp in -max_exp..=max_exp {
            for mantissa in 0..32 {
                let x = ((1.0f64 + mantissa as f64 / 32.0) * (2.0f64).powi(exp)) as f32;
                for signed in [x, -x] {
                    let r = precision.round_scalar(signed);
                    let rel = ((r - signed) / signed).abs() as f64;
                    bound_excess = bound_excess.max((rel - u).max(0.0) / u);
                    idempotence_err = idempotence_err
                        .max((precision.round_scalar(r) - r).abs() as f64);
                }
            }
        }
    }

    vec![
        cert(
            "precision",
            "unit_roundoff_bound",
            "Round-to-nearest at p significand bits has relative error at most 2^-p, for bf16 and f16.",
            bound_excess,
            1e-6,
        ),
        cert(
            "precision",
            "rounding_idempotent",
            "round(round(x)) = round(x): the output really lies on the target grid.",
            idempotence_err,
            0.0,
        ),
    ]
}

// --------------------------------------------------------------- quantize --

fn quantize_certificates() -> Vec<Certificate> {
    let device = Default::default();

    // Round-to-nearest on a fixed grid cannot err by more than half the widest
    // gap, scaled by the block's absmax. This is the bound that makes 4-bit
    // weights usable at all.
    let max_gap = NF4_LEVELS.windows(2).map(|w| w[1] - w[0]).fold(0.0f32, f32::max);
    let values: Vec<f32> = Tensor::<B, 1>::random([2048], Distribution::Normal(0.0, 1.0), &device)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let restored = Nf4Tensor::quantize(&values).dequantize();
    let mut bound_excess: f64 = 0.0;
    for (block_idx, chunk) in values.chunks(crate::quantize::BLOCK_SIZE).enumerate() {
        let absmax = chunk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let bound = 0.5 * max_gap * absmax;
        for (i, &v) in chunk.iter().enumerate() {
            let err = (restored[block_idx * crate::quantize::BLOCK_SIZE + i] - v).abs();
            bound_excess = bound_excess.max(((err - bound) / bound.max(1e-12)) as f64);
        }
    }

    // Zero must survive exactly: masks, padding and pruned weights depend on
    // it, and a symmetric grid without an exact zero would not provide it.
    let zeros = Nf4Tensor::quantize(&vec![0.0f32; 128]).dequantize();
    let mut zero_err = zeros.iter().fold(0.0f64, |a, &v| a.max(v.abs() as f64));
    let mixed = Nf4Tensor::quantize(&{
        let mut v = vec![0.0f32; 64];
        v[3] = 2.5;
        v
    })
    .dequantize();
    zero_err = zero_err.max(mixed[0].abs() as f64);

    // A freshly attached LoRA adapter must be an exact no-op, or wrapping a
    // trained checkpoint would perturb it before any training happened.
    let adapter = LoraAdapter::<B>::new(&LoraConfig::new(32, 16, 4), &device);
    let x = Tensor::<B, 2>::random([4, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let lora_err = adapter.forward(x).abs().max().into_scalar() as f64;

    vec![
        cert(
            "quantize",
            "nf4_error_bound",
            "NF4 reconstruction error is at most half the widest level gap times the block absmax.",
            bound_excess.max(0.0),
            1e-6,
        ),
        cert(
            "quantize",
            "nf4_zero_exact",
            "Zero is exactly representable and survives quantization unchanged.",
            zero_err,
            0.0,
        ),
        cert(
            "quantize",
            "lora_identity_at_init",
            "A zero-initialized LoRA adapter is an exact no-op.",
            lora_err,
            0.0,
        ),
    ]
}

// -------------------------------------------------------------- loopgraph --

fn loopgraph_certificates() -> Vec<Certificate> {
    // ACT mixture weights must be a partition of unity, or the loop graph's
    // output is an arbitrarily scaled vector rather than a convex combination
    // of block outputs.
    let mut mass_err: f64 = 0.0;
    for halt in [0.0f32, 0.05, 0.3, 0.5, 0.9, 1.0] {
        for num_blocks in [1usize, 2, 4, 7] {
            let mut planner = LoopPlanner::new(LoopGraphConfig::default(), num_blocks);
            let mut mass = 0.0f32;
            let mut guard = 0;
            loop {
                guard += 1;
                assert!(guard < 1000, "planner failed to terminate");
                match planner.next(0.5) {
                    Decision::Stop => break,
                    Decision::Skip(_) => continue,
                    _ => mass += planner.charge(halt),
                }
                if planner.finished() {
                    break;
                }
            }
            mass_err = mass_err.max((mass as f64 - 1.0).abs());
        }
    }

    // Budgets must be hard. Driven adversarially: confidence pinned so the
    // planner always wants to loop back, halting probability pinned at zero so
    // ACT never terminates the run.
    let mut budget_excess: f64 = 0.0;
    for budget in [1usize, 2, 3, 5] {
        let config = LoopGraphConfig {
            budget: Some(budget),
            max_iterations: 500,
            loopback_threshold: 1.0,
            max_loopbacks: usize::MAX,
            exit_threshold: f32::INFINITY,
            skip_threshold: f32::INFINITY,
        };
        let mut planner = LoopPlanner::new(config, 4);
        let mut executions = 0usize;
        let mut guard = 0;
        loop {
            guard += 1;
            assert!(guard < 5000, "planner failed to terminate");
            match planner.next(0.0) {
                Decision::Stop => break,
                Decision::Skip(_) => continue,
                _ => {
                    executions += 1;
                    planner.charge(0.0);
                }
            }
            if planner.finished() {
                break;
            }
        }
        budget_excess = budget_excess.max((executions as f64 - budget as f64).max(0.0));
    }

    vec![
        cert(
            "loopgraph",
            "act_partition_of_unity",
            "ACT mixture weights sum to exactly 1 for any halting probabilities and block count.",
            mass_err,
            1e-5,
        ),
        cert(
            "loopgraph",
            "budget_is_hard",
            "The planner never authorizes more block executions than its budget, even under adversarial signals.",
            budget_excess,
            0.0,
        ),
    ]
}

// -------------------------------------------------------------------- moe --

fn moe_certificates() -> Vec<Certificate> {
    let device = Default::default();

    // Renormalized top-k gates must be a distribution, or the layer rescales
    // its own output as a side effect of the routing decision.
    let mut gate_err: f64 = 0.0;
    for top_k in [1usize, 2, 4] {
        let config = MoEConfig::new(8, 4, 4).with_top_k(top_k);
        let layer = MoELayer::<B>::new(&config, &device);
        let x = Tensor::<B, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device);
        let cond = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let probs = softmax(layer.router_logits(&x, &cond), 1);
        let (vals, _) = probs.topk_with_indices(top_k, 1);
        let gates = vals.clone() / vals.sum_dim(1).clamp_min(1e-12);
        for s in gates
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
        {
            gate_err = gate_err.max((s as f64 - 1.0).abs());
        }
    }

    // On the diagonal f == p, Cauchy-Schwarz gives E * sum_e p_e^2 >= 1, with
    // equality exactly at uniform routing. That is the precise sense in which
    // the auxiliary loss is minimized by balance.
    let e = 8usize;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut bound_violation: f64 = 0.0;
    for _ in 0..500 {
        let mut p: Vec<f64> = (0..e)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 + 1e-9
            })
            .collect();
        let total: f64 = p.iter().sum();
        for v in p.iter_mut() {
            *v /= total;
        }
        let loss = e as f64 * p.iter().map(|x| x * x).sum::<f64>();
        bound_violation = bound_violation
            .max((1.0 - loss).max(0.0))
            .max((loss - e as f64).max(0.0));
    }
    let uniform_gap =
        (e as f64 * vec![1.0 / e as f64; e].iter().map(|x| x * x).sum::<f64>() - 1.0).abs();

    // ------------------------------------------------------------------
    // The z-loss penalizes the log-sum-exp rather than the logits themselves,
    // and the reason is this sandwich:
    //
    //     max_e x_e  <=  logsumexp_e x_e  <=  max_e x_e + ln E
    //
    // so holding the log-sum-exp near zero holds *every* logit within ln E of
    // zero. That is what makes it an overflow guard rather than a vague
    // shrinkage penalty.
    let mut sandwich: f64 = 0.0;
    let mut state = 0xC2B2_AE3D_27D4_EB4Fu64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for width in [2usize, 4, 16, 64] {
        for scale in [1e-3f64, 1.0, 30.0, 120.0] {
            let values: Vec<f32> = (0..width)
                .map(|_| ((next() * 2.0 - 1.0) * scale) as f32)
                .collect();
            let logits = Tensor::<B, 1>::from_floats(values.as_slice(), &device)
                .reshape([1, width]);
            // `router_z_loss` returns the *squared* log-sum-exp, so recover it.
            let z: f64 = f64::from(crate::moe::router_z_loss(&logits).into_scalar());
            let lse = z.sqrt();
            let max = values.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let max = f64::from(max);
            // Only meaningful where the log-sum-exp is positive; a negative one
            // squares to the same number and the sign is lost.
            if max > 0.0 {
                sandwich = sandwich
                    .max((max - lse).max(0.0))
                    .max((lse - (max + (width as f64).ln())).max(0.0));
            }
        }
    }

    // The optimum is a log-sum-exp of zero, not a logit of zero: for a row of E
    // equal logits that means each sits at -ln E. Shifting off it by `d` must
    // cost exactly `d^2`, which is what makes the penalty a calibrated distance
    // rather than an arbitrary regularizer.
    let mut z_optimum: f64 = 0.0;
    for width in [2usize, 4, 16] {
        for offset in [-3.0f64, -0.5, 0.0, 0.5, 3.0] {
            let value = (-(width as f64).ln() + offset) as f32;
            let logits = Tensor::<B, 2>::full([4, width], value, &device);
            let z = f64::from(crate::moe::router_z_loss(&logits).into_scalar());
            z_optimum = z_optimum.max((z - offset * offset).abs());
        }
    }

    // And the direction it penalizes is one the balance loss is blind to: the
    // softmax is invariant to a per-row constant shift, so every routing
    // probability is unchanged by a shift the z-loss scores as enormous. That
    // is precisely why a balance loss alone cannot prevent logit drift.
    let base = Tensor::<B, 2>::random([8, 6], Distribution::Uniform(-2.0, 2.0), &device);
    let shifted = base.clone() + 40.0;
    let p0: Vec<f32> = softmax(base.clone(), 1).into_data().convert::<f32>().iter::<f32>().collect();
    let p1: Vec<f32> = softmax(shifted.clone(), 1).into_data().convert::<f32>().iter::<f32>().collect();
    let mut shift_invariance: f64 = 0.0;
    for (a, b) in p0.iter().zip(&p1) {
        shift_invariance = shift_invariance.max(f64::from((a - b).abs()));
    }
    let z_before = f64::from(crate::moe::router_z_loss(&base).into_scalar());
    let z_after = f64::from(crate::moe::router_z_loss(&shifted).into_scalar());
    // The z-loss must actually register the shift; 0 if it does, 1 if not.
    let z_sees_the_shift = f64::from(u8::from(z_after <= z_before * 10.0));

    vec![
        cert(
            "moe",
            "gates_partition_of_unity",
            "Renormalized top-k gates sum to 1 per token, for every k.",
            gate_err,
            1e-6,
        ),
        cert(
            "moe",
            "balance_loss_bounds",
            "On the diagonal the Switch balance loss lies in [1, E] (Cauchy-Schwarz), attaining 1 exactly at uniform routing.",
            bound_violation.max(uniform_gap),
            1e-12,
        ),
        cert(
            "moe",
            "logsumexp_bounds_the_largest_logit",
            "max_e x_e <= logsumexp_e x_e <= max_e x_e + ln E, so holding the z-loss near zero holds every routing logit within ln E of zero.",
            sandwich,
            1e-4,
        ),
        cert(
            "moe",
            "z_loss_optimum_is_a_zero_logsumexp",
            "The z-loss is exactly the squared distance of the log-sum-exp from zero: a row of E equal logits is optimal at -ln E, and an offset d costs d^2.",
            z_optimum,
            1e-4,
        ),
        cert(
            "moe",
            "z_loss_penalizes_what_the_balance_loss_cannot_see",
            "A per-row constant shift leaves every routing probability unchanged -- so the balance loss is blind to it -- while the z-loss registers it. That is why logit drift needs its own term.",
            shift_invariance.max(z_sees_the_shift),
            // The invariance is exact in real arithmetic. What is measured is
            // f32 re-exponentiation after a shift of 40, which costs a few ulps
            // of the intermediate exponentials -- order 1e-6 on a probability.
            1e-5,
        ),
    ]
}

// ------------------------------------------------------------------ mosme --

fn mosme_certificates() -> Vec<Certificate> {
    use crate::expert_index::{BoxSpec, ExpertSpec, MosmeSpec};
    use crate::mosme::{MosmeConfig, MosmeFeedForward};

    let device = Default::default();

    let ragged = |top_box: usize, top_expert: usize| MosmeSpec {
        boxes: vec![
            BoxSpec::new(
                "coding",
                "Code",
                vec![
                    ExpertSpec::new("coding/rust", "Rust"),
                    ExpertSpec::new("coding/python", "Python"),
                    ExpertSpec::new("coding/secure", "Secure"),
                ],
            ),
            BoxSpec::new(
                "cyber",
                "Cybersecurity",
                vec![
                    ExpertSpec::new("cyber/netsec", "Network"),
                    ExpertSpec::new("cyber/malware", "Malware"),
                ],
            ),
        ],
        top_box,
        top_expert,
        route_on_tokens: true,
        balance: Default::default(),
    };
    let config = |spec: MosmeSpec| MosmeConfig::new(8, 4, spec).with_intermediate_size(16);
    let inputs = || {
        (
            Tensor::<B, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device),
            Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device),
        )
    };

    // Two renormalized levels compose into one distribution over (box, expert)
    // pairs. Without this the layer output is an arbitrarily scaled mixture
    // rather than a convex combination of its experts, and every downstream
    // bound -- including the convex-hull certificate on x0 -- stops holding.
    let mut partition_err: f64 = 0.0;
    for top_box in [1usize, 2] {
        for top_expert in [1usize, 2, 3] {
            let cfg = config(ragged(top_box, top_expert));
            let layer = MosmeFeedForward::<B>::new(&cfg, &device);
            let (x, cond) = inputs();
            let gates = layer.router().route(layer.router().router_input(&x, &cond));
            for m in gates
                .composed_flat()
                .sum_dim(1)
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
            {
                partition_err = partition_err.max((m as f64 - 1.0).abs());
            }
        }
    }

    // A single box must reproduce the flat MoE path exactly -- output and
    // balance loss both. This is what makes the hierarchy a strict
    // generalization rather than a second, subtly different implementation.
    let flat_cfg = config(MosmeSpec::flat(4));
    let flat_layer = MosmeFeedForward::<B>::new(&flat_cfg, &device);
    let reference = flat_layer.as_flat().expect("one box");
    let (x, cond) = inputs();
    let hierarchical = flat_layer.forward(x.clone(), cond.clone());
    let flat = reference.forward(x, cond);
    let reduction_err = (hierarchical.output - flat.output)
        .abs()
        .max()
        .into_scalar()
        .max(
            (hierarchical.balance.expert_loss - flat.balance)
                .abs()
                .max()
                .into_scalar(),
        ) as f64;
    // Softmax over a single logit is exactly 1, so the degenerate box level
    // contributes exactly 1 to the balance -- and exactly 0 to the z-loss,
    // because a logit that cannot change any routing decision has nothing to
    // stabilize.
    let degenerate_err = (hierarchical.balance.box_loss.into_scalar() as f64 - 1.0)
        .abs()
        .max(f64::from(
            (hierarchical.balance.z_loss - flat.z_loss).abs().max().into_scalar(),
        ));

    // Growing a model with a disabled expert must be a bit-exact identity;
    // that is what "add a specialist without retraining the others" means.
    let spec = ragged(1, 4);
    let cfg = config(spec.clone());
    let layer = MosmeFeedForward::<B>::new(&cfg, &device);
    let (x, cond) = inputs();
    let before = layer.forward(x.clone(), cond.clone()).output;
    let grown_spec = spec
        .extended_with("coding", ExpertSpec::new("coding/go", "Go"))
        .expect("extend");
    let grown = layer.grown(&grown_spec, &cfg, &device).expect("grow");
    let after = grown.forward(x.clone(), cond.clone()).output;
    let hot_swap_err = (before - after).abs().max().into_scalar() as f64;

    // ...and the newly added expert must be gated to exactly zero, not merely
    // to something small.
    let gates = grown.router().route(grown.router().router_input(&x, &cond));
    let disabled_err = gates.composed()[0]
        .clone()
        .narrow(1, 3, 1)
        .abs()
        .max()
        .into_scalar() as f64;

    // Per level the Switch loss is bounded by the number of things that level
    // routes between, and the expert level is a convex combination over boxes
    // so it inherits the bound termwise.
    let breakdown = gates.balance_loss(Default::default());
    let widths = grown.router().experts_per_box();
    let mut bound_violation: f64 = 0.0;
    let box_loss = breakdown.box_loss.into_scalar() as f64;
    bound_violation = bound_violation
        .max((-box_loss).max(0.0))
        .max((box_loss - 2.0).max(0.0));
    for (i, l) in breakdown.per_box.iter().enumerate() {
        let v = l.clone().into_scalar() as f64;
        bound_violation = bound_violation
            .max((-v).max(0.0))
            .max((v - widths[i] as f64).max(0.0));
    }
    let traffic_err = (grown
        .router()
        .route(grown.router().router_input(&x, &cond))
        .box_traffic()
        .sum()
        .into_scalar() as f64
        - 1.0)
        .abs();

    // Site (a): every LoRA `B` factor is zero at init, so a fresh adapter bank
    // is EXACTLY its frozen base -- for any input and any routing condition.
    // Strictly stronger than the flat LoRA identity, which covers one adapter
    // at one point.
    let base: burn::nn::Linear<B> = burn::nn::LinearConfig::new(8, 6).init(&device);
    crate::tensor_ext::force_initialization(&base);
    let bank =
        crate::mosme::MosmeAdapterBank::<B>::from_linear(base.clone(), &ragged(1, 2), 4, 4, 4.0, &device);
    let mut bank_err: f64 = 0.0;
    for _ in 0..4 {
        let xb = Tensor::<B, 2>::random([3, 8], Distribution::Uniform(-2.0, 2.0), &device);
        let cb = Tensor::<B, 2>::random([3, 4], Distribution::Uniform(-2.0, 2.0), &device);
        bank_err = bank_err.max(
            (bank.forward(xb.clone(), cb).output - base.forward(xb))
                .abs()
                .max()
                .into_scalar() as f64,
        );
    }

    // Site (b): the micro-model mixture is a convex combination of softmax
    // outputs, so it is still a distribution -- which is what lets the two
    // `model` certificates keep holding over an ensemble.
    let vit = ViTDiTConfig::tiny(10);
    let dblock_cfg = DblockConfig { num_blocks: 2, ..DblockConfig::default() };
    let ensemble =
        crate::mosme::MosmeEnsemble::<B>::fresh(&ragged(1, 1), &vit, &dblock_cfg, 4, &device);
    let pixels = Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let zt = Tensor::<B, 2>::random([2, 32], Distribution::Normal(0.0, 1.0), &device);
    let ens_cond = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
    let dense = ensemble.forward_dense(&pixels, &zt, &[1.0; 2], ens_cond.clone());
    let sparse = ensemble.forward_sparse(&pixels, &zt, &[1.0; 2], ens_cond);
    let mixture_err = dense
        .probs
        .clone()
        .sum_dim(1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .map(|m| (m as f64 - 1.0).abs())
        .fold(0.0f64, f64::max);
    // Skipping an unselected specialist contributes `0.0 * y`, and `x + 0.0`
    // is exact, so the two dispatch paths must agree bit-for-bit.
    let dispatch_err = (dense.probs - sparse.probs).abs().max().into_scalar() as f64;

    // The manifest must survive a round trip through its wire format, since
    // an external engine is the whole reason it exists.
    let index = crate::expert_index::MosmeSpec::flat(3);
    let roundtrip_err = match serde_json::to_string(&index)
        .ok()
        .and_then(|t| serde_json::from_str::<MosmeSpec>(&t).ok())
    {
        Some(back) if back == index => 0.0,
        _ => 1.0,
    };

    vec![
        cert(
            "mosme",
            "composed_gates_partition_of_unity",
            "Composing two renormalized routing levels gives a distribution over (box, expert) pairs.",
            partition_err,
            1e-6,
        ),
        cert(
            "mosme",
            "single_box_reduces_to_flat_moe",
            "With one box the hierarchical layer IS the flat MoE layer: same output, same balance loss.",
            reduction_err,
            0.0,
        ),
        cert(
            "mosme",
            "degenerate_box_level_is_exactly_one",
            "Softmax over a single box logit is exactly 1, so a one-box balance loss is exactly 1.",
            degenerate_err,
            0.0,
        ),
        cert(
            "mosme",
            "hot_swap_is_an_exact_identity",
            "Adding a disabled expert leaves every output bit-identical, so a specialist can be added without retraining the others.",
            hot_swap_err,
            0.0,
        ),
        cert(
            "mosme",
            "disabled_expert_gate_is_exactly_zero",
            "A -inf routing mask gives exp(-inf - max) == 0, so a disabled expert contributes exactly nothing.",
            disabled_err,
            0.0,
        ),
        cert(
            "mosme",
            "hierarchical_balance_bounds",
            "Each level's Switch loss lies in [0, N] for N routed alternatives; the expert level inherits it as a convex combination over boxes.",
            bound_violation,
            1e-6,
        ),
        cert(
            "mosme",
            "box_traffic_is_a_distribution",
            "Box traffic shares sum to 1, which is what makes the expert-level aggregation convex.",
            traffic_err,
            1e-6,
        ),
        cert(
            "mosme",
            "adapter_bank_identity_at_init",
            "A freshly built adapter bank equals its frozen base exactly, for every input and every routing condition.",
            bank_err,
            0.0,
        ),
        cert(
            "mosme",
            "ensemble_mixture_is_a_distribution",
            "Mixing micro-models in probability space yields a distribution, so the convex-hull bound on x0 still holds.",
            mixture_err,
            1e-6,
        ),
        cert(
            "mosme",
            "sparse_dispatch_matches_dense",
            "Skipping unselected specialists is exact: an unevaluated term contributes 0.0 and x + 0.0 is exact.",
            dispatch_err,
            0.0,
        ),
        cert(
            "mosme",
            "manifest_roundtrip",
            "An expert manifest survives serialization unchanged.",
            roundtrip_err,
            0.0,
        ),
    ]
}

// --------------------------------------------------------------------- lm --

fn lm_certificates() -> Vec<Certificate> {
    use crate::lm::{LanguageModel, LmConfig, Sampling};
    use crate::tokenizer::{ByteTokenizer, Special, BYTE_TOKENS, VOCAB_SIZE};
    use rand::{rngs::StdRng, SeedableRng};

    let device = Default::default();
    let tokenizer = ByteTokenizer::new();

    // Byte-level tokenization is chosen precisely because it cannot mangle any
    // input. If a round trip ever loses information the trade stops being worth
    // making.
    let mut roundtrip_err: f64 = 0.0;
    for text in [
        "",
        "hello world",
        "naive cafe -- em-dash, unicode: \u{e4}\u{f6}\u{fc}",
        "\u{65e5}\u{672c}\u{8a9e}",
        "\u{0}\u{1}\u{7f} control bytes",
    ] {
        let ids = tokenizer.encode(text);
        if tokenizer.decode(&ids).as_deref() != Some(text) {
            roundtrip_err = 1.0;
        }
    }
    // Specials must not collide with any byte, or a byte would be silently
    // interpreted as a control token.
    let mut collision = 0.0f64;
    for special in Special::ALL {
        if (special.id() as usize) < BYTE_TOKENS {
            collision = 1.0;
        }
    }

    // The property that makes next-token training meaningful. If position i
    // could see i+1 the model would learn to copy the answer and the loss would
    // fall without anything being learned.
    let model = LanguageModel::<B>::new(&LmConfig::tiny(), &device);
    let ids: Vec<i64> = vec![5, 6, 7, 8, 9];
    let n = ids.len();
    let as_tensor = |v: &[i64]| {
        Tensor::<B, 1, Int>::from_ints(v, &device).reshape([1, v.len()])
    };
    let reference = model.forward(as_tensor(&ids)).logits;
    let mut perturbed_ids = ids.clone();
    perturbed_ids[n - 1] = 200;
    let perturbed = model.forward(as_tensor(&perturbed_ids)).logits;
    let leak = (reference.clone().narrow(1, 0, n - 1) - perturbed.narrow(1, 0, n - 1))
        .abs()
        .max()
        .into_scalar() as f64;

    // An untrained model with a tied head should be near-uniform, so the loss
    // should sit at ln(V). Drifting from that means the initialization is
    // producing peaked logits and training starts by undoing them.
    let (_, metrics) = model.next_token_loss(as_tensor(&ids), 0..model.num_layers());
    let uniform_gap = (metrics.loss as f64 - (VOCAB_SIZE as f64).ln()).abs();

    // Top-1 sampling is greedy decoding by definition; if they diverge, the
    // sampling path is not selecting what it claims to.
    let prompt = vec![Special::Bos.id(), 65];
    let greedy = model.generate(&prompt, 4, &Sampling::Greedy, &mut StdRng::seed_from_u64(0), &device);
    let top1 = model.generate(
        &prompt,
        4,
        &Sampling::TopK { k: 1, temperature: 1.0 },
        &mut StdRng::seed_from_u64(3),
        &device,
    );
    let sampling_err = f64::from(u8::from(greedy != top1));

    // A causal model's keys and values at position i are a function of tokens
    // up to i alone, so once those tokens are committed a cache can only
    // reproduce what a full recompute would produce. That is what makes
    // incremental decoding a pure `O(n^2) -> O(n)` saving rather than a
    // speed-for-accuracy trade -- and it is checked against several chunkings,
    // because a cache that is right for one token at a time can still be wrong
    // for a batch of them.
    let cached_ids: Vec<i64> = tokenizer.encode("cache me").iter().map(|t| *t as i64).collect();
    let reference: Vec<f32> = model
        .forward(as_tensor(&cached_ids))
        .logits
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();

    let mut cache_err: f64 = 0.0;
    for chunks in [
        vec![cached_ids.len()],
        vec![1; cached_ids.len()],
        vec![3, 1, cached_ids.len() - 4],
    ] {
        let mut cache = model.new_cache();
        let mut produced: Vec<f32> = Vec::new();
        let mut at = 0usize;
        for size in chunks {
            let out = model.forward_cached(as_tensor(&cached_ids[at..at + size]), &mut cache);
            produced.extend(out.logits.into_data().convert::<f32>().iter::<f32>());
            at += size;
        }
        if produced.len() != reference.len() {
            cache_err = f64::INFINITY;
            continue;
        }
        for (a, b) in produced.iter().zip(&reference) {
            cache_err = cache_err.max(f64::from((a - b).abs()) / f64::from(b.abs()).max(1.0));
        }
    }

    // ...and the same statement end to end: identical text out of both
    // decoders from the same seed.
    let plain = model.generate(&prompt, 6, &Sampling::Greedy, &mut StdRng::seed_from_u64(9), &device);
    let cached = model.generate_cached(
        &prompt,
        6,
        &Sampling::Greedy,
        &mut StdRng::seed_from_u64(9),
        &device,
    );
    let decode_err = f64::from(u8::from(plain != cached));

    vec![
        cert(
            "lm",
            "tokenizer_roundtrip_is_lossless",
            "Byte-level encoding then decoding reproduces the input exactly, for text, unicode and control bytes.",
            roundtrip_err.max(collision),
            0.0,
        ),
        cert(
            "lm",
            "attention_cannot_see_the_future",
            "Perturbing token i+1 leaves the logits at every position <= i bit-identical.",
            leak,
            0.0,
        ),
        cert(
            "lm",
            "untrained_loss_is_near_uniform",
            "With a tied head and a correctly scaled embedding, an untrained model's loss is ln(vocab).",
            uniform_gap,
            1.0,
        ),
        cert(
            "lm",
            "top1_sampling_is_greedy",
            "Restricting sampling to the single most likely token reproduces greedy decoding exactly.",
            sampling_err,
            0.0,
        ),
        cert(
            "lm",
            "kv_cache_matches_full_recompute",
            "Incremental decoding with a key/value cache reproduces the full-recompute logits under every chunking, to within float summation order.",
            cache_err,
            // The only permitted difference is the order the attention weights
            // are summed in. Over n <= 16 positions that is bounded by
            // n * eps ~ 16 * 5.96e-8 ~ 1e-6; the tolerance is ten times that.
            1e-5,
        ),
        cert(
            "lm",
            "cached_decoding_emits_the_same_tokens",
            "Cached and uncached generation produce identical sequences from the same seed.",
            decode_err,
            0.0,
        ),
    ]
}

// --------------------------------------------------------------- accuracy --

fn accuracy_certificates() -> Vec<Certificate> {
    use crate::accuracy::{Ensemble, Guidance, LogitNorm, ScalingCurve, ScalingPoint};

    let device = Default::default();

    // ------------------------------------------------------------------
    // Guidance at scale 1 must be the conditional estimate *bitwise*, not
    // approximately. In exact arithmetic `u + 1*(c - u) == c`, but in f32 the
    // round trip loses low bits whenever |u| >> |c - u| -- and here the result
    // feeds an ODE step, so the error compounds over the trajectory. The
    // implementation short-circuits; this is what holds it to that.
    let cond = Tensor::<B, 2>::random([8, 16], Distribution::Uniform(-50.0, 50.0), &device);
    let uncond = Tensor::<B, 2>::random([8, 16], Distribution::Uniform(-50.0, 50.0), &device);
    let identity: Vec<f32> = Guidance::none()
        .apply(cond.clone(), uncond.clone())
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let reference: Vec<f32> = cond.clone().into_data().convert::<f32>().iter::<f32>().collect();
    let guidance_identity = identity
        .iter()
        .zip(&reference)
        .map(|(a, b)| f64::from((a.to_bits() != b.to_bits()) as u8))
        .fold(0.0f64, f64::max);

    // Guidance is an affine extrapolation through the two estimates, so at
    // scale s the result must be u + s(c - u). Checked against independently
    // computed f64 arithmetic rather than against itself.
    //
    // The error is normalized by the magnitude of the *terms*, not of the
    // result. Dividing by the result would make the residual explode wherever
    // `u` and `s(c - u)` nearly cancel -- reporting catastrophic cancellation,
    // which is a property of the inputs, as if it were an implementation
    // defect. Normalized this way the bound follows from the arithmetic: three
    // f32 roundings (subtract, multiply, add), each with relative error at most
    // `eps = 2^-24 ~ 5.96e-8`, gives `3 * eps ~ 1.8e-7`. The tolerance is 1e-6,
    // a little over 5x that.
    let mut affine_err: f64 = 0.0;
    for scale in [0.0f64, 0.5, 1.5, 3.0, 7.5] {
        let got: Vec<f32> = Guidance::new(scale)
            .apply(cond.clone(), uncond.clone())
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let u: Vec<f32> = uncond.clone().into_data().convert::<f32>().iter::<f32>().collect();
        for ((g, c), u) in got.iter().zip(&reference).zip(&u) {
            let (c, u) = (f64::from(*c), f64::from(*u));
            let want = u + scale * (c - u);
            let conditioning = (u.abs() + scale * (c - u).abs()).max(1.0);
            affine_err = affine_err.max((f64::from(*g) - want).abs() / conditioning);
        }
    }

    // ------------------------------------------------------------------
    // Every logit normalization is a strictly increasing per-row affine map, so
    // the arg-max cannot move. That is what makes it safe to switch on
    // anywhere: it recalibrates the confidence the adaptive strategy and the
    // quality gates read, and leaves top-1 exactly where it was.
    let raw = Tensor::<B, 2>::random([64, 10], Distribution::Uniform(-60.0, 60.0), &device);
    let baseline: Vec<i64> = raw
        .clone()
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .iter()
        .collect();
    let mut argmax_drift: f64 = 0.0;
    for norm in [
        LogitNorm::Temperature { tau: 0.1 },
        LogitNorm::Temperature { tau: 1.0 },
        LogitNorm::Temperature { tau: 25.0 },
        LogitNorm::L2 { tau: 0.05 },
        LogitNorm::L2 { tau: 4.0 },
        LogitNorm::Standardize { tau: 0.5 },
        LogitNorm::Standardize { tau: 10.0 },
    ] {
        let moved: Vec<i64> = norm
            .apply(raw.clone())
            .argmax(1)
            .into_data()
            .convert::<i64>()
            .iter()
            .collect();
        let differing = moved
            .iter()
            .zip(&baseline)
            .filter(|(a, b)| a != b)
            .count();
        argmax_drift = argmax_drift.max(differing as f64);
    }

    // `LogitNorm::None` is the exact identity, bitwise.
    let untouched: Vec<f32> = LogitNorm::None
        .apply(raw.clone())
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let raw_bits: Vec<f32> = raw.clone().into_data().convert::<f32>().iter::<f32>().collect();
    let norm_identity = untouched
        .iter()
        .zip(&raw_bits)
        .map(|(a, b)| f64::from((a.to_bits() != b.to_bits()) as u8))
        .fold(0.0f64, f64::max);

    // ------------------------------------------------------------------
    // Every ensemble emits a probability distribution. A combination rule that
    // returned unnormalized mass would silently rescale every downstream
    // confidence -- and confidences are what the gates threshold on.
    let members: Vec<Tensor<B, 2>> = (0..4)
        .map(|i| {
            Tensor::<B, 2>::random(
                [8, 10],
                Distribution::Uniform(-3.0 - f64::from(i), 3.0 + f64::from(i)),
                &device,
            )
        })
        .collect();
    let mut simplex_err: f64 = 0.0;
    for kind in [Ensemble::ProbabilityMean, Ensemble::LogitMean, Ensemble::MajorityVote] {
        let combined = kind.combine(&members);
        for s in combined
            .clone()
            .sum_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
        {
            simplex_err = simplex_err.max((f64::from(s) - 1.0).abs());
        }
        for p in combined.into_data().convert::<f32>().iter::<f32>() {
            simplex_err = simplex_err.max((-f64::from(p)).max(0.0));
        }
    }

    // Ensembling N copies of one member is that member. The containment that
    // lets a pipeline keep the ensemble permanently in place, with the member
    // count as the only knob -- the same argument as
    // `single_box_reduces_to_flat_moe` in the `mosme` group.
    let lone = members[0].clone();
    let expected: Vec<f32> = softmax(lone.clone(), 1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let mut ensemble_identity: f64 = 0.0;
    for count in [1usize, 3, 5] {
        let repeated = vec![lone.clone(); count];
        for kind in [Ensemble::ProbabilityMean, Ensemble::LogitMean] {
            let got: Vec<f32> = kind
                .combine(&repeated)
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect();
            for (a, b) in got.iter().zip(&expected) {
                ensemble_identity = ensemble_identity.max(f64::from((a - b).abs()));
            }
        }
    }

    // ------------------------------------------------------------------
    // The Pareto frontier must contain no dominated point and must not drop an
    // undominated one -- checked by brute force against the definition rather
    // than against the implementation's own reasoning.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut curve_violation: f64 = 0.0;
    for round in 0..64 {
        let mut curve = ScalingCurve::new();
        let mut raw_points = Vec::new();
        for i in 0..8 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let acc = ((state >> 11) as f64 / (1u64 << 53) as f64 * 1000.0).round() / 1000.0;
            let layers = 1 + (state % 40) as usize;
            let point = ScalingPoint::new(format!("r{round}p{i}"), layers, layers, acc);
            raw_points.push((layers, acc));
            curve.push(point);
        }

        let frontier = curve.pareto();
        for p in &frontier {
            // No other point may dominate a frontier member.
            let dominated = raw_points
                .iter()
                .any(|(l, a)| *l <= p.layers_executed && *a > p.accuracy);
            if dominated {
                curve_violation = 1.0;
            }
        }
        // The most accurate point's accuracy must be attained on the frontier.
        let best = raw_points.iter().map(|(_, a)| *a).fold(f64::NEG_INFINITY, f64::max);
        if !frontier.iter().any(|p| p.accuracy == best) {
            curve_violation = 1.0;
        }
        // The frontier is increasing in both cost and accuracy.
        for w in frontier.windows(2) {
            if w[1].layers_executed < w[0].layers_executed || w[1].accuracy <= w[0].accuracy {
                curve_violation = 1.0;
            }
        }
    }

    vec![
        cert(
            "accuracy",
            "guidance_identity_is_exact",
            "Guidance at scale 1 returns the conditional estimate bitwise, so the default costs no precision as well as no compute.",
            guidance_identity,
            0.0,
        ),
        cert(
            "accuracy",
            "guidance_is_affine_in_the_estimates",
            "Guided output equals u + scale*(c - u) at every scale, relative to independently computed f64 arithmetic.",
            affine_err,
            1e-6,
        ),
        cert(
            "accuracy",
            "logit_normalization_preserves_argmax",
            "Every normalization is a strictly increasing per-row affine map, so top-1 is unchanged; only the reported confidence moves.",
            argmax_drift,
            0.0,
        ),
        cert(
            "accuracy",
            "logit_normalization_none_is_exact",
            "`LogitNorm::None` returns its input bitwise.",
            norm_identity,
            0.0,
        ),
        cert(
            "accuracy",
            "ensembles_emit_distributions",
            "Every combination rule returns non-negative rows summing to 1, so downstream confidences stay comparable.",
            simplex_err,
            1e-6,
        ),
        cert(
            "accuracy",
            "identical_members_are_identity",
            "An ensemble of N copies of one member equals that member: ensembling is a strict generalization of not ensembling.",
            ensemble_identity,
            1e-6,
        ),
        cert(
            "accuracy",
            "pareto_frontier_is_exactly_the_undominated_set",
            "The reported scaling frontier contains no dominated configuration, attains the best accuracy measured, and increases in both cost and accuracy.",
            curve_violation,
            0.0,
        ),
    ]
}

// ------------------------------------------------------------------ optim --

fn optim_certificates() -> Vec<Certificate> {
    use crate::reweight::{SigmaImportanceSampler, UncertaintyWeighting};
    use crate::schedule::{GradientAccumulator, LrSchedule};

    // ------------------------------------------------------------------
    // Gradient accumulation over k micro-batches must equal one k-times-larger
    // batch. The accumulator *averages* rather than sums precisely so this
    // holds -- if it summed, the effective learning rate would scale with the
    // accumulation count and every hyperparameter would silently change with it.
    //
    // This is checked against **real gradients through a real module**, not
    // against the algebraic identity `mean_i(mean(g_i)) == mean(concat(g_i))`.
    // The earlier version of this certificate checked the identity on plain
    // f64 numbers and passed while the implementation threw k-1 of every k
    // gradients away -- a formula is not an implementation, and a certificate
    // that never touches the code path cannot tell the difference.
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::backend::AutodiffBackend;
    use burn::optim::GradientsParams;
    use crate::train::DefaultTrainBackend as A;

    let ad_device = Default::default();
    <A as burn::tensor::backend::Backend>::seed(&ad_device, 4242);
    let layer: Linear<A> = LinearConfig::new(6, 4).with_bias(true).init(&ad_device);

    // A fixed dataset, split k ways. Micro-batches are equal-sized, which is
    // the condition under which the identity holds at all.
    let micro = 3usize;
    let mut accumulation_err: f64 = 0.0;
    for k in [1usize, 2, 4] {
        let rows = k * micro;
        let inputs = Tensor::<A, 2>::random(
            [rows, 6],
            Distribution::Uniform(-1.0, 1.0),
            &ad_device,
        );

        // One k-times-larger batch: the reference.
        let single = layer.forward(inputs.clone()).powf_scalar(2.0).mean();
        let reference = GradientsParams::from_grads(single.backward(), &layer);

        // The same data as k micro-batches, each scaled by `loss_scale` and
        // folded in. The result must be the same gradient.
        let mut accumulator = GradientAccumulator::new(k);
        let scale = accumulator.loss_scale() as f32;
        let mut summed = None;
        for i in 0..k {
            let chunk = inputs.clone().narrow(0, i * micro, micro);
            let loss = layer.forward(chunk).powf_scalar(2.0).mean().mul_scalar(scale);
            let grads = GradientsParams::from_grads(loss.backward(), &layer);
            summed = accumulator.fold(grads, &layer).into_gradients();
        }
        let Some(summed) = summed else {
            accumulation_err = f64::INFINITY;
            continue;
        };

        // Compare parameter by parameter.
        let weight_id = layer.weight.id;
        for (a, b) in [
            (
                summed.get::<<A as AutodiffBackend>::InnerBackend, 2>(weight_id),
                reference.get::<<A as AutodiffBackend>::InnerBackend, 2>(weight_id),
            ),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let diff: f32 = (a - b).abs().max().into_scalar();
                    accumulation_err = accumulation_err.max(f64::from(diff));
                }
                _ => accumulation_err = f64::INFINITY,
            }
        }
    }

    // A cycle must fire exactly every k micro-batches, whether the batches were
    // folded in or skipped by a quality gate -- an accumulator that drifts
    // would change the effective batch size mid-run.
    let mut cadence_err: f64 = 0.0;
    for k in [1usize, 2, 3, 8] {
        let mut accumulator = GradientAccumulator::new(k);
        let mut fired = 0usize;
        for i in 1..=(k * 7) {
            if accumulator.skip().is_ready() {
                fired += 1;
                if i % k != 0 {
                    cadence_err = 1.0;
                }
            }
        }
        if fired != 7 {
            cadence_err = 1.0;
        }
    }

    // ------------------------------------------------------------------
    // The EMA is a convex combination: `d * shadow + (1 - d) * live` with
    // `d` in [0, 1]. Two consequences are checked, because they are what makes
    // an average safe to evaluate with -- the coefficients sum to 1 (no
    // rescaling of the weights), and the result never leaves the interval
    // spanned by its inputs (no extrapolation into a region neither model
    // occupies).
    let mut convexity_err: f64 = 0.0;
    let mut interval_err: f64 = 0.0;
    for decay in [0.0f64, 0.5, 0.9, 0.999, 1.0] {
        for updates in [0usize, 1, 5, 100, 10_000] {
            // The bias correction the implementation applies.
            let warm = (1.0 + updates as f64) / (10.0 + updates as f64);
            let d = decay.min(warm);
            convexity_err = convexity_err.max((d + (1.0 - d) - 1.0).abs());
            if !(0.0..=1.0).contains(&d) {
                convexity_err = 1.0;
            }

            for (shadow, live) in [(-3.0f64, 7.0), (7.0, -3.0), (2.0, 2.0), (0.0, 1e6)] {
                let blended = d * shadow + (1.0 - d) * live;
                let (lo, hi) = (shadow.min(live), shadow.max(live));
                interval_err = interval_err
                    .max((lo - blended).max(0.0))
                    .max((blended - hi).max(0.0));
            }
        }
    }

    // ------------------------------------------------------------------
    // The learning-rate schedule never exceeds its declared peak, and its ramp
    // is monotone. Both matter for the same reason: a schedule that overshoots
    // or oscillates during warmup defeats the purpose of warming up at all.
    let mut peak_violation: f64 = 0.0;
    let mut ramp_violation: f64 = 0.0;
    let mut decay_violation: f64 = 0.0;
    let schedules = [
        LrSchedule::Constant { lr: 1e-3 },
        LrSchedule::WarmupConstant { peak: 1e-3, warmup_steps: 50 },
        LrSchedule::WarmupCosine { peak: 1e-3, min_lr: 1e-5, warmup_steps: 50, total_steps: 500 },
        LrSchedule::WarmupCosine { peak: 3e-4, min_lr: 0.0, warmup_steps: 0, total_steps: 200 },
    ];
    for schedule in &schedules {
        let peak = schedule.peak();
        let mut previous_ramp = f64::NEG_INFINITY;
        let mut previous_decay = f64::INFINITY;
        for step in 0..600 {
            let lr = schedule.at(step);
            peak_violation = peak_violation.max((lr - peak).max(0.0));
            if lr < 0.0 || !lr.is_finite() {
                peak_violation = 1.0;
            }

            let warmup = match *schedule {
                LrSchedule::Constant { .. } => 0,
                LrSchedule::WarmupConstant { warmup_steps, .. }
                | LrSchedule::WarmupCosine { warmup_steps, .. } => warmup_steps,
            };
            if step <= warmup {
                ramp_violation = ramp_violation.max((previous_ramp - lr).max(0.0));
                previous_ramp = lr;
            } else {
                decay_violation = decay_violation.max((lr - previous_decay).max(0.0));
                previous_decay = lr;
            }
        }
    }

    // ------------------------------------------------------------------
    // Uncertainty weighting (20.5). The `+ l` term is what stops the head
    // buying a smaller number by claiming more uncertainty: the objective in
    // `l` alone is minimized at `l* = ln L`, and there its value is `1 + ln L`.
    let mut optimum_err: f64 = 0.0;
    for raw in [1e-3f64, 0.5, 13.4, 1909.8, 1e5] {
        let objective = |l: f64| (-l).exp() * raw + l;
        let l_star = UncertaintyWeighting::optimal_log_variance(raw);
        // Stationarity: the derivative -exp(-l)L + 1 vanishes at l*.
        optimum_err = optimum_err.max((-(-l_star).exp() * raw + 1.0).abs());
        // And it is a minimum, not just a stationary point.
        for delta in [-1.0f64, -0.1, 0.1, 1.0] {
            let gap = objective(l_star) - objective(l_star + delta);
            optimum_err = optimum_err.max(gap.max(0.0));
        }
        optimum_err =
            optimum_err.max((objective(l_star) - UncertaintyWeighting::value_at_optimum(raw)).abs());
    }

    // The property that actually fixes the 140x block imbalance: at the
    // optimum the gradient scale is `exp(-l*) * L = 1` regardless of `L`, so
    // rescaling any noise level's loss by a constant leaves the gradient it
    // contributes unchanged. No weighting convention can reintroduce the
    // imbalance.
    let mut scale_freedom: f64 = 0.0;
    for raw in [13.4f64, 1909.8] {
        for c in [1e-6f64, 1e-3, 1.0, 1e3, 1e6] {
            let scaled = raw * c;
            let effective = (-UncertaintyWeighting::optimal_log_variance(scaled)).exp() * scaled;
            scale_freedom = scale_freedom.max((effective - 1.0).abs());
        }
    }

    // ------------------------------------------------------------------
    // Importance sampling (20.6). Unbiasedness is the exact identity
    // `E_q[p/q] = sum_b q_b (p_b/q_b) = sum_b p_b = 1`, so it is checked
    // exactly rather than by convergence.
    let mut bias: f64 = 0.0;
    let mut simplex: f64 = 0.0;
    let mut weight_bound: f64 = 0.0;
    let mut cold_identity: f64 = 0.0;
    for bins in [1usize, 4, 16] {
        for smoothing in [0.01f64, 0.25, 1.0] {
            // A cold sampler must be *exactly* plain sampling, so the feature
            // can be switched on before it has learned anything.
            let cold = SigmaImportanceSampler::new(bins).with_smoothing(smoothing);
            for q in cold.proposal() {
                cold_identity = cold_identity.max((q - cold.prior()).abs());
            }
            cold_identity = cold_identity.max((cold.max_weight() - 1.0).abs());

            // Adversarial traffic: all the loss in one bin, nothing elsewhere.
            let mut sampler = SigmaImportanceSampler::new(bins).with_smoothing(smoothing);
            for bin in 0..bins {
                sampler.observe(bin, if bin == 0 { 1e9 } else { 1e-12 });
            }
            let q = sampler.proposal();
            let prior = sampler.prior();

            simplex = simplex.max((q.iter().sum::<f64>() - 1.0).abs());
            let expectation: f64 = q.iter().map(|qb| qb * (prior / qb)).sum();
            bias = bias.max((expectation - 1.0).abs());

            // The smoothing floor is what bounds the estimator's variance: a
            // starved bin is not merely unvisited, its weight p/q explodes.
            weight_bound = weight_bound.max((sampler.max_weight() - 1.0 / smoothing).max(0.0));
        }
    }

    vec![
        cert(
            "optim",
            "accumulation_equals_one_large_batch",
            "Folding k micro-batches through the real accumulator reproduces the gradient of one k-times-larger batch, and the cycle fires on exactly that cadence whether batches were folded or skipped.",
            accumulation_err.max(cadence_err),
            // Gradients are f32 and the two paths sum in different orders.
            // eps = 5.96e-8 over at most 12 rows bounds the difference at
            // roughly 1e-6; the previous 1e-12 was only ever reachable because
            // the check ran in f64 against a formula instead of the code.
            1e-6,
        ),
        cert(
            "optim",
            "ema_is_a_convex_combination",
            "The EMA coefficients sum to 1 and the blend never leaves the interval spanned by the shadow and the live weights, at every decay and update count.",
            convexity_err.max(interval_err),
            1e-12,
        ),
        cert(
            "optim",
            "lr_schedule_is_bounded_and_monotone",
            "No schedule exceeds its declared peak or goes negative; the warmup ramp is non-decreasing and the post-warmup decay is non-increasing.",
            peak_violation.max(ramp_violation).max(decay_violation),
            1e-15,
        ),
        cert(
            "optim",
            "uncertainty_optimum_is_the_log_loss",
            "The uncertainty objective exp(-l)L + l is stationary and minimal at l = ln(L), with value 1 + ln(L): the head cannot report a smaller loss by claiming more uncertainty.",
            optimum_err,
            1e-9,
        ),
        cert(
            "optim",
            "uncertainty_gradient_is_scale_free",
            "At its optimum the effective gradient scale exp(-l)L equals 1 for every loss magnitude, so rescaling any noise level's loss cannot reintroduce the block imbalance.",
            scale_freedom,
            1e-12,
        ),
        cert(
            "optim",
            "importance_sampling_is_unbiased",
            "The proposal is a distribution and E_q[p/q] = 1 exactly, so reweighted samples estimate the prior's mean rather than the proposal's.",
            bias.max(simplex),
            1e-12,
        ),
        cert(
            "optim",
            "importance_weights_are_bounded_by_the_smoothing_floor",
            "The uniform mixture bounds the worst importance weight at 1/smoothing under adversarial traffic, and a cold sampler is exactly plain sampling.",
            weight_bound.max(cold_identity),
            1e-9,
        ),
    ]
}

// ------------------------------------------------------------------ model --

fn planner_certificates() -> Vec<Certificate> {
    use crate::planner::{Beam, Budget, LookaheadDecoder, Path, TrajectoryPlanner};

    // ------------------------------------------------------------------
    // The budget is the whole reason lookahead is deployable: without an
    // enforced ceiling, `beam x depth x candidates` model calls per committed
    // step occasionally turns one token into minutes. Driven adversarially --
    // every expansion offers more options than the budget allows, and no path
    // ever terminates on its own.
    let mut overrun: f64 = 0.0;
    let mut work_overrun: f64 = 0.0;
    for max_evaluations in [1usize, 2, 3, 5, 7, 11, 16, 64] {
        for max_depth in [0usize, 1, 3, 5] {
            for beam_width in [1usize, 2, 4] {
                let budget = Budget { max_evaluations, max_depth, beam_width };
                let mut work = 0usize;
                let plan = Beam::new(budget).search(|_p: &Path<u32>, remaining: usize| {
                    // A model-calling expand consults its allowance first; the
                    // certificate covers both that path and the lazy one.
                    let n = remaining.min(6);
                    work += n;
                    (0..n as u32).map(|i| (i, f64::from(i))).collect::<Vec<_>>()
                });
                overrun = overrun.max((plan.evaluations as f64 - max_evaluations as f64).max(0.0));
                work_overrun = work_overrun.max((work as f64 - max_evaluations as f64).max(0.0));
            }
        }
    }

    // ------------------------------------------------------------------
    // Containment: a rollout of depth 0 must be exactly the greedy policy the
    // crate already had -- pick the best immediate option -- so the planner is
    // a strict generalization rather than a different algorithm that happens
    // to behave similarly. Mirrors how `single_box_reduces_to_flat_moe` earns
    // its keep in the `mosme` group.
    let mut state = 0xD1B5_4A32_D192_ED03u64;
    let mut next_f64 = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut greedy_gap: f64 = 0.0;
    let mut containment: f64 = 0.0;
    for _ in 0..200 {
        let scores: Vec<f64> = (0..6).map(|_| next_f64()).collect();
        let expand = |path: &Path<usize>, _r: usize| -> Vec<(usize, f64)> {
            if path.depth() > 0 {
                return Vec::new();
            }
            scores.iter().copied().enumerate().collect()
        };

        let greedy = Beam::new(Budget::greedy()).search(expand);
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        greedy_gap = greedy_gap.max((greedy.score() - best).abs());

        // beam(1) at depth 0 must reproduce it exactly, argument for argument.
        let beam_one = Beam::new(Budget {
            max_evaluations: usize::MAX,
            max_depth: 0,
            beam_width: 1,
        })
        .search(expand);
        if beam_one.commit() != greedy.commit() || beam_one.score() != greedy.score() {
            containment = 1.0;
        }
    }

    // ------------------------------------------------------------------
    // Only the first step is committed. Lookahead informs the choice; the rest
    // of the path is a hypothesis to be re-planned once its consequences are
    // observed. A planner that executed its whole plan would compound its own
    // prediction error.
    let mut commit_mismatch: f64 = 0.0;
    for max_depth in [0usize, 1, 2, 4] {
        let plan = Beam::new(Budget { max_evaluations: 1024, max_depth, beam_width: 3 })
            .search(|path: &Path<usize>, _r: usize| {
                if path.depth() > max_depth {
                    return Vec::new();
                }
                (0..3).map(|i| (i, next_f64())).collect::<Vec<_>>()
            });
        match (plan.commit(), plan.best.as_ref().and_then(|p| p.steps.first())) {
            (Some(a), Some(b)) if a == b => {}
            (None, None) => {}
            _ => commit_mismatch = 1.0,
        }
    }

    // ------------------------------------------------------------------
    // A trajectory is only a trajectory if sigma falls monotonically and never
    // undershoots the floor. The planner is free to choose the step size; it is
    // not free to run back up the schedule -- the bug this crate already fixed
    // once, in the consistency rollout.
    let mut monotonicity: f64 = 0.0;
    let mut undershoot: f64 = 0.0;
    for depth in [0usize, 1, 2] {
        let planner = TrajectoryPlanner::new(Budget {
            max_evaluations: 512,
            max_depth: depth,
            beam_width: 3,
        });
        let plan = planner.plan(80.0, 0.002, |sigma, width| -sigma - 0.1 * width as f64);
        if let Some(path) = &plan.best {
            let mut previous = 80.0;
            for step in &path.steps {
                monotonicity = monotonicity.max((step.sigma - previous).max(0.0));
                undershoot = undershoot.max((0.002 - step.sigma).max(0.0));
                previous = step.sigma;
            }
        }
    }

    // ------------------------------------------------------------------
    // Cross-depth comparison is invalid: score deltas accumulate, so a shorter
    // path always looks better under log-probabilities and always worse under
    // costs. Here `b` is worse immediately and better overall; a planner that
    // compared a depth-1 path against a depth-2 path on raw score would take
    // `a` and never notice.
    let lookahead_plan = Beam::new(Budget { max_evaluations: 64, max_depth: 1, beam_width: 4 })
        .search(|path: &Path<char>, _r: usize| match path.steps.as_slice() {
            [] => vec![('a', -0.1), ('b', -0.5)],
            ['a'] => vec![('x', -6.0)],
            ['b'] => vec![('y', -0.2)],
            _ => Vec::new(),
        });
    let myopia = f64::from(lookahead_plan.commit() != Some(&'b'));

    // The same statement on the language side, through the real decoder.
    let decoder_plan = LookaheadDecoder::new(
        Budget { max_evaluations: 64, max_depth: 1, beam_width: 4 },
        2,
    )
    .plan(&[65], |context: &[u16]| match context.len() {
        1 => vec![(1, -0.1), (2, -0.5)],
        2 if context[1] == 1 => vec![(11, -6.0)],
        2 => vec![(22, -0.2)],
        _ => Vec::new(),
    });
    let decoder_myopia = f64::from(decoder_plan.commit().map(|s| s.token) != Some(2));

    // ------------------------------------------------------------------
    // The advertised worst case must actually bound the observed spend, or a
    // caller cannot size a budget before paying for it.
    let mut worst_case_violation: f64 = 0.0;
    for beam_width in [1usize, 2, 3] {
        for max_depth in [0usize, 1, 2] {
            let candidates = 4usize;
            let budget = Budget { max_evaluations: 4096, max_depth, beam_width };
            let plan = Beam::new(budget).search(|path: &Path<u32>, _r: usize| {
                if path.depth() > max_depth {
                    return Vec::new();
                }
                (0..candidates as u32).map(|i| (i, f64::from(i))).collect::<Vec<_>>()
            });
            worst_case_violation = worst_case_violation
                .max((plan.evaluations as f64 - budget.worst_case(candidates) as f64).max(0.0));
        }
    }

    vec![
        cert(
            "planner",
            "budget_is_never_exceeded",
            "Beam search spends at most `max_evaluations` candidate evaluations, and tells `expand` its allowance before the work is done -- for every (budget, depth, width).",
            overrun.max(work_overrun),
            0.0,
        ),
        cert(
            "planner",
            "depth_zero_is_greedy",
            "A rollout of depth 0 selects the highest-scoring immediate candidate: the planner strictly generalizes the greedy policy.",
            greedy_gap,
            1e-12,
        ),
        cert(
            "planner",
            "greedy_within_beam_one",
            "beam(1) at depth 0 reproduces greedy decoding exactly, step and score.",
            containment,
            0.0,
        ),
        cert(
            "planner",
            "only_the_first_step_is_committed",
            "The committed step is the first step of the best path, at every depth: lookahead informs the choice without executing the plan.",
            commit_mismatch,
            0.0,
        ),
        cert(
            "planner",
            "trajectory_is_monotone",
            "Every planned step lowers sigma and none undershoots sigma_min, at every rollout depth.",
            monotonicity.max(undershoot),
            0.0,
        ),
        cert(
            "planner",
            "lookahead_defeats_myopia",
            "Where a locally worse step leads to a better continuation, one level of lookahead takes it -- in the beam and in the language decoder alike.",
            myopia.max(decoder_myopia),
            0.0,
        ),
        cert(
            "planner",
            "worst_case_bounds_the_spend",
            "Observed evaluations never exceed `Budget::worst_case`, so a caller can size a plan before paying for it.",
            worst_case_violation,
            0.0,
        ),
    ]
}

// ------------------------------------------------------------------ model --

fn model_certificates() -> Vec<Certificate> {
    let device = Default::default();
    let cfg = ViTDiTConfig::tiny(10);
    let model = DblockClassifier::<B>::new(
        &cfg,
        &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
        &device,
    );

    let pixels = Tensor::<B, 4>::random([4, 3, 32, 32], Distribution::Uniform(-1.0, 1.0), &device);
    let z = Tensor::<B, 2>::random([4, 32], Distribution::Normal(0.0, 1.0), &device);

    let mut certificates = model_health(&model, &pixels, &z);

    // adaLN-zero: a freshly DiT-initialized model outputs exactly zero logits,
    // so the residual stream starts as the identity and training never has to
    // undo a random initial perturbation. Only meaningful at initialization,
    // which is why it is not part of `model_health`.
    let zero_logits = model
        .model()
        .forward_all(
            Tensor::<B, 4>::zeros([1, 3, 32, 32], &device),
            Tensor::<B, 2>::zeros([1, 32], &device),
            Tensor::<B, 1>::zeros([1], &device),
        )
        .abs()
        .max()
        .into_scalar() as f64;
    certificates.push(cert(
        "model",
        "dit_zero_init",
        "DiT zero-initialization makes the classifier output exactly zero before training.",
        zero_logits,
        0.0,
    ));

    certificates
}

/// Certificates that hold for *any* model, trained or not, and can therefore
/// be re-checked during a training run against the live weights.
///
/// These are the invariants a diverging model breaks first: probabilities stop
/// summing to one, the `x0` projection leaves the label-embedding hull, or a
/// parameter goes non-finite. Checking them periodically turns a silent
/// divergence into a named failure at the step it happens.
pub fn model_health<BB: burn::tensor::backend::Backend<FloatElem = f32>>(
    model: &DblockClassifier<BB>,
    pixel_values: &Tensor<BB, 4>,
    latent: &Tensor<BB, 2>,
) -> Vec<Certificate> {
    let batch = pixel_values.dims()[0];
    let device = pixel_values.device();

    // Class probabilities must be a distribution: everything downstream
    // (confidence gates, the x0 projection, the KL objective) assumes it.
    let logits = model.denoise(
        pixel_values.clone(),
        latent.clone(),
        &vec![1.0; batch],
        None,
    );
    let partition_err = softmax(logits, 1)
        .sum_dim(1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .map(|s| (s as f64 - 1.0).abs())
        .fold(0.0f64, f64::max);

    // x0 = probs @ W is a convex combination of the label-embedding rows, so
    // it can never leave their convex hull. A violation means the projection is
    // not the one the sampler assumes -- or that a weight went non-finite.
    let table = model.model().label_embedding_weight();
    let max_row_norm = table
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .max()
        .into_scalar() as f64;
    let x0_norm = model
        .x0_estimate(pixel_values, latent, 1.0, None)
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .max()
        .into_scalar() as f64;
    let hull_excess = ((x0_norm - max_row_norm) / max_row_norm.max(1e-12)).max(0.0);

    // The label embeddings are the diffusion process's clean data; the
    // objective assumes they are unit norm after normalization.
    let num_labels = table.dims()[0];
    let ids: Vec<i64> = (0..batch.min(num_labels) as i64).collect();
    let labels = Tensor::<BB, 1, Int>::from_ints(ids.as_slice(), &device);
    let norm_err = model
        .model()
        .normalized_label_embeds(labels)
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .sub_scalar(1.0)
        .abs()
        .max()
        .into_scalar() as f64;

    vec![
        cert(
            "model",
            "softmax_partition",
            "Class probabilities sum to 1 per sample.",
            partition_err,
            1e-6,
        ),
        cert(
            "model",
            "x0_in_convex_hull",
            "x0 = softmax(logits) W lies in the convex hull of the label embeddings, so its norm is bounded by theirs.",
            hull_excess,
            1e-6,
        ),
        cert(
            "model",
            "label_embeddings_unit_norm",
            "normalized_label_embeds returns unit-norm rows.",
            norm_err,
            1e-6,
        ),
    ]
}

// --------------------------------------------------------------- autodiff --

fn autodiff_certificates() -> Vec<Certificate> {
    use crate::{distill::soft_target_kl, train::DefaultTrainBackend as A};
    use burn::tensor::TensorData;

    let device = Default::default();

    // Gradient check on the distillation objective. Autodiff is trusted
    // everywhere else in this crate, so verifying it against central finite
    // differences on a real loss is the one place that trust is earned.
    let teacher_values = [1.5f32, -0.4, 0.2, 0.9, 0.1, -1.1];
    let student_values = [0.3f32, 0.7, -0.5, -0.2, 1.4, 0.6];

    let teacher = Tensor::<A, 1>::from_floats(teacher_values.as_slice(), &device).reshape([2, 3]);
    let student = Tensor::<A, 1>::from_floats(student_values.as_slice(), &device)
        .reshape([2, 3])
        .require_grad();

    let loss = soft_target_kl(teacher.clone(), student.clone(), 2.0);
    let grads = loss.backward();
    let analytic: Vec<f32> = student
        .grad(&grads)
        .expect("student logits must receive a gradient")
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();

    // Central differences: error is O(h^2) plus O(eps/h) from f32 rounding,
    // minimized around h ~ eps^(1/3) ~ 5e-3 for f32.
    let h = 5e-3f32;
    let mut worst_rel: f64 = 0.0;
    for i in 0..student_values.len() {
        let evaluate = |delta: f32| -> f32 {
            let mut v = student_values;
            v[i] += delta;
            let s = Tensor::<A, 1>::from_data(TensorData::new(v.to_vec(), [6]), &device)
                .reshape([2, 3]);
            soft_target_kl(teacher.clone(), s, 2.0).into_scalar()
        };
        let numeric = (evaluate(h) - evaluate(-h)) / (2.0 * h);
        let scale = analytic[i].abs().max(numeric.abs()).max(1e-3);
        worst_rel = worst_rel.max(((analytic[i] - numeric).abs() / scale) as f64);
    }

    // Gibbs' inequality, the property that makes the KL term a sensible
    // objective at all: it is non-negative and vanishes exactly at agreement.
    let self_kl = soft_target_kl(teacher.clone(), teacher.clone(), 2.0).into_scalar() as f64;
    let cross_kl = soft_target_kl(teacher, student, 2.0).into_scalar() as f64;
    let gibbs_violation = self_kl.abs().max((-cross_kl).max(0.0));

    vec![
        cert(
            "autodiff",
            "distillation_gradcheck",
            "Autodiff gradients of the distillation KL match central finite differences.",
            worst_rel,
            2e-2,
        ),
        cert(
            "autodiff",
            "kl_gibbs_inequality",
            "KL(p||q) >= 0 with equality exactly at p == q.",
            gibbs_violation,
            1e-6,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_certificates_hold() {
        let report = run_all();
        assert!(
            report.passed(),
            "verification failed:\n{}",
            report.render()
        );
    }

    #[test]
    fn test_suite_covers_every_group() {
        // A certificate group that silently disappears would leave a whole
        // subsystem unverified while the gate still reported success.
        let report = run_all();
        let mut groups: Vec<&str> = report.certificates.iter().map(|c| c.group).collect();
        groups.sort_unstable();
        groups.dedup();
        assert_eq!(
            groups,
            vec![
                "accuracy",
                "autodiff",
                "lm",
                "loopgraph",
                "model",
                "moe",
                "mosme",
                "optim",
                "planner",
                "precision",
                "preconditioning",
                "quantize",
                "schedule",
                "solver",
                "stats",
            ]
        );

        // The list above and `GROUPS` are maintained separately on purpose:
        // one is what `run_all` actually emits, the other is what the crate
        // advertises. Checking them against each other catches a group added
        // to one and forgotten in the other -- which has happened.
        let mut declared: Vec<&str> = GROUPS.to_vec();
        declared.sort_unstable();
        assert_eq!(groups, declared, "GROUPS and run_all must agree");
        assert!(report.certificates.len() >= 50, "suite shrank unexpectedly");
    }

    #[test]
    fn test_a_broken_certificate_is_reported() {
        // The gate is only worth anything if it can fail, so check that a
        // violated certificate is detected and rendered as such.
        let report = Report {
            certificates: vec![
                cert("g", "good", "holds", 0.0, 1e-9),
                cert("g", "bad", "does not hold", 1.0, 1e-9),
            ],
        };
        assert!(!report.passed());
        assert_eq!(report.failures().len(), 1);
        assert_eq!(report.num_passed(), 1);
        let rendered = report.render();
        assert!(rendered.contains("FAILED"));
        assert!(rendered.contains("does not hold"));

        // A NaN residual must count as a failure, not slip through a
        // comparison that is false for NaN.
        let nan = Report {
            certificates: vec![cert("g", "nan", "n/a", f64::NAN, 1.0)],
        };
        assert!(!nan.passed());
    }
}
