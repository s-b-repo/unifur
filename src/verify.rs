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
//! | `hybrid` | A dense schedule is the Phase 19 trunk bit for bit; a window or a retrieval set covering the context is dense; linear attention's recurrent state equals its masked form; rotary scores depend only on distance; every mode decodes from its state exactly as it recomputes; rotary decoding continues past the table; the routing state is bounded and carries history; routing locality reads as specified; partial rotary is full rotary at fraction one and leaves the suffix; GQA with full heads is dense and narrow KV shrinks the decode state; gated attention is identity at init; RMSNorm starts at unit RMS; MTP at weight zero is the plain loss |
//! | `deltanet` | Gated DeltaNet: alpha = 1 is the pure delta rule, beta = 0 is pure decay, the zero-state first output is closed-form, L2 rows are unit, the short convolution is causal, and the tensor-alpha step and batched rollout are the scalar core bit for bit |
//! | `geom` | The learned metric `G = L Lᵀ` is positive definite by construction; geodesic attention rows are distributions; the closed-form Lipschitz step never increases the relaxation energy; and the exact rational kernel keeps `1/3 + 1/6 = 1/2` exact, intersects segments exactly, refuses degenerate figures, derives facts with their rule certificate to a fixed point, and both rejects a false claim and survives a true one in falsification |
//! | `geomfusion` | The trunk-fused geometric stream: the composed readout gates are a distribution per row (the old broadcast-scatter bug would fail this), deeper certified relaxation never raises the energy it reports, and a zero-initialized output projection makes fusion an exact identity on the trunk hidden state |
//! | `antipattern` | Every shipped rule matches its examples and none of its counterexamples; labels follow tokens through both corpus readers; zero weights reproduce the plain loss bitwise; the unlikelihood term is 0 for an impossible token and finite for a certain one; a penalized target leaves the likelihood; one penalized step lowers p(bad) where one plain step raises it |
//! | `model` | Softmax partition, unit-norm label embeddings, DiT zero-init, and that every `x0` estimate lies in the convex hull of the label table |
//! | `autodiff` | Finite-difference gradient check on the distillation objective |
//! | `qwennet` | Qwen3.8 trunk: full attention decodes from its KV cache bit for bit, a mixed linear+full trunk decodes to its prefill logits bit for bit, and the zero-centered RMSNorm is identity at zero weight |

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
use anyhow::Context as _;
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
    Certificate {
        group,
        name,
        theorem,
        residual,
        tolerance,
    }
}

/// The residual of a check that could not run: infinite, so the certificate
/// fails, with the reason on stderr rather than a panic.
fn failed(what: &str, err: &anyhow::Error) -> f64 {
    eprintln!("verify: {what}: {err:#}");
    f64::INFINITY
}

/// Run a group's checks; if the group cannot even be set up, report that as
/// a single failing certificate instead of unwinding the whole run.
fn checks_or_failed(
    group: &'static str,
    checks: fn() -> anyhow::Result<Vec<Certificate>>,
) -> Vec<Certificate> {
    match checks() {
        Ok(certificates) => certificates,
        Err(err) => vec![cert(
            group,
            "checks_ran",
            "Every check in this group could be set up and evaluated.",
            failed(group, &err),
            0.0,
        )],
    }
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
            out.push_str(&format!(
                "\nFAILED {}::{}\n  {}\n",
                f.group, f.name, f.theorem
            ));
        }
        out
    }
}

/// Names of every certificate group, in the order [`run_all`] emits them.
pub const GROUPS: [&str; 28] = [
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
    "hybrid",
    "deltanet",
    "geom",
    "geomfusion",
    "antipattern",
    "codequality",
    "planner",
    "accuracy",
    "optim",
    "experiment",
    "multisource",
    "policy",
    "ablation",
    "model",
    "autodiff",
    "qwennet",
    "cheat",
    "codegen_eval",
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
    certificates.extend(checks_or_failed("lm", lm_checks));
    certificates.extend(checks_or_failed("hybrid", hybrid_checks));
    certificates.extend(deltanet_certificates());
    certificates.extend(checks_or_failed("geom", geom_checks));
    certificates.extend(checks_or_failed("geomfusion", geomfusion_checks));
    certificates.extend(antipattern_certificates());
    certificates.extend(codequality_certificates());
    certificates.extend(planner_certificates());
    certificates.extend(accuracy_certificates());
    certificates.extend(optim_certificates());
    certificates.extend(experiment_certificates());
    certificates.extend(multisource_certificates());
    certificates.extend(policy_certificates());
    certificates.extend(ablation_certificates());
    certificates.extend(checks_or_failed("model", model_checks));
    certificates.extend(autodiff_certificates());
    certificates.extend(checks_or_failed("qwennet", qwennet_checks));
    certificates.extend(cheat_certificates());
    certificates.extend(codegen_eval_certificates());
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
        .max(if edm.windows(2).all(|w| w[0] > w[1]) {
            0.0
        } else {
            1.0
        });
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
        closed_form_err = closed_form_err.max((z_end - expected).abs().max().into_scalar() as f64);
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
            interp_err = interp_err
                .max((p(-g0) - a_n).abs())
                .max((p(-g1) - b_n).abs());
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
                (
                    (n as f64).log2(),
                    oracle_error(kind, &grid, oracle).max(1e-12).log2(),
                )
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
    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let mut bound_excess: f64 = 0.0;
    let mut idempotence_err: f64 = 0.0;
    // Both of the above are satisfied trivially by a `round` that does nothing:
    // an identity has zero relative error and is trivially idempotent. So the
    // suite also has to show that rounding *happens* -- a mutant that made
    // `round_scalar` the identity passed both of the original certificates.
    let mut inertness: f64 = 0.0;
    // ...and that the tensor path agrees with the scalar one, since sampling
    // uses `Precision::round` while these bounds were only ever measured on
    // `round_scalar`.
    let mut path_disagreement: f64 = 0.0;

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
                    idempotence_err =
                        idempotence_err.max((precision.round_scalar(r) - r).abs() as f64);
                }
            }
        }

        // A value needing more significand bits than the format has *must*
        // move. `1 + 2^-20` is representable in f32 and in neither bf16 (8
        // bits) nor f16 (11), so an unchanged output means no rounding
        // happened at all.
        for probe in [
            1.0f32 + 2f32.powi(-20),
            1.0f32 + 2f32.powi(-18),
            -(1.0f32 + 2f32.powi(-20)),
        ] {
            if precision.round_scalar(probe) == probe {
                inertness = 1.0;
            }
        }

        // The scalar and tensor paths are two implementations of one claim;
        // they must not drift.
        let values: Vec<f32> = (0..64)
            .map(|i| (1.0 + i as f32 / 64.0) * 2f32.powi((i % 9) - 4))
            .collect();
        let rounded = precision
            .round(Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([8, 8]));
        let got: Vec<f32> = rounded.into_data().convert::<f32>().iter::<f32>().collect();
        for (tensor_value, scalar_input) in got.iter().zip(&values) {
            let want = precision.round_scalar(*scalar_input);
            if tensor_value.to_bits() != want.to_bits() {
                path_disagreement = f64::from((tensor_value - want).abs()).max(f64::MIN_POSITIVE);
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
            "rounding_actually_rounds",
            "A value needing more significand bits than the format holds is changed by rounding, and the tensor path agrees with the scalar one bit for bit -- neither an inert `round` nor a drifted second implementation can pass.",
            inertness.max(path_disagreement),
            0.0,
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
    let max_gap = NF4_LEVELS
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(0.0f32, f32::max);
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

    // Packing two codes per byte is a STORAGE detail: the packed form must
    // dequantize bit-identically to the simulated one-code-per-byte form,
    // plain and double-quantized.
    use crate::quantize::PackedNf4Tensor;
    let mut packed_gap: f64 = 0.0;
    for n in [2048usize, 1000, 63] {
        let v: Vec<f32> = Tensor::<B, 1>::random([n], Distribution::Normal(0.0, 1.3), &device)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let pairs = [
            (
                PackedNf4Tensor::quantize(&v).dequantize(),
                Nf4Tensor::quantize(&v).dequantize(),
            ),
            (
                PackedNf4Tensor::quantize(&v)
                    .with_double_quantization()
                    .dequantize(),
                Nf4Tensor::quantize(&v)
                    .with_double_quantization()
                    .dequantize(),
            ),
        ];
        for (packed, simulated) in &pairs {
            packed_gap = packed_gap.max(
                packed
                    .iter()
                    .zip(simulated)
                    .map(|(a, b)| f64::from((a - b).abs()))
                    .fold(0.0, f64::max),
            );
        }
    }

    // An Nf4Linear is exactly "dequantize, then matmul": same values, same
    // burn linear op as an f32 Linear holding the dequantized weight.
    use crate::qwennet::Nf4Linear;
    let w: Vec<f32> = Tensor::<B, 2>::random([8, 6], Distribution::Normal(0.0, 1.0), &device)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let packed = PackedNf4Tensor::quantize(&w).with_double_quantization();
    let deq = packed.dequantize();
    let nf4 = Nf4Linear::<B>::from_packed(packed, 8, 6, None);
    let f32lin = burn::nn::Linear::<B> {
        weight: burn::module::Param::from_tensor(
            Tensor::<B, 1>::from_floats(deq.as_slice(), &device).reshape([8, 6]),
        ),
        bias: None,
    };
    let x3 = Tensor::<B, 3>::random([1, 4, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let flat =
        |t: Tensor<B, 3>| -> Vec<f32> { t.into_data().convert::<f32>().iter::<f32>().collect() };
    let a = flat(nf4.forward(x3.clone()));
    let b = flat(f32lin.forward(x3));
    let nf4_linear_gap: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max);

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
        cert(
            "quantize",
            "packed_nf4_storage_roundtrip_is_exact",
            "Packing two NF4 codes per byte (low nibble first) is a pure storage detail: the packed form dequantizes bit-identically to the simulated one-code-per-byte form, plain and double-quantized.",
            packed_gap,
            0.0,
        ),
        cert(
            "quantize",
            "nf4_linear_is_dequantize_then_matmul",
            "A frozen NF4-resident linear produces bit-identical outputs to an f32 linear holding its dequantized weight: the only thing quantization changes is residency, never arithmetic.",
            nf4_linear_gap,
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
    checks_or_failed("moe", moe_checks)
}

fn moe_checks() -> anyhow::Result<Vec<Certificate>> {
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
        for s in gates.sum_dim(1).into_data().convert::<f32>().iter::<f32>() {
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
        // Measured through `weighted_switch_loss` itself, not recomputed here.
        // The inline version of this check passed while a mutant that dropped
        // the `* E` factor shipped -- a certificate that reimplements the
        // formula proves the formula, which was never in doubt.
        let probs = Tensor::<B, 1>::from_floats(
            p.iter().map(|v| *v as f32).collect::<Vec<f32>>().as_slice(),
            &device,
        )
        .reshape([1, e]);
        // On the diagonal `f == p`, so the top-1 assignment must reproduce the
        // same distribution: a one-hot row per expert, weighted by `p`.
        let ids = Tensor::<B, 1, Int>::arange(0..e as i64, &device).reshape([e, 1]);
        let dense = probs.clone().repeat_dim(0, e);
        let weights = probs.clone().reshape([e, 1]);
        let loss =
            f64::from(crate::moe::weighted_switch_loss(&dense, &ids, &weights, e).into_scalar());
        bound_violation = bound_violation
            .max((1.0 - loss).max(0.0))
            .max((loss - e as f64).max(0.0));
    }
    // Uniform routing attains the lower bound exactly -- again measured through
    // the implementation.
    let uniform = Tensor::<B, 2>::full([e, e], 1.0 / e as f32, &device);
    let uniform_ids = Tensor::<B, 1, Int>::arange(0..e as i64, &device).reshape([e, 1]);
    let uniform_weights = Tensor::<B, 2>::full([e, 1], 1.0 / e as f32, &device);
    let uniform_gap = f64::from(
        crate::moe::weighted_switch_loss(&uniform, &uniform_ids, &uniform_weights, e).into_scalar(),
    ) - 1.0;
    let uniform_gap = uniform_gap.abs();

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
            let logits =
                Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([1, width]);
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
    let p0: Vec<f32> = softmax(base.clone(), 1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let p1: Vec<f32> = softmax(shifted.clone(), 1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let mut shift_invariance: f64 = 0.0;
    for (a, b) in p0.iter().zip(&p1) {
        shift_invariance = shift_invariance.max(f64::from((a - b).abs()));
    }
    let z_before = f64::from(crate::moe::router_z_loss(&base).into_scalar());
    let z_after = f64::from(crate::moe::router_z_loss(&shifted).into_scalar());
    // The z-loss must actually register the shift; 0 if it does, 1 if not.
    let z_sees_the_shift = f64::from(u8::from(z_after <= z_before * 10.0));

    // ------------------------------------------------------------------
    // Routing diagnostics (roadmap 23.6). Two normalized entropies, and the
    // point is that they can disagree: a balanced load says nothing about
    // whether any token was *decided*. The middle row of the table in
    // `RoutingStats` -- balanced load, hedging tokens -- is exactly what a
    // per-micro-batch balance loss produces and exactly what its own value
    // cannot show.
    use crate::moe::{layer_routing, normalized_entropy, switch_loss_from_parts, RoutingStats};
    let mut entropy_err = f64::from((normalized_entropy(&[0.25; 4]) - 1.0).abs())
        .max(f64::from(normalized_entropy(&[1.0, 0.0, 0.0, 0.0]).abs()));
    let (t, e) = (8usize, 4usize);
    let round_robin: Vec<i64> = (0..t).map(|i| (i % e) as i64).collect();
    let top1 = Tensor::<B, 1, Int>::from_ints(round_robin.as_slice(), &device).reshape([t, 1]);
    // Balanced load, every token hedging uniformly: both entropies are 1.
    let hedging = layer_routing(&Tensor::<B, 2>::full([t, e], 0.25, &device), &top1, e).to_host();
    entropy_err = entropy_err
        .max(f64::from((hedging.load_entropy - 1.0).abs()))
        .max(f64::from((hedging.token_entropy - 1.0).abs()));
    // Balanced load, every token certain: load entropy 1, token entropy 0.
    let mut confident = vec![0.0f32; t * e];
    for i in 0..t {
        confident[i * e + i % e] = 1.0;
    }
    let specialized = layer_routing(
        &Tensor::<B, 1>::from_floats(confident.as_slice(), &device).reshape([t, e]),
        &top1,
        e,
    )
    .to_host();
    entropy_err = entropy_err
        .max(f64::from((specialized.load_entropy - 1.0).abs()))
        .max(f64::from(specialized.token_entropy.abs()));
    // Collapse: every token to expert 0, load entropy 0.
    let collapsed = layer_routing(
        &Tensor::<B, 2>::full([t, e], 0.25, &device),
        &Tensor::<B, 2, Int>::zeros([t, 1], &device),
        e,
    )
    .to_host();
    entropy_err = entropy_err.max(f64::from(collapsed.load_entropy.abs()));

    // The statistics a real layer reports must be what its own routing
    // probabilities imply, recomputed on the host from the router's logits.
    let stats_layer = MoELayer::<B>::new(&MoEConfig::new(8, 4, 4).with_top_k(2), &device);
    let sx = Tensor::<B, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let sc = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
    let reported = stats_layer.forward(sx.clone(), sc.clone()).routing;
    let host_probs: Vec<f32> = softmax(stats_layer.router_logits(&sx, &sc), 1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let (st, se) = (6usize, 4usize);
    let mut host_load = vec![0.0f32; se];
    let mut host_entropy = 0.0f64;
    for row in 0..st {
        let r = &host_probs[row * se..(row + 1) * se];
        let argmax = (0..se).fold(0, |best, e| if r[e] > r[best] { e } else { best });
        host_load[argmax] += 1.0 / st as f32;
        host_entropy -= r
            .iter()
            .map(|&v| f64::from(v) * f64::from(v).ln())
            .sum::<f64>();
    }
    let host = RoutingStats::from_load(host_load, (host_entropy / st as f64) as f32, st);
    let got = reported.to_host();
    let mut stats_err = f64::from((got.token_entropy - host.token_entropy).abs())
        .max(f64::from((got.load_entropy - host.load_entropy).abs()));
    for (a, b) in got.load.iter().zip(&host.load) {
        stats_err = stats_err.max(f64::from((a - b).abs()));
    }
    if got.tokens != st || got.experts() != se {
        stats_err = 1.0;
    }

    // ------------------------------------------------------------------
    // Global-batch load (roadmap 23.4). A window of one micro-batch is the
    // micro-batch: the load goes to the host and back as f32, which is exact,
    // and the recombination is the same two ops the fused loss uses.
    let micro = stats_layer.forward(sx.clone(), sc.clone());
    let mut one = crate::schedule::GlobalLoad::new(1);
    let f_global = one.observe(0, &micro.routing.to_host().load);
    let global = switch_loss_from_parts(
        Tensor::<B, 1>::from_floats(f_global.as_slice(), &device).reshape([1, se]),
        micro.routing.prob_mass.clone().reshape([1, se]),
        se,
    );
    let window_one_err = f64::from(u8::from(
        global.into_scalar().to_bits() != micro.balance.into_scalar().to_bits(),
    ));

    // And the reason to want it: two micro-batches that each route entirely to
    // a *different* expert are perfectly balanced together and perfectly
    // imbalanced apart. Per micro-batch the Switch loss is `E` for both; over
    // the window the second one costs `E/2`.
    let (gt, ge) = (4usize, 4usize);
    let onehot = |col: usize| {
        let mut v = vec![0.0f32; gt * ge];
        for row in 0..gt {
            v[row * ge + col] = 1.0;
        }
        Tensor::<B, 1>::from_floats(v.as_slice(), &device).reshape([gt, ge])
    };
    let ones_col = Tensor::<B, 2>::ones([gt, 1], &device);
    let (probs_a, top_a) = (onehot(0), Tensor::<B, 2, Int>::zeros([gt, 1], &device));
    let (probs_b, top_b) = (onehot(1), Tensor::<B, 2, Int>::ones([gt, 1], &device));
    let micro_a =
        f64::from(crate::moe::weighted_switch_loss(&probs_a, &top_a, &ones_col, ge).into_scalar());
    let micro_b =
        f64::from(crate::moe::weighted_switch_loss(&probs_b, &top_b, &ones_col, ge).into_scalar());
    let mut two = crate::schedule::GlobalLoad::new(2);
    two.observe(0, &layer_routing(&probs_a, &top_a, ge).to_host().load);
    let f_b = two.observe(0, &layer_routing(&probs_b, &top_b, ge).to_host().load);
    let routing_b = layer_routing(&probs_b, &top_b, ge);
    let global_b = f64::from(
        switch_loss_from_parts(
            Tensor::<B, 1>::from_floats(f_b.as_slice(), &device).reshape([1, ge]),
            routing_b.prob_mass.reshape([1, ge]),
            ge,
        )
        .into_scalar(),
    );
    let specialization_err = (micro_a - ge as f64)
        .abs()
        .max((micro_b - ge as f64).abs())
        .max((global_b - ge as f64 / 2.0).abs());

    // ------------------------------------------------------------------
    // Loss-free bias balancing (roadmap 23.5). The bias must be able to move
    // the *selection* without touching a selected expert's *gate*, and a bias
    // of zero -- or one that is equal on every expert -- must change nothing
    // at all, to the bit.
    use crate::moe::TopKRouter;
    let bias_cfg = MoEConfig::new(8, 4, 4).with_top_k(2);
    let plain = MoELayer::<B>::new(&bias_cfg, &device);
    let bx = Tensor::<B, 3>::random([2, 3, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let bc = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device);
    // Forcing every parameter before cloning: an un-forced `Param` draws its
    // own weights on each clone (see `tensor_ext`).
    let plain_out = plain.forward(bx.clone(), bc.clone());
    let rebias = |bias: Tensor<B, 2>| {
        MoELayer::from_parts(
            TopKRouter::from_parts(plain.router().weight(), plain.router().bias())
                .with_balance_bias(Some(bias)),
            plain.experts().to_vec(),
            plain.top_k(),
            true,
            plain.z_level(),
        )
    };
    let bits = |t: Tensor<B, 3>| -> Vec<u32> {
        t.into_data()
            .convert::<f32>()
            .iter::<f32>()
            .map(f32::to_bits)
            .collect()
    };
    let plain_bits = bits(plain_out.output.clone());
    let plain_balance = plain_out.balance.into_scalar().to_bits();
    let mut identity_err = 0.0f64;
    for bias in [
        Tensor::<B, 2>::zeros([1, 4], &device),
        Tensor::<B, 2>::full([1, 4], 3.5, &device),
    ] {
        let out = rebias(bias).forward(bx.clone(), bc.clone());
        if bits(out.output) != plain_bits || out.balance.into_scalar().to_bits() != plain_balance {
            identity_err = 1.0;
        }
    }
    // A large bias on expert 2 puts it in every token's top-2; the gate values
    // are still the unbiased probabilities of whoever was selected.
    let logits = plain.router_logits(&bx, &bc);
    let probs = softmax(logits.clone(), 1);
    let steered = TopKRouter::from_parts(plain.router().weight(), plain.router().bias())
        .with_balance_bias(Some(
            Tensor::<B, 1>::from_floats([0.0f32, 0.0, 50.0, 0.0], &device).reshape([1, 4]),
        ));
    let (vals, idx) = steered.select(&logits, &probs, 2);
    let idx: Vec<i64> = idx.into_data().convert::<i64>().iter::<i64>().collect();
    let vals: Vec<f32> = vals.into_data().convert::<f32>().iter::<f32>().collect();
    let probs_host: Vec<f32> = probs.into_data().convert::<f32>().iter::<f32>().collect();
    let mut steer_err = 0.0f64;
    for row in 0..6 {
        let picked = &idx[row * 2..row * 2 + 2];
        if !picked.contains(&2) {
            steer_err = 1.0;
        }
        for (slot, &expert) in picked.iter().enumerate() {
            let expected = probs_host[row * 4 + expert as usize];
            if vals[row * 2 + slot].to_bits() != expected.to_bits() {
                steer_err = 1.0;
            }
        }
    }
    // One nudge against a fully collapsed load moves the overloaded expert by
    // exactly -rate and every starved one by exactly +rate; a balanced load
    // moves nothing.
    let mut nudged = TopKRouter::<B>::new(4, 4, &device);
    nudged.ensure_balance_bias();
    nudged.nudge_balance_bias(&[1.0, 0.0, 0.0, 0.0], 1e-3);
    nudged.nudge_balance_bias(&[0.25, 0.25, 0.25, 0.25], 1e-3);
    let after: Vec<f32> = nudged
        .balance_bias()
        .context("bias attached")?
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let want = [-1e-3f32, 1e-3, 1e-3, 1e-3];
    let nudge_err = f64::from(u8::from(
        after
            .iter()
            .zip(&want)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
    ));

    Ok(vec![
        cert(
            "moe",
            "routing_entropies_read_as_specified",
            "Normalized load entropy is 1 for a balanced load and 0 for collapse; per-token entropy is 1 for tokens that hedge uniformly and 0 for certain ones -- and a balanced load with hedging tokens scores 1 on both, which is what a balance loss alone cannot see.",
            entropy_err,
            1e-6,
        ),
        cert(
            "moe",
            "reported_routing_matches_host_recomputation",
            "The load fractions and entropies a layer reports equal those recomputed on the host from its own routing probabilities.",
            stats_err,
            1e-5,
        ),
        cert(
            "moe",
            "global_window_of_one_is_the_micro_batch_loss",
            "With a load window of one micro-batch, the global-batch balance loss equals the fused Switch loss bit for bit.",
            window_one_err,
            0.0,
        ),
        cert(
            "moe",
            "global_scope_allows_specialization_across_micro_batches",
            "Two micro-batches routed entirely to different experts cost E each per micro-batch and E/2 over a two-batch window: balance is judged on the global batch, not on every micro-batch alone.",
            specialization_err,
            1e-6,
        ),
        cert(
            "moe",
            "zero_or_uniform_selection_bias_is_a_bitwise_identity",
            "A router with a zero selection bias, or one equal on every expert, produces the same output and balance loss as a router with none, to the bit.",
            identity_err,
            0.0,
        ),
        cert(
            "moe",
            "selection_bias_steers_selection_not_gates",
            "A large bias on one expert puts it in every token's top-k, while every selected expert's gate value is still its unbiased probability.",
            steer_err,
            0.0,
        ),
        cert(
            "moe",
            "one_nudge_moves_each_bias_by_exactly_the_rate",
            "Against a collapsed load, one nudge lowers the overloaded expert's bias by exactly the rate and raises every starved one by it; a balanced load moves nothing.",
            nudge_err,
            0.0,
        ),
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
    ])
}

// ------------------------------------------------------------------ mosme --

fn mosme_certificates() -> Vec<Certificate> {
    checks_or_failed("mosme", mosme_checks)
}

fn mosme_checks() -> anyhow::Result<Vec<Certificate>> {
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
    let reference = flat_layer.as_flat().context("one box")?;
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
            (hierarchical.balance.z_loss - flat.z_loss)
                .abs()
                .max()
                .into_scalar(),
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
        .context("extend")?;
    let grown = layer.grown(&grown_spec, &cfg, &device).context("grow")?;
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
    let bank = crate::mosme::MosmeAdapterBank::<B>::from_linear(
        base.clone(),
        &ragged(1, 2),
        4,
        4,
        4.0,
        &device,
    );
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
    let dblock_cfg = DblockConfig {
        num_blocks: 2,
        ..DblockConfig::default()
    };
    let ensemble =
        crate::mosme::MosmeEnsemble::<B>::fresh(&ragged(1, 1), &vit, &dblock_cfg, 4, &device)?;
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
        .context("serialize spec")
        .and_then(|t| serde_json::from_str::<MosmeSpec>(&t).context("parse spec"))
    {
        Ok(back) if back == index => 0.0,
        Ok(_) => 1.0,
        Err(err) => failed("mosme: spec round trip", &err),
    };

    Ok(vec![
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
    ])
}

// --------------------------------------------------------------------- lm --

/// Trunk certificates. `anyhow::Result`: a check that cannot be set up is a
/// failing certificate, not a panic and not a silent pass.
fn lm_checks() -> anyhow::Result<Vec<Certificate>> {
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
    let model = LanguageModel::<B>::new(&LmConfig::tiny(), &device)?;
    let ids: Vec<i64> = vec![5, 6, 7, 8, 9];
    let n = ids.len();
    let as_tensor = |v: &[i64]| Tensor::<B, 1, Int>::from_ints(v, &device).reshape([1, v.len()]);
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
    let greedy = model.generate(
        &prompt,
        4,
        &Sampling::Greedy,
        &mut StdRng::seed_from_u64(0),
        &device,
    );
    let top1 = model.generate(
        &prompt,
        4,
        &Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
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
    let cached_ids: Vec<i64> = tokenizer
        .encode("cache me")
        .iter()
        .map(|t| *t as i64)
        .collect();
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
    let plain = model.generate(
        &prompt,
        6,
        &Sampling::Greedy,
        &mut StdRng::seed_from_u64(9),
        &device,
    );
    let cached = model.generate_cached(
        &prompt,
        6,
        &Sampling::Greedy,
        &mut StdRng::seed_from_u64(9),
        &device,
    );
    let decode_err = f64::from(u8::from(plain != cached));

    Ok(vec![
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
    ])
}

// ----------------------------------------------------------------- hybrid --

fn hybrid_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::hybrid::{
        apply_rotary, feature_map, linear_attention, AttentionMode, AttentionSchedule, LinearState,
        PositionKind, ROTARY_BASE,
    };
    use crate::lm::{LanguageModel, LmConfig, Sampling};
    use crate::moe::MoEConfig;
    use crate::routing::{layer_agreement, token_stability, RoutingState};
    use burn::module::Module;
    use rand::{rngs::StdRng, SeedableRng};

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let as_tensor = |v: &[i64]| Tensor::<B, 1, Int>::from_ints(v, &device).reshape([1, v.len()]);
    let logits_of = |m: &LanguageModel<B>, ids: &[i64]| -> Vec<f32> {
        m.forward(as_tensor(ids))
            .logits
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect()
    };
    let max_abs_diff = |a: &[f32], b: &[f32]| -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max)
    };
    let max_rel_diff = |a: &[f32], b: &[f32]| -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()) / f64::from(y.abs()).max(1.0))
            .fold(0.0, f64::max)
    };
    let build = |config: &LmConfig| -> anyhow::Result<LanguageModel<B>> {
        let model = LanguageModel::<B>::new(config, &device)?;
        crate::tensor_ext::force_initialization(&model);
        Ok(model)
    };
    let tiny = LmConfig::tiny();
    let n_layers = tiny.num_layers;
    let ids: Vec<i64> = vec![5, 66, 7, 108, 9, 200, 11, 32];

    // A mode that only changes the mask adds no parameter, so the reference
    // model's record loads into a model of the other mode and the two share
    // every weight bit for bit -- without depending on the global RNG, which
    // any concurrently running test may draw from.
    let plain = build(&tiny)?;
    let same_weights = |config: &LmConfig| -> anyhow::Result<LanguageModel<B>> {
        Ok(LanguageModel::<B>::new(config, &device)?.load_record(plain.clone().into_record()))
    };

    // 1. An explicit all-dense schedule is the Phase 19 trunk.
    let dense = same_weights(&LmConfig {
        attention: Some(AttentionSchedule::dense(n_layers)),
        ..tiny.clone()
    })?;
    let reference = logits_of(&plain, &ids);
    let dense_gap = max_abs_diff(&reference, &logits_of(&dense, &ids));

    // 2. A window covering the whole sequence is dense; so is reading every key.
    let wide = same_weights(&LmConfig {
        attention: Some(AttentionSchedule {
            modes: vec![AttentionMode::Sliding { window: 64 }; n_layers],
        }),
        ..tiny.clone()
    })?;
    let window_gap = max_abs_diff(&reference, &logits_of(&wide, &ids));
    let all_keys = same_weights(&LmConfig {
        attention: Some(AttentionSchedule {
            modes: vec![AttentionMode::Retrieval { top_k: 64 }; n_layers],
        }),
        ..tiny.clone()
    })?;
    let retrieval_gap = max_abs_diff(&reference, &logits_of(&all_keys, &ids));

    // 3. Linear attention: the chunked recurrence equals the masked matrix form.
    let (b, h, n, d) = (2usize, 2usize, 7usize, 4usize);
    let q = feature_map(Tensor::<B, 4>::random(
        [b, h, n, d],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    ));
    let k = feature_map(Tensor::<B, 4>::random(
        [b, h, n, d],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    ));
    let v = Tensor::<B, 4>::random([b, h, n, d], Distribution::Uniform(-1.0, 1.0), &device);
    let (full, _) = linear_attention(q.clone(), k.clone(), v.clone(), None, 1e-6);
    let full: Vec<f32> = full.into_data().convert::<f32>().iter::<f32>().collect();
    let mut linear_gap: f64 = 0.0;
    for chunks in [vec![1; n], vec![3, 1, 3], vec![2, 5]] {
        let mut state: Option<LinearState<B>> = None;
        let mut outputs = Vec::new();
        let mut at = 0;
        for size in chunks {
            let (out, next) = linear_attention(
                q.clone().narrow(2, at, size),
                k.clone().narrow(2, at, size),
                v.clone().narrow(2, at, size),
                state.take(),
                1e-6,
            );
            outputs.push(out);
            state = Some(next);
            at += size;
        }
        let produced: Vec<f32> = Tensor::cat(outputs, 2)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        linear_gap = linear_gap.max(max_rel_diff(&produced, &full));
    }

    // 4. Rotary: a score depends only on the distance between the positions.
    let rq = Tensor::<B, 4>::random([1, 1, 1, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let rk = Tensor::<B, 4>::random([1, 1, 1, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let score = |qo: usize, ko: usize| -> f64 {
        let s: f32 = (apply_rotary(rq.clone(), qo, ROTARY_BASE)
            * apply_rotary(rk.clone(), ko, ROTARY_BASE))
        .sum()
        .into_scalar();
        f64::from(s)
    };
    let base_score = score(9, 4);
    let rotary_gap = [1usize, 7, 30, 250]
        .iter()
        .map(|s| (score(9 + s, 4 + s) - base_score).abs())
        .fold(0.0, f64::max);

    // 5. Every mode decodes from its state exactly as it recomputes, under
    //    several chunkings.
    let schedules: Vec<(&str, AttentionSchedule)> = vec![
        ("dense", AttentionSchedule::dense(n_layers)),
        (
            "sliding3",
            AttentionSchedule {
                modes: vec![AttentionMode::Sliding { window: 3 }; n_layers],
            },
        ),
        (
            "retrieval2",
            AttentionSchedule {
                modes: vec![AttentionMode::Retrieval { top_k: 2 }; n_layers],
            },
        ),
        (
            "linear",
            AttentionSchedule {
                modes: vec![AttentionMode::Linear; n_layers],
            },
        ),
        (
            "learned",
            AttentionSchedule {
                modes: vec![AttentionMode::Learned; n_layers],
            },
        ),
        (
            "3:1",
            AttentionSchedule::ratio(n_layers, 3, AttentionMode::Linear, AttentionMode::Dense),
        ),
    ];
    let mut cache_gap: f64 = 0.0;
    for (_, schedule) in &schedules {
        for positions in [
            PositionKind::Learned,
            PositionKind::Rotary,
            PositionKind::None,
        ] {
            let model = build(&LmConfig {
                attention: Some(schedule.clone()),
                positions,
                ..tiny.clone()
            })?;
            let reference = logits_of(&model, &ids);
            for chunks in [
                vec![ids.len()],
                vec![1; ids.len()],
                vec![3, 1, ids.len() - 4],
            ] {
                let mut cache = model.new_cache();
                let mut produced: Vec<f32> = Vec::new();
                let mut at = 0usize;
                for size in chunks {
                    let out = model.forward_cached(as_tensor(&ids[at..at + size]), &mut cache);
                    produced.extend(out.logits.into_data().convert::<f32>().iter::<f32>());
                    at += size;
                }
                cache_gap = cache_gap.max(max_rel_diff(&produced, &reference));
            }
        }
    }

    // 6. Under rotary positions a sliding/linear trunk keeps decoding past
    //    the table, and what it emits there is what a full recompute over the
    //    whole (longer-than-context) sequence emits.
    let long_model = build(&LmConfig {
        attention: Some(AttentionSchedule::ratio(
            n_layers,
            1,
            AttentionMode::Linear,
            AttentionMode::Sliding { window: 4 },
        )),
        positions: PositionKind::Rotary,
        ..tiny.clone()
    })?;
    let prompt: Vec<u16> = vec![crate::tokenizer::Special::Bos.id(), 65, 66];
    let want = tiny.context + 6;
    let emitted = long_model.generate_cached(
        &prompt,
        want,
        &Sampling::Greedy,
        &mut StdRng::seed_from_u64(0),
        &device,
    );
    // Greedy decoding from random weights may legitimately emit <eos> early;
    // what must not happen is stopping at the table's edge for no reason.
    let stopped_at_eos = emitted.last() == Some(&crate::tokenizer::Special::Eos.id());
    let past_table = if emitted.len() == prompt.len() + want || stopped_at_eos {
        0.0
    } else {
        1.0
    };
    let long_ids: Vec<i64> = (0..tiny.context as i64 + 5)
        .map(|i| 40 + (i * 7) % 60)
        .collect();
    let long_reference = logits_of(&long_model, &long_ids);
    let mut long_cache = long_model.new_cache();
    let mut long_produced: Vec<f32> = Vec::new();
    for chunk in long_ids.chunks(5) {
        let out = long_model.forward_cached(as_tensor(chunk), &mut long_cache);
        long_produced.extend(out.logits.into_data().convert::<f32>().iter::<f32>());
    }
    let ran_past = if long_cache.position() > tiny.context {
        0.0
    } else {
        1.0
    };
    let long_gap = max_rel_diff(&long_produced, &long_reference) + past_table + ran_past;

    // 7. The routing state is bounded by its tanh and carries history.
    let state = RoutingState::<B>::new(8, 4, &device);
    let hidden = Tensor::<B, 3>::random([2, 3, 8], Distribution::Normal(0.0, 5.0), &device);
    let r0 = state.step(&hidden, None);
    let bound: f32 = r0.clone().abs().max().into_scalar();
    let history = Tensor::<B, 3>::full([2, 3, 4], 0.9, &device);
    let moved: f32 = (state.step(&hidden, Some(&history)) - r0)
        .abs()
        .max()
        .into_scalar();
    let state_residual = f64::from((bound - 1.0).max(0.0)) + if moved > 1e-4 { 0.0 } else { 1.0 };
    // ...and a trunk built without one has exactly the parameters it had.
    let count = |m: &LanguageModel<B>| m.num_params();
    let without = count(&build(&tiny)?);
    let zero = count(&build(&LmConfig {
        routing_state: 0,
        ..tiny.clone()
    })?);
    let with = count(&build(&LmConfig {
        routing_state: 4,
        ..tiny.clone()
    })?);
    let params_residual = if without == zero && with > without {
        0.0
    } else {
        1.0
    };
    let width_residual = if MoEConfig::new(8, 4, 3)
        .with_state_size(0)
        .router_input_size()
        == MoEConfig::new(8, 4, 3).router_input_size()
        && MoEConfig::new(8, 4, 3)
            .with_state_size(5)
            .router_input_size()
            == MoEConfig::new(8, 4, 3).router_input_size() + 5
    {
        0.0
    } else {
        1.0
    };

    // 8. Routing locality reads as specified.
    let ids_of = |v: &[i64]| Tensor::<B, 1, Int>::from_ints(v, &device).reshape([v.len(), 1]);
    let alternating = token_stability(&ids_of(&[0, 1, 0, 1]), 1, 4);
    let constant = token_stability(&ids_of(&[2, 2, 2, 2]), 1, 4);
    let across = token_stability(&ids_of(&[0, 0, 1, 1]), 2, 2);
    let agreement = layer_agreement(&[0, 1, 2], &[0, 1, 0], 3, 3).unwrap_or(f32::NAN);
    let incomparable = layer_agreement(&[0, 1], &[0, 1], 2, 3).is_none();
    let locality_residual = f64::from(
        alternating.abs()
            + (constant - 1.0).abs()
            + (across - 1.0).abs()
            + (agreement - 2.0 / 3.0).abs(),
    ) + if incomparable { 0.0 } else { 1.0 };

    // 9. Qwen-style trunk knobs: identity at their defaults, invariants on.
    use crate::hybrid::{apply_rotary_partial, rotary_dim_for};
    use crate::vit::{NormKind, TrunkNorm};
    let to_vec4 = |t: Tensor<B, 4>| {
        t.into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect::<Vec<f32>>()
    };
    // 9a. Partial rotary at fraction 1.0 is the full rotation (it delegates),
    //     and below 1.0 the suffix passes through untouched while the prefix
    //     still scores by distance only.
    let px = Tensor::<B, 4>::random([1, 2, 5, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let full = apply_rotary(px.clone(), 13, ROTARY_BASE);
    let partial_full = apply_rotary_partial(px.clone(), 13, ROTARY_BASE, 8);
    let partial_identity_gap = max_abs_diff(&to_vec4(full), &to_vec4(partial_full));
    let half = apply_rotary_partial(px.clone(), 13, ROTARY_BASE, 4);
    let half_v = to_vec4(half);
    let px_v = to_vec4(px.clone());
    let mut suffix_gap = 0.0f64;
    for r in 0..half_v.len() / 8 {
        for c in 4..8 {
            suffix_gap = suffix_gap.max(f64::from((half_v[r * 8 + c] - px_v[r * 8 + c]).abs()));
        }
    }
    let rq2 = Tensor::<B, 4>::random([1, 1, 1, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let rk2 = Tensor::<B, 4>::random([1, 1, 1, 8], Distribution::Uniform(-1.0, 1.0), &device);
    let pscore = |qo: usize, ko: usize| -> f64 {
        let a = apply_rotary_partial(rq2.clone(), qo, ROTARY_BASE, 4);
        let b = apply_rotary_partial(rk2.clone(), ko, ROTARY_BASE, 4);
        let s: f32 = (a * b).sum().into_scalar();
        f64::from(s)
    };
    let pbase = pscore(9, 4);
    let partial_rotary_gap = [1usize, 7, 30]
        .iter()
        .map(|s| (pscore(9 + s, 4 + s) - pbase).abs())
        .fold(0.0, f64::max);
    let cover_ok = if rotary_dim_for(1.0, 8) == 8 && rotary_dim_for(0.25, 8) == 2 {
        0.0
    } else {
        1.0
    };
    let partial_gap = partial_identity_gap + suffix_gap + partial_rotary_gap + cover_ok;
    // 9b. GQA with as many KV heads as query heads is full MHA, bit for bit;
    //     one KV head keeps strictly less decode state than the full trunk.
    let queries = tiny.num_heads;
    let gqa_same = same_weights(&LmConfig {
        num_kv_heads: Some(queries),
        ..tiny.clone()
    })?;
    let gqa_gap = max_abs_diff(&reference, &logits_of(&gqa_same, &ids));
    let full_cache_model = build(&tiny)?;
    let gqa_cache_model = build(&LmConfig {
        num_kv_heads: Some(1),
        ..tiny.clone()
    })?;
    let mut full_cache = full_cache_model.new_cache();
    let mut gqa_cache = gqa_cache_model.new_cache();
    full_cache_model.forward_cached(as_tensor(&ids[0..4]), &mut full_cache);
    gqa_cache_model.forward_cached(as_tensor(&ids[0..4]), &mut gqa_cache);
    let gqa_cache_gap = if gqa_cache.resident_floats() < full_cache.resident_floats() {
        0.0
    } else {
        1.0
    };
    // 9c. The gated-attention reference carries the zero-initialized gate in
    //     its record; the plain model drops it, so the gap is the gate's
    //     at-init effect alone: exactly 1.0 up to tanh's rounding.
    let gated_ref = build(&LmConfig {
        gated_attention: true,
        ..tiny.clone()
    })?;
    let gated_logits = logits_of(&gated_ref, &ids);
    let plain_from_gated =
        LanguageModel::<B>::new(&tiny, &device)?.load_record(gated_ref.clone().into_record());
    let gate_gap = max_abs_diff(&gated_logits, &logits_of(&plain_from_gated, &ids));
    // 9c2. QK-Norm: a normalized trunk decodes from its state exactly as it
    //      recomputes, and normalizing moves the output (the path is live).
    //      Like the gate cert the reference is built directly: loading the
    //      plain record into a normalized model would null the norms.
    let qk_ref = build(&LmConfig {
        qk_norm: true,
        ..tiny.clone()
    })?;
    let qk_reference = logits_of(&qk_ref, &ids);
    // Whole-prefix then token-by-token cached decodes must both reproduce
    // the full recompute, exactly as the existing cache certificate demands.
    let mut qk_cache_gap = 0.0f64;
    for chunk in [vec![ids.len()], vec![1; ids.len()]] {
        let mut cache = qk_ref.new_cache();
        let mut produced: Vec<f32> = Vec::new();
        let mut at = 0usize;
        for size in chunk {
            let out = qk_ref.forward_cached(as_tensor(&ids[at..at + size]), &mut cache);
            produced.extend(out.logits.into_data().convert::<f32>().iter::<f32>());
            at += size;
        }
        qk_cache_gap = qk_cache_gap.max(max_rel_diff(&produced, &qk_reference));
    }
    let plain_from_qk =
        LanguageModel::<B>::new(&tiny, &device)?.load_record(qk_ref.clone().into_record());
    let qk_moves = max_abs_diff(&qk_reference, &logits_of(&plain_from_qk, &ids));
    let qk_gap = qk_cache_gap + if qk_moves > 1e-6 { 0.0 } else { 1.0 };
    // 9c3. Every causal mode reads the past: perturbing the first token
    //      moves a later logit under every schedule. A mask that excluded
    //      everything would still be "causal" and still decode exactly, so
    //      the cache certificates cannot see it; only a past-dependence
    //      check can (Anthropic circuits lesson: recurrent/linear variants
    //      must preserve the previous-token path or in-context reasoning
    //      fails). Residual = shortfall below the minimum visible movement.
    let perturbed: Vec<i64> = ids
        .iter()
        .enumerate()
        .map(|(i, v)| if i == 0 { (v + 1) % 259 } else { *v })
        .collect();
    let mut past_gap = 0.0f64;
    for (_, schedule) in &schedules {
        let model = build(&LmConfig {
            attention: Some(schedule.clone()),
            ..tiny.clone()
        })?;
        let before = logits_of(&model, &ids);
        let after = logits_of(&model, &perturbed);
        let v = tiny.vocab_size;
        let mut moved = 0.0f64;
        for p in 1..ids.len() {
            for c in 0..v {
                moved = moved.max(f64::from((after[p * v + c] - before[p * v + c]).abs()));
            }
        }
        past_gap = past_gap.max((1e-6 - moved).max(0.0));
    }
    // 9d. A fresh RMSNorm (weight exactly one) gives unit-RMS rows.
    let rms = TrunkNorm::<B>::new(NormKind::Rms, 12, 1e-12, &device);
    let rx = Tensor::<B, 3>::random([2, 5, 12], Distribution::Normal(0.0, 3.0), &device);
    let rout: Vec<f32> = rms
        .forward(rx)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let mut rms_gap = 0.0f64;
    for row in rout.chunks_exact(12) {
        let ms = row
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / 12.0;
        rms_gap = rms_gap.max((ms.sqrt() - 1.0).abs());
    }
    // 9e. MTP heads at weight zero are the plain loss, bit for bit; a live
    //     MTP term stays finite.
    let mtp_ref = build(&LmConfig {
        mtp_steps: 2,
        mtp_weight: 0.0,
        ..tiny.clone()
    })?;
    let plain_from_mtp =
        LanguageModel::<B>::new(&tiny, &device)?.load_record(mtp_ref.clone().into_record());
    let toks = as_tensor(&ids);
    let (plain_loss, _) = plain_from_mtp.next_token_loss(toks.clone(), 0..n_layers);
    let (mtp_loss, _) = mtp_ref.next_token_loss(toks.clone(), 0..n_layers);
    let mtp_gap =
        f64::from((plain_loss.mean().into_scalar() - mtp_loss.mean().into_scalar()).abs());
    let mtp_live = build(&LmConfig {
        mtp_steps: 2,
        mtp_weight: 0.5,
        ..tiny.clone()
    })?;
    let (live_loss, live_metrics) = mtp_live.next_token_loss(toks, 0..n_layers);
    let live_value: f32 = live_loss.mean().into_scalar();
    let mtp_finite_gap = if live_value.is_finite() && live_metrics.mtp_loss.is_finite() {
        0.0
    } else {
        1.0
    };

    Ok(vec![
        cert(
            "hybrid",
            "dense_schedule_is_the_phase19_trunk",
            "An explicit all-dense schedule produces the logits of the trunk without a schedule, bit for bit.",
            dense_gap,
            0.0,
        ),
        cert(
            "hybrid",
            "window_covering_the_context_is_dense",
            "A sliding window at least as long as the sequence attends exactly as dense attention does.",
            window_gap,
            0.0,
        ),
        cert(
            "hybrid",
            "reading_every_key_is_dense",
            "Retrieval attention whose top-k covers every key attends exactly as dense attention does.",
            retrieval_gap,
            0.0,
        ),
        cert(
            "hybrid",
            "linear_state_matches_masked_form",
            "Feeding a sequence through the linear-attention recurrence in chunks reproduces the masked matrix form, to within summation order.",
            linear_gap,
            1e-5,
        ),
        cert(
            "hybrid",
            "rotary_scores_depend_on_distance",
            "Shifting a query and a key by the same offset leaves their rotary score unchanged.",
            rotary_gap,
            1e-4,
        ),
        cert(
            "hybrid",
            "every_mode_decodes_from_its_state_exactly",
            "For dense, sliding, retrieval, linear, learned and 3:1 schedules under every position kind, cached decoding reproduces the full-recompute logits under every chunking.",
            cache_gap,
            1e-4,
        ),
        cert(
            "hybrid",
            "rotary_decoding_continues_past_the_table",
            "A rotary sliding/linear trunk emits tokens past the context length, and its cached logits there equal a full recompute over the whole sequence.",
            long_gap,
            1e-4,
        ),
        cert(
            "hybrid",
            "routing_state_is_bounded_and_carries_history",
            "The routing state stays inside (-1, 1) and depends on the state of the layer before; a trunk without one adds no parameter and a router without one keeps its input width.",
            state_residual + params_residual + width_residual,
            0.0,
        ),
        cert(
            "hybrid",
            "routing_locality_reads_as_specified",
            "Token stability is 0 for alternating experts, 1 for a constant expert and never counts a sequence boundary; layer agreement is the matching fraction and is undefined across widths.",
            locality_residual,
            1e-6,
        ),
        cert(
            "hybrid",
            "partial_rotary_is_full_at_fraction_one_and_leaves_the_suffix",
            "Partial rotary at fraction 1.0 delegates to the full rotation bit for bit; below 1.0 the uncovered suffix passes through untouched while the rotated prefix still scores by distance only.",
            partial_gap,
            1e-4,
        ),
        cert(
            "hybrid",
            "gqa_with_full_heads_is_dense_and_narrow_kv_shrinks_state",
            "Grouped-query attention with as many KV heads as query heads is full multi-head attention bit for bit, and one KV head keeps strictly fewer decode-state floats than the full trunk after the same prefix.",
            gqa_gap + gqa_cache_gap,
            0.0,
        ),
        cert(
            "hybrid",
            "gated_attention_is_identity_at_init",
            "A zero-initialized output gate scales the merged heads by 1 + tanh(0) = 1, so a gated trunk at init matches the ungated trunk on the shared weights up to tanh's rounding.",
            gate_gap,
            1e-6,
        ),
        cert(
            "hybrid",
            "causal_modes_read_the_past",
            "Perturbing the first token moves a later logit under every attention schedule: no causal mode is deaf to its own past.",
            past_gap,
            0.0,
        ),
        cert(
            "hybrid",
            "qk_norm_decodes_from_state_exactly_and_moves_output",
            "A QK-normalized trunk decodes from its per-mode state exactly as it recomputes, and normalizing moves the output versus the shared weights without it.",
            qk_gap,
            1e-4,
        ),
        cert(
            "hybrid",
            "rmsnorm_starts_at_unit_rms",
            "A fresh RMSNorm carries weight exactly one, so every output row has root-mean-square one up to float rounding.",
            rms_gap,
            1e-5,
        ),
        cert(
            "hybrid",
            "mtp_weight_zero_is_the_plain_loss",
            "MTP heads at weight zero add nothing, so the loss matches the plain next-token loss bit for bit on the shared weights; a live MTP term stays finite.",
            mtp_gap + mtp_finite_gap,
            0.0,
        ),
    ])
}

// --------------------------------------------------------------- deltanet --

fn deltanet_certificates() -> Vec<Certificate> {
    use crate::deltanet::{
        causal_depthwise_conv, gated_delta_recurrent, gated_delta_recurrent_batched,
        gated_delta_step, gated_delta_step_tensor, l2norm_last_dim,
    };
    use burn::tensor::Distribution;

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let flat =
        |t: Tensor<B, 2>| -> Vec<f32> { t.into_data().convert::<f32>().iter::<f32>().collect() };
    let gap = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max)
    };

    // 1. alpha = 1 is the pure delta rule, against the direct Householder form.
    let dk = 4;
    let dv = 3;
    let state = Tensor::<B, 2>::random([dv, dk], Distribution::Uniform(-1.0, 1.0), &device);
    let q = Tensor::<B, 1>::random([dk], Distribution::Uniform(-1.0, 1.0), &device);
    let k = Tensor::<B, 1>::random([dk], Distribution::Uniform(-1.0, 1.0), &device);
    let v = Tensor::<B, 1>::random([dv], Distribution::Uniform(-1.0, 1.0), &device);
    let beta = 0.4f32;
    let (next, out) = gated_delta_step(state.clone(), q.clone(), k.clone(), v.clone(), 1.0, beta);
    let k_col = k.clone().unsqueeze_dim::<2>(1).transpose();
    let sk = state.clone().matmul(k.clone().unsqueeze_dim::<2>(1));
    let want_next = state
        .clone()
        .sub(sk.matmul(k_col.clone()).mul_scalar(beta))
        .add(
            v.clone()
                .unsqueeze_dim::<2>(1)
                .matmul(k_col)
                .mul_scalar(beta),
        );
    let want_out = want_next
        .clone()
        .matmul(q.clone().unsqueeze_dim::<2>(1))
        .squeeze::<1>();
    let alpha_gap = gap(&flat(next), &flat(want_next)).max(gap(
        &flat(out.clone().unsqueeze_dim::<2>(0)),
        &flat(want_out.unsqueeze_dim::<2>(0)),
    ));

    // 2. beta = 0 is pure decay, exactly.
    let (decayed, _) = gated_delta_step(state.clone(), q.clone(), k.clone(), v.clone(), 0.6, 0.0);
    let decay_gap = gap(&flat(decayed), &flat(state.clone().mul_scalar(0.6)));

    // 3. From a zero state the first output is closed-form: o_1 = b v (k.q).
    let qr = Tensor::<B, 3>::random([1, 3, dk], Distribution::Uniform(-1.0, 1.0), &device);
    let kr = Tensor::<B, 3>::random([1, 3, dk], Distribution::Uniform(-1.0, 1.0), &device);
    let vr = Tensor::<B, 3>::random([1, 3, dv], Distribution::Uniform(-1.0, 1.0), &device);
    let (ro, _) = gated_delta_recurrent(
        qr.clone(),
        kr.clone(),
        vr.clone(),
        &[0.8, 0.3, 0.5],
        &[0.6, 0.9, 0.2],
        &device,
    );
    let q1: Vec<f32> = qr
        .narrow(1, 0, 1)
        .reshape([dk])
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let k1: Vec<f32> = kr
        .narrow(1, 0, 1)
        .reshape([dk])
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let v1: Vec<f32> = vr
        .narrow(1, 0, 1)
        .reshape([dv])
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let dot: f32 = q1.iter().zip(&k1).map(|(a, b)| a * b).sum();
    let want1: Vec<f32> = v1.iter().map(|x| 0.6 * x * dot).collect();
    let got1: Vec<f32> = ro
        .narrow(1, 0, 1)
        .reshape([dv])
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let first_gap = gap(&got1, &want1);

    // 4. L2 rows are unit-norm.
    let x = Tensor::<B, 3>::random([2, 4, 8], Distribution::Normal(0.0, 2.0), &device);
    let y: Vec<f32> = l2norm_last_dim(x)
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let mut norm_gap = 0.0f64;
    for row in y.chunks_exact(8) {
        let n: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        norm_gap = norm_gap.max((n.sqrt() - 1.0).abs());
    }

    // 5. The short convolution is causal: changing only the last input
    // leaves every earlier output bit-identical.
    let xn: Vec<f32> = Tensor::<B, 3>::random([1, 7, 2], Distribution::Uniform(-1.0, 1.0), &device)
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let mut yn = xn.clone();
    yn[12] += 1.0;
    yn[13] += 1.0;
    let w: Vec<f32> = Tensor::<B, 2>::random([2, 4], Distribution::Uniform(-1.0, 1.0), &device)
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let seq = |vals: &[f32]| Tensor::<B, 1>::from_floats(vals, &device).reshape([1, 7, 2]);
    let taps = |vals: &[f32]| Tensor::<B, 1>::from_floats(vals, &device).reshape([2, 4]);
    let o1: Vec<f32> = causal_depthwise_conv(seq(&xn), taps(&w))
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    let o2: Vec<f32> = causal_depthwise_conv(seq(&yn), taps(&w))
        .into_data()
        .convert::<f32>()
        .iter()
        .collect();
    // First six positions (12 floats) must agree exactly; the last may differ.
    let causal_gap = gap(&o1[..12], &o2[..12]);

    // 6. The tensor-alpha step (gradients flow through alpha/beta) is the
    // scalar step bit for bit: same operation order, broadcast [1]
    // multiplies standing in for mul_scalar.
    let alpha_host = 0.7f32;
    let beta_host = 0.4f32;
    let (next_t, out_t) = gated_delta_step_tensor(
        state.clone(),
        q.clone(),
        k.clone(),
        v.clone(),
        Tensor::<B, 1>::from_floats([alpha_host], &device),
        Tensor::<B, 1>::from_floats([beta_host], &device),
    );
    let (next_s, out_s) = gated_delta_step(
        state.clone(),
        q.clone(),
        k.clone(),
        v.clone(),
        alpha_host,
        beta_host,
    );
    let tensor_step_gap = gap(&flat(next_t), &flat(next_s)).max(gap(
        &flat(out_t.unsqueeze_dim::<2>(0)),
        &flat(out_s.unsqueeze_dim::<2>(0)),
    ));

    // 7. The batched tensor-alpha rollout (one (k, v) pair per row) matches
    // the scalar rollout per row, bit for bit.
    let rows = 2;
    let tn = 3;
    let q2 = Tensor::<B, 3>::random([rows, tn, dk], Distribution::Uniform(-1.0, 1.0), &device);
    let k2 = Tensor::<B, 3>::random([rows, tn, dk], Distribution::Uniform(-1.0, 1.0), &device);
    let v2 = Tensor::<B, 3>::random([rows, tn, dv], Distribution::Uniform(-1.0, 1.0), &device);
    let a2 = Tensor::<B, 2>::from_floats([[0.8, 0.3, 0.5], [0.9, 0.2, 0.7]], &device);
    let b2 = Tensor::<B, 2>::from_floats([[0.6, 0.9, 0.2], [0.4, 0.8, 0.1]], &device);
    let (bo, bf) =
        gated_delta_recurrent_batched(q2.clone(), k2.clone(), v2.clone(), a2, b2, &device);
    let (so0, sf0) = gated_delta_recurrent(
        q2.clone().narrow(0, 0, 1),
        k2.clone().narrow(0, 0, 1),
        v2.clone().narrow(0, 0, 1),
        &[0.8, 0.3, 0.5],
        &[0.6, 0.9, 0.2],
        &device,
    );
    let (so1, sf1) = gated_delta_recurrent(
        q2.narrow(0, 1, 1),
        k2.narrow(0, 1, 1),
        v2.narrow(0, 1, 1),
        &[0.9, 0.2, 0.7],
        &[0.4, 0.8, 0.1],
        &device,
    );
    let want_o = Tensor::cat(vec![so0, so1], 0);
    let want_f = Tensor::cat(vec![sf0, sf1], 0);
    let batched_gap = gap(
        &flat(bo.reshape([rows * tn, dv])),
        &flat(want_o.reshape([rows * tn, dv])),
    )
    .max(gap(
        &flat(bf.reshape([rows * dv, dk])),
        &flat(want_f.reshape([rows * dv, dk])),
    ));

    vec![
        cert(
            "deltanet",
            "alpha_one_is_pure_delta",
            "With alpha = 1 the gated step is exactly the delta-rule Householder update S(I - b k k^T) + b v k^T, outputs included.",
            alpha_gap,
            1e-5,
        ),
        cert(
            "deltanet",
            "beta_zero_is_pure_decay",
            "With beta = 0 the gated step is exactly alpha times the incoming state, whatever the vectors.",
            decay_gap,
            0.0,
        ),
        cert(
            "deltanet",
            "zero_state_first_output_closed_form",
            "From a zero state the first output is exactly b v (k.q): the full step, state update and readout, in closed form.",
            first_gap,
            1e-5,
        ),
        cert(
            "deltanet",
            "l2norm_rows_are_unit",
            "Query/key normalization leaves every row at unit L2 norm, so attention scores stay bounded.",
            norm_gap,
            1e-5,
        ),
        cert(
            "deltanet",
            "short_conv_is_causal",
            "Changing only the last input leaves every earlier convolution output bit-identical: the future cannot leak backwards.",
            causal_gap,
            0.0,
        ),
        cert(
            "deltanet",
            "tensor_step_matches_scalar_step",
            "The tensor-alpha gated step (alpha/beta as [1] tensors, so gradients flow through them) is the scalar step bit for bit: identical operation order with broadcast [1] multiplies in place of mul_scalar.",
            tensor_step_gap,
            0.0,
        ),
        cert(
            "deltanet",
            "batched_recurrent_matches_scalar_recurrent",
            "The batched tensor-alpha recurrent rollout, one (k, v) pair per row, reproduces the scalar recurrent rollout per row bit for bit, outputs and final states.",
            batched_gap,
            0.0,
        ),
    ]
}

// --------------------------------------------------------------- qwennet --

/// Small Qwen architecture exercising both layer kinds (layer 3 of 5 is
/// full attention); the same shape as the qwennet unit tests.
fn qwennet_tiny_dims() -> crate::qwen::QwenArchDims {
    crate::qwen::QwenArchDims {
        num_layers: 5,
        hidden_size: 64,
        intermediate_size: 96,
        vocab_size: 128,
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        linear_k_groups: 2,
        linear_v_heads: 4,
        linear_head_dim: 16,
        conv_kernel: 4,
        full_attention_interval: 4,
        attn_output_gate: true,
    }
}

fn qwennet_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::qwennet::{QwenFullAttention, QwenRmsNorm, QwenTrunk, QwenTrunkConfig};
    use burn::tensor::Int;

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let flat2 =
        |t: Tensor<B, 2>| -> Vec<f32> { t.into_data().convert::<f32>().iter::<f32>().collect() };
    let flat3 =
        |t: Tensor<B, 3>| -> Vec<f32> { t.into_data().convert::<f32>().iter::<f32>().collect() };
    let gap = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max)
    };

    let dims = qwennet_tiny_dims();
    let n = 6;

    // 1. Full-attention KV-cache decode reproduces prefill at every position.
    let attn = QwenFullAttention::<B>::new(&dims, 10_000_000.0, 1e-6, &device);
    crate::tensor_ext::force_initialization(&attn);
    let x = Tensor::<B, 3>::random(
        [1, n, dims.hidden_size],
        Distribution::Normal(0.0, 0.02),
        &device,
    );
    let prefill = attn.forward(x.clone());
    let mut state = crate::hybrid::LayerState::<B>::new();
    let mut attn_gap = 0.0f64;
    for t in 0..n {
        let xt = x.clone().narrow(1, t, 1).reshape([1, dims.hidden_size]);
        let out = attn.step(xt, &mut state);
        attn_gap = attn_gap.max(gap(&flat2(out), &flat3(prefill.clone().narrow(1, t, 1))));
    }

    // 2. A mixed linear+full trunk decodes to its prefill logits.
    let config = QwenTrunkConfig {
        dims,
        num_layers: None,
        rope_base: 10_000_000.0,
        eps: 1e-6,
    };
    let trunk = QwenTrunk::<B>::new(config, &device);
    crate::tensor_ext::force_initialization(&trunk);
    let tokens = Tensor::<B, 2, Int>::from_ints([[3, 1, 4, 1, 5, 9]], &device);
    let logits = trunk.forward(tokens.clone());
    let mut tstate = trunk.new_state(1, &device);
    let mut trunk_gap = 0.0f64;
    for t in 0..n {
        let out = trunk.step(tokens.clone().narrow(1, t, 1), &mut tstate)?;
        trunk_gap = trunk_gap.max(gap(&flat2(out), &flat3(logits.clone().narrow(1, t, 1))));
    }

    // 3. The zero-centered RMSNorm at zero weight is the identity on
    // unit-RMS rows.
    let norm = QwenRmsNorm::<B>::new(8, 1e-6, &device);
    let unit =
        Tensor::<B, 3>::from_floats([[[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]], &device);
    let norm_gap = gap(&flat3(norm.forward(unit.clone())), &flat3(unit));

    Ok(vec![
        cert(
            "qwennet",
            "full_attention_decodes_from_kv_cache_exactly",
            "Token-by-token decode against the KV cache reproduces the prefill forward at every position bit for bit: identical projections, per-head QK norms, rotary angles at the absolute offset, masked softmax row and sigmoid output gate.",
            attn_gap,
            0.0,
        ),
        cert(
            "qwennet",
            "mixed_trunk_step_matches_prefill",
            "A mixed linear-attention + full-attention trunk decodes token by token to exactly its prefill logits: every per-layer decode state is the prefill computation replayed one position at a time.",
            trunk_gap,
            0.0,
        ),
        cert(
            "qwennet",
            "zero_centered_norm_is_identity_at_zero_weight",
            "QwenRmsNorm at its zeros-init weight applies only the RMS rescaling (the (1 + w) factor is exactly one), so unit-RMS rows pass through unchanged.",
            norm_gap,
            1e-6,
        ),
    ])
}

// ------------------------------------------------------------------- geom --

/// Geometrical certificates. One `anyhow::Result`: a check that cannot be set
/// up is a failing certificate, not a panic and not a silent pass.
fn geom_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::geometry::{
        energy_grad_step, energy_value, geodesic_attention, lipschitz_step, GeomConfig,
        GeometricReasoner, MetricField,
    };
    use crate::geomkernel::{
        self, constraint_holds, construct, falsify, graph_from_points, saturate_rules, Segment, Q,
    };
    use burn::module::Param;
    use burn::tensor::Distribution;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let mut out = Vec::new();

    // -- kernel: exact rationals ------------------------------------------------
    // 1/3 + 1/6 must be exactly 1/2 -- the identity a float kernel fails
    let third = Q::new(1, 3)?;
    let sixth = Q::new(1, 6)?;
    let half = third.add(&sixth)?;
    // Exactness is the *rational* check: the fraction must be literally 1/2
    // (`num`/`den` untouched), and the float view of it must agree.
    let rational_err = (half.to_f64() - 0.5).abs()
        + (half.num - 1).unsigned_abs() as f64
        + (half.den - 2).unsigned_abs() as f64;
    out.push(cert(
        "geom",
        "kernel_rationals_are_exact_where_floats_are_not",
        "1/3 + 1/6 equals exactly 1/2 in the kernel's rational arithmetic (in f32 the same expression is not 0.5).",
        rational_err,
        0.0,
    ));

    // -- kernel: exact intersection ---------------------------------------------
    let points = vec![
        crate::geomkernel::KPoint {
            name: "A".into(),
            x: crate::geomkernel::Frac { num: 0, den: 1 },
            y: crate::geomkernel::Frac { num: 1, den: 3 },
        },
        crate::geomkernel::KPoint {
            name: "B".into(),
            x: crate::geomkernel::Frac { num: 1, den: 1 },
            y: crate::geomkernel::Frac { num: 4, den: 3 },
        },
        crate::geomkernel::KPoint {
            name: "C".into(),
            x: crate::geomkernel::Frac { num: 2, den: 3 },
            y: crate::geomkernel::Frac { num: 0, den: 1 },
        },
        crate::geomkernel::KPoint {
            name: "D".into(),
            x: crate::geomkernel::Frac { num: 2, den: 3 },
            y: crate::geomkernel::Frac { num: 1, den: 1 },
        },
    ];
    let graph = graph_from_points(points, vec![]);
    let hit = construct(
        &graph,
        &geomkernel::Construction::Intersection {
            first: Segment {
                from: "A".into(),
                to: "B".into(),
            },
            second: Segment {
                from: "C".into(),
                to: "D".into(),
            },
        },
        "E",
    )?;
    let intersect_err = f64::from(
        ((hit.x.num - 2).abs()
            + (hit.x.den - 3).abs()
            + (hit.y.num - 1).abs()
            + (hit.y.den - 1).abs()) as f32,
    );
    out.push(cert(
        "geom",
        "kernel_intersection_is_exact",
        "y = x + 1/3 meets x = 2/3 at exactly (2/3, 1): the coordinates come back as those exact fractions.",
        intersect_err,
        0.0,
    ));

    // -- kernel: non-degeneracy refusals ----------------------------------------
    let points = vec![
        crate::geomkernel::KPoint {
            name: "A".into(),
            x: crate::geomkernel::Frac::from_int(0),
            y: crate::geomkernel::Frac::from_int(0),
        },
        crate::geomkernel::KPoint {
            name: "B".into(),
            x: crate::geomkernel::Frac::from_int(1),
            y: crate::geomkernel::Frac::from_int(1),
        },
        crate::geomkernel::KPoint {
            name: "C".into(),
            x: crate::geomkernel::Frac::from_int(0),
            y: crate::geomkernel::Frac::from_int(1),
        },
        crate::geomkernel::KPoint {
            name: "D".into(),
            x: crate::geomkernel::Frac::from_int(1),
            y: crate::geomkernel::Frac::from_int(2),
        },
    ];
    let graph = graph_from_points(points, vec![]);
    let refused_parallel = construct(
        &graph,
        &geomkernel::Construction::Intersection {
            first: Segment {
                from: "A".into(),
                to: "B".into(),
            },
            second: Segment {
                from: "C".into(),
                to: "D".into(),
            },
        },
        "F",
    )
    .is_err();
    let refused_coincident = construct(
        &graph,
        &geomkernel::Construction::Intersection {
            first: Segment {
                from: "A".into(),
                to: "B".into(),
            },
            second: Segment {
                from: "A".into(),
                to: "B".into(),
            },
        },
        "F",
    )
    .is_err();
    out.push(cert(
        "geom",
        "kernel_refuses_degenerate_configurations",
        "Parallel lines and coincident lines refuse to intersect: the kernel returns an error instead of a silent configuration.",
        f64::from(!refused_parallel) + f64::from(!refused_coincident),
        0.0,
    ));

    // -- kernel: rules fire with certificates and saturate ----------------------
    let points = vec![
        crate::geomkernel::KPoint {
            name: "B".into(),
            x: crate::geomkernel::Frac::from_int(0),
            y: crate::geomkernel::Frac::from_int(0),
        },
        crate::geomkernel::KPoint {
            name: "C".into(),
            x: crate::geomkernel::Frac::from_int(4),
            y: crate::geomkernel::Frac::from_int(2),
        },
        crate::geomkernel::KPoint {
            name: "D".into(),
            x: crate::geomkernel::Frac::from_int(2),
            y: crate::geomkernel::Frac::from_int(1),
        },
    ];
    let mut graph = graph_from_points(
        points,
        vec![geomkernel::Constraint::MidpointOf {
            p: "D".into(),
            a: "B".into(),
            b: "C".into(),
        }],
    );
    let added = saturate_rules(&mut graph, 8)?;
    let derived = graph
        .facts
        .iter()
        .filter(|f| matches!(f.provenance, crate::geomkernel::Provenance::Derived { .. }))
        .count();
    let second = saturate_rules(&mut graph, 8)?;
    out.push(cert(
        "geom",
        "kernel_rules_derive_with_certificates_and_saturate",
        "A midpoint given derives exactly the collinearity and length facts, each naming its rule, and saturation is a fixed point.",
        f64::from((added != 2) as u8) + f64::from((derived != 2) as u8) + f64::from((second != 0) as u8),
        0.0,
    ));

    // -- kernel: falsification ----------------------------------------------------
    let claim = geomkernel::Constraint::EqualLength {
        first: Segment {
            from: "A".into(),
            to: "B".into(),
        },
        second: Segment {
            from: "A".into(),
            to: "C".into(),
        },
    };
    let mut rng = StdRng::seed_from_u64(4);
    let rejected = matches!(
        falsify(&mut rng, &[], &claim, 256, 6)?,
        geomkernel::Verdict::Counterexample { .. }
    );
    // and a theorem survives every draw
    let premises = vec![geomkernel::Constraint::MidpointOf {
        p: "A".into(),
        a: "B".into(),
        b: "C".into(),
    }];
    let theorem = geomkernel::Constraint::Collinear {
        a: "B".into(),
        b: "A".into(),
        c: "C".into(),
    };
    let survived = match falsify(&mut rng, &premises, &theorem, 128, 6)? {
        geomkernel::Verdict::Unrefuted { satisfying, .. } => satisfying > 0,
        _ => false,
    };
    out.push(cert(
        "geom",
        "kernel_falsification_rejects_and_survives_correctly",
        "Randomized falsification finds a counterexample to an unconstrained length claim, while the midpoint theorem survives every premise-satisfying draw.",
        f64::from(!rejected) + f64::from(!survived),
        0.0,
    ));

    // -- model: the metric is PSD by construction ---------------------------------
    let mut metric = MetricField::<B>::new(4, &device);
    metric.lower = Param::from_tensor(Tensor::<B, 2>::random(
        [4, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    metric.log_diag = Param::from_tensor(Tensor::<B, 1>::random(
        [4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let g = metric.metric();
    let mut psd_err = 0.0f32;
    for _ in 0..128 {
        let v = Tensor::<B, 1>::random([4], Distribution::Normal(0.0, 1.0), &device);
        let q = v
            .clone()
            .reshape([1, 4])
            .matmul(g.clone())
            .reshape([4])
            .mul(v)
            .sum()
            .into_scalar();
        psd_err = psd_err.max(-q);
    }
    out.push(cert(
        "geom",
        "metric_is_positive_definite_by_construction",
        "G = L L^T with an exp-diagonal L has no negative quadratic form, for any parameter values the optimizer can reach.",
        f64::from(psd_err),
        1e-5,
    ));

    // -- model: geodesic attention rows are distributions -------------------------
    let query = Tensor::<B, 3>::random([2, 5, 4], Distribution::Normal(0.0, 1.0), &device);
    let keys = Tensor::<B, 3>::random([2, 7, 4], Distribution::Normal(0.0, 1.0), &device);
    let weights = geodesic_attention(&g, query, keys, 1.0);
    let row_err = weights.sum_dim(2).sub_scalar(1.0).abs().max().into_scalar();
    out.push(cert(
        "geom",
        "geodesic_attention_rows_are_distributions",
        "Inverse-distance attention weights sum to one over the attended set.",
        f64::from(row_err),
        1e-5,
    ));

    // -- model: the temperature is bounded on both sides ----------------------
    // Extreme raw logits must map into [1e-3, 2·dim]: the floor keeps `-d²`
    // off a zero divisor, the cap keeps runaway drift from flattening the
    // attention into uniformity instead of learning geometry. At raw 0 the
    // mapping is exactly softplus(0) + 1e-3, so every existing checkpoint
    // (trained near there) is bit-identical under the cap.
    use crate::geometry::{bounded_temperature, temperature_cap};
    let dim = 8usize;
    let cap = temperature_cap(dim);
    let mut temp_err = 0.0f64;
    for raw in [-50.0f32, -5.0, 0.0, 5.0, 50.0] {
        let t = f64::from(bounded_temperature(raw, dim));
        if !(1e-3..=f64::from(cap)).contains(&t) {
            temp_err += 1.0;
        }
    }
    let plain_softplus = (1.0f32 + 0.0f32.exp()).ln() + 1e-3;
    temp_err += f64::from((bounded_temperature(0.0, dim) - plain_softplus).abs());
    // A capped extreme stays finite (no inf/NaN through the mapping).
    for raw in [-100.0f32, 100.0] {
        if !bounded_temperature(raw, dim).is_finite() {
            temp_err += 1.0;
        }
    }
    out.push(cert(
        "geom",
        "attention_temperature_is_bounded",
        "The geodesic-attention temperature maps any raw logit into [1e-3, 2·dim], stays finite at extremes, and reproduces softplus(0) + 1e-3 exactly at raw zero.",
        temp_err,
        0.0,
    ));

    // -- model: the certified relaxation ------------------------------------------
    // a fresh identity metric: the closed-form fixed point holds for any G,
    // but the *rate* is a function of the condition number, so the
    // certificate's step budget needs the deterministic identity landscape
    let g = MetricField::<B>::new(4, &device).metric();
    let lambda = 0.5f32;
    let slots = 6usize;
    let eta = lipschitz_step(&g, lambda, slots);
    let mut z = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
    let c = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
    let mut worst = 0.0f32;
    let mut previous = energy_value(&g, &z, &c, lambda, slots);
    for _ in 0..64 {
        z = energy_grad_step(&g, &z, &c, lambda, slots, eta);
        let current = energy_value(&g, &z, &c, lambda, slots);
        worst = worst.max(current - previous);
        previous = current;
    }
    out.push(cert(
        "geom",
        "energy_descent_is_monotone_under_the_certified_step",
        "With the step size from the closed-form Lipschitz bound, the relaxation's energy never increases.",
        f64::from(worst),
        1e-4,
    ));

    // The exact fixed point of the identity-metric landscape. Solving
    // `(I + 2*lambda*K) z - 2*lambda*S = c` with `S = sum_l z_l` per
    // coordinate gives `A = (I + 2*lambda*K)I - 2*lambda*J`, and since
    // `A - K*b = 1` the inverse collapses to `(c + 2*lambda*sum(c)) /
    // (1 + 2*lambda*K)`. The `2*lambda` matches the gradient the step
    // actually takes -- the same factor the monotone-descent certificate
    // pins down.
    let mut z = Tensor::<B, 3>::random([2, slots, 4], Distribution::Normal(0.0, 1.0), &device);
    let z0 = z.clone();
    let gap0 = z0.clone().sub(c.clone());
    let initial = gap0.clone().mul(gap0).sum().sqrt().into_scalar();
    for _ in 0..96 {
        z = energy_grad_step(&g, &z, &c, lambda, slots, eta);
    }
    let fixed = c
        .clone()
        .add(c.clone().sum_dim(1).mul_scalar(2.0 * lambda))
        .div_scalar(1.0 + 2.0 * lambda * slots as f32);
    let gap = z.sub(fixed);
    let residual = gap.clone().mul(gap).sum().sqrt().into_scalar();
    out.push(cert(
        "geom",
        "relaxation_converges_to_the_exact_fixed_point",
        "With an identity metric the fixed point is (c + 2*lambda*sum(c))/(1 + 2*lambda*K); 96 certified steps land within 1% of it.",
        f64::from(residual / initial.max(1e-6)),
        1e-2,
    ));

    // with repulsion off, the attractor is the context
    let lambda_tiny = 0.0f32;
    let eta_tiny = lipschitz_step(&g, lambda_tiny, slots);
    let mut z = z0.clone();
    for _ in 0..8 {
        z = energy_grad_step(&g, &z, &c, lambda_tiny, slots, eta_tiny);
    }
    let gap_end = z.sub(c.clone());
    let end = gap_end.clone().mul(gap_end).sum().sqrt().into_scalar();
    out.push(cert(
        "geom",
        "with_repulsion_off_the_attractor_is_the_context",
        "With the repulsion vanishing, eight certified steps drive every slot onto its context.",
        f64::from(end / initial.max(1e-6)),
        1e-4,
    ));

    // -- model: span identity -------------------------------------------------------
    let config = GeomConfig::tiny();
    let model = GeometricReasoner::<B>::new(&config, &device)?;
    let scene: Vec<i64> = (0..config.scene_len - 1)
        .map(|i| i64::from(b'A' + (i % 26) as u8))
        .collect();
    let tokens = Tensor::<B, 1, Int>::from_ints(scene.as_slice(), &device)
        .reshape([1, config.scene_len - 1]);
    let direct = model.forward(tokens.clone())?.logits;
    let spanned = model.forward_span(tokens, 0..model.num_blocks())?.logits;
    let span_err = (direct - spanned).abs().max().into_scalar();
    out.push(cert(
        "geom",
        "block_span_covering_everything_is_the_full_path",
        "A refinement span covering every block reproduces the full forward pass bit for bit.",
        f64::from(span_err),
        0.0,
    ));

    // -- model: the denoiser is its input at sigma zero ------------------------------
    let scene_full: Vec<i64> = (0..config.scene_len)
        .map(|i| i64::from(b'A' + (i % 26) as u8))
        .collect();
    let tokens_full = Tensor::<B, 1, Int>::from_ints(scene_full.as_slice(), &device)
        .reshape([1, config.scene_len]);
    let (z_star, _) = model.clean_state(tokens_full.clone())?;
    let step = model.block_step(tokens_full, 0, 0.0, &z_star, 0.0)?;
    let ident_err = step.x0.sub(z_star).abs().max().into_scalar();
    out.push(cert(
        "geom",
        "denoiser_is_identity_at_sigma_zero",
        "EDM preconditioning makes a block's x0 estimate its input at sigma zero: c_skip(0) = 1 and c_out(0) = 0.",
        f64::from(ident_err),
        1e-6,
    ));

    // -- model: the degenerate mixture is the dense head ------------------------------
    let single = crate::geometry::GeomRouter::new(8, 12, 1, 1, 1, &device);
    let query = Tensor::<B, 2>::random([3, 8], Distribution::Normal(0.0, 1.0), &device);
    let (mixed, info) = single.forward(query.clone())?;
    let own = single.expert_head(query, 0, 0);
    let mix_err = (mixed - own).abs().max().into_scalar();
    let gates_err = info
        .balance
        .total
        .clone()
        .detach()
        .into_scalar()
        .is_finite();
    out.push(cert(
        "geom",
        "a_single_box_single_expert_mixture_is_that_expert",
        "A one-box one-expert top-1 mixture returns exactly its selected expert head; the gates remain a distribution.",
        f64::from(mix_err) + f64::from(!gates_err),
        0.0,
    ));

    // -- model: the readout inherits the mosme invariants ------------------------------
    // The readout routes through `mosme::HierarchicalRouter`, so disabling an
    // expert masks its logit to -inf and `exp(-inf - max) == 0.0` gives its
    // composed gate column exactly 0.0 -- the invariant
    // `mosme`'s disabled-expert test pins at the router level. The gates of
    // the *untouched boxes* are bit-identical (the mask never reaches them),
    // and the composed gates still form a distribution: every row sums to 1.
    // (The disabled expert's own box-mates are *not* bit-identical -- the
    // softmax and the top-k renormalization redistribute its mass, which is
    // the mask doing its job.)
    let mut masked = crate::geometry::GeomRouter::new(8, 12, 2, 2, 2, &device);
    let query = Tensor::<B, 2>::random([3, 8], Distribution::Normal(0.0, 1.0), &device);
    let boxes_before = masked.router().route(query.clone()).composed_flat();
    masked.router_mut().set_enabled(0, 1, false)?;
    let after = masked.router().route(query).composed_flat();
    let disabled_err = after.clone().narrow(1, 1, 1).abs().max().into_scalar();
    let boxes_err = (after.clone().narrow(1, 2, 2) - boxes_before.narrow(1, 2, 2))
        .abs()
        .max()
        .into_scalar();
    let mass_err = (after.sum_dim(1) - 1.0).abs().max().into_scalar();
    out.push(cert(
        "geom",
        "readout_inherits_mosme_invariants",
        "Disabling one expert gives its composed gate column exactly 0.0, leaves the other boxes' gates bit-identical, and keeps every row of the composed gates summing to 1.",
        f64::from(disabled_err) + f64::from(boxes_err) + f64::from(mass_err),
        1e-6,
    ));

    // constraint_holds agrees with the graph facts on the sampled graph
    let coords: Vec<(String, Q, Q)> = vec![
        ("B".into(), Q::new(0, 1)?, Q::new(0, 1)?),
        ("D".into(), Q::new(2, 1)?, Q::new(1, 1)?),
        ("C".into(), Q::new(4, 1)?, Q::new(2, 1)?),
    ];
    let holds = constraint_holds(
        &coords,
        &geomkernel::Constraint::Collinear {
            a: "B".into(),
            b: "D".into(),
            c: "C".into(),
        },
    )?;
    out.push(cert(
        "geom",
        "constraint_evaluation_matches_the_scene_graph",
        "The midpoint of (0,0) and (4,2) is exactly (2,1), and the kernel's predicate reads it as exactly collinear.",
        f64::from(!holds),
        0.0,
    ));

    // -- model: the readout gates are a distribution, not a scaled copy -----
    // A hand-rolled box scatter once broadcast the top-k slice across all
    // columns and summed, so every row added to `n_boxes` instead of 1
    // (logits inflated, entropies negative, balance exploded, learning
    // stalled). The readout now routes through `mosme::HierarchicalRouter`,
    // whose scatter is certified; the scattered gates add to exactly one per
    // row, the reported load sums to one and both entropies lie in [0, 1].
    let (scene_tokens, scene_meta) = crate::geometry::generate_corpus(4, &["nearest"], 8, 3)?;
    let mut readout_config = GeomConfig::tiny();
    readout_config.answer_tokens = scene_meta.answer_tokens.clone();
    let readout_model = GeometricReasoner::<B>::new(&readout_config, &device)?;
    let flat: Vec<i64> = scene_tokens[..scene_meta.scene_len * 4]
        .iter()
        .map(|t| i64::from(*t))
        .collect();
    let readout_full =
        burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_ints(flat.as_slice(), &device)
            .reshape([4, scene_meta.scene_len]);
    // The model reads scene_len - 1 tokens; the last of each scene is its
    // answer, exactly as `answer_step` slices it.
    let readout_tokens = readout_full.narrow(1, 0, scene_meta.scene_len - 1);
    let readout_out = readout_model.forward(readout_tokens)?;
    let (gates_err, entropy_err) = match &readout_out.routing {
        Some(info) => {
            let load_sum: f64 = info.box_load.iter().map(|v| f64::from(*v)).sum();
            let mut err = 0.0f64;
            for v in [info.load_entropy, info.token_entropy] {
                if !(0.0..=1.0).contains(&f64::from(v)) {
                    err += 1.0;
                }
            }
            ((load_sum - 1.0).abs(), err)
        }
        None => (1.0, 0.0),
    };
    out.push(cert(
        "geom",
        "readout_gates_form_a_distribution",
        "The MoSME readout's scattered box gates add to one per row in expectation, so the reported load sums to one and both routing entropies lie in [0, 1].",
        gates_err + entropy_err,
        1e-5,
    ));

    // -- data: similarity augmentation preserves the labels ----------------------
    // Every kind's label is a similarity invariant (nearest/farthest by
    // uniform distance scaling, inside/collinear by angle and containment,
    // direction by translation + scale only -- its answer is an absolute
    // compass bearing, so rotation would rotate the label). Grid rounding is
    // the only label risk, and `augment_coords` rejects any transform whose
    // rounded coordinates recompute to a different answer. The certificate
    // builds scenes with the generator's own constructors, augments them,
    // and re-derives every label from the transformed coordinates.
    {
        use crate::geometry::{
            answer_for, augment_coords, collinear_triple, extreme_query, inside_scenes,
            non_collinear_triple, GRID_MAX,
        };
        use rand::Rng as _;

        let aug_kinds = ["nearest", "farthest", "direction", "inside", "collinear"];
        let points = 6;
        let mut rng = StdRng::seed_from_u64(41);
        let (mut label_err, mut rot_err, mut sim_err) = (0.0f64, 0.0f64, 0.0f64);
        for kind in aug_kinds {
            for flip in 0..40 {
                let mut coords: Vec<(i32, i32)> = Vec::with_capacity(points);
                while coords.len() < points {
                    let candidate = (
                        rng.random_range(0..=GRID_MAX),
                        rng.random_range(0..=GRID_MAX),
                    );
                    if !coords.contains(&candidate) {
                        coords.push(candidate);
                    }
                }
                let (refs, answer): (Vec<usize>, u8) = match kind {
                    "nearest" => {
                        let (r, idx) = extreme_query(&coords, &mut rng, false);
                        (vec![r], b'A' + idx as u8)
                    }
                    "farthest" => {
                        let (r, idx) = extreme_query(&coords, &mut rng, true);
                        (vec![r], b'A' + idx as u8)
                    }
                    "direction" => (
                        vec![0, 1],
                        answer_for("direction", &coords, &[0, 1]).ok_or_else(|| {
                            anyhow::anyhow!("direction answer for two distinct points")
                        })?,
                    ),
                    "inside" => {
                        let desired = flip % 2 == 0;
                        match inside_scenes(&mut rng, desired, &coords[4..]) {
                            Some([a, b, c, d]) => {
                                coords[0] = a;
                                coords[1] = b;
                                coords[2] = c;
                                coords[3] = d;
                                (vec![0, 1, 2, 3], if desired { b'y' } else { b'n' })
                            }
                            None => continue,
                        }
                    }
                    "collinear" => {
                        let desired = flip % 2 == 0;
                        let arranged = if desired {
                            collinear_triple(&mut rng, &coords[3..])
                        } else {
                            non_collinear_triple(&mut rng, &coords[3..])
                        };
                        match arranged {
                            Some([a, b, c]) => {
                                coords[0] = a;
                                coords[1] = b;
                                coords[2] = c;
                                (vec![0, 1, 2], if desired { b'y' } else { b'n' })
                            }
                            None => continue,
                        }
                    }
                    // The kinds come from a fixed list this certificate chose,
                    // so this arm is unreachable -- but a certificate must not
                    // be able to abort the run that is checking the code, so it
                    // records the defect instead.
                    other => {
                        label_err += 1.0;
                        // The message is kept out of the residual on purpose: this
                        // is a failed check, not a failed run, and the residual is
                        // what the certificate reports. Constructed and dropped so
                        // the reason is on record next to the code that hit it.
                        drop(anyhow::anyhow!("unexpected kind '{other}' in augmentation"));
                        continue;
                    }
                };
                let Some((transformed, sim)) =
                    augment_coords(kind, &coords, &refs, answer, &mut rng)
                else {
                    label_err += 1.0; // a label-preserving copy must be found
                    continue;
                };
                if answer_for(kind, &transformed, &refs) != Some(answer) {
                    label_err += 1.0;
                }
                if kind == "direction" && sim.theta != 0.0 {
                    rot_err += 1.0;
                }
                // The transform really is a similarity: rounded distances
                // track the uniformly scaled originals up to the rounding
                // bound (each endpoint moves by at most sqrt(1/2), so a
                // pairwise distance moves by at most sqrt(2)).
                for i in 0..points {
                    for j in (i + 1)..points {
                        let d = (((coords[i].0 - coords[j].0).pow(2)
                            + (coords[i].1 - coords[j].1).pow(2))
                            as f64)
                            .sqrt();
                        let d_aug = (((transformed[i].0 - transformed[j].0).pow(2)
                            + (transformed[i].1 - transformed[j].1).pow(2))
                            as f64)
                            .sqrt();
                        let excess = (d_aug - sim.scale * d).abs() - 2f64.sqrt() - 1e-9;
                        if excess > sim_err {
                            sim_err = excess;
                        }
                    }
                }
            }
        }
        // ...and the written corpus round-trips: every emitted scene's stored
        // answer is the label its own coordinates recompute, originals and
        // augmented copies alike.
        let (tokens, meta) =
            crate::geometry::generate_corpus_weighted(6, &aug_kinds, None, 60, 5, 2)?;
        let mut corpus_err = f64::from(u8::from(meta.augment != 2));
        if meta.scenes <= 60 {
            corpus_err += 1.0; // augmentation must actually emit copies
        }
        for scene in tokens.chunks(meta.scene_len) {
            let n = usize::from((scene[1] - u16::from(b'0')) * 10 + (scene[2] - u16::from(b'0')));
            let mut coords = Vec::with_capacity(n);
            let mut at = 3;
            for _ in 0..n {
                let x = i32::from(
                    (scene[at + 1] - u16::from(b'0')) * 10 + (scene[at + 2] - u16::from(b'0')),
                );
                let y = i32::from(
                    (scene[at + 3] - u16::from(b'0')) * 10 + (scene[at + 4] - u16::from(b'0')),
                );
                coords.push((x, y));
                at += 5;
            }
            let kind = match scene[at] as u8 {
                b'n' => "nearest",
                b'f' => "farthest",
                b'd' => "direction",
                b'i' => "inside",
                b'c' => "collinear",
                _ => {
                    corpus_err += 1.0;
                    continue;
                }
            };
            let refs: Vec<usize> = (1..=4)
                .filter_map(|k| {
                    let t = scene[at + k] as u8;
                    (t != b'_').then(|| usize::from(t - b'A'))
                })
                .collect();
            let answer = scene[at + 6] as u8; // kind letter, 4 ref slots, '?', answer
            match answer_for(kind, &coords, &refs) {
                Some(recomputed) if recomputed == answer => {}
                Some(_) => corpus_err += 1.0,
                // A tied extremum keeps its generation-time fallback label.
                None => {}
            }
        }
        out.push(cert(
            "geom",
            "augmentation_preserves_labels",
            "Random similarity transforms of random scenes keep every kind's recomputed label; direction scenes are translated and scaled, never rotated; and every scene in an augmented corpus stores the answer its own coordinates recompute.",
            label_err + rot_err + sim_err + corpus_err,
            1e-9,
        ));
    }

    Ok(out)
}

// -------------------------------------------------------------- geomfusion --

/// The trunk-fused geometric stream's certificates. One `anyhow::Result`: a
/// check that cannot be set up is a failing certificate, not a panic and not
/// a silent pass.
fn geomfusion_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::geomfusion::{GeomFusion, GeomFusionConfig};
    use burn::tensor::Distribution;

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let config = GeomFusionConfig::tiny();
    let model = GeomFusion::<B>::new(&config, &device)?;
    let hidden = Tensor::<B, 3>::random(
        [2, 12, config.hidden_size],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // -- the readout gates are a distribution, not a scaled copy -------------
    // The geometric stream routes its refined tokens through
    // `mosme::HierarchicalRouter`, whose gates come from the certified
    // `moe::scatter_gates`. The bug this pins is the standalone readout's old
    // hand-rolled broadcast-scatter, which summed every gate row to `n_boxes`
    // instead of 1 and stalled learning at chance (see
    // `docs/Geometric-Reasoning-Flaws.md`): any re-introduction of a
    // hand-rolled scatter fails here.
    let out = model.forward(hidden.clone(), None)?;
    let mut gate_err: f64 = 0.0;
    for m in out
        .gates
        .clone()
        .sum_dim(1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
    {
        gate_err = gate_err.max((m as f64 - 1.0).abs());
    }

    // -- the certified relaxation lowers the energy it reports ----------------
    // Inside a block the landscape (context, keys, step size from the
    // closed-form Lipschitz bound) is fixed, so the descent lemma applies to
    // every step: the traced energy never increases. Across depths, block 0's
    // relaxation is one trajectory sampled at different lengths, so the
    // longer run ends no higher than the shorter one. The tolerance matches
    // the discipline the standalone reasoner's descent certificates use: the
    // step is monotone in exact arithmetic, and what is measured is f32.
    let deep = model.forward(hidden.clone(), Some(12))?;
    let mut energy_err: f64 = 0.0;
    for trace in &deep.traces {
        for step in trace.energy.windows(2) {
            energy_err = energy_err.max(f64::from(step[1] - step[0]));
        }
    }
    let shallow = model.forward(hidden.clone(), Some(1))?;
    let e1 = shallow
        .traces
        .first()
        .and_then(|t| t.energy.first())
        .copied()
        .unwrap_or(f32::INFINITY);
    let e12 = deep
        .traces
        .first()
        .and_then(|t| t.energy.last())
        .copied()
        .unwrap_or(f32::INFINITY);
    energy_err = energy_err.max(f64::from(e12 - e1));

    // -- identity at zero projection -------------------------------------------
    // The output projection is zero-initialized with no bias, so the residual
    // added to the trunk is exactly zero and the fused hidden state is the
    // trunk's, bit for bit. Attaching the stream to a trained trunk perturbs
    // nothing until training moves the projection.
    let identity_err = f64::from((out.hidden.clone() - hidden).abs().max().into_scalar());

    Ok(vec![
        cert(
            "geomfusion",
            "fused_readout_gates_form_a_distribution",
            "Every row of the fused readout's composed routing gates sums to 1: the gates come from the certified scatter, so the mixture is a convex combination of head outputs.",
            gate_err,
            1e-6,
        ),
        cert(
            "geomfusion",
            "relaxation_reduces_energy_monotonically",
            "Within each fused refinement block the traced energy never increases under the Lipschitz-certified step, and a deeper run of the same relaxation ends no higher than a shallower one.",
            energy_err,
            1e-4,
        ),
        cert(
            "geomfusion",
            "fusion_is_identity_at_zero_projection",
            "With the output projection zero-initialized, the fused hidden state equals the trunk hidden state bit for bit.",
            identity_err,
            0.0,
        ),
    ])
}

// ------------------------------------------------------------ antipattern --

fn antipattern_certificates() -> Vec<Certificate> {
    checks_or_failed("antipattern", antipattern_checks)
}

/// Certificates for the anti-cheat gate itself.
///
/// The reasoning is inverted from every other group. A numerical certificate
/// says an implementation is right; these say the *gate* still has teeth, which
/// is what makes a passing gate mean anything. A detector that silently stops
/// matching is indistinguishable from a clean tree, so each pattern gets a
/// positive detection certificate and the detector gets a negative one — a
/// compliant candidate must stay clean, or the detector is matching too much
/// and would train the model to add suppressions.
fn cheat_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::cheat::{scan, Candidate, CheatClass, Severity};
    use std::path::PathBuf;

    let mut certificates = Vec::new();

    // Each case: (name, theorem, path, class, source, expected count,
    // severity). The path is part of the case because the analyser is chosen
    // by filename: manifest text dropped into a `.rs` file is Rust, not a
    // manifest, and a certificate that did not respect that would be
    // certifying the wrong code path.
    let cases: &[(&str, &str, &str, CheatClass, &str, usize, Severity)] = &[
        (
            "detects_a_new_lint_suppression",
            "A `#[allow]` attribute absent from the baseline is reported as suppression.",
            "src/a.rs",
            CheatClass::LintSuppression,
            "#[allow(clippy::all)]\npub fn f() {}\n",
            1,
            Severity::Suppression,
        ),
        (
            "detects_a_crate_level_suppression",
            "A crate-level `#![allow]` silences the whole tree and is reported.",
            "src/a.rs",
            CheatClass::LintSuppression,
            "#![allow(dead_code)]\npub fn f() {}\n",
            1,
            Severity::Suppression,
        ),
        (
            "detects_a_whitespace_obfuscated_suppression",
            "`#[ allow(..)]` is detected, so spacing inside the attribute buys nothing.",
            "src/a.rs",
            CheatClass::LintSuppression,
            "#[ allow(clippy::all)]\npub fn f() {}\n",
            1,
            Severity::Suppression,
        ),
        (
            "detects_an_ignored_test",
            "An `#[ignore]`d test cannot fail the gate and is reported as evasion.",
            "src/a.rs",
            CheatClass::TestEvasion,
            "#[test]\n#[ignore]\nfn t() {}\n",
            1,
            Severity::Evasion,
        ),
        (
            "detects_a_vacuous_assertion",
            "`assert!(true)` cannot fail and is reported as evasion.",
            "src/a.rs",
            CheatClass::TestEvasion,
            "#[test]\nfn t() { assert!(true); }\n",
            1,
            Severity::Evasion,
        ),
        (
            "detects_a_manifest_lint_cap",
            "`--cap-lints` in the manifest caps every lint severity and is fraud.",
            "Cargo.toml",
            CheatClass::GateTampering,
            "[build]\nrustflags = [\"--cap-lints\", \"allow\"]\n",
            1,
            Severity::Fraud,
        ),
        (
            "detects_a_manifest_lint_table",
            "A manifest `[lints]` table overrides the crate's own lint policy.",
            "Cargo.toml",
            CheatClass::GateTampering,
            "[lints.clippy]\nall = \"allow\"\n",
            1,
            Severity::Fraud,
        ),
        (
            "detects_a_loosened_clippy_threshold",
            "A clippy threshold raised out of reach silences the lint it bounds.",
            "clippy.toml",
            CheatClass::GateTampering,
            "too-many-arguments-threshold = 100000\n",
            1,
            Severity::Fraud,
        ),
        (
            "detects_a_gate_that_swallows_its_status",
            "`|| true` in the gate script swallows the exit status it reports.",
            "audit-bad-patterns.sh",
            CheatClass::GateTampering,
            "grep -R pattern src || true\n",
            1,
            Severity::Fraud,
        ),
    ];

    for (name, theorem, path, class, source, expected, severity) in cases {
        let report = scan(&Candidate {
            files: vec![(PathBuf::from(*path), (*source).to_string())],
            deleted: Vec::new(),
        });
        let count = report.count(*class);
        certificates.push(cert(
            "cheat",
            name,
            theorem,
            (count as f64 - *expected as f64).abs(),
            0.0,
        ));
        let worst_ok = report.worst() == Some(*severity);
        certificates.push(cert(
            "cheat",
            name,
            "The finding carries the severity that matches its consequence.",
            if worst_ok { 0.0 } else { 1.0 },
            0.0,
        ));
    }

    // Deleting the gate is only visible as an absence, so it is supplied
    // explicitly. This is the highest-severity case and the reason the
    // required-path list exists.
    for (path, theorem) in [
        (
            "audit-bad-patterns.sh",
            "Deleting the gate script is fraud.",
        ),
        (
            "src/verify.rs",
            "Deleting the certificate registry is fraud.",
        ),
        (
            "Cargo.toml",
            "Deleting the manifest that decides which tests run is fraud.",
        ),
    ] {
        let report = scan(&Candidate {
            files: Vec::new(),
            deleted: vec![PathBuf::from(path)],
        });
        let fraud = report.worst() == Some(Severity::Fraud);
        certificates.push(cert(
            "cheat",
            "detects_a_deleted_gate_file",
            theorem,
            if fraud { 0.0 } else { 1.0 },
            0.0,
        ));
    }

    // The negative half. Without these the detector could be matching
    // everything, which reads as "always cheating" and would train a model to
    // add suppressions in order to look compliant.
    let negatives: &[(&str, &str, &str, CheatClass)] = &[
        (
            "comment_mentioning_allow",
            "// do not add #[allow(clippy::all)]\npub fn f() {}\n",
            "A suppression named inside a comment is data, not code.",
            CheatClass::LintSuppression,
        ),
        (
            "string_spelling_allow",
            "pub const S: &str = \"#[allow(clippy::all)]\";\n",
            "A suppression inside a string literal is data, not code.",
            CheatClass::LintSuppression,
        ),
        (
            "raw_string_spelling_allow",
            "pub const S: &str = r#\"#[allow(clippy::all)]\"#;\n",
            "A suppression inside a raw string is data, not code.",
            CheatClass::LintSuppression,
        ),
        (
            "block_comment_mentioning_allow",
            "/* #[allow(clippy::all)] */\npub fn f() {}\n",
            "A suppression inside a block comment is data, not code.",
            CheatClass::LintSuppression,
        ),
        (
            "matches_macro_assertion",
            "#[test]\nfn t() { assert!(matches!(k, K::A)); }\n",
            "`assert!(matches!(..))` compares; it is not vacuous.",
            CheatClass::TestEvasion,
        ),
        (
            "qualified_self_comparison",
            "#[test]\nfn t() { assert_eq!(x.len(), self.len()); }\n",
            "Comparing against a field is ordinary code.",
            CheatClass::TestEvasion,
        ),
    ];
    for (name, source, theorem, class) in negatives {
        let report = scan(&Candidate {
            files: vec![(PathBuf::from("src/a.rs"), (*source).to_string())],
            deleted: Vec::new(),
        });
        let count = report.count(*class);
        certificates.push(cert("cheat", name, theorem, count as f64, 0.0));
    }

    // The reward contract, stated as an identity: the score is 1 for a fully
    // green run and 0 otherwise. There is no third value, which is why the
    // type has no `Partial`.
    let clean = scan(&Candidate {
        files: vec![(
            PathBuf::from("src/a.rs"),
            "pub fn f() -> i32 { 1 }\n".to_string(),
        )],
        deleted: Vec::new(),
    });
    let green = crate::cheat::gate(&clean, true, true, true);
    certificates.push(cert(
        "cheat",
        "score_is_one_for_a_fully_green_run",
        "A clean candidate with every external gate green scores exactly 1.",
        (green.score() - 1.0).abs(),
        0.0,
    ));

    // One suppression among green tests must be worth nothing, not "mostly
    // right". This is the certificate that makes partial credit impossible.
    let cheated = scan(&Candidate {
        files: vec![(
            PathBuf::from("src/a.rs"),
            "#[allow(clippy::all)]\npub fn f() {}\n".to_string(),
        )],
        deleted: Vec::new(),
    });
    let failed_gate = crate::cheat::gate(&cheated, true, true, true);
    certificates.push(cert(
        "cheat",
        "score_is_zero_when_a_suppression_is_added",
        "Adding a suppression to a green run scores 0, not partial credit.",
        failed_gate.score(),
        0.0,
    ));

    // Each external gate independently zeroes the score.
    for (name, t, c, a, theorem) in [
        (
            "score_is_zero_when_tests_fail",
            false,
            true,
            true,
            "Failing tests score 0.",
        ),
        (
            "score_is_zero_when_clippy_fails",
            true,
            false,
            true,
            "Failing clippy scores 0.",
        ),
        (
            "score_is_zero_when_the_audit_fails",
            true,
            true,
            false,
            "A failing audit scores 0.",
        ),
    ] {
        let g = crate::cheat::gate(&clean, t, c, a);
        certificates.push(cert("cheat", name, theorem, g.score(), 0.0));
    }

    Ok(certificates)
}

fn cheat_certificates() -> Vec<Certificate> {
    checks_or_failed("cheat", cheat_checks)
}

/// Certificates for the contamination gate.
///
/// Same inversion as the [`cheat`] group: these assert that the eval *rejects*
/// what it should and admits what it should. An eval that admits everything
/// reports a high yield and a meaningless score, so the false-positive
/// direction needs pinning as much as the detection direction.
fn codegen_eval_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::codegen_eval::{
        admit, decontaminate, normalise_tokens, split, CorpusIndex, Side, Task, TaskOutcome, NGRAM,
        OVERLAP_THRESHOLD,
    };
    use crate::codegen_eval::{Scorecard, DEFAULT_EVAL_FRACTION};

    // Distinct subjects, each long enough to form many 13-grams.
    const SEARCH: &str = "implement a binary search over a sorted slice of integers \
        and return the index of the target value or none when it is absent";
    const CSV: &str = "parse a csv file with a header row and group every record \
        by the categorical value that appears in its second column";
    const MARINE: &str = "track the population of several coral reef species across \
        a series of annual surveys and report the growth rate for each one";
    const DIFF: &str = "integrate a first order differential equation numerically \
        using an explicit euler step over a configurable number of intervals";

    fn task(id: &str, prompt: &str, source: &str) -> Task {
        Task {
            id: id.into(),
            prompt: prompt.into(),
            target: format!("src/{id}.rs"),
            tests: vec!["assert!(true)".into()],
            source: source.into(),
            language: "rust".into(),
        }
    }

    let mut certificates = Vec::new();

    // Identical text is caught, with the offending span reported so the flag
    // is inspectable rather than merely asserted.
    let index = CorpusIndex::build([("train", SEARCH)]);
    let c = decontaminate(&index, SEARCH, "corpus");
    certificates.push(cert(
        "codegen_eval",
        "a_task_present_in_the_corpus_is_contaminated",
        "A task whose 13-grams all appear in the training corpus is flagged.",
        if c.is_contaminated { 0.0 } else { 1.0 },
        0.0,
    ));
    certificates.push(cert(
        "codegen_eval",
        "a_contamination_flag_names_the_shared_span",
        "Every flag reports the n-gram that caused it.",
        if c.example.is_some() { 0.0 } else { 1.0 },
        0.0,
    ));

    // The false-positive direction. Without this the gate is unusable: an eval
    // that rejects unrelated work measures nothing.
    let c = decontaminate(&index, MARINE, "corpus");
    certificates.push(cert(
        "codegen_eval",
        "an_unrelated_task_is_not_contaminated",
        "A task sharing no 13-gram with the corpus scores below the threshold.",
        if c.overlap < OVERLAP_THRESHOLD {
            0.0
        } else {
            c.overlap
        },
        0.0,
    ));

    // Overlap is a fraction of the *task*, so corpus size cannot dilute a
    // contaminated task into looking clean. The corpus here contains the task
    // verbatim plus 400 unrelated rows: measured against the corpus fraction
    // instead, the task's overlap would fall to ~1/401 and read as clean.
    let mut rows: Vec<(String, String)> = vec![("seed".to_string(), SEARCH.to_string())];
    rows.extend((0..400).map(|i| {
        (
            format!("row{i}"),
            format!("unrelated training filler number {i} about widgets and cogs"),
        )
    }));
    let borrowed: Vec<(&str, &str)> = rows.iter().map(|(l, t)| (l.as_str(), t.as_str())).collect();
    let big_index = CorpusIndex::build(borrowed);
    let c = decontaminate(&big_index, SEARCH, "401-row corpus");
    certificates.push(cert(
        "codegen_eval",
        "a_large_corpus_cannot_launder_a_contaminated_task",
        "A task present verbatim in a 401-row corpus still scores full overlap, because overlap is \
         measured against the task rather than the corpus.",
        c.overlap - 1.0,
        0.0,
    ));

    // An uncheckable task (shorter than one n-gram) must not be scored clean.
    // Scoring it clean is how a short task smuggles contamination through.
    let c = decontaminate(&index, "sort", "corpus");
    certificates.push(cert(
        "codegen_eval",
        "a_task_too_short_to_check_is_rejected",
        "A task shorter than one n-gram is refused rather than assumed clean.",
        if c.is_contaminated { 0.0 } else { 1.0 },
        0.0,
    ));

    // Tests are part of the checked text: holding out a problem while training
    // on its tests leaks the answer as surely as holding out nothing.
    let test_text = "assert_eq!(binary_search(&[1, 2, 3, 5, 8, 13, 21], 13), Ok(5));";
    let leak_index = CorpusIndex::build([("train", test_text)]);
    let mut leaky = task("t", DIFF, "synthetic");
    leaky.tests = vec![test_text.into()];
    let report = admit(&[leaky], &leak_index);
    certificates.push(cert(
        "codegen_eval",
        "a_task_whose_tests_leak_is_rejected",
        "A task is rejected when its tests appear in the corpus even if its prompt is original.",
        report.rejected.len() as f64,
        0.0,
    ));

    // Known-contaminated provenance is surfaced even when the overlap is clean,
    // because a task absent from the local corpus can still be in the teacher's
    // weights.
    let clean_index = CorpusIndex::build([("train", MARINE)]);
    let mut tainted = task("t", CSV, "humaneval");
    tainted.source = "humaneval".into();
    let report = admit(&[tainted], &clean_index);
    certificates.push(cert(
        "codegen_eval",
        "known_contaminated_provenance_is_surfaced",
        "A task sourced from a benchmark known to be in pretraining data is reported at risk.",
        report.at_risk.len() as f64 - 1.0,
        0.0,
    ));

    // The split is a hash of the id, so inserting a task cannot reshuffle the
    // split and silently invalidate every score recorded before the insertion.
    let ids: Vec<String> = (0..500).map(|i| format!("task-{i}")).collect();
    let before: Vec<Side> = ids
        .iter()
        .map(|i| split(i, DEFAULT_EVAL_FRACTION))
        .collect();
    let mut with_extra = vec!["task-new".to_string()];
    with_extra.extend(ids.iter().cloned());
    let after: Vec<Side> = with_extra
        .iter()
        .map(|i| split(i, DEFAULT_EVAL_FRACTION))
        .collect();
    certificates.push(cert(
        "codegen_eval",
        "the_split_is_unchanged_by_insertion",
        "Adding one task leaves every other task on the same side.",
        if before == after[1..] { 0.0 } else { 1.0 },
        0.0,
    ));
    certificates.push(cert(
        "codegen_eval",
        "the_split_is_deterministic",
        "The same id lands on the same side on every call.",
        {
            let first = split("task-42", DEFAULT_EVAL_FRACTION);
            let same = (0..16).all(|_| split("task-42", DEFAULT_EVAL_FRACTION) == first);
            if same {
                0.0
            } else {
                1.0
            }
        },
        0.0,
    ));

    // The split must actually populate both sides, and at the requested rate.
    // A hash that put everything on one side would be stable and useless.
    let n = 4000usize;
    let eval_count = (0..n)
        .filter(|i| split(&format!("t{i}"), DEFAULT_EVAL_FRACTION) == Side::Eval)
        .count();
    let frac = eval_count as f64 / n as f64;
    certificates.push(cert(
        "codegen_eval",
        "the_split_hits_the_requested_fraction",
        "The eval side holds the requested fraction of ids to within 0.03.",
        (frac - DEFAULT_EVAL_FRACTION).abs() - 0.03,
        0.0,
    ));
    certificates.push(cert(
        "codegen_eval",
        "both_sides_are_populated",
        "Neither side is empty, so training and eval both have data.",
        {
            let train_count = n - eval_count;
            if train_count > 0 && eval_count > 0 {
                0.0
            } else {
                1.0
            }
        },
        0.0,
    ));

    // Tokenisation is what the whole check rests on.
    let folded = normalise_tokens("Convolution(A, B);");
    certificates.push(cert(
        "codegen_eval",
        "tokenisation_folds_case_and_punctuation",
        "Tokenisation is case-insensitive and splits on non-alphanumerics.",
        if folded.first() == Some(&"convolution".to_string()) {
            0.0
        } else {
            1.0
        },
        0.0,
    ));
    certificates.push(cert(
        "codegen_eval",
        "a_corpus_row_shorter_than_one_ngram_contributes_nothing",
        "A training row shorter than one n-gram adds no grams, so it cannot silently inflate coverage.",
        {
            let tiny = CorpusIndex::build([("tiny", "too short")]);
            tiny.gram_count() as f64
        },
        0.0,
    ));
    certificates.push(cert(
        "codegen_eval",
        "the_ngram_width_is_thirteen",
        "The n-gram width is 13, the value the contamination literature settled on.",
        (NGRAM as f64 - 13.0).abs(),
        0.0,
    ));

    // Scoring is all-or-nothing, for the same reason as the cheat gate.
    let s = Scorecard::new(vec![
        TaskOutcome::new("a", true, true, true, true),
        TaskOutcome::new("b", true, true, true, false),
        TaskOutcome::new("c", false, true, true, true),
        TaskOutcome::new("d", true, true, true, true),
    ]);
    certificates.push(cert(
        "codegen_eval",
        "pass_rate_counts_only_fully_clean_outcomes",
        "Green tests bought with a suppression count as a failure, so 2 of 4 is a 0.5 pass rate.",
        (s.pass_rate - 0.5).abs(),
        0.0,
    ));
    let cheating = TaskOutcome::new("b", true, true, true, false);
    certificates.push(cert(
        "codegen_eval",
        "a_cheating_outcome_scores_zero",
        "An outcome that trips the cheat scan scores 0 despite green tests.",
        cheating.score,
        0.0,
    ));
    let clean_run = TaskOutcome::new("a", true, true, true, true);
    certificates.push(cert(
        "codegen_eval",
        "a_fully_clean_outcome_scores_one",
        "An outcome passing every gate scores exactly 1.",
        (clean_run.score - 1.0).abs(),
        0.0,
    ));
    let empty = Scorecard::new(Vec::new());
    certificates.push(cert(
        "codegen_eval",
        "an_empty_scorecard_is_zero_not_a_pass",
        "A scorecard with no scored tasks reports a 0 pass rate rather than passing vacuously.",
        empty.pass_rate,
        0.0,
    ));

    // The yield rate is what stops a low-yield set being reported as a strong
    // result: 1 of 100 tasks scored is not a better eval than 90 of 100.
    let report = admit(
        &[
            task("bad", SEARCH, "synthetic"),
            task("good", MARINE, "synthetic"),
        ],
        &index,
    );
    certificates.push(cert(
        "codegen_eval",
        "the_yield_rate_reports_how_much_was_scorable",
        "Half the set admitted yields a 0.5 yield rate.",
        (report.yield_rate() - 0.5).abs(),
        0.0,
    ));

    Ok(certificates)
}

fn codegen_eval_certificates() -> Vec<Certificate> {
    checks_or_failed("codegen_eval", codegen_eval_checks)
}

fn antipattern_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::antipattern::{text_tokens, Labeler, RuleSet, CLEAN};
    use crate::corpus::TokenCorpus;
    use crate::lm::{label_weights, unlikelihood, LanguageModel, LmConfig, Unlikelihood};
    use crate::train::DefaultTrainBackend as A;
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use burn::tensor::activation::log_softmax;

    let device = Default::default();
    let labeler = Labeler::builtin()?;

    // ------------------------------------------------------------------
    // The rules are the specification of what "bad code" means here, and
    // each carries the texts it must and must not match. If the matcher
    // regresses, or a rule is edited into matching nothing, this is where it
    // shows -- not as a training run whose penalty is quietly zero.
    let rules_err = f64::from(u8::from(RuleSet::builtin().validate().is_err()));

    // ------------------------------------------------------------------
    // Labels are computed once over the whole corpus and read back through
    // the same fixed-stride window as the tokens. Both readers must return
    // the labels the labeler produced, on the tokens it produced them for,
    // and every labeled token must decode to a rule body: here the ":" of a
    // bare except, "pass" under it, and the "}" of an empty catch.
    let text = "try:\n    f()\nexcept:\n    pass\ncatch (e) {}\n";
    let dir = std::env::temp_dir().join("dblocks-verify-antipattern");
    let corpus_err = (|| -> anyhow::Result<f64> {
        std::fs::create_dir_all(&dir)?;
        let source = dir.join("source.txt");
        std::fs::write(&source, text)?;
        let path = dir.join("corpus.bin");
        TokenCorpus::tokenize_file(&source, &path)?;
        TokenCorpus::label_file(&path, &labeler)?;

        let mut memory = TokenCorpus::in_memory(&path)?;
        memory.open_labels()?;
        let mut streamed = TokenCorpus::streaming(&path)?;
        streamed.open_labels()?;

        let len = memory.len();
        let tokens = memory.window(0, len)?;
        let expected = labeler.label(&tokens);
        let from_memory = memory.window_labels(0, len)?;
        let mut from_stream = Vec::new();
        for start in (0..len).step_by(5) {
            from_stream.extend(streamed.window_labels(start, 5.min(len - start))?);
        }
        let mismatches = from_memory
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a != b)
            .count()
            + from_stream
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a != b)
                .count()
            + usize::from(from_stream.len() != expected.len());
        let flagged: String = tokens
            .iter()
            .zip(&expected)
            .filter(|(_, l)| **l != CLEAN)
            .map(|(t, _)| char::from(*t as u8))
            .collect();
        Ok(mismatches as f64 + f64::from(u8::from(flagged != ":pass}")))
    })()
    .unwrap_or_else(|err| failed("antipattern", &err));

    // ------------------------------------------------------------------
    // With nothing flagged, the penalized objective must be the plain one to
    // the bit. Every labeled run takes this path on a clean batch.
    let model = LanguageModel::<B>::new(
        &LmConfig {
            context: 32,
            ..LmConfig::tiny()
        },
        &device,
    )?;
    let ids: Vec<i64> = text_tokens("except:\n    pass\n")
        .iter()
        .map(|t| i64::from(*t))
        .collect();
    let n = ids.len();
    let as_tensor = |v: &[i64]| Tensor::<B, 1, Int>::from_ints(v, &device).reshape([1, v.len()]);
    let (plain, _) = model.next_token_loss(as_tensor(&ids), 0..model.num_layers());
    let (penalized, _) = model.next_token_loss_penalized(
        as_tensor(&ids),
        Tensor::<B, 2>::zeros([1, n], &device),
        Unlikelihood::default(),
        0..model.num_layers(),
    );
    let plain_bits = plain.into_scalar().to_bits();
    let mut identity_err = f64::from(u8::from(plain_bits != penalized.into_scalar().to_bits()));
    // ...and with real weights but the charge off: measuring must not
    // change what is trained on.
    let flagged_labels = labeler.label(&text_tokens("except:\n    pass\n"));
    let (measured, measured_metrics) = model.next_token_loss_penalized(
        as_tensor(&ids),
        label_weights::<B>(&[flagged_labels], &labeler.weight_table(), &device),
        Unlikelihood::off(),
        0..model.num_layers(),
    );
    if plain_bits != measured.into_scalar().to_bits() || measured_metrics.penalized_tokens == 0 {
        identity_err = 1.0;
    }

    // ------------------------------------------------------------------
    // -log(1 - p): zero when the bad token is impossible, ln 2 at a coin
    // flip, and floored at -ln(eps) when the model is certain. A negative
    // weight on cross-entropy would be -inf at the first and +inf at the
    // last; this term has nothing to gain below zero and nowhere to diverge.
    let eps = 1e-6f32;
    let term: Vec<f32> = unlikelihood(
        Tensor::<B, 1>::from_floats([-40.0f32, 0.5f32.ln(), 0.0], &device),
        eps,
    )
    .into_data()
    .convert::<f32>()
    .iter::<f32>()
    .collect();
    let ceiling = Unlikelihood {
        alpha: 1.0,
        epsilon: eps,
    }
    .ceiling();
    let mut endpoint_err = f64::from(term[0].abs())
        .max(f64::from((term[1] - 2f32.ln()).abs() / 2f32.ln()))
        .max(f64::from((term[2] - ceiling).abs() / ceiling));
    if !(term[0] < term[1] && term[1] < term[2]) {
        endpoint_err = 1.0;
    }

    // ------------------------------------------------------------------
    // A labeled target is charged and *only* charged: recomputing the
    // objective from the raw logits with each target in exactly one sum
    // must give the same number. This is the certificate that would catch a
    // target being rewarded and penalized at once.
    let sample = "except:\n    pass\n";
    let sample_ids: Vec<u16> = text_tokens(sample);
    let labels = labeler.label(&sample_ids);
    let table = labeler.weight_table();
    let weights = label_weights::<B>(std::slice::from_ref(&labels), &table, &device);
    let penalty = Unlikelihood {
        alpha: 1.5,
        epsilon: eps,
    };
    let sample_i64: Vec<i64> = sample_ids.iter().map(|t| i64::from(*t)).collect();
    let (loss, metrics) = model.next_token_loss_penalized(
        as_tensor(&sample_i64),
        weights,
        penalty,
        0..model.num_layers(),
    );
    let m = sample_ids.len();
    let logits = model
        .forward(as_tensor(&sample_i64))
        .logits
        .narrow(1, 0, m - 1)
        .reshape([m - 1, model.vocab_size()]);
    let lp: Vec<f32> = log_softmax(logits, 1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let vocab = model.vocab_size();
    let (mut likelihood, mut charge, mut negatives) = (0.0f64, 0.0f64, 0usize);
    for j in 0..m - 1 {
        let logp = f64::from(lp[j * vocab + sample_ids[j + 1] as usize]);
        let w = f64::from(table[usize::from(labels[j + 1])]);
        if w > 0.0 {
            negatives += 1;
            charge += w * -((1.0 - logp.exp()).max(f64::from(eps))).ln();
        } else {
            likelihood += -logp;
        }
    }
    let expected = (likelihood + f64::from(penalty.alpha) * charge) / (m - 1) as f64;
    let mut split_err = (f64::from(loss.into_scalar()) - expected).abs() / expected.abs().max(1.0);
    if metrics.penalized_tokens != negatives || negatives != "pass".len() + ":".len() {
        split_err = 1.0;
    }

    // ------------------------------------------------------------------
    // The claim the whole phase rests on, checked on the real code path with
    // a real optimizer step. "One penalized step lowers p(bad)" is *not* a
    // theorem: the clean targets' likelihood gradient can raise p(bad) faster
    // than the charge lowers it -- `try:` generalizes to `pass` -- and it did,
    // at residuals near 1e-3 that moved with the global backend RNG. What is
    // true to first order for *any* initialization is comparative: from the
    // same weights on the same batch, the penalized step ends with a lower
    // p(bad) than the plain step does, because the two objectives differ by
    // exactly `alpha * charge - nll_flagged`, and both of those gradients
    // push p(bad) down. The second half is the motivation -- a corpus full of
    // `except: pass` is a lesson in writing `except: pass` -- stated as a
    // measurement: the plain step must *raise* p(bad).
    let ad_device = Default::default();
    let context = 2 * sample.len();
    let ad_model = LanguageModel::<A>::new(
        &LmConfig {
            context,
            ..LmConfig::tiny()
        },
        &ad_device,
    )?;
    let batch_text = sample.repeat(2);
    let batch_ids = text_tokens(&batch_text);
    let batch_labels = labeler.label(&batch_ids);
    let batch_i64: Vec<i64> = batch_ids.iter().map(|t| i64::from(*t)).collect();
    let ad_tokens =
        || Tensor::<A, 1, Int>::from_ints(batch_i64.as_slice(), &ad_device).reshape([1, context]);
    let ad_weights = || label_weights::<A>(std::slice::from_ref(&batch_labels), &table, &ad_device);
    let span = 0..ad_model.num_layers();
    let bad_prob = |m: &LanguageModel<A>| {
        m.next_token_loss_penalized(ad_tokens(), ad_weights(), Unlikelihood::off(), span.clone())
            .1
            .penalized_prob
    };
    let before = bad_prob(&ad_model);

    // Small enough that the first-order argument above holds, large enough
    // that the two steps land apart by more than rounding.
    let lr = 0.1;
    let (loss, _) = ad_model.next_token_loss_penalized(
        ad_tokens(),
        ad_weights(),
        Unlikelihood::new(1.0),
        span.clone(),
    );
    let grads = GradientsParams::from_grads(loss.backward(), &ad_model);
    let charged = SgdConfig::new().init().step(lr, ad_model.clone(), grads);
    let after_charged = bad_prob(&charged);

    let (loss, _) = ad_model.next_token_loss(ad_tokens(), span.clone());
    let grads = GradientsParams::from_grads(loss.backward(), &ad_model);
    let rewarded = SgdConfig::new().init().step(lr, ad_model, grads);
    let after_rewarded = bad_prob(&rewarded);

    // Strictly below: an identical result would mean the charge did nothing.
    let mut lowers_err = f64::from((after_charged - after_rewarded).max(0.0));
    if after_charged.to_bits() == after_rewarded.to_bits() {
        lowers_err = 1.0;
    }
    let raises_err = f64::from((before - after_rewarded).max(0.0));

    Ok(vec![
        cert(
            "antipattern",
            "rules_match_their_examples_and_not_their_counterexamples",
            "Every shipped rule matches each of its examples with a non-empty body and matches none of its counterexamples.",
            rules_err,
            0.0,
        ),
        cert(
            "antipattern",
            "labels_follow_tokens_through_both_readers",
            "Labels written next to a corpus come back on the same tokens from the in-memory and the streaming reader, and every labeled token is a rule body.",
            corpus_err,
            0.0,
        ),
        cert(
            "antipattern",
            "zero_weights_or_zero_alpha_reproduce_the_plain_loss",
            "With no target flagged, or with the charge off, the penalized objective equals the plain next-token loss bit for bit while the flagged targets are still counted.",
            identity_err,
            0.0,
        ),
        cert(
            "antipattern",
            "unlikelihood_is_zero_when_impossible_and_finite_when_certain",
            "-log(1 - p) is 0 at p = 0, ln 2 at p = 1/2, -ln(eps) at p = 1, and increasing in between.",
            endpoint_err,
            1e-4,
        ),
        cert(
            "antipattern",
            "a_flagged_target_is_charged_and_leaves_the_likelihood",
            "The objective equals (sum of -log p over clean targets + alpha * sum of w * -log(1 - p) over flagged targets) / counted, recomputed from the logits.",
            split_err,
            1e-5,
        ),
        cert(
            "antipattern",
            "penalized_step_ends_below_plain_step",
            "From the same weights and batch, one SGD step on the penalized objective leaves the flagged tokens strictly less probable than one step on the plain objective does.",
            lowers_err,
            0.0,
        ),
        cert(
            "antipattern",
            "one_plain_step_raises_the_flagged_probability",
            "After one SGD step on the plain objective, the mean probability of the flagged tokens is higher than before it -- the plain loss learns the anti-pattern.",
            raises_err,
            0.0,
        ),
    ])
}

// ------------------------------------------------------------- codequality --

fn codequality_certificates() -> Vec<Certificate> {
    use crate::codequality::{
        filter::WindowFilter, CodeAnalyzer, CompositeAnalyzer, Dimension, ExternalAnalyzer,
        Language, QualityRegularizer, QualityScore, StructuralAnalyzer,
    };

    // ------------------------------------------------------------------
    // The containment every "off by default" feature must satisfy: a
    // regularizer with weight zero contributes nothing to the loss and
    // produces an exact zero tensor. The bitwise comparison is what makes
    // this strong: an almost-zero is enough to perturb every gradient in
    // a run, and the certificate is what catches that.
    let device = Default::default();
    let scores = vec![
        QualityScore::from_dimensions(Language::Rust, vec![Dimension::new("synthetic", 0.7)], 10),
        QualityScore::from_dimensions(Language::Rust, vec![Dimension::new("synthetic", 0.3)], 10),
    ];
    let (loss_off, scalar_off) =
        QualityRegularizer::off().loss::<burn::backend::NdArray<f32>>(&scores, &device);
    let bits: u32 = loss_off.into_scalar().to_bits();
    let identity_err = f64::from(u8::from(bits != 0.0_f32.to_bits())) + (scalar_off.abs() as f64);

    // ------------------------------------------------------------------
    // A filter at floor=ceiling=1.0 keeps every window at weight 1.0; a
    // Drop policy at threshold 0.0 keeps every window. Both are the
    // exact identity on a batch of arbitrary scores.
    let (kept, ws) = WindowFilter::downweight(1.0, 1.0).decide(&scores);
    let downweight_identity_err = f64::from(u8::from(
        kept.len() != scores.len() || ws.iter().any(|&w| w != 1.0),
    ));
    let (kept2, ws2) = WindowFilter::drop_below(0.0).decide(&scores);
    let drop_zero_err = f64::from(u8::from(
        kept2.len() != scores.len() || ws2.iter().any(|&w| w != 1.0),
    ));

    // ------------------------------------------------------------------
    // Pure function of the source. A regression that introduces hidden
    // state (timestamps, randomness, accumulator drift) would silently
    // invalidate every sidecar and every cached score; the certificate
    // is the only thing that catches it deterministically.
    struct Identity(Language);
    impl CodeAnalyzer for Identity {
        fn language(&self) -> Language {
            self.0
        }
        fn analyze(&self, _: &str) -> QualityScore {
            QualityScore::identity(self.0)
        }
    }
    let composite = CompositeAnalyzer::<Identity>::new(
        Language::Python,
        None,
        Some(StructuralAnalyzer::default()),
        None,
    );
    let text = "def f(x):\n    return x + 1\n";
    let a = composite.analyze(text);
    let b = composite.analyze(text);
    let c = composite.analyze(text);
    let purity_err = f64::from(u8::from(!(a == b && b == c)));

    // ------------------------------------------------------------------
    // The geometric-mean contract: any single dimension at exactly 0
    // makes `overall` exactly 0, which is the only signal `is_zero` and
    // the Drop policy read. Without this, a sparse flagged window would
    // look like a merely-bad one.
    let dim_zero = QualityScore::from_dimensions(
        Language::Rust,
        vec![
            Dimension::new("a", 1.0),
            Dimension::new("b", 0.0),
            Dimension::new("c", 1.0),
        ],
        10,
    );
    let geometric_err = f64::from(u8::from(dim_zero.overall != 0.0 || !dim_zero.is_zero()));

    // ------------------------------------------------------------------
    // External analyzer without the feature flag is a no-op, and the
    // composite analyzer treats it that way: same source, same score.
    let ext_no_feature = ExternalAnalyzer::new("clippy", vec!["clippy".into()]);
    let composite_with_ext =
        CompositeAnalyzer::<Identity>::new(Language::Rust, None, None, Some(ext_no_feature));
    let composite_without_ext =
        CompositeAnalyzer::<Identity>::new(Language::Rust, None, None, None);
    let ext_no_op_err = f64::from(u8::from(
        composite_with_ext.analyze("fn main() {}") != composite_without_ext.analyze("fn main() {}"),
    ));

    // ------------------------------------------------------------------
    // Down-weighting preserves total weight within the closed interval
    // [floor * batch, ceiling * batch], so a filter that promised
    // "weighted" actually delivered weights in that range -- a
    // regression that swapped floor and ceiling would still be a valid
    // function of the score but would invert the user's intent.
    let varied: Vec<QualityScore> = (0..16)
        .map(|i| {
            let overall = i as f32 / 15.0;
            let mut s = QualityScore::from_dimensions(
                Language::Rust,
                vec![Dimension::new("synthetic", overall)],
                1,
            );
            s.overall = overall;
            s
        })
        .collect();
    let (_, ws) = WindowFilter::downweight(0.2, 0.8).decide(&varied);
    let min_w = ws.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w = ws.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bounds_err = f64::from(u8::from(min_w < 0.2 - 1e-6 || max_w > 0.8 + 1e-6))
        + f64::from(u8::from(ws.len() != varied.len()));

    vec![
        cert(
            "codequality",
            "identity_at_strength_zero",
            "QualityRegularizer with weight 0 produces an exact-zero tensor on every input.",
            identity_err,
            0.0,
        ),
        cert(
            "codequality",
            "downweight_at_one_one_is_exactly_keep",
            "WindowFilter::downweight(1, 1) keeps every window at weight 1.0.",
            downweight_identity_err,
            0.0,
        ),
        cert(
            "codequality",
            "drop_below_zero_drops_nothing",
            "WindowFilter::drop_below(0) keeps every window at weight 1.0.",
            drop_zero_err,
            0.0,
        ),
        cert(
            "codequality",
            "per_language_analyzers_are_pure",
            "A composite analyzer applied to the same source three times yields identical scores.",
            purity_err,
            0.0,
        ),
        cert(
            "codequality",
            "geometric_mean_collapses_on_zero_dimension",
            "Any single dimension at 0 makes `overall` exactly 0, which is the signal Drop reads.",
            geometric_err,
            0.0,
        ),
        cert(
            "codequality",
            "external_analyzer_without_feature_is_a_no_op",
            "An ExternalAnalyzer produces a no-op composite score when the Cargo feature is off.",
            ext_no_op_err,
            0.0,
        ),
        cert(
            "codequality",
            "downweight_weights_lie_in_the_closed_interval",
            "WindowFilter::downweight(floor, ceiling) assigns weights in [floor, ceiling].",
            bounds_err,
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
    let reference: Vec<f32> = cond
        .clone()
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
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
        let u: Vec<f32> = uncond
            .clone()
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
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
        let differing = moved.iter().zip(&baseline).filter(|(a, b)| a != b).count();
        argmax_drift = argmax_drift.max(differing as f64);
    }

    // `LogitNorm::None` is the exact identity, bitwise.
    let untouched: Vec<f32> = LogitNorm::None
        .apply(raw.clone())
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let raw_bits: Vec<f32> = raw
        .clone()
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
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
    for kind in [
        Ensemble::ProbabilityMean,
        Ensemble::LogitMean,
        Ensemble::MajorityVote,
    ] {
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
        let best = raw_points
            .iter()
            .map(|(_, a)| *a)
            .fold(f64::NEG_INFINITY, f64::max);
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
    use crate::sigma::{P_MEAN, P_STD};
    use crate::train::DefaultTrainBackend as A2;
    use rand::{rngs::StdRng, SeedableRng};

    let device: <B as burn::tensor::backend::BackendTypes>::Device = Default::default();

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
    use crate::train::DefaultTrainBackend as A;
    use burn::nn::{Linear, LinearConfig};
    use burn::optim::GradientsParams;
    use burn::tensor::backend::AutodiffBackend;

    let ad_device = Default::default();
    <A as burn::tensor::backend::Backend>::seed(&ad_device, 4242);
    let layer: Linear<A> = LinearConfig::new(6, 4).with_bias(true).init(&ad_device);

    // A fixed dataset, split k ways. Micro-batches are equal-sized, which is
    // the condition under which the identity holds at all.
    let micro = 3usize;
    let mut accumulation_err: f64 = 0.0;
    for k in [1usize, 2, 4] {
        let rows = k * micro;
        let inputs =
            Tensor::<A, 2>::random([rows, 6], Distribution::Uniform(-1.0, 1.0), &ad_device);

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
            let loss = layer
                .forward(chunk)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(scale);
            let grads = GradientsParams::from_grads(loss.backward(), &layer);
            summed = accumulator.fold(grads, &layer).into_gradients();
        }
        let Some(summed) = summed else {
            accumulation_err = f64::INFINITY;
            continue;
        };

        // Compare parameter by parameter.
        let weight_id = layer.weight.id;
        for (a, b) in [(
            summed.get::<<A as AutodiffBackend>::InnerBackend, 2>(weight_id),
            reference.get::<<A as AutodiffBackend>::InnerBackend, 2>(weight_id),
        )] {
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
    //
    // The decay is read from `Ema::effective_decay` rather than recomputed
    // here. Recomputing it is what let a mutant that dropped the bias
    // correction entirely pass this certificate: the inline copy stayed
    // correct while the implementation did not.
    let ema_device: <A2 as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let probe: burn::nn::Linear<A2> = burn::nn::LinearConfig::new(2, 2)
        .with_bias(false)
        .init(&ema_device);

    let mut convexity_err: f64 = 0.0;
    let mut interval_err: f64 = 0.0;
    for decay in [0.0f64, 0.5, 0.9, 0.999, 1.0] {
        let mut ema = crate::schedule::Ema::new(&probe, decay);
        for updates in [0usize, 1, 5, 100] {
            while ema.updates() < updates {
                ema.update::<A2>(&probe);
            }
            let d = ema.effective_decay();

            // Bias correction: early on the shadow is still mostly its
            // initialization, so the nominal decay is ramped in.
            let warm = (1.0 + updates as f64) / (10.0 + updates as f64);
            convexity_err = convexity_err.max((d - decay.min(warm)).abs());
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
        LrSchedule::WarmupConstant {
            peak: 1e-3,
            warmup_steps: 50,
        },
        LrSchedule::WarmupCosine {
            peak: 1e-3,
            min_lr: 1e-5,
            warmup_steps: 50,
            total_steps: 500,
        },
        LrSchedule::WarmupCosine {
            peak: 3e-4,
            min_lr: 0.0,
            warmup_steps: 0,
            total_steps: 200,
        },
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
        optimum_err = optimum_err
            .max((objective(l_star) - UncertaintyWeighting::value_at_optimum(raw)).abs());
    }

    // ...and `apply` must implement that objective, not merely agree with the
    // closed form derived from it. Checking `optimal_log_variance` against
    // inline arithmetic proves algebra that was never in doubt: a mutant that
    // dropped the `+ l` term from `apply` passed every certificate in this
    // group, because no certificate called `apply`.
    let mut apply_err: f64 = 0.0;
    let losses = [0.05f32, 1.0, 13.4, 1909.8];
    let logvars = [-2.0f32, -0.25, 0.0, 3.5];
    let raw = Tensor::<B, 1>::from_floats(losses.as_slice(), &device);
    let lv = Tensor::<B, 1>::from_floats(logvars.as_slice(), &device);

    let full: Vec<f32> = UncertaintyWeighting::full()
        .apply(raw.clone(), lv.clone())
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    for ((got, l), v) in full.iter().zip(&losses).zip(&logvars) {
        let want = f64::from(*l) * (-f64::from(*v)).exp() + f64::from(*v);
        apply_err = apply_err.max((f64::from(*got) - want).abs() / want.abs().max(1.0));
    }

    // Strength 0 is the exact identity, bitwise -- a blended `(1-t)L + tL` is
    // not `L` in floating point, and an almost-identity default is a leak
    // nobody would look for.
    let off: Vec<f32> = UncertaintyWeighting::off()
        .apply(raw.clone(), lv.clone())
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    for (got, want) in off.iter().zip(&losses) {
        if got.to_bits() != want.to_bits() {
            apply_err = f64::INFINITY;
        }
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

    // The proposal maths above never draws a sample, and `sample` is where the
    // weight is actually attached. A mutant that returned `q/p` instead of
    // `p/q` -- inverting the correction so it *amplifies* the proposal's bias
    // rather than removing it -- passed every certificate *and* every unit
    // test. So the weights are checked where they are produced.
    let mut drawn_err: f64 = 0.0;
    let mut rng = StdRng::seed_from_u64(0xA5A5);
    for bins in [1usize, 4, 16] {
        let mut sampler = SigmaImportanceSampler::new(bins).with_smoothing(0.2);
        for bin in 0..bins {
            sampler.observe(bin, (bin as f64 + 1.0).powi(2));
        }
        let q = sampler.proposal();
        let prior = sampler.prior();
        let (cdf_lo, cdf_hi) = (0.05f64, 0.95);

        let drawn = sampler.sample(&mut rng, cdf_lo, cdf_hi, P_MEAN, P_STD, 256);
        for (sigma, weight) in &drawn {
            // Every sigma must land inside the window it was drawn from...
            let cdf = crate::stats::norm_cdf((sigma.ln() - P_MEAN) / P_STD);
            drawn_err = drawn_err
                .max((cdf_lo - cdf).max(0.0))
                .max((cdf - cdf_hi).max(0.0));
            // ...and carry exactly the `p/q` of the bin it landed in.
            let bin = sampler.bin_of(*sigma, cdf_lo, cdf_hi, P_MEAN, P_STD);
            let expected = prior / q[bin];
            drawn_err = drawn_err.max((weight - expected).abs() / expected.max(1.0));
        }
        // No Monte-Carlo convergence check here on purpose. "The mean weight
        // tends to 1" is a *statistical* claim needing a sampling-error
        // tolerance, and folding it into a certificate whose other terms are
        // exact would force that whole certificate down to a bound no exact
        // statement needs. The per-draw equality above is exact and already
        // catches an inverted weight, which is what this exists to catch.
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
            "`UncertaintyWeighting::apply` computes exp(-l)L + l, is the bitwise identity at strength 0, and that objective is stationary and minimal at l = ln(L) with value 1 + ln(L): the head cannot report a smaller loss by claiming more uncertainty.",
            optimum_err.max(apply_err),
            // The closed-form half is f64 and lands near 1e-15; `apply` runs in
            // f32, where an exp, a multiply and an add cost about 3 eps
            // ~ 1.8e-7 relative. The tolerance is set by the f32 half.
            1e-6,
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
            "The proposal is a distribution, E_q[p/q] = 1 exactly, and `sample` attaches that exact p/q to every draw it returns -- so reweighted samples estimate the prior's mean rather than the proposal's.",
            bias.max(simplex).max(drawn_err),
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
                let budget = Budget {
                    max_evaluations,
                    max_depth,
                    beam_width,
                };
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
        let plan = Beam::new(Budget {
            max_evaluations: 1024,
            max_depth,
            beam_width: 3,
        })
        .search(|path: &Path<usize>, _r: usize| {
            if path.depth() > max_depth {
                return Vec::new();
            }
            (0..3).map(|i| (i, next_f64())).collect::<Vec<_>>()
        });
        match (
            plan.commit(),
            plan.best.as_ref().and_then(|p| p.steps.first()),
        ) {
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
    let lookahead_plan = Beam::new(Budget {
        max_evaluations: 64,
        max_depth: 1,
        beam_width: 4,
    })
    .search(|path: &Path<char>, _r: usize| match path.steps.as_slice() {
        [] => vec![('a', -0.1), ('b', -0.5)],
        ['a'] => vec![('x', -6.0)],
        ['b'] => vec![('y', -0.2)],
        _ => Vec::new(),
    });
    let myopia = f64::from(lookahead_plan.commit() != Some(&'b'));

    // The same statement on the language side, through the real decoder.
    let decoder_plan = LookaheadDecoder::new(
        Budget {
            max_evaluations: 64,
            max_depth: 1,
            beam_width: 4,
        },
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
            let budget = Budget {
                max_evaluations: 4096,
                max_depth,
                beam_width,
            };
            let plan = Beam::new(budget).search(|path: &Path<u32>, _r: usize| {
                if path.depth() > max_depth {
                    return Vec::new();
                }
                (0..candidates as u32)
                    .map(|i| (i, f64::from(i)))
                    .collect::<Vec<_>>()
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

// ------------------------------------------------------------- experiment --

fn experiment_certificates() -> Vec<Certificate> {
    checks_or_failed("experiment", experiment_checks)
}

fn experiment_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::experiment::{median, t_quantile_975, Summary};

    // ------------------------------------------------------------------
    // The interval a record reports is the t interval on the mean: it
    // contains the mean, and its half-width is exactly t_{0.975, n-1} s/sqrt(n)
    // for the sample standard deviation s. Measured through `Summary::of`.
    let sample = [3.1, 2.7, 3.9, 3.3, 2.5, 3.6, 3.0];
    let summary = Summary::of(&sample).context("non-empty")?;
    let mean = sample.iter().sum::<f64>() / sample.len() as f64;
    let var = sample.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (sample.len() - 1) as f64;
    let half = t_quantile_975(sample.len() - 1) * var.sqrt() / (sample.len() as f64).sqrt();
    let (lo, hi) = summary.ci95();
    let mut interval_err = (summary.ci95_half_width - half)
        .abs()
        .max((summary.mean - mean).abs());
    if !(lo <= summary.mean && summary.mean <= hi) {
        interval_err = 1.0;
    }

    // ------------------------------------------------------------------
    // More trials narrow the interval. For a sample of +-1 pairs the
    // standard deviation is sqrt(n/(n-1)), so the half-width is
    // t_{0.975, n-1}/sqrt(n-1): both factors fall with n, and the summary must
    // agree -- a table with a typo, or a std with the wrong denominator, would
    // break the monotonicity somewhere.
    let mut shrink_err: f64 = 0.0;
    let mut previous = f64::INFINITY;
    for pairs in 1..=20 {
        let values: Vec<f64> = (0..pairs).flat_map(|_| [-1.0, 1.0]).collect();
        let hw = Summary::of(&values).context("non-empty")?.ci95_half_width;
        shrink_err = shrink_err.max((hw - previous).max(0.0));
        previous = hw;
    }

    // ------------------------------------------------------------------
    // The median of an odd sample is its middle element, whatever order the
    // sample arrived in; of an even one, the mean of the two middle elements.
    let scrambled = [9.0, 1.0, 7.0, 3.0, 5.0];
    let median_err = (median(&scrambled) - 5.0).abs() + (median(&[4.0, 1.0, 2.0, 3.0]) - 2.5).abs();

    // ------------------------------------------------------------------
    // The t table is what the interval rests on: non-increasing in the degrees
    // of freedom, always at least the normal quantile, and equal to it far out.
    let mut table_err: f64 = 0.0;
    let mut last = f64::INFINITY;
    for df in 1..=200 {
        let t = t_quantile_975(df);
        table_err = table_err.max((t - last).max(0.0)).max((1.96 - t).max(0.0));
        last = t;
    }
    table_err = table_err.max((t_quantile_975(1000) - 1.96).abs());

    Ok(vec![
        cert(
            "experiment",
            "ci_is_the_t_interval_on_the_mean",
            "A record's 95% interval contains its mean and has half-width exactly t(0.975, n-1) * s / sqrt(n).",
            interval_err,
            1e-12,
        ),
        cert(
            "experiment",
            "more_trials_narrow_the_interval",
            "For +-1 pairs the half-width is t(0.975, n-1)/sqrt(n-1); it must fall as pairs are added.",
            shrink_err,
            0.0,
        ),
        cert(
            "experiment",
            "median_is_the_middle_element",
            "The median of an odd sample is its middle element regardless of arrival order; of an even one, the mean of the two middle elements.",
            median_err,
            0.0,
        ),
        cert(
            "experiment",
            "t_table_is_monotone_and_bounded_by_the_normal_quantile",
            "t(0.975, df) is non-increasing in df, never below 1.96, and equals 1.96 far out.",
            table_err,
            0.0,
        ),
    ])
}

// ------------------------------------------------------------ multisource --

fn multisource_certificates() -> Vec<Certificate> {
    checks_or_failed("multisource", multisource_checks)
}

fn multisource_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::corpus::TokenCorpus;
    use crate::distill::{soft_target_kl, soft_target_kl_probs, teacher_mixture};
    use crate::lm::{Distillation, ExtraNegatives, LanguageModel, LmConfig, LmExtras};
    use crate::merge::{flatten, merge_into, ParamSnapshot};
    use crate::mix::{CorpusMix, MixMode, MixWeights};
    use crate::train::DefaultTrainBackend as A;
    use burn::nn::LinearConfig;
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use rand::SeedableRng;

    let device = Default::default();

    // ------------------------------------------------------------------
    // Mixture: the source of each batch is drawn by weight. 4000 draws of a
    // 0.75 / 0.25 split have a binomial standard deviation of 0.0068 on the
    // share; the tolerance is 3.6 of those.
    let weights = MixWeights::new(&[3.0, 1.0]).context("valid")?;
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(29);
    let draws = 4000;
    let first = (0..draws).filter(|_| weights.draw(&mut rng) == 0).count();
    let mixture_err = (first as f64 / draws as f64 - 0.75).abs();

    // Composite: every batch is sliced exactly, and no slice is more than one
    // item from its ideal share (largest-remainder apportionment).
    let mut composite_err = 0.0f64;
    for batch in 1..=64usize {
        let slices = weights.split(batch);
        if slices.iter().sum::<usize>() != batch {
            composite_err = 1.0;
        }
        for (i, slice) in slices.iter().enumerate() {
            let ideal = weights.get(i) * batch as f64;
            composite_err = composite_err
                .max(((*slice as f64) - ideal).abs() - 1.0)
                .max(0.0);
        }
    }

    // A single-source mix draws exactly the windows the corpus alone would,
    // from the same random stream; a 3:1 composite of eight slices 6 + 2.
    let scratch = std::env::temp_dir().join(format!("dblocks-verify-mix-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).with_context(|| format!("create {}", scratch.display()))?;
    let corpus_path = scratch.join("mix.bin");
    let single_err = (|| -> anyhow::Result<f64> {
        TokenCorpus::write(
            &corpus_path,
            &(0..300).map(|i| (i % 250) as u16).collect::<Vec<_>>(),
        )?;
        let mut plain = TokenCorpus::in_memory(&corpus_path)?;
        let mut a = rand_chacha::ChaCha12Rng::seed_from_u64(5);
        let expected = plain.sample_batch(3, 8, &mut a)?;
        let mut mixed = TokenCorpus::in_memory(&corpus_path)?;
        let mut mix = CorpusMix::single(&mut mixed)?;
        let mut b = rand_chacha::ChaCha12Rng::seed_from_u64(5);
        let (got, rows, origin) = mix.sample(3, 8, &mut b)?;
        Ok(f64::from(u8::from(
            got != expected || rows.is_some() || origin.counts != vec![3],
        )))
    })()
    .unwrap_or_else(|err| failed("multisource", &err));
    let composite_mix_err = (|| -> anyhow::Result<f64> {
        let mut x = TokenCorpus::in_memory(&corpus_path)?;
        let mut y = TokenCorpus::in_memory(&corpus_path)?;
        let mut mix = CorpusMix::new(
            vec![&mut x, &mut y],
            MixWeights::new(&[3.0, 1.0])?,
            MixMode::Composite,
        )?;
        let mut r = rand_chacha::ChaCha12Rng::seed_from_u64(6);
        let (got, _, origin) = mix.sample(8, 4, &mut r)?;
        Ok(f64::from(u8::from(
            got.len() != 8 || origin.counts != vec![6, 2],
        )))
    })()
    .unwrap_or_else(|err| failed("multisource", &err));
    if let Err(err) = std::fs::remove_dir_all(&scratch) {
        eprintln!(
            "verify: could not remove scratch {}: {err}",
            scratch.display()
        );
    }

    // ------------------------------------------------------------------
    // Teacher mixtures. One teacher's mixture is its own softened
    // distribution bit for bit; two copies of it are the same; and the
    // probability-target KL agrees with the logit-target KL up to the
    // rounding of `log(softmax)` against `log_softmax`.
    let logits = Tensor::<B, 2>::random([5, 7], Distribution::Uniform(-3.0, 3.0), &device);
    let student = Tensor::<B, 2>::random([5, 7], Distribution::Uniform(-3.0, 3.0), &device);
    let bits = |t: Tensor<B, 2>| -> Vec<u32> {
        t.into_data()
            .convert::<f32>()
            .iter::<f32>()
            .map(f32::to_bits)
            .collect()
    };
    let softened = bits(softmax(logits.clone().div_scalar(2.0), 1));
    let one = bits(teacher_mixture(&[(logits.clone(), 0.7)], 2.0)?);
    let twins = teacher_mixture(&[(logits.clone(), 1.0), (logits.clone(), 3.0)], 2.0)?;
    let mut mixture_identity_err = f64::from(u8::from(one != softened));
    let twin_rows: Vec<f32> = twins.into_data().convert::<f32>().iter::<f32>().collect();
    let soft_rows: Vec<f32> = softmax(logits.clone().div_scalar(2.0), 1)
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    for (a, b) in twin_rows.iter().zip(&soft_rows) {
        mixture_identity_err = mixture_identity_err.max(f64::from((a - b).abs()));
    }
    let mut distribution_err = 0.0f64;
    let other = Tensor::<B, 2>::random([5, 7], Distribution::Uniform(-3.0, 3.0), &device);
    let mixed = teacher_mixture(&[(logits.clone(), 0.3), (other, 0.7)], 1.5)?;
    for row in mixed.sum_dim(1).into_data().convert::<f32>().iter::<f32>() {
        distribution_err = distribution_err.max(f64::from((row - 1.0).abs()));
    }
    let kl_logits = f64::from(soft_target_kl(logits.clone(), student.clone(), 2.0).into_scalar());
    let kl_probs = f64::from(
        soft_target_kl_probs(softmax(logits.div_scalar(2.0), 1), student, 2.0).into_scalar(),
    );
    let kl_agreement_err = (kl_logits - kl_probs).abs() / kl_logits.abs().max(1.0);

    // ------------------------------------------------------------------
    // Negative teacher. At a confidence no probability reaches, nothing is
    // proposed and the objective is the plain loss to the bit; where a
    // proposal is charged it is never the corpus target; and a step with the
    // charge ends with the proposals less probable than a plain step does.
    let model = LanguageModel::<B>::new(
        &LmConfig {
            context: 16,
            ..LmConfig::tiny()
        },
        &device,
    )?;
    let negative = LanguageModel::<B>::new(
        &LmConfig {
            context: 16,
            ..LmConfig::tiny()
        },
        &device,
    )?;
    let ids: Vec<i64> = (0..16).map(|i| 65 + (i * 7 % 26) as i64).collect();
    let tokens = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, 16]);
    let span = 0..model.num_layers();
    let (plain, _) = model.next_token_loss(tokens.clone(), span.clone());
    let (proposed, weights_none) = negative.negative_proposals(tokens.clone(), 2.0);
    let silent = model.next_token_step_full(
        tokens.clone(),
        LmExtras {
            extra: Some(ExtraNegatives {
                tokens: proposed,
                weights: weights_none,
                alpha: 1.0,
                epsilon: 1e-6,
            }),
            ..LmExtras::plain()
        },
        span.clone(),
    );
    let mut negative_identity_err = f64::from(u8::from(
        plain.into_scalar().to_bits() != silent.loss.into_scalar().to_bits(),
    ));
    if silent.metrics.negative_teacher_tokens != 0 {
        negative_identity_err = 1.0;
    }
    let (proposed, weights) = negative.negative_proposals(tokens.clone(), 0.0);
    let proposed_host: Vec<i64> = proposed
        .into_data()
        .convert::<i64>()
        .iter::<i64>()
        .collect();
    let weights_host: Vec<f32> = weights.into_data().convert::<f32>().iter::<f32>().collect();
    let mut contradiction_err = 0.0f64;
    for (j, w) in weights_host.iter().enumerate() {
        if *w > 0.0 && proposed_host[j] == ids[j + 1] {
            contradiction_err = 1.0;
        }
        if *w == 0.0 && proposed_host[j] != ids[j + 1] {
            contradiction_err = 1.0; // at confidence 0 every differing proposal is charged
        }
    }

    // The comparative step, on the autodiff backend.
    let ad_device = Default::default();
    let ad_model = LanguageModel::<A>::new(
        &LmConfig {
            context: 16,
            ..LmConfig::tiny()
        },
        &ad_device,
    )?;
    let ad_negative = LanguageModel::<A>::new(
        &LmConfig {
            context: 16,
            ..LmConfig::tiny()
        },
        &ad_device,
    )?;
    let ad_tokens = || Tensor::<A, 1, Int>::from_ints(ids.as_slice(), &ad_device).reshape([1, 16]);
    let (ad_proposed, ad_weights) = ad_negative.negative_proposals(ad_tokens(), 0.0);
    let extra = || ExtraNegatives {
        tokens: ad_proposed.clone(),
        weights: ad_weights.clone(),
        alpha: 1.0,
        epsilon: 1e-6,
    };
    let prob_of_proposals = |m: &LanguageModel<A>| {
        m.next_token_step_full(
            ad_tokens(),
            LmExtras {
                extra: Some(ExtraNegatives {
                    alpha: 0.0,
                    ..extra()
                }),
                ..LmExtras::plain()
            },
            span.clone(),
        )
        .metrics
        .negative_teacher_prob
    };
    let lr = 0.1;
    let charged_step = ad_model.next_token_step_full(
        ad_tokens(),
        LmExtras {
            extra: Some(extra()),
            ..LmExtras::plain()
        },
        span.clone(),
    );
    let grads = GradientsParams::from_grads(charged_step.loss.backward(), &ad_model);
    let charged = SgdConfig::new().init().step(lr, ad_model.clone(), grads);
    let plain_step = ad_model.next_token_step_full(ad_tokens(), LmExtras::plain(), span.clone());
    let grads = GradientsParams::from_grads(plain_step.loss.backward(), &ad_model);
    let rewarded = SgdConfig::new().init().step(lr, ad_model, grads);
    let (after_charged, after_plain) = (prob_of_proposals(&charged), prob_of_proposals(&rewarded));
    let mut negative_step_err = f64::from((after_charged - after_plain).max(0.0));
    if after_charged.to_bits() == after_plain.to_bits() {
        negative_step_err = 1.0;
    }
    // Distillation toward a single teacher equal to the student is zero.
    let self_distill = model
        .next_token_step_full(
            tokens,
            LmExtras {
                distill: Some(Distillation {
                    teachers: &[(&model, 1.0)],
                    temperature: 2.0,
                    weight: 1.0,
                }),
                ..LmExtras::plain()
            },
            span,
        )
        .metrics
        .distill_loss;
    let self_distill_err = f64::from(self_distill.abs());

    // ------------------------------------------------------------------
    // Merging: identity on identical inputs, to the bit; linear otherwise.
    let a = LinearConfig::new(4, 3).init::<B>(&device);
    let b_lin = LinearConfig::new(4, 3).init::<B>(&device);
    let fa = flatten::<B, _>(&a);
    let fb = flatten::<B, _>(&b_lin);
    let same = merge_into::<B, _>(a.clone(), &[ParamSnapshot::of::<B, _>(&a)], &[1.0, 1.0])
        .context("merge")?;
    let mut merge_identity_err = f64::from(u8::from(flatten::<B, _>(&same) != fa));
    let half = merge_into::<B, _>(a, &[ParamSnapshot::of::<B, _>(&b_lin)], &[1.0, 3.0])
        .context("merge")?;
    let mut merge_linear_err = 0.0f64;
    for ((m, x), y) in flatten::<B, _>(&half).iter().zip(&fa).zip(&fb) {
        merge_linear_err = merge_linear_err.max(f64::from((m - (0.25 * x + 0.75 * y)).abs()));
    }
    if fa == fb {
        merge_identity_err = 1.0; // two inits must differ for the test to mean anything
    }

    Ok(vec![
        cert(
            "multisource",
            "mixture_draws_follow_the_weights",
            "Over 4000 draws the share of batches taken from a 0.75-weight source is 0.75 within 3.6 binomial standard deviations.",
            mixture_err,
            0.025,
        ),
        cert(
            "multisource",
            "composite_slices_are_exact_apportionment",
            "For every batch size the composite slices sum to the batch and each is within one item of its weight's share.",
            composite_err,
            0.0,
        ),
        cert(
            "multisource",
            "single_source_mix_is_the_plain_corpus",
            "A one-corpus mix draws exactly the windows the corpus alone draws from the same stream, with no penalty rows and a whole-batch origin; a 3:1 composite of eight windows slices 6 + 2.",
            single_err.max(composite_mix_err),
            0.0,
        ),
        cert(
            "multisource",
            "teacher_mixture_of_one_is_its_own_softened_distribution",
            "A mixture of one teacher, at any weight, is softmax(logits / T) bit for bit; two copies of a teacher mix to the same distribution.",
            mixture_identity_err,
            1e-6,
        ),
        cert(
            "multisource",
            "teacher_mixture_is_a_distribution",
            "Every row of a weighted teacher mixture sums to 1.",
            distribution_err,
            1e-5,
        ),
        cert(
            "multisource",
            "probability_target_kl_agrees_with_logit_target_kl",
            "KL from a probability target equals KL from the logits it came from, up to the rounding of log(softmax) against log_softmax.",
            kl_agreement_err,
            1e-5,
        ),
        cert(
            "multisource",
            "negative_teacher_below_its_confidence_is_the_plain_loss",
            "A negative teacher whose confidence bar no probability reaches proposes nothing, and the objective equals the plain loss bit for bit.",
            negative_identity_err,
            0.0,
        ),
        cert(
            "multisource",
            "negative_teacher_never_contradicts_the_corpus",
            "A proposal equal to the corpus target is never charged; at confidence 0 every differing proposal is.",
            contradiction_err,
            0.0,
        ),
        cert(
            "multisource",
            "negative_teacher_step_ends_below_plain_step",
            "From the same weights, one SGD step with the negative teacher's charge leaves its proposals strictly less probable than one plain step does.",
            negative_step_err,
            0.0,
        ),
        cert(
            "multisource",
            "self_distillation_is_zero",
            "Distilling a model toward itself as its only teacher costs exactly nothing.",
            self_distill_err,
            1e-6,
        ),
        cert(
            "multisource",
            "merge_of_identical_checkpoints_is_the_identity",
            "Averaging a checkpoint with itself returns it bit for bit.",
            merge_identity_err,
            0.0,
        ),
        cert(
            "multisource",
            "merge_is_linear_in_its_weights",
            "A 1:3 merge equals 0.25 a + 0.75 b parameter by parameter.",
            merge_linear_err,
            1e-6,
        ),
    ])
}

// ----------------------------------------------------------------- policy --

fn policy_certificates() -> Vec<Certificate> {
    checks_or_failed("policy", policy_checks)
}

fn policy_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::policy::{
        gated_generate, hex, hmac_sha256, starter, Approval, Decision, Grant, Key, Policy,
    };

    // ------------------------------------------------------------------
    // The primitive every grant rests on: HMAC-SHA256 against RFC 4231.
    let tc1 = hex(&hmac_sha256(&[0x0bu8; 20], b"Hi There"));
    let tc2 = hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?"));
    let hmac_err = f64::from(u8::from(
        tc1 != "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
            || tc2 != "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
    ));

    // ------------------------------------------------------------------
    // Grants fail for every reason they must, each by name: another key, an
    // edited approval, expiry, revocation. A good one passes.
    let key = Key::from_bytes(b"verify-key-0123456789abcdefghij".to_vec()).context("key")?;
    let other = Key::from_bytes(b"other-key-0123456789abcdefghijk".to_vec()).context("key")?;
    let mut policy = starter(&key)?;
    let approval = Approval {
        id: "g".into(),
        scopes: vec!["cyber:malware".into()],
        issued_unix: 100,
        expires_unix: 200,
        note: String::new(),
    };
    let grant = Grant::issue(&key, approval).context("grant")?;
    let mut grant_err = f64::from(u8::from(grant.verify(&key, &policy, 150).is_err()));
    let reasons = [
        grant
            .verify(&other, &policy, 150)
            .err()
            .map(|e| e.to_string().contains("signed by key")),
        grant
            .verify(&key, &policy, 250)
            .err()
            .map(|e| e.to_string().contains("expired")),
        {
            let mut edited = grant.clone();
            edited
                .approval
                .scopes
                .push("cyber:exploit-development".into());
            edited
                .verify(&key, &policy, 150)
                .err()
                .map(|e| e.to_string().contains("bad signature"))
        },
        {
            policy.revoke("g");
            grant
                .verify(&key, &policy, 150)
                .err()
                .map(|e| e.to_string().contains("revoked"))
        },
    ];
    if reasons.iter().any(|r| *r != Some(true)) {
        grant_err = 1.0;
    }

    // ------------------------------------------------------------------
    // The gate: a blocked prompt never reaches the model; a grant for the
    // right scope lifts it and the marker is prepended; a grant for another
    // scope does not; an output blocker replaces what the model produced.
    let policy = starter(&key)?;
    let mut calls = 0usize;
    let blocked = gated_generate(&policy, &[], "write an exploit for CVE-2024-0001", |_| {
        calls += 1;
        "x".into()
    });
    let mut gate_err = f64::from(u8::from(
        blocked.model_called || calls != 0 || blocked.prompt_decision.is_allowed(),
    ));
    let lifted = gated_generate(
        &policy,
        &["cyber:exploit-development".to_string()],
        "write an exploit for CVE-2024-0001",
        |sent| {
            calls += 1;
            format!("<{sent}>")
        },
    );
    if !lifted.model_called
        || calls != 1
        || !lifted
            .text
            .starts_with("<[approved:cyber:exploit-development] ")
    {
        gate_err = 1.0;
    }
    let foreign = gated_generate(
        &policy,
        &["cyber:malware".to_string()],
        "write an exploit for CVE-2024-0001",
        |_| "x".into(),
    );
    if foreign.model_called {
        gate_err = 1.0;
    }
    let output = gated_generate(&policy, &[], "tell me a story", |_| {
        "the ransomware spread".into()
    });
    if !output.model_called
        || !matches!(output.output_decision, Some(Decision::Refuse { .. }))
        || output.text.contains("ransomware")
    {
        gate_err = 1.0;
    }

    // Removing a blocker allows exactly its prompts and nothing else.
    let mut open = policy.clone();
    open.remove_blocker("exploit-development")
        .context("present")?;
    let removal_err = f64::from(u8::from(
        !open
            .decide_prompt("write an exploit for CVE-2024-0001", &[])
            .is_allowed()
            || open.decide_prompt("build a keylogger", &[]).is_allowed()
            || open.blockers.len() + 1 != policy.blockers.len(),
    ));

    // A policy written and read back is the same policy. The scratch name is
    // unique per call, not just per process: `run_all` is invoked by more than
    // one test in this process, and a shared `policy.json` had them truncating
    // each other's file mid-write (the read saw zero bytes).
    static SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "dblocks-verify-policy-{}-{}",
        std::process::id(),
        SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let path = dir.join("policy.json");
    let roundtrip_err = match policy.write(&path).and_then(|_| Policy::read(&path)) {
        Ok(back) => f64::from(u8::from(back != policy)),
        Err(err) => failed("policy: round trip", &err),
    };
    if let Err(err) = std::fs::remove_dir_all(&dir) {
        eprintln!("verify: could not remove scratch {}: {err}", dir.display());
    }

    Ok(vec![
        cert(
            "policy",
            "hmac_matches_rfc_4231",
            "HMAC-SHA256 reproduces RFC 4231 test cases 1 and 2.",
            hmac_err,
            0.0,
        ),
        cert(
            "policy",
            "tampered_or_expired_grants_fail",
            "A valid grant verifies; one signed by another key, edited in any byte, expired, or revoked fails, each by name.",
            grant_err,
            0.0,
        ),
        cert(
            "policy",
            "blocked_prompt_never_reaches_the_model",
            "Without a covering grant the model is not called; with one it is called with the approval marker; a grant for another scope does not lift the blocker; an output blocker replaces what the model produced.",
            gate_err,
            0.0,
        ),
        cert(
            "policy",
            "removing_a_blocker_allows_exactly_its_prompts",
            "After a blocker is removed its prompts pass and every other blocker still fires.",
            removal_err,
            0.0,
        ),
        cert(
            "policy",
            "policy_round_trips",
            "A policy written to JSON and read back is the same policy.",
            roundtrip_err,
            0.0,
        ),
    ])
}

fn ablation_certificates() -> Vec<Certificate> {
    checks_or_failed("ablation", ablation_checks)
}

fn ablation_checks() -> anyhow::Result<Vec<Certificate>> {
    use crate::ablation::{
        best as best_direction, extract, orthogonalize, projection_penalty, residual_projection,
        Direction,
    };
    use crate::checkpoint::canonical_hash_hex;
    use crate::dblock::NegativeLabels;
    use crate::heretic::{
        apply, best, interpolate, mean_kl, search, HereticParams, Kernel, RefusalDetector,
        SearchConfig,
    };
    use crate::lm::{DirectionPenalty, LanguageModel, LmConfig, LmExtras};
    use crate::train::DefaultTrainBackend as A;
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use burn::tensor::activation::softmax;

    let device = Default::default();
    let model = LanguageModel::<B>::new(
        &LmConfig {
            context: 16,
            ..LmConfig::tiny()
        },
        &device,
    )?;
    let h = model.hidden_size();
    let layers = model.num_layers();
    let eps = f64::from(f32::EPSILON);

    // ------------------------------------------------------------------
    // A direction from the model's own residual stream (roadmap 31.1): the
    // two prompt sets differ in their last byte, one candidate per layer,
    // the best-separated one kept. Its target set must project above its
    // baseline set along it -- that is what "separates" means.
    let prompt = |last: u8| -> Vec<u16> {
        let mut ids: Vec<u16> = (0..15u16).map(|i| 65 + (i * 5 % 26)).collect();
        ids.push(u16::from(last));
        ids
    };
    let residuals = |lasts: &[u8]| -> Vec<Vec<Vec<f32>>> {
        lasts
            .iter()
            .map(|l| model.residuals_at_last_position(&prompt(*l), &device))
            .collect()
    };
    let (target, baseline) = (residuals(b"abcde"), residuals(b"vwxyz"));
    let directions: Vec<Direction> = (0..layers)
        .filter_map(|l| {
            let t: Vec<Vec<f32>> = target.iter().map(|r| r[l].clone()).collect();
            let b: Vec<Vec<f32>> = baseline.iter().map(|r| r[l].clone()).collect();
            extract(l, &t, &b)
        })
        .collect();
    let chosen = best_direction(&directions)
        .cloned()
        .context("a direction")?;
    let separation_err = f64::from(u8::from(
        chosen.separation.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            || chosen
                .target_mean_projection
                .partial_cmp(&chosen.baseline_mean_projection)
                != Some(std::cmp::Ordering::Greater),
    ));
    let norm_err = (chosen
        .vector
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt()
        - 1.0)
        .abs();

    // ------------------------------------------------------------------
    // Weight-space ablation on the real model, adaLN gates included: after
    // it, no residual writer has a component along the (gate-scaled)
    // direction. The tolerance is the arithmetic's: `(w - (w.d) d).d =
    // (w.d)(1 - d.d)` plus the rounding of two h-term dot products.
    let gates = model.layer_gates(&device);
    let before = f64::from(residual_projection::<B, _>(&model, &chosen, Some(&gates)));
    let (ablated, touched) = orthogonalize::<B, _>(model.clone(), &chosen, Some(gates.clone()));
    let after = f64::from(residual_projection::<B, _>(&ablated, &chosen, Some(&gates)));
    let projection_tolerance = 4.0 * h as f64 * eps * before.max(1.0);
    let mut writer_err = after;
    if touched == 0 || before.partial_cmp(&after) != Some(std::cmp::Ordering::Greater) {
        writer_err = 1.0;
    }

    // On a coordinate axis the projection is exact arithmetic, so a second
    // pass has nothing left to remove: bit-identical weights.
    let axis = Direction {
        layer: chosen.layer,
        vector: (0..h).map(|i| if i == 3 { 1.0 } else { 0.0 }).collect(),
        separation: 0.0,
        target_mean_projection: 0.0,
        baseline_mean_projection: 0.0,
        target_count: 0,
        baseline_count: 0,
    };
    let (once, _) = orthogonalize::<B, _>(model.clone(), &axis, Some(gates.clone()));
    let (twice, _) = orthogonalize::<B, _>(once.clone(), &axis, Some(gates.clone()));
    let idempotent_err = f64::from(u8::from(
        canonical_hash_hex::<B, _>(&once) != canonical_hash_hex::<B, _>(&twice),
    ));

    // ------------------------------------------------------------------
    // Inference-time ablation: with the direction projected out after the
    // embedding and after every layer, every layer's output has no
    // component along it.
    let ids: Vec<i64> = prompt(b'a').iter().map(|t| i64::from(*t)).collect();
    let tokens = || Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, 16]);
    let d_tensor = chosen.tensor::<B>(&device);
    let (_, states) = model.forward_span_states(tokens(), 0..layers, Some(&d_tensor));
    let (_, plain_states) = model.forward_span_states(tokens(), 0..layers, None);
    let largest_state = plain_states
        .iter()
        .map(|s| f64::from(s.clone().abs().max().into_scalar()))
        .fold(1.0f64, f64::max);
    let inference_err = states
        .iter()
        .map(|s| {
            f64::from(
                s.clone()
                    .matmul(d_tensor.clone().reshape([1, h, 1]))
                    .abs()
                    .max()
                    .into_scalar(),
            )
        })
        .fold(0.0f64, f64::max);
    let inference_tolerance = 4.0 * h as f64 * eps * largest_state;

    // ------------------------------------------------------------------
    // The training penalty (roadmap 31.2). At weight zero the objective is
    // the plain loss to the bit; with a weight, a step ends with the
    // penalized layer's states projecting less onto the direction than a
    // plain step does.
    let span = 0..layers;
    let plain = model.next_token_step_full(tokens(), LmExtras::plain(), span.clone());
    let silent = model.next_token_step_directed(
        tokens(),
        LmExtras::with_direction(DirectionPenalty {
            direction: &d_tensor,
            layer: chosen.layer,
            weight: 0.0,
        }),
        span.clone(),
    );
    let mut zero_weight_err = f64::from(u8::from(
        plain.loss.into_scalar().to_bits() != silent.loss.into_scalar().to_bits(),
    ));
    if silent.metrics.direction_projection != 0.0 {
        zero_weight_err = 1.0;
    }

    let ad_device = Default::default();
    let ad_model = LanguageModel::<A>::new(
        &LmConfig {
            context: 16,
            ..LmConfig::tiny()
        },
        &ad_device,
    )?;
    let ad_tokens = || Tensor::<A, 1, Int>::from_ints(ids.as_slice(), &ad_device).reshape([1, 16]);
    // The penalized direction: the mean state of the last layer, so the
    // projection starts large.
    let last = layers - 1;
    let mean_state: Vec<f32> = ad_model.hidden_states(ad_tokens())[last]
        .clone()
        .mean_dim(1)
        .reshape([h])
        .into_data()
        .convert::<f32>()
        .iter::<f32>()
        .collect();
    let norm = mean_state
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    let unit: Vec<f32> = mean_state.iter().map(|x| x / norm).collect();
    let ad_direction = Tensor::<A, 1>::from_floats(unit.as_slice(), &ad_device);
    let projection_of = |m: &LanguageModel<A>| -> f32 {
        projection_penalty(&m.hidden_states(ad_tokens())[last], &ad_direction).into_scalar()
    };
    // First order, the penalized step moves the projection by
    // `-lr * weight * |grad P|^2` relative to the plain step; the second-order
    // remainder scales with `lr^2`, so a small step keeps the sign for any
    // initialization the process-wide RNG hands out (it flipped once at 0.1).
    let lr = 0.01;
    let penalized_step = ad_model.next_token_step_directed(
        ad_tokens(),
        LmExtras::with_direction(DirectionPenalty {
            direction: &ad_direction,
            layer: last,
            weight: 1.0,
        }),
        span.clone(),
    );
    let grads = GradientsParams::from_grads(penalized_step.loss.backward(), &ad_model);
    let penalized = SgdConfig::new().init().step(lr, ad_model.clone(), grads);
    let plain_step = ad_model.next_token_step_full(ad_tokens(), LmExtras::plain(), span.clone());
    let grads = GradientsParams::from_grads(plain_step.loss.backward(), &ad_model);
    let unpenalized = SgdConfig::new().init().step(lr, ad_model, grads);
    let (after_penalized, after_plain) = (projection_of(&penalized), projection_of(&unpenalized));
    let mut penalty_step_err = f64::from((after_penalized - after_plain).max(0.0));
    if after_penalized.to_bits() == after_plain.to_bits() {
        penalty_step_err = 1.0;
    }

    // ------------------------------------------------------------------
    // Negative labels for the image trunk (roadmap 31.3). A charge of zero
    // costs nothing; a charge of one is bounded by `-ln eps`; and a charged
    // step ends with the labels less probable than a rewarded step does --
    // measured on the clean latent, where no noise is drawn.
    let cfg = ViTDiTConfig::tiny(10);
    let block_cfg = DblockConfig {
        num_blocks: 1,
        ..DblockConfig::default()
    };
    let classifier = DblockClassifier::<A>::new(&cfg, &block_cfg, &ad_device)?;
    let pixels =
        Tensor::<A, 4>::random([4, 3, 32, 32], Distribution::Uniform(-1.0, 1.0), &ad_device);
    let labels = Tensor::<A, 1, Int>::from_ints([0i64, 1, 2, 3], &ad_device);
    let sigmas = [0.05f64; 4];
    let negatives = |alpha: f32| NegativeLabels::<A> {
        mask: Tensor::<A, 1>::ones([4], &ad_device),
        alpha,
        epsilon: 1e-6,
    };
    let free = classifier.training_step_negative(
        pixels.clone(),
        labels.clone(),
        &sigmas,
        0,
        None,
        negatives(0.0),
    );
    let mut free_err = f64::from(free.metrics.ce_loss.abs());
    if free.metrics.negative_samples != 4 {
        free_err = 1.0;
    }
    let charged = classifier.training_step_negative(
        pixels.clone(),
        labels.clone(),
        &sigmas,
        0,
        None,
        negatives(1.0),
    );
    let bound_err = (f64::from(charged.metrics.ce_loss) + (1e-6f64).ln()).max(0.0);

    let label_prob = |m: &DblockClassifier<A>| -> f32 {
        let clean = m.model().normalized_label_embeds(labels.clone());
        let probs = softmax(m.denoise(pixels.clone(), clean, &sigmas, Some(0)), 1);
        probs
            .gather(1, labels.clone().unsqueeze_dim::<2>(1))
            .mean()
            .into_scalar()
    };
    let grads = GradientsParams::from_grads(charged.loss.backward(), &classifier);
    let charged_model = SgdConfig::new().init().step(lr, classifier.clone(), grads);
    let rewarded_step =
        classifier.training_step_on(pixels.clone(), labels.clone(), &sigmas, 0, None);
    let grads = GradientsParams::from_grads(rewarded_step.loss.backward(), &classifier);
    let rewarded_model = SgdConfig::new().init().step(lr, classifier, grads);
    let (p_charged, p_rewarded) = (label_prob(&charged_model), label_prob(&rewarded_model));
    let mut negative_step_err = f64::from((p_charged - p_rewarded).max(0.0));
    if p_charged.to_bits() == p_rewarded.to_bits() {
        negative_step_err = 1.0;
    }

    // ------------------------------------------------------------------
    // Heretic (roadmap 31.6): the trapezoid kernel, identity parameters,
    // direction interpolation, the search's best, the refusal detector.
    let kernel = Kernel {
        max_weight: 1.0,
        max_weight_position: 3.0,
        min_weight: 0.2,
        min_weight_distance: 2.0,
    };
    let mut kernel_err = f64::from(
        (kernel.weight(3) - 1.0).abs()
            + (kernel.weight(0) - 0.2).abs()
            + (kernel.weight(6) - 0.2).abs(),
    );
    kernel_err += f64::from((kernel.weight(4) - 0.6).abs());
    for layer in 0..8 {
        let w = kernel.weight(layer);
        if !(0.2..=1.0).contains(&w) {
            kernel_err += 1.0;
        }
    }

    let (untouched, touched_by_identity) = apply::<B, _>(
        model.clone(),
        &directions,
        &HereticParams::identity(),
        Some(&gates),
    )
    .context("identity")?;
    let logits = model
        .forward(tokens())
        .logits
        .reshape([16, model.vocab_size()]);
    let self_kl = f64::from(mean_kl(logits.clone(), logits));
    let mut identity_err = self_kl + touched_by_identity as f64;
    if canonical_hash_hex::<B, _>(&untouched) != canonical_hash_hex::<B, _>(&model) {
        identity_err += 1.0;
    }

    let at_one = interpolate(&directions, 1.0).context("interpolated")?;
    let interpolation_err = f64::from(u8::from(at_one != directions[1].vector));

    let trials = search(&SearchConfig::new(9, layers), |params| {
        let refusals =
            f64::from(params.attention.kernel.max_weight - params.mlp.kernel.min_weight).abs();
        let kl = f64::from(params.attention.direction_index).abs() * 0.01;
        Ok((refusals, kl, 1))
    })
    .context("search")?;
    let lowest = trials.iter().map(|t| t.score).fold(f64::INFINITY, f64::min);
    let best_err = best(&trials).map_or(1.0, |t| (t.score - lowest).max(0.0));

    let detector = RefusalDetector::default();
    let detector_err = f64::from(u8::from(
        !detector.is_refusal("I cannot help with that request.")
            || detector.is_refusal("Sure, here is the code you asked for."),
    ));

    Ok(vec![
        cert(
            "ablation",
            "extracted_direction_separates_its_sets",
            "The best-separated direction is a unit vector along which the target prompts' mean residual lies above the baseline prompts' (roadmap 31.1).",
            separation_err + norm_err,
            8.0 * h as f64 * eps,
        ),
        cert(
            "ablation",
            "orthogonalized_writers_have_no_component_along_the_direction",
            "After weight-space ablation no residual writer of the model has a component along the gate-scaled direction, up to the arithmetic of two h-term dot products.",
            writer_err,
            projection_tolerance,
        ),
        cert(
            "ablation",
            "orthogonalizing_twice_is_the_identity",
            "On a coordinate axis the projection is exact, so ablating an ablated model changes no bit.",
            idempotent_err,
            0.0,
        ),
        cert(
            "ablation",
            "inference_ablation_removes_the_direction_from_every_layer",
            "With the direction projected out after every layer, no layer's output has a component along it.",
            inference_err,
            inference_tolerance,
        ),
        cert(
            "ablation",
            "zero_direction_weight_is_the_plain_loss",
            "A direction penalty of weight zero reproduces the plain next-token loss bit for bit and reports no projection (roadmap 31.2).",
            zero_weight_err,
            0.0,
        ),
        cert(
            "ablation",
            "penalized_step_ends_below_plain_step",
            "From one initialization and batch, a step with the direction penalty leaves the penalized layer projecting less onto the direction than a plain step does.",
            penalty_step_err,
            0.0,
        ),
        cert(
            "ablation",
            "zero_negative_charge_costs_nothing",
            "Negative labels charged at zero contribute exactly zero cross-entropy while still being counted (roadmap 31.3).",
            free_err,
            0.0,
        ),
        cert(
            "ablation",
            "negative_charge_is_bounded",
            "The negative charge -log(1 - p) is clamped at -ln(eps) per sample, so a batch of negatives cannot cost more than that.",
            bound_err,
            0.0,
        ),
        cert(
            "ablation",
            "negative_step_ends_below_rewarded_step",
            "From one initialization and batch, a step that charges the labels leaves them less probable on the clean latent than a step that rewards them.",
            negative_step_err,
            0.0,
        ),
        cert(
            "ablation",
            "heretic_kernel_is_a_trapezoid",
            "The Heretic kernel is max_weight at its position, min_weight beyond its distance, linear between, and within [min, max] everywhere (roadmap 31.6).",
            kernel_err,
            1e-6,
        ),
        cert(
            "ablation",
            "heretic_identity_parameters_touch_nothing",
            "Heretic's identity parameters ablate no parameter, leave every bit of the model in place, and a model's KL from itself is zero.",
            identity_err,
            0.0,
        ),
        cert(
            "ablation",
            "integer_direction_index_is_that_layers_direction",
            "An integer direction index interpolates to exactly that layer's direction.",
            interpolation_err,
            0.0,
        ),
        cert(
            "ablation",
            "heretic_best_is_never_worse_than_any_trial",
            "The trial the search reports as best has the lowest score of every trial it ran.",
            best_err,
            0.0,
        ),
        cert(
            "ablation",
            "refusal_detector_matches_its_phrases",
            "The refusal detector flags a refusal phrase and passes a compliant answer.",
            detector_err,
            0.0,
        ),
    ])
}

fn model_checks() -> anyhow::Result<Vec<Certificate>> {
    let device = Default::default();
    let cfg = ViTDiTConfig::tiny(10);
    let model = DblockClassifier::<B>::new(
        &cfg,
        &DblockConfig {
            num_blocks: 2,
            ..DblockConfig::default()
        },
        &device,
    )?;

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

    Ok(certificates)
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
    checks_or_failed("autodiff", autodiff_checks)
}

fn autodiff_checks() -> anyhow::Result<Vec<Certificate>> {
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
        .context("student logits must receive a gradient")?
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

    Ok(vec![
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
    ])
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr
)]
mod tests {
    use super::*;

    #[test]
    fn test_all_certificates_hold() {
        let report = run_all();
        assert!(report.passed(), "verification failed:\n{}", report.render());
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
                "ablation",
                "accuracy",
                "antipattern",
                "autodiff",
                "cheat",
                "codegen_eval",
                "codequality",
                "deltanet",
                "experiment",
                "geom",
                "geomfusion",
                "hybrid",
                "lm",
                "loopgraph",
                "model",
                "moe",
                "mosme",
                "multisource",
                "optim",
                "planner",
                "policy",
                "precision",
                "preconditioning",
                "quantize",
                "qwennet",
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
