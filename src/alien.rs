//! ALIEN-FINDER: between-space discovery over specialist vectors (TRISKELE
//! plan §14, §15a, §16).
//!
//! The known vectors are specialists -- the ideas that have been had. A point
//! between them is a *candidate* when it is **coherent** (an affine
//! combination of the knowns reconstructs it, so it recombines things that
//! exist) and **unlikely** (the knowns' kernel density there is low, so no
//! specialist sits on it). This module generates such points, ranks them,
//! records what happened when somebody piloted one, and learns from those
//! records which kinds of candidate tend to survive.
//!
//! # What this does not do
//!
//! It ranks *candidates*. A high alien-gap says a point is reconstructable
//! from known specialists and far from every one of them; it does not say the
//! idea there works, is absent from the literature, or deserves a pilot.
//! [`Plan`]s are written by a person or a later realizer (a teacher or the NCM
//! decoder), never by this module. The survival predictor is only as good as
//! the [`PilotOutcome`]s it was fitted on: below
//! [`SurvivalConfig::min_fit`] outcomes it refuses to fit, and below
//! [`SurvivalConfig::min_report`] held-out outcomes it refuses to report an
//! accuracy.
//!
//! # The between-space math (a port of `prototypes/alien_vectors.py`)
//!
//! With known vectors `V = {v_1..v_n}` in `R^d`:
//!
//! - `coherence(p) = exp(-‖p - affine_recon(p; V)‖ / scale)`, where
//!   `affine_recon` is the closest point of the affine hull `{Σ λ_i v_i :
//!   Σ λ_i = 1}` and `scale` is the median nearest-neighbour distance among
//!   `V` (floored at `1e-9`, as in the prototype).
//! - `likelihood(p)` is the Gaussian KDE `mean_i exp(-‖v_i - p‖² / 2h²)`
//!   with bandwidth `h = factor × median nearest-neighbour distance`
//!   (`factor` 1.5 in the prototype's verified `--dim 64 --seed 7` run),
//!   normalised by `max + 1e-12` over the scored set, as `rank_alien` does.
//! - `alien_gap(p) = coherence(p) · (1 - likelihood_norm(p))`.
//!
//! **The least-squares solve.** The affine constraint is removed by centring
//! on `v_1`: the residual of `p - v_1` against the column space of
//! `A = [v_2 - v_1, …, v_n - v_1]`. That column space is found by
//! **Householder QR with column pivoting** ([`AffineHull`]), and the residual
//! is the norm of the trailing components of `Qᵀ(p - v_1)`. Why this one:
//! Householder QR is backward stable and works on `A` itself, where the
//! normal equations square its condition number (knowns that are nearly
//! collinear -- the normal case for specialists of one field -- would lose
//! half the digits); column pivoting makes it *rank-revealing*, so a
//! rank-deficient `A` (more knowns than the manifold has dimensions) is
//! handled by truncating at the numerical rank with numpy `lstsq`'s own
//! cutoff (`eps · max(d, n-1) · |R_11|`), not by an arbitrary solution; and a
//! Tikhonov term is deliberately absent, because ridge shrinkage moves the
//! reconstruction off the hull and would make an exact affine extrapolation
//! look incoherent in proportion to how far it extrapolates. SVD would also
//! be correct; pivoted QR reaches the same projection with a fraction of the
//! code, and only the projection (never the non-unique `λ`) is used.
//!
//! **The precondition the prototype is explicit about.** Coherence only
//! discriminates when the knowns lie on a lower-dimensional manifold inside
//! the ambient space. If their affine hull has full rank `d` (for instance
//! `n ≥ d + 1` knowns in general position), every point reconstructs exactly
//! and every coherence is 1. [`AffineHull::require_discriminating`] refuses
//! that case with the numbers, instead of returning scores that mean nothing.
//!
//! **Generators** ([`Generator`]): pairwise midpoints; simplex-interior
//! blends with `λ ~ Dirichlet(1)` (seeded); farthest-interior selection
//! (Voronoi emptiness: keep the pool points farthest from their nearest
//! known); slerp midpoints; and Poincaré-ball geodesic midpoints. The curved
//! generators work on directions *from the knowns' centroid* (slerp with the
//! radius interpolated, the Poincaré chart through the exponential map at the
//! centroid), so their betweens lie in the affine hull and are judged by the
//! same coherence as the straight ones.
//!
//! **Controls** that must lose, scored with every run: every known vector
//! (on-known: high likelihood, rejected as obvious) and points pushed from the
//! centroid along the orthogonal complement of the hull (null-space: large
//! residual, coherence ≈ 0, rejected as incoherent).
//!
//! # Deliberate differences from the prototype
//!
//! - The prototype's `poincare_midpoint` computes `a ⊕ (0.5 · ((-a) ⊕ b))`
//!   with a *Euclidean* 0.5: a point on the geodesic, but not its midpoint.
//!   [`poincare_geodesic`] uses Möbius scalar multiplication, so `t = 0.5` is
//!   equidistant from both ends; the prototype's composition is still
//!   reproduced exactly by [`mobius_add`] in the parity test.
//! - Every known is an on-known control (the prototype scores the first
//!   three), which also anchors the likelihood normaliser at the densest
//!   known rather than at whichever candidate happens to be densest.
//! - Pools are streamed through a bounded farthest-interior selection rather
//!   than materialised, so a 10k-dimensional embedding with thousands of
//!   pairs does not hold every pool point in memory.
//!
//! A caveat inherited from the normalisation: likelihood is relative to the
//! densest known, so an isolated specialist (no neighbours in its region)
//! scores a lower likelihood at its own position than a clustered one, and
//! can score a non-zero gap on itself. The on-known control summary reports
//! the worst known's gap for exactly this reason.
//!
//! # Why `crate::geomnonclid` is not reused
//!
//! It is an *exact rational* incidence kernel in the plane (projective,
//! affine, Cayley-Klein disk, spherical rays over `Q`) and deliberately
//! refuses hyperbolic trigonometry. A Poincaré-ball geodesic in `R^d` needs
//! `tanh`/`artanh` in `f64`, which is exactly what that module will not
//! approximate. So the Möbius operations live here, with every point kept at
//! norm `≤ 1 - BALL_EPS`.
//!
//! # Specialist vectors
//!
//! Known vectors come from a JSON file of named vectors ([`read_vectors`]),
//! optionally with a category label per vector (so this module does not
//! depend on a taxonomy module), or from a nina student checkpoint plus a
//! field corpus ([`embed_student`]: the mean hidden state over evenly spaced
//! windows, through the student's public forward pass). Per-category finders
//! run within a category; the top-level finder runs between category
//! centroids ([`rank`]), with its likelihood measured against every member
//! specialist rather than the handful of centroids.
//!
//! Every scope reports the prototype's four discrimination self-checks, and
//! [`ScopeReport::discriminates`] is false (the rendered report says so in a
//! VERDICT line) when any fails. That happens, for instance, inside a
//! category whose specialists are one undifferentiated cluster: a
//! nearest-neighbour bandwidth then spans the whole set, nothing between them
//! is less likely than they are, and the scores rank nothing.

use crate::verify::Certificate;
use anyhow::{bail, ensure, Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

/// Floor on the coherence scale (the prototype's `max(scale, 1e-9)`).
const SCALE_FLOOR: f64 = 1e-9;
/// Added to the likelihood normaliser (the prototype's `lik.max() + 1e-12`).
const NORMALISER_EPS: f64 = 1e-12;
/// A projected null-space direction shorter than this is dropped (prototype).
const NULL_DIRECTION_FLOOR: f64 = 1e-9;
/// Below this angle slerp returns its first direction (prototype).
const SLERP_COLLINEAR: f64 = 1e-6;
/// Every Poincaré-ball point is kept at norm `<= 1 - BALL_EPS`. geoopt's
/// float64 projection margin: closer to the boundary the conformal factor
/// `2 / (1 - |x|^2)` passes 1e5 and Möbius addition between near-antipodal
/// points cancels below f64 resolution, so a geodesic could not even return
/// its own endpoint.
pub const BALL_EPS: f64 = 1e-5;
/// Bandwidth = this × median nearest-neighbour distance in the prototype's
/// verified `--dim 64 --seed 7` configuration.
pub const DEFAULT_BANDWIDTH_FACTOR: f64 = 1.5;
/// A specialist counts as blended into a candidate -- for the feature count
/// and the category-cross flag -- at this weight or more.
pub const BLEND_WEIGHT: f64 = 0.05;
/// Weights below this are not listed in a proposal's `between`.
const LIST_WEIGHT: f64 = 1e-3;
/// Nearest knowns listed per proposal.
const NEAREST_LISTED: usize = 3;

// ------------------------------------------------------------ vectors ---

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn scaled(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    dist2(a, b).sqrt()
}

/// numpy's median: the mean of the two middle values for an even count.
fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.is_empty() {
        None
    } else if sorted.len() % 2 == 1 {
        sorted.get(mid).copied()
    } else {
        Some((*sorted.get(mid.checked_sub(1)?)? + *sorted.get(mid)?) / 2.0)
    }
}

/// Validate a set of points: non-empty, one dimension, finite. Returns `d`.
fn validate_points(points: &[Vec<f64>], what: &str) -> Result<usize> {
    let first = points
        .first()
        .ok_or_else(|| anyhow::anyhow!("{what}: no vectors"))?;
    let dim = first.len();
    ensure!(dim > 0, "{what}: the vectors are empty (dimension 0)");
    for (i, p) in points.iter().enumerate() {
        ensure!(
            p.len() == dim,
            "{what}: vector {i} has dimension {}, vector 0 has {dim}",
            p.len()
        );
        ensure!(
            p.iter().all(|x| x.is_finite()),
            "{what}: vector {i} has a non-finite component"
        );
    }
    Ok(dim)
}

fn check_point(p: &[f64], dim: usize) -> Result<()> {
    ensure!(
        p.len() == dim,
        "a point of dimension {} cannot be scored against knowns of dimension {dim}",
        p.len()
    );
    ensure!(
        p.iter().all(|x| x.is_finite()),
        "a point with a non-finite component cannot be scored"
    );
    Ok(())
}

fn centroid(points: &[Vec<f64>]) -> Vec<f64> {
    let dim = points.first().map_or(0, Vec::len);
    let mut c = vec![0.0; dim];
    for p in points {
        for (ci, x) in c.iter_mut().zip(p) {
            *ci += x;
        }
    }
    let n = points.len().max(1) as f64;
    for ci in &mut c {
        *ci /= n;
    }
    c
}

/// `Σ w_i v_i` over the listed `(index, weight)` pairs.
fn combination(known: &[Vec<f64>], weights: &[(usize, f64)]) -> Result<Vec<f64>> {
    let dim = known.first().map_or(0, Vec::len);
    let mut out = vec![0.0; dim];
    for &(i, w) in weights {
        let v = known
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("weight on known {i}, but there are {}", known.len()))?;
        for (o, x) in out.iter_mut().zip(v) {
            *o += w * x;
        }
    }
    Ok(out)
}

/// One standard normal draw (Box-Muller; `u1` in `(0, 1]` so `ln` is finite).
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = 1.0 - rng.random::<f64>();
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn gaussian_vector(rng: &mut StdRng, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| standard_normal(rng)).collect()
}

/// Every unordered pair `(i, j)`, `i < j`, in the prototype's order.
fn pairs(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).flat_map(move |i| (i + 1..n).map(move |j| (i, j)))
}

// ------------------------------------------------------ the affine hull ---

/// One Householder reflector `H = I - 2 v vᵀ` acting on rows `offset..`.
#[derive(Debug, Clone)]
struct Reflector {
    offset: usize,
    /// Unit vector of length `dim - offset`.
    v: Vec<f64>,
}

impl Reflector {
    fn apply(&self, x: &mut [f64]) {
        if let Some(tail) = x.get_mut(self.offset..) {
            let s = 2.0 * dot(&self.v, tail);
            for (xi, vi) in tail.iter_mut().zip(&self.v) {
                *xi -= s * vi;
            }
        }
    }
}

/// Whether coherence can discriminate on a set of knowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifoldCheck {
    pub ambient_dim: usize,
    pub known: usize,
    /// Numerical rank of the hull's direction space.
    pub affine_rank: usize,
    /// `ambient_dim - affine_rank`: the dimensions a point can leave the
    /// hull along. Zero means every point reconstructs exactly.
    pub codimension: usize,
}

/// The affine hull of the knowns, factored once by Householder QR with column
/// pivoting so any number of points can be projected onto it.
///
/// See the module docs for why this factorisation. The reflectors are kept,
/// not `Q`: projecting a point costs `O(rank · d)`.
#[derive(Debug, Clone)]
pub struct AffineHull {
    base: Vec<f64>,
    reflectors: Vec<Reflector>,
    dim: usize,
    known: usize,
}

impl AffineHull {
    /// Factor the hull of `known` (at least one vector, one dimension,
    /// finite).
    pub fn fit(known: &[Vec<f64>]) -> Result<Self> {
        let dim = validate_points(known, "affine hull")?;
        let base = known
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("affine hull: no vectors"))?;
        let mut cols: Vec<Vec<f64>> = known.iter().skip(1).map(|v| sub(v, &base)).collect();
        let ncols = cols.len();
        let steps = dim.min(ncols);
        let mut reflectors = Vec::with_capacity(steps);
        let mut tol = 0.0;
        for k in 0..steps {
            // Pivot: the remaining column with the largest norm below row k.
            let mut best: Option<(usize, f64)> = None;
            for (j, col) in cols.iter().enumerate().skip(k) {
                let n = norm(col.get(k..).unwrap_or(&[]));
                if best.is_none_or(|(_, b)| n > b) {
                    best = Some((j, n));
                }
            }
            let Some((pivot, pnorm)) = best else { break };
            if k == 0 {
                // numpy lstsq's cutoff: eps * max(M, N) * largest singular
                // value; the first pivot norm is |R_11|, its QR estimate.
                tol = f64::EPSILON * dim.max(ncols) as f64 * pnorm;
            }
            if pnorm <= tol {
                break;
            }
            cols.swap(k, pivot);
            let mut v: Vec<f64> = cols
                .get(k)
                .and_then(|c| c.get(k..))
                .ok_or_else(|| anyhow::anyhow!("affine hull: pivot column {k} out of range"))?
                .to_vec();
            let x0 = *v
                .first()
                .ok_or_else(|| anyhow::anyhow!("affine hull: empty pivot column"))?;
            // H x = alpha e_1 with the sign that avoids cancellation.
            let alpha = if x0 >= 0.0 { -pnorm } else { pnorm };
            if let Some(v0) = v.first_mut() {
                *v0 -= alpha;
            }
            let vn = norm(&v);
            ensure!(
                vn > 0.0 && vn.is_finite(),
                "affine hull: degenerate reflector at step {k}"
            );
            for vi in &mut v {
                *vi /= vn;
            }
            let reflector = Reflector { offset: k, v };
            for col in cols.iter_mut().skip(k + 1) {
                reflector.apply(col);
            }
            reflectors.push(reflector);
        }
        Ok(Self {
            base,
            reflectors,
            dim,
            known: known.len(),
        })
    }

    /// Numerical rank of the hull's direction space.
    pub fn rank(&self) -> usize {
        self.reflectors.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn check(&self) -> ManifoldCheck {
        ManifoldCheck {
            ambient_dim: self.dim,
            known: self.known,
            affine_rank: self.rank(),
            codimension: self.dim - self.rank(),
        }
    }

    /// The precondition: refuse when the hull fills the ambient space, so
    /// that no score is returned that could not have been anything but 1.
    pub fn require_discriminating(&self) -> Result<ManifoldCheck> {
        let check = self.check();
        ensure!(
            check.codimension > 0,
            "coherence cannot discriminate here: the {} known vectors' affine hull has \
             rank {} in a {}-dimensional space, so every point reconstructs exactly and \
             every coherence would be 1. The between-space method needs the knowns on a \
             lower-dimensional manifold inside the ambient space (n - 1 < d at the very \
             least): embed them in a higher-dimensional space, or rank fewer at once",
            check.known,
            check.affine_rank,
            check.ambient_dim
        );
        Ok(check)
    }

    /// `Qᵀ (p - base)`.
    fn rotated(&self, p: &[f64]) -> Result<Vec<f64>> {
        check_point(p, self.dim)?;
        let mut b = sub(p, &self.base);
        for r in &self.reflectors {
            r.apply(&mut b);
        }
        Ok(b)
    }

    /// The part of a rotated vector outside the hull's directions, rotated
    /// back.
    fn unrotate_complement(&self, mut b: Vec<f64>) -> Vec<f64> {
        for x in b.iter_mut().take(self.rank()) {
            *x = 0.0;
        }
        for r in self.reflectors.iter().rev() {
            r.apply(&mut b);
        }
        b
    }

    /// `‖p - affine_recon(p)‖`.
    pub fn residual(&self, p: &[f64]) -> Result<f64> {
        let b = self.rotated(p)?;
        Ok(norm(b.get(self.rank()..).unwrap_or(&[])))
    }

    /// `p - affine_recon(p)`: the component of `p` orthogonal to the hull.
    pub fn complement(&self, p: &[f64]) -> Result<Vec<f64>> {
        let b = self.rotated(p)?;
        Ok(self.unrotate_complement(b))
    }

    /// The closest point of the affine hull, `Σ λ_i v_i` with `Σ λ_i = 1`.
    pub fn reconstruct(&self, p: &[f64]) -> Result<Vec<f64>> {
        let c = self.complement(p)?;
        Ok(sub(p, &c))
    }

    /// A *direction* with its components along the hull removed (no base
    /// point involved): what the null-space controls are built from.
    pub fn project_out(&self, direction: &[f64]) -> Result<Vec<f64>> {
        check_point(direction, self.dim)?;
        let mut b = direction.to_vec();
        for r in &self.reflectors {
            r.apply(&mut b);
        }
        Ok(self.unrotate_complement(b))
    }
}

/// `‖p - affine_recon(p; V)‖`, factoring the hull for this one call.
pub fn affine_residual(p: &[f64], known: &[Vec<f64>]) -> Result<f64> {
    AffineHull::fit(known)?.residual(p)
}

/// `exp(-residual / max(scale, 1e-9))`.
pub fn coherence_from_residual(residual: f64, scale: f64) -> f64 {
    (-residual / scale.max(SCALE_FLOOR)).exp()
}

/// `coherence(p; V)` for a stated scale.
pub fn coherence(p: &[f64], known: &[Vec<f64>], scale: f64) -> Result<f64> {
    Ok(coherence_from_residual(affine_residual(p, known)?, scale))
}

/// Gaussian KDE of the knowns at `p` (unnormalised, as the prototype).
pub fn kde_likelihood(p: &[f64], known: &[Vec<f64>], bandwidth: f64) -> Result<f64> {
    ensure!(
        bandwidth > 0.0 && bandwidth.is_finite(),
        "KDE bandwidth must be positive and finite, got {bandwidth}"
    );
    let dim = validate_points(known, "KDE")?;
    check_point(p, dim)?;
    let two_h2 = 2.0 * bandwidth * bandwidth;
    let total: f64 = known.iter().map(|v| (-dist2(v, p) / two_h2).exp()).sum();
    Ok(total / known.len() as f64)
}

/// The two length scales of a known set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scales {
    /// Each known's distance to its nearest other known.
    pub nearest: Vec<f64>,
    /// Median nearest-neighbour distance: the coherence scale.
    pub scale: f64,
    /// `factor × scale`: the KDE bandwidth.
    pub bandwidth: f64,
}

/// Nearest-neighbour scales, as the prototype derives them: the *median
/// nearest-neighbour* distance, not the median of all pairs (which between-
/// cluster gaps dominate, flattening the density).
pub fn scales(known: &[Vec<f64>], bandwidth_factor: f64) -> Result<Scales> {
    validate_points(known, "scales")?;
    ensure!(
        known.len() >= 2,
        "nearest-neighbour scales need at least two known vectors, got {}",
        known.len()
    );
    ensure!(
        bandwidth_factor > 0.0 && bandwidth_factor.is_finite(),
        "the bandwidth factor must be positive and finite, got {bandwidth_factor}"
    );
    let nearest: Vec<f64> = known
        .iter()
        .enumerate()
        .map(|(i, v)| {
            known
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, w)| dist(v, w))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let scale = median(&nearest).ok_or_else(|| anyhow::anyhow!("scales: no distances"))?;
    ensure!(
        scale > 0.0 && scale.is_finite(),
        "the median nearest-neighbour distance is {scale}: at least half of the known \
         vectors duplicate another, so there is no length scale to judge a between by"
    );
    Ok(Scales {
        nearest,
        scale,
        bandwidth: scale * bandwidth_factor,
    })
}

/// One point's scores.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PointScore {
    pub residual: f64,
    pub coherence: f64,
    pub likelihood_raw: f64,
    /// `likelihood_raw / (max over the scored set + 1e-12)`.
    pub likelihood: f64,
    /// `coherence · (1 - likelihood)`.
    pub gap: f64,
}

/// Score a set of points together: the prototype's `rank_alien` without the
/// sort. The likelihood is normalised over exactly this set, so what is in
/// the set matters (the finder always includes every known).
pub fn score_points(
    points: &[Vec<f64>],
    known: &[Vec<f64>],
    hull: &AffineHull,
    scales: &Scales,
) -> Result<Vec<PointScore>> {
    let mut raw = Vec::with_capacity(points.len());
    for p in points {
        let residual = hull.residual(p)?;
        raw.push((
            residual,
            coherence_from_residual(residual, scales.scale),
            kde_likelihood(p, known, scales.bandwidth)?,
        ));
    }
    let denom = raw.iter().map(|r| r.2).fold(f64::NEG_INFINITY, f64::max) + NORMALISER_EPS;
    Ok(raw
        .into_iter()
        .map(|(residual, coherence, likelihood_raw)| {
            let likelihood = likelihood_raw / denom;
            PointScore {
                residual,
                coherence,
                likelihood_raw,
                likelihood,
                gap: coherence * (1.0 - likelihood),
            }
        })
        .collect())
}

// --------------------------------------------------------- generators ---

/// How a candidate was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Generator {
    /// `(v_i + v_j) / 2`.
    Midpoint,
    /// `Σ λ_i v_i`, `λ ~ Dirichlet(1)`.
    SimplexBlend,
    /// Slerp midpoint of two directions from the knowns' centroid, radius
    /// interpolated.
    Slerp,
    /// Poincaré-ball geodesic midpoint in the exponential chart at the
    /// knowns' centroid.
    PoincareGeodesic,
}

impl Generator {
    pub const ALL: [Generator; 4] = [
        Generator::Midpoint,
        Generator::SimplexBlend,
        Generator::Slerp,
        Generator::PoincareGeodesic,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Generator::Midpoint => "midpoint",
            Generator::SimplexBlend => "simplex_blend",
            Generator::Slerp => "slerp",
            Generator::PoincareGeodesic => "poincare_geodesic",
        }
    }
}

impl std::fmt::Display for Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A generated point with the knowns it was made from.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub point: Vec<f64>,
    pub generator: Generator,
    /// `(known index, weight)`; for the pair generators `[(i, .5), (j, .5)]`
    /// (the geodesic parameter, not an affine weight, for the curved ones).
    pub weights: Vec<(usize, f64)>,
}

fn nearest_distance(p: &[f64], known: &[Vec<f64>]) -> f64 {
    known
        .iter()
        .map(|v| dist(p, v))
        .fold(f64::INFINITY, f64::min)
}

/// A bounded farthest-interior selection: keeps the `k` offered candidates
/// farthest from their nearest known. Ties keep the earlier offer.
struct Emptiest {
    k: usize,
    kept: Vec<(f64, Candidate)>,
}

impl Emptiest {
    fn new(k: usize) -> Self {
        Self {
            k,
            kept: Vec::with_capacity(k + 1),
        }
    }

    fn offer(&mut self, known: &[Vec<f64>], candidate: Candidate) {
        let distance = nearest_distance(&candidate.point, known);
        let pos = self
            .kept
            .iter()
            .position(|(d, _)| *d < distance)
            .unwrap_or(self.kept.len());
        if pos < self.k {
            self.kept.insert(pos, (distance, candidate));
            self.kept.truncate(self.k);
        }
    }

    fn into_candidates(self) -> Vec<Candidate> {
        self.kept.into_iter().map(|(_, c)| c).collect()
    }
}

fn midpoint(known: &[Vec<f64>], i: usize, j: usize) -> Result<Candidate> {
    let (a, b) = (
        known
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("no known {i}"))?,
        known
            .get(j)
            .ok_or_else(|| anyhow::anyhow!("no known {j}"))?,
    );
    Ok(Candidate {
        point: a.iter().zip(b).map(|(x, y)| (x + y) / 2.0).collect(),
        generator: Generator::Midpoint,
        weights: vec![(i, 0.5), (j, 0.5)],
    })
}

/// Every pairwise midpoint -- the simplest "between two ideas".
pub fn midpoints(known: &[Vec<f64>]) -> Result<Vec<Candidate>> {
    pairs(known.len())
        .map(|(i, j)| midpoint(known, i, j))
        .collect()
}

fn simplex_blend(known: &[Vec<f64>], rng: &mut StdRng) -> Result<Candidate> {
    // Dirichlet(1, ..., 1) = normalised Exp(1) draws.
    let raw: Vec<f64> = known
        .iter()
        .map(|_| -(1.0 - rng.random::<f64>()).ln())
        .collect();
    let total: f64 = raw.iter().sum();
    let uniform = 1.0 / known.len().max(1) as f64;
    let weights: Vec<(usize, f64)> = raw
        .iter()
        .enumerate()
        .map(|(i, r)| (i, if total > 0.0 { r / total } else { uniform }))
        .collect();
    Ok(Candidate {
        point: combination(known, &weights)?,
        generator: Generator::SimplexBlend,
        weights,
    })
}

/// `count` interior blends `Σ λ_i v_i`, `λ ~ Dirichlet(1)`: several known
/// ideas at once.
pub fn simplex_blends(
    known: &[Vec<f64>],
    count: usize,
    rng: &mut StdRng,
) -> Result<Vec<Candidate>> {
    (0..count).map(|_| simplex_blend(known, rng)).collect()
}

/// The `k` pool candidates farthest from their nearest known (Voronoi
/// emptiness: the most under-proposed spots).
pub fn farthest_interior(known: &[Vec<f64>], pool: Vec<Candidate>, k: usize) -> Vec<Candidate> {
    let mut keep = Emptiest::new(k);
    for c in pool {
        keep.offer(known, c);
    }
    keep.into_candidates()
}

/// Spherical interpolation of two *directions* on the unit sphere (an exact
/// port of the prototype's `slerp`, refusing the cases it would divide by
/// zero in).
pub fn slerp(a: &[f64], b: &[f64], t: f64) -> Result<Vec<f64>> {
    ensure!(
        a.len() == b.len(),
        "slerp: dimensions {} and {}",
        a.len(),
        b.len()
    );
    let (na, nb) = (norm(a), norm(b));
    ensure!(
        na > 0.0 && nb > 0.0 && na.is_finite() && nb.is_finite(),
        "slerp needs two non-zero finite vectors"
    );
    let (a1, b1) = (scaled(a, 1.0 / na), scaled(b, 1.0 / nb));
    let omega = dot(&a1, &b1).clamp(-1.0, 1.0).acos();
    if omega < SLERP_COLLINEAR {
        return Ok(a1);
    }
    ensure!(
        std::f64::consts::PI - omega >= SLERP_COLLINEAR,
        "slerp between antipodal directions has no unique path"
    );
    let s = omega.sin();
    let (ca, cb) = (((1.0 - t) * omega).sin() / s, (t * omega).sin() / s);
    Ok(a1.iter().zip(&b1).map(|(x, y)| ca * x + cb * y).collect())
}

/// Slerp midpoint of `v_i - c` and `v_j - c` with the radius interpolated;
/// `None` when the pair has no unique slerp (a known at the centroid, or
/// antipodal directions).
fn slerp_midpoint(known: &[Vec<f64>], c: &[f64], i: usize, j: usize) -> Result<Option<Candidate>> {
    let a = sub(
        known
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("no known {i}"))?,
        c,
    );
    let b = sub(
        known
            .get(j)
            .ok_or_else(|| anyhow::anyhow!("no known {j}"))?,
        c,
    );
    let (na, nb) = (norm(&a), norm(&b));
    if na <= 0.0 || nb <= 0.0 {
        return Ok(None);
    }
    let cosine = (dot(&a, &b) / (na * nb)).clamp(-1.0, 1.0);
    if std::f64::consts::PI - cosine.acos() < SLERP_COLLINEAR {
        return Ok(None);
    }
    let direction = slerp(&a, &b, 0.5)?;
    let radius = 0.5 * (na + nb);
    Ok(Some(Candidate {
        point: c
            .iter()
            .zip(&direction)
            .map(|(ci, d)| ci + radius * d)
            .collect(),
        generator: Generator::Slerp,
        weights: vec![(i, 0.5), (j, 0.5)],
    }))
}

/// Every pair's slerp midpoint (pairs with no unique slerp are skipped).
pub fn slerp_midpoints(known: &[Vec<f64>]) -> Result<Vec<Candidate>> {
    let c = centroid(known);
    let mut out = Vec::new();
    for (i, j) in pairs(known.len()) {
        if let Some(cand) = slerp_midpoint(known, &c, i, j)? {
            out.push(cand);
        }
    }
    Ok(out)
}

// ------------------------------------------------------ Poincaré ball ---

fn clamp_to_ball(x: Vec<f64>) -> Vec<f64> {
    let n = norm(&x);
    let max = 1.0 - BALL_EPS;
    if n > max {
        scaled(&x, max / n)
    } else {
        x
    }
}

/// `artanh` kept finite. Clamped at one ULP below 1, not at the ball's clamp
/// radius: the Möbius *difference* of two points inside that radius can lie
/// far closer to the boundary than either, and shrinking it would stop the
/// geodesic short of its endpoint.
fn artanh_clamped(x: f64) -> f64 {
    x.clamp(0.0, 1.0 - f64::EPSILON).atanh()
}

/// A point strictly inside the ball, moved onto the clamp radius if it lies
/// between it and the boundary. A point on or outside the boundary has no
/// hyperbolic meaning and is refused rather than clamped.
fn into_ball(x: &[f64], what: &str) -> Result<Vec<f64>> {
    let n = norm(x);
    ensure!(
        n.is_finite() && n < 1.0,
        "{what} has norm {n}: a Poincaré-ball point must lie strictly inside the unit ball"
    );
    Ok(clamp_to_ball(x.to_vec()))
}

/// Möbius addition `x ⊕ y` in the Poincaré ball.
pub fn mobius_add(x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
    ensure!(
        x.len() == y.len(),
        "mobius_add: dimensions {} and {}",
        x.len(),
        y.len()
    );
    let (x2, y2, xy) = (dot(x, x), dot(y, y), dot(x, y));
    // 1 + 2<x,y> + |x|^2 |y|^2, written by Lagrange's identity as
    // (1 + <x,y>)^2 + (|x|^2 |y|^2 - <x,y>^2): a sum of two non-negative
    // terms, so rounding cannot drive it negative near the boundary.
    let den = (1.0 + xy) * (1.0 + xy) + (x2 * y2 - xy * xy).max(0.0);
    ensure!(
        den > 0.0 && den.is_finite(),
        "mobius_add: denominator {den} (are both points inside the ball?)"
    );
    let (cx, cy) = (1.0 + 2.0 * xy + y2, 1.0 - x2);
    Ok(x.iter()
        .zip(y)
        .map(|(a, b)| (cx * a + cy * b) / den)
        .collect())
}

/// Möbius scalar multiplication `r ⊗ x = tanh(r · artanh‖x‖) x / ‖x‖`.
pub fn mobius_scale(r: f64, x: &[f64]) -> Vec<f64> {
    let n = norm(x);
    if n <= 0.0 {
        return vec![0.0; x.len()];
    }
    scaled(x, (r * artanh_clamped(n)).tanh() / n)
}

/// Hyperbolic distance `2 artanh ‖(-x) ⊕ y‖`.
pub fn poincare_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    let neg: Vec<f64> = x.iter().map(|v| -v).collect();
    Ok(2.0 * artanh_clamped(norm(&mobius_add(&neg, y)?)))
}

/// The geodesic `γ(t) = a ⊕ (t ⊗ ((-a) ⊕ b))`: `γ(0) = a`, `γ(1) = b`, and
/// `d(a, γ(t)) = t · d(a, b)`. Inputs must be strictly inside the ball (a
/// point on or outside it is refused, not clamped); an input between the
/// clamp radius `1 - BALL_EPS` and the boundary is moved onto that radius,
/// and the output is clamped to it against rounding.
pub fn poincare_geodesic(a: &[f64], b: &[f64], t: f64) -> Result<Vec<f64>> {
    let a = into_ball(a, "the geodesic's start")?;
    let b = into_ball(b, "the geodesic's end")?;
    ensure!(t.is_finite(), "geodesic parameter {t} is not finite");
    let neg: Vec<f64> = a.iter().map(|v| -v).collect();
    let step = mobius_scale(t, &mobius_add(&neg, &b)?);
    Ok(clamp_to_ball(mobius_add(&a, &step)?))
}

/// The exponential chart at the knowns' centroid: `x ↦ tanh(‖x‖/s) x/‖x‖`
/// on `x = v - c`, with `s` the largest known's distance from `c`.
struct PoincareChart {
    centre: Vec<f64>,
    radius: f64,
}

impl PoincareChart {
    fn new(known: &[Vec<f64>]) -> Result<Self> {
        let centre = centroid(known);
        let radius = known.iter().map(|v| dist(v, &centre)).fold(0.0, f64::max);
        ensure!(
            radius > 0.0 && radius.is_finite(),
            "the known vectors all coincide; there is no chart to draw geodesics in"
        );
        Ok(Self { centre, radius })
    }

    fn enter_ball(&self, v: &[f64]) -> Vec<f64> {
        let x = sub(v, &self.centre);
        let n = norm(&x);
        if n <= 0.0 {
            return x;
        }
        clamp_to_ball(scaled(&x, (n / self.radius).tanh() / n))
    }

    fn leave_ball(&self, y: &[f64]) -> Vec<f64> {
        let n = norm(y);
        if n <= 0.0 {
            return self.centre.clone();
        }
        let s = self.radius * artanh_clamped(n) / n;
        self.centre
            .iter()
            .zip(y)
            .map(|(c, yi)| c + s * yi)
            .collect()
    }

    fn midpoint(&self, known: &[Vec<f64>], i: usize, j: usize) -> Result<Candidate> {
        let a = self.enter_ball(
            known
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("no known {i}"))?,
        );
        let b = self.enter_ball(
            known
                .get(j)
                .ok_or_else(|| anyhow::anyhow!("no known {j}"))?,
        );
        Ok(Candidate {
            point: self.leave_ball(&poincare_geodesic(&a, &b, 0.5)?),
            generator: Generator::PoincareGeodesic,
            weights: vec![(i, 0.5), (j, 0.5)],
        })
    }
}

/// Every pair's hyperbolic geodesic midpoint, through the exponential chart
/// at the knowns' centroid (a tree-like field's between bends toward the
/// common root).
pub fn poincare_midpoints(known: &[Vec<f64>]) -> Result<Vec<Candidate>> {
    let chart = PoincareChart::new(known)?;
    pairs(known.len())
        .map(|(i, j)| chart.midpoint(known, i, j))
        .collect()
}

/// CONTROL points that must be rejected: the centroid pushed a typical
/// known's distance along a random direction orthogonal to the hull. Novel,
/// but not reconstructable.
pub fn nullspace_controls(
    known: &[Vec<f64>],
    hull: &AffineHull,
    count: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f64>>> {
    let c = centroid(known);
    let reach = known.iter().map(|v| dist(v, &c)).sum::<f64>() / known.len().max(1) as f64;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let u = hull.project_out(&gaussian_vector(rng, hull.dim()))?;
        let n = norm(&u);
        if n > NULL_DIRECTION_FLOOR {
            out.push(
                c.iter()
                    .zip(&u)
                    .map(|(ci, ui)| ci + ui / n * reach)
                    .collect(),
            );
        }
    }
    Ok(out)
}

// ------------------------------------------------- specialist vectors ---

/// One known vector: a specialist (or a category centroid).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecialistVector {
    pub name: String,
    /// Category label (a taxonomy node); needed for `--per-category`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    pub vector: Vec<f64>,
    /// Per-dimension variance of the window means it was averaged from;
    /// what diagonal whitening pools. Absent for hand-written vectors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_var: Option<Vec<f64>>,
    /// Where the vector came from, in words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// The on-disk vector file: `{"vectors": [...]}` (a bare array also reads).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorFile {
    pub vectors: Vec<SpecialistVector>,
}

/// Parse either file shape. Through a `Value` rather than an untagged enum:
/// this crate's `serde_json` keeps number lexemes (`arbitrary_precision`),
/// and serde's untagged buffering cannot hand those to an `f64`.
fn parse_vector_file(path: &Path) -> Result<Vec<SpecialistVector>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    // Through the `Value`, the f64s are parsed correctly rounded (see
    // `from_json_exact`).
    let parsed = if value.is_array() {
        serde_json::from_value::<Vec<SpecialistVector>>(value)
    } else {
        serde_json::from_value::<VectorFile>(value).map(|file| file.vectors)
    };
    parsed.with_context(|| {
        format!(
            "parse {}: expected {{\"vectors\": [{{\"name\", \"vector\", \"category\"?}}, ...]}} \
             or a bare array of those",
            path.display()
        )
    })
}

/// Read a vector file and validate it.
pub fn read_vectors(path: &Path) -> Result<Vec<SpecialistVector>> {
    let vectors = parse_vector_file(path)?;
    validate_vectors(&vectors).with_context(|| format!("validate {}", path.display()))?;
    Ok(vectors)
}

/// Write a vector file atomically.
pub fn write_vectors(path: &Path, vectors: &[SpecialistVector]) -> Result<()> {
    if !vectors.is_empty() {
        validate_vectors_allow_one(vectors)?;
    }
    let file = VectorFile {
        vectors: vectors.to_vec(),
    };
    let mut bytes = serde_json::to_vec_pretty(&file).context("serialize vector file")?;
    bytes.push(b'\n');
    crate::interop::write_atomic(path, &bytes)
}

/// Insert `vector` into the file at `path` (created if absent), replacing an
/// entry of the same name. Returns whether an entry was replaced.
pub fn upsert_vector(path: &Path, vector: SpecialistVector) -> Result<bool> {
    let mut vectors = if path
        .try_exists()
        .with_context(|| format!("stat {}", path.display()))?
    {
        parse_vector_file(path)?
    } else {
        Vec::new()
    };
    let replaced = match vectors.iter_mut().find(|v| v.name == vector.name) {
        Some(slot) => {
            *slot = vector;
            true
        }
        None => {
            vectors.push(vector);
            false
        }
    };
    write_vectors(path, &vectors)?;
    Ok(replaced)
}

fn validate_vectors_allow_one(vectors: &[SpecialistVector]) -> Result<usize> {
    let mut names = HashSet::new();
    let first = vectors
        .first()
        .ok_or_else(|| anyhow::anyhow!("no specialist vectors"))?;
    let dim = first.vector.len();
    ensure!(dim > 0, "specialist '{}' has an empty vector", first.name);
    for v in vectors {
        ensure!(
            !v.name.trim().is_empty(),
            "a specialist vector has an empty name"
        );
        ensure!(
            names.insert(v.name.as_str()),
            "specialist '{}' appears twice",
            v.name
        );
        ensure!(
            v.vector.len() == dim,
            "specialist '{}' has dimension {}, '{}' has {dim}",
            v.name,
            v.vector.len(),
            first.name
        );
        ensure!(
            v.vector.iter().all(|x| x.is_finite()),
            "specialist '{}' has a non-finite component",
            v.name
        );
        if let Some(var) = &v.window_var {
            ensure!(
                var.len() == dim && var.iter().all(|x| x.is_finite() && *x >= 0.0),
                "specialist '{}' has a window_var that is not {dim} finite non-negative numbers",
                v.name
            );
        }
        if let Some(category) = &v.category {
            ensure!(
                !category.trim().is_empty(),
                "specialist '{}' has an empty category",
                v.name
            );
        }
    }
    Ok(dim)
}

/// At least two vectors, unique non-empty names, one dimension, finite.
/// Returns the dimension.
pub fn validate_vectors(vectors: &[SpecialistVector]) -> Result<usize> {
    ensure!(
        vectors.len() >= 2,
        "{} specialist vector(s): a between needs at least two",
        vectors.len()
    );
    validate_vectors_allow_one(vectors)
}

/// Diagonal whitening by the *pooled within-specialist* window variance:
/// dimension `k` is divided by `sqrt(pooled_k + 1e-3 · mean(pooled))`.
///
/// One transform for every specialist (whitening each by its own variance
/// would put them in different coordinates). Diagonal, because a full
/// covariance is not estimable from fewer windows than hidden dimensions;
/// the `1e-3` shrinkage toward isotropic keeps a dimension that never varied
/// within any specialist from dominating every distance.
pub fn whiten_diagonal(vectors: &[SpecialistVector]) -> Result<Vec<SpecialistVector>> {
    let dim = validate_vectors(vectors)?;
    let mut pooled = vec![0.0; dim];
    for v in vectors {
        let var = v.window_var.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "specialist '{}' carries no window_var (embed it with `dblocks alien embed` \
                 over at least two windows) -- whitening needs every specialist's",
                v.name
            )
        })?;
        for (p, x) in pooled.iter_mut().zip(var) {
            *p += x;
        }
    }
    let n = vectors.len() as f64;
    for p in &mut pooled {
        *p /= n;
    }
    let shrink = 1e-3 * pooled.iter().sum::<f64>() / dim as f64;
    ensure!(
        shrink > 0.0 && shrink.is_finite(),
        "the pooled window variance is zero in every dimension: nothing to whiten by"
    );
    let inv_sd: Vec<f64> = pooled.iter().map(|p| 1.0 / (p + shrink).sqrt()).collect();
    Ok(vectors
        .iter()
        .map(|v| SpecialistVector {
            vector: v.vector.iter().zip(&inv_sd).map(|(x, s)| x * s).collect(),
            ..v.clone()
        })
        .collect())
}

/// Group by category (every vector must carry one).
fn by_category(vectors: &[SpecialistVector]) -> Result<BTreeMap<String, Vec<SpecialistVector>>> {
    let mut groups: BTreeMap<String, Vec<SpecialistVector>> = BTreeMap::new();
    for v in vectors {
        let category = v.category.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "specialist '{}' has no category; per-category finding needs one on every \
                 vector (a \"category\" key in the vector file)",
                v.name
            )
        })?;
        groups.entry(category).or_default().push(v.clone());
    }
    Ok(groups)
}

/// One centroid per category, named by the category: the knowns of the
/// top-level (cross-domain) finder.
pub fn category_centroids(vectors: &[SpecialistVector]) -> Result<Vec<SpecialistVector>> {
    validate_vectors(vectors)?;
    Ok(by_category(vectors)?
        .into_iter()
        .map(|(category, members)| {
            let points: Vec<Vec<f64>> = members.iter().map(|m| m.vector.clone()).collect();
            SpecialistVector {
                name: category.clone(),
                category: Some(category),
                vector: centroid(&points),
                window_var: None,
                source: Some(format!("centroid of {} specialists", members.len())),
            }
        })
        .collect())
}

// ------------------------------------------------------------ proposals ---

/// One specialist's share of a candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Share {
    pub specialist: String,
    pub weight: f64,
}

/// A nearby known and its distance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Nearest {
    pub specialist: String,
    pub distance: f64,
}

/// A ranked candidate: a point between specialists, with the scores that
/// ranked it. A proposal is a *candidate to judge*, not a finding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    /// `alien-` + 12 hex of a hash of scope, generator and composition:
    /// stable across runs with the same vectors and seed.
    pub id: String,
    /// `all`, `category:<name>` or `top-level`.
    #[serde(default)]
    pub scope: String,
    /// The knowns it was built from, largest weight first.
    pub between: Vec<Share>,
    pub coherence: f64,
    /// Normalised as in the prototype (over the scored set).
    pub likelihood: f64,
    pub gap: f64,
    pub generator: Generator,
    pub nearest_knowns: Vec<Nearest>,
    /// Distance to the nearest known over the scope's coherence scale.
    #[serde(default)]
    pub nearest_distance_scaled: f64,
    /// Whether the blended specialists (weight >= `BLEND_WEIGHT`) span more
    /// than one category; always true at the top level.
    #[serde(default)]
    pub category_cross: bool,
    #[serde(default)]
    pub residual: f64,
    #[serde(default)]
    pub likelihood_raw: f64,
    /// Predicted pilot survival, when a predictor ranked this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub survival: Option<f64>,
    /// What the ranking sorted by: `gap`, or `gap × survival`.
    #[serde(default)]
    pub rank_score: f64,
    /// The between-point itself, for a realizer to decode.
    #[serde(default)]
    pub point: Vec<f64>,
}

fn proposal_id(scope: &str, generator: Generator, between: &[Share]) -> String {
    let mut h = Sha256::new();
    h.update(scope.as_bytes());
    h.update(b"|");
    h.update(generator.as_str().as_bytes());
    for s in between {
        h.update(format!("|{}:{:.6}", s.specialist, s.weight).as_bytes());
    }
    let digest = h.finalize();
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    format!("alien-{hex}")
}

fn scope_seed(seed: u64, scope: &str) -> u64 {
    let digest = Sha256::digest(scope.as_bytes());
    let bytes: [u8; 8] = digest
        .get(..8)
        .and_then(|s| s.try_into().ok())
        .unwrap_or([0; 8]);
    seed ^ u64::from_le_bytes(bytes)
}

fn sort_proposals(proposals: &mut [Proposal]) {
    proposals.sort_by(|a, b| {
        b.rank_score
            .total_cmp(&a.rank_score)
            .then(
                b.nearest_distance_scaled
                    .total_cmp(&a.nearest_distance_scaled),
            )
            .then_with(|| a.id.cmp(&b.id))
    });
}

// -------------------------------------------------------------- finder ---

/// How a finder generates and scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinderConfig {
    /// Dirichlet blends drawn into the Euclidean pool (prototype: 2000).
    pub simplex_samples: usize,
    /// Candidates kept per pool by farthest-interior selection (prototype: 12).
    pub interior: usize,
    /// Bandwidth = factor × median nearest-neighbour distance.
    pub bandwidth_factor: f64,
    /// Null-space control points (prototype: 3).
    pub null_controls: usize,
    /// Also generate slerp and Poincaré-geodesic betweens.
    pub curved: bool,
    pub seed: u64,
}

impl Default for FinderConfig {
    fn default() -> Self {
        Self {
            simplex_samples: 2000,
            interior: 12,
            bandwidth_factor: DEFAULT_BANDWIDTH_FACTOR,
            null_controls: 3,
            curved: true,
            seed: 7,
        }
    }
}

/// What went into a scope's candidate pools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolStats {
    pub simplex: usize,
    pub midpoints: usize,
    pub slerp: usize,
    /// Pairs with no unique slerp (a known at the centroid, or antipodal).
    pub slerp_skipped: usize,
    pub poincare: usize,
    /// Kept per pool by farthest-interior selection.
    pub kept_per_pool: usize,
}

/// The controls' scores: what must lose.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlSummary {
    pub on_known: usize,
    pub on_known_mean_gap: f64,
    pub on_known_max_gap: f64,
    /// The known with the largest gap (an isolated specialist, usually).
    pub on_known_worst: String,
    pub null_space: usize,
    pub null_mean_coherence: f64,
    pub null_max_coherence: f64,
}

/// One of the prototype's discrimination self-checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfCheck {
    pub name: String,
    pub passed: bool,
}

/// One finder's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopeReport {
    pub scope: String,
    pub knowns: Vec<String>,
    pub manifold: ManifoldCheck,
    pub scale: f64,
    pub bandwidth: f64,
    /// How many points the likelihood was measured against.
    #[serde(default)]
    pub density_points: usize,
    pub pools: PoolStats,
    /// Sorted best first.
    pub proposals: Vec<Proposal>,
    pub controls: ControlSummary,
    pub self_checks: Vec<SelfCheck>,
}

impl ScopeReport {
    /// Whether every one of the prototype's discrimination checks held. When
    /// one fails, this scope's scores do not separate betweens from controls
    /// and its ranking should not be read as one -- typical for knowns with
    /// no cluster structure for a nearest-neighbour bandwidth to resolve.
    pub fn discriminates(&self) -> bool {
        self.self_checks.iter().all(|c| c.passed)
    }
}

fn spans_categories(weights: &[(usize, f64)], categories: &[Option<String>]) -> bool {
    let mut seen: HashSet<&str> = HashSet::new();
    for &(i, w) in weights {
        if w >= BLEND_WEIGHT {
            if let Some(Some(c)) = categories.get(i) {
                seen.insert(c.as_str());
            }
        }
    }
    seen.len() > 1
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Run one ALIEN-FINDER over `members`. `cross_scope` marks every proposal
/// as category-crossing (the top-level finder, whose knowns are categories).
///
/// `density` is what the likelihood (and the scale and bandwidth) is
/// measured against -- the knowns themselves when `None`. The top-level
/// finder passes every member specialist: coherence asks whether a point
/// recombines the category centroids, likelihood asks whether any actual
/// specialist sits there. Three centroids alone would put the bandwidth at
/// 1.5x the distance between categories, and every between would look
/// obvious.
pub fn find_in_scope(
    scope: &str,
    members: &[SpecialistVector],
    config: &FinderConfig,
    cross_scope: bool,
    density: Option<&[Vec<f64>]>,
) -> Result<ScopeReport> {
    ensure!(
        config.interior > 0,
        "the finder must keep at least one candidate per pool"
    );
    let names: Vec<String> = members.iter().map(|m| m.name.clone()).collect();
    let categories: Vec<Option<String>> = members.iter().map(|m| m.category.clone()).collect();
    let known: Vec<Vec<f64>> = members.iter().map(|m| m.vector.clone()).collect();
    ensure!(
        known.len() >= 2,
        "scope {scope}: {} known vector(s); a between needs at least two",
        known.len()
    );
    let hull = AffineHull::fit(&known).with_context(|| format!("scope {scope}"))?;
    let manifold = hull
        .require_discriminating()
        .with_context(|| format!("scope {scope}"))?;
    let density: Vec<Vec<f64>> = match density {
        Some(points) => {
            let dim = validate_points(points, "density points")?;
            ensure!(
                dim == hull.dim(),
                "scope {scope}: density points have dimension {dim}, the knowns {}",
                hull.dim()
            );
            points.to_vec()
        }
        None => known.clone(),
    };
    let scales =
        scales(&density, config.bandwidth_factor).with_context(|| format!("scope {scope}"))?;
    let mut rng = StdRng::seed_from_u64(scope_seed(config.seed, scope));
    let n = known.len();
    let mut pools = PoolStats {
        simplex: config.simplex_samples,
        midpoints: n * (n - 1) / 2,
        kept_per_pool: config.interior,
        ..PoolStats::default()
    };

    // The prototype's pool: simplex interior + pairwise midpoints, then the
    // emptiest (most under-proposed) coherent spots.
    let mut euclid = Emptiest::new(config.interior);
    for _ in 0..config.simplex_samples {
        euclid.offer(&known, simplex_blend(&known, &mut rng)?);
    }
    for (i, j) in pairs(n) {
        euclid.offer(&known, midpoint(&known, i, j)?);
    }
    let mut candidates = euclid.into_candidates();
    if config.curved {
        let c = centroid(&known);
        let mut sl = Emptiest::new(config.interior);
        for (i, j) in pairs(n) {
            match slerp_midpoint(&known, &c, i, j)? {
                Some(cand) => {
                    pools.slerp += 1;
                    sl.offer(&known, cand);
                }
                None => pools.slerp_skipped += 1,
            }
        }
        candidates.extend(sl.into_candidates());
        let chart = PoincareChart::new(&known)?;
        let mut pc = Emptiest::new(config.interior);
        for (i, j) in pairs(n) {
            pools.poincare += 1;
            pc.offer(&known, chart.midpoint(&known, i, j)?);
        }
        candidates.extend(pc.into_candidates());
    }
    let nulls = nullspace_controls(&known, &hull, config.null_controls, &mut rng)?;

    // One scored set -- candidates, every known, the null controls -- so the
    // likelihood is normalised over all of it, as the prototype does.
    let mut scored: Vec<Vec<f64>> = candidates.iter().map(|c| c.point.clone()).collect();
    scored.extend(known.iter().cloned());
    scored.extend(nulls.iter().cloned());
    let scores = score_points(&scored, &density, &hull, &scales)?;
    let (cand_scores, rest) = scores
        .split_at_checked(candidates.len())
        .ok_or_else(|| anyhow::anyhow!("scope {scope}: fewer scores than candidates"))?;
    let (known_scores, null_scores) = rest
        .split_at_checked(n)
        .ok_or_else(|| anyhow::anyhow!("scope {scope}: fewer scores than knowns"))?;

    let mut proposals = Vec::with_capacity(candidates.len());
    let mut ids = HashSet::new();
    for (cand, s) in candidates.iter().zip(cand_scores) {
        let mut between: Vec<Share> = cand
            .weights
            .iter()
            .filter(|(_, w)| w.abs() >= LIST_WEIGHT)
            .filter_map(|&(i, w)| {
                names.get(i).map(|name| Share {
                    specialist: name.clone(),
                    weight: w,
                })
            })
            .collect();
        between.sort_by(|a, b| {
            b.weight
                .total_cmp(&a.weight)
                .then_with(|| a.specialist.cmp(&b.specialist))
        });
        let mut near: Vec<Nearest> = known
            .iter()
            .zip(&names)
            .map(|(v, name)| Nearest {
                specialist: name.clone(),
                distance: dist(&cand.point, v),
            })
            .collect();
        near.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        near.truncate(NEAREST_LISTED);
        let nearest_distance_scaled = near.first().map_or(0.0, |x| x.distance) / scales.scale;
        let mut id = proposal_id(scope, cand.generator, &between);
        let mut suffix = 1usize;
        while !ids.insert(id.clone()) {
            suffix += 1;
            id = format!("{}-{suffix}", proposal_id(scope, cand.generator, &between));
        }
        proposals.push(Proposal {
            id,
            scope: scope.to_string(),
            between,
            coherence: s.coherence,
            likelihood: s.likelihood,
            gap: s.gap,
            generator: cand.generator,
            nearest_knowns: near,
            nearest_distance_scaled,
            category_cross: cross_scope || spans_categories(&cand.weights, &categories),
            residual: s.residual,
            likelihood_raw: s.likelihood_raw,
            survival: None,
            rank_score: s.gap,
            point: cand.point.clone(),
        });
    }
    sort_proposals(&mut proposals);

    let known_gaps: Vec<f64> = known_scores.iter().map(|s| s.gap).collect();
    let worst = known_gaps
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .and_then(|(i, _)| names.get(i).cloned())
        .unwrap_or_default();
    let null_coh: Vec<f64> = null_scores.iter().map(|s| s.coherence).collect();
    let controls = ControlSummary {
        on_known: known_scores.len(),
        on_known_mean_gap: mean(&known_gaps),
        on_known_max_gap: max_of(&known_gaps),
        on_known_worst: worst,
        null_space: null_scores.len(),
        null_mean_coherence: mean(&null_coh),
        null_max_coherence: if null_coh.is_empty() {
            0.0
        } else {
            max_of(&null_coh)
        },
    };

    // The prototype's four self-checks.
    let cand_gaps: Vec<f64> = cand_scores.iter().map(|s| s.gap).collect();
    let cand_coh: Vec<f64> = cand_scores.iter().map(|s| s.coherence).collect();
    let control_best = known_scores
        .iter()
        .chain(null_scores)
        .map(|s| s.gap)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut self_checks = vec![SelfCheck {
        name: "between-points beat on-known points on alien-gap".into(),
        passed: mean(&cand_gaps) > controls.on_known_mean_gap,
    }];
    if config.null_controls > 0 {
        self_checks.push(SelfCheck {
            name: "null-space points are incoherent (mean coherence < 0.5)".into(),
            passed: !null_coh.is_empty() && controls.null_mean_coherence < 0.5,
        });
    }
    self_checks.push(SelfCheck {
        name: "between-points are coherent (mean coherence > 0.6)".into(),
        passed: mean(&cand_coh) > 0.6,
    });
    self_checks.push(SelfCheck {
        name: "top-ranked point is a between, not a control".into(),
        passed: max_of(&cand_gaps) > control_best,
    });

    Ok(ScopeReport {
        scope: scope.to_string(),
        knowns: names,
        manifold,
        scale: scales.scale,
        bandwidth: scales.bandwidth,
        density_points: density.len(),
        pools,
        proposals,
        controls,
        self_checks,
    })
}

/// What `rank` does.
#[derive(Debug, Clone, Default)]
pub struct RankOptions {
    /// One finder per category plus a top-level finder between category
    /// centroids, instead of one finder over everything.
    pub per_category: bool,
    /// Whiten by the pooled within-specialist window variance first.
    pub whiten: bool,
    pub finder: FinderConfig,
}

/// A scope that was not ranked, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skipped {
    pub scope: String,
    pub reason: String,
}

/// What a ranking sorted by.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RankingMode {
    /// No survival predictor: the alien-gap alone.
    GapOnly,
    /// `gap × predicted survival`, from a predictor fitted on `outcomes`.
    GapTimesSurvival {
        outcomes: usize,
        holdout: usize,
        holdout_accuracy: Option<f64>,
    },
}

impl RankingMode {
    pub fn describe(&self) -> String {
        match self {
            RankingMode::GapOnly => "alien-gap alone -- no survival predictor was given, so this \
                                     orders candidates by coherence x (1 - likelihood) and \
                                     predicts nothing about which would survive a pilot"
                .to_string(),
            RankingMode::GapTimesSurvival {
                outcomes,
                holdout,
                holdout_accuracy,
            } => {
                let accuracy = match holdout_accuracy {
                    Some(a) => format!("held-out accuracy {a:.3} on {holdout}"),
                    None => format!(
                        "held-out accuracy NOT reported ({holdout} held-out outcomes is too few)"
                    ),
                };
                format!(
                    "alien-gap x predicted survival -- logistic predictor fitted on {outcomes} \
                     recorded pilot outcomes; {accuracy}"
                )
            }
        }
    }
}

/// Every finder's output for one `rank` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankReport {
    pub scopes: Vec<ScopeReport>,
    pub skipped: Vec<Skipped>,
    pub ranking: RankingMode,
    pub whitened: bool,
}

/// Rank between-space candidates. Flat mode refuses outright when its one
/// scope cannot be scored; per-category mode skips (and reports) a scope it
/// cannot score and ranks the rest.
pub fn rank(
    vectors: &[SpecialistVector],
    options: &RankOptions,
    predictor: Option<&SurvivalModel>,
) -> Result<RankReport> {
    validate_vectors(vectors)?;
    let vectors = if options.whiten {
        whiten_diagonal(vectors)?
    } else {
        vectors.to_vec()
    };
    let mut scopes = Vec::new();
    let mut skipped = Vec::new();
    if options.per_category {
        for (category, members) in by_category(&vectors)? {
            let scope = format!("category:{category}");
            if members.len() < 2 {
                skipped.push(Skipped {
                    scope,
                    reason: format!(
                        "only {} specialist; a between needs at least two",
                        members.len()
                    ),
                });
                continue;
            }
            match find_in_scope(&scope, &members, &options.finder, false, None) {
                Ok(report) => scopes.push(report),
                Err(err) => skipped.push(Skipped {
                    scope,
                    reason: format!("{err:#}"),
                }),
            }
        }
        let centroids = category_centroids(&vectors)?;
        if centroids.len() < 2 {
            skipped.push(Skipped {
                scope: "top-level".into(),
                reason: format!(
                    "{} category; the top-level finder needs at least two",
                    centroids.len()
                ),
            });
        } else {
            let specialists: Vec<Vec<f64>> = vectors.iter().map(|v| v.vector.clone()).collect();
            match find_in_scope(
                "top-level",
                &centroids,
                &options.finder,
                true,
                Some(&specialists),
            ) {
                Ok(report) => scopes.push(report),
                Err(err) => skipped.push(Skipped {
                    scope: "top-level".into(),
                    reason: format!("{err:#}"),
                }),
            }
        }
    } else {
        scopes.push(find_in_scope(
            "all",
            &vectors,
            &options.finder,
            false,
            None,
        )?);
    }
    let ranking = match predictor {
        None => RankingMode::GapOnly,
        Some(model) => {
            for scope in &mut scopes {
                for p in &mut scope.proposals {
                    let s = model.predict(p)?;
                    p.survival = Some(s);
                    p.rank_score = p.gap * s;
                }
                sort_proposals(&mut scope.proposals);
            }
            RankingMode::GapTimesSurvival {
                outcomes: model.report.train,
                holdout: model.report.holdout,
                holdout_accuracy: model.report.holdout_accuracy,
            }
        }
    };
    Ok(RankReport {
        scopes,
        skipped,
        ranking,
        whitened: options.whiten,
    })
}

fn describe_between(between: &[Share]) -> String {
    let shown: Vec<String> = between
        .iter()
        .take(3)
        .map(|s| format!("{} {:.2}", s.specialist, s.weight))
        .collect();
    let mut text = shown.join(" + ");
    if between.len() > 3 {
        text.push_str(&format!(" (+{} more)", between.len() - 3));
    }
    text
}

impl RankReport {
    /// Every proposal, scope by scope.
    pub fn proposals(&self) -> impl Iterator<Item = &Proposal> {
        self.scopes.iter().flat_map(|s| s.proposals.iter())
    }

    /// The human-readable report: the top `top` proposals per scope, the
    /// controls and self-checks, and what was skipped.
    pub fn render(&self, top: usize) -> String {
        let mut out = String::new();
        out.push_str(
            "ALIEN-FINDER -- between-space candidates. This ranks candidates; it does not \
             establish that any of them works.\n",
        );
        out.push_str(&format!("ranking: {}\n", self.ranking.describe()));
        if self.whitened {
            out.push_str(
                "vectors: diagonally whitened by the pooled within-specialist window variance\n",
            );
        }
        let with_survival = matches!(self.ranking, RankingMode::GapTimesSurvival { .. });
        for s in &self.scopes {
            out.push_str(&format!(
                "\n== {}  ({} knowns in R^{}; affine rank {}, codimension {})\n",
                s.scope,
                s.manifold.known,
                s.manifold.ambient_dim,
                s.manifold.affine_rank,
                s.manifold.codimension
            ));
            out.push_str(&format!(
                "scale {:.4} (median nearest-neighbour distance), bandwidth {:.4}, likelihood \
                 over {} points{}\n",
                s.scale,
                s.bandwidth,
                s.density_points,
                if s.density_points == s.manifold.known {
                    " (the knowns)"
                } else {
                    " (the member specialists)"
                }
            ));
            let p = &s.pools;
            let mut pools = format!(
                "candidates: {} farthest-interior of {} simplex blends + {} midpoints",
                p.kept_per_pool.min(p.simplex + p.midpoints),
                p.simplex,
                p.midpoints
            );
            if p.poincare > 0 || p.slerp > 0 {
                pools.push_str(&format!(
                    "; {} farthest of {} slerp midpoints ({} pairs skipped); {} farthest of {} \
                     Poincare geodesic midpoints",
                    p.kept_per_pool.min(p.slerp),
                    p.slerp,
                    p.slerp_skipped,
                    p.kept_per_pool.min(p.poincare),
                    p.poincare
                ));
            }
            out.push_str(&pools);
            out.push('\n');
            out.push_str(&format!(
                "{:>4}  {:<17} {:>9} {:>10} {:>9}{}  {:<6} between  |  nearest\n",
                "rank",
                "generator",
                "coherence",
                "likelihood",
                "alien-gap",
                if with_survival { "  survival" } else { "" },
                "cross"
            ));
            for (r, prop) in s.proposals.iter().take(top).enumerate() {
                let survival = match (with_survival, prop.survival) {
                    (true, Some(v)) => format!("  {v:>8.3}"),
                    (true, None) => format!("  {:>8}", "-"),
                    (false, _) => String::new(),
                };
                let near: Vec<String> = prop
                    .nearest_knowns
                    .iter()
                    .map(|n| format!("{} ({:.2})", n.specialist, n.distance))
                    .collect();
                out.push_str(&format!(
                    "{:>4}  {:<17} {:>9.3} {:>10.3} {:>9.3}{}  {:<6} {}  |  {}\n",
                    r + 1,
                    prop.generator.as_str(),
                    prop.coherence,
                    prop.likelihood,
                    prop.gap,
                    survival,
                    if prop.category_cross { "yes" } else { "no" },
                    describe_between(&prop.between),
                    near.join(", ")
                ));
            }
            if s.proposals.len() > top {
                out.push_str(&format!("      ... {} more\n", s.proposals.len() - top));
            }
            let c = &s.controls;
            out.push_str(&format!(
                "controls: {} on-known: mean gap {:.3}, max {:.3} ({}) -- should be ~0 (obvious); \
                 {} null-space: mean coherence {:.3}, max {:.3} -- should be ~0 (incoherent)\n",
                c.on_known,
                c.on_known_mean_gap,
                c.on_known_max_gap,
                c.on_known_worst,
                c.null_space,
                c.null_mean_coherence,
                c.null_max_coherence
            ));
            out.push_str("self-checks:\n");
            for check in &s.self_checks {
                out.push_str(&format!(
                    "  [{}] {}\n",
                    if check.passed { "OK  " } else { "FAIL" },
                    check.name
                ));
            }
            if !s.discriminates() {
                out.push_str(
                    "VERDICT: the scores in this scope do NOT discriminate (a self-check failed), \
                     so its ranking is not a ranking -- typical when the knowns have no cluster \
                     structure for a nearest-neighbour bandwidth to resolve, or are too few\n",
                );
            }
        }
        for skip in &self.skipped {
            out.push_str(&format!("\nskipped {}: {}\n", skip.scope, skip.reason));
        }
        out
    }
}

// --------------------------------------------- plans, outcomes, survival ---

/// A falsifiable plan for one proposal (TRISKELE §14 step 4). Written by a
/// person or a teacher/NCM realizer -- this module only stores and checks
/// them, it never writes one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub proposal_id: String,
    pub mechanism: String,
    pub expected_effect: String,
    pub baseline: String,
    pub failure_mode: String,
    pub falsification_test: String,
}

impl Plan {
    /// Every field present and non-blank, naming the first one that is not.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("proposal_id", &self.proposal_id),
            ("mechanism", &self.mechanism),
            ("expected_effect", &self.expected_effect),
            ("baseline", &self.baseline),
            ("failure_mode", &self.failure_mode),
            ("falsification_test", &self.falsification_test),
        ] {
            ensure!(
                !value.trim().is_empty(),
                "plan for '{}' has an empty {field}: a plan without it is not falsifiable",
                self.proposal_id
            );
        }
        Ok(())
    }
}

/// Every plan valid and naming a known proposal.
pub fn check_plans(plans: &[Plan], proposals: &[Proposal]) -> Result<()> {
    let ids: HashSet<&str> = proposals.iter().map(|p| p.id.as_str()).collect();
    for plan in plans {
        plan.validate()?;
        ensure!(
            ids.contains(plan.proposal_id.as_str()),
            "plan names proposal '{}', which is not among the {} proposals given",
            plan.proposal_id,
            proposals.len()
        );
    }
    Ok(())
}

/// What happened when a proposal was piloted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PilotOutcome {
    pub proposal_id: String,
    pub survived: bool,
    #[serde(default)]
    pub notes: String,
}

/// Parse JSON through a `Value`. This crate's `serde_json` keeps each
/// number's lexeme (`arbitrary_precision`), and reading an `f64` from a
/// `Value` parses that lexeme with std's correctly rounded parser, where
/// `from_str` straight into an `f64` can land one ULP off -- so a proposal
/// or model written and read back would not be the one that was written.
fn from_json_exact<T: serde::de::DeserializeOwned>(text: &str) -> serde_json::Result<T> {
    serde_json::from_value(serde_json::from_str::<serde_json::Value>(text)?)
}

/// Read a JSONL file (blank lines skipped; errors name the line).
pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            from_json_exact(line).with_context(|| format!("{}:{}: parse", path.display(), i + 1))
        })
        .collect()
}

/// Write a JSONL file atomically.
pub fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> Result<()> {
    let mut out = Vec::new();
    for item in items {
        serde_json::to_writer(&mut out, item).context("serialize a JSONL record")?;
        out.push(b'\n');
    }
    crate::interop::write_atomic(path, &out)
}

/// The survival predictor's features, in order.
pub const FEATURES: [&str; 10] = [
    "gap",
    "coherence",
    "likelihood",
    "generator_midpoint",
    "generator_simplex_blend",
    "generator_slerp",
    "generator_poincare_geodesic",
    "nearest_distance_scaled",
    "blended_specialists",
    "category_cross",
];

/// A proposal's feature vector, in [`FEATURES`] order.
pub fn proposal_features(p: &Proposal) -> Vec<f64> {
    let mut x = vec![p.gap, p.coherence, p.likelihood];
    for g in Generator::ALL {
        x.push(if p.generator == g { 1.0 } else { 0.0 });
    }
    x.push(p.nearest_distance_scaled);
    x.push(
        p.between
            .iter()
            .filter(|s| s.weight >= BLEND_WEIGHT)
            .count() as f64,
    );
    x.push(if p.category_cross { 1.0 } else { 0.0 });
    x
}

/// The honesty thresholds of the survival predictor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurvivalConfig {
    /// Fewest training outcomes it will fit on.
    pub min_fit: usize,
    /// Fewest of *each* outcome (survived, failed) in training.
    pub min_per_class: usize,
    /// Fewest held-out outcomes it will report an accuracy for.
    pub min_report: usize,
    /// One proposal in this many is held out, chosen by a hash of its id (so
    /// adding outcomes never moves an old proposal between folds).
    pub holdout_every: u8,
    /// L2 penalty on the standardised weights (not the bias).
    pub ridge: f64,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            min_fit: 20,
            min_per_class: 5,
            min_report: 30,
            holdout_every: 5,
            ridge: 1.0,
        }
    }
}

/// What the predictor was fitted on and how well it held out.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SurvivalReport {
    pub outcomes: usize,
    /// Outcomes whose proposal id was not among the proposals.
    pub unmatched: usize,
    pub train: usize,
    pub train_survived: usize,
    pub holdout: usize,
    pub holdout_survived: usize,
    pub holdout_accuracy: Option<f64>,
    /// Wilson 95% interval of the held-out accuracy.
    pub holdout_accuracy_interval: Option<(f64, f64)>,
    /// Accuracy of always predicting the training majority, on the same
    /// held-out set: the number the predictor has to beat.
    pub majority_baseline_accuracy: Option<f64>,
    pub holdout_log_loss: Option<f64>,
    /// Why no accuracy is reported, when it is not.
    pub refusal: Option<String>,
    pub iterations: usize,
    pub converged: bool,
}

/// A fitted logistic survival predictor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurvivalModel {
    pub features: Vec<String>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub config: SurvivalConfig,
    pub report: SurvivalReport,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn in_holdout(id: &str, every: u8) -> bool {
    Sha256::digest(id.as_bytes())
        .first()
        .is_some_and(|b| b % every == 0)
}

/// Solve `A x = b` for symmetric positive-definite `A` (Cholesky).
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    ensure!(
        a.len() == n,
        "cholesky: {}x? matrix against {n} right-hand sides",
        a.len()
    );
    let mut l: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, row) in a.iter().enumerate() {
        ensure!(
            row.len() == n,
            "cholesky: row {i} has {} entries, not {n}",
            row.len()
        );
        let mut li: Vec<f64> = Vec::with_capacity(i + 1);
        for (j, &aij) in row.iter().enumerate().take(i + 1) {
            if j == i {
                let s: f64 = li.iter().map(|x| x * x).sum();
                let d = aij - s;
                ensure!(
                    d > 0.0 && d.is_finite(),
                    "cholesky: the system is not positive definite at pivot {i}"
                );
                li.push(d.sqrt());
            } else {
                let lj = l
                    .get(j)
                    .ok_or_else(|| anyhow::anyhow!("cholesky: row {j}"))?;
                let s: f64 = li.iter().zip(lj).take(j).map(|(x, y)| x * y).sum();
                let ljj = *lj
                    .get(j)
                    .ok_or_else(|| anyhow::anyhow!("cholesky: pivot {j}"))?;
                li.push((aij - s) / ljj);
            }
        }
        l.push(li);
    }
    // L y = b
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for (li, bi) in l.iter().zip(b) {
        let s: f64 = li.iter().zip(&y).map(|(x, yk)| x * yk).sum();
        let d = *li
            .last()
            .ok_or_else(|| anyhow::anyhow!("cholesky: empty row"))?;
        y.push((bi - s) / d);
    }
    // Lᵀ x = y, from the bottom up.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = l
            .iter()
            .zip(&x)
            .skip(i + 1)
            .map(|(lk, xk)| lk.get(i).map_or(0.0, |v| v * xk))
            .sum();
        let lii = l
            .get(i)
            .and_then(|r| r.get(i))
            .copied()
            .ok_or_else(|| anyhow::anyhow!("cholesky: pivot {i}"))?;
        let yi = *y.get(i).ok_or_else(|| anyhow::anyhow!("cholesky: y {i}"))?;
        if let Some(xi) = x.get_mut(i) {
            *xi = (yi - s) / lii;
        }
    }
    Ok(x)
}

/// Ridge logistic regression by Newton/IRLS on standardised features.
/// Returns `(weights, bias, iterations, converged)`.
fn fit_logistic(xs: &[Vec<f64>], ys: &[f64], ridge: f64) -> Result<(Vec<f64>, f64, usize, bool)> {
    const MAX_ITERS: usize = 100;
    let f = xs.first().map_or(0, Vec::len);
    let width = f + 1; // bias last
    let mut w = vec![0.0; width];
    let mut iterations = 0;
    let mut converged = false;
    while iterations < MAX_ITERS {
        iterations += 1;
        let mut grad = vec![0.0; width];
        let mut hess = vec![vec![0.0; width]; width];
        for (x, y) in xs.iter().zip(ys) {
            let mut aug = x.clone();
            aug.push(1.0);
            let p = sigmoid(dot(&w, &aug));
            let wt = (p * (1.0 - p)).max(1e-12);
            for ((g, row), ak) in grad.iter_mut().zip(hess.iter_mut()).zip(&aug) {
                *g += (y - p) * ak;
                for (h, al) in row.iter_mut().zip(&aug) {
                    *h += wt * ak * al;
                }
            }
        }
        for (k, ((g, row), wk)) in grad.iter_mut().zip(hess.iter_mut()).zip(&w).enumerate() {
            let penalty = if k < f { ridge } else { 1e-9 };
            *g -= if k < f { ridge * wk } else { 0.0 };
            if let Some(d) = row.get_mut(k) {
                *d += penalty;
            }
        }
        let step = cholesky_solve(&hess, &grad)?;
        let largest = step.iter().fold(0.0f64, |m, s| m.max(s.abs()));
        for (wk, s) in w.iter_mut().zip(&step) {
            *wk += s;
        }
        if largest < 1e-10 {
            converged = true;
            break;
        }
    }
    ensure!(
        w.iter().all(|x| x.is_finite()),
        "the logistic fit diverged (non-finite weights)"
    );
    let bias = w.pop().unwrap_or(0.0);
    Ok((w, bias, iterations, converged))
}

/// Wilson 95% interval for `k` successes in `n`.
fn wilson(k: usize, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let (k, n) = (k as f64, n as f64);
    let z = 1.959_963_984_540_054;
    let p = k / n;
    let den = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / den;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / den;
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// Fit the survival predictor on recorded outcomes, refusing when they are
/// too few to fit on, and withholding the accuracy when the held-out fold is
/// too small to say anything.
pub fn train_survival(
    proposals: &[Proposal],
    outcomes: &[PilotOutcome],
    config: &SurvivalConfig,
) -> Result<SurvivalModel> {
    ensure!(
        config.holdout_every >= 2,
        "holdout_every must be at least 2"
    );
    ensure!(
        config.ridge > 0.0 && config.ridge.is_finite(),
        "the ridge penalty must be positive and finite"
    );
    let mut by_id: BTreeMap<&str, &Proposal> = BTreeMap::new();
    for p in proposals {
        if let Some(previous) = by_id.insert(p.id.as_str(), p) {
            ensure!(
                proposal_features(previous) == proposal_features(p),
                "proposal id '{}' appears twice with different scores; ids must name one \
                 candidate",
                p.id
            );
        }
    }
    let mut train: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut holdout: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut unmatched = 0;
    for o in outcomes {
        let Some(p) = by_id.get(o.proposal_id.as_str()) else {
            unmatched += 1;
            continue;
        };
        let sample = (proposal_features(p), if o.survived { 1.0 } else { 0.0 });
        if in_holdout(&o.proposal_id, config.holdout_every) {
            holdout.push(sample);
        } else {
            train.push(sample);
        }
    }
    let train_survived = train.iter().filter(|s| s.1 > 0.5).count();
    let train_failed = train.len() - train_survived;
    ensure!(
        train.len() >= config.min_fit
            && train_survived >= config.min_per_class
            && train_failed >= config.min_per_class,
        "refusing to fit a survival predictor: {} training outcomes ({} survived, {} failed; \
         {} outcomes total, {} held out, {} with no matching proposal). It needs at least {} \
         with at least {} of each -- record more pilot outcomes first",
        train.len(),
        train_survived,
        train_failed,
        outcomes.len(),
        holdout.len(),
        unmatched,
        config.min_fit,
        config.min_per_class
    );

    // Standardise on the training fold only.
    let f = FEATURES.len();
    let m = train.len() as f64;
    let mut mean = vec![0.0; f];
    for (x, _) in &train {
        for (mk, xk) in mean.iter_mut().zip(x) {
            *mk += xk / m;
        }
    }
    let mut var = vec![0.0; f];
    for (x, _) in &train {
        for ((vk, xk), mk) in var.iter_mut().zip(x).zip(&mean) {
            *vk += (xk - mk) * (xk - mk) / m;
        }
    }
    let std: Vec<f64> = var.iter().map(|v| v.sqrt()).collect();
    let standardise = |x: &[f64]| -> Vec<f64> {
        x.iter()
            .zip(&mean)
            .zip(&std)
            .map(|((xk, mk), sk)| if *sk > 1e-12 { (xk - mk) / sk } else { 0.0 })
            .collect()
    };
    let xs: Vec<Vec<f64>> = train.iter().map(|(x, _)| standardise(x)).collect();
    let ys: Vec<f64> = train.iter().map(|(_, y)| *y).collect();
    let (weights, bias, iterations, converged) = fit_logistic(&xs, &ys, config.ridge)?;

    let holdout_survived = holdout.iter().filter(|s| s.1 > 0.5).count();
    let mut report = SurvivalReport {
        outcomes: outcomes.len(),
        unmatched,
        train: train.len(),
        train_survived,
        holdout: holdout.len(),
        holdout_survived,
        iterations,
        converged,
        ..SurvivalReport::default()
    };
    if holdout.len() >= config.min_report {
        let mut correct = 0usize;
        let mut log_loss = 0.0;
        for (x, y) in &holdout {
            let p = sigmoid(dot(&weights, &standardise(x)) + bias);
            if (p >= 0.5) == (*y > 0.5) {
                correct += 1;
            }
            let pc = p.clamp(1e-12, 1.0 - 1e-12);
            log_loss -= y * pc.ln() + (1.0 - y) * (1.0 - pc).ln();
        }
        let n = holdout.len();
        report.holdout_accuracy = Some(correct as f64 / n as f64);
        report.holdout_accuracy_interval = wilson(correct, n);
        report.holdout_log_loss = Some(log_loss / n as f64);
        let majority_survives = train_survived * 2 >= train.len();
        let majority_correct = if majority_survives {
            holdout_survived
        } else {
            n - holdout_survived
        };
        report.majority_baseline_accuracy = Some(majority_correct as f64 / n as f64);
    } else {
        report.refusal = Some(format!(
            "held-out accuracy not reported: {} held-out outcomes, fewer than the {} an \
             accuracy is worth reading at",
            holdout.len(),
            config.min_report
        ));
    }
    Ok(SurvivalModel {
        features: FEATURES.iter().map(|s| (*s).to_string()).collect(),
        mean,
        std,
        weights,
        bias,
        config: config.clone(),
        report,
    })
}

impl SurvivalModel {
    /// Predicted probability that `p` survives a pilot.
    pub fn predict(&self, p: &Proposal) -> Result<f64> {
        let x = proposal_features(p);
        ensure!(
            x.len() == self.weights.len()
                && self.mean.len() == x.len()
                && self.std.len() == x.len(),
            "survival model has {} weights for {} features",
            self.weights.len(),
            x.len()
        );
        let z: f64 = x
            .iter()
            .zip(&self.mean)
            .zip(&self.std)
            .zip(&self.weights)
            .map(|(((xk, mk), sk), wk)| {
                if *sk > 1e-12 {
                    wk * (xk - mk) / sk
                } else {
                    0.0
                }
            })
            .sum();
        Ok(sigmoid(z + self.bias))
    }

    /// Save as JSON, atomically.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(self).context("serialize survival model")?;
        bytes.push(b'\n');
        crate::interop::write_atomic(path, &bytes)
    }

    /// Load, refusing a model fitted on a different feature set.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let model: Self =
            from_json_exact(&text).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            model
                .features
                .iter()
                .map(String::as_str)
                .eq(FEATURES.iter().copied()),
            "{} was fitted on features {:?}, this build scores {:?}",
            path.display(),
            model.features,
            FEATURES
        );
        ensure!(
            model.weights.len() == FEATURES.len()
                && model.mean.len() == FEATURES.len()
                && model.std.len() == FEATURES.len(),
            "{} has the wrong number of coefficients",
            path.display()
        );
        Ok(model)
    }

    /// The report, with every count, in words.
    pub fn render(&self) -> String {
        let r = &self.report;
        let mut out = format!(
            "survival predictor (logistic, ridge {}) over {} outcomes: {} train ({} survived), \
             {} held out ({} survived), {} with no matching proposal; {} Newton steps{}\n",
            self.config.ridge,
            r.outcomes,
            r.train,
            r.train_survived,
            r.holdout,
            r.holdout_survived,
            r.unmatched,
            r.iterations,
            if r.converged { "" } else { " (NOT converged)" }
        );
        match (r.holdout_accuracy, &r.refusal) {
            (Some(acc), _) => {
                let (lo, hi) = r.holdout_accuracy_interval.unwrap_or((f64::NAN, f64::NAN));
                out.push_str(&format!(
                    "held-out accuracy {acc:.3} (95% Wilson [{lo:.3}, {hi:.3}]) vs \
                     majority baseline {:.3}; log loss {:.3}\n",
                    r.majority_baseline_accuracy.unwrap_or(f64::NAN),
                    r.holdout_log_loss.unwrap_or(f64::NAN)
                ));
            }
            (None, Some(reason)) => {
                out.push_str(reason);
                out.push('\n');
            }
            (None, None) => out.push_str("held-out accuracy not reported\n"),
        }
        out.push_str("weights (standardised features):\n");
        for (name, w) in self.features.iter().zip(&self.weights) {
            out.push_str(&format!("  {name:<28} {w:>+8.3}\n"));
        }
        out.push_str(&format!("  {:<28} {:>+8.3}\n", "bias", self.bias));
        out.push_str(
            "This predictor is only as good as the recorded pilot outcomes it was fitted on.\n",
        );
        out
    }
}

// ------------------------------------------------ specialists from a model ---

/// How [`embed_student`] samples the corpus.
#[derive(Debug, Clone)]
pub struct EmbedOptions {
    /// Windows averaged over, evenly spaced across the corpus.
    pub windows: usize,
    /// Tokens per window; the student's configured context when `None`
    /// (and never more than the corpus holds).
    pub window_len: Option<usize>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            windows: 32,
            window_len: None,
        }
    }
}

/// A specialist's known vector from a nina student checkpoint and a sample
/// of its field corpus: the mean of the student's hidden states (the
/// residual stream after the geometric write-back, through the public
/// [`crate::student::GeometricStudent::forward`]) over every position of
/// `windows` evenly spaced windows. The per-dimension variance of the window
/// means is kept as `window_var`, for [`whiten_diagonal`].
///
/// The student's config comes from the sidecar or deployment manifest beside
/// the checkpoint, the same lookup the nina scheduler uses.
pub fn embed_student(
    checkpoint: &Path,
    corpus: &Path,
    name: &str,
    category: Option<&str>,
    options: &EmbedOptions,
) -> Result<SpecialistVector> {
    use crate::corpus::TokenCorpus;
    use crate::student::GeometricStudent;
    use burn::tensor::{Int, Tensor};
    type Eval = burn_ndarray::NdArray<f32>;

    ensure!(!name.trim().is_empty(), "the specialist needs a name");
    ensure!(options.windows > 0, "embedding needs at least one window");
    let config = crate::nina::student_config_for_checkpoint(checkpoint)?;
    let device: <Eval as burn::tensor::backend::BackendTypes>::Device = Default::default();
    <Eval as burn::tensor::backend::Backend>::seed(&device, 0);
    let model = GeometricStudent::<Eval>::new(&config, &device)?;
    crate::tensor_ext::force_initialization(&model);
    let model = crate::checkpoint::load::<Eval, _>(model, checkpoint, &device)?;

    let mut tokens = TokenCorpus::streaming(corpus)?;
    let len = options
        .window_len
        .unwrap_or(config.context)
        .min(tokens.len());
    ensure!(
        len > 0,
        "{} holds no tokens to embed (or the window length is 0)",
        corpus.display()
    );
    let span = tokens.len() - len;
    let windows = options.windows;
    let starts: Vec<usize> = (0..windows)
        .map(|i| {
            if windows == 1 {
                0
            } else {
                // i * span / (windows - 1), without overflow.
                ((i as u128 * span as u128) / (windows as u128 - 1)) as usize
            }
        })
        .collect();

    // Welford over the window means, per dimension.
    let hidden = config.hidden_size;
    let mut mean = vec![0.0f64; hidden];
    let mut m2 = vec![0.0f64; hidden];
    let mut count = 0.0f64;
    for &start in &starts {
        let window = tokens.window(start, len)?;
        if let Some(bad) = window
            .iter()
            .find(|&&t| usize::from(t) >= config.vocab_size)
        {
            bail!(
                "{} holds token {bad} at or after offset {start}, outside the student's \
                 vocabulary of {}: is this corpus tokenized for this student?",
                corpus.display(),
                config.vocab_size
            );
        }
        let ids: Vec<i64> = window.iter().map(|&t| i64::from(t)).collect();
        let input = Tensor::<Eval, 1, Int>::from_ints(ids.as_slice(), &device).reshape([1, len]);
        let out = model.forward(input)?;
        let window_mean: Vec<f32> = out
            .hidden
            .mean_dim(1)
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        ensure!(
            window_mean.len() == hidden,
            "the student returned {} hidden values per position, its config says {hidden}",
            window_mean.len()
        );
        count += 1.0;
        for ((mk, sk), &x) in mean.iter_mut().zip(m2.iter_mut()).zip(&window_mean) {
            let x = f64::from(x);
            let delta = x - *mk;
            *mk += delta / count;
            *sk += delta * (x - *mk);
        }
    }
    ensure!(
        mean.iter().all(|x| x.is_finite()),
        "the student produced non-finite hidden states on {}",
        corpus.display()
    );
    let window_var =
        (starts.len() >= 2).then(|| m2.iter().map(|s| s / (count - 1.0)).collect::<Vec<f64>>());
    Ok(SpecialistVector {
        name: name.to_string(),
        category: category.map(str::to_string),
        vector: mean,
        window_var,
        source: Some(format!(
            "mean hidden state of student checkpoint {} over {} windows of {len} tokens of {}",
            checkpoint.display(),
            starts.len(),
            corpus.display()
        )),
    })
}

// ---------------------------------------------------------- certificates ---

/// A planted configuration: clusters on a `manifold_dim`-dimensional affine
/// subspace of `R^ambient_dim`, in a random orientation.
#[derive(Debug, Clone, Copy)]
struct Planted {
    clusters: usize,
    per_cluster: usize,
    manifold_dim: usize,
    ambient_dim: usize,
    spread: f64,
    noise: f64,
}

/// `(points, frame, labels)` of a planted configuration; the frame spans the
/// manifold's directions.
type PlantedSet = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>);
/// `(knowns, labels, farthest-interior betweens)` of the demo configuration.
type DemoSet = (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>);

/// Orthonormal vectors (Gram-Schmidt, twice) of `R^dim`, continuing `frame`.
fn extend_frame(
    rng: &mut StdRng,
    frame: &mut Vec<Vec<f64>>,
    count: usize,
    dim: usize,
) -> Result<()> {
    ensure!(
        frame.len() + count <= dim,
        "cannot fit {count} more orthonormal vectors in R^{dim}"
    );
    let target = frame.len() + count;
    let mut attempts = 0;
    while frame.len() < target {
        attempts += 1;
        ensure!(
            attempts <= 10 * target + 10,
            "Gram-Schmidt kept producing degenerate vectors"
        );
        let mut g = gaussian_vector(rng, dim);
        for _ in 0..2 {
            for q in frame.iter() {
                let c = dot(q, &g);
                for (gi, qi) in g.iter_mut().zip(q) {
                    *gi -= c * qi;
                }
            }
        }
        let n = norm(&g);
        if n > 1e-6 {
            frame.push(scaled(&g, 1.0 / n));
        }
    }
    Ok(())
}

impl Planted {
    /// `(points, frame, labels)`: the frame spans the manifold's directions.
    fn generate(&self, rng: &mut StdRng) -> Result<PlantedSet> {
        let mut frame = Vec::new();
        extend_frame(rng, &mut frame, self.manifold_dim, self.ambient_dim)?;
        let offset = gaussian_vector(rng, self.ambient_dim);
        let mut points = Vec::new();
        let mut labels = Vec::new();
        for cluster in 0..self.clusters {
            let centre: Vec<f64> = scaled(&gaussian_vector(rng, self.manifold_dim), self.spread);
            for _ in 0..self.per_cluster {
                let coords: Vec<f64> = centre
                    .iter()
                    .map(|c| c + self.noise * standard_normal(rng))
                    .collect();
                let mut p = offset.clone();
                for (ck, q) in coords.iter().zip(&frame) {
                    for (pi, qi) in p.iter_mut().zip(q) {
                        *pi += ck * qi;
                    }
                }
                points.push(p);
                labels.push(cluster);
            }
        }
        Ok((points, frame, labels))
    }
}

/// The operators a certificate checks, as parameters, so a test can hand in
/// a broken one and watch the certificate fail.
type Reconstruct = dyn Fn(&[Vec<f64>], &[f64]) -> Result<Vec<f64>>;
type CoherenceOf = dyn Fn(&[Vec<f64>], &[f64]) -> Result<f64>;
type Scorer = dyn Fn(&[Vec<f64>], &[Vec<f64>]) -> Result<Vec<PointScore>>;
type Geodesic = dyn Fn(&[f64], &[f64], f64) -> Result<Vec<f64>>;

fn hull_scorer(known: &[Vec<f64>], points: &[Vec<f64>]) -> Result<Vec<PointScore>> {
    let hull = AffineHull::fit(known)?;
    let s = scales(known, DEFAULT_BANDWIDTH_FACTOR)?;
    score_points(points, known, &hull, &s)
}

/// Max over affine combinations of `|recon(p) - p| / (1 + |p|)`, and over
/// orthogonal offsets of `|residual - offset|`.
fn hull_exactness_residual(reconstruct: &Reconstruct) -> Result<f64> {
    let mut rng = StdRng::seed_from_u64(0xA11E);
    // 8 knowns on a 5-dim manifold of R^24: rank-deficient (7 differences,
    // rank 5), which is the case pivoting exists for.
    let planted = Planted {
        clusters: 2,
        per_cluster: 4,
        manifold_dim: 5,
        ambient_dim: 24,
        spread: 4.0,
        noise: 0.5,
    };
    let (known, mut frame, _) = planted.generate(&mut rng)?;
    let mut worst: f64 = 0.0;
    for _ in 0..16 {
        // Affine weights, negative ones included: extrapolation is on the hull too.
        let mut weights: Vec<(usize, f64)> = (0..known.len())
            .map(|i| (i, standard_normal(&mut rng)))
            .collect();
        let others: f64 = weights.iter().skip(1).map(|(_, w)| w).sum();
        if let Some(first) = weights.first_mut() {
            first.1 = 1.0 - others;
        }
        let p = combination(&known, &weights)?;
        let r = reconstruct(&known, &p)?;
        worst = worst.max(dist(&r, &p) / (1.0 + norm(&p)));
    }
    // Control: a known orthogonal offset must come back as exactly that.
    extend_frame(&mut rng, &mut frame, 1, planted.ambient_dim)?;
    let w = frame
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no orthogonal direction"))?;
    let base = combination(&known, &[(0, 0.3), (5, 0.7)])?;
    for offset in [1.0, 0.37, 5.0] {
        let p: Vec<f64> = base.iter().zip(&w).map(|(b, wi)| b + offset * wi).collect();
        let r = reconstruct(&known, &p)?;
        worst = worst.max((dist(&r, &p) - offset).abs() / offset);
    }
    Ok(worst)
}

/// `max(coherence of complement points, 1 - coherence of hull points)`, plus
/// 1 if a full-rank configuration is not refused (or a just-discriminating
/// one is).
fn complement_coherence_residual(coherence_of: &CoherenceOf) -> Result<f64> {
    let mut rng = StdRng::seed_from_u64(0xC0DE);
    let planted = Planted {
        clusters: 2,
        per_cluster: 5,
        manifold_dim: 3,
        ambient_dim: 40,
        spread: 8.0,
        noise: 0.2,
    };
    let (known, mut frame, _) = planted.generate(&mut rng)?;
    let c = centroid(&known);
    let reach = known.iter().map(|v| dist(v, &c)).sum::<f64>() / known.len() as f64;
    let mut worst: f64 = 0.0;
    extend_frame(&mut rng, &mut frame, 4, planted.ambient_dim)?;
    for w in frame.iter().skip(planted.manifold_dim) {
        let p: Vec<f64> = c.iter().zip(w).map(|(ci, wi)| ci + reach * wi).collect();
        worst = worst.max(coherence_of(&known, &p)?);
    }
    // The generator's own null controls too.
    let hull = AffineHull::fit(&known)?;
    for p in nullspace_controls(&known, &hull, 4, &mut rng)? {
        worst = worst.max(coherence_of(&known, &p)?);
    }
    // Control: hull points are coherent (a scorer that is 0 everywhere fails).
    for (i, j) in [(0, 7), (2, 3), (4, 9)] {
        let p = combination(&known, &[(i, 0.4), (j, 0.6)])?;
        worst = worst.max(1.0 - coherence_of(&known, &p)?);
    }
    // The precondition: 9 points in general position in R^6 fill it.
    let full: Vec<Vec<f64>> = (0..9).map(|_| gaussian_vector(&mut rng, 6)).collect();
    if AffineHull::fit(&full)?.require_discriminating().is_ok() {
        worst += 1.0;
    }
    let tight: Vec<Vec<f64>> = (0..6).map(|_| gaussian_vector(&mut rng, 6)).collect();
    if AffineHull::fit(&tight)?.require_discriminating().is_err() {
        worst += 1.0;
    }
    Ok(worst)
}

/// The prototype's demo configuration (3 fields x 6 specialists in R^64),
/// drawn with this crate's RNG: knowns plus the farthest-interior betweens.
fn demo_configuration(rng: &mut StdRng) -> Result<DemoSet> {
    let planted = Planted {
        clusters: 3,
        per_cluster: 6,
        manifold_dim: 64,
        ambient_dim: 64,
        spread: 6.0,
        noise: 0.3,
    };
    let (known, _, labels) = planted.generate(rng)?;
    let mut pool = simplex_blends(&known, 2000, rng)?;
    pool.extend(midpoints(&known)?);
    let betweens = farthest_interior(&known, pool, 12)
        .into_iter()
        .map(|c| c.point)
        .collect();
    Ok((known, labels, betweens))
}

/// The worst known's gap against the best between's shortfall from 1.
fn known_obvious_residual(score: &Scorer) -> Result<f64> {
    let mut rng = StdRng::seed_from_u64(0x0B71);
    let (known, _, betweens) = demo_configuration(&mut rng)?;
    let n_between = betweens.len();
    let mut points = betweens;
    points.extend(known.iter().cloned());
    let scores = score(&known, &points)?;
    let (between_scores, known_scores) = scores
        .split_at_checked(n_between)
        .ok_or_else(|| anyhow::anyhow!("scorer returned too few scores"))?;
    ensure!(
        known_scores.len() == known.len(),
        "scorer returned the wrong number of scores"
    );
    let worst_known = known_scores
        .iter()
        .map(|s| s.gap)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_between = between_scores
        .iter()
        .map(|s| s.gap)
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(worst_known.max(1.0 - best_between))
}

/// The margin by which every between-cluster midpoint must outrank every
/// within-cluster midpoint.
const OUTRANK_MARGIN: f64 = 0.5;

/// `max(0, margin - (min between gap - max within gap))`, plus the shortfall
/// of the within-cluster points' coherence below 0.999.
fn between_within_residual(score: &Scorer) -> Result<f64> {
    let mut rng = StdRng::seed_from_u64(0xB7);
    let planted = Planted {
        clusters: 2,
        per_cluster: 5,
        manifold_dim: 3,
        ambient_dim: 48,
        spread: 8.0,
        noise: 0.25,
    };
    let (known, _, labels) = planted.generate(&mut rng)?;
    let mut within = Vec::new();
    let mut between = Vec::new();
    for (i, j) in pairs(known.len()) {
        let p = midpoint(&known, i, j)?.point;
        if labels.get(i) == labels.get(j) {
            within.push(p);
        } else {
            between.push(p);
        }
    }
    let (n_within, n_between) = (within.len(), between.len());
    let mut points = within;
    points.extend(between);
    points.extend(known.iter().cloned());
    let scores = score(&known, &points)?;
    let (w, rest) = scores
        .split_at_checked(n_within)
        .ok_or_else(|| anyhow::anyhow!("scorer returned too few scores"))?;
    let b = rest
        .get(..n_between)
        .ok_or_else(|| anyhow::anyhow!("scorer returned too few scores"))?;
    let max_within = w.iter().map(|s| s.gap).fold(f64::NEG_INFINITY, f64::max);
    let min_between = b.iter().map(|s| s.gap).fold(f64::INFINITY, f64::min);
    let min_within_coherence = w.iter().map(|s| s.coherence).fold(f64::INFINITY, f64::min);
    Ok((OUTRANK_MARGIN - (min_between - max_within)).max(0.0)
        + (0.999 - min_within_coherence).max(0.0))
}

/// Endpoints, in-ball, and the geodesic distance identities
/// `d(a,γ(t)) + d(γ(t),b) = d(a,b)` and `d(a,γ(t)) = t·d(a,b)`; plus 1 if a
/// point outside the ball is not refused.
fn geodesic_residual(geodesic: &Geodesic) -> Result<f64> {
    let mut rng = StdRng::seed_from_u64(0x9E0);
    let dim = 8;
    let at_norm = |rng: &mut StdRng, r: f64| -> Vec<f64> {
        let g = gaussian_vector(rng, dim);
        scaled(&g, r / norm(&g))
    };
    let mut ends = vec![vec![0.0; dim]];
    for r in [0.1, 0.5, 0.9, 0.999] {
        ends.push(at_norm(&mut rng, r));
        ends.push(at_norm(&mut rng, r));
    }
    let mut worst: f64 = 0.0;
    for (i, j) in pairs(ends.len()) {
        let (Some(a), Some(b)) = (ends.get(i), ends.get(j)) else {
            continue;
        };
        let d_ab = poincare_distance(a, b)?;
        let unit = d_ab.max(1.0);
        worst = worst.max(dist(&geodesic(a, b, 0.0)?, a));
        worst = worst.max(dist(&geodesic(a, b, 1.0)?, b));
        for t in [0.25, 0.5, 0.75] {
            let g = geodesic(a, b, t)?;
            let n = norm(&g);
            if !(n.is_finite() && n < 1.0) {
                worst = worst.max(1.0);
                continue;
            }
            let d_ag = poincare_distance(a, &g)?;
            let d_gb = poincare_distance(&g, b)?;
            worst = worst.max((d_ag + d_gb - d_ab).abs() / unit);
            worst = worst.max((d_ag - t * d_ab).abs() / unit);
        }
    }
    let outside = scaled(&at_norm(&mut rng, 1.0), 1.5);
    let inside = at_norm(&mut rng, 0.3);
    if geodesic(&outside, &inside, 0.5).is_ok() {
        worst += 1.0;
    }
    Ok(worst)
}

fn synthetic_proposal(i: usize, rng: &mut StdRng) -> (Proposal, bool) {
    let gap = rng.random::<f64>();
    let coherence = rng.random::<f64>();
    let generator = Generator::ALL
        .get(i % Generator::ALL.len())
        .copied()
        .unwrap_or(Generator::Midpoint);
    let survived = coherence + 0.3 * gap > 0.65;
    let p = Proposal {
        id: format!("synthetic-{i}"),
        scope: "synthetic".into(),
        between: vec![
            Share {
                specialist: "a".into(),
                weight: 0.5,
            },
            Share {
                specialist: "b".into(),
                weight: 0.5,
            },
        ],
        coherence,
        likelihood: rng.random::<f64>(),
        gap,
        generator,
        nearest_knowns: Vec::new(),
        nearest_distance_scaled: 1.0 + rng.random::<f64>(),
        category_cross: i.is_multiple_of(3),
        residual: 0.0,
        likelihood_raw: 0.0,
        survival: None,
        rank_score: gap,
        point: Vec::new(),
    };
    (p, survived)
}

/// Synthetic proposals whose survival follows a planted rule.
fn synthetic_outcomes(count: usize, seed: u64) -> (Vec<Proposal>, Vec<PilotOutcome>) {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let (p, survived) = synthetic_proposal(i, &mut rng);
            let o = PilotOutcome {
                proposal_id: p.id.clone(),
                survived,
                notes: String::new(),
            };
            (p, o)
        })
        .unzip()
}

/// 1 for each way the predictor fails to be honest about sample size: a
/// tiny sample fitted; a small one fitted with an accuracy reported; a large
/// planted-separable one with no accuracy, or one below 0.9.
fn survival_refusal_residual(config: &SurvivalConfig) -> Result<f64> {
    let mut worst = 0.0;
    let (p, o) = synthetic_outcomes(12, 1);
    if train_survival(&p, &o, config).is_ok() {
        worst += 1.0;
    }
    let (p, o) = synthetic_outcomes(60, 2);
    match train_survival(&p, &o, config) {
        Ok(model) => {
            if model.report.holdout_accuracy.is_some() || model.report.refusal.is_none() {
                worst += 1.0;
            }
        }
        Err(_) => worst += 1.0,
    }
    let (p, o) = synthetic_outcomes(400, 3);
    match train_survival(&p, &o, config) {
        Ok(model) => match model.report.holdout_accuracy {
            Some(acc) if acc >= 0.9 => {}
            _ => worst += 1.0,
        },
        Err(_) => worst += 1.0,
    }
    Ok(worst)
}

fn alien_cert(
    name: &'static str,
    theorem: &'static str,
    residual: f64,
    tolerance: f64,
) -> Certificate {
    Certificate {
        group: "alien",
        name,
        theorem,
        residual,
        tolerance,
    }
}

/// The `alien` certificate group (registered in [`crate::verify`]). Each
/// residual carries a control, so a scorer that is trivially right (constant,
/// identity, coherence-only) fails it.
pub fn certificate_checks() -> Result<Vec<Certificate>> {
    let reconstruct = |known: &[Vec<f64>], p: &[f64]| AffineHull::fit(known)?.reconstruct(p);
    let coherence_of = |known: &[Vec<f64>], p: &[f64]| -> Result<f64> {
        let s = scales(known, DEFAULT_BANDWIDTH_FACTOR)?;
        Ok(coherence_from_residual(
            AffineHull::fit(known)?.residual(p)?,
            s.scale,
        ))
    };
    let survival = survival_refusal_residual(&SurvivalConfig::default())?;
    Ok(vec![
        alien_cert(
            "affine_reconstruction_is_exact_on_the_hull",
            "A point sum_i l_i v_i with sum l_i = 1 (negative weights included) reconstructs to \
             itself, on a rank-deficient planted hull; a unit-, 0.37- and 5-length offset \
             orthogonal to the hull comes back as exactly that residual (control: an identity \
             reconstruction fails it).",
            hull_exactness_residual(&reconstruct)?,
            1e-9,
        ),
        alien_cert(
            "orthogonal_complement_has_zero_coherence",
            "Points pushed from the centroid along the orthogonal complement of the knowns' hull \
             (analytic and generated) score coherence below 1e-6 while hull points score 1; a \
             configuration whose hull fills the space is refused and one with codimension 1 is \
             not.",
            complement_coherence_residual(&coherence_of)?,
            1e-6,
        ),
        alien_cert(
            "known_points_score_as_obvious",
            "On the prototype's 3 x 6 clustered configuration in R^64, every known vector scores \
             alien-gap at most 0.15 (high likelihood: obvious) while the best farthest-interior \
             between scores at least 0.85 (control: a coherence-only scorer fails it).",
            known_obvious_residual(&hull_scorer)?,
            0.15,
        ),
        alien_cert(
            "between_cluster_points_outrank_within_cluster_points",
            "On two planted clusters on a 3-dim manifold in R^48, every between-cluster midpoint \
             outranks every within-cluster midpoint by at least 0.5 in alien-gap, and the \
             within-cluster midpoints are coherent (>= 0.999), so the order comes from the \
             likelihood term rather than from coherence.",
            between_within_residual(&hull_scorer)?,
            0.0,
        ),
        alien_cert(
            "poincare_geodesic_stays_in_the_ball_and_hits_its_endpoints",
            "For pairs in the 8-dim Poincare ball up to norm 0.999, gamma(0) = a, gamma(1) = b, \
             gamma(t) stays inside the ball, d(a,g) + d(g,b) = d(a,b) and d(a,g) = t d(a,b) \
             (a Euclidean chord fails these); a point outside the ball is refused.",
            geodesic_residual(&poincare_geodesic)?,
            1e-8,
        ),
        alien_cert(
            "survival_predictor_refuses_tiny_samples",
            "The survival predictor refuses to fit on 12 outcomes, fits 60 but withholds the \
             held-out accuracy (fewer than 30 held out), and on 400 planted-separable outcomes \
             reports a held-out accuracy of at least 0.9 -- the refusal is not unconditional.",
            survival,
            0.0,
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
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const FIXTURE: &str = include_str!("../tests/fixtures/alien_parity.json");

    fn scratch(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("alien-test-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn fixture() -> (Value, Vec<Vec<f64>>) {
        let fx: Value = serde_json::from_str(FIXTURE).unwrap();
        let v: Vec<Vec<f64>> = fx["V"].as_array().unwrap().iter().map(floats).collect();
        (fx, v)
    }

    fn fixture_specialists() -> Vec<SpecialistVector> {
        let (fx, v) = fixture();
        let labels: Vec<String> = fx["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_str().unwrap().to_string())
            .collect();
        v.into_iter()
            .zip(&labels)
            .enumerate()
            .map(|(i, (vector, label))| SpecialistVector {
                name: format!("{label}-{i}"),
                category: Some(label.clone()),
                vector,
                window_var: None,
                source: None,
            })
            .collect()
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * (1.0 + a.abs().max(b.abs()))
    }

    /// Parity with `prototypes/alien_vectors.py` on deterministic inputs:
    /// the prototype's own `--dim 64 --seed 7` V, fixed candidate
    /// constructions, numbers frozen by `tests/fixtures/alien_parity.py`.
    /// Tolerance 1e-9 relative: the two sides solve the least squares by
    /// different factorizations (numpy SVD `lstsq` vs pivoted Householder),
    /// so they agree to rounding, not bit for bit.
    #[test]
    fn prototype_parity_on_the_frozen_fixture() {
        let (fx, v) = fixture();
        let s = scales(&v, 1.5).unwrap();
        for (mine, theirs) in s.nearest.iter().zip(floats(&fx["nn"])) {
            assert!(close(*mine, theirs, 1e-12), "nn {mine} vs {theirs}");
        }
        assert!(close(s.scale, fx["scale"].as_f64().unwrap(), 1e-12));
        assert!(close(s.bandwidth, fx["bandwidth"].as_f64().unwrap(), 1e-12));
        let hull = AffineHull::fit(&v).unwrap();
        assert_eq!(hull.rank() as u64, fx["affine_rank"].as_u64().unwrap());

        let mut points = Vec::new();
        for spec in fx["candidates"].as_array().unwrap() {
            match spec["kind"].as_str().unwrap() {
                "weights" => {
                    let weights: Vec<(usize, f64)> = spec["weights"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, w)| (k.parse().unwrap(), w.as_f64().unwrap()))
                        .collect();
                    points.push(combination(&v, &weights).unwrap());
                }
                "point" => points.push(floats(&spec["point"])),
                other => panic!("unknown candidate kind {other}"),
            }
        }
        let scores = score_points(&points, &v, &hull, &s).unwrap();
        let residual = floats(&fx["residual"]);
        let coherence = floats(&fx["coherence"]);
        let kde = floats(&fx["kde"]);
        let lik = floats(&fx["likelihood_norm"]);
        let gap = floats(&fx["gap"]);
        assert_eq!(scores.len(), residual.len());
        for (k, sc) in scores.iter().enumerate() {
            // On-hull residuals are rounding (~1e-14) on both sides.
            assert!(
                (sc.residual - residual[k]).abs() <= 1e-9 * (1.0 + residual[k]),
                "candidate {k}: residual {} vs {}",
                sc.residual,
                residual[k]
            );
            assert!(
                (sc.coherence - coherence[k]).abs() <= 1e-9,
                "candidate {k} coherence"
            );
            assert!(
                (sc.likelihood_raw - kde[k]).abs() <= 1e-9 * kde[k] + 1e-300,
                "candidate {k}: kde {} vs {}",
                sc.likelihood_raw,
                kde[k]
            );
            assert!(
                (sc.likelihood - lik[k]).abs() <= 1e-9,
                "candidate {k} likelihood"
            );
            assert!((sc.gap - gap[k]).abs() <= 1e-9, "candidate {k} gap");
        }

        // Farthest-interior selection over the midpoints is deterministic.
        let picked = farthest_interior(&v, midpoints(&v).unwrap(), 12);
        let theirs: Vec<(usize, usize)> = fx["farthest_midpoint_pairs_k12"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_u64().unwrap() as usize,
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        let mine: Vec<(usize, usize)> = picked
            .iter()
            .map(|c| (c.weights[0].0, c.weights[1].0))
            .collect();
        assert_eq!(mine, theirs);

        // slerp, and the prototype's Poincare composition through mobius_add.
        let (a, b) = (&v[0], &v[17]);
        for t in ["0.25", "0.5"] {
            let mine = slerp(a, b, t.parse().unwrap()).unwrap();
            let theirs = floats(&fx["slerp_v0_v17"][t]);
            assert!(dist(&mine, &theirs) <= 1e-12, "slerp t={t}");
        }
        let pp = &fx["poincare_prototype_point"];
        let (pa, pb) = (floats(&pp["a"]), floats(&pp["b"]));
        let neg: Vec<f64> = pa.iter().map(|x| -x).collect();
        let diff = mobius_add(&neg, &pb).unwrap();
        let mine = mobius_add(&pa, &scaled(&diff, 0.5)).unwrap();
        assert!(dist(&mine, &floats(&pp["out"])) <= 1e-12);
    }

    /// The prototype's verified qualitative result, on its exact V: the top
    /// candidates are cross-field betweens at coherence ~1 / likelihood ~0,
    /// and both controls score ~0.
    #[test]
    fn prototype_demo_result_reproduces_on_its_own_vectors() {
        let specs = fixture_specialists();
        let report = find_in_scope("all", &specs, &FinderConfig::default(), false, None).unwrap();
        assert!(report.discriminates());
        assert!(
            report.self_checks.iter().all(|c| c.passed),
            "{:?}",
            report.self_checks
        );
        let top: Vec<&Proposal> = report.proposals.iter().take(12).collect();
        for p in &top {
            assert!(p.coherence > 0.999, "{p:?}");
            assert!(p.likelihood < 0.01, "{p:?}");
            assert!(p.gap > 0.99, "{p:?}");
            assert!(p.category_cross, "a top candidate within one field: {p:?}");
        }
        assert!(
            report.controls.on_known_max_gap < 0.15,
            "{:?}",
            report.controls
        );
        assert!(
            report.controls.null_max_coherence < 1e-3,
            "{:?}",
            report.controls
        );
        assert_eq!(report.manifold.affine_rank, 17);
        assert_eq!(report.manifold.codimension, 47);
    }

    /// The same, re-drawn with this crate's RNG (the prototype's
    /// distribution, not its numbers).
    #[test]
    fn prototype_demo_result_reproduces_with_this_rng() {
        let mut rng = StdRng::seed_from_u64(7);
        let (known, labels, betweens) = demo_configuration(&mut rng).unwrap();
        let mut points = betweens.clone();
        points.extend(known.iter().take(3).cloned());
        let hull = AffineHull::fit(&known).unwrap();
        let nulls = nullspace_controls(&known, &hull, 3, &mut rng).unwrap();
        assert_eq!(nulls.len(), 3);
        points.extend(nulls);
        let scores = hull_scorer(&known, &points).unwrap();
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|a, b| scores[*b].gap.total_cmp(&scores[*a].gap));
        assert!(
            order.iter().take(12).all(|&i| i < 12),
            "a control ranked in the top 12"
        );
        for s in &scores[..12] {
            assert!(s.coherence > 0.999 && s.likelihood < 0.01, "{s:?}");
        }
        for s in &scores[12..15] {
            assert!(s.gap < 0.15, "on-known {s:?}");
        }
        for s in &scores[15..] {
            assert!(s.coherence < 1e-3, "null-space {s:?}");
        }
        assert_eq!(labels.len(), 18);
    }

    #[test]
    fn certificates_all_pass() {
        let certs = certificate_checks().unwrap();
        assert_eq!(certs.len(), 6);
        for c in &certs {
            println!(
                "alien::{:<58} residual {:.3e}  tolerance {:.1e}",
                c.name, c.residual, c.tolerance
            );
            assert!(
                c.passed(),
                "{} residual {} > {}",
                c.name,
                c.residual,
                c.tolerance
            );
        }
    }

    #[test]
    fn hull_certificate_fails_for_an_identity_or_base_reconstruction() {
        let identity = |_: &[Vec<f64>], p: &[f64]| -> Result<Vec<f64>> { Ok(p.to_vec()) };
        assert!(hull_exactness_residual(&identity).unwrap() > 1e-3);
        let base = |known: &[Vec<f64>], _: &[f64]| -> Result<Vec<f64>> { Ok(known[0].clone()) };
        assert!(hull_exactness_residual(&base).unwrap() > 1e-3);
    }

    #[test]
    fn complement_certificate_fails_for_constant_coherence() {
        let one = |_: &[Vec<f64>], _: &[f64]| -> Result<f64> { Ok(1.0) };
        let zero = |_: &[Vec<f64>], _: &[f64]| -> Result<f64> { Ok(0.0) };
        assert!(complement_coherence_residual(&one).unwrap() >= 1.0);
        assert!(complement_coherence_residual(&zero).unwrap() >= 1.0);
    }

    #[test]
    fn obvious_and_outrank_certificates_fail_for_a_coherence_only_scorer() {
        let coherence_only = |known: &[Vec<f64>], points: &[Vec<f64>]| -> Result<Vec<PointScore>> {
            let mut s = hull_scorer(known, points)?;
            for x in &mut s {
                x.gap = x.coherence;
            }
            Ok(s)
        };
        assert!(known_obvious_residual(&coherence_only).unwrap() > 0.5);
        assert!(between_within_residual(&coherence_only).unwrap() > 0.4);
        let inverted = |known: &[Vec<f64>], points: &[Vec<f64>]| -> Result<Vec<PointScore>> {
            let mut s = hull_scorer(known, points)?;
            for x in &mut s {
                x.gap = x.coherence * x.likelihood;
            }
            Ok(s)
        };
        assert!(between_within_residual(&inverted).unwrap() > 1.0);
    }

    #[test]
    fn geodesic_certificate_fails_for_a_chord_and_for_the_prototype_midpoint() {
        let chord = |a: &[f64], b: &[f64], t: f64| -> Result<Vec<f64>> {
            Ok(a.iter()
                .zip(b)
                .map(|(x, y)| (1.0 - t) * x + t * y)
                .collect())
        };
        assert!(geodesic_residual(&chord).unwrap() > 1e-3);
        // The prototype's composition: Euclidean scaling of the Mobius
        // difference. On the geodesic, but not at parameter t.
        let prototype = |a: &[f64], b: &[f64], t: f64| -> Result<Vec<f64>> {
            let neg: Vec<f64> = a.iter().map(|x| -x).collect();
            mobius_add(a, &scaled(&mobius_add(&neg, b)?, t))
        };
        assert!(geodesic_residual(&prototype).unwrap() > 1e-3);
    }

    #[test]
    fn survival_certificate_fails_without_the_minimums() {
        let reckless = SurvivalConfig {
            min_fit: 0,
            min_per_class: 0,
            min_report: 0,
            ..SurvivalConfig::default()
        };
        assert!(survival_refusal_residual(&reckless).unwrap() >= 1.0);
    }

    #[test]
    fn a_hull_that_fills_the_space_is_refused_with_its_numbers() {
        let mut rng = StdRng::seed_from_u64(1);
        let full: Vec<Vec<f64>> = (0..10).map(|_| gaussian_vector(&mut rng, 5)).collect();
        let err = AffineHull::fit(&full)
            .unwrap()
            .require_discriminating()
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank 5 in a 5-dimensional space"), "{err}");
        let specs: Vec<SpecialistVector> = full
            .into_iter()
            .enumerate()
            .map(|(i, vector)| SpecialistVector {
                name: format!("s{i}"),
                category: None,
                vector,
                window_var: None,
                source: None,
            })
            .collect();
        assert!(rank(&specs, &RankOptions::default(), None).is_err());
    }

    #[test]
    fn duplicates_and_degenerate_inputs_are_refused_or_handled() {
        // All knowns identical: rank 0, no scale.
        let same = vec![vec![1.0, 2.0, 3.0]; 4];
        assert_eq!(AffineHull::fit(&same).unwrap().rank(), 0);
        assert!(scales(&same, 1.5).is_err());
        // Collinear knowns: rank 1, a point on the line reconstructs.
        let line: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64, 2.0 * i as f64, 1.0])
            .collect();
        let hull = AffineHull::fit(&line).unwrap();
        assert_eq!(hull.rank(), 1);
        assert!(hull.residual(&[10.0, 20.0, 1.0]).unwrap() < 1e-12);
        assert!((hull.residual(&[0.0, 0.0, 4.0]).unwrap() - 3.0).abs() < 1e-12);
        // Dimension mismatch is an error, not a truncated dot product.
        assert!(hull.residual(&[1.0, 2.0]).is_err());
        assert!(AffineHull::fit(&[vec![1.0, f64::NAN]]).is_err());
    }

    #[test]
    fn poincare_points_stay_inside_and_refuse_the_outside() {
        // Past the clamp radius: moved onto it, and the geodesic between the
        // clamped ends still returns them.
        let a = vec![0.999_999_999_9, 0.0];
        let b = vec![-0.999_999_999_9, 0.0];
        for t in [0.0, 0.5, 1.0] {
            let g = poincare_geodesic(&a, &b, t).unwrap();
            assert!(norm(&g) <= 1.0 - BALL_EPS);
        }
        // At the clamp radius Mobius addition of near-antipodal points
        // cancels to ~1e-6 relative (the reason for the clamp); the
        // certificate checks 1e-8 up to norm 0.999.
        let end = poincare_geodesic(&a, &b, 1.0).unwrap();
        assert!((end[0] + (1.0 - BALL_EPS)).abs() < 1e-4, "{end:?}");
        assert!(poincare_geodesic(&[1.0, 0.0], &[0.0, 0.0], 0.5).is_err());
        assert!(poincare_geodesic(&[f64::NAN, 0.0], &[0.0, 0.0], 0.5).is_err());
        // The true midpoint is equidistant; the prototype's point is not.
        let (a, b) = (vec![0.6, 0.1], vec![-0.2, 0.7]);
        let m = poincare_geodesic(&a, &b, 0.5).unwrap();
        let (da, db) = (
            poincare_distance(&a, &m).unwrap(),
            poincare_distance(&m, &b).unwrap(),
        );
        assert!((da - db).abs() < 1e-12);
        let neg: Vec<f64> = a.iter().map(|x| -x).collect();
        let proto = mobius_add(&a, &scaled(&mobius_add(&neg, &b).unwrap(), 0.5)).unwrap();
        let (pa, pb) = (
            poincare_distance(&a, &proto).unwrap(),
            poincare_distance(&proto, &b).unwrap(),
        );
        assert!((pa - pb).abs() > 1e-2, "{pa} {pb}");
    }

    #[test]
    fn curved_betweens_lie_on_the_hull() {
        let specs = fixture_specialists();
        let known: Vec<Vec<f64>> = specs.iter().map(|s| s.vector.clone()).collect();
        let hull = AffineHull::fit(&known).unwrap();
        for c in slerp_midpoints(&known)
            .unwrap()
            .iter()
            .chain(poincare_midpoints(&known).unwrap().iter())
        {
            let r = hull.residual(&c.point).unwrap();
            assert!(
                r < 1e-9 * (1.0 + norm(&c.point)),
                "{:?} residual {r}",
                c.generator
            );
        }
        let s = slerp(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 0.5).unwrap();
        assert!((norm(&s) - 1.0).abs() < 1e-15);
        assert!(slerp(&[1.0, 0.0], &[-1.0, 0.0], 0.5).is_err());
    }

    #[test]
    fn per_category_rank_runs_each_category_and_the_top_level() {
        let mut specs = fixture_specialists();
        specs.push(SpecialistVector {
            name: "lonely".into(),
            category: Some("solo".into()),
            vector: specs[0].vector.iter().map(|x| x + 40.0).collect(),
            window_var: None,
            source: None,
        });
        let options = RankOptions {
            per_category: true,
            ..RankOptions::default()
        };
        let report = rank(&specs, &options, None).unwrap();
        let scopes: Vec<&str> = report.scopes.iter().map(|s| s.scope.as_str()).collect();
        assert_eq!(
            scopes,
            [
                "category:crypto",
                "category:ml",
                "category:networking",
                "top-level"
            ]
        );
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].scope, "category:solo");
        let top = report.scopes.last().unwrap();
        assert_eq!(top.knowns, ["crypto", "ml", "networking", "solo"]);
        assert!(top.proposals.iter().all(|p| p.category_cross));
        // Likelihood over the 19 member specialists, not the 4 centroids:
        // the between-category points are unproposed and the finder says so.
        assert_eq!(top.density_points, 19);
        assert!(top.discriminates(), "{:?}", top.self_checks);
        assert!(top.proposals[0].gap > 0.9);
        // The caveat in the module docs, visible in the summary: the lone
        // "solo" specialist is its own centroid, sits in no dense region, and
        // is the worst-scoring on-known control.
        assert_eq!(top.controls.on_known_worst, "solo");
        // Six specialists of pure noise around one centre have no structure
        // for the bandwidth to resolve; the finder reports that rather than
        // presenting a ranking.
        assert!(!report.scopes[0].discriminates());
        assert!(report.render(3).contains("do NOT discriminate"));
        assert!(report.scopes[0].proposals.iter().all(|p| !p.category_cross));
        let rendered = report.render(5);
        assert!(rendered.contains("alien-gap alone"), "{rendered}");
        assert!(rendered.contains("skipped category:solo"), "{rendered}");
        // Untagged vectors cannot be ranked per category.
        let mut untagged = fixture_specialists();
        untagged[3].category = None;
        assert!(rank(&untagged, &options, None).is_err());
    }

    #[test]
    fn ranking_is_deterministic_and_ids_are_unique() {
        let specs = fixture_specialists();
        let a = rank(&specs, &RankOptions::default(), None).unwrap();
        let b = rank(&specs, &RankOptions::default(), None).unwrap();
        let ids_a: Vec<&str> = a.proposals().map(|p| p.id.as_str()).collect();
        let ids_b: Vec<&str> = b.proposals().map(|p| p.id.as_str()).collect();
        assert_eq!(ids_a, ids_b);
        let unique: HashSet<&str> = ids_a.iter().copied().collect();
        assert_eq!(unique.len(), ids_a.len());
        assert!(a.proposals().all(|p| p.id.starts_with("alien-")));
    }

    #[test]
    fn survival_predictor_reports_honestly_and_ranks_with_it() {
        let (p, o) = synthetic_outcomes(12, 1);
        let err = train_survival(&p, &o, &SurvivalConfig::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("refusing to fit"), "{err}");

        let (p, o) = synthetic_outcomes(60, 2);
        let model = train_survival(&p, &o, &SurvivalConfig::default()).unwrap();
        assert!(model.report.holdout_accuracy.is_none());
        assert!(
            model.render().contains("not reported"),
            "{}",
            model.render()
        );

        let (mut p, mut o) = synthetic_outcomes(400, 3);
        o.push(PilotOutcome {
            proposal_id: "nowhere".into(),
            survived: true,
            notes: String::new(),
        });
        let model = train_survival(&p, &o, &SurvivalConfig::default()).unwrap();
        let r = &model.report;
        assert_eq!(r.unmatched, 1);
        assert_eq!(r.train + r.holdout, 400);
        let acc = r.holdout_accuracy.unwrap();
        assert!(
            acc >= 0.9 && acc > r.majority_baseline_accuracy.unwrap(),
            "{r:?}"
        );
        let (lo, hi) = r.holdout_accuracy_interval.unwrap();
        assert!(lo <= acc && acc <= hi);
        // The planted rule: coherence raises survival.
        let coherence_weight = model.weights[1];
        assert!(coherence_weight > 0.0);

        // Save, load, and rank with it.
        let dir = scratch("survival");
        let path = dir.join("model.json");
        model.save(&path).unwrap();
        let loaded = SurvivalModel::load(&path).unwrap();
        assert_eq!(loaded, model);
        let report = rank(
            &fixture_specialists(),
            &RankOptions::default(),
            Some(&loaded),
        )
        .unwrap();
        assert!(report.render(3).contains("predicted survival"));
        for prop in report.proposals() {
            let s = prop.survival.unwrap();
            assert!((prop.rank_score - prop.gap * s).abs() < 1e-15);
        }
        // A duplicated id with different scores is refused.
        p.push(Proposal {
            gap: 0.123,
            ..p[0].clone()
        });
        assert!(train_survival(&p, &o, &SurvivalConfig::default()).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn plans_must_be_complete_and_name_a_proposal() {
        let (p, _) = synthetic_outcomes(3, 4);
        let plan = Plan {
            proposal_id: p[0].id.clone(),
            mechanism: "m".into(),
            expected_effect: "e".into(),
            baseline: "b".into(),
            failure_mode: "f".into(),
            falsification_test: "t".into(),
        };
        check_plans(std::slice::from_ref(&plan), &p).unwrap();
        let blank = Plan {
            falsification_test: "  ".into(),
            ..plan.clone()
        };
        let err = blank.validate().unwrap_err().to_string();
        assert!(err.contains("falsification_test"), "{err}");
        let orphan = Plan {
            proposal_id: "alien-000000000000".into(),
            ..plan
        };
        assert!(check_plans(&[orphan], &p).is_err());
    }

    #[test]
    fn vector_files_round_trip_and_upsert() {
        let dir = scratch("vectors");
        let path = dir.join("v.json");
        let a = SpecialistVector {
            name: "a".into(),
            category: Some("c".into()),
            vector: vec![1.0, 2.0],
            window_var: Some(vec![0.5, 0.25]),
            source: None,
        };
        assert!(!upsert_vector(&path, a.clone()).unwrap());
        let b = SpecialistVector {
            name: "b".into(),
            vector: vec![3.0, 4.0],
            ..a.clone()
        };
        assert!(!upsert_vector(&path, b.clone()).unwrap());
        let a2 = SpecialistVector {
            vector: vec![0.0, 1.0],
            ..a.clone()
        };
        assert!(upsert_vector(&path, a2.clone()).unwrap());
        assert_eq!(read_vectors(&path).unwrap(), vec![a2, b.clone()]);
        // A bare array reads too; a dimension mismatch does not.
        let bare = dir.join("bare.json");
        std::fs::write(
            &bare,
            r#"[{"name":"x","vector":[1,2]},{"name":"y","vector":[3,4]}]"#,
        )
        .unwrap();
        assert_eq!(read_vectors(&bare).unwrap().len(), 2);
        std::fs::write(
            &bare,
            r#"[{"name":"x","vector":[1,2]},{"name":"y","vector":[3]}]"#,
        )
        .unwrap();
        assert!(read_vectors(&bare).is_err());
        // JSONL round trip.
        let (props, outcomes) = synthetic_outcomes(5, 9);
        let pj = dir.join("p.jsonl");
        write_jsonl(&pj, &props).unwrap();
        assert_eq!(read_jsonl::<Proposal>(&pj).unwrap(), props);
        let oj = dir.join("o.jsonl");
        write_jsonl(&oj, &outcomes).unwrap();
        assert_eq!(read_jsonl::<PilotOutcome>(&oj).unwrap(), outcomes);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn whitening_pools_window_variance() {
        let mk = |name: &str, vector: Vec<f64>, var: Option<Vec<f64>>| SpecialistVector {
            name: name.into(),
            category: None,
            vector,
            window_var: var,
            source: None,
        };
        let v = vec![
            mk("a", vec![2.0, 10.0], Some(vec![4.0, 100.0])),
            mk("b", vec![4.0, 30.0], Some(vec![4.0, 100.0])),
        ];
        let w = whiten_diagonal(&v).unwrap();
        // shrink = 1e-3 * mean(4, 100) = 0.052
        assert!((w[0].vector[0] - 2.0 / (4.052f64).sqrt()).abs() < 1e-12);
        assert!((w[1].vector[1] - 30.0 / (100.052f64).sqrt()).abs() < 1e-12);
        let missing = vec![v[0].clone(), mk("c", vec![1.0, 1.0], None)];
        assert!(whiten_diagonal(&missing).is_err());
    }

    fn tiny_student_config() -> crate::student::StudentConfig {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 2,
            "num_kv_heads": 1,
            "context": 24,
            "intermediate_size": 16,
            "cond_hidden_size": 8,
            "frequency_embedding_size": 8,
            "attention": "dense",
            "experts_per_box": 1,
            "geom_slots": 2,
            "geom_dim": 4,
            "geom_refine_steps": 1,
            "geom_cond_hidden": 8,
            "geom_frequency": 4,
        }))
        .unwrap()
    }

    #[test]
    fn embedding_a_student_checkpoint_is_deterministic_and_corpus_dependent() {
        type Eval = burn_ndarray::NdArray<f32>;
        let dir = scratch("embed");
        let config = tiny_student_config();
        let device = Default::default();
        <Eval as burn::tensor::backend::Backend>::seed(&device, 3);
        let model = crate::student::GeometricStudent::<Eval>::new(&config, &device).unwrap();
        crate::tensor_ext::force_initialization(&model);
        let ckpt =
            crate::checkpoint::save_content_addressed::<Eval, _>(model, &dir, "student").unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&serde_json::json!({ "student_config": config })).unwrap(),
        )
        .unwrap();
        let tok = crate::tokenizer::ByteTokenizer::new();
        let rust = dir.join("rust.bin");
        let prose = dir.join("prose.bin");
        crate::corpus::TokenCorpus::write(
            &rust,
            &tok.encode_document(&"fn main() { let x = 1 + 2; println!(\"{x}\"); }\n".repeat(4)),
        )
        .unwrap();
        crate::corpus::TokenCorpus::write(
            &prose,
            &tok.encode_document(&"The quick brown fox jumps over the lazy dog. ".repeat(4)),
        )
        .unwrap();
        let options = EmbedOptions {
            windows: 3,
            window_len: Some(16),
        };
        let a = embed_student(&ckpt, &rust, "rust", Some("coding"), &options).unwrap();
        let again = embed_student(&ckpt, &rust, "rust", Some("coding"), &options).unwrap();
        let b = embed_student(&ckpt, &prose, "prose", Some("text"), &options).unwrap();
        assert_eq!(a.vector.len(), 32);
        assert_eq!(a.vector, again.vector);
        assert!(dist(&a.vector, &b.vector) > 1e-6);
        assert_eq!(a.window_var.as_ref().unwrap().len(), 32);
        assert!(a
            .source
            .as_ref()
            .unwrap()
            .contains("3 windows of 16 tokens"));
        // A token outside the vocabulary is refused.
        let bad = dir.join("bad.bin");
        crate::corpus::TokenCorpus::write(&bad, &[1, 2, 60000, 4]).unwrap();
        assert!(embed_student(&ckpt, &bad, "bad", None, &options).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
