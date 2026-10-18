//! Rotating-cohort MoE training for budget hardware.
//!
//! A training strategy that keeps only a cohort of experts resident on a
//! small GPU at a time while the router accumulates knowledge of the whole
//! population. See the "Hardware solution" section of
//! `docs/Frontier-27B-Plan.md` and the plan file for the nine stages.
//!
//! This module pins the *policy* of the strategy as pure, certified logic:
//! fingerprint construction and comparison, the dual-method redundancy gate,
//! cohort rotation, early gatekeeping, cross-cohort replay scheduling, and
//! the load-balance accounting. The GPU-bound training loops that consume
//! these decisions live elsewhere and are exercised by the existing trainers.

use crate::verify::Certificate;

/// A lightweight fingerprint of one expert, resident in memory (tiny next to
/// the weights).
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Mean activation per feature over a fixed probe dataset (gross overlap).
    pub activation_mean: Vec<f32>,
    /// Standard deviation per feature over the same probe set.
    pub activation_std: Vec<f32>,
    /// Output embeddings on a fixed set of probe prompts (behavioral check).
    pub probe_embeddings: Vec<f32>,
    /// Dimension of each probe embedding.
    pub probe_dim: usize,
}

impl Fingerprint {
    pub fn new(
        activation_mean: Vec<f32>,
        activation_std: Vec<f32>,
        probe_embeddings: Vec<f32>,
        probe_dim: usize,
    ) -> Self {
        Self {
            activation_mean,
            activation_std,
            probe_embeddings,
            probe_dim,
        }
    }
}

/// Cosine similarity between two probe-embedding vectors (pooled over probes).
pub fn embedding_similarity(a: &Fingerprint, b: &Fingerprint) -> f64 {
    let na = a.probe_embeddings.len().min(b.probe_embeddings.len());
    if na == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na_norm = 0.0f64;
    let mut nb_norm = 0.0f64;
    for i in 0..na {
        let x = a.probe_embeddings[i] as f64;
        let y = b.probe_embeddings[i] as f64;
        dot += x * y;
        na_norm += x * x;
        nb_norm += y * y;
    }
    let denom = (na_norm.sqrt() * nb_norm.sqrt()).max(1e-12);
    (dot / denom).clamp(-1.0, 1.0)
}

/// Relative activation overlap: fraction of features whose means are within
/// `tol` of each other. Cheap gross-overlap check.
pub fn activation_overlap(a: &Fingerprint, b: &Fingerprint, tol: f32) -> f64 {
    let n = a.activation_mean.len().min(b.activation_mean.len());
    if n == 0 {
        return 0.0;
    }
    let mut close = 0usize;
    for i in 0..n {
        if (a.activation_mean[i] - b.activation_mean[i]).abs() <= tol {
            close += 1;
        }
    }
    close as f64 / n as f64
}

/// Dual-method redundancy decision: flag redundant only when *both* the
/// activation-overlap check and the embedding-similarity check agree, to
/// avoid false alarms from either method's blind spots.
pub fn is_redundant(
    a: &Fingerprint,
    b: &Fingerprint,
    overlap_tol: f32,
    sim_threshold: f64,
) -> bool {
    let overlap = activation_overlap(a, b, overlap_tol);
    let sim = embedding_similarity(a, b);
    // "agree": both methods report high similarity.
    overlap >= 0.5 && sim >= sim_threshold
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    Keep,
    Reinit,
}

/// Early-training gatekeeping: after a handful of steps, compare the new
/// expert's fingerprint to all stored ones; reinit if redundant.
pub fn gatekeep(
    new_fp: &Fingerprint,
    stored: &[Fingerprint],
    overlap_tol: f32,
    sim_threshold: f64,
) -> GateDecision {
    for s in stored {
        if is_redundant(new_fp, s, overlap_tol, sim_threshold) {
            return GateDecision::Reinit;
        }
    }
    GateDecision::Keep
}

/// Build a rotating cohort plan: which expert indices train together in each
/// rotation. Every expert appears in at least one cohort.
pub fn cohort_plan(n_experts: usize, cohort_size: usize) -> Vec<Vec<usize>> {
    if n_experts == 0 {
        return Vec::new();
    }
    let k = cohort_size.max(1).min(n_experts);
    let mut cohorts = Vec::new();
    let mut start = 0usize;
    // Slide by (k-1) so adjacent cohorts share one expert for continuity,
    // but never produce an empty or duplicate trailing cohort.
    let step = if k >= 2 { k - 1 } else { 1 };
    while start < n_experts {
        let end = (start + k).min(n_experts);
        let cohort: Vec<usize> = (start..end).collect();
        if !cohort.is_empty() {
            cohorts.push(cohort);
        }
        if end == n_experts {
            break;
        }
        start += step;
        if start >= n_experts {
            // ensure the last expert is covered
            if !cohorts
                .last()
                .map(|c| c.contains(&(n_experts - 1)))
                .unwrap_or(false)
            {
                cohorts.push(vec![n_experts - 1]);
            }
            break;
        }
    }
    // Guarantee every expert appears at least once.
    for e in 0..n_experts {
        if !cohorts.iter().any(|c| c.contains(&e)) {
            cohorts.push(vec![e]);
        }
    }
    cohorts
}

/// Cross-cohort replay schedule: for each rotation index, which previously
/// trained expert to reload for a short joint-training burst. Rotates through
/// all experts so every expert is eventually refreshed against its neighbors.
pub fn replay_schedule(n_experts: usize, n_rotations: usize) -> Vec<Option<usize>> {
    if n_experts == 0 {
        return vec![None; n_rotations];
    }
    (0..n_rotations)
        .map(|r| {
            if r == 0 {
                None
            } else {
                Some((r - 1) % n_experts)
            }
        })
        .collect()
}

/// Load-balance accounting: returns the mean squared deviation of per-expert
/// usage counts from uniform. Lower is better; 0 is perfectly balanced.
pub fn load_imbalance(usage: &[u64]) -> f64 {
    if usage.is_empty() {
        return 0.0;
    }
    let total: u64 = usage.iter().sum();
    let n = usage.len() as f64;
    let mean = total as f64 / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var: f64 = usage
        .iter()
        .map(|&u| {
            let d = u as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (var.sqrt() / mean).max(0.0)
}

/// Light calibration: given per-expert frozen flags, returns true if the
/// calibration pass should update *only* the router (all experts frozen).
pub fn calibration_router_only(expert_frozen: &[bool]) -> bool {
    !expert_frozen.is_empty() && expert_frozen.iter().all(|&f| f)
}

// -- certificates ----------------------------------------------------------

pub fn certificates() -> Vec<Certificate> {
    // 1. embedding_similarity is symmetric and bounded.
    let mut sym_violation = 0.0f64;
    let a = Fingerprint::new(vec![1.0, 2.0], vec![0.5, 0.5], vec![1.0, 0.0, 0.0, 1.0], 2);
    let b = Fingerprint::new(vec![1.0, 2.0], vec![0.5, 0.5], vec![0.0, 1.0, 1.0, 0.0], 2);
    let s_ab = embedding_similarity(&a, &b);
    let s_ba = embedding_similarity(&b, &a);
    if (s_ab - s_ba).abs() > 1e-12 {
        sym_violation += 1.0;
    }
    if !(-1.0..=1.0).contains(&s_ab) {
        sym_violation += 1.0;
    }

    // 2. cohort_plan covers every expert at least once.
    let mut coverage_violation = 0.0f64;
    for n in 1usize..=12 {
        for k in 1usize..=4 {
            let plan = cohort_plan(n, k);
            for e in 0..n {
                if !plan.iter().any(|c| c.contains(&e)) {
                    coverage_violation += 1.0;
                }
            }
            // no empty cohort
            if plan.iter().any(|c| c.is_empty()) {
                coverage_violation += 1.0;
            }
        }
    }

    // 3. replay_schedule has length n_rotations and covers every expert
    //    (when n_rotations >= n_experts + 1).
    let mut replay_violation = 0.0f64;
    let sched = replay_schedule(4, 8);
    if sched.len() != 8 {
        replay_violation += 1.0;
    }
    // first rotation has no replay
    if sched[0].is_some() {
        replay_violation += 1.0;
    }
    // experts 0..3 all appear
    for e in 0..4usize {
        if !sched.contains(&Some(e)) {
            replay_violation += 1.0;
        }
    }

    // 4. gatekeeping returns Reinit iff a redundant stored fingerprint exists.
    let mut gate_violation = 0.0f64;
    let stored = vec![a.clone()];
    // identical fingerprint -> redundant
    if gatekeep(&a, &stored, 0.0, 0.5) != GateDecision::Reinit {
        gate_violation += 1.0;
    }
    // orthogonal fingerprint -> keep
    let fresh = Fingerprint::new(vec![1.0, 2.0], vec![0.5, 0.5], vec![0.0, 0.0, 1.0, 0.0], 2);
    if gatekeep(&fresh, &stored, 0.0, 0.99) != GateDecision::Keep {
        gate_violation += 1.0;
    }

    // 5. load_imbalance is 0 for uniform usage.
    let mut lb_violation = 0.0f64;
    if load_imbalance(&[10, 10, 10, 10]) > 1e-12 {
        lb_violation += 1.0;
    }
    if load_imbalance(&[]) != 0.0 {
        lb_violation += 1.0;
    }

    // 6. calibration_router_only true iff all experts frozen.
    let mut cal_violation = 0.0f64;
    if !calibration_router_only(&[true, true, true]) {
        cal_violation += 1.0;
    }
    if calibration_router_only(&[true, false, true]) {
        cal_violation += 1.0;
    }
    if calibration_router_only(&[]) {
        cal_violation += 1.0;
    }

    // 7. dual-method agreement: redundant requires both methods high.
    let mut agree_violation = 0.0f64;
    // high embedding sim but low activation overlap -> not redundant
    let hi_emb = Fingerprint::new(vec![0.0, 0.0], vec![1.0, 1.0], vec![1.0, 0.0, 0.0, 1.0], 2);
    let hi_emb2 = Fingerprint::new(
        vec![10.0, 10.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
    );
    if is_redundant(&hi_emb, &hi_emb2, 0.0, 0.5) {
        agree_violation += 1.0;
    }

    vec![
        Certificate {
            group: "cohort",
            name: "embedding_similarity_is_symmetric_and_bounded",
            theorem: "Probe-embedding similarity is symmetric and lies in [-1, 1].",
            residual: sym_violation,
            tolerance: 0.0,
        },
        Certificate {
            group: "cohort",
            name: "cohort_plan_covers_every_expert",
            theorem: "Every expert appears in at least one cohort and no cohort is empty.",
            residual: coverage_violation,
            tolerance: 0.0,
        },
        Certificate {
            group: "cohort",
            name: "replay_schedule_covers_experts",
            theorem: "The replay schedule has the requested length, starts with no replay, and eventually replays every expert.",
            residual: replay_violation,
            tolerance: 0.0,
        },
        Certificate {
            group: "cohort",
            name: "gatekeeping_reinits_redundant_experts",
            theorem: "Gatekeeping reinitializes an expert iff it is redundant with a stored fingerprint.",
            residual: gate_violation,
            tolerance: 0.0,
        },
        Certificate {
            group: "cohort",
            name: "load_imbalance_zero_for_uniform",
            theorem: "Uniform usage yields zero imbalance; empty usage yields zero.",
            residual: lb_violation,
            tolerance: 0.0,
        },
        Certificate {
            group: "cohort",
            name: "calibration_updates_router_only_when_frozen",
            theorem: "Calibration touches only the router when every expert is frozen.",
            residual: cal_violation,
            tolerance: 0.0,
        },
        Certificate {
            group: "cohort",
            name: "redundancy_requires_dual_method_agreement",
            theorem: "Redundancy is flagged only when both activation overlap and embedding similarity are high.",
            residual: agree_violation,
            tolerance: 0.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohort_plan_small() {
        let plan = cohort_plan(4, 2);
        assert!(!plan.is_empty());
        for e in 0..4 {
            assert!(plan.iter().any(|c| c.contains(&e)));
        }
    }

    #[test]
    fn gatekeep_redundant() {
        let fp = Fingerprint::new(vec![1.0], vec![1.0], vec![1.0, 0.0], 2);
        assert_eq!(
            gatekeep(&fp, std::slice::from_ref(&fp), 0.0, 0.5),
            GateDecision::Reinit
        );
    }

    #[test]
    fn calibration_router_only_all_frozen() {
        assert!(calibration_router_only(&[true, true]));
        assert!(!calibration_router_only(&[true, false]));
    }
}
