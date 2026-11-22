//! Next-Concept Modeling (NCM): the training method for the nina models.
//!
//! NCM (Kumbong et al., Stanford, arXiv:2610.14162) trains in two stages:
//!
//! 1. A **multiscale residual-VQ tokenizer** encodes a token sequence of length
//!    `L` into `K` levels of discrete codes with lengths `ℓ_1 < … < ℓ_K = L`.
//!    Level `k` quantizes the residual the coarser levels left behind, so the
//!    cumulative reconstruction `ẑ^(k) = ẑ^(k-1) + R_k(e_k(r_k))` is a usable
//!    partial reconstruction at every `k`: these are the *concept vectors*.
//! 2. With the tokenizer frozen, a **level-wise generator** predicts every code
//!    of level `k` in one forward pass, conditioned on `ẑ^(k-1)`. Generation is
//!    `K` network calls (NFE = K), coarse to fine.
//!
//! This file holds only what both stages share: the level schedule and the
//! [`ConceptCodec`] interface through which the generator sees the frozen
//! tokenizer. The tokenizer and the generator live in their own submodules,
//! so they can be built and tested independently against this contract.
//!
//! # Honesty about the claims
//!
//! The paper reports 10–31× fewer network calls than diffusion LMs at matched
//! quality for 25M–725M models on general text. That is a *decode-call*
//! result on the paper's corpora. It has not been measured here on
//! byte-level security specialists or on phones; any speed or quality claim
//! about nina models must come from a measurement in this crate.
//!
//! # Training cost (read before promising a training run)
//!
//! NCM's speed-up is a *decode* result: it cuts the number of generation-time
//! network calls. It does not make training cheaper. A training step still
//! pushes every token through the network forwards and backwards, so the cost
//! is `FLOPs ≈ 6 · N · D` (`N` parameters, `D` training tokens). The unified
//! concept-field generator re-runs its block once per thinking step, which
//! multiplies that by `(1 + thinking_steps)`. The tokenizer (0 encoder + 1
//! decoder block) is cheap by comparison and is trained once for every field.
//!
//! Release-build NdArray samples on the dev laptop measured 2.7–15.3 GFLOP/s
//! on generator widths 64–512. Use the Training Guide's worker qualification
//! procedure to measure the intended batch, sequence length, device and
//! latent depth before budgeting a full fit.

use anyhow::{bail, ensure, Result};
use burn::tensor::{backend::Backend, Int, Tensor};
use serde::{Deserialize, Serialize};

fn check_stop_step(steps: usize, stop_at: Option<usize>, can_checkpoint: bool) -> Result<()> {
    if let Some(step) = stop_at {
        ensure!(
            step > 0 && step <= steps,
            "stop_at must be between 1 and the planned {steps} steps, got {step}"
        );
        ensure!(
            can_checkpoint,
            "stop_at requires a checkpoint output directory"
        );
    }
    Ok(())
}

pub mod cli;
pub mod decode;
pub mod generator;
pub mod released;
pub mod tokenizer;

/// The level lengths `ℓ_1 < … < ℓ_K = L`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LevelSchedule {
    lengths: Vec<usize>,
}

impl LevelSchedule {
    /// A schedule from explicit lengths. They must be strictly increasing,
    /// start at 1 or more, and end at the sequence length.
    pub fn new(lengths: Vec<usize>) -> Result<Self> {
        let schedule = Self { lengths };
        schedule.validate()?;
        Ok(schedule)
    }

    /// A geometric schedule: `ℓ_k ≈ L^{k/K}`, rounded, forced strictly
    /// increasing, with `ℓ_K = L` exactly. Coarse levels are short (a few
    /// codes summarise the whole sequence) and each level roughly multiplies
    /// the length by `L^{1/K}`, which is the shape the paper's schedules have.
    pub fn geometric(seq_len: usize, levels: usize) -> Result<Self> {
        ensure!(levels >= 1, "an NCM schedule needs at least one level");
        ensure!(
            levels <= seq_len,
            "{levels} strictly increasing levels cannot fit a sequence of length {seq_len}"
        );
        let mut lengths = Vec::with_capacity(levels);
        let mut previous = 0usize;
        for k in 1..=levels {
            // Leave room so the remaining levels can still increase strictly
            // and the last one lands exactly on `seq_len`.
            let ceiling = seq_len - (levels - k);
            let ideal = (seq_len as f64).powf(k as f64 / levels as f64).round() as usize;
            let length = ideal.max(previous + 1).min(ceiling);
            lengths.push(length);
            previous = length;
        }
        Self::new(lengths)
    }

    /// Check the structural promises: non-empty, starts at ≥1, strictly
    /// increasing.
    pub fn validate(&self) -> Result<()> {
        let Some(&first) = self.lengths.first() else {
            bail!("an NCM schedule needs at least one level");
        };
        ensure!(first >= 1, "level 1 must have at least one code");
        for pair in self.lengths.windows(2) {
            if let [a, b] = pair {
                ensure!(
                    a < b,
                    "level lengths must strictly increase, got {a} then {b}"
                );
            }
        }
        Ok(())
    }

    /// Number of levels `K`.
    pub fn levels(&self) -> usize {
        self.lengths.len()
    }

    /// Sequence length `L = ℓ_K`.
    pub fn seq_len(&self) -> usize {
        self.lengths.last().copied().unwrap_or(0)
    }

    /// `ℓ_k` for a 0-based level index.
    pub fn len_of(&self, level: usize) -> Option<usize> {
        self.lengths.get(level).copied()
    }

    /// All lengths, coarse to fine.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// The paper's level weights `w_k = ℓ_k^{-α} / Σ_j ℓ_j^{1-α}` for the
    /// generator loss `−Σ_k w_k Σ_i log p(r_k^i | z̃^(k-1))`. `α = 0` weights
    /// every code equally; `α = 1` weights every level equally.
    pub fn loss_weights(&self, alpha: f64) -> Result<Vec<f64>> {
        ensure!(
            alpha.is_finite() && (0.0..=1.0).contains(&alpha),
            "alpha must be in [0, 1], got {alpha}"
        );
        let norm: f64 = self
            .lengths
            .iter()
            .map(|&l| (l as f64).powf(1.0 - alpha))
            .sum();
        ensure!(
            norm > 0.0 && norm.is_finite(),
            "degenerate schedule for loss weights"
        );
        Ok(self
            .lengths
            .iter()
            .map(|&l| (l as f64).powf(-alpha) / norm)
            .collect())
    }
}

/// Traverse the prepared corpus order, starting at a resumable batch index.
pub(crate) fn ordered_record_offsets(
    tokens: usize,
    width: usize,
    count: usize,
    batch_index: usize,
) -> Result<impl Iterator<Item = usize>> {
    ensure!(
        width > 0 && tokens >= width && tokens.is_multiple_of(width),
        "expected whole records of width {width}, got {tokens} tokens"
    );
    let records = (tokens / width) as u128;
    let start = (batch_index as u128 * count as u128) % records;
    Ok((0..count).map(move |offset| ((start + offset as u128) % records) as usize * width))
}

/// Codes for every level, coarse to fine: `codes[k]` is `[batch, ℓ_k]` with
/// values in `0..codebook_size`.
#[derive(Debug, Clone)]
pub struct LevelCodes<B: Backend> {
    pub codes: Vec<Tensor<B, 2, Int>>,
}

/// The frozen tokenizer as the generator sees it.
///
/// Implemented by the residual-VQ tokenizer; the generator is written and
/// tested against this trait only, so either side can change internally
/// without the other noticing.
pub trait ConceptCodec<B: Backend> {
    /// The level schedule the codes follow.
    fn schedule(&self) -> &LevelSchedule;
    /// Codebook size per level (the generator's output classes).
    fn codebook_size(&self) -> usize;
    /// Latent width `d` of the concept vectors.
    fn latent_dim(&self) -> usize;
    /// Token vocabulary size of the decoder head.
    fn vocab_size(&self) -> usize;
    /// Encode token ids `[batch, L]` into codes for all `K` levels.
    fn encode(&self, tokens: Tensor<B, 2, Int>) -> Result<LevelCodes<B>>;
    /// The cumulative reconstruction `ẑ^(k)` `[batch, L, d]` from the codes of
    /// the first `codes.len()` levels (an empty slice gives `ẑ^(0)`, zeros).
    fn reconstruct(&self, codes: &[Tensor<B, 2, Int>], batch: usize) -> Result<Tensor<B, 3>>;
    /// Token logits `[batch, L, vocab]` from a (possibly partial) reconstruction.
    fn decode_logits(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>>;
}

/// What the concept-field generator needs from the frozen tokenizer beyond
/// [`ConceptCodec`]: each level's codebook, whose rows are the attractors the
/// generator's relaxation settles into.
///
/// Row `j` of `level_codebook(k)` must be the vector code `j` of level `k`
/// stands for, i.e. the vector `encode` matched and `reconstruct` adds back
/// for that code. The generator only reads it (the codebook stays frozen).
pub trait SoftConceptCodec<B: Backend>: ConceptCodec<B> {
    /// The frozen codebook of 0-based `level`, `[codebook_size, code_dim]`.
    fn level_codebook(&self, level: usize) -> Result<Tensor<B, 2>>;
}

/// A level-wise generator as the decoding strategies see it (early exit,
/// level speculation, repair, particles — `ncm::decode`): one call produces
/// code logits for every position of one level.
///
/// Implemented by the unified generator; a draft model for speculation
/// implements it too. Keeping the strategies behind this trait means they
/// can be certified against small test doubles and reused for any generator.
pub trait LevelModel<B: Backend> {
    /// Code logits `[batch, ℓ_level, codebook]` for 0-based `level`, given the
    /// cumulative reconstruction of every coarser level `ẑ^(level)`
    /// (`[batch, L, d]`, zeros for level 0) and an optional prompt of token
    /// ids `[batch, P]`.
    fn level_logits(
        &self,
        level: usize,
        zhat_coarser: Tensor<B, 3>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 3>>;

    /// Several levels in one call, for speculative verification. The default
    /// makes one `level_logits` call per request; an implementation that can
    /// batch levels into a single forward pass overrides it and says so in
    /// [`LevelModel::batches_levels`], because only then does verifying `m`
    /// drafted levels cost one call instead of `m`.
    fn levels_logits(
        &self,
        requests: Vec<(usize, Tensor<B, 3>)>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Vec<Tensor<B, 3>>> {
        requests
            .into_iter()
            .map(|(level, zhat)| self.level_logits(level, zhat, prompt.clone()))
            .collect()
    }

    /// Whether [`LevelModel::levels_logits`] really evaluates all requests in
    /// one forward pass. Decoding reports count calls with this, so a
    /// speculative speed-up is never claimed for a model that loops.
    fn batches_levels(&self) -> bool {
        false
    }
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
    fn ordered_batches_cover_each_record_and_resume_at_wrap() {
        let offsets = |batch| {
            ordered_record_offsets(20, 4, 3, batch)
                .unwrap()
                .collect::<Vec<_>>()
        };
        assert_eq!(offsets(0), [0, 4, 8]);
        assert_eq!(offsets(1), [12, 16, 0]);
        assert_eq!(offsets(2), [4, 8, 12]);
        assert!(ordered_record_offsets(22, 4, 3, 0).is_err());
    }

    #[test]
    fn test_geometric_schedule_is_strictly_increasing_and_ends_at_l() {
        for (l, k) in [
            (1024, 32),
            (128, 12),
            (48, 8),
            (64, 11),
            (4096, 32),
            (8, 8),
            (5, 1),
        ] {
            let s = LevelSchedule::geometric(l, k).unwrap();
            assert_eq!(s.levels(), k);
            assert_eq!(s.seq_len(), l);
            assert!(
                s.lengths().windows(2).all(|w| w[0] < w[1]),
                "{l}/{k}: {:?}",
                s.lengths()
            );
            assert!(s.lengths()[0] >= 1);
        }
    }

    #[test]
    fn test_impossible_schedules_are_refused() {
        assert!(LevelSchedule::geometric(4, 5).is_err());
        assert!(LevelSchedule::geometric(4, 0).is_err());
        assert!(LevelSchedule::new(vec![]).is_err());
        assert!(LevelSchedule::new(vec![0, 4]).is_err());
        assert!(LevelSchedule::new(vec![2, 2, 4]).is_err());
    }

    #[test]
    fn test_loss_weights_match_the_paper_formula_at_both_ends() {
        let s = LevelSchedule::new(vec![1, 4, 16]).unwrap();
        // α = 0: every code weighs the same, 1 / Σ ℓ_j.
        let w0 = s.loss_weights(0.0).unwrap();
        for w in &w0 {
            assert!((w - 1.0 / 21.0).abs() < 1e-15);
        }
        // Total weight over all codes is 1 at α = 0.
        let total: f64 = w0.iter().zip(s.lengths()).map(|(w, &l)| w * l as f64).sum();
        assert!((total - 1.0).abs() < 1e-12);
        // α = 1: every level weighs the same in total, w_k ℓ_k = 1/K.
        let w1 = s.loss_weights(1.0).unwrap();
        for (w, &l) in w1.iter().zip(s.lengths()) {
            assert!((w * l as f64 - 1.0 / 3.0).abs() < 1e-12);
        }
        assert!(s.loss_weights(1.5).is_err());
        assert!(s.loss_weights(f64::NAN).is_err());
    }
}
