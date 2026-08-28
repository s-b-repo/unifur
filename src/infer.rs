//! Offline inference API (roadmap item 15.8).
//!
//! [`InferenceEngine`] wraps a trained [`DblockClassifier`] with everything a
//! caller needs to actually use it: the sampling strategy, solver, precision
//! policy and quality gates in one owned configuration, arbitrary input sizes
//! split into fixed batches, top-k outputs, and the per-run statistics the
//! sampler produces.
//!
//! It is an in-process library API rather than a network service. Serving is a
//! deployment concern with its own dependency tree (HTTP stack, batching
//! queue, metrics endpoint), and this crate deliberately has neither network
//! nor async dependencies -- but everything a server would need to call is
//! here, behind one type.

use crate::{
    accuracy::{Guidance, LogitNorm},
    dblock::DblockClassifier,
    multi_block::{Gated, MultiBlockConfig, SamplingStats, Strategy},
    precision::PrecisionPolicy,
    profile::Profiler,
    quality::{LayerGates, QualityGateConfig},
    solver::SolverKind,
};
use burn::tensor::{activation::softmax, backend::Backend, Tensor};
use rand::Rng;

/// How an engine runs inference.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub strategy: Strategy,
    pub solver: SolverKind,
    /// Sampling windows; `None` uses the model's block count.
    pub num_steps: Option<usize>,
    pub precision: PrecisionPolicy,
    pub gates: LayerGates,
    /// Guidance applied to every x0 estimate (roadmap 22.5). The default is the
    /// exact identity; anything else doubles the model calls.
    pub guidance: Guidance,
    /// Logit normalization applied to the returned logits (roadmap 22.6). It
    /// leaves the predicted label untouched and only rescales the reported
    /// confidence, which is what makes a confidence threshold portable between
    /// checkpoints.
    pub logit_norm: LogitNorm,
    /// Inputs larger than this are split into several forward passes.
    pub batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::Sequential,
            solver: SolverKind::default(),
            num_steps: None,
            precision: PrecisionPolicy::default(),
            gates: LayerGates::uniform(QualityGateConfig::lenient()),
            guidance: Guidance::none(),
            logit_norm: LogitNorm::None,
            batch_size: 64,
        }
    }
}

impl InferenceConfig {
    fn to_multi_block(&self) -> MultiBlockConfig {
        MultiBlockConfig {
            strategy: Gated { inner: self.strategy, gate: self.gates.clone() },
            solver: self.solver,
            num_steps: self.num_steps,
            precision: self.precision,
            guidance: self.guidance,
            logit_norm: self.logit_norm,
        }
    }
}

/// One batch's predictions.
#[derive(Debug, Clone)]
pub struct Predictions {
    /// Raw class logits `[n, num_labels]`.
    pub logits: Vec<Vec<f32>>,
    /// Arg-max class per input.
    pub labels: Vec<usize>,
    /// Max class probability per input.
    pub confidence: Vec<f32>,
    /// Sampling statistics accumulated across every batch.
    pub stats: SamplingStats,
}

impl Predictions {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// The `k` highest-scoring `(label, probability)` pairs per input,
    /// descending.
    pub fn top_k(&self, k: usize) -> Vec<Vec<(usize, f32)>> {
        self.logits
            .iter()
            .map(|row| {
                let probs = softmax_row(row);
                let mut ranked: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
                });
                ranked.truncate(k.min(row.len()));
                ranked
            })
            .collect()
    }

    /// Accuracy against ground-truth labels.
    pub fn accuracy(&self, truth: &[usize]) -> f32 {
        if truth.is_empty() {
            return 0.0;
        }
        let correct = self
            .labels
            .iter()
            .zip(truth)
            .filter(|(a, b)| a == b)
            .count();
        correct as f32 / truth.len() as f32
    }
}

/// Numerically stable softmax of one logit row.
fn softmax_row(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; row.len()];
    }
    let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum.max(f32::MIN_POSITIVE)).collect()
}

/// A model plus its inference policy.
pub struct InferenceEngine<B: Backend<FloatElem = f32>> {
    model: DblockClassifier<B>,
    config: InferenceConfig,
}

impl<B: Backend<FloatElem = f32>> InferenceEngine<B> {
    pub fn new(model: DblockClassifier<B>, config: InferenceConfig) -> Self {
        Self { model, config }
    }

    pub fn model(&self) -> &DblockClassifier<B> {
        &self.model
    }

    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut InferenceConfig {
        &mut self.config
    }

    /// Classify `pixel_values` `[n, c, h, w]`, splitting into batches of at
    /// most [`InferenceConfig::batch_size`].
    ///
    /// Splitting is not just a memory convenience: the sampler draws its
    /// initial latent per batch, so batching also bounds how much of a run one
    /// unlucky noise draw can affect.
    pub fn classify(&self, pixel_values: Tensor<B, 4>, rng: &mut impl Rng) -> Predictions {
        self.classify_profiled(pixel_values, rng, &mut Profiler::new())
    }

    /// [`Self::classify`] with per-batch timings recorded into `profiler`.
    pub fn classify_profiled(
        &self,
        pixel_values: Tensor<B, 4>,
        rng: &mut impl Rng,
        profiler: &mut Profiler,
    ) -> Predictions {
        let n = pixel_values.dims()[0];
        let batch_size = self.config.batch_size.max(1);
        let mb = self.config.to_multi_block();

        let mut logits_out: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        let mut confidence = Vec::with_capacity(n);
        let mut stats = SamplingStats::default();

        let mut offset = 0usize;
        while offset < n {
            let take = batch_size.min(n - offset);
            let chunk = pixel_values.clone().narrow(0, offset, take);

            let start = std::time::Instant::now();
            let (logits, batch_stats) = self.model.sample_multi_block(&chunk, &mb, rng);
            profiler.record("sample_multi_block", start.elapsed());

            merge_stats(&mut stats, batch_stats);

            let num_labels = logits.dims()[1];
            let probs: Vec<f32> = softmax(logits.clone(), 1)
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .collect();
            let raw: Vec<f32> = logits.into_data().convert::<f32>().iter::<f32>().collect();

            for row in 0..take {
                let span = row * num_labels..(row + 1) * num_labels;
                let row_probs = &probs[span.clone()];
                let (best, conf) = row_probs.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |acc, (i, &p)| if p > acc.1 { (i, p) } else { acc },
                );
                labels.push(best);
                confidence.push(conf);
                logits_out.push(raw[span].to_vec());
            }
            offset += take;
        }

        Predictions { logits: logits_out, labels, confidence, stats }
    }
}

/// Accumulate one batch's statistics into a running total.
fn merge_stats(acc: &mut SamplingStats, batch: SamplingStats) {
    acc.model_calls += batch.model_calls;
    acc.gated_samples += batch.gated_samples;
    acc.layers_executed += batch.layers_executed;
    acc.reduced_precision_windows += batch.reduced_precision_windows;
    // Spans are identical across batches under a deterministic policy, but an
    // adaptive one can differ, so keep them all.
    acc.spans.extend(batch.spans);
    acc.ledger.merge(&batch.ledger);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dblock::DblockConfig, vit::ViTDiTConfig};
    use burn::backend::NdArray;
    use burn::tensor::Distribution;
    use rand::{rngs::StdRng, SeedableRng};

    type B = NdArray<f32>;

    fn engine(batch_size: usize) -> InferenceEngine<B> {
        let device = Default::default();
        <B as Backend>::seed(&device, 7);
        let model = DblockClassifier::<B>::new(
            &ViTDiTConfig::tiny(10),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        InferenceEngine::new(
            model,
            InferenceConfig { batch_size, num_steps: Some(3), ..InferenceConfig::default() },
        )
    }

    fn inputs(n: usize) -> Tensor<B, 4> {
        Tensor::<B, 4>::random([n, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), &Default::default())
    }

    #[test]
    fn test_shapes_and_confidence_range() {
        let e = engine(64);
        let preds = e.classify(inputs(5), &mut StdRng::seed_from_u64(0));
        assert_eq!(preds.len(), 5);
        assert_eq!(preds.logits.len(), 5);
        assert!(preds.logits.iter().all(|r| r.len() == 10));
        assert!(
            preds.confidence.iter().all(|&c| (0.0..=1.0).contains(&c)),
            "confidence must be a probability"
        );
        assert!(preds.labels.iter().all(|&l| l < 10));
    }

    #[test]
    fn test_batching_does_not_change_the_number_of_results() {
        // Splitting is an implementation detail of memory use; it must not
        // change how many predictions come back or their shape.
        let n = 7usize;
        for batch_size in [1usize, 2, 3, 64] {
            let e = engine(batch_size);
            let preds = e.classify(inputs(n), &mut StdRng::seed_from_u64(1));
            assert_eq!(preds.len(), n, "batch_size {batch_size} lost results");
            // Every batch performs its own sampling run, so calls scale with
            // the number of chunks -- reported honestly rather than hidden.
            let chunks = n.div_ceil(batch_size);
            assert!(
                preds.stats.model_calls >= chunks,
                "stats must accumulate across {chunks} chunks"
            );
        }
    }

    #[test]
    fn test_top_k_is_sorted_and_normalized() {
        let preds = Predictions {
            logits: vec![vec![1.0, 3.0, 2.0, 0.0]],
            labels: vec![1],
            confidence: vec![0.6],
            stats: SamplingStats::default(),
        };
        let top = preds.top_k(3);
        assert_eq!(top[0].iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 2, 0]);
        assert!(top[0].windows(2).all(|w| w[0].1 >= w[1].1), "must be descending");

        // The probabilities are a genuine softmax of the full row, so the
        // whole row sums to one even though only k entries are returned.
        let all = preds.top_k(4);
        let mass: f32 = all[0].iter().map(|(_, p)| p).sum();
        assert!((mass - 1.0).abs() < 1e-6, "softmax mass = {mass}");

        // Asking for more than there are classes is clamped, not a panic.
        assert_eq!(preds.top_k(99)[0].len(), 4);
    }

    #[test]
    fn test_softmax_row_is_stable_for_large_logits() {
        // Naive exp() would overflow to inf/NaN here; the max-subtraction is
        // what keeps a confident prediction from becoming NaN.
        let probs = softmax_row(&[1000.0, 999.0, -1000.0]);
        assert!(probs.iter().all(|p| p.is_finite()));
        let mass: f32 = probs.iter().sum();
        assert!((mass - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn test_accuracy() {
        let preds = Predictions {
            logits: vec![vec![0.0; 3]; 4],
            labels: vec![0, 1, 2, 1],
            confidence: vec![0.5; 4],
            stats: SamplingStats::default(),
        };
        assert!((preds.accuracy(&[0, 1, 9, 9]) - 0.5).abs() < 1e-6);
        assert_eq!(preds.accuracy(&[]), 0.0);
    }

    #[test]
    fn test_merged_gate_rates_stay_in_range() {
        // Regression guard for the ledger merge: rates must remain
        // probabilities after accumulating across batches.
        let e = engine(2);
        let preds = e.classify(inputs(5), &mut StdRng::seed_from_u64(4));
        for block in 0..preds.stats.ledger.num_blocks() {
            let rate = preds.stats.ledger.rejection_rate(block);
            assert!((0.0..=1.0).contains(&rate), "block {block} rate {rate} out of range");
            assert!(
                preds.stats.ledger.rejected(block) <= preds.stats.ledger.evaluated(block),
                "more rejections than evaluations in block {block}"
            );
        }
    }

    #[test]
    fn test_profiler_records_each_batch() {
        let e = engine(2);
        let mut profiler = Profiler::new();
        let _ = e.classify_profiled(inputs(5), &mut StdRng::seed_from_u64(3), &mut profiler);
        // 5 inputs at batch size 2 => 3 sampling runs.
        assert_eq!(profiler.stats("sample_multi_block").unwrap().count(), 3);
    }
}
