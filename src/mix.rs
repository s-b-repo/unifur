//! Multi-source training (roadmap Phase 29): several datasets or corpora in
//! one run, mixed either by drawing each batch from one source (**mixture**)
//! or by composing every batch from all of them (**composite**).
//!
//! The weights are the whole interface. A mixture draws the source of each
//! batch from the host RNG in proportion to them -- so the draw is part of
//! the resumable state -- and a composite splits every batch between the
//! sources in proportion to them, exactly, by largest-remainder
//! apportionment. Both report where each batch came from, so a per-source
//! loss can be kept: a source whose loss never falls is invisible in the
//! aggregate and obvious here.

use anyhow::Context;
use burn::tensor::{backend::Backend, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::data::Batch;

/// How several sources share a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MixMode {
    /// Every batch comes from one source, drawn by weight.
    #[default]
    Mixture,
    /// Every batch holds a slice from every source, sized by weight.
    Composite,
}

impl MixMode {
    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "mixture" => Ok(Self::Mixture),
            "composite" => Ok(Self::Composite),
            other => anyhow::bail!("unknown mix mode {other:?}; expected mixture | composite"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Mixture => "mixture",
            Self::Composite => "composite",
        }
    }
}

/// Normalized, positive weights over the sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixWeights {
    weights: Vec<f64>,
}

impl MixWeights {
    /// Normalize `raw` to sum to 1. Every weight must be positive: a source
    /// with weight zero should not be listed at all.
    pub fn new(raw: &[f64]) -> anyhow::Result<Self> {
        anyhow::ensure!(!raw.is_empty(), "at least one source is needed");
        anyhow::ensure!(
            raw.iter().all(|w| w.is_finite() && *w > 0.0),
            "every source weight must be positive and finite: {raw:?}"
        );
        let total: f64 = raw.iter().sum();
        Ok(Self { weights: raw.iter().map(|w| w / total).collect() })
    }

    pub fn uniform(n: usize) -> Self {
        let n = n.max(1);
        Self { weights: vec![1.0 / n as f64; n] }
    }

    /// `"0.7,0.3"`, or empty for uniform over `n` sources.
    pub fn parse(text: &str, n: usize) -> anyhow::Result<Self> {
        if text.trim().is_empty() {
            return Ok(Self::uniform(n));
        }
        let raw: Vec<f64> = text
            .split(',')
            .map(|s| s.trim().parse::<f64>().with_context(|| format!("weight {s:?} is not a number")))
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(raw.len() == n, "{} weight(s) given for {n} source(s)", raw.len());
        Self::new(&raw)
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn get(&self, i: usize) -> f64 {
        self.weights[i]
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.weights
    }

    /// Draw a source index by weight (inverse CDF on one uniform). A single
    /// source consumes **no** randomness, so a one-source mix draws exactly
    /// the batches the source alone would from the same stream.
    pub fn draw<R: Rng>(&self, rng: &mut R) -> usize {
        if self.weights.len() == 1 {
            return 0;
        }
        let u: f64 = rng.random::<f64>();
        let mut acc = 0.0;
        for (i, w) in self.weights.iter().enumerate() {
            acc += w;
            if u < acc {
                return i;
            }
        }
        self.weights.len() - 1
    }

    /// Split `batch` items between the sources in proportion to the weights,
    /// exactly: floors first, then the remainder to the largest fractional
    /// parts (Hamilton's apportionment). The slices always sum to `batch`.
    pub fn split(&self, batch: usize) -> Vec<usize> {
        let exact: Vec<f64> = self.weights.iter().map(|w| w * batch as f64).collect();
        let mut slices: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
        let mut remainder = batch - slices.iter().sum::<usize>();
        let mut order: Vec<usize> = (0..self.weights.len()).collect();
        order.sort_by(|a, b| {
            let fa = exact[*a] - exact[*a].floor();
            let fb = exact[*b] - exact[*b].floor();
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
        });
        for i in order {
            if remainder == 0 {
                break;
            }
            slices[i] += 1;
            remainder -= 1;
        }
        slices
    }
}

/// How many items of a batch came from each source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOrigin {
    pub counts: Vec<usize>,
}

impl BatchOrigin {
    /// A whole batch from one source.
    pub fn single(source: usize, sources: usize, batch: usize) -> Self {
        let mut counts = vec![0; sources];
        counts[source] = batch;
        Self { counts }
    }

    /// The source that supplied the batch, when exactly one did.
    pub fn sole_source(&self) -> Option<usize> {
        let mut found = None;
        for (i, c) in self.counts.iter().enumerate() {
            if *c > 0 {
                if found.is_some() {
                    return None;
                }
                found = Some(i);
            }
        }
        found
    }

    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }
}

/// Concatenate batches from several sources along the batch axis, in order.
pub fn concat_batches<B: Backend>(parts: Vec<Batch<B>>) -> anyhow::Result<Batch<B>> {
    let mut parts = parts.into_iter().filter(|b| b.batch_size() > 0);
    let first = parts.next().context("a composite batch needs at least one non-empty part")?;
    let Some(second) = parts.next() else {
        return Ok(first);
    };
    let (mut pixels, mut labels) = (vec![first.pixel_values, second.pixel_values], vec![first.labels, second.labels]);
    for b in parts {
        pixels.push(b.pixel_values);
        labels.push(b.labels);
    }
    Ok(Batch { pixel_values: Tensor::cat(pixels, 0), labels: Tensor::cat(labels, 0) })
}

/// Per-source loss bookkeeping for the end-of-run report.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceStats {
    pub names: Vec<String>,
    pub batches: Vec<usize>,
    pub samples: Vec<usize>,
    loss_sum: Vec<f64>,
}

impl SourceStats {
    pub fn new(names: Vec<String>) -> Self {
        let n = names.len();
        Self { names, batches: vec![0; n], samples: vec![0; n], loss_sum: vec![0.0; n] }
    }

    /// Attribute a step's loss to the sources its batch came from, weighted
    /// by their share of the batch. A composite batch's loss is one number;
    /// it is credited to every contributing source in proportion.
    pub fn record(&mut self, origin: &BatchOrigin, loss: f32) {
        if self.names.is_empty() || !loss.is_finite() {
            return;
        }
        let total = origin.total().max(1) as f64;
        for (i, count) in origin.counts.iter().enumerate() {
            if *count == 0 || i >= self.names.len() {
                continue;
            }
            self.batches[i] += 1;
            self.samples[i] += count;
            self.loss_sum[i] += f64::from(loss) * (*count as f64 / total);
        }
    }

    /// Mean loss attributed to source `i`, per batch it contributed to.
    pub fn mean_loss(&self, i: usize) -> f32 {
        if self.batches.get(i).copied().unwrap_or(0) == 0 {
            0.0
        } else {
            (self.loss_sum[i] / self.batches[i] as f64) as f32
        }
    }

    pub fn is_multi(&self) -> bool {
        self.names.len() > 1
    }

    pub fn render(&self) -> String {
        let mut out = format!("{:<32} {:>8} {:>10} {:>12}\n", "source", "batches", "samples", "mean loss");
        out.push_str(&"-".repeat(66));
        out.push('\n');
        for (i, name) in self.names.iter().enumerate() {
            out.push_str(&format!(
                "{:<32} {:>8} {:>10} {:>12.4}\n",
                name,
                self.batches[i],
                self.samples[i],
                self.mean_loss(i)
            ));
        }
        out
    }
}

/// Windows plus, when any source is labeled, one penalty-weight row per window.
pub type WeightedWindows = (Vec<Vec<u16>>, Option<Vec<Vec<f32>>>);

/// [`WeightedWindows`] plus where the windows came from.
pub type MixedWindows = (Vec<Vec<u16>>, Option<Vec<Vec<f32>>>, BatchOrigin);

/// Several corpora in one LM run (roadmap 29.1): the token-side counterpart
/// of the dataset mix. Borrowed, so the caller keeps ownership of each corpus
/// and its labels.
pub struct CorpusMix<'a> {
    sources: Vec<&'a mut crate::corpus::TokenCorpus>,
    names: Vec<String>,
    weights: MixWeights,
    mode: MixMode,
    /// Per source: its label -> penalty-weight table, when it has labels open.
    tables: Vec<Option<[f32; 256]>>,
}

impl<'a> CorpusMix<'a> {
    pub fn new(
        sources: Vec<&'a mut crate::corpus::TokenCorpus>,
        weights: MixWeights,
        mode: MixMode,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!sources.is_empty(), "at least one corpus is needed");
        anyhow::ensure!(weights.len() == sources.len(), "{} weight(s) for {} corpora", weights.len(), sources.len());
        let mut tables = Vec::with_capacity(sources.len());
        let mut names = Vec::with_capacity(sources.len());
        for c in &sources {
            tables.push(if c.has_labels() { Some(c.manifest()?.weight_table()) } else { None });
            names.push(c.path().display().to_string());
        }
        Ok(Self { sources, names, weights, mode, tables })
    }

    /// One corpus, no mixing: what every earlier run was.
    pub fn single(corpus: &'a mut crate::corpus::TokenCorpus) -> anyhow::Result<Self> {
        Self::new(vec![corpus], MixWeights::uniform(1), MixMode::Mixture)
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn mode(&self) -> MixMode {
        self.mode
    }

    pub fn weights(&self) -> &MixWeights {
        &self.weights
    }

    /// Whether any source has labels open (the batch then carries penalty
    /// weights, zero for windows from unlabeled sources).
    pub fn any_labels(&self) -> bool {
        self.tables.iter().any(Option::is_some)
    }

    pub fn total_tokens(&self) -> usize {
        self.sources.iter().map(|c| c.len()).sum()
    }

    /// The first source's path -- what a single-corpus run reports.
    pub fn primary_path(&self) -> &std::path::Path {
        self.sources[0].path()
    }

    /// Identity of every source's bytes (corpus, and labels when open).
    pub fn identities(&self) -> anyhow::Result<Vec<crate::checkpoint::DatasetIdentity>> {
        self.sources
            .iter()
            .map(|c| {
                let mut paths = vec![c.path().to_path_buf()];
                if c.has_labels() {
                    paths.push(crate::corpus::labels_path(c.path()));
                }
                crate::checkpoint::DatasetIdentity::of_paths(&paths, "corpus")
            })
            .collect()
    }

    fn draw_from(
        &mut self,
        source: usize,
        count: usize,
        context: usize,
        rng: &mut impl Rng,
        want_weights: bool,
    ) -> anyhow::Result<WeightedWindows> {
        let table = self.tables[source];
        let corpus = &mut self.sources[source];
        match table {
            Some(table) if want_weights => {
                let (windows, labels) = corpus.sample_batch_labeled(count, context, rng)?;
                let rows = labels
                    .iter()
                    .map(|row| row.iter().map(|l| table[usize::from(*l)]).collect())
                    .collect();
                Ok((windows, Some(rows)))
            }
            _ => {
                let windows = corpus.sample_batch(count, context, rng)?;
                let rows = want_weights.then(|| windows.iter().map(|w| vec![0.0f32; w.len()]).collect());
                Ok((windows, rows))
            }
        }
    }

    /// A batch of `batch` windows of `context + 1` tokens: from one source
    /// drawn by weight (mixture) or sliced from every source (composite).
    /// Returns the windows, the penalty-weight rows when any source is
    /// labeled, and where the windows came from.
    pub fn sample<R: Rng>(
        &mut self,
        batch: usize,
        context: usize,
        rng: &mut R,
    ) -> anyhow::Result<MixedWindows> {
        let want_weights = self.any_labels();
        match self.mode {
            MixMode::Mixture => {
                let source = self.weights.draw(rng);
                let (windows, rows) = self.draw_from(source, batch, context, rng, want_weights)?;
                Ok((windows, rows, BatchOrigin::single(source, self.sources.len(), batch)))
            }
            MixMode::Composite => {
                let slices = self.weights.split(batch);
                let mut windows = Vec::with_capacity(batch);
                let mut rows: Option<Vec<Vec<f32>>> = want_weights.then(Vec::new);
                for (source, count) in slices.iter().enumerate() {
                    if *count == 0 {
                        continue;
                    }
                    let (w, r) = self.draw_from(source, *count, context, rng, want_weights)?;
                    windows.extend(w);
                    if let (Some(acc), Some(r)) = (rows.as_mut(), r) {
                        acc.extend(r);
                    }
                }
                Ok((windows, rows, BatchOrigin { counts: slices }))
            }
        }
    }
}

/// `[batch, n]` penalty weights from already-resolved rows (one per window).
pub fn weight_rows<B: Backend>(rows: &[Vec<f32>], device: &B::Device) -> Tensor<B, 2> {
    let batch = rows.len();
    let n = rows.first().map_or(0, Vec::len);
    let flat: Vec<f32> = rows
        .iter()
        .flat_map(|r| {
            assert_eq!(r.len(), n, "every weight row must be the window length");
            r.iter().copied()
        })
        .collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch, n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_weights_normalize_split_exactly_and_draw_by_weight() {
        let w = MixWeights::new(&[3.0, 1.0]).unwrap();
        assert_eq!(w.as_slice(), &[0.75, 0.25]);
        assert_eq!(w.split(8), vec![6, 2]);
        assert_eq!(w.split(7), vec![5, 2], "the remainder goes to the larger fractional part");
        assert_eq!(w.split(1), vec![1, 0]);
        assert_eq!(MixWeights::uniform(3).split(10).iter().sum::<usize>(), 10);
        assert_eq!(MixWeights::parse("", 2).unwrap(), MixWeights::uniform(2));
        assert!(MixWeights::parse("1,0", 2).is_err(), "zero weights are refused");
        assert!(MixWeights::parse("1,2,3", 2).is_err());

        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(1);
        let mut counts = [0usize; 2];
        for _ in 0..4000 {
            counts[w.draw(&mut rng)] += 1;
        }
        let share = counts[0] as f64 / 4000.0;
        assert!((share - 0.75).abs() < 0.03, "{share}");
    }

    #[test]
    fn test_origin_and_source_stats() {
        let single = BatchOrigin::single(1, 3, 8);
        assert_eq!(single.sole_source(), Some(1));
        assert_eq!(single.total(), 8);
        let composite = BatchOrigin { counts: vec![6, 2, 0] };
        assert_eq!(composite.sole_source(), None);

        let mut stats = SourceStats::new(vec!["a".into(), "b".into(), "c".into()]);
        stats.record(&single, 2.0);
        stats.record(&composite, 4.0);
        assert_eq!(stats.batches, vec![1, 2, 0]);
        assert_eq!(stats.samples, vec![6, 10, 0]);
        assert!((stats.mean_loss(0) - 3.0).abs() < 1e-6, "6/8 of 4.0");
        assert!((stats.mean_loss(1) - (2.0 + 1.0) / 2.0).abs() < 1e-6);
        assert_eq!(stats.mean_loss(2), 0.0);
        assert!(stats.render().contains("mean loss"));
    }
}
