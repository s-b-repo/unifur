//! Small, fully known stand-ins for the codec and the level models.
//!
//! The `ncm-decode` certificates run inside `dblocks verify`, i.e. in a
//! normal (non-test) build, so the doubles they are stated against live here
//! rather than in a test module. Everything is host arithmetic on seeded
//! values, so every distribution a certificate compares against can be
//! computed exactly.

use std::cell::Cell;
use std::ops::Range;

use anyhow::{anyhow, ensure, Context, Result};
use burn::tensor::{backend::Backend, Int, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::ncm::{ConceptCodec, LevelCodes, LevelModel, LevelSchedule};

/// The positions of a length-`seq_len` sequence that level segment `seg`
/// (of `len`) covers: those with `⌊pos · len / seq_len⌋ = seg`.
fn segment_span(seg: usize, len: usize, seq_len: usize) -> Range<usize> {
    (seg * seq_len).div_ceil(len)..((seg + 1) * seq_len).div_ceil(len)
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (a, b) in dst.iter_mut().zip(src) {
        *a += b;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn uniform_vec(rng: &mut ChaCha8Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| rng.random_range(-1.0f32..1.0) * scale)
        .collect()
}

fn host_ids<B: Backend>(tensor: Tensor<B, 2, Int>, bound: usize, what: &str) -> Result<Vec<usize>> {
    tensor
        .into_data()
        .convert::<i64>()
        .iter::<i64>()
        .map(|v| {
            usize::try_from(v)
                .ok()
                .filter(|&v| v < bound)
                .ok_or_else(|| anyhow!("{what} holds {v}, outside 0..{bound}"))
        })
        .collect()
}

fn ints<B: Backend>(
    values: &[usize],
    shape: [usize; 2],
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>> {
    let flat = values
        .iter()
        .map(|&v| i64::try_from(v).with_context(|| format!("{v} does not fit an i64")))
        .collect::<Result<Vec<i64>>>()?;
    Ok(Tensor::<B, 1, Int>::from_ints(flat.as_slice(), device).reshape(shape))
}

/// A residual-VQ codec in miniature: tokens embed to fixed random vectors,
/// level `k` average-pools the residual over its segments and snaps it to
/// the nearest row of a fixed random codebook (scale `1/(k+1)`), the
/// reconstruction upsamples the chosen rows back, and the decoder's logits
/// are `sharpness · ⟨ẑ, embedding⟩`.
pub(super) struct ToyCodec<B: Backend> {
    schedule: LevelSchedule,
    codebook: usize,
    dim: usize,
    vocab: usize,
    codebooks: Vec<Vec<f32>>,
    embed: Vec<f32>,
    sharpness: f32,
    device: B::Device,
}

impl<B: Backend> ToyCodec<B> {
    pub(super) fn new(
        lengths: Vec<usize>,
        codebook: usize,
        dim: usize,
        vocab: usize,
        sharpness: f32,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self> {
        let schedule = LevelSchedule::new(lengths)?;
        ensure!(
            codebook >= 1 && dim >= 1 && vocab >= 1,
            "toy codec needs codebook, dim and vocab >= 1, got {codebook}, {dim}, {vocab}"
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let codebooks = (0..schedule.levels())
            .map(|k| uniform_vec(&mut rng, codebook * dim, 1.0 / (1.0 + k as f32)))
            .collect();
        let embed = uniform_vec(&mut rng, vocab * dim, 1.0);
        Ok(Self {
            schedule,
            codebook,
            dim,
            vocab,
            codebooks,
            embed,
            sharpness,
            device: device.clone(),
        })
    }

    fn book(&self, level: usize) -> Result<&[f32]> {
        self.codebooks
            .get(level)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow!("toy codec has no level {level}"))
    }

    fn row<'v>(&self, table: &'v [f32], index: usize) -> Result<&'v [f32]> {
        table
            .get(index * self.dim..(index + 1) * self.dim)
            .ok_or_else(|| anyhow!("toy codec has no row {index}"))
    }
}

impl<B: Backend> ConceptCodec<B> for ToyCodec<B> {
    fn schedule(&self) -> &LevelSchedule {
        &self.schedule
    }

    fn codebook_size(&self) -> usize {
        self.codebook
    }

    fn latent_dim(&self) -> usize {
        self.dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn encode(&self, tokens: Tensor<B, 2, Int>) -> Result<LevelCodes<B>> {
        let [rows, seq_len] = tokens.dims();
        ensure!(
            seq_len == self.schedule.seq_len(),
            "toy codec: {seq_len} tokens, schedule wants {}",
            self.schedule.seq_len()
        );
        let ids = host_ids(tokens, self.vocab, "toy codec tokens")?;
        let d = self.dim;
        // The residual starts as the embedding and loses each chosen code.
        let mut residual = Vec::with_capacity(ids.len() * d);
        for &id in &ids {
            residual.extend_from_slice(self.row(&self.embed, id)?);
        }
        let mut codes = Vec::with_capacity(self.schedule.levels());
        for (level, &len) in self.schedule.lengths().iter().enumerate() {
            let book = self.book(level)?;
            let mut chosen = Vec::with_capacity(rows * len);
            for row in residual.chunks_exact_mut(seq_len * d) {
                for seg in 0..len {
                    let span = segment_span(seg, len, seq_len);
                    let count = span.len().max(1) as f32;
                    let block = row
                        .get_mut(span.start * d..span.end * d)
                        .ok_or_else(|| anyhow!("toy codec: segment {seg} outside the sequence"))?;
                    let mut mean = vec![0.0f32; d];
                    for position in block.chunks_exact(d) {
                        add_into(&mut mean, position);
                    }
                    for m in &mut mean {
                        *m /= count;
                    }
                    let mut best = 0usize;
                    let mut best_distance = f32::INFINITY;
                    for (code, entry) in book.chunks_exact(d).enumerate() {
                        let distance: f32 = entry
                            .iter()
                            .zip(&mean)
                            .map(|(e, m)| (e - m) * (e - m))
                            .sum();
                        if distance < best_distance {
                            best = code;
                            best_distance = distance;
                        }
                    }
                    let entry = self.row(book, best)?;
                    for position in block.chunks_exact_mut(d) {
                        for (p, e) in position.iter_mut().zip(entry) {
                            *p -= e;
                        }
                    }
                    chosen.push(best);
                }
            }
            codes.push(ints::<B>(&chosen, [rows, len], &self.device)?);
        }
        Ok(LevelCodes { codes })
    }

    fn reconstruct(&self, codes: &[Tensor<B, 2, Int>], batch: usize) -> Result<Tensor<B, 3>> {
        ensure!(
            codes.len() <= self.schedule.levels(),
            "toy codec: {} levels of codes for a {}-level schedule",
            codes.len(),
            self.schedule.levels()
        );
        let (seq_len, d) = (self.schedule.seq_len(), self.dim);
        let mut zhat = vec![0.0f32; batch * seq_len * d];
        for (level, (tensor, &len)) in codes.iter().zip(self.schedule.lengths()).enumerate() {
            let dims = tensor.dims();
            ensure!(
                dims == [batch, len],
                "toy codec: level {level} codes have shape {dims:?}, want [{batch}, {len}]"
            );
            let ids = host_ids(tensor.clone(), self.codebook, "toy codec codes")?;
            let book = self.book(level)?;
            for (row_z, row_codes) in zhat
                .chunks_exact_mut(seq_len * d)
                .zip(ids.chunks_exact(len))
            {
                for (position, z) in row_z.chunks_exact_mut(d).enumerate() {
                    let code = row_codes
                        .get(position * len / seq_len)
                        .copied()
                        .ok_or_else(|| anyhow!("toy codec: no code for position {position}"))?;
                    add_into(z, self.row(book, code)?);
                }
            }
        }
        Ok(Tensor::<B, 1>::from_floats(zhat.as_slice(), &self.device).reshape([batch, seq_len, d]))
    }

    fn decode_logits(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [rows, seq_len, d] = zhat.dims();
        ensure!(
            d == self.dim,
            "toy codec: latent width {d}, want {}",
            self.dim
        );
        let z: Vec<f32> = zhat.into_data().convert::<f32>().iter::<f32>().collect();
        let mut logits = Vec::with_capacity(rows * seq_len * self.vocab);
        for position in z.chunks_exact(d) {
            for token in self.embed.chunks_exact(d) {
                logits.push(self.sharpness * dot(position, token));
            }
        }
        Ok(Tensor::<B, 1>::from_floats(logits.as_slice(), &self.device)
            .reshape([rows, seq_len, self.vocab]))
    }
}

/// A level model with known logits: for segment `s` of level `k`,
/// `bias[k][s] + gain · mix[k] · mean(ẑ over s) + prompt term`. With
/// `gain = 0` every level's distribution is fixed; with `gain > 0` each level
/// depends on the coarser codes through `ẑ`, which is what makes level
/// speculation's prefix rule matter.
pub(super) struct ToyModel<B: Backend> {
    lengths: Vec<usize>,
    seq_len: usize,
    codebook: usize,
    dim: usize,
    bias: Vec<Vec<f32>>,
    mix: Vec<Vec<f32>>,
    gain: f32,
    prompt_gain: f32,
    batched: bool,
    device: B::Device,
}

impl<B: Backend> ToyModel<B> {
    pub(super) fn random(
        lengths: Vec<usize>,
        codebook: usize,
        dim: usize,
        gain: f32,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self> {
        let schedule = LevelSchedule::new(lengths)?;
        ensure!(
            codebook >= 1 && dim >= 1,
            "toy model needs codebook and dim >= 1, got {codebook}, {dim}"
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let bias = schedule
            .lengths()
            .iter()
            .map(|&len| uniform_vec(&mut rng, len * codebook, 2.0))
            .collect();
        let mix = schedule
            .lengths()
            .iter()
            .map(|_| uniform_vec(&mut rng, codebook * dim, 1.0))
            .collect();
        Ok(Self {
            seq_len: schedule.seq_len(),
            lengths: schedule.lengths().to_vec(),
            codebook,
            dim,
            bias,
            mix,
            gain,
            prompt_gain: 0.5,
            batched: false,
            device: device.clone(),
        })
    }

    /// The same model with uniform noise of half-width `scale` added to every
    /// bias and mixing weight: a draft that agrees with the target often but
    /// not always.
    pub(super) fn perturbed(&self, scale: f32, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut shake = |table: &[Vec<f32>]| -> Vec<Vec<f32>> {
            table
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| v + rng.random_range(-1.0f32..1.0) * scale)
                        .collect()
                })
                .collect()
        };
        let bias = shake(&self.bias);
        let mix = shake(&self.mix);
        Self {
            lengths: self.lengths.clone(),
            seq_len: self.seq_len,
            codebook: self.codebook,
            dim: self.dim,
            bias,
            mix,
            gain: self.gain,
            prompt_gain: self.prompt_gain,
            batched: self.batched,
            device: self.device.clone(),
        }
    }

    /// Whether this model claims [`LevelModel::batches_levels`].
    pub(super) fn batching(mut self, batched: bool) -> Self {
        self.batched = batched;
        self
    }
}

impl<B: Backend> LevelModel<B> for ToyModel<B> {
    fn level_logits(
        &self,
        level: usize,
        zhat_coarser: Tensor<B, 3>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 3>> {
        let len = self
            .lengths
            .get(level)
            .copied()
            .ok_or_else(|| anyhow!("toy model has no level {level}"))?;
        let [rows, seq_len, d] = zhat_coarser.dims();
        ensure!(
            seq_len == self.seq_len && d == self.dim,
            "toy model: ẑ has shape [{rows}, {seq_len}, {d}], want [_, {}, {}]",
            self.seq_len,
            self.dim
        );
        let prompt_sums: Vec<i64> = match prompt {
            Some(p) => {
                let [prompt_rows, width] = p.dims();
                ensure!(
                    prompt_rows == rows,
                    "toy model: prompt has {prompt_rows} rows, ẑ has {rows}"
                );
                let flat: Vec<i64> = p.into_data().convert::<i64>().iter::<i64>().collect();
                flat.chunks(width.max(1))
                    .map(|r| r.iter().sum())
                    .chain(std::iter::repeat(0))
                    .take(rows)
                    .collect()
            }
            None => vec![0; rows],
        };
        let z: Vec<f32> = zhat_coarser
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .collect();
        let bias = self
            .bias
            .get(level)
            .ok_or_else(|| anyhow!("toy model has no bias for level {level}"))?;
        let mix = self
            .mix
            .get(level)
            .ok_or_else(|| anyhow!("toy model has no mix for level {level}"))?;
        let mut logits = Vec::with_capacity(rows * len * self.codebook);
        for (row_z, &prompt_sum) in z.chunks_exact(seq_len * d).zip(&prompt_sums) {
            for (seg, seg_bias) in bias.chunks_exact(self.codebook).enumerate() {
                let span = segment_span(seg, len, seq_len);
                let count = span.len().max(1) as f32;
                let mut pooled = vec![0.0f32; d];
                for position in row_z
                    .get(span.start * d..span.end * d)
                    .unwrap_or(&[])
                    .chunks_exact(d)
                {
                    add_into(&mut pooled, position);
                }
                for (code, (b, weights)) in seg_bias.iter().zip(mix.chunks_exact(d)).enumerate() {
                    let prompt_term =
                        (prompt_sum * 31 + code as i64 * 7).rem_euclid(11) as f32 / 11.0;
                    logits.push(
                        b + self.gain * dot(weights, &pooled) / count
                            + self.prompt_gain * prompt_term,
                    );
                }
            }
        }
        Ok(
            Tensor::<B, 1>::from_floats(logits.as_slice(), &self.device).reshape([
                rows,
                len,
                self.codebook,
            ]),
        )
    }

    fn batches_levels(&self) -> bool {
        self.batched
    }
}

/// Counts the forward passes a level model is asked for, independently of
/// what the decoder reports. A `levels_logits` call on a model that
/// [`LevelModel::batches_levels`] is one pass; otherwise each level is one.
pub(super) struct CountingModel<'m, B: Backend> {
    inner: &'m dyn LevelModel<B>,
    single: Cell<usize>,
    multi: Cell<usize>,
    levels: Cell<usize>,
}

impl<'m, B: Backend> CountingModel<'m, B> {
    pub(super) fn new(inner: &'m dyn LevelModel<B>) -> Self {
        Self {
            inner,
            single: Cell::new(0),
            multi: Cell::new(0),
            levels: Cell::new(0),
        }
    }

    /// Forward passes made.
    pub(super) fn passes(&self) -> usize {
        self.single.get() + self.multi.get()
    }

    /// Of those, multi-level passes.
    pub(super) fn multi_level_passes(&self) -> usize {
        self.multi.get()
    }

    /// Levels evaluated in total.
    pub(super) fn levels(&self) -> usize {
        self.levels.get()
    }
}

impl<B: Backend> LevelModel<B> for CountingModel<'_, B> {
    fn level_logits(
        &self,
        level: usize,
        zhat_coarser: Tensor<B, 3>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Tensor<B, 3>> {
        self.single.set(self.single.get() + 1);
        self.levels.set(self.levels.get() + 1);
        self.inner.level_logits(level, zhat_coarser, prompt)
    }

    fn levels_logits(
        &self,
        requests: Vec<(usize, Tensor<B, 3>)>,
        prompt: Option<Tensor<B, 2, Int>>,
    ) -> Result<Vec<Tensor<B, 3>>> {
        if self.inner.batches_levels() {
            self.multi.set(self.multi.get() + 1);
            self.levels.set(self.levels.get() + requests.len());
            requests
                .into_iter()
                .map(|(level, zhat)| self.inner.level_logits(level, zhat, prompt.clone()))
                .collect()
        } else {
            requests
                .into_iter()
                .map(|(level, zhat)| self.level_logits(level, zhat, prompt.clone()))
                .collect()
        }
    }

    fn batches_levels(&self) -> bool {
        self.inner.batches_levels()
    }
}

/// Counts codec calls independently of the decoder's report.
pub(super) struct CountingCodec<'c, B: Backend> {
    inner: &'c dyn ConceptCodec<B>,
    reconstructs: Cell<usize>,
    decodes: Cell<usize>,
    encodes: Cell<usize>,
}

impl<'c, B: Backend> CountingCodec<'c, B> {
    pub(super) fn new(inner: &'c dyn ConceptCodec<B>) -> Self {
        Self {
            inner,
            reconstructs: Cell::new(0),
            decodes: Cell::new(0),
            encodes: Cell::new(0),
        }
    }

    /// `(reconstructs, decodes, encodes)` so far.
    pub(super) fn counts(&self) -> (usize, usize, usize) {
        (
            self.reconstructs.get(),
            self.decodes.get(),
            self.encodes.get(),
        )
    }
}

impl<B: Backend> ConceptCodec<B> for CountingCodec<'_, B> {
    fn schedule(&self) -> &LevelSchedule {
        self.inner.schedule()
    }

    fn codebook_size(&self) -> usize {
        self.inner.codebook_size()
    }

    fn latent_dim(&self) -> usize {
        self.inner.latent_dim()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn encode(&self, tokens: Tensor<B, 2, Int>) -> Result<LevelCodes<B>> {
        self.encodes.set(self.encodes.get() + 1);
        self.inner.encode(tokens)
    }

    fn reconstruct(&self, codes: &[Tensor<B, 2, Int>], batch: usize) -> Result<Tensor<B, 3>> {
        self.reconstructs.set(self.reconstructs.get() + 1);
        self.inner.reconstruct(codes, batch)
    }

    fn decode_logits(&self, zhat: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        self.decodes.set(self.decodes.get() + 1);
        self.inner.decode_logits(zhat)
    }
}
