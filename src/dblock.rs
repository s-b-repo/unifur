//! Block-wise denoising classifier, ported from `ViTDBlockModel`.
//!
//! The class label embedding plays the role of the "clean data": during
//! training we noise it with per-sample sigmas sampled inside one block's
//! window and train the corresponding transformer block to recover it
//! (cross-entropy against the label, weighted by the EDM loss weighting).
//! At inference the latent embedding is integrated from pure noise down to
//! the smallest sigma with Euler steps across blocks.

use crate::{
    sigma::{
        discrete_sigmas_dblock, edm_loss_weight, estimate_target_layer, DblockSigmaSampler,
        EdmPreconditioning, SIGMA_MAX, SIGMA_MIN, P_MEAN, P_STD,
    },
    solver::SolverKind,
    vit::ViTDiTConfig,
    vit::ViTDiTForImageClassification,
};
use burn::{
    tensor::{activation::log_softmax, activation::softmax, backend::Backend, Distribution, Int, Tensor},
};
use rand::Rng;

/// Configuration of the block-wise denoising wrapper.
#[derive(Debug, Clone)]
pub struct DblockConfig {
    /// Number of blocks the transformer layers are partitioned into.
    pub num_blocks: usize,
    /// Sigma-range extension factor for training windows.
    pub gamma: f64,
    /// EDM `sigma_data`.
    pub sigma_data: f64,
    /// Number of Euler steps at inference (defaults to `num_blocks`).
    pub num_inference_steps: Option<usize>,
    /// Weight of the MoE load-balancing auxiliary loss, when the trunk has
    /// sparse layers. Ignored by a fully dense trunk.
    pub moe_aux_weight: f64,
}

impl Default for DblockConfig {
    fn default() -> Self {
        Self {
            num_blocks: 3,
            gamma: 0.05,
            sigma_data: 0.5,
            num_inference_steps: None,
            // Switch Transformer's default auxiliary weight: large enough to
            // prevent router collapse, small enough not to distort the task
            // loss.
            moe_aux_weight: 0.01,
        }
    }
}

/// Metrics returned by [`DblockClassifier::training_step`].
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub loss: f32,
    pub ce_loss: f32,
    pub block_idx: usize,
    /// MoE load-balancing auxiliary loss of the executed span; `0.0` for a
    /// dense trunk.
    pub balance_loss: f32,
}

/// The pieces of one training step, kept separate so a caller can reweight
/// them independently.
///
/// [`Self::loss`] already has the balance term folded in — that is the number a
/// plain run backpropagates. [`Self::balance`] is handed back *as well* because
/// a caller that reweights [`Self::per_sample`] must re-add it afterwards
/// rather than reweight it: the balance term is a router regularizer, not a
/// per-noise-level quantity, and scaling it by a sigma-indexed weight would tie
/// load balancing to whichever noise levels a batch happened to draw.
#[derive(Debug)]
pub struct StepParts<B: Backend> {
    /// Scalar loss, balance term included.
    pub loss: Tensor<B, 1>,
    /// Per-sample weighted losses `[b]`, before the mean and before the
    /// balance term.
    pub per_sample: Tensor<B, 1>,
    /// The executed span's load-balancing loss; `None` for a dense trunk.
    pub balance: Option<Tensor<B, 1>>,
    pub metrics: StepMetrics,
}

/// DiffusionBlocks classifier.
#[derive(burn::module::Module, Debug)]
pub struct DblockClassifier<B: Backend> {
    model: ViTDiTForImageClassification<B>,
    num_blocks: usize,
    sigma_data: f64,
    moe_aux_weight: f64,
    layer_split: usize,
    inference_sigmas: Vec<f64>,
}

impl<B: Backend<FloatElem = f32>> DblockClassifier<B> {
    /// Build the model with DiT initialization and precompute schedules.
    pub fn new(
        vit_config: &ViTDiTConfig,
        dblock_config: &DblockConfig,
        device: &B::Device,
    ) -> Self {
        assert!(
            vit_config.num_hidden_layers % dblock_config.num_blocks == 0,
            "num_hidden_layers ({}) must be divisible by num_blocks ({})",
            vit_config.num_hidden_layers,
            dblock_config.num_blocks
        );
        assert!(
            dblock_config.num_blocks >= 1,
            "need at least one block"
        );

        let steps = dblock_config
            .num_inference_steps
            .unwrap_or(dblock_config.num_blocks);
        let inference_sigmas =
            discrete_sigmas_dblock(steps, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD);

        let model = ViTDiTForImageClassification::new(vit_config, device)
            .with_dit_init(vit_config, device);

        Self {
            model,
            num_blocks: dblock_config.num_blocks,
            sigma_data: dblock_config.sigma_data,
            moe_aux_weight: dblock_config.moe_aux_weight,
            layer_split: vit_config.num_hidden_layers / dblock_config.num_blocks,
            inference_sigmas,
        }
    }

    pub fn sampler(&self, gamma: f64) -> DblockSigmaSampler {
        DblockSigmaSampler::new(self.num_blocks, gamma)
    }

    /// Contiguous transformer-layer window owned by `block_idx`.
    pub fn layer_range(&self, block_idx: usize) -> std::ops::Range<usize> {
        assert!(block_idx < self.num_blocks);
        let start = block_idx * self.layer_split;
        start..start + self.layer_split
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// EDM `sigma_data` used by the preconditioning.
    pub fn sigma_data(&self) -> f64 {
        self.sigma_data
    }

    /// Weight applied to the MoE load-balancing auxiliary loss.
    pub fn moe_aux_weight(&self) -> f64 {
        self.moe_aux_weight
    }

    /// Discrete (descending) inference schedule.
    pub fn inference_sigmas(&self) -> &[f64] {
        &self.inference_sigmas
    }

    /// EDM-preconditioned denoiser -> class logits.
    ///
    /// `sigmas` holds one sigma per sample; when `block_idx` is `None` it is
    /// inferred from the sigmas by majority vote (`estimate_target_layer`).
    pub fn denoise(
        &self,
        pixel_values: Tensor<B, 4>,
        zt: Tensor<B, 2>,
        sigmas: &[f64],
        block_idx: Option<usize>,
    ) -> Tensor<B, 2> {
        let block_idx = block_idx.unwrap_or_else(|| {
            estimate_target_layer(&self.block_bounds(), sigmas)
        });
        self.denoise_span(pixel_values, zt, sigmas, self.layer_range(block_idx))
    }

    /// [`Self::denoise`] over an arbitrary contiguous transformer-layer
    /// window (`start..end`). This is the primitive behind parallel /
    /// multi-block inference strategies; running only the span also acts as
    /// gradient routing during training, since gradients flow exclusively
    /// through executed layers.
    pub fn denoise_span(
        &self,
        pixel_values: Tensor<B, 4>,
        zt: Tensor<B, 2>,
        sigmas: &[f64],
        span: std::ops::Range<usize>,
    ) -> Tensor<B, 2> {
        self.denoise_span_with_aux(pixel_values, zt, sigmas, span).0
    }

    /// [`Self::denoise_span`] that also returns the MoE load-balancing loss of
    /// the executed span (`None` for a dense trunk).
    ///
    /// Inference paths discard it; training paths must add it, or a sparse
    /// router is free to collapse onto a single expert.
    pub fn denoise_span_with_aux(
        &self,
        pixel_values: Tensor<B, 4>,
        zt: Tensor<B, 2>,
        sigmas: &[f64],
        span: std::ops::Range<usize>,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 1>>) {
        let device = zt.device();
        let b = zt.dims()[0];
        assert_eq!(sigmas.len(), b, "one sigma per sample required");

        // Per-sample EDM preconditioning coefficients.
        let s = Tensor::<B, 1>::from_floats(
            sigmas.iter().map(|&v| v as f32).collect::<Vec<_>>().as_slice(),
            &device,
        );
        let sd = self.sigma_data;
        let s2 = s.clone().powf_scalar(2.0);
        let denom = (s2.clone() + sd * sd).sqrt();
        let c_skip = (sd * sd) / (s2.clone() + sd * sd);
        let c_out = s.clone() * sd / denom.clone();
        let c_in = denom.recip();
        let c_noise = s.clone().log().mul_scalar(0.25);

        let zt_scaled = zt.clone() * c_in.unsqueeze_dim::<2>(1);
        let out =
            self.model
                .vit()
                .forward_block(span, pixel_values, zt_scaled, c_noise);

        // model_out = hidden * c_out + zt * c_skip, on the CLS/noisy token.
        let pooled = out.last_hidden_state.narrow(1, 0, 1); // [b, 1, h]
        let model_out = pooled * c_out.unsqueeze_dim::<3>(1) + zt.unsqueeze_dim::<3>(1) * c_skip.unsqueeze_dim::<3>(1);

        (
            self.model.forward_output_embeddings(model_out, out.conditioning),
            out.balance_loss,
        )
    }

    /// Denoised "clean" embedding estimate at scalar `sigma`: class
    /// probabilities projected through the label-embedding table.
    ///
    /// `span` selects the contiguous layer window to run (see
    /// [`Self::denoise_span`]); `None` infers the block from `sigma`.
    pub fn x0_estimate(
        &self,
        pixel_values: &Tensor<B, 4>,
        z: &Tensor<B, 2>,
        sigma: f64,
        span: Option<std::ops::Range<usize>>,
    ) -> Tensor<B, 2> {
        self.x0_estimate_probs(pixel_values, z, sigma, span, false).0
    }

    /// [`Self::x0_estimate`] that can additionally return the class
    /// probabilities feeding the estimate (used by confidence-based
    /// strategies).
    pub fn x0_estimate_probs(
        &self,
        pixel_values: &Tensor<B, 4>,
        z: &Tensor<B, 2>,
        sigma: f64,
        span: Option<std::ops::Range<usize>>,
        with_probs: bool,
    ) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
        let b = z.dims()[0];
        let logits = match span {
            Some(span) => {
                self.denoise_span(pixel_values.clone(), z.clone(), &[sigma].repeat(b), span)
            }
            None => self.denoise(pixel_values.clone(), z.clone(), &vec![sigma; b], None),
        };
        let probs = softmax(logits, 1);
        let probs_out = if with_probs { Some(probs.clone()) } else { None };
        (probs.matmul(self.model.label_embedding_weight()), probs_out)
    }

    /// x0 estimate with the image conditioning removed.
    ///
    /// The unconditional half of a guided pair (roadmap 22.5). "Unconditional"
    /// here means a zero image rather than a learned null embedding: this model
    /// has no null token to fall back on, and zeros are the input the patch
    /// embedding maps to its own bias — the closest thing to "no evidence" that
    /// exists without retraining. A learned null embedding would be stronger
    /// and is the natural upgrade if guidance proves worth training for.
    pub fn x0_estimate_unconditional(
        &self,
        pixel_values: &Tensor<B, 4>,
        z: &Tensor<B, 2>,
        sigma: f64,
        span: Option<std::ops::Range<usize>>,
    ) -> Tensor<B, 2> {
        let null = Tensor::<B, 4>::zeros(pixel_values.dims(), &pixel_values.device());
        self.x0_estimate(&null, z, sigma, span)
    }

    /// Solve the full probability-flow ODE from pure noise down to
    /// `sigma_min` with an arbitrary [`SolverKind`] and a fixed block-span
    /// policy (`None` = infer per-sigma majority block).
    pub fn solve(
        &self,
        pixel_values: &Tensor<B, 4>,
        kind: SolverKind,
        num_steps: usize,
        span: Option<std::ops::Range<usize>>,
        rng: &mut impl Rng,
    ) -> Tensor<B, 2> {
        let b = pixel_values.dims()[0];
        let h_dim = self.model.label_embedding_weight().dims()[1];

        let schedule =
            crate::sigma::discrete_sigmas_dblock(num_steps, SIGMA_MIN, SIGMA_MAX, P_MEAN, P_STD);
        let s0 = schedule[0];
        let initial = Tensor::<B, 2>::random(
            [b, h_dim],
            Distribution::Normal(0.0, 1.0),
            &pixel_values.device(),
        )
        .mul_scalar((1.0 + s0 * s0).sqrt() as f32);

        let z_end = crate::solver::integrate(
            initial,
            &schedule,
            |sigma, z| self.x0_estimate(pixel_values, z, sigma, span.clone()),
            kind,
            rng,
        );

        let min_sigma = *schedule.last().expect("non-empty schedule");
        self.denoise(pixel_values.clone(), z_end, &vec![min_sigma; b], None)
    }

    pub(crate) fn block_bounds(&self) -> Vec<f64> {
        self.sampler(0.0).block_sigmas
    }

    /// One block-wise training step (`shared_step` in the reference).
    ///
    /// Picks a random block, samples sigmas in its (extended) window, noises
    /// the label embeddings and computes the EDM-weighted cross-entropy loss
    /// of the target block's denoising prediction.
    pub fn training_step<R: Rng>(
        &self,
        pixel_values: Tensor<B, 4>,
        labels: Tensor<B, 1, Int>,
        gamma: f64,
        rng: &mut R,
    ) -> (Tensor<B, 1>, StepMetrics) {
        let b = pixel_values.dims()[0];
        let block_idx = rng.random_range(0..self.num_blocks);
        let sigmas = self.sampler(gamma).sample(rng, block_idx, b);
        let parts = self.training_step_on(pixel_values, labels, &sigmas, block_idx, None);
        (parts.loss, parts.metrics)
    }

    /// [`Self::training_step`] with the block and noise levels chosen by the
    /// caller, and optional per-sample importance weights (roadmap 20.6).
    ///
    /// Splitting this out is what lets [`crate::reweight::SigmaImportanceSampler`]
    /// choose the sigmas without the sampler having to know anything about the
    /// model, and what lets a caller reuse one set of noise levels across
    /// objectives.
    ///
    /// Returns the loss, the metrics, the **per-sample** weighted losses `[b]`
    /// and the balance term separately — see [`StepParts`], and the note there
    /// on why the balance loss is handed back rather than only folded in.
    pub fn training_step_on(
        &self,
        pixel_values: Tensor<B, 4>,
        labels: Tensor<B, 1, Int>,
        sigmas: &[f64],
        block_idx: usize,
        importance: Option<&[f64]>,
    ) -> StepParts<B> {
        let device = pixel_values.device();
        let b = pixel_values.dims()[0];
        assert_eq!(sigmas.len(), b, "one sigma per sample required");
        assert!(block_idx < self.num_blocks, "block {block_idx} out of range");

        // Clean "data": normalized label embeddings.
        let z = self.model.normalized_label_embeds(labels.clone());

        // zt = z + sigma * eps
        let eps = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
        let s = Tensor::<B, 1>::from_floats(
            sigmas.iter().map(|&v| v as f32).collect::<Vec<_>>().as_slice(),
            &device,
        );
        let zt = z + eps * s.unsqueeze_dim::<2>(1);

        let (logits, balance) =
            self.denoise_span_with_aux(pixel_values, zt, sigmas, self.layer_range(block_idx));

        // Per-sample cross entropy.
        let log_probs = log_softmax(logits, 1);
        let nll = -log_probs
            .gather(1, labels.unsqueeze_dim::<2>(1))
            .squeeze_dim::<1>(1); // [b]

        let ce_loss = nll.clone().mean();

        // EDM loss weighting per sample.
        let weights: Vec<f32> = sigmas
            .iter()
            .map(|&sg| edm_loss_weight(sg, self.sigma_data) as f32)
            .collect();
        let w = Tensor::<B, 1>::from_floats(weights.as_slice(), &device);
        let mut per_sample = nll * w;

        // Importance weights, when the noise levels were not drawn from the
        // prior. `p/q` is what keeps the estimator unbiased -- without it a
        // proposal that favours the high-loss region would report a
        // systematically larger loss and the optimizer would chase it.
        if let Some(importance) = importance {
            assert_eq!(importance.len(), b, "one importance weight per sample required");
            let iw = Tensor::<B, 1>::from_floats(
                importance.iter().map(|&v| v as f32).collect::<Vec<_>>().as_slice(),
                &device,
            );
            per_sample = per_sample * iw;
        }

        let mut loss = per_sample.clone().mean();

        let mut m_balance = 0.0f32;
        if let Some(aux) = balance.clone() {
            m_balance = aux.clone().into_scalar();
            loss = loss + aux.mul_scalar(self.moe_aux_weight as f32);
        }

        let metrics = StepMetrics {
            loss: loss.clone().into_scalar(),
            ce_loss: ce_loss.into_scalar(),
            block_idx,
            balance_loss: m_balance,
        };
        StepParts { loss, metrics, per_sample, balance }
    }

    /// Euler-integrated classification (`diffusion_step`): integrate the
    /// latent class embedding from `sigma_max` down to `sigma_min` across
    /// blocks, then return the final logits.
    ///
    /// Must be called with a plain (non-autodiff) backend tensor set.
    pub fn diffusion_step(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 2> {
        let device = pixel_values.device();
        let b = pixel_values.dims()[0];
        let h = self.model.label_embedding_weight().dims()[1];

        // Start from N(0, I) scaled to sqrt(1 + sigma_0^2).
        let sigma0 = self.inference_sigmas[0];
        let mut z = Tensor::<B, 2>::random([b, h], Distribution::Normal(0.0, 1.0), &device)
            .mul_scalar((1.0 + sigma0 * sigma0).sqrt());

        let w = self.model.label_embedding_weight(); // [V, H]

        for window in self.inference_sigmas.windows(2) {
            let sigma = window[0];
            let next_sigma = window[1];

            let logits = self.denoise(
                pixel_values.clone(),
                z.clone(),
                &vec![sigma; b],
                None,
            );
            let probs = softmax(logits, 1);
            // Denoised embedding estimate: probabilities over the vocab
            // projected through the label embedding table.
            let denoised = probs.matmul(w.clone()); // [b, H]

            // Euler step: z <- z + (sigma_next - sigma) * (z - x0) / sigma
            let d = (z.clone() - denoised) / sigma;
            z = z + (next_sigma - sigma) * d;
        }

        let min_sigma = *self.inference_sigmas.last().expect("non-empty schedule");
        self.denoise(pixel_values, z, &vec![min_sigma; b], None)
    }

    /// Access to the underlying ViT-DiT model.
    pub fn model(&self) -> &ViTDiTForImageClassification<B> {
        &self.model
    }
}

/// Convenience: build preconditioning for a scalar sigma (host side).
pub fn precondition(sigma: f64, sigma_data: f64) -> EdmPreconditioning {
    EdmPreconditioning::new(sigma, sigma_data)
}
