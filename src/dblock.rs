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
}

impl Default for DblockConfig {
    fn default() -> Self {
        Self {
            num_blocks: 3,
            gamma: 0.05,
            sigma_data: 0.5,
            num_inference_steps: None,
        }
    }
}

/// Metrics returned by [`DblockClassifier::training_step`].
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub loss: f32,
    pub ce_loss: f32,
    pub block_idx: usize,
}

/// DiffusionBlocks classifier.
#[derive(burn::module::Module, Debug)]
pub struct DblockClassifier<B: Backend> {
    model: ViTDiTForImageClassification<B>,
    num_blocks: usize,
    sigma_data: f64,
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

    /// [`Denoiser::denoise`] over an arbitrary contiguous transformer-layer
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

        self.model.forward_output_embeddings(model_out, out.conditioning)
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
        let device = pixel_values.device();
        let b = pixel_values.dims()[0];

        // Clean "data": normalized label embeddings.
        let z = self.model.normalized_label_embeds(labels.clone());

        // Random block + per-sample sigmas within its window.
        let block_idx = rng.random_range(0..self.num_blocks);
        let sigmas = self.sampler(gamma).sample(rng, block_idx, b);

        // zt = z + sigma * eps
        let eps = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
        let s = Tensor::<B, 1>::from_floats(
            sigmas.iter().map(|&v| v as f32).collect::<Vec<_>>().as_slice(),
            &device,
        );
        let zt = z + eps * s.unsqueeze_dim::<2>(1);

        let logits = self.denoise(pixel_values, zt, &sigmas, Some(block_idx));

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
        let loss = (nll * w).mean();

        let metrics = StepMetrics {
            loss: loss.clone().into_scalar(),
            ce_loss: ce_loss.into_scalar(),
            block_idx,
        };
        (loss, metrics)
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
