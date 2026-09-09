//! Error-propagation audit of a block-wise model (roadmap Phase 28; issue #1
//! §7).
//!
//! Block-wise training scores each block on its own window and never on what
//! it does to the blocks after it. This harness measures, on a batch, the
//! four numbers that decide whether a locally good block can still damage the
//! whole chain:
//!
//! - **local loss**: the block's own training objective at a sigma inside its
//!   window;
//! - **boundary mismatch**: how far this block's `x0` estimate is from the
//!   next block's at the sigma they share;
//! - **sensitivity**: a finite-difference proxy for the block map's Lipschitz
//!   constant, `‖H(z + ε) − H(z)‖ / ‖ε‖`, which is what Proposition 5 in
//!   `docs/Mathematical-Foundation.md` assumes and never measured;
//! - **downstream amplification**: the product of the sensitivities of every
//!   later block -- how much an error made here has grown by the output.
//!
//! The block map `H` is one Euler step of the EDM ODE across the block's
//! window using that block's `x0` estimate, which is exactly what the
//! sequential sampler does with one step per block. On random weights the
//! numbers describe the initialization; on a checkpoint they describe the
//! model, and that is the run the roadmap leaves GPU-blocked.

use burn::tensor::{activation::log_softmax, backend::Backend, Distribution, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::dblock::DblockClassifier;
use crate::sigma::DblockSigmaSampler;

/// One block's row of the audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockReport {
    pub block: usize,
    /// The window in inference order: from `sigma_hi` down to `sigma_lo`.
    pub sigma_hi: f64,
    pub sigma_lo: f64,
    /// EDM-weighted training loss and plain cross-entropy at the window's
    /// geometric midpoint.
    pub local_loss: f32,
    pub local_ce: f32,
    /// Mean squared distance between this block's and the next block's `x0`
    /// estimate at `sigma_lo`, on the same latent. `None` for the last block.
    pub boundary_mismatch: Option<f32>,
    /// `‖H(z + ε) − H(z)‖ / ‖ε‖` for the block map `H`.
    pub sensitivity: f32,
    /// Product of the sensitivities of every later block; 1 for the last.
    pub downstream_amplification: f32,
}

/// The whole audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropagationReport {
    pub blocks: Vec<BlockReport>,
    /// Cross-entropy and accuracy of the model's own inference path on the batch.
    pub end_to_end_ce: f32,
    pub end_to_end_accuracy: f32,
    pub batch_size: usize,
    pub epsilon: f64,
}

impl PropagationReport {
    pub fn render(&self) -> String {
        let mut out = format!(
            "{:<6} {:>10} {:>10} {:>11} {:>10} {:>12} {:>12} {:>14}\n",
            "block", "sigma hi", "sigma lo", "local loss", "local ce", "boundary", "sensitivity", "amplification"
        );
        out.push_str(&"-".repeat(92));
        out.push('\n');
        for b in &self.blocks {
            out.push_str(&format!(
                "{:<6} {:>10.4} {:>10.4} {:>11.4} {:>10.4} {:>12} {:>12.4} {:>14.4}\n",
                b.block,
                b.sigma_hi,
                b.sigma_lo,
                b.local_loss,
                b.local_ce,
                b.boundary_mismatch.map_or("-".to_string(), |v| format!("{v:.4e}")),
                b.sensitivity,
                b.downstream_amplification
            ));
        }
        out.push_str(&format!(
            "\nend to end: cross-entropy {:.4}, accuracy {:.1}% on {} samples (perturbation {:.1e})\n",
            self.end_to_end_ce,
            100.0 * self.end_to_end_accuracy,
            self.batch_size,
            self.epsilon
        ));
        out
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        serde_json::to_string_pretty(self).map_err(|err| anyhow::anyhow!("serialize audit: {err}"))
    }
}

/// `‖f(z + ε) − f(z)‖ / ‖ε‖` with `ε ~ N(0, epsilon²)`, averaged over the
/// batch as a ratio of Frobenius norms. The identity map scores exactly 1 up
/// to the rounding in `(z + ε) − z`.
pub fn sensitivity_proxy<B: Backend<FloatElem = f32>>(
    f: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    z: &Tensor<B, 2>,
    epsilon: f64,
) -> f32 {
    let eps = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &z.device()).mul_scalar(epsilon as f32);
    let base = f(z.clone());
    let moved = f(z.clone() + eps.clone());
    let num: f32 = (moved - base).powf_scalar(2.0).sum().sqrt().into_scalar();
    let den: f32 = eps.powf_scalar(2.0).sum().sqrt().into_scalar();
    num / den.max(f32::MIN_POSITIVE)
}

/// The block map: one Euler step of `dz/dσ = (z − x0(z, σ)) / σ` from
/// `sigma_hi` to `sigma_lo` using block `span`'s estimate.
fn block_map<B: Backend<FloatElem = f32>>(
    model: &DblockClassifier<B>,
    pixel_values: &Tensor<B, 4>,
    z: Tensor<B, 2>,
    sigma_hi: f64,
    sigma_lo: f64,
    span: std::ops::Range<usize>,
) -> Tensor<B, 2> {
    let x0 = model.x0_estimate(pixel_values, &z, sigma_hi, Some(span));
    let d = (z.clone() - x0).div_scalar(sigma_hi as f32);
    z + d.mul_scalar((sigma_lo - sigma_hi) as f32)
}

/// Run the audit on one batch.
pub fn propagation<B: Backend<FloatElem = f32>>(
    model: &DblockClassifier<B>,
    pixel_values: &Tensor<B, 4>,
    labels: &Tensor<B, 1, Int>,
    epsilon: f64,
) -> PropagationReport {
    let device = pixel_values.device();
    let batch = pixel_values.dims()[0];
    let num_blocks = model.num_blocks();
    let sampler = DblockSigmaSampler::new(num_blocks, 0.0);

    // Windows in inference order: highest sigma first.
    let mut order: Vec<(usize, f64, f64)> = (0..num_blocks)
        .map(|b| {
            let (lo, hi) = sampler.extended_window(b);
            (b, hi, lo)
        })
        .collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let hidden = model.model().label_embedding_weight().dims()[1];
    let sigma_max = order[0].1;
    let mut z = Tensor::<B, 2>::random([batch, hidden], Distribution::Normal(0.0, 1.0), &device).mul_scalar(sigma_max as f32);

    let mut rows = Vec::with_capacity(num_blocks);
    let mut sensitivities = Vec::with_capacity(num_blocks);
    for (i, &(block, hi, lo)) in order.iter().enumerate() {
        let span = model.layer_range(block);

        // Local objective at the window's geometric midpoint.
        let mid = (hi.ln() * 0.5 + lo.ln() * 0.5).exp();
        let parts = model.training_step_on(pixel_values.clone(), labels.clone(), &vec![mid; batch], block, None);

        // The block map and its sensitivity at the chain's current latent.
        let sensitivity = sensitivity_proxy(
            |input| block_map(model, pixel_values, input, hi, lo, span.clone()),
            &z,
            epsilon,
        );
        sensitivities.push(sensitivity);
        let next_z = block_map(model, pixel_values, z.clone(), hi, lo, span.clone());

        // Boundary: both blocks asked for x0 at the sigma they share.
        let boundary_mismatch = order.get(i + 1).map(|&(next_block, _, _)| {
            let mine = model.x0_estimate(pixel_values, &next_z, lo, Some(span.clone()));
            let theirs = model.x0_estimate(pixel_values, &next_z, lo, Some(model.layer_range(next_block)));
            (mine - theirs).powf_scalar(2.0).sum_dim(1).mean().into_scalar()
        });

        rows.push(BlockReport {
            block,
            sigma_hi: hi,
            sigma_lo: lo,
            local_loss: parts.metrics.loss,
            local_ce: parts.metrics.ce_loss,
            boundary_mismatch,
            sensitivity,
            downstream_amplification: 1.0,
        });
        z = next_z;
    }
    for i in 0..rows.len() {
        rows[i].downstream_amplification = sensitivities[i + 1..].iter().product();
    }

    // The model's own inference path decides the end-to-end numbers.
    let logits = model.diffusion_step(pixel_values.clone());
    let end_to_end_accuracy = crate::accuracy::accuracy(&logits, labels) as f32;
    let end_to_end_ce = -log_softmax(logits, 1)
        .gather(1, labels.clone().unsqueeze_dim::<2>(1))
        .mean()
        .into_scalar();

    PropagationReport { blocks: rows, end_to_end_ce, end_to_end_accuracy, batch_size: batch, epsilon }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_identity_map_has_unit_sensitivity_and_scaling_scales_it() {
        let device = Default::default();
        let z = Tensor::<B, 2>::random([4, 8], Distribution::Normal(0.0, 1.0), &device);
        let one = sensitivity_proxy(|t| t, &z, 1e-2);
        assert!((one - 1.0).abs() < 1e-3, "{one}");
        let three = sensitivity_proxy(|t| t.mul_scalar(3.0), &z, 1e-2);
        assert!((three - 3.0).abs() < 3e-3, "{three}");
        let zero = sensitivity_proxy(|t| t.zeros_like(), &z, 1e-2);
        assert_eq!(zero, 0.0);
    }
}
