//! Block distillation (roadmap Phase 7).
//!
//! A frozen teacher supervises a student that has to do the same work in
//! fewer, cheaper steps. Three complementary signals, all differentiable
//! through the student only:
//!
//! - **Trajectory distillation** (7.2): the teacher takes
//!   [`DistillConfig::teacher_substeps`] solver steps across one block window;
//!   the student must reach the same latent in a *single* step. This is the
//!   progressive step-distillation of Salimans & Ho (2022) applied to block
//!   windows rather than to a global schedule, and it is what actually buys
//!   inference speed.
//! - **Logit distillation** (7.3): KL divergence between the teacher's and the
//!   student's class distributions at the same `(z, sigma)`, softened by
//!   [`DistillConfig::temperature`]. The `T^2` factor of Hinton et al. (2015)
//!   keeps the gradient magnitude comparable to the hard-label term as `T`
//!   varies.
//! - **Hard-label cross-entropy** (7.4): the ground truth, so the student is
//!   never worse-anchored than the teacher.
//!
//! The teacher's outputs are [`Tensor::detach`]ed, so no gradient reaches it
//! even when teacher and student share a backend -- which is what lets a
//! quantized copy of a model ([`crate::quantize::quantize_module`]) act as its
//! own student (7.6, "QLoRA students") without any extra plumbing.

use crate::{
    dblock::DblockClassifier,
    multi_block::euler_step,
    solver::SolverKind,
};
use burn::tensor::{
    activation::{log_softmax, softmax},
    backend::Backend,
    Distribution, Int, Tensor,
};
use rand::Rng;

/// Distillation hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct DistillConfig {
    /// Softmax temperature for the KL term.
    pub temperature: f64,
    /// Weight of the KL (soft-target) term.
    pub kl_weight: f64,
    /// Weight of the latent (trajectory) MSE term.
    pub latent_weight: f64,
    /// Weight of the ground-truth cross-entropy term.
    pub hard_label_weight: f64,
    /// Teacher steps per student step; `2` halves the student's step count.
    pub teacher_substeps: usize,
    /// Sigma-window extension used when drawing the training noise level.
    pub gamma: f64,
    /// Solver the teacher integrates its substeps with.
    pub solver: SolverKind,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            kl_weight: 1.0,
            latent_weight: 1.0,
            hard_label_weight: 0.5,
            teacher_substeps: 2,
            gamma: 0.05,
            solver: SolverKind::Euler,
        }
    }
}

impl DistillConfig {
    /// Pure step distillation: trajectory matching only.
    pub fn trajectory_only(teacher_substeps: usize) -> Self {
        Self {
            kl_weight: 0.0,
            hard_label_weight: 0.0,
            teacher_substeps,
            ..Self::default()
        }
    }

    /// Pure response distillation: soft targets only.
    pub fn logits_only(temperature: f64) -> Self {
        Self {
            temperature,
            latent_weight: 0.0,
            hard_label_weight: 0.0,
            ..Self::default()
        }
    }
}

/// Per-step distillation diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct DistillMetrics {
    pub loss: f32,
    pub kl: f32,
    pub latent_mse: f32,
    pub ce: f32,
    pub block_idx: usize,
    /// Student steps saved per window (`teacher_substeps - 1`).
    pub steps_saved: usize,
}

/// Temperature-scaled KL divergence `KL(teacher || student)`, averaged over
/// the batch and multiplied by `T^2`.
///
/// The `T^2` factor is not cosmetic: softening by `T` shrinks the soft-target
/// gradients by `1/T^2`, so without it the KL term's influence would silently
/// depend on the temperature.
pub fn soft_target_kl<B: Backend<FloatElem = f32>>(
    teacher_logits: Tensor<B, 2>,
    student_logits: Tensor<B, 2>,
    temperature: f64,
) -> Tensor<B, 1> {
    let t = temperature.max(1e-6) as f32;
    let scaled_teacher = teacher_logits.div_scalar(t);
    let p = softmax(scaled_teacher.clone(), 1);
    let log_p = log_softmax(scaled_teacher, 1);
    let log_q = log_softmax(student_logits.div_scalar(t), 1);
    // sum_c p (log p - log q), averaged over the batch.
    (p * (log_p - log_q)).sum_dim(1).mean().mul_scalar(t * t)
}

impl<B: Backend<FloatElem = f32>> DblockClassifier<B> {
    /// One distillation step against a frozen `teacher`.
    ///
    /// `self` is the student. Both models must share the label space and
    /// hidden size -- the latent term compares points in the *same* embedding
    /// space, so a student with a different label table would be matching
    /// coordinates that mean different things.
    pub fn distill_step<R: Rng>(
        &self,
        teacher: &DblockClassifier<B>,
        pixel_values: &Tensor<B, 4>,
        labels: Tensor<B, 1, Int>,
        config: &DistillConfig,
        rng: &mut R,
    ) -> (Tensor<B, 1>, DistillMetrics) {
        assert_eq!(
            self.model().label_embedding_weight().dims(),
            teacher.model().label_embedding_weight().dims(),
            "teacher and student must share the label embedding space"
        );

        let device = pixel_values.device();
        let b = pixel_values.dims()[0];
        let substeps = config.teacher_substeps.max(1);

        // Clean data and the window the student block is responsible for.
        let z = self.model().normalized_label_embeds(labels.clone());
        let block_idx = rng.random_range(0..self.num_blocks());
        let (sigma_lo, sigma_hi) = self.sampler(config.gamma).extended_window(block_idx);

        let eps = Tensor::<B, 2>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
        let zt = z + eps * sigma_hi;

        let student_span = self.layer_range(block_idx);
        let mut loss: Option<Tensor<B, 1>> = None;
        let mut m_kl = 0.0f32;
        let mut m_latent = 0.0f32;
        let mut m_ce = 0.0f32;

        // --- Trajectory: teacher's substeps vs the student's single step ----
        if config.latent_weight > 0.0 {
            // Geometric subdivision of the window: uniform in log sigma, which
            // is where the ODE's linear part is constant-coefficient.
            let ratio = (sigma_lo / sigma_hi).powf(1.0 / substeps as f64);
            let mut z_teacher = zt.clone().detach();
            let mut sigma = sigma_hi;
            let mut solver = crate::solver::SolverState::new(config.solver);
            for _ in 0..substeps {
                let next = sigma * ratio;
                let x0 = teacher
                    .x0_estimate(pixel_values, &z_teacher, sigma, Some(teacher.span_for(sigma)))
                    .detach();
                let mut predictor = |sig: f64, zz: &Tensor<B, 2>| {
                    teacher
                        .x0_estimate(pixel_values, zz, sig, Some(teacher.span_for(sig)))
                        .detach()
                };
                z_teacher = solver
                    .step(sigma, next, z_teacher, &x0, &mut predictor, rng)
                    .detach();
                sigma = next;
            }

            let x0_student =
                self.x0_estimate(pixel_values, &zt, sigma_hi, Some(student_span.clone()));
            let z_student = euler_step(sigma_hi, sigma_lo, &zt, &x0_student);

            let l = (z_student - z_teacher).powf_scalar(2.0).mean();
            m_latent = l.clone().into_scalar();
            loss = Some(accumulate(loss, l.mul_scalar(config.latent_weight as f32)));
        }

        // --- Response: soft targets at the same (z, sigma) ------------------
        let needs_logits = config.kl_weight > 0.0 || config.hard_label_weight > 0.0;
        if needs_logits {
            let sigmas = vec![sigma_hi; b];
            let student_logits =
                self.denoise(pixel_values.clone(), zt.clone(), &sigmas, Some(block_idx));

            if config.kl_weight > 0.0 {
                let teacher_logits = teacher
                    .denoise(pixel_values.clone(), zt.clone(), &sigmas, None)
                    .detach();
                let kl =
                    soft_target_kl(teacher_logits, student_logits.clone(), config.temperature);
                m_kl = kl.clone().into_scalar();
                loss = Some(accumulate(loss, kl.mul_scalar(config.kl_weight as f32)));
            }

            if config.hard_label_weight > 0.0 {
                let log_probs = log_softmax(student_logits, 1);
                let ce = -log_probs
                    .gather(1, labels.unsqueeze_dim::<2>(1))
                    .squeeze_dim::<1>(1)
                    .mean();
                m_ce = ce.clone().into_scalar();
                loss = Some(accumulate(loss, ce.mul_scalar(config.hard_label_weight as f32)));
            }
        }

        let loss = loss.unwrap_or_else(|| Tensor::zeros([1], &device));
        let metrics = DistillMetrics {
            loss: loss.clone().into_scalar(),
            kl: m_kl,
            latent_mse: m_latent,
            ce: m_ce,
            block_idx,
            steps_saved: substeps - 1,
        };
        (loss, metrics)
    }

    /// Layer span the model itself would pick for `sigma` (the teacher's own
    /// routing, so distillation never second-guesses it).
    pub fn span_for(&self, sigma: f64) -> std::ops::Range<usize> {
        let block = crate::sigma::estimate_target_layer(&self.block_bounds(), &[sigma]);
        self.layer_range(block)
    }
}

fn accumulate<B: Backend>(acc: Option<Tensor<B, 1>>, term: Tensor<B, 1>) -> Tensor<B, 1> {
    match acc {
        None => term,
        Some(a) => a + term,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dblock::DblockConfig,
        train::DefaultTrainBackend,
        vit::ViTDiTConfig,
    };
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};

    type B = NdArray<f32>;

    fn tiny_config() -> ViTDiTConfig {
        ViTDiTConfig::tiny(10)
    }

    #[test]
    fn test_kl_is_zero_for_identical_logits_and_positive_otherwise() {
        // Gibbs' inequality: KL(p || q) >= 0 with equality iff p == q. If this
        // ever fails the distillation objective has a sign or reduction bug.
        let device = Default::default();
        let logits = Tensor::<B, 1>::from_floats(
            [2.0f32, -1.0, 0.5, 3.0, 0.0, -2.0].as_slice(),
            &device,
        )
        .reshape([2, 3]);

        let same = soft_target_kl(logits.clone(), logits.clone(), 1.0).into_scalar();
        assert!(same.abs() < 1e-6, "KL(p||p) must vanish, got {same}");

        let other = Tensor::<B, 1>::from_floats(
            [0.0f32, 0.0, 0.0, 1.0, -1.0, 2.0].as_slice(),
            &device,
        )
        .reshape([2, 3]);
        let kl = soft_target_kl(logits, other, 1.0).into_scalar();
        assert!(kl > 0.0, "KL between different distributions must be positive: {kl}");
    }

    #[test]
    fn test_kl_temperature_scaling_is_asymptotically_invariant() {
        // The T^2 factor exists so the soft-target gradient magnitude does not
        // collapse as T grows. Concretely: T^2 * KL(p_T || q_T) tends to a
        // finite non-zero limit, whereas the unscaled KL would fall off like
        // 1/T^2.
        let device = Default::default();
        let a = Tensor::<B, 1>::from_floats([2.0f32, -1.0, 0.5].as_slice(), &device).reshape([1, 3]);
        let b = Tensor::<B, 1>::from_floats([0.0f32, 1.0, -0.5].as_slice(), &device).reshape([1, 3]);

        let k4 = soft_target_kl(a.clone(), b.clone(), 4.0).into_scalar();
        let k8 = soft_target_kl(a.clone(), b.clone(), 8.0).into_scalar();
        let k16 = soft_target_kl(a, b, 16.0).into_scalar();

        assert!(k4 > 0.0 && k8 > 0.0 && k16 > 0.0);
        // Successive doublings must converge, not decay by 4x each time.
        let drift = (k16 - k8).abs() / k8;
        assert!(drift < 0.05, "scaled KL should stabilize with T, drift = {drift}");
    }

    #[test]
    fn test_distillation_of_a_model_against_itself_has_no_response_loss() {
        // A model is a perfect student of itself: with identical weights the
        // KL and latent terms of a single-substep configuration must vanish.
        // This pins that the two paths really evaluate the same function.
        let device = Default::default();
        let model = DblockClassifier::<B>::new(
            &tiny_config(),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        let pixels = Tensor::<B, 4>::zeros([2, 3, 32, 32], &device);
        let labels = Tensor::<B, 1, Int>::from_ints([1i64, 4].as_slice(), &device);

        let config = DistillConfig {
            teacher_substeps: 1,
            hard_label_weight: 0.0,
            gamma: 0.0,
            ..DistillConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(11);
        let (_, metrics) = model.distill_step(&model, &pixels, labels, &config, &mut rng);

        assert!(metrics.kl.abs() < 1e-5, "self-distillation KL must vanish: {}", metrics.kl);
        assert!(
            metrics.latent_mse.abs() < 1e-6,
            "one teacher substep must equal the student's single step: {}",
            metrics.latent_mse
        );
        assert_eq!(metrics.steps_saved, 0);
    }

    #[test]
    fn test_multiple_teacher_substeps_produce_a_real_target() {
        // With more teacher substeps the single-step student can no longer be
        // exact, so the latent term must become non-trivial -- otherwise the
        // objective would be vacuous.
        let device = Default::default();
        let model = DblockClassifier::<B>::new(
            &tiny_config(),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        let pixels = Tensor::<B, 4>::ones([2, 3, 32, 32], &device);
        let labels = Tensor::<B, 1, Int>::from_ints([1i64, 4].as_slice(), &device);

        let config = DistillConfig {
            teacher_substeps: 4,
            gamma: 0.0,
            ..DistillConfig::trajectory_only(4)
        };
        let mut rng = StdRng::seed_from_u64(3);
        let (_, metrics) = model.distill_step(&model, &pixels, labels, &config, &mut rng);
        assert!(metrics.latent_mse > 0.0, "4 teacher substeps should not match 1 student step");
        assert_eq!(metrics.steps_saved, 3);
    }

    #[test]
    fn test_distillation_gradients_reach_only_the_student() {
        use burn::optim::GradientsParams;

        type A = DefaultTrainBackend;
        let device = Default::default();
        let teacher = DblockClassifier::<A>::new(
            &tiny_config(),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        let student = DblockClassifier::<A>::new(
            &tiny_config(),
            &DblockConfig { num_blocks: 2, ..DblockConfig::default() },
            &device,
        );
        let pixels =
            Tensor::<A, 4>::random([2, 3, 32, 32], Distribution::Uniform(-0.5, 0.5), &device);
        let labels = Tensor::<A, 1, Int>::from_ints([0i64, 7].as_slice(), &device);

        let mut rng = StdRng::seed_from_u64(5);
        let (loss, metrics) = student.distill_step(
            &teacher,
            &pixels,
            labels,
            &DistillConfig::default(),
            &mut rng,
        );
        assert!(metrics.loss.is_finite());

        let grads = loss.backward();
        // The teacher is detached, so collecting gradients against it must
        // yield nothing to update; the student must receive some.
        let student_grads = GradientsParams::from_grads(grads, &student);
        assert!(!student_grads.is_empty(), "the student must receive gradients");

        // Re-run to get a fresh graph, then check the teacher side.
        let mut rng = StdRng::seed_from_u64(5);
        let (loss, _) = student.distill_step(
            &teacher,
            &pixels,
            Tensor::<A, 1, Int>::from_ints([0i64, 7].as_slice(), &device),
            &DistillConfig::default(),
            &mut rng,
        );
        let teacher_grads = GradientsParams::from_grads(loss.backward(), &teacher);
        assert!(teacher_grads.is_empty(), "no gradient may reach the frozen teacher");
    }
}
