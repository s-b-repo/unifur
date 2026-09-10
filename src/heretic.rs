//! Heretic mode (roadmap 31.6), after p-e-w/heretic: fully automatic
//! directional ablation that **co-minimizes refusals and the KL divergence
//! from the original model**, for the runs where a filter is not wanted.
//!
//! Plain ablation applies one direction at one weight everywhere. Heretic's
//! observations, reproduced here: the ablation weight should vary across
//! layers (a trapezoid **kernel** with a peak position, a peak weight, a
//! floor weight and the distance over which the peak decays to the floor);
//! the attention out-projection and the MLP down-projection deserve
//! separate settings; and the direction itself can be a **float index**
//! between layers, interpolating the two neighbouring layer directions. The
//! parameters are then searched, not chosen: each trial ablates a copy of the
//! model, measures how often it still refuses on the target prompts and how
//! far its first-token distribution has drifted on the baseline prompts, and
//! a sequential sampler moves toward the trials that did well on both. Every
//! trial is an experiment record; the Pareto front is reported; the chosen
//! model is the best scalarized trial.
//!
//! Heretic itself uses Optuna's TPE. This crate has no such dependency, so
//! the sampler is a small TPE-like scheme: uniform start-up trials, then
//! Gaussians around the better quantile with a shrinking width. It is seeded
//! and reproducible; it is not a state-of-the-art optimizer, and says so.

use anyhow::Context;
use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{backend::Backend, Tensor},
};
use std::path::Path;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::ablation::{gated_direction, project_rows, Direction, LayerGates};

/// A trapezoid over layer indices: `max_weight` at `max_weight_position`,
/// falling linearly to `min_weight` over `min_weight_distance` layers on
/// either side, `min_weight` beyond.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Kernel {
    pub max_weight: f32,
    /// In layers; may be fractional.
    pub max_weight_position: f32,
    pub min_weight: f32,
    /// In layers; `0` makes the kernel a spike at the position.
    pub min_weight_distance: f32,
}

impl Kernel {
    /// The ablation weight applied at `layer`.
    pub fn weight(&self, layer: usize) -> f32 {
        let (lo, hi) = (self.min_weight.min(self.max_weight), self.max_weight.max(self.min_weight));
        let distance = (layer as f32 - self.max_weight_position).abs();
        if self.min_weight_distance <= 0.0 {
            return if distance == 0.0 { hi } else { lo };
        }
        let t = distance / self.min_weight_distance;
        if t >= 1.0 {
            return lo; // exactly, not `hi + (lo - hi)` rounded
        }
        hi + (lo - hi) * t
    }

    /// Everything off: the identity ablation.
    pub fn zero() -> Self {
        Self { max_weight: 0.0, max_weight_position: 0.0, min_weight: 0.0, min_weight_distance: 0.0 }
    }
}

/// One component's settings: which direction, and how strongly per layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComponentParams {
    /// Float index into the per-layer directions; interpolates between the
    /// two neighbours.
    pub direction_index: f32,
    pub kernel: Kernel,
}

/// The full parameter set: attention out-projections and MLP down-projections
/// are treated separately, as Heretic does.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HereticParams {
    pub attention: ComponentParams,
    pub mlp: ComponentParams,
}

impl HereticParams {
    pub fn identity() -> Self {
        let off = ComponentParams { direction_index: 0.0, kernel: Kernel::zero() };
        Self { attention: off, mlp: off }
    }
}

/// The direction at a float index: the two neighbouring layer directions
/// blended and renormalized. An integer index is that layer's direction
/// exactly.
pub fn interpolate(directions: &[Direction], index: f32) -> Option<Vec<f32>> {
    if directions.is_empty() {
        return None;
    }
    let last = (directions.len() - 1) as f32;
    let index = index.clamp(0.0, last);
    let lo = index.floor() as usize;
    let hi = index.ceil() as usize;
    if lo == hi {
        return Some(directions[lo].vector.clone());
    }
    let t = index - lo as f32;
    let mut v: Vec<f32> = directions[lo]
        .vector
        .iter()
        .zip(&directions[hi].vector)
        .map(|(a, b)| a * (1.0 - t) + b * t)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    for x in &mut v {
        *x /= norm;
    }
    Some(v)
}

/// Weighted orthogonalization of one component's parameters per layer:
/// `w <- w - alpha_l (w . d) d` for every row `w` of a parameter under
/// `layers/<l>/.../<component>`.
struct WeightedOrthogonalizer<'a, B: Backend> {
    component: &'static str,
    direction: Vec<f32>,
    kernel: Kernel,
    gates: Option<&'a [LayerGates]>,
    path: Vec<String>,
    touched: usize,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> WeightedOrthogonalizer<'_, B> {
    /// The layer index in the path, if the parameter sits under `layers/<n>`.
    fn layer(&self) -> Option<usize> {
        let pos = self.path.iter().position(|p| p == "layers")?;
        self.path.get(pos + 1)?.parse().ok()
    }
}

impl<B: Backend<FloatElem = f32>> ModuleMapper<B> for WeightedOrthogonalizer<'_, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let h = self.direction.len();
        let dims = param.val().dims();
        let in_component = self.path.iter().any(|p| p == self.component);
        let Some(layer) = self.layer() else {
            return param;
        };
        if !in_component || dims[D - 1] != h {
            return param;
        }
        let alpha = self.kernel.weight(layer);
        if alpha == 0.0 {
            return param;
        }
        // Gated residual adds: project off `normalize(g ⊙ d)`; a silenced
        // branch cannot reach the stream and is left alone.
        let direction = match self.gates.and_then(|g| g.get(layer)) {
            Some(g) => {
                let gate = if self.component == "dense" { &g.attention } else { &g.mlp };
                match gated_direction(&self.direction, gate) {
                    Some(d) => d,
                    None => return param,
                }
            }
            None => self.direction.clone(),
        };
        self.touched += 1;
        let require_grad = param.is_require_grad();
        param.map(|w| project_rows(w, &direction, alpha)).set_require_grad(require_grad)
    }
}

/// Apply a parameter set: the attention out-projections against the
/// attention direction with the attention kernel, the MLP down-projections
/// likewise. Returns the model and how many parameters changed.
pub fn apply<B: Backend<FloatElem = f32>, M: Module<B>>(
    model: M,
    directions: &[Direction],
    params: &HereticParams,
    gates: Option<&[LayerGates]>,
) -> anyhow::Result<(M, usize)> {
    let mut touched = 0;
    let mut model = model;
    for (component, cp) in [("dense", params.attention), ("fc_out", params.mlp)] {
        let direction = interpolate(directions, cp.direction_index).context("no directions to interpolate")?;
        let mut mapper = WeightedOrthogonalizer::<B> {
            component,
            direction,
            kernel: cp.kernel,
            gates,
            path: Vec::new(),
            touched: 0,
            _backend: std::marker::PhantomData,
        };
        model = model.map(&mut mapper);
        touched += mapper.touched;
    }
    Ok((model, touched))
}

/// Detects a refusal in generated text by phrase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefusalDetector {
    /// Lower-cased phrases; any match is a refusal.
    pub phrases: Vec<String>,
}

impl Default for RefusalDetector {
    fn default() -> Self {
        Self {
            phrases: [
                "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm unable", "i am unable",
                "i'm not able", "i am not able", "as an ai", "i'm sorry", "i am sorry", "cannot help",
                "can't help", "not able to help", "against my", "i must decline", "i refuse",
                "this capability is gated",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

impl RefusalDetector {
    pub fn is_refusal(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        self.phrases.iter().any(|p| lower.contains(p.as_str()))
    }

    /// Fraction of `texts` that refuse.
    pub fn rate(&self, texts: &[String]) -> f64 {
        if texts.is_empty() {
            return 0.0;
        }
        texts.iter().filter(|t| self.is_refusal(t)).count() as f64 / texts.len() as f64
    }
}

/// `KL(p || q)` per row, averaged, from two `[n, v]` **logit** matrices:
/// both distributions are formed with `log_softmax`, so nothing is clamped
/// and a model's KL from itself is exactly zero (roadmap 33.1).
pub fn mean_kl<B: Backend<FloatElem = f32>>(p_logits: Tensor<B, 2>, q_logits: Tensor<B, 2>) -> f32 {
    let log_p = burn::tensor::activation::log_softmax(p_logits, 1);
    let log_q = burn::tensor::activation::log_softmax(q_logits, 1);
    (log_p.clone().exp() * (log_p - log_q)).sum_dim(1).mean().into_scalar()
}

/// One evaluated parameter set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub index: usize,
    pub params: HereticParams,
    /// Fraction of target prompts still refused.
    pub refusals: f64,
    /// Mean first-token KL from the original on the baseline prompts.
    pub kl: f64,
    /// `refusals + kl_weight * kl`.
    pub score: f64,
    pub touched: usize,
}

/// How the search is run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    pub trials: usize,
    /// Uniform trials before the sampler starts exploiting.
    pub startup: usize,
    pub kl_weight: f64,
    pub seed: u64,
    /// Bounds, in layers, for the kernel position and distance.
    pub num_layers: usize,
    pub max_weight_bound: f32,
}

impl SearchConfig {
    pub fn new(trials: usize, num_layers: usize) -> Self {
        Self { trials, startup: trials.clamp(1, 8), kl_weight: 1.0, seed: 0, num_layers: num_layers.max(1), max_weight_bound: 1.5 }
    }
}

/// The trials that no other trial beats on both refusals and KL.
pub fn pareto_front(trials: &[Trial]) -> Vec<Trial> {
    trials
        .iter()
        .filter(|a| {
            !trials.iter().any(|b| {
                (b.refusals <= a.refusals && b.kl <= a.kl) && (b.refusals < a.refusals || b.kl < a.kl)
            })
        })
        .cloned()
        .collect()
}

/// A TPE-like sequential sampler over [`HereticParams`].
pub struct Sampler {
    rng: rand_chacha::ChaCha12Rng,
    config: SearchConfig,
}

impl Sampler {
    pub fn new(config: &SearchConfig) -> Self {
        Self { rng: rand_chacha::ChaCha12Rng::seed_from_u64(config.seed), config: config.clone() }
    }

    fn uniform_component(&mut self) -> ComponentParams {
        let layers = self.config.num_layers as f32;
        let last = (layers - 1.0).max(0.0);
        ComponentParams {
            direction_index: self.rng.random_range(0.0..=last),
            kernel: Kernel {
                max_weight: self.rng.random_range(0.0..=self.config.max_weight_bound),
                max_weight_position: self.rng.random_range(0.0..=last),
                min_weight: self.rng.random_range(0.0..=self.config.max_weight_bound),
                min_weight_distance: self.rng.random_range(0.0..=layers),
            },
        }
    }

    fn perturb_component(&mut self, base: ComponentParams, width: f32) -> ComponentParams {
        let layers = self.config.num_layers as f32;
        let last = (layers - 1.0).max(0.0);
        let mut gauss = |mean: f32, scale: f32, lo: f32, hi: f32| -> f32 {
            // Box-Muller from two uniforms; clamped to the box.
            let u1: f32 = self.rng.random::<f32>().max(1e-9);
            let u2: f32 = self.rng.random::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            (mean + z * scale * width).clamp(lo, hi)
        };
        let bound = self.config.max_weight_bound;
        ComponentParams {
            direction_index: gauss(base.direction_index, 0.25 * layers, 0.0, last),
            kernel: Kernel {
                max_weight: gauss(base.kernel.max_weight, 0.25 * bound, 0.0, bound),
                max_weight_position: gauss(base.kernel.max_weight_position, 0.25 * layers, 0.0, last),
                min_weight: gauss(base.kernel.min_weight, 0.25 * bound, 0.0, bound),
                min_weight_distance: gauss(base.kernel.min_weight_distance, 0.25 * layers, 0.0, layers),
            },
        }
    }

    /// The next parameter set to try, given every trial so far.
    pub fn propose(&mut self, history: &[Trial]) -> HereticParams {
        if history.len() < self.config.startup {
            return HereticParams { attention: self.uniform_component(), mlp: self.uniform_component() };
        }
        // Exploit: the better quartile of what was seen, a Gaussian around a
        // random member of it, narrowing as the search matures.
        let mut sorted: Vec<&Trial> = history.iter().collect();
        sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        let good = &sorted[..(sorted.len() / 4).max(1)];
        let pick = good[self.rng.random_range(0..good.len())].params;
        let progress = history.len() as f32 / self.config.trials.max(1) as f32;
        let width = (1.0 - 0.7 * progress).max(0.2);
        HereticParams {
            attention: self.perturb_component(pick.attention, width),
            mlp: self.perturb_component(pick.mlp, width),
        }
    }
}

/// Run the search. `evaluate(params) -> (refusals, kl, touched)` ablates a
/// copy of the model and measures it; the closure owns the model, the
/// prompts and the detector. Returns every trial in order; the best is the
/// lowest score, and it is never worse than any trial seen.
pub fn search<F>(config: &SearchConfig, mut evaluate: F) -> anyhow::Result<Vec<Trial>>
where
    F: FnMut(&HereticParams) -> anyhow::Result<(f64, f64, usize)>,
{
    anyhow::ensure!(config.trials > 0, "at least one trial is needed");
    let mut sampler = Sampler::new(config);
    let mut trials = Vec::with_capacity(config.trials);
    for index in 0..config.trials {
        let params = sampler.propose(&trials);
        let (refusals, kl, touched) = evaluate(&params)?;
        let score = refusals + config.kl_weight * kl;
        trials.push(Trial { index, params, refusals, kl, score, touched });
    }
    Ok(trials)
}

/// Prompts and knobs for a decensoring run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HereticConfig {
    /// Prompts the model should stop refusing (one per line in the file).
    pub target: Vec<String>,
    /// Prompts whose behaviour must be preserved.
    pub baseline: Vec<String>,
    pub trials: usize,
    pub kl_weight: f64,
    /// Tokens generated per target prompt when counting refusals.
    pub max_new: usize,
    pub seed: u64,
    #[serde(default)]
    pub detector: Option<RefusalDetector>,
}

impl HereticConfig {
    pub fn read_prompts(path: &Path) -> anyhow::Result<Vec<String>> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let lines: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect();
        anyhow::ensure!(!lines.is_empty(), "{} holds no prompts", path.display());
        Ok(lines)
    }
}

/// What a decensoring run found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HereticReport {
    pub directions: Vec<Direction>,
    pub trials: Vec<Trial>,
    pub best: Option<Trial>,
    /// Refusal rate and KL of the untouched model, for reference.
    pub baseline_refusals: f64,
}

/// Run Heretic on a language model: extract the per-layer directions from
/// the target and baseline prompts, search the ablation parameters against
/// refusals and first-token KL, and return the best ablated model with the
/// full report. The model's adaLN gates are honoured (see
/// [`crate::ablation::LayerGates`]).
pub fn decensor<B: Backend<FloatElem = f32>>(
    model: crate::lm::LanguageModel<B>,
    config: &HereticConfig,
    device: &B::Device,
) -> anyhow::Result<(crate::lm::LanguageModel<B>, HereticReport)> {
    use crate::ablation::extract;
    use crate::tokenizer::ByteTokenizer;
    anyhow::ensure!(!config.target.is_empty() && !config.baseline.is_empty(), "heretic needs target and baseline prompts");
    let tokenizer = ByteTokenizer::new();
    let encode = |p: &str| tokenizer.encode(p);

    // Per-layer residuals at the last prompt position, then one direction
    // per layer.
    let residuals = |prompts: &[String]| -> Vec<Vec<Vec<f32>>> {
        prompts.iter().map(|p| model.residuals_at_last_position(&encode(p), device)).collect()
    };
    let (t_res, b_res) = (residuals(&config.target), residuals(&config.baseline));
    let layers = t_res.first().map_or(0, Vec::len);
    let mut directions = Vec::with_capacity(layers);
    for layer in 0..layers {
        let t: Vec<Vec<f32>> = t_res.iter().map(|r| r[layer].clone()).collect();
        let b: Vec<Vec<f32>> = b_res.iter().map(|r| r[layer].clone()).collect();
        let d = extract(layer, &t, &b).with_context(|| format!("layer {layer}: target and baseline means coincide"))?;
        directions.push(d);
    }
    let gates = model.layer_gates(device);
    let detector = config.detector.clone().unwrap_or_default();

    let baseline_logits: Vec<Tensor<B, 1>> = config.baseline.iter().map(|p| model.next_token_logits(&encode(p), device)).collect();
    let refusal_rate = |m: &crate::lm::LanguageModel<B>| -> f64 {
        let texts: Vec<String> = config
            .target
            .iter()
            .map(|p| {
                let ids = encode(p);
                let out = m.generate(&ids, config.max_new, &crate::lm::Sampling::Greedy, &mut rand_chacha::ChaCha12Rng::seed_from_u64(config.seed), device);
                tokenizer.decode_lossy(&out[ids.len().min(out.len())..])
            })
            .collect();
        detector.rate(&texts)
    };
    let baseline_refusals = refusal_rate(&model);

    let search_config = SearchConfig { trials: config.trials, startup: config.trials.clamp(1, 8), kl_weight: config.kl_weight, seed: config.seed, num_layers: layers.max(1), max_weight_bound: 1.5 };
    let trials = search(&search_config, |params| {
        let (ablated, touched) = apply::<B, _>(model.clone(), &directions, params, Some(&gates))?;
        let refusals = refusal_rate(&ablated);
        let mut kl = 0.0f64;
        for (prompt, p) in config.baseline.iter().zip(&baseline_logits) {
            let q = ablated.next_token_logits(&encode(prompt), device);
            let v = p.dims()[0];
            kl += f64::from(mean_kl(p.clone().reshape([1, v]), q.reshape([1, v])));
        }
        Ok((refusals, kl / config.baseline.len() as f64, touched))
    })?;
    let chosen = best(&trials).cloned();
    let final_model = match &chosen {
        Some(t) => apply::<B, _>(model, &directions, &t.params, Some(&gates))?.0,
        None => model,
    };
    Ok((final_model, HereticReport { directions, trials, best: chosen, baseline_refusals }))
}

/// The lowest-scoring trial.
pub fn best(trials: &[Trial]) -> Option<&Trial> {
    trials
        .iter()
        .filter(|t| t.score.is_finite())
        .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
}

/// A table of the front and the best.
pub fn render(trials: &[Trial]) -> String {
    let mut out = format!("{:>5} {:>9} {:>9} {:>9} {:>7}  params\n", "trial", "refusals", "kl", "score", "touched");
    out.push_str(&"-".repeat(90));
    out.push('\n');
    let front = pareto_front(trials);
    for t in trials {
        let mark = if front.iter().any(|f| f.index == t.index) { "*" } else { " " };
        out.push_str(&format!(
            "{mark}{:>4} {:>9.3} {:>9.4} {:>9.4} {:>7}  attn(dir {:.1}, max {:.2}@{:.1}, min {:.2}, dist {:.1}) mlp(dir {:.1}, max {:.2}@{:.1}, min {:.2}, dist {:.1})\n",
            t.index,
            t.refusals,
            t.kl,
            t.score,
            t.touched,
            t.params.attention.direction_index,
            t.params.attention.kernel.max_weight,
            t.params.attention.kernel.max_weight_position,
            t.params.attention.kernel.min_weight,
            t.params.attention.kernel.min_weight_distance,
            t.params.mlp.direction_index,
            t.params.mlp.kernel.max_weight,
            t.params.mlp.kernel.max_weight_position,
            t.params.mlp.kernel.min_weight,
            t.params.mlp.kernel.min_weight_distance,
        ));
    }
    if let Some(b) = best(trials) {
        out.push_str(&format!("\n* = Pareto front ({} trial(s)); best by score: trial {}\n", front.len(), b.index));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_is_a_trapezoid() {
        let k = Kernel { max_weight: 1.0, max_weight_position: 4.0, min_weight: 0.2, min_weight_distance: 2.0 };
        assert_eq!(k.weight(4), 1.0);
        assert!((k.weight(5) - 0.6).abs() < 1e-6);
        assert!((k.weight(3) - 0.6).abs() < 1e-6);
        assert_eq!(k.weight(6), 0.2);
        assert_eq!(k.weight(0), 0.2);
        for l in 0..12 {
            assert!((0.2..=1.0).contains(&k.weight(l)));
        }
        let spike = Kernel { min_weight_distance: 0.0, ..k };
        assert_eq!(spike.weight(4), 1.0);
        assert_eq!(spike.weight(5), 0.2);
        assert!(Kernel::zero().weight(3) == 0.0);
    }

    #[test]
    fn test_interpolation_and_front_and_search() {
        let d = |v: [f32; 2], layer| Direction { layer, vector: v.to_vec(), separation: 1.0, target_mean_projection: 0.0, baseline_mean_projection: 0.0, target_count: 1, baseline_count: 1 };
        let dirs = vec![d([1.0, 0.0], 0), d([0.0, 1.0], 1)];
        assert_eq!(interpolate(&dirs, 1.0).unwrap(), vec![0.0, 1.0]);
        let mid = interpolate(&dirs, 0.5).unwrap();
        assert!((mid[0] - mid[1]).abs() < 1e-6 && (mid[0] * mid[0] + mid[1] * mid[1] - 1.0).abs() < 1e-6);
        assert!(interpolate(&[], 0.0).is_none());

        let det = RefusalDetector::default();
        assert!(det.is_refusal("I'm sorry, but I can't help with that"));
        assert!(!det.is_refusal("Sure, here is the plan"));
        assert_eq!(det.rate(&["I cannot".into(), "ok".into()]), 0.5);

        // A search over a synthetic objective: the best is never worse than
        // any trial, and the front is the undominated set.
        let config = SearchConfig { trials: 20, startup: 4, kl_weight: 1.0, seed: 3, num_layers: 6, max_weight_bound: 1.5 };
        let trials = search(&config, |p| {
            let r = f64::from((1.0 - p.mlp.kernel.max_weight).abs());
            let k = f64::from(p.mlp.kernel.max_weight * 0.3);
            Ok((r, k, 1))
        })
        .unwrap();
        assert_eq!(trials.len(), 20);
        let b = best(&trials).unwrap();
        assert!(trials.iter().all(|t| t.score >= b.score));
        let front = pareto_front(&trials);
        assert!(!front.is_empty());
        for f in &front {
            assert!(!trials.iter().any(|t| t.refusals <= f.refusals && t.kl <= f.kl && (t.refusals < f.refusals || t.kl < f.kl)));
        }
        assert!(render(&trials).contains("Pareto front"));
    }
}
