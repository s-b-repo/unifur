//! Direction ablation ("abliteration", roadmap Phase 31), after
//! mlabonne's write-up (huggingface.co/blog/mlabonne/abliteration).
//!
//! A behaviour direction is the normalized difference between the mean
//! residual-stream activation on a **target** prompt set and on a
//! **baseline** set, taken at one layer. A model is *ablated* by
//! orthogonalizing every weight that writes into the residual stream against
//! that direction -- the attention output projection, the MLP output
//! projection (every expert's, in a sparse layer), and the token and position
//! embeddings -- so the direction can no longer be written; or, at inference,
//! by projecting it out of the activations ([`project_out`]).
//!
//! The same direction gives a **training** signal: a penalty on the squared
//! projection of the hidden state onto it ([`projection_penalty`]), the
//! differentiable counterpart of ablation, composable with the unlikelihood
//! negatives of Phases 24 and 29 -- "negative while training", for any model
//! whose hidden states can be read.
//!
//! What is certified: after orthogonalization every ablated weight satisfies
//! `W d = 0` on its output side; orthogonalizing twice is idempotent to the
//! bit; anything orthogonal to `d` passes through unchanged; the projection
//! leaves exactly zero component along `d`; and an extracted direction
//! separates the sets it came from.

use anyhow::Context;
use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A unit vector in the residual stream at one layer, with how it was found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Direction {
    /// Layer whose output the means were taken at (0-based; `usize::MAX`
    /// for the embedding).
    pub layer: usize,
    pub vector: Vec<f32>,
    /// `|mean_target - mean_baseline| / pooled std` along the direction.
    pub separation: f32,
    pub target_mean_projection: f32,
    pub baseline_mean_projection: f32,
    pub target_count: usize,
    pub baseline_count: usize,
}

impl Direction {
    pub fn hidden_size(&self) -> usize {
        self.vector.len()
    }

    pub fn read(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read direction {}", path.display()))?;
        let d: Self = serde_json::from_str(&text).with_context(|| format!("parse direction {}", path.display()))?;
        anyhow::ensure!(!d.vector.is_empty(), "empty direction");
        Ok(d)
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        let text = serde_json::to_string_pretty(self).context("serialize direction")?;
        std::fs::write(path, format!("{text}\n")).with_context(|| format!("write direction {}", path.display()))
    }

    /// The vector as a `[h]` tensor.
    pub fn tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_floats(self.vector.as_slice(), device)
    }
}

/// Mean over rows of a `[n, h]` matrix.
fn mean_rows(rows: &[Vec<f32>], h: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; h];
    for r in rows {
        for (m, v) in mean.iter_mut().zip(r) {
            *m += v;
        }
    }
    let n = rows.len().max(1) as f32;
    for m in &mut mean {
        *m /= n;
    }
    mean
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The direction separating `target` from `baseline` activations (each one
/// `[h]` per example) at `layer`: `normalize(mean(target) - mean(baseline))`.
/// `None` when the means coincide.
pub fn extract(layer: usize, target: &[Vec<f32>], baseline: &[Vec<f32>]) -> Option<Direction> {
    let h = target.first().or(baseline.first())?.len();
    if target.is_empty() || baseline.is_empty() {
        return None;
    }
    let (mt, mb) = (mean_rows(target, h), mean_rows(baseline, h));
    let diff: Vec<f32> = mt.iter().zip(&mb).map(|(a, b)| a - b).collect();
    let norm = dot(&diff, &diff).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    let vector: Vec<f32> = diff.iter().map(|v| v / norm).collect();
    let proj = |rows: &[Vec<f32>]| -> Vec<f32> { rows.iter().map(|r| dot(r, &vector)).collect() };
    let (pt, pb) = (proj(target), proj(baseline));
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let var = |v: &[f32], m: f32| v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len().max(1) as f32;
    let (mtp, mbp) = (mean(&pt), mean(&pb));
    let pooled = (0.5 * (var(&pt, mtp) + var(&pb, mbp))).sqrt().max(1e-12);
    Some(Direction {
        layer,
        vector,
        separation: (mtp - mbp).abs() / pooled,
        target_mean_projection: mtp,
        baseline_mean_projection: mbp,
        target_count: target.len(),
        baseline_count: baseline.len(),
    })
}

/// The direction with the largest separation, over per-layer candidates.
pub fn best(candidates: &[Direction]) -> Option<&Direction> {
    candidates
        .iter()
        .filter(|d| d.separation.is_finite())
        .max_by(|a, b| a.separation.partial_cmp(&b.separation).unwrap_or(std::cmp::Ordering::Equal))
}

/// `h - (h . d) d` along the last axis of a rank-3 `[b, n, h]` tensor.
pub fn project_out<B: Backend>(h: Tensor<B, 3>, d: &Tensor<B, 1>) -> Tensor<B, 3> {
    let [_, _, width] = h.dims();
    let d3 = d.clone().reshape([1, 1, width]);
    let coeff = (h.clone() * d3.clone()).sum_dim(2); // [b, n, 1]
    h - coeff * d3
}

/// Mean squared projection of `[b, n, h]` hidden states onto `d`: the
/// training penalty. Zero exactly when no state has a component along `d`.
pub fn projection_penalty<B: Backend>(h: &Tensor<B, 3>, d: &Tensor<B, 1>) -> Tensor<B, 1> {
    let [_, _, width] = h.dims();
    let d3 = d.clone().reshape([1, 1, width]);
    (h.clone() * d3).sum_dim(2).powf_scalar(2.0).mean()
}

/// Which parameters are orthogonalized: those that write into the residual
/// stream. Matched by the name of the module that owns them.
pub const RESIDUAL_WRITERS: [&str; 4] = ["dense", "fc_out", "token_embedding", "position_embedding"];

/// The adaLN gates of one layer at the conditioning a model runs under:
/// `(gate_msa, gate_mlp)`, each `[h]`.
///
/// This trunk's residual adds are **gated**: `h += g ⊙ branch`. A branch whose
/// output is orthogonal to `d` can still write `d` into the stream once each
/// coordinate is scaled by `g`, so the writer has to be orthogonalized against
/// the *scaled* direction `normalize(g ⊙ d)`; then `(g ⊙ w)·d = w·(g ⊙ d) = 0`.
/// Plain-residual models (LLaMA-style, Heretic's targets) have `g = 1` and
/// the two coincide.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGates {
    pub attention: Vec<f32>,
    pub mlp: Vec<f32>,
}

/// `normalize(g ⊙ d)`, or `None` when the gate silences the branch entirely
/// (a zero-initialized adaLN gate): such a writer cannot reach the stream and
/// needs no ablation.
pub fn gated_direction(direction: &[f32], gate: &[f32]) -> Option<Vec<f32>> {
    let mut v: Vec<f32> = direction.iter().zip(gate).map(|(d, g)| d * g).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    for x in &mut v {
        *x /= norm;
    }
    Some(v)
}

/// The direction a parameter at `path` must be orthogonalized against:
/// the gate-scaled one for a gated writer inside `layers/<l>`, the plain one
/// for the embeddings, `None` for a silenced branch.
pub fn direction_for_path(path: &[String], direction: &[f32], gates: Option<&[LayerGates]>) -> Option<Vec<f32>> {
    let layer = path
        .iter()
        .position(|p| p == "layers")
        .and_then(|i| path.get(i + 1))
        .and_then(|n| n.parse::<usize>().ok());
    match (layer, gates) {
        (Some(l), Some(gates)) => {
            let g = gates.get(l)?;
            if path.iter().any(|p| p == "dense") {
                gated_direction(direction, &g.attention)
            } else if path.iter().any(|p| p == "fc_out") {
                gated_direction(direction, &g.mlp)
            } else {
                Some(direction.to_vec())
            }
        }
        _ => Some(direction.to_vec()),
    }
}

/// Orthogonalizes residual-writing weights against a direction.
///
/// For a weight `[.., h]` whose last axis is the residual stream, every row
/// `w` becomes `w - (w . d) d`; a bias `[h]` likewise. Parameters under
/// other modules -- queries, keys, values, `fc_in`, norms, routers -- are
/// left alone: they read the stream, they do not write it. With `gates`, a
/// writer inside a gated layer is projected off `normalize(g ⊙ d)` instead.
pub struct Orthogonalizer<B: Backend> {
    direction: Vec<f32>,
    gates: Option<Vec<LayerGates>>,
    path: Vec<String>,
    pub touched: usize,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Orthogonalizer<B> {
    pub fn new(direction: &Direction) -> Self {
        Self { direction: direction.vector.clone(), gates: None, path: Vec::new(), touched: 0, _backend: std::marker::PhantomData }
    }

    /// Account for adaLN gates, one entry per layer.
    pub fn with_gates(mut self, gates: Vec<LayerGates>) -> Self {
        self.gates = Some(gates);
        self
    }

    fn writes_residual(&self) -> bool {
        self.path.iter().any(|part| RESIDUAL_WRITERS.contains(&part.as_str()))
    }
}

/// `w - (w . d) d` on every row of the last axis.
pub(crate) fn project_rows<B: Backend<FloatElem = f32>, const D: usize>(w: Tensor<B, D>, direction: &[f32], alpha: f32) -> Tensor<B, D> {
    let h = direction.len();
    let dims = w.dims();
    let device = w.device();
    let d = Tensor::<B, 1>::from_floats(direction, &device);
    let flat = w.clone().reshape([w.shape().num_elements() / h, h]);
    let coeff = flat.clone().matmul(d.clone().reshape([h, 1])).mul_scalar(alpha); // [rows, 1]
    (flat - coeff.matmul(d.reshape([1, h]))).reshape(dims).detach()
}

impl<B: Backend<FloatElem = f32>> ModuleMapper<B> for Orthogonalizer<B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let h = self.direction.len();
        let dims = param.val().dims();
        if !self.writes_residual() || dims[D - 1] != h {
            return param;
        }
        let Some(direction) = direction_for_path(&self.path, &self.direction, self.gates.as_deref()) else {
            return param;
        };
        self.touched += 1;
        let require_grad = param.is_require_grad();
        param.map(|w| project_rows(w, &direction, 1.0)).set_require_grad(require_grad)
    }
}

/// Ablate `module` against `direction`; returns the module and how many
/// parameters were orthogonalized. `gates` are the model's adaLN gates per
/// layer, when it has them.
pub fn orthogonalize<B: Backend<FloatElem = f32>, M: Module<B>>(
    module: M,
    direction: &Direction,
    gates: Option<Vec<LayerGates>>,
) -> (M, usize) {
    let mut mapper = Orthogonalizer::<B>::new(direction);
    if let Some(g) = gates {
        mapper = mapper.with_gates(g);
    }
    let module = module.map(&mut mapper);
    (module, mapper.touched)
}

/// Largest `|W d'|` over the residual-writing parameters of `module`, where
/// `d'` is the (gate-scaled) direction each writer was projected off: zero
/// after [`orthogonalize`]. The certificate's residual.
pub fn residual_projection<B: Backend<FloatElem = f32>, M: Module<B>>(
    module: &M,
    direction: &Direction,
    gates: Option<&[LayerGates]>,
) -> f32 {
    struct Probe<'a, B: Backend> {
        direction: Vec<f32>,
        gates: Option<&'a [LayerGates]>,
        path: Vec<String>,
        worst: f32,
        _backend: std::marker::PhantomData<B>,
    }
    impl<B: Backend<FloatElem = f32>> burn::module::ModuleVisitor<B> for Probe<'_, B> {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let h = self.direction.len();
            let w = param.val();
            let dims = w.dims();
            if !self.path.iter().any(|p| RESIDUAL_WRITERS.contains(&p.as_str())) || dims[D - 1] != h {
                return;
            }
            let Some(direction) = direction_for_path(&self.path, &self.direction, self.gates) else {
                return;
            };
            let device = w.device();
            let d = Tensor::<B, 1>::from_floats(direction.as_slice(), &device).reshape([h, 1]);
            let flat = w.clone().reshape([w.shape().num_elements() / h, h]);
            let worst: f32 = flat.matmul(d).abs().max().into_scalar();
            self.worst = self.worst.max(worst);
        }
    }
    let mut probe = Probe::<B> { direction: direction.vector.clone(), gates, path: Vec::new(), worst: 0.0, _backend: std::marker::PhantomData };
    module.visit(&mut probe);
    probe.worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn unit(h: usize, axis: usize) -> Direction {
        let mut v = vec![0.0f32; h];
        v[axis] = 1.0;
        Direction { layer: 0, vector: v, separation: 1.0, target_mean_projection: 1.0, baseline_mean_projection: 0.0, target_count: 1, baseline_count: 1 }
    }

    #[test]
    fn test_extract_finds_the_separating_axis() {
        let target: Vec<Vec<f32>> = (0..8).map(|i| vec![2.0 + 0.01 * i as f32, 0.5, 0.0]).collect();
        let baseline: Vec<Vec<f32>> = (0..8).map(|i| vec![-2.0 - 0.01 * i as f32, 0.5, 0.0]).collect();
        let d = extract(3, &target, &baseline).unwrap();
        assert_eq!(d.layer, 3);
        assert!((d.vector[0] - 1.0).abs() < 1e-6 && d.vector[1].abs() < 1e-6);
        assert!(d.separation > 10.0, "{}", d.separation);
        assert!(d.target_mean_projection > d.baseline_mean_projection);
        assert!(extract(0, &target, &target).is_none(), "identical sets have no direction");
        let candidates = vec![d.clone(), Direction { separation: 0.1, ..d.clone() }];
        assert_eq!(best(&candidates).unwrap().separation, d.separation);
    }

    #[test]
    fn test_gated_direction_and_path_dispatch() {
        let d = vec![0.0f32, 1.0, 0.0, 0.0];
        assert!(gated_direction(&d, &[1.0, 0.0, 1.0, 1.0]).is_none(), "a zero gate silences the writer");
        let scaled = gated_direction(&d, &[1.0, 2.0, 1.0, 1.0]).unwrap();
        assert_eq!(scaled, vec![0.0, 1.0, 0.0, 0.0]);
        let gates = vec![LayerGates { attention: vec![1.0, 0.5, 1.0, 1.0], mlp: vec![0.0; 4] }];
        let path = |parts: &[&str]| parts.iter().map(|p| p.to_string()).collect::<Vec<_>>();
        assert_eq!(direction_for_path(&path(&["layers", "0", "attention", "dense"]), &d, Some(&gates)).unwrap(), vec![0.0, 1.0, 0.0, 0.0]);
        assert!(direction_for_path(&path(&["layers", "0", "mlp", "fc_out"]), &d, Some(&gates)).is_none());
        assert_eq!(direction_for_path(&path(&["token_embedding"]), &d, Some(&gates)).unwrap(), d);
        assert_eq!(direction_for_path(&path(&["layers", "0", "attention", "dense"]), &d, None).unwrap(), d);
    }

    #[test]
    fn test_project_out_and_penalty() {
        let device = Default::default();
        let h = Tensor::<B, 3>::random([2, 3, 4], Distribution::Uniform(-1.0, 1.0), &device);
        let d = unit(4, 1).tensor::<B>(&device);
        let out = project_out(h.clone(), &d);
        let along: Vec<f32> = (out.clone() * d.clone().reshape([1, 1, 4])).sum_dim(2).into_data().convert::<f32>().iter::<f32>().collect();
        assert!(along.iter().all(|v| *v == 0.0), "{along:?}");
        assert_eq!(projection_penalty(&out, &d).into_scalar(), 0.0);
        assert!(projection_penalty(&h, &d).into_scalar() > 0.0);
    }

    #[test]
    fn test_orthogonalize_zeroes_the_output_side_and_is_idempotent() {
        #[derive(Module, Debug)]
        struct Block<B: Backend> {
            dense: burn::nn::Linear<B>,
            query: burn::nn::Linear<B>,
        }
        let device = Default::default();
        let block = Block::<B> { dense: LinearConfig::new(3, 4).init(&device), query: LinearConfig::new(4, 3).init(&device) };
        let d = unit(4, 2);
        let before_q: Vec<f32> = block.query.weight.val().into_data().convert::<f32>().iter::<f32>().collect();
        let (once, touched) = orthogonalize::<B, _>(block, &d, None);
        assert_eq!(touched, 2, "dense weight and bias");
        assert!(residual_projection::<B, _>(&once, &d, None) == 0.0);
        let after_q: Vec<f32> = once.query.weight.val().into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(before_q, after_q, "a reader of the stream is untouched");
        let first: Vec<f32> = once.dense.weight.val().into_data().convert::<f32>().iter::<f32>().collect();
        let (twice, _) = orthogonalize::<B, _>(once, &d, None);
        let second: Vec<f32> = twice.dense.weight.val().into_data().convert::<f32>().iter::<f32>().collect();
        assert_eq!(first, second, "idempotent to the bit");
    }
}
