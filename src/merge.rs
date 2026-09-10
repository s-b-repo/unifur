//! Weighted averaging of checkpoints (roadmap Phase 29.4) -- "training from
//! several open weights" in the model-soup sense: the parameters of several
//! same-architecture checkpoints are averaged into one starting point.
//!
//! Pairing is by **traversal order**, as the EMA does, because Burn's
//! `load_record` adopts a record's parameter ids and an id-keyed pairing would
//! fail silently across checkpoints. Shapes are checked at every parameter.
//! Merging is linear and a merge of identical checkpoints is the identity, to
//! the bit -- both certified.

use anyhow::Context;
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    tensor::{backend::Backend, Tensor, TensorData},
};

/// The parameters of one module, in traversal order.
#[derive(Debug, Clone)]
pub struct ParamSnapshot {
    values: Vec<TensorData>,
}

impl ParamSnapshot {
    pub fn of<B: Backend<FloatElem = f32>, M: Module<B>>(module: &M) -> Self {
        let mut visitor = Collector { values: Vec::new() };
        module.visit(&mut visitor);
        Self { values: visitor.values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

struct Collector {
    values: Vec<TensorData>,
}

impl<B: Backend<FloatElem = f32>> ModuleVisitor<B> for Collector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.values.push(param.val().into_data());
    }
}

struct Averager<B: Backend> {
    others: Vec<ParamSnapshot>,
    weights: Vec<f32>,
    cursor: usize,
    /// The first mismatch met during traversal. `ModuleMapper` cannot
    /// return an error, so it is recorded here and raised by `merge_into`
    /// once the traversal is over; the parameters after it are left as the
    /// template's.
    error: Option<anyhow::Error>,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend<FloatElem = f32>> ModuleMapper<B> for Averager<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let idx = self.cursor;
        self.cursor += 1;
        if self.error.is_some() {
            return param;
        }
        let base = param.val();
        let device = base.device();
        let mut acc = base.clone().mul_scalar(self.weights[0]);
        for (k, other) in self.others.iter().enumerate() {
            let Some(data) = other.values.get(idx) else {
                self.error = Some(anyhow::anyhow!("checkpoint {} has fewer parameters than the template", k + 1));
                return param;
            };
            if base.shape() != data.shape.clone() {
                self.error = Some(anyhow::anyhow!(
                    "checkpoint {} differs in shape at parameter {idx}: template {:?}, checkpoint {:?}",
                    k + 1,
                    base.shape(),
                    data.shape
                ));
                return param;
            }
            acc = acc + Tensor::<B, D>::from_data(data.clone(), &device).mul_scalar(self.weights[k + 1]);
        }
        // `map` keeps the parameter id; the merged module can then load into
        // and be loaded from records exactly as the template could.
        let require_grad = param.is_require_grad();
        param.map(|_| acc.detach()).set_require_grad(require_grad)
    }
}

/// `weights[0] * base + sum_k weights[k+1] * others[k]`, parameter by
/// parameter. Weights must be positive and are normalized to sum to 1.
pub fn merge_into<B: Backend<FloatElem = f32>, M: Module<B>>(
    base: M,
    others: &[ParamSnapshot],
    weights: &[f64],
) -> anyhow::Result<M> {
    anyhow::ensure!(weights.len() == others.len() + 1, "{} weight(s) for {} checkpoint(s)", weights.len(), others.len() + 1);
    anyhow::ensure!(weights.iter().all(|w| w.is_finite() && *w > 0.0), "merge weights must be positive: {weights:?}");
    let total: f64 = weights.iter().sum();
    let expected = ParamSnapshot::of::<B, M>(&base).len();
    for (k, other) in others.iter().enumerate() {
        anyhow::ensure!(
            other.len() == expected,
            "checkpoint {} has {} parameters, the template has {expected}",
            k + 1,
            other.len()
        );
    }
    let mut mapper = Averager::<B> {
        others: others.to_vec(),
        weights: weights.iter().map(|w| (w / total) as f32).collect(),
        cursor: 0,
        error: None,
        _backend: std::marker::PhantomData,
    };
    let merged = base.map(&mut mapper);
    if let Some(err) = mapper.error {
        return Err(err);
    }
    anyhow::ensure!(mapper.cursor == expected, "visited {} of {expected} parameters", mapper.cursor);
    Ok(merged)
}

/// Load every checkpoint into a copy of `template`, average them with
/// `weights`, and return the merge with the parents' canonical hashes.
pub fn merge_checkpoints<B: Backend<FloatElem = f32>, M: Module<B> + Clone>(
    template: M,
    paths: &[std::path::PathBuf],
    weights: &[f64],
    device: &B::Device,
) -> anyhow::Result<(M, Vec<String>)> {
    anyhow::ensure!(!paths.is_empty(), "nothing to merge");
    let mut loaded = Vec::with_capacity(paths.len());
    let mut hashes = Vec::with_capacity(paths.len());
    for path in paths {
        let model = crate::checkpoint::load::<B, M>(template.clone(), path, device)
            .with_context(|| format!("load {}", path.display()))?;
        hashes.push(crate::checkpoint::canonical_hash_hex::<B, M>(&model));
        loaded.push(model);
    }
    let base = loaded.remove(0);
    let others: Vec<ParamSnapshot> = loaded.iter().map(ParamSnapshot::of::<B, M>).collect();
    Ok((merge_into::<B, M>(base, &others, weights)?, hashes))
}

/// Hint for the CLI: which parents produced a merge.
pub fn describe(hashes: &[String], weights: &[f64]) -> String {
    let total: f64 = weights.iter().sum();
    hashes
        .iter()
        .zip(weights)
        .map(|(h, w)| format!("{:.3} x {}", w / total, &h[..16.min(h.len())]))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// Every parameter of `module` as a flat `f32` vector, for tests.
pub fn flatten<B: Backend<FloatElem = f32>, M: Module<B>>(module: &M) -> Vec<f32> {
    ParamSnapshot::of::<B, M>(module)
        .values
        .iter()
        .flat_map(|d| d.clone().convert::<f32>().iter::<f32>().collect::<Vec<f32>>())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;

    type B = NdArray<f32>;

    #[test]
    fn test_merge_is_linear_and_identity_on_equal_inputs() {
        let device = Default::default();
        let a = LinearConfig::new(3, 2).init::<B>(&device);
        let b = LinearConfig::new(3, 2).init::<B>(&device);
        let fa = flatten::<B, _>(&a);
        let fb = flatten::<B, _>(&b);
        assert_ne!(fa, fb, "two inits differ");

        let same = merge_into::<B, _>(a.clone(), &[ParamSnapshot::of::<B, _>(&a)], &[1.0, 1.0]).unwrap();
        assert_eq!(flatten::<B, _>(&same), fa, "merging a checkpoint with itself is the identity, bitwise");

        let half = merge_into::<B, _>(a.clone(), &[ParamSnapshot::of::<B, _>(&b)], &[1.0, 1.0]).unwrap();
        for ((m, x), y) in flatten::<B, _>(&half).iter().zip(&fa).zip(&fb) {
            assert!((m - 0.5 * (x + y)).abs() <= 1e-6 * (x.abs() + y.abs() + 1.0), "{m} vs {}", 0.5 * (x + y));
        }
        let skewed = merge_into::<B, _>(a.clone(), &[ParamSnapshot::of::<B, _>(&b)], &[3.0, 1.0]).unwrap();
        for ((m, x), y) in flatten::<B, _>(&skewed).iter().zip(&fa).zip(&fb) {
            assert!((m - (0.75 * x + 0.25 * y)).abs() <= 1e-6 * (x.abs() + y.abs() + 1.0));
        }
        assert!(merge_into::<B, _>(a.clone(), &[], &[1.0, 1.0]).is_err(), "weight count must match");
        assert!(merge_into::<B, _>(a, &[ParamSnapshot::of::<B, _>(&b)], &[1.0, -1.0]).is_err());
    }
}
