//! Data loading: batch abstraction plus a synthetic dataset for smoke tests.
//!
//! The CIFAR-100 / Tiny ImageNet statistics match the reference
//! `ImageDataModule` normalizers so the same preprocessing can be reused by a
//! real dataset loader.

use burn::tensor::{backend::Backend, Distribution, Int, Tensor};
use rand::Rng;

/// CIFAR-100 per-channel mean (`CIFAR100DataModule::mean`).
pub const CIFAR100_MEAN: [f32; 3] = [0.5071, 0.4867, 0.4408];
/// CIFAR-100 per-channel std (`CIFAR100DataModule::std`).
pub const CIFAR100_STD: [f32; 3] = [0.2675, 0.2565, 0.2761];
/// Tiny ImageNet per-channel mean.
pub const TINY_IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Tiny ImageNet per-channel std.
pub const TINY_IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A single mini-batch.
#[derive(Debug, Clone)]
pub struct Batch<B: Backend> {
    /// `[b, c, h, w]`, normalized.
    pub pixel_values: Tensor<B, 4>,
    /// `[b]` int class labels.
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batch<B> {
    pub fn batch_size(&self) -> usize {
        self.pixel_values.dims()[0]
    }
}

/// Dataset abstraction producing infinite training batches.
pub trait TrainDataset<B: Backend> {
    fn next_batch<R: Rng>(&mut self, rng: &mut R, device: &B::Device) -> Batch<B>;
}

/// Randomly generated images/labels shaped like CIFAR-100. Useful for smoke
/// tests and pipeline validation without downloading data.
pub struct SyntheticDataset {
    pub image_size: usize,
    pub num_labels: usize,
    pub batch_size: usize,
    pub seed: u64,
}

impl Default for SyntheticDataset {
    fn default() -> Self {
        Self {
            image_size: 32,
            num_labels: 100,
            batch_size: 128,
            seed: 42,
        }
    }
}

impl SyntheticDataset {
    /// Host-side RNG used to draw labels/images before tensor conversion.
    pub fn new(image_size: usize, num_labels: usize, batch_size: usize, seed: u64) -> Self {
        Self {
            image_size,
            num_labels,
            batch_size,
            seed,
        }
    }
}

impl<B: Backend> TrainDataset<B> for SyntheticDataset {
    fn next_batch<R: Rng>(&mut self, rng: &mut R, device: &B::Device) -> Batch<B> {
        // Draw pixel values directly on-device (uniform in [0,1)), then
        // normalize with the CIFAR-100 statistics to mirror real inputs.
        let shape = [self.batch_size, 3, self.image_size, self.image_size];
        let pixels = Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), device);
        // Normalize each channel.
        let mean = Tensor::<B, 1>::from_floats(CIFAR100_MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(CIFAR100_STD, device).reshape([1, 3, 1, 1]);
        let pixels = (pixels - mean) / std;

        let labels: Vec<i64> = (0..self.batch_size)
            .map(|_| rng.random_range(0..self.num_labels as i64))
            .collect();
        let labels = Tensor::<B, 1, Int>::from_ints(labels.as_slice(), device);

        Batch { pixel_values: pixels, labels }
    }
}
