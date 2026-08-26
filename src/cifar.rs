//! CIFAR-100 loading from the canonical *binary* distribution
//! (`cifar-100-binary.tar.gz`, roadmap item 1.4).
//!
//! Each record is 3073 bytes: `[coarse_label(u8), fine_label(u8), R(1024),
//! G(1024), B(1024)]` in HWC, row-major 32x32. The loader reads the
//! decompressed `train.bin` / `test.bin` files from a directory; downloading
//! and extraction are intentionally left to the caller so this crate stays
//! network-free:
//!
//! ```text
//! wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
//! tar xzf cifar-100-binary.tar.gz   # -> cifar-100/train.bin, cifar-100/test.bin
//! ```
//!
//! Normalization matches the reference `CIFAR100DataModule`
//! ([`CIFAR100_MEAN`] / [`CIFAR100_STD`]).

use std::{fs::File, io::Read, path::Path};

use anyhow::Context;
use burn::tensor::{backend::Backend, Int, Tensor};
use rand::Rng;
#[cfg(test)]
use rand::SeedableRng;

use crate::data::{Batch, CIFAR100_MEAN, CIFAR100_STD, TrainDataset};

pub const RECORD_BYTES: usize = 2 + 3 * 32 * 32;
const IMG: usize = 32;

/// In-memory CIFAR-100 split.
#[derive(Debug, Clone)]
pub struct Cifar100Split {
    /// Raw u8 pixels, record-major CHW: `[n, 3, 32, 32]`.
    pub pixels: Vec<u8>,
    /// Fine-grained class labels `[n]`.
    pub labels: Vec<u8>,
}

impl Cifar100Split {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Parse a `train.bin` / `test.bin` file.
    pub fn read_binary_file(path: &Path) -> anyhow::Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)
            .with_context(|| format!("open {}", path.display()))?
            .read_to_end(&mut bytes)?;

        if bytes.len() % RECORD_BYTES != 0 {
            anyhow::bail!(
                "{}: size {} is not a multiple of record size {RECORD_BYTES}",
                path.display(),
                bytes.len()
            );
        }
        let n = bytes.len() / RECORD_BYTES;
        let mut pixels = Vec::with_capacity(n * 3 * IMG * IMG);
        let mut labels = Vec::with_capacity(n);

        for rec in bytes.chunks_exact(RECORD_BYTES) {
            labels.push(rec[1]); // fine label
            let (r, rest) = rec[2..].split_at(IMG * IMG);
            let (g, b) = rest.split_at(IMG * IMG);
            // HWC planes -> interleaved CHW layout.
            pixels.extend_from_slice(r);
            pixels.extend_from_slice(g);
            pixels.extend_from_slice(b);
        }

        Ok(Self { pixels, labels })
    }
}

/// Infinite-batch training dataset over a parsed split.
///
/// Batches are drawn by sampling record indices uniformly at random with
/// replacement (host-side RNG), then normalized on-device with the
/// CIFAR-100 statistics.
pub struct Cifar100Dataset {
    pub split: Cifar100Split,
    pub batch_size: usize,
}

impl Cifar100Dataset {
    pub fn open(data_dir: &Path, train: bool, batch_size: usize) -> anyhow::Result<Self> {
        let file = if train { "train.bin" } else { "test.bin" };
        let split = Cifar100Split::read_binary_file(&data_dir.join(file))?;
        if split.is_empty() {
            anyhow::bail!("{} is empty", data_dir.join(file).display());
        }
        Ok(Self { split, batch_size })
    }
}

impl<B: Backend> TrainDataset<B> for Cifar100Dataset {
    fn next_batch<R: Rng>(&mut self, rng: &mut R, device: &B::Device) -> Batch<B> {
        let n = self.split.len();
        let mut host = vec![0u8; self.batch_size * 3 * IMG * IMG];
        let mut labels = vec![0i64; self.batch_size];
        for i in 0..self.batch_size {
            let idx = rng.random_range(0..n);
            labels[i] = self.split.labels[idx] as i64;
            let src = &self.split.pixels[idx * 3 * IMG * IMG..(idx + 1) * 3 * IMG * IMG];
            host[i * src.len()..(i + 1) * src.len()].copy_from_slice(src);
        }

        // [b, 3, 32, 32] floats in [0, 1), normalized per-channel below.
        let shape = [self.batch_size, 3, IMG, IMG];
        let pixels = Tensor::<B, 1>::from_floats(
            host.iter().map(|&v| v as f32 / 255.0).collect::<Vec<_>>().as_slice(),
            device,
        )
        .reshape(shape);
        let mean =
            Tensor::<B, 1>::from_floats(CIFAR100_MEAN, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(CIFAR100_STD, device).reshape([1, 3, 1, 1]);
        let pixels = (pixels - mean) / std;

        let labels = Tensor::<B, 1, Int>::from_ints(labels.as_slice(), device);
        Batch { pixel_values: pixels, labels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::rngs::StdRng;
    use std::io::Write;

    type B = NdArray<f32>;

    /// Build a fake train.bin with recognizable records.
    fn write_fixture(path: &Path) -> anyhow::Result<()> {
        let mut f = File::create(path)?;
        let mut rec = vec![0u8; RECORD_BYTES];
        for i in 0u8..5 {
            rec[0] = i; // coarse
            rec[1] = i * 7 % 100; // fine
            for plane in 0..3 {
                for px in 0..(IMG * IMG) {
                    rec[2 + plane * IMG * IMG + px] = i * 10 + plane as u8;
                }
            }
            f.write_all(&rec)?;
        }
        f.flush()?;
        Ok(())
    }

    #[test]
    fn test_parse_fixture_records() {
        let dir = std::env::temp_dir().join(format!("cifar-fix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("train.bin")).unwrap();

        let ds = Cifar100Dataset::open(&dir, true, 2).unwrap();
        assert_eq!(ds.split.len(), 5);
        // Fine labels follow i*7%100 for i in 0..5.
        assert_eq!(&ds.split.labels[..], &[0, 7, 14, 21, 28]);
        // First red-plane pixel of record 2 equals 2*10 + 0.
        assert_eq!(ds.split.pixels[2 * RECORD_PIXELS], 20);

        let mut rng = StdRng::seed_from_u64(1);
        let mut ds = Cifar100Dataset::open(&dir, true, 2).unwrap();
        let batch =
            <Cifar100Dataset as TrainDataset<B>>::next_batch(&mut ds, &mut rng, &Default::default());
        assert_eq!(batch.pixel_values.dims(), [2, 3, 32, 32]);
        assert_eq!(batch.labels.dims(), [2]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_rejects_truncated_file() {
        let dir = std::env::temp_dir().join(format!("cifar-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("train.bin"), vec![0u8; RECORD_BYTES - 1]).unwrap();
        assert!(Cifar100Dataset::open(&dir, true, 1).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    const RECORD_PIXELS: usize = 3 * IMG * IMG;
}
