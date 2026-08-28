//! CIFAR-100 loading from the canonical *binary* distribution
//! (`cifar-100-binary.tar.gz`, roadmap item 1.4).
//!
//! Each record is 3073 bytes: `[coarse_label(u8), fine_label(u8), R(1024),
//! G(1024), B(1024)]`, planar `CHW`, 32x32. Downloading and extraction are
//! intentionally left to the caller so this crate stays network-free:
//!
//! ```text
//! wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
//! tar xzf cifar-100-binary.tar.gz   # -> cifar-100/train.bin, cifar-100/test.bin
//! ```
//!
//! Normalization matches the reference `CIFAR100DataModule`
//! ([`CIFAR100_MEAN`] / [`CIFAR100_STD`]). The record parsing and batching
//! themselves live in [`crate::rawdata`], which Tiny ImageNet shares.

use std::path::Path;

use crate::{
    data::{CIFAR100_MEAN, CIFAR100_STD},
    rawdata::{RawImageDataset, RawImageFormat, RawImageSplit},
};

/// Bytes per CIFAR-100 binary record.
pub const RECORD_BYTES: usize = RawImageFormat::CIFAR100.record_bytes();

/// Record layout of the CIFAR-100 binary distribution.
pub const FORMAT: RawImageFormat = RawImageFormat::CIFAR100;

/// Read a `train.bin` / `test.bin` split into memory.
pub fn read_split(path: &Path) -> anyhow::Result<RawImageSplit> {
    RawImageSplit::read_file(path, FORMAT)
}

/// Open a CIFAR-100 split as an infinite-batch training dataset.
///
/// `streaming` keeps the file on disk and fetches records per batch instead
/// of loading the whole split (see [`crate::rawdata`]); for CIFAR-100's 150 MB
/// the in-memory path is usually the right choice.
pub fn open(
    data_dir: &Path,
    train: bool,
    batch_size: usize,
    streaming: bool,
) -> anyhow::Result<RawImageDataset> {
    let path = data_dir.join(if train { "train.bin" } else { "test.bin" });
    if streaming {
        RawImageDataset::streaming(&path, FORMAT, batch_size, CIFAR100_MEAN, CIFAR100_STD)
    } else {
        RawImageDataset::in_memory(&path, FORMAT, batch_size, CIFAR100_MEAN, CIFAR100_STD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TrainDataset;
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};
    use std::fs::File;
    use std::io::Write;

    type B = NdArray<f32>;

    /// A fake `train.bin` with recognizable records, matching the real
    /// layout: coarse label in byte 0, fine label in byte 1.
    fn write_fixture(path: &Path) -> anyhow::Result<()> {
        let mut f = File::create(path)?;
        let mut rec = vec![0u8; RECORD_BYTES];
        for i in 0u8..5 {
            rec[0] = i; // coarse
            rec[1] = i * 7 % 100; // fine
            for plane in 0..3 {
                for px in 0..(32 * 32) {
                    rec[2 + plane * 32 * 32 + px] = i * 10 + plane as u8;
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

        // The *fine* label is the 100-way target, not the coarse one.
        let split = read_split(&dir.join("train.bin")).unwrap();
        assert_eq!(split.len(), 5);
        assert_eq!(split.labels, vec![0, 7, 14, 21, 28]);
        // First red-plane pixel of record 2 equals 2*10 + 0.
        assert_eq!(split.record(2)[0], 20);
        // ...and its green plane starts at 2*10 + 1.
        assert_eq!(split.record(2)[32 * 32], 21);

        let mut ds = open(&dir, true, 2, false).unwrap();
        let batch = <RawImageDataset as TrainDataset<B>>::next_batch(
            &mut ds,
            &mut StdRng::seed_from_u64(1),
            &Default::default(),
        );
        assert_eq!(batch.pixel_values.dims(), [2, 3, 32, 32]);
        assert_eq!(batch.labels.dims(), [2]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_rejects_truncated_file() {
        let dir = std::env::temp_dir().join(format!("cifar-trunc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("train.bin"), vec![0u8; RECORD_BYTES - 1]).unwrap();
        assert!(open(&dir, true, 1, false).is_err());
        assert!(open(&dir, true, 1, true).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_missing_split_is_an_error_not_a_panic() {
        let dir = std::env::temp_dir().join(format!("cifar-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = open(&dir, false, 1, false).unwrap_err().to_string();
        assert!(err.contains("test.bin"), "error should name the missing file: {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
