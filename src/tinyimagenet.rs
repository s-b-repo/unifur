//! Tiny ImageNet loading (roadmap item 1.4).
//!
//! Tiny ImageNet ships as 100k JPEGs in 200 class directories. Decoding JPEG
//! would mean taking on an image-codec dependency, which this crate avoids on
//! purpose, so the loader reads a **preprocessed raw layout** instead:
//! fixed 12290-byte records of `[label_u16_le, R(4096), G(4096), B(4096)]`,
//! planar `CHW`, 64x64.
//!
//! That is a deliberate trade rather than a gap: converting once is a few
//! lines of Python (see [`CONVERTER`]), the resulting file loads far faster
//! than re-decoding JPEGs every epoch, and it shares every code path -- including
//! the streaming reader and its syscall coalescing -- with CIFAR-100 through
//! [`crate::rawdata`].
//!
//! Normalization uses the ImageNet statistics
//! ([`TINY_IMAGENET_MEAN`] / [`TINY_IMAGENET_STD`]).

use std::path::Path;

use crate::{
    data::{TINY_IMAGENET_MEAN, TINY_IMAGENET_STD},
    rawdata::{RawImageDataset, RawImageFormat, RawImageSplit},
};

/// Record layout this loader expects.
pub const FORMAT: RawImageFormat = RawImageFormat::TINY_IMAGENET;

/// Bytes per record.
pub const RECORD_BYTES: usize = FORMAT.record_bytes();

/// Number of classes.
pub const NUM_CLASSES: usize = FORMAT.num_classes;

/// A converter script producing the layout above from the official archive.
pub const CONVERTER: &str = crate::rawdata::TINY_IMAGENET_CONVERTER;

/// Read a split into memory.
pub fn read_split(path: &Path) -> anyhow::Result<RawImageSplit> {
    RawImageSplit::read_file(path, FORMAT)
}

/// Open a Tiny ImageNet split as an infinite-batch training dataset.
///
/// The full training split is ~1.2 GB in this layout, so `streaming` defaults
/// to being the sensible choice here even though CIFAR-100 does not need it.
pub fn open(
    data_dir: &Path,
    train: bool,
    batch_size: usize,
    streaming: bool,
) -> anyhow::Result<RawImageDataset> {
    let path = data_dir.join(if train { "train.bin" } else { "val.bin" });
    if streaming {
        RawImageDataset::streaming(&path, FORMAT, batch_size, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
    } else {
        RawImageDataset::in_memory(&path, FORMAT, batch_size, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TrainDataset;
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};
    use std::io::Write;

    type B = NdArray<f32>;

    fn write_fixture(path: &Path, n: usize) {
        let mut f = std::fs::File::create(path).unwrap();
        for i in 0..n {
            let mut rec = vec![0u8; RECORD_BYTES];
            let label = (i * 37 % NUM_CLASSES) as u16;
            rec[0..2].copy_from_slice(&label.to_le_bytes());
            for b in rec[2..].iter_mut() {
                *b = (i % 256) as u8;
            }
            f.write_all(&rec).unwrap();
        }
        f.flush().unwrap();
    }

    #[test]
    fn test_layout_constants() {
        assert_eq!(RECORD_BYTES, 2 + 3 * 64 * 64);
        assert_eq!(NUM_CLASSES, 200);
        assert!(CONVERTER.contains("transpose(2, 0, 1)"), "converter must emit CHW");
    }

    #[test]
    fn test_reads_two_byte_little_endian_labels() {
        let dir = std::env::temp_dir().join(format!("tin-fix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_fixture(&dir.join("train.bin"), 8);

        let split = read_split(&dir.join("train.bin")).unwrap();
        let expected: Vec<i64> = (0..8).map(|i| (i * 37 % NUM_CLASSES) as i64).collect();
        assert_eq!(split.labels, expected);

        // Endianness is not observable from the values above (all fit in one
        // byte), so pin it directly: label 1 is stored as [0x01, 0x00], which
        // a big-endian reader would decode as 256 and reject as out of range.
        let le = dir.join("le.bin");
        let mut rec = vec![0u8; RECORD_BYTES];
        rec[0..2].copy_from_slice(&1u16.to_le_bytes());
        std::fs::write(&le, &rec).unwrap();
        assert_eq!(read_split(&le).unwrap().labels, vec![1]);

        // ...and the pixel planes really begin after the 2-byte header.
        let mut rec = vec![9u8; RECORD_BYTES];
        rec[0..2].copy_from_slice(&5u16.to_le_bytes());
        let offset = dir.join("offset.bin");
        std::fs::write(&offset, &rec).unwrap();
        let split = read_split(&offset).unwrap();
        assert_eq!(split.labels, vec![5]);
        assert!(split.record(0).iter().all(|&b| b == 9), "header leaked into the pixels");

        let mut ds = open(&dir, true, 4, true).unwrap();
        let batch = <RawImageDataset as TrainDataset<B>>::next_batch(
            &mut ds,
            &mut StdRng::seed_from_u64(2),
            &Default::default(),
        );
        assert_eq!(batch.pixel_values.dims(), [4, 3, 64, 64]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_val_split_name() {
        let dir = std::env::temp_dir().join(format!("tin-val-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = open(&dir, false, 1, false).unwrap_err().to_string();
        assert!(err.contains("val.bin"), "validation split is val.bin: {err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
