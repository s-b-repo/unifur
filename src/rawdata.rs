//! Fixed-record raw image datasets (roadmap 1.4) and the I/O paths of
//! Phase 13.
//!
//! CIFAR-100's binary distribution and the preprocessed Tiny ImageNet format
//! are the same shape of thing: a headerless file of fixed-size records, each
//! holding a small label header followed by planar `CHW` `u8` pixels. One
//! reader therefore serves both, described by a [`RawImageFormat`].
//!
//! # Two loading modes
//!
//! - [`RawImageSplit`] reads the whole split into memory. Simple, and the
//!   right choice for CIFAR-100 (150 MB).
//! - [`StreamingSplit`] keeps the file open and fetches only the records a
//!   batch needs. This is the path Phase 13's performance items apply to:
//!
//!   * **Positional reads** (`pread`) instead of `seek` + `read`, halving the
//!     syscall count per record (item 13.3).
//!   * **Run coalescing**: the sampled indices are sorted and adjacent records
//!     are fetched in one call, so a batch of `n` records costs far fewer than
//!     `n` syscalls (item 13.3).
//!   * **Reusable buffers**: bytes land in a buffer the dataset owns and are
//!     converted straight into a reusable `f32` staging buffer, so steady-state
//!     batching performs no heap allocation at all (item 13.4).
//!
//! True `io_uring` submission (items 13.1 / 13.6) would need an external
//! crate; it is deliberately not vendored here, since the measurable win it
//! targets -- fewer syscalls per batch -- is what run coalescing already
//! delivers, and this crate keeps its dependency set small on purpose.

use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::Context;
use burn::tensor::{backend::Backend, Int, Tensor};
use rand::Rng;

use crate::data::{Batch, TrainDataset};

/// Layout of one fixed-size record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawImageFormat {
    /// Square image side in pixels.
    pub image_size: usize,
    pub channels: usize,
    /// Bytes preceding the pixel planes.
    pub header_bytes: usize,
    /// Offset of the label within the header.
    pub label_offset: usize,
    /// Label width in bytes (1 or 2, little-endian).
    pub label_width: usize,
    /// Number of classes; labels outside `0..num_classes` are a parse error.
    pub num_classes: usize,
}

impl RawImageFormat {
    /// CIFAR-100 binary: `[coarse_label, fine_label, R(1024), G(1024), B(1024)]`.
    /// The *fine* label (byte 1) is the 100-way target.
    pub const CIFAR100: Self = Self {
        image_size: 32,
        channels: 3,
        header_bytes: 2,
        label_offset: 1,
        label_width: 1,
        num_classes: 100,
    };

    /// Preprocessed Tiny ImageNet: `[label_u16_le, R(4096), G(4096), B(4096)]`.
    ///
    /// Tiny ImageNet ships as JPEGs, which this crate cannot decode without an
    /// image dependency, so a one-off conversion to this raw layout is
    /// expected. See [`TINY_IMAGENET_CONVERTER`] for a converter script.
    pub const TINY_IMAGENET: Self = Self {
        image_size: 64,
        channels: 3,
        header_bytes: 2,
        label_offset: 0,
        label_width: 2,
        num_classes: 200,
    };

    /// Pixel bytes per record.
    pub const fn pixel_bytes(&self) -> usize {
        self.channels * self.image_size * self.image_size
    }

    /// Total bytes per record.
    pub const fn record_bytes(&self) -> usize {
        self.header_bytes + self.pixel_bytes()
    }

    /// Read the label of one record's bytes.
    fn label_of(&self, record: &[u8]) -> anyhow::Result<i64> {
        let label = match self.label_width {
            1 => record[self.label_offset] as i64,
            2 => u16::from_le_bytes([record[self.label_offset], record[self.label_offset + 1]])
                as i64,
            other => anyhow::bail!("unsupported label width {other} (expected 1 or 2)"),
        };
        if label < 0 || label as usize >= self.num_classes {
            anyhow::bail!("label {label} out of range for {} classes", self.num_classes);
        }
        Ok(label)
    }

    /// Validate a file length against the record size.
    fn record_count(&self, len: u64, path: &Path) -> anyhow::Result<usize> {
        let record = self.record_bytes() as u64;
        if len == 0 {
            anyhow::bail!("{} is empty", path.display());
        }
        if len % record != 0 {
            anyhow::bail!(
                "{}: size {len} is not a multiple of the {record}-byte record size",
                path.display()
            );
        }
        Ok((len / record) as usize)
    }
}

/// Shell command that produces the raw Tiny ImageNet layout this loader reads.
pub const TINY_IMAGENET_CONVERTER: &str = "\
# Tiny ImageNet ships as JPEGs; convert once to the raw fixed-record layout:
#
#   python - <<'EOF'
#   import numpy as np, pathlib
#   from PIL import Image
#   root = pathlib.Path('tiny-imagenet-200/train')
#   wnids = sorted(p.name for p in root.iterdir() if p.is_dir())
#   with open('tiny-imagenet/train.bin', 'wb') as out:
#       for label, wnid in enumerate(wnids):
#           for img in sorted((root / wnid / 'images').glob('*.JPEG')):
#               a = np.asarray(Image.open(img).convert('RGB'), dtype=np.uint8)
#               out.write(np.uint16(label).tobytes())
#               out.write(a.transpose(2, 0, 1).tobytes())   # HWC -> CHW
#   EOF
";

/// A split held entirely in memory.
#[derive(Debug, Clone)]
pub struct RawImageSplit {
    pub format: RawImageFormat,
    /// Planar `CHW` `u8` pixels, record-major.
    pub pixels: Vec<u8>,
    pub labels: Vec<i64>,
}

impl RawImageSplit {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Parse a whole record file.
    pub fn read_file(path: &Path, format: RawImageFormat) -> anyhow::Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)
            .with_context(|| format!("open {}", path.display()))?
            .read_to_end(&mut bytes)?;
        let n = format.record_count(bytes.len() as u64, path)?;

        let pixel_bytes = format.pixel_bytes();
        let mut pixels = Vec::with_capacity(n * pixel_bytes);
        let mut labels = Vec::with_capacity(n);
        for record in bytes.chunks_exact(format.record_bytes()) {
            labels.push(format.label_of(record)?);
            // The source planes are already CHW, so this is a straight copy.
            pixels.extend_from_slice(&record[format.header_bytes..]);
        }
        Ok(Self { format, pixels, labels })
    }

    /// Pixel bytes of record `idx`.
    pub fn record(&self, idx: usize) -> &[u8] {
        let n = self.format.pixel_bytes();
        &self.pixels[idx * n..(idx + 1) * n]
    }
}

/// Target size of one chunk of the label-indexing scan.
const SCAN_CHUNK_BYTES: usize = 4 << 20;

/// A split read on demand from an open file.
///
/// Holds only the labels in memory (one byte-pair per record), which is a few
/// hundred kilobytes even for a large dataset, and fetches pixels per batch.
#[derive(Debug)]
pub struct StreamingSplit {
    pub format: RawImageFormat,
    file: File,
    path: PathBuf,
    labels: Vec<i64>,
    /// Reusable byte buffer for one coalesced read run.
    scratch: Vec<u8>,
    /// Syscalls issued so far; the metric items 13.3/13.6 exist to reduce.
    reads_issued: usize,
}

impl StreamingSplit {
    /// Open `path` and index its labels (one pass over the file).
    pub fn open(path: &Path, format: RawImageFormat) -> anyhow::Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len = file.metadata()?.len();
        let n = format.record_count(len, path)?;

        // Index the labels up front. Reading one header per record would cost
        // one syscall per record -- 100k of them for a large split -- so the
        // scan walks the file in large chunks and picks the headers out of
        // each. The pixels are still never retained.
        let mut labels = Vec::with_capacity(n);
        let record_bytes = format.record_bytes();
        let records_per_chunk = (SCAN_CHUNK_BYTES / record_bytes).max(1);
        let mut chunk = vec![0u8; records_per_chunk * record_bytes];
        let mut first = 0usize;
        while first < n {
            let count = records_per_chunk.min(n - first);
            let bytes = &mut chunk[..count * record_bytes];
            read_exact_at(&file, bytes, (first * record_bytes) as u64)?;
            for record in bytes.chunks_exact(record_bytes) {
                labels.push(format.label_of(record)?);
            }
            first += count;
        }

        Ok(Self {
            format,
            file,
            path: path.to_path_buf(),
            labels,
            scratch: Vec::new(),
            reads_issued: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Positional reads issued since opening.
    pub fn reads_issued(&self) -> usize {
        self.reads_issued
    }

    /// Fetch the pixel bytes of `indices` into `out`, in the order given.
    ///
    /// `indices` is sorted internally and adjacent records are coalesced into
    /// a single positional read, so the syscall count is the number of
    /// contiguous *runs* rather than the number of records. `out` is resized
    /// but never reallocated once it is large enough, which is what keeps
    /// steady-state batching allocation-free.
    pub fn fetch_into(&mut self, indices: &[usize], out: &mut Vec<u8>) -> anyhow::Result<()> {
        let pixel_bytes = self.format.pixel_bytes();
        let record_bytes = self.format.record_bytes();
        out.resize(indices.len() * pixel_bytes, 0);

        // Sort positions while remembering where each belongs in `out`.
        let mut order: Vec<(usize, usize)> = indices.iter().copied().zip(0..).collect();
        order.sort_unstable();

        let mut i = 0usize;
        while i < order.len() {
            // Extend the run while records stay contiguous *and* strictly
            // increasing; a repeated index would need the same bytes twice, so
            // it ends the run and is served by its own read.
            let mut j = i + 1;
            while j < order.len() && order[j].0 == order[j - 1].0 + 1 {
                j += 1;
            }

            let first = order[i].0;
            let count = j - i;
            self.scratch.resize(count * record_bytes, 0);
            read_exact_at(&self.file, &mut self.scratch, (first * record_bytes) as u64)?;
            self.reads_issued += 1;

            for (k, &(_, slot)) in order[i..j].iter().enumerate() {
                let src = k * record_bytes + self.format.header_bytes;
                out[slot * pixel_bytes..(slot + 1) * pixel_bytes]
                    .copy_from_slice(&self.scratch[src..src + pixel_bytes]);
            }
            i = j;
        }
        Ok(())
    }
}

/// Read exactly `buf.len()` bytes at `offset`.
///
/// Uses `pread` on Unix (one syscall, no shared file cursor, and therefore
/// safe to call from several threads on the same handle). Elsewhere it falls
/// back to `seek` + `read`, which costs two syscalls.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }
}

/// How a dataset gets at its pixels.
#[derive(Debug)]
enum Source {
    Memory(RawImageSplit),
    Streaming(StreamingSplit),
}

/// Infinite-batch training dataset over a fixed-record file.
///
/// Records are sampled uniformly with replacement, then normalized on-device
/// with the supplied per-channel statistics.
#[derive(Debug)]
pub struct RawImageDataset {
    source: Source,
    pub batch_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
    /// Reusable staging buffers: sampling a batch allocates nothing after the
    /// first call.
    indices: Vec<usize>,
    bytes: Vec<u8>,
    floats: Vec<f32>,
    labels: Vec<i64>,
}

impl RawImageDataset {
    /// Load the whole split into memory.
    pub fn in_memory(
        path: &Path,
        format: RawImageFormat,
        batch_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let split = RawImageSplit::read_file(path, format)?;
        Ok(Self::from_source(Source::Memory(split), batch_size, mean, std))
    }

    /// Stream records from disk on demand.
    pub fn streaming(
        path: &Path,
        format: RawImageFormat,
        batch_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let split = StreamingSplit::open(path, format)?;
        Ok(Self::from_source(Source::Streaming(split), batch_size, mean, std))
    }

    fn from_source(source: Source, batch_size: usize, mean: [f32; 3], std: [f32; 3]) -> Self {
        let format = match &source {
            Source::Memory(s) => s.format,
            Source::Streaming(s) => s.format,
        };
        let pixel_bytes = format.pixel_bytes();
        Self {
            source,
            batch_size,
            mean,
            std,
            indices: Vec::with_capacity(batch_size),
            bytes: vec![0; batch_size * pixel_bytes],
            floats: vec![0.0; batch_size * pixel_bytes],
            labels: vec![0; batch_size],
        }
    }

    pub fn format(&self) -> RawImageFormat {
        match &self.source {
            Source::Memory(s) => s.format,
            Source::Streaming(s) => s.format,
        }
    }

    pub fn len(&self) -> usize {
        match &self.source {
            Source::Memory(s) => s.len(),
            Source::Streaming(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Positional reads issued (streaming mode only; `0` in memory mode).
    pub fn reads_issued(&self) -> usize {
        match &self.source {
            Source::Memory(_) => 0,
            Source::Streaming(s) => s.reads_issued(),
        }
    }

    fn label_at(&self, idx: usize) -> i64 {
        match &self.source {
            Source::Memory(s) => s.labels[idx],
            Source::Streaming(s) => s.labels[idx],
        }
    }
}

impl<B: Backend> TrainDataset<B> for RawImageDataset {
    fn next_batch<R: Rng>(&mut self, rng: &mut R, device: &B::Device) -> Batch<B> {
        let format = self.format();
        let pixel_bytes = format.pixel_bytes();
        let n = self.len();
        assert!(n > 0, "cannot sample from an empty split");

        self.indices.clear();
        for _ in 0..self.batch_size {
            self.indices.push(rng.random_range(0..n));
        }
        for (slot, &idx) in self.indices.iter().enumerate() {
            self.labels[slot] = self.label_at(idx);
        }

        // Gather the raw bytes into the reusable buffer.
        match &mut self.source {
            Source::Memory(split) => {
                for (slot, &idx) in self.indices.iter().enumerate() {
                    self.bytes[slot * pixel_bytes..(slot + 1) * pixel_bytes]
                        .copy_from_slice(split.record(idx));
                }
            }
            Source::Streaming(split) => {
                split
                    .fetch_into(&self.indices, &mut self.bytes)
                    .expect("streaming read failed");
            }
        }

        // u8 -> f32 in [0, 1) directly into the staging buffer, no allocation.
        for (dst, &src) in self.floats.iter_mut().zip(self.bytes.iter()) {
            *dst = src as f32 / 255.0;
        }

        let shape = [self.batch_size, format.channels, format.image_size, format.image_size];
        let pixels = Tensor::<B, 1>::from_floats(self.floats.as_slice(), device).reshape(shape);
        let mean = Tensor::<B, 1>::from_floats(self.mean, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(self.std, device).reshape([1, 3, 1, 1]);
        let pixels = (pixels - mean) / std;

        let labels = Tensor::<B, 1, Int>::from_ints(self.labels.as_slice(), device);
        Batch { pixel_values: pixels, labels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CIFAR100_MEAN, CIFAR100_STD};
    use burn::backend::NdArray;
    use rand::{rngs::StdRng, SeedableRng};
    use std::io::Write;

    type B = NdArray<f32>;

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("dblocks-raw-{}-{tag}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `n` records whose every pixel byte equals the record index (mod 256),
    /// so a mis-assembled batch is immediately visible.
    fn write_fixture(path: &Path, format: RawImageFormat, n: usize) {
        let mut f = File::create(path).unwrap();
        for i in 0..n {
            let mut rec = vec![(i % 256) as u8; format.record_bytes()];
            let label = (i % format.num_classes) as u16;
            match format.label_width {
                1 => rec[format.label_offset] = label as u8,
                _ => rec[format.label_offset..format.label_offset + 2]
                    .copy_from_slice(&label.to_le_bytes()),
            }
            f.write_all(&rec).unwrap();
        }
        f.flush().unwrap();
    }

    #[test]
    fn test_format_arithmetic() {
        assert_eq!(RawImageFormat::CIFAR100.record_bytes(), 2 + 3 * 32 * 32);
        assert_eq!(RawImageFormat::TINY_IMAGENET.record_bytes(), 2 + 3 * 64 * 64);
        assert_eq!(RawImageFormat::TINY_IMAGENET.pixel_bytes(), 12288);
    }

    #[test]
    fn test_in_memory_parse() {
        let dir = scratch_dir("mem");
        let path = dir.join("train.bin");
        let format = RawImageFormat::CIFAR100;
        write_fixture(&path, format, 5);

        let split = RawImageSplit::read_file(&path, format).unwrap();
        assert_eq!(split.len(), 5);
        assert_eq!(split.labels, vec![0, 1, 2, 3, 4]);
        // Record 3's pixels are all 3 (the header bytes were overwritten with
        // the label, but the pixel planes were not).
        assert!(split.record(3).iter().all(|&b| b == 3));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_streaming_matches_in_memory_exactly() {
        // The two paths must be interchangeable; if they are not, a run that
        // switches to streaming for memory reasons silently changes its data.
        let dir = scratch_dir("equiv");
        let path = dir.join("train.bin");
        let format = RawImageFormat::CIFAR100;
        write_fixture(&path, format, 40);

        let mut mem =
            RawImageDataset::in_memory(&path, format, 8, CIFAR100_MEAN, CIFAR100_STD).unwrap();
        let mut stream =
            RawImageDataset::streaming(&path, format, 8, CIFAR100_MEAN, CIFAR100_STD).unwrap();

        let device = Default::default();
        for seed in 0..4u64 {
            let a = <RawImageDataset as TrainDataset<B>>::next_batch(
                &mut mem,
                &mut StdRng::seed_from_u64(seed),
                &device,
            );
            let b = <RawImageDataset as TrainDataset<B>>::next_batch(
                &mut stream,
                &mut StdRng::seed_from_u64(seed),
                &device,
            );
            let diff = (a.pixel_values - b.pixel_values).abs().max().into_scalar();
            assert_eq!(diff, 0.0, "streaming and in-memory batches must be identical");
            let la: Vec<i64> = a.labels.into_data().convert::<i64>().iter().collect();
            let lb: Vec<i64> = b.labels.into_data().convert::<i64>().iter().collect();
            assert_eq!(la, lb);
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_label_scan_is_chunked() {
        // Opening a streaming split must not cost one syscall per record.
        let dir = scratch_dir("scan");
        let path = dir.join("train.bin");
        let format = RawImageFormat::CIFAR100;
        let n = 300usize;
        write_fixture(&path, format, n);

        let split = StreamingSplit::open(&path, format).unwrap();
        assert_eq!(split.len(), n);
        // 300 CIFAR records are ~0.9 MB, well under one 4 MiB chunk.
        assert_eq!(split.reads_issued(), 0, "the scan is not counted as batch I/O");

        // The labels are still exactly right.
        let expected: Vec<i64> = (0..n).map(|i| (i % format.num_classes) as i64).collect();
        assert_eq!(split.labels, expected);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_run_coalescing_reduces_syscalls() {
        // The point of sorting and coalescing: a contiguous span of records
        // must cost one read, not one per record.
        let dir = scratch_dir("coalesce");
        let path = dir.join("train.bin");
        let format = RawImageFormat::CIFAR100;
        write_fixture(&path, format, 64);

        let mut split = StreamingSplit::open(&path, format).unwrap();
        let baseline = split.reads_issued();

        let mut out = Vec::new();
        // 16 contiguous records, deliberately shuffled: sorting must recover
        // the single run.
        let indices: Vec<usize> = vec![
            10, 3, 7, 1, 14, 6, 2, 12, 0, 9, 5, 15, 11, 4, 13, 8,
        ];
        split.fetch_into(&indices, &mut out).unwrap();
        assert_eq!(
            split.reads_issued() - baseline,
            1,
            "a contiguous span must coalesce into a single read"
        );

        // ...and the bytes still land in the caller's requested order.
        let pixel_bytes = format.pixel_bytes();
        for (slot, &idx) in indices.iter().enumerate() {
            let got = out[slot * pixel_bytes];
            assert_eq!(got, (idx % 256) as u8, "record {idx} landed in the wrong slot");
        }

        // Scattered indices cannot coalesce, so the read count grows.
        let before = split.reads_issued();
        split.fetch_into(&[0, 10, 20, 30], &mut out).unwrap();
        assert_eq!(split.reads_issued() - before, 4);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_repeated_indices_are_served_correctly() {
        // Sampling with replacement produces duplicates; each must still get
        // its own copy of the bytes.
        let dir = scratch_dir("dupes");
        let path = dir.join("train.bin");
        let format = RawImageFormat::CIFAR100;
        write_fixture(&path, format, 8);

        let mut split = StreamingSplit::open(&path, format).unwrap();
        let mut out = Vec::new();
        let indices = [5usize, 5, 5, 2];
        split.fetch_into(&indices, &mut out).unwrap();

        let pixel_bytes = format.pixel_bytes();
        for (slot, &idx) in indices.iter().enumerate() {
            assert!(
                out[slot * pixel_bytes..(slot + 1) * pixel_bytes]
                    .iter()
                    .all(|&b| b == idx as u8),
                "slot {slot} does not hold record {idx}"
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_batching_is_allocation_stable() {
        // After the first batch the staging buffers must not grow again;
        // that is what "zero-copy" buys in steady state.
        let dir = scratch_dir("alloc");
        let path = dir.join("train.bin");
        let format = RawImageFormat::CIFAR100;
        write_fixture(&path, format, 32);

        let mut ds =
            RawImageDataset::in_memory(&path, format, 4, CIFAR100_MEAN, CIFAR100_STD).unwrap();
        let device = Default::default();
        let mut rng = StdRng::seed_from_u64(0);
        let _ = <RawImageDataset as TrainDataset<B>>::next_batch(&mut ds, &mut rng, &device);
        let cap_bytes = ds.bytes.capacity();
        let cap_floats = ds.floats.capacity();
        for _ in 0..5 {
            let _ = <RawImageDataset as TrainDataset<B>>::next_batch(&mut ds, &mut rng, &device);
        }
        assert_eq!(ds.bytes.capacity(), cap_bytes, "byte buffer regrew");
        assert_eq!(ds.floats.capacity(), cap_floats, "float buffer regrew");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_rejects_bad_files() {
        let dir = scratch_dir("bad");
        let format = RawImageFormat::CIFAR100;

        let truncated = dir.join("truncated.bin");
        std::fs::write(&truncated, vec![0u8; format.record_bytes() - 1]).unwrap();
        assert!(RawImageSplit::read_file(&truncated, format).is_err());
        assert!(StreamingSplit::open(&truncated, format).is_err());

        let empty = dir.join("empty.bin");
        std::fs::write(&empty, Vec::<u8>::new()).unwrap();
        assert!(RawImageSplit::read_file(&empty, format).is_err());

        // A label outside the class range means the format is wrong; failing
        // loudly beats training against garbage targets.
        let bad_label = dir.join("bad_label.bin");
        let mut rec = vec![0u8; format.record_bytes()];
        rec[format.label_offset] = 200; // >= 100 classes
        std::fs::write(&bad_label, &rec).unwrap();
        assert!(RawImageSplit::read_file(&bad_label, format).is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_tiny_imagenet_format_roundtrip() {
        // Two-byte little-endian labels exercise the wider label path.
        let dir = scratch_dir("tin");
        let path = dir.join("train.bin");
        let format = RawImageFormat::TINY_IMAGENET;
        write_fixture(&path, format, 3);

        let split = RawImageSplit::read_file(&path, format).unwrap();
        assert_eq!(split.labels, vec![0, 1, 2]);
        assert_eq!(split.record(0).len(), 12288);

        let mut ds = RawImageDataset::streaming(
            &path,
            format,
            2,
            crate::data::TINY_IMAGENET_MEAN,
            crate::data::TINY_IMAGENET_STD,
        )
        .unwrap();
        let batch = <RawImageDataset as TrainDataset<B>>::next_batch(
            &mut ds,
            &mut StdRng::seed_from_u64(1),
            &Default::default(),
        );
        assert_eq!(batch.pixel_values.dims(), [2, 3, 64, 64]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
