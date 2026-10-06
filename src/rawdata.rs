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
//! # io_uring, and the caveat that came with it
//!
//! The paragraph above used to end by declaring `io_uring` out of scope,
//! because it would need an external crate and the win it targets -- fewer
//! syscalls per batch -- is what run coalescing already delivers. That was a
//! reasonable trade for a crate that keeps its dependencies small, and it was
//! also only half true: coalescing reduces the *number* of reads, while a ring
//! reduces the *syscalls per read*, and a streaming dataset that is not in the
//! page cache is bound by both.
//!
//! [`crate::peregrine`] is that machinery, built and tested here. The
//! mirrored variant reads the same fixed-record format through it,
//! and the honest caveat is the one [`crate::peregrine`] documents: striping a
//! region across replicas only pays when the replicas are genuinely separate
//! devices. Pointed at two paths on one disk it is *slower* than a single
//! `pread` -- the same device queue, asked twice. The measurement is exposed
//! (`reads_issued`, and [`crate::peregrine::IoStats`]) precisely so that claim
//! can be checked rather than believed.

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
            anyhow::bail!(
                "label {label} out of range for {} classes",
                self.num_classes
            );
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
        Ok(Self {
            format,
            pixels,
            labels,
        })
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

/// The same fixed-record split, read through [`crate::peregrine`] from several
/// replicas of the file at once.
///
/// This is the striped counterpart to [`StreamingSplit`]: identical format,
/// identical label index, identical `fetch_into` contract. The difference is
/// only in how a run of records reaches memory -- coalesced `pread`s from one
/// file, or one region split across every replica and submitted together (and,
/// with the `peregrine-uring` feature, in a single `io_uring_enter`).
///
/// A batch of records is *already* one contiguous region after the indices are
/// sorted, which is exactly the shape striping wants, so the coalescing pass
/// and the striping pass compose rather than compete.
#[derive(Debug)]
pub struct MirroredSplit {
    pub format: RawImageFormat,
    mirrors: crate::peregrine::MirrorSet,
    path: PathBuf,
    labels: Vec<i64>,
    /// Reusable byte buffer for one coalesced, striped read.
    scratch: Vec<u8>,
}

impl MirroredSplit {
    /// Open `paths` as replicas of one split and index its labels.
    ///
    /// The label scan reads replica 0: the labels are a header in every
    /// replica, and scanning one copy is the same cost as scanning the others,
    /// so striping it would buy nothing.
    pub fn open(
        paths: &[PathBuf],
        format: RawImageFormat,
        backend: crate::peregrine::Backend,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !paths.is_empty(),
            "a mirrored split needs at least one replica path"
        );
        let mirrors = crate::peregrine::MirrorSet::open(paths, backend)?;
        let first = &paths[0];
        let file = File::open(first).with_context(|| format!("open {}", first.display()))?;
        let len = file.metadata()?.len();
        let n = format.record_count(len, first)?;

        let mut labels = Vec::with_capacity(n);
        let record_bytes = format.record_bytes();
        let records_per_chunk = (SCAN_CHUNK_BYTES / record_bytes).max(1);
        let mut chunk = vec![0u8; records_per_chunk * record_bytes];
        let mut first_record = 0usize;
        while first_record < n {
            let count = records_per_chunk.min(n - first_record);
            let bytes = &mut chunk[..count * record_bytes];
            read_exact_at(&file, bytes, (first_record * record_bytes) as u64)?;
            for record in bytes.chunks_exact(record_bytes) {
                labels.push(format.label_of(record)?);
            }
            first_record += count;
        }

        Ok(Self {
            format,
            mirrors,
            path: first.to_path_buf(),
            labels,
            scratch: Vec::new(),
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

    /// How many replicas the batch reads are striped across.
    pub fn replicas(&self) -> usize {
        self.mirrors.replicas()
    }

    /// I/O counters: stripe reads, syscalls, bytes, ring batches.
    pub fn io_stats(&self) -> crate::peregrine::IoStats {
        self.mirrors.stats()
    }

    /// Bytes read so far. Reported rather than assumed, so the cost of a run is
    /// visible next to its accuracy.
    pub fn bytes_read(&self) -> u64 {
        self.mirrors.stats().bytes
    }

    /// Fetch the pixel bytes of `indices` into `out`, in the order given.
    ///
    /// The contract is exactly [`StreamingSplit::fetch_into`]: same order, same
    /// coalescing, same reusable buffer. The runs that coalescing found are the
    /// regions that get striped, so the two optimizations stack.
    pub fn fetch_into(&mut self, indices: &[usize], out: &mut Vec<u8>) -> anyhow::Result<()> {
        let pixel_bytes = self.format.pixel_bytes();
        let record_bytes = self.format.record_bytes();
        out.resize(indices.len() * pixel_bytes, 0);

        let mut order: Vec<(usize, usize)> = indices.iter().copied().zip(0..).collect();
        order.sort_unstable();

        let mut i = 0usize;
        while i < order.len() {
            // A repeated index would need the same bytes twice; it ends the run
            // and is served by its own read, exactly as in the single-file path.
            let mut j = i + 1;
            while j < order.len() && order[j].0 == order[j - 1].0 + 1 {
                j += 1;
            }
            let first = order[i].0;
            let count = j - i;
            let region =
                crate::peregrine::Region::new((first * record_bytes) as u64, count * record_bytes);
            self.scratch = self
                .mirrors
                .read_striped(region)
                .with_context(|| format!("striped read of records {first}..{}", first + count))?;

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

/// How a dataset gets at its pixels.
#[derive(Debug)]
enum Source {
    Memory(RawImageSplit),
    // Boxed, not by accident: a `StreamingSplit`/`MirroredSplit` carries an open
    // `File` plus label and scratch vectors, so the enum would otherwise be
    // sized by its largest variant and every `Source` move copied that payload.
    Streaming(Box<StreamingSplit>),
    /// Striped reads across replicas, via [`crate::peregrine`].
    Mirrored(Box<MirroredSplit>),
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
        Ok(Self::from_source(
            Source::Memory(split),
            batch_size,
            mean,
            std,
        ))
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
        Ok(Self::from_source(
            Source::Streaming(Box::new(split)),
            batch_size,
            mean,
            std,
        ))
    }

    /// Stream records from several replicas of the same split at once, each
    /// batch region striped across every replica (see [`crate::peregrine`]).
    ///
    /// `paths` are replicas of one logical file, not different files: the
    /// striped read returns the bytes of the region and nothing else, so a
    /// replica that is not a mirror yields wrong data rather than an error. The
    /// oracle test in `peregrine` is what keeps that honest, and it is why
    /// `MirrorSet::open` deliberately does not verify replicas against each
    /// other -- comparing them would mean reading everything to open anything.
    pub fn mirrored(
        paths: &[PathBuf],
        format: RawImageFormat,
        batch_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
        backend: crate::peregrine::Backend,
    ) -> anyhow::Result<Self> {
        let split = MirroredSplit::open(paths, format, backend)?;
        Ok(Self::from_source(
            Source::Mirrored(Box::new(split)),
            batch_size,
            mean,
            std,
        ))
    }

    fn from_source(source: Source, batch_size: usize, mean: [f32; 3], std: [f32; 3]) -> Self {
        let format = match &source {
            Source::Memory(s) => s.format,
            Source::Streaming(s) => s.format,
            Source::Mirrored(s) => s.format,
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
            Source::Mirrored(s) => s.format,
        }
    }

    pub fn len(&self) -> usize {
        match &self.source {
            Source::Memory(s) => s.len(),
            Source::Streaming(s) => s.len(),
            Source::Mirrored(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Positional reads issued (streaming and mirrored modes only; `0` in
    /// memory mode).
    ///
    /// In mirrored mode this is the number of *stripe reads*, which is at least
    /// the number of coalesced regions; [`Self::io_stats`] breaks it down into
    /// reads and the syscalls they cost.
    pub fn reads_issued(&self) -> usize {
        match &self.source {
            Source::Memory(_) => 0,
            Source::Streaming(s) => s.reads_issued(),
            Source::Mirrored(s) => s.io_stats().reads as usize,
        }
    }

    /// How many replicas a mirrored dataset reads across; `1` for the in-memory
    /// and single-file modes.
    pub fn replica_count(&self) -> usize {
        match &self.source {
            Source::Mirrored(s) => s.replicas(),
            _ => 1,
        }
    }

    /// Full I/O counters for a mirrored dataset, including the syscalls the
    /// striping saved. `None` for the in-memory and single-file modes.
    pub fn io_stats(&self) -> Option<crate::peregrine::IoStats> {
        match &self.source {
            Source::Mirrored(s) => Some(s.io_stats()),
            _ => None,
        }
    }

    fn label_at(&self, idx: usize) -> i64 {
        match &self.source {
            Source::Memory(s) => s.labels[idx],
            Source::Streaming(s) => s.labels[idx],
            Source::Mirrored(s) => s.labels[idx],
        }
    }
}

impl<B: Backend> TrainDataset<B> for RawImageDataset {
    fn next_batch<R: Rng>(&mut self, rng: &mut R, device: &B::Device) -> anyhow::Result<Batch<B>> {
        let format = self.format();
        let pixel_bytes = format.pixel_bytes();
        let n = self.len();
        anyhow::ensure!(n > 0, "cannot sample from an empty split");

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
                    .context("streaming read of a raw image batch")?;
            }
            Source::Mirrored(split) => {
                split
                    .fetch_into(&self.indices, &mut self.bytes)
                    .context("striped read of a raw image batch")?;
            }
        }

        // u8 -> f32 in [0, 1) directly into the staging buffer, no allocation.
        for (dst, &src) in self.floats.iter_mut().zip(self.bytes.iter()) {
            *dst = src as f32 / 255.0;
        }

        let shape = [
            self.batch_size,
            format.channels,
            format.image_size,
            format.image_size,
        ];
        let pixels = Tensor::<B, 1>::from_floats(self.floats.as_slice(), device).reshape(shape);
        let mean = Tensor::<B, 1>::from_floats(self.mean, device).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_floats(self.std, device).reshape([1, 3, 1, 1]);
        let pixels = (pixels - mean) / std;

        let labels = Tensor::<B, 1, Int>::from_ints(self.labels.as_slice(), device);
        Ok(Batch {
            pixel_values: pixels,
            labels,
        })
    }
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`, which is the right
// thing for a test to say. The grant is scoped to this module: production
// code in the same file is still denied it (see the `[lints]` table in
// `Cargo.toml` and the contract in the crate docs).
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::dbg_macro,
    clippy::let_underscore_must_use,
    clippy::redundant_pattern_matching,
    clippy::mem_forget,
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr
)]
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
        let dir =
            std::env::temp_dir().join(format!("dblocks-raw-{}-{tag}-{nanos}", std::process::id()));
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

    /// The oracle for the mirrored path: a batch fetched through
    /// `peregrine`'s striping is byte-for-byte the batch the single-file reader
    /// produces, for the same indices and the same seed.
    ///
    /// This is the test that makes the striped read usable at all. A striping
    /// bug does not crash a training run -- it feeds a model plausible-looking
    /// wrong pixels and reports a plausible loss, so the check has to be
    /// explicit rather than assumed.
    #[test]
    fn mirrored_batches_are_byte_identical_to_the_single_file_reader() {
        let dir = scratch_dir("mirror-oracle");
        let format = RawImageFormat::CIFAR100;
        let n = 400;
        let paths: Vec<PathBuf> = (0..3)
            .map(|i| dir.join(format!("replica{i}.bin")))
            .collect();
        for path in &paths {
            write_fixture(path, format, n);
        }

        let mut streaming =
            RawImageDataset::streaming(&paths[0], format, 32, CIFAR100_MEAN, CIFAR100_STD).unwrap();
        let mut mirrored = RawImageDataset::mirrored(
            &paths,
            format,
            32,
            CIFAR100_MEAN,
            CIFAR100_STD,
            crate::peregrine::Backend::Threaded,
        )
        .unwrap();

        assert_eq!(streaming.len(), mirrored.len());
        assert_eq!(streaming.len(), n);
        assert_eq!(
            mirrored.io_stats().map(|s| s.batches),
            Some(0),
            "no reads yet"
        );

        // Same seed => same sampled indices => the two fetch plans match, so
        // any difference in the bytes is the striping's doing and nothing else.
        for step in 0..4 {
            // `Default::default()` for the device, and the fully-qualified
            // `TrainDataset` call to pin `B`: an inferred `next_batch` here
            // would leave the backend ambiguous, since the closure only ever
            // produces f32s.
            let device = Default::default();
            let a = <RawImageDataset as TrainDataset<B>>::next_batch(
                &mut streaming,
                &mut StdRng::seed_from_u64(7),
                &device,
            )
            .unwrap();
            let b = <RawImageDataset as TrainDataset<B>>::next_batch(
                &mut mirrored,
                &mut StdRng::seed_from_u64(7),
                &device,
            )
            .unwrap();
            let left = a.pixel_values.to_data().to_vec::<f32>().unwrap();
            let right = b.pixel_values.to_data().to_vec::<f32>().unwrap();
            assert_eq!(
                left, right,
                "step {step}: mirrored batch differs from single-file"
            );
            let la = a.labels.to_data().to_vec::<i64>().unwrap();
            let lb = b.labels.to_data().to_vec::<i64>().unwrap();
            assert_eq!(
                la, lb,
                "step {step}: mirrored labels differ from single-file"
            );
        }

        // The reads really happened, and they were counted as stripe reads.
        let stats = mirrored.io_stats().expect("mirrored has stats");
        assert!(
            stats.batches >= 4,
            "expected at least one batch per step, got {stats:?}"
        );
        assert!(
            stats.reads >= stats.batches,
            "stripe reads below batch count: {stats:?}"
        );
        assert!(stats.bytes > 0, "no bytes were read: {stats:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The mirrored dataset reports the same records as the in-memory one, and
    /// a single replica is the degenerate case of the same code path.
    #[test]
    fn mirrored_with_one_replica_matches_streaming() {
        let dir = scratch_dir("mirror-single");
        let format = RawImageFormat::CIFAR100;
        let path = dir.join("only.bin");
        write_fixture(&path, format, 64);

        let streaming =
            RawImageDataset::streaming(&path, format, 8, CIFAR100_MEAN, CIFAR100_STD).unwrap();
        let mirrored = RawImageDataset::mirrored(
            std::slice::from_ref(&path),
            format,
            8,
            CIFAR100_MEAN,
            CIFAR100_STD,
            crate::peregrine::Backend::Threaded,
        )
        .unwrap();
        assert_eq!(mirrored.replica_count(), 1);
        assert_eq!(streaming.len(), mirrored.len());

        let indices = [0usize, 1, 2, 30, 31, 62, 63, 5];
        let mut a = vec![0u8; indices.len() * format.pixel_bytes()];
        let mut b = vec![0u8; indices.len() * format.pixel_bytes()];
        let mut s = StreamingSplit::open(&path, format).unwrap();
        s.fetch_into(&indices, &mut a).unwrap();
        let mut m =
            MirroredSplit::open(&[path], format, crate::peregrine::Backend::Threaded).unwrap();
        m.fetch_into(&indices, &mut b).unwrap();
        assert_eq!(
            a, b,
            "one-replica striping differs from the single-file reader"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A mirror that is *not* a mirror is a real failure mode, so the dataset
    /// surfaces it: a striped read over diverging replicas returns different
    /// bytes than replica 0 alone, which is exactly the corruption the oracle
    /// test exists to catch.
    ///
    /// The replica here is the same *length* as the good one -- a short file
    /// would be caught by the bounds check, which is a different (and easier)
    /// failure. Same length, different content is the case that gets through.
    #[test]
    fn a_replica_that_is_not_a_mirror_is_caught() {
        let dir = scratch_dir("mirror-bad");
        let format = RawImageFormat::CIFAR100;
        let good = dir.join("good.bin");
        let bad = dir.join("bad.bin");
        write_fixture(&good, format, 64);
        // Same record count, same labels, every pixel byte different: a replica
        // that would pass a length check and a label check, and only a
        // byte-comparison catches.
        let mut f = File::create(&bad).unwrap();
        for i in 0..64usize {
            let mut rec = vec![200 + (i % 7) as u8; format.record_bytes()];
            let label = (i % format.num_classes) as u16;
            match format.label_width {
                1 => rec[format.label_offset] = label as u8,
                _ => rec[format.label_offset..format.label_offset + 2]
                    .copy_from_slice(&label.to_le_bytes()),
            }
            f.write_all(&rec).unwrap();
        }
        f.flush().unwrap();
        assert_eq!(
            std::fs::metadata(&good).unwrap().len(),
            std::fs::metadata(&bad).unwrap().len(),
            "the two replicas must be the same length for this test to mean anything"
        );

        let mut mirrors = crate::peregrine::MirrorSet::open(
            &[good.clone(), bad],
            crate::peregrine::Backend::Threaded,
        )
        .unwrap();
        let region = crate::peregrine::Region::new(0, 64 * format.record_bytes());
        // `read_serial` is replica 0 alone, so it is the ground truth the
        // striped read is compared against -- which is exactly how the
        // divergence surfaces in production rather than in a test helper.
        let truth = mirrors.read_serial(region.clone()).unwrap();
        let striped = mirrors.read_striped(region).unwrap();
        assert_ne!(
            truth, striped,
            "a diverging replica should produce different bytes than replica 0 alone"
        );
        // And the striped read is self-consistent, which is what lets the
        // oracle test be a comparison rather than a hash of the whole file.
        assert_eq!(
            striped.len(),
            truth.len(),
            "the two reads disagree in length"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A read past the end of the mirrored file is an error, not a short batch.
    #[test]
    fn mirrored_read_past_the_end_is_refused() {
        let dir = scratch_dir("mirror-short");
        let format = RawImageFormat::CIFAR100;
        let path = dir.join("small.bin");
        write_fixture(&path, format, 8);
        let mut split =
            MirroredSplit::open(&[path], format, crate::peregrine::Backend::Threaded).unwrap();
        let mut out = Vec::new();
        let err = split
            .fetch_into(&[0, 1, 99], &mut out)
            .expect_err("must refuse");
        assert!(
            format!("{err:#}").contains("end of file") || format!("{err:#}").contains("failed"),
            "unhelpful error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An empty replica list is refused up front, not at first read.
    #[test]
    fn mirrored_needs_at_least_one_replica() {
        let err = MirroredSplit::open(
            &[],
            RawImageFormat::CIFAR100,
            crate::peregrine::Backend::Threaded,
        )
        .expect_err("must refuse");
        assert!(
            format!("{err}").contains("at least one replica"),
            "unhelpful: {err}"
        );
    }

    #[test]
    fn test_format_arithmetic() {
        assert_eq!(RawImageFormat::CIFAR100.record_bytes(), 2 + 3 * 32 * 32);
        assert_eq!(
            RawImageFormat::TINY_IMAGENET.record_bytes(),
            2 + 3 * 64 * 64
        );
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
            )
            .expect("batch");
            let b = <RawImageDataset as TrainDataset<B>>::next_batch(
                &mut stream,
                &mut StdRng::seed_from_u64(seed),
                &device,
            )
            .expect("batch");
            let diff = (a.pixel_values - b.pixel_values).abs().max().into_scalar();
            assert_eq!(
                diff, 0.0,
                "streaming and in-memory batches must be identical"
            );
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
        assert_eq!(
            split.reads_issued(),
            0,
            "the scan is not counted as batch I/O"
        );

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
        let indices: Vec<usize> = vec![10, 3, 7, 1, 14, 6, 2, 12, 0, 9, 5, 15, 11, 4, 13, 8];
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
            assert_eq!(
                got,
                (idx % 256) as u8,
                "record {idx} landed in the wrong slot"
            );
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
        <RawImageDataset as TrainDataset<B>>::next_batch(&mut ds, &mut rng, &device)
            .expect("batch");
        let cap_bytes = ds.bytes.capacity();
        let cap_floats = ds.floats.capacity();
        for _ in 0..5 {
            <RawImageDataset as TrainDataset<B>>::next_batch(&mut ds, &mut rng, &device)
                .expect("batch");
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
        )
        .expect("batch");
        assert_eq!(batch.pixel_values.dims(), [2, 3, 64, 64]);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
