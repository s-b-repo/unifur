//! safetensors reader for Phase B of `docs/Frontier-27B-Plan.md`.
//!
//! Reads HuggingFace `safetensors` weight files by tensor name and decodes
//! to `f32` for inspection, validation and (later phases) remapping into
//! Burn modules. Deliberately decoupled from Burn tensors: import is an I/O
//! and format problem, and the remap needs raw named tensors, not module
//! records.
//!
//! Format refresher (stable since 2022, no dependency needed):
//! `[u64 LE header_len][header_len bytes of JSON][raw tensor data]`, where
//! the JSON maps each name to `{dtype, shape, data_offsets}` and offsets are
//! relative to the start of the data section.
//!
//! Honest scope limits:
//!
//! - Decoded dtypes: `F32`, `F16`, `BF16`. Anything else (quantized,
//!   integer, bool) is a loud error naming the dtype, not a silent cast.
//!   Both float conversions are bit-exact (wider format absorbs the
//!   narrower one, NaN stays NaN).
//! - Reads are ranged (`File` + `Seek`): opening a 5 GB shard parses only
//!   the header, and decoding one tensor reads only its byte range. Nothing
//!   here ever loads a whole shard into RAM.
//! - No GGUF parsing (the training source is BF16 safetensors; GGUF quants
//!   are covered by `gguf-rs`-style crates, not reimplemented here).
//! - Name remapping into Qwen modules is the next step (needs real shard
//!   tensor names, inspected after download, not guessed here).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

/// Dtypes this reader decodes. Everything else errors loudly at header time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportDtype {
    F32,
    F16,
    BF16,
}

impl ImportDtype {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            other => anyhow::bail!(
                "unsupported safetensors dtype {other:?}: only F32/F16/BF16 decode (no silent casts)"
            ),
        }
    }

    /// Bytes per element.
    pub fn size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// One tensor's location inside a shard, with offsets relative to the start
/// of the data section (the format's convention).
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: ImportDtype,
    pub shape: Vec<usize>,
    /// `[start, end)` within the data section.
    pub start: u64,
    pub end: u64,
}

impl TensorMeta {
    /// Expected byte length from shape × dtype; `None` on arithmetic overflow.
    pub fn byte_len(&self) -> Option<u64> {
        let mut elements: u64 = 1;
        for dim in &self.shape {
            elements = elements.checked_mul(*dim as u64)?;
        }
        elements.checked_mul(self.dtype.size() as u64)
    }
}

#[derive(Debug, Deserialize)]
struct TensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

/// Header-only view of a shard: names, shapes and byte ranges, no weights.
#[derive(Debug, Clone)]
pub struct SafetensorsIndex {
    pub tensors: Vec<TensorMeta>,
    data_start: u64,
    file_len: u64,
}

impl SafetensorsIndex {
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn find(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total file bytes (header + data section), for accounting.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Total data-section bytes covered by tensor ranges (for accounting).
    pub fn covered_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.end.saturating_sub(t.start))
            .sum()
    }
}

/// Parse only the header of `path`: names, dtypes, shapes and offsets.
/// Every range is validated against the file length before anything trusts it.
pub fn open_index(path: &Path) -> Result<SafetensorsIndex> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .context("read header length")?;
    let header_len = u64::from_le_bytes(len_buf);
    let data_start = 8u64
        .checked_add(header_len)
        .context("header length overflows u64")?;
    ensure!(
        data_start <= file_len,
        "header ({} bytes) runs past end of {} ({} bytes)",
        data_start,
        path.display(),
        file_len
    );
    let mut header_buf = vec![0u8; header_len as usize];
    file.read_exact(&mut header_buf)
        .context("read header JSON")?;
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header_buf).context("parse header JSON")?;
    let mut tensors = Vec::new();
    for (name, value) in &raw {
        // Sidecar metadata (`__metadata__`) is not a tensor.
        if name == "__metadata__" {
            continue;
        }
        let entry: TensorEntry = serde_json::from_value(value.clone())
            .with_context(|| format!("parse header entry for tensor {name:?}"))?;
        let dtype = ImportDtype::parse(&entry.dtype)?;
        let (start, end) = entry.data_offsets;
        ensure!(
            start <= end,
            "tensor {name:?} has inverted offsets [{start}, {end})"
        );
        ensure!(
            data_start.saturating_add(end) <= file_len,
            "tensor {name:?} range [{start}, {end}) runs past end of file"
        );
        let meta = TensorMeta {
            name: name.clone(),
            dtype,
            shape: entry.shape,
            start,
            end,
        };
        match meta.byte_len() {
            Some(want) => ensure!(
                end - start == want,
                "tensor {name:?} range length {} != shape×dtype {}",
                end - start,
                want
            ),
            None => anyhow::bail!("tensor {name:?} shape overflows u64"),
        }
        tensors.push(meta);
    }
    ensure!(!tensors.is_empty(), "no tensors in {}", path.display());
    Ok(SafetensorsIndex {
        tensors,
        data_start,
        file_len,
    })
}

/// A decoded tensor: row-major `f32` values with the file's shape.
#[derive(Debug, Clone)]
pub struct DecodedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Decode one named tensor to `f32`, reading only its byte range.
pub fn read_tensor_f32(path: &Path, index: &SafetensorsIndex, name: &str) -> Result<DecodedTensor> {
    let meta = index
        .find(name)
        .with_context(|| format!("tensor {name:?} not in {}", path.display()))?;
    let len = (meta.end - meta.start) as usize;
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    file.seek(SeekFrom::Start(index.data_start + meta.start))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)
        .with_context(|| format!("read {} bytes for tensor {name:?}", len))?;
    let data = match meta.dtype {
        ImportDtype::F32 => {
            ensure!(
                len % 4 == 0,
                "F32 tensor {name:?} has non-multiple-of-4 length"
            );
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        ImportDtype::F16 => {
            ensure!(len % 2 == 0, "F16 tensor {name:?} has odd length");
            buf.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        ImportDtype::BF16 => {
            ensure!(len % 2 == 0, "BF16 tensor {name:?} has odd length");
            buf.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
    };
    Ok(DecodedTensor {
        name: name.to_string(),
        shape: meta.shape.clone(),
        data,
    })
}

/// IEEE-754 binary16 -> f32. Exact: every f16 value is representable in f32,
/// including subnormals, infinities and NaN payloads (quieted).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out: u32 = match exp {
        0 => {
            // Zero or subnormal: renormalize into f32 range. Value is
            // mant x 2^-24; shifting mant left until bit 10 is set keeps
            // `m x 2^(e-127)` invariant with e starting at 127-14 (then one
            // less per shift), and the leading 1 goes implicit.
            if mant == 0 {
                sign << 31
            } else {
                let mut m = mant;
                let mut e = 127u32.wrapping_sub(14);
                while m & 0x400 == 0 {
                    m <<= 1;
                    e = e.wrapping_sub(1);
                }
                m &= 0x3ff;
                (sign << 31) | (e << 23) | (m << 13)
            }
        }
        0x1f => {
            // Infinity (zero payload) or NaN (payload preserved, quiet bit
            // forced so a signaling input cannot trap downstream).
            let payload = if mant == 0 {
                0
            } else {
                ((mant << 13) | 0x0040_0000) & 0x007F_FFFF
            };
            (sign << 31) | (0xff << 23) | payload
        }
        _ => (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

/// bfloat16 -> f32. Exact: bf16 is the top 16 bits of f32, so the value is
/// preserved bit-for-bit (NaN stays NaN).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
    use std::io::Write;

    use std::sync::atomic::{AtomicU64, Ordering};

    /// Fixture counter: parallel tests sharing one temp filename raced
    /// (one test deleting another's file), so every fixture gets a unique
    /// name even within one process.
    static FIXTURE_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Write a minimal safetensors file: header JSON + raw LE data section.
    /// `tensors` is `(name, dtype_str, shape, raw_data_bytes)`.
    fn write_fixture(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let end = offset + bytes.len() as u64;
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
            data.extend_from_slice(bytes);
        }
        let header_str = serde_json::to_string(&header).unwrap();
        let path = std::env::temp_dir().join(format!(
            "dblocks-import-fixture-{}-{}.st",
            std::process::id(),
            FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header_str.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_str.as_bytes()).unwrap();
        file.write_all(&data).unwrap();
        path
    }

    fn f32le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn u16le(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn test_bf16_bit_patterns_are_exact() {
        // Well-known encodings, not computed values.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x8000), -0.0);
        assert_eq!(bf16_to_f32(0x7F80), f32::INFINITY);
        assert_eq!(bf16_to_f32(0xFF80), f32::NEG_INFINITY);
        assert!(bf16_to_f32(0x7FC0).is_nan());
        // Round-trip on exactly-representable values.
        for v in [0.5f32, -3.25, 100.0, 1e-6, 1e30] {
            let back = bf16_to_f32((v.to_bits() >> 16) as u16);
            assert_eq!(
                back.to_bits(),
                (v.to_bits() >> 16) << 16,
                "bf16 round-trip of {v}"
            );
        }
    }

    #[test]
    fn test_f16_covers_subnormals_infinities_and_nan() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        // Smallest subnormal 2^-24 lands exactly in f32 range.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        // Largest finite 65504.
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    }

    #[test]
    fn test_header_indexes_without_touching_weights() {
        let path = write_fixture(&[
            ("w.f32", "F32", vec![2, 2], f32le(&[1.0, 2.0, 3.0, 4.0])),
            ("w.bf16", "BF16", vec![3], u16le(&[0x3F80, 0xC000, 0x7F80])),
            ("w.f16", "F16", vec![2], u16le(&[0x3C00, 0x0001])),
        ]);
        let index = open_index(&path).unwrap();
        assert_eq!(index.len(), 3);
        let meta = index.find("w.bf16").unwrap();
        assert_eq!(meta.shape, vec![3]);
        assert_eq!(meta.dtype, ImportDtype::BF16);
        assert!(index.find("missing").is_none());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_ranged_reads_decode_all_three_dtypes() {
        let path = write_fixture(&[
            ("w.f32", "F32", vec![2, 2], f32le(&[1.0, 2.0, 3.0, 4.0])),
            ("w.bf16", "BF16", vec![3], u16le(&[0x3F80, 0xC000, 0x7F80])),
            ("w.f16", "F16", vec![2], u16le(&[0x3C00, 0x0001])),
        ]);
        let index = open_index(&path).unwrap();
        let f32t = read_tensor_f32(&path, &index, "w.f32").unwrap();
        assert_eq!(f32t.shape, vec![2, 2]);
        assert_eq!(f32t.data, vec![1.0, 2.0, 3.0, 4.0]);
        let bf16t = read_tensor_f32(&path, &index, "w.bf16").unwrap();
        assert_eq!(bf16t.data, vec![1.0, -2.0, f32::INFINITY]);
        let f16t = read_tensor_f32(&path, &index, "w.f16").unwrap();
        assert_eq!(f16t.data, vec![1.0, 2f32.powi(-24)]);
        // Missing-name and shape queries fail loudly.
        assert!(read_tensor_f32(&path, &index, "nope").is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_corrupt_headers_are_refused_not_misread() {
        // Unknown dtype.
        let path = write_fixture(&[("w.q", "Q4_K", vec![4], vec![0u8; 4])]);
        assert!(open_index(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        // Range longer than shape x dtype.
        let path = write_fixture(&[("w.bad", "F32", vec![2], f32le(&[1.0, 2.0, 3.0]))]);
        assert!(open_index(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        // Truncated file (header claims more than exists).
        let full = write_fixture(&[("w.ok", "F32", vec![2], f32le(&[1.0, 2.0]))]);
        let bytes = std::fs::read(&full).unwrap();
        let cut =
            std::env::temp_dir().join(format!("dblocks-import-cut-{}.st", std::process::id()));
        std::fs::write(&cut, &bytes[..bytes.len() - 3]).unwrap();
        assert!(open_index(&cut).is_err());
        std::fs::remove_file(&full).unwrap();
        std::fs::remove_file(&cut).unwrap();
    }
}
