//! Pre-tokenized text corpora (roadmap 19.2).
//!
//! A corpus is a flat little-endian `u16` file with no header: token `i` lives
//! at byte offset `2i`. That is the same shape as the fixed-record image
//! datasets in [`crate::rawdata`] with a record size of two bytes, and it is
//! chosen for the same reason — a fixed stride means any window can be read
//! with one seek, so a corpus larger than memory costs no more per batch than
//! one that fits.
//!
//! # Why `u16` and not `u8`
//!
//! [`crate::tokenizer`] has 259 tokens: the 256 byte values plus three
//! specials. The specials are what mark document boundaries, and a corpus
//! without them would train the model to run one document straight into the
//! next. Two bytes per token is the price of being able to say where a document
//! ends. Text is stored uncompressed either way, so a corpus is exactly twice
//! the size of its source — worth knowing before tokenizing a large one.
//!
//! # No header
//!
//! A header would carry the tokenizer version and the vocabulary size, which is
//! genuinely useful. It is omitted because the byte-level tokenizer has no
//! vocabulary file to drift from: the mapping is fixed by the format itself and
//! cannot change without changing this crate. If a learned tokenizer is ever
//! added, a header becomes necessary rather than merely nice.
//!
//! # Labels (roadmap Phase 24)
//!
//! A corpus may carry a sidecar `book.labels`: one `u8` per token, the same
//! index space as the tokens, holding the anti-pattern category of that token
//! or [`crate::antipattern::CLEAN`]. It is read with the same one-seek window
//! as the tokens, so a labeled window costs two reads rather than one. What the
//! label bytes *mean* — category names and penalty weights — is written next to
//! them in `book.labels.json` ([`crate::antipattern::LabelManifest`]), so the
//! corpus can be trained on without the rule set that produced it.
//!
//! Labels are computed over the whole corpus at once, not per window: a rule
//! whose match would straddle a window boundary is still found, because the
//! window is cut *after* labeling.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::antipattern::{LabelManifest, Labeler};
use crate::tokenizer::ByteTokenizer;

/// Bytes per token on disk.
pub const TOKEN_BYTES: usize = 2;

/// Bytes per label on disk.
pub const LABEL_BYTES: usize = 1;

/// `book.bin` -> `book.labels`.
pub fn labels_path(corpus: &Path) -> PathBuf {
    corpus.with_extension("labels")
}

/// `book.bin` -> `book.labels.json`.
pub fn manifest_path(corpus: &Path) -> PathBuf {
    corpus.with_extension("labels.json")
}

/// `book.bin` -> `book.mask`.
pub fn mask_path(corpus: &Path) -> PathBuf {
    corpus.with_extension("mask")
}

/// The windows of a batch and, aligned with them, their label bytes.
pub type LabeledBatch = (Vec<Vec<u16>>, Vec<Vec<u8>>);

/// Where a corpus reads from.
#[derive(Debug)]
enum Source {
    /// The whole corpus, resident.
    Memory(Vec<u16>),
    /// A handle plus a reusable buffer; windows are read on demand.
    Streaming {
        file: File,
        scratch: Vec<u8>,
        reads_issued: usize,
    },
}

/// Where a corpus's labels read from; mirrors [`Source`].
#[derive(Debug)]
enum LabelSource {
    Memory(Vec<u8>),
    Streaming { file: File, scratch: Vec<u8> },
}

#[derive(Debug)]
enum MaskSource {
    Memory(Vec<u8>),
    Streaming { file: File, scratch: Vec<u8> },
}

#[derive(Debug)]
enum GradeSource {
    Memory(Vec<i8>),
    Streaming { file: File, scratch: Vec<u8> },
}

/// A pre-tokenized corpus, addressed by token index.
#[derive(Debug)]
pub struct TokenCorpus {
    source: Source,
    /// Present once [`TokenCorpus::open_labels`] has succeeded.
    labels: Option<LabelSource>,
    masks: Option<MaskSource>,
    /// Present once [`TokenCorpus::open_grades`] has succeeded.
    grades: Option<GradeSource>,
    path: PathBuf,
    /// Tokens in the file.
    len: usize,
}

/// One sampled batch plus the optional per-token side tables: labels, masks,
/// grades. Named so that signatures stay readable instead of spelling out the
/// four-tuple.
pub type BatchWithSides = (
    Vec<Vec<u16>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<u8>>>,
    Option<Vec<Vec<i8>>>,
);

/// A sampled batch of token windows and the mask bytes marking which tokens in
/// each window were masked out.
pub type MaskedBatch = (Vec<Vec<u16>>, Vec<Vec<u8>>);

impl TokenCorpus {
    /// Write `tokens` to `path` in the corpus format.
    pub fn write(path: &Path, tokens: &[u16]) -> Result<()> {
        let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut bytes = Vec::with_capacity(tokens.len() * TOKEN_BYTES);
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        file.write_all(&bytes)
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    /// Tokenize `input` as UTF-8 text and write the corpus to `output`.
    ///
    /// The whole document is wrapped in `<bos>` / `<eos>`. Returns the token
    /// count written.
    pub fn tokenize_file(input: &Path, output: &Path) -> Result<usize> {
        let text =
            std::fs::read_to_string(input).with_context(|| format!("read {}", input.display()))?;
        let tokens = ByteTokenizer::new().encode_document(&text);
        Self::write(output, &tokens)?;
        Ok(tokens.len())
    }

    /// Write one label byte per token to `path`.
    pub fn write_labels(path: &Path, labels: &[u8]) -> Result<()> {
        std::fs::write(path, labels).with_context(|| format!("write {}", path.display()))
    }

    pub fn write_masks(path: &Path, masks: &[u8]) -> Result<()> {
        std::fs::write(path, masks).with_context(|| format!("write {}", path.display()))
    }

    /// Label every token of the corpus at `corpus` with `labeler`, writing the
    /// `.labels` sidecar and its `.labels.json` manifest next to it.
    ///
    /// The corpus is loaded into memory for this: a rule is allowed to match
    /// across any distance, so there is no chunk size that would be safe.
    /// Labeling is a one-off preprocessing step, like tokenizing, and the
    /// text was in memory for that too. Returns the manifest.
    pub fn label_file(corpus: &Path, labeler: &Labeler) -> Result<LabelManifest> {
        let mut resident = Self::in_memory(corpus)?;
        let len = resident.len();
        let tokens = resident.window(0, len)?;
        let (labels, spans) = labeler.label_with_spans(&tokens);
        let manifest = labeler.manifest(&labels, &spans);
        Self::write_labels(&labels_path(corpus), &labels)?;
        manifest.write(&manifest_path(corpus))?;
        Ok(manifest)
    }

    /// Open the `.labels` sidecar in the same mode as the tokens.
    ///
    /// # Errors
    ///
    /// If the sidecar is missing, or holds a different number of labels than
    /// the corpus has tokens — which means it was made from a different
    /// corpus, and every label after the first divergence would land on the
    /// wrong token.
    pub fn open_labels(&mut self) -> Result<()> {
        let path = labels_path(&self.path);
        anyhow::ensure!(
            path.exists(),
            "{} has no label sidecar {}; run `dblocks lm label --corpus {}` first",
            self.path.display(),
            path.display(),
            self.path.display()
        );
        let bytes = std::fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len() as usize;
        anyhow::ensure!(
            bytes / LABEL_BYTES == self.len,
            "{} holds {} labels but {} holds {} tokens: the sidecar was made from a different corpus",
            path.display(),
            bytes / LABEL_BYTES,
            self.path.display(),
            self.len
        );
        self.labels = Some(match self.source {
            Source::Memory(_) => LabelSource::Memory(
                std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
            ),
            Source::Streaming { .. } => LabelSource::Streaming {
                file: File::open(&path).with_context(|| format!("open {}", path.display()))?,
                scratch: Vec::new(),
            },
        });
        Ok(())
    }

    pub fn open_mask(&mut self) -> Result<()> {
        let path = mask_path(&self.path);
        anyhow::ensure!(
            path.exists(),
            "{} has no mask sidecar {}",
            self.path.display(),
            path.display()
        );
        let bytes = std::fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len() as usize;
        anyhow::ensure!(
            bytes == self.len,
            "{} holds {} mask bytes but {} holds {} tokens",
            path.display(),
            bytes,
            self.path.display(),
            self.len
        );
        let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        anyhow::ensure!(
            data.iter().all(|value| *value <= 1),
            "{} contains a value other than 0 or 1",
            path.display()
        );
        self.masks = Some(match self.source {
            Source::Memory(_) => MaskSource::Memory(data),
            Source::Streaming { .. } => MaskSource::Streaming {
                file: File::open(&path).with_context(|| format!("open {}", path.display()))?,
                scratch: Vec::new(),
            },
        });
        Ok(())
    }

    pub fn has_mask(&self) -> bool {
        self.masks.is_some()
    }

    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }

    /// Open the `.grades` sidecar — one supervision point per token — in the
    /// same mode as the tokens.
    ///
    /// # Errors
    ///
    /// If the sidecar is missing, holds a different number of grades than the
    /// corpus has tokens, or holds a byte that is not one of the five grades.
    pub fn open_grades(&mut self) -> Result<()> {
        let path = crate::grade::grades_path(&self.path);
        anyhow::ensure!(
            path.exists(),
            "{} has no grade sidecar {}; run `dblocks lm grade --corpus {}` first",
            self.path.display(),
            path.display(),
            self.path.display()
        );
        let bytes = std::fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len() as usize;
        anyhow::ensure!(
            bytes / crate::grade::GRADE_BYTES == self.len,
            "{} holds {} grades but {} holds {} tokens: the sidecar was made from a different corpus",
            path.display(),
            bytes / crate::grade::GRADE_BYTES,
            self.path.display(),
            self.len
        );
        // Decode once here, even in streaming mode: a corrupt byte found at
        // train time, ten thousand steps in, is a corrupt byte found too late.
        let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let plane = crate::grade::GradePlane::from_bytes(&data)
            .with_context(|| format!("decoding {}", path.display()))?;
        self.grades = Some(match self.source {
            Source::Memory(_) => GradeSource::Memory(plane.as_i8().to_vec()),
            Source::Streaming { .. } => GradeSource::Streaming {
                file: File::open(&path).with_context(|| format!("open {}", path.display()))?,
                scratch: Vec::new(),
            },
        });
        Ok(())
    }

    pub fn has_grades(&self) -> bool {
        self.grades.is_some()
    }

    /// Grade every token of the corpus at `corpus` from its own label sidecar:
    /// a token a heavy rule fired on earns `−1`, a lightly-flagged one `−0.5`,
    /// and an unflagged token stays *ungraded* rather than correct — the rules
    /// say what is wrong, never what is right.
    ///
    /// Returns the manifest that was written.
    pub fn grade_file_from_labels(corpus: &Path) -> Result<crate::grade::GradeManifest> {
        let mut resident = Self::in_memory(corpus)?;
        resident.open_labels()?;
        let len = resident.len();
        let ids = resident.window_labels(0, len)?;
        let table = resident.manifest()?.weight_table();
        let plane = crate::grade::GradePlane::from_label_ids(&ids, &table);
        let manifest = crate::grade::GradeManifest::for_plane(
            &plane,
            "labels(rule weight >= 0.5 -> -1, 0 < weight < 0.5 -> -0.5, else ungraded)",
            len,
        );
        plane.write(&crate::grade::grades_path(corpus))?;
        manifest.write(&crate::grade::manifest_path(corpus))?;
        Ok(manifest)
    }

    /// The grade manifest written next to the sidecar, if grading has run.
    pub fn grade_manifest(&self) -> Result<crate::grade::GradeManifest> {
        crate::grade::GradeManifest::read(&crate::grade::manifest_path(&self.path))
    }

    /// The manifest written next to the labels, if labeling has been run.
    pub fn manifest(&self) -> Result<LabelManifest> {
        LabelManifest::read(&manifest_path(&self.path))
    }

    /// Read the whole corpus into memory.
    pub fn in_memory(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let len = Self::token_count(bytes.len(), path)?;
        let tokens = bytes
            .chunks_exact(TOKEN_BYTES)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Self {
            source: Source::Memory(tokens),
            labels: None,
            masks: None,
            grades: None,
            path: path.to_path_buf(),
            len,
        })
    }

    /// Read windows from disk on demand.
    pub fn streaming(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len = Self::token_count(file.metadata()?.len() as usize, path)?;
        Ok(Self {
            source: Source::Streaming {
                file,
                scratch: Vec::new(),
                reads_issued: 0,
            },
            labels: None,
            masks: None,
            grades: None,
            path: path.to_path_buf(),
            len,
        })
    }

    fn token_count(bytes: usize, path: &Path) -> Result<usize> {
        anyhow::ensure!(
            bytes % TOKEN_BYTES == 0,
            "{} is {bytes} bytes, not a whole number of {TOKEN_BYTES}-byte tokens",
            path.display()
        );
        Ok(bytes / TOKEN_BYTES)
    }

    /// Tokens in the corpus.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether windows are read from disk rather than memory.
    pub fn is_streaming(&self) -> bool {
        matches!(self.source, Source::Streaming { .. })
    }

    /// Reads issued so far; always 0 for an in-memory corpus.
    ///
    /// The metric roadmap items 13.3 and 13.6 exist to reduce, exposed here for
    /// the same reason [`crate::rawdata`] exposes it: an I/O optimization that
    /// cannot be measured cannot be claimed.
    pub fn reads_issued(&self) -> usize {
        match &self.source {
            Source::Memory(_) => 0,
            Source::Streaming { reads_issued, .. } => *reads_issued,
        }
    }

    /// Distinct training windows of length `context + 1`.
    ///
    /// One more token than the context is read per window because next-token
    /// training needs a target for the final input position. A corpus with
    /// fewer tokens than that yields no windows at all rather than a short one:
    /// a truncated window would train the model on a sequence length it will
    /// never see at inference.
    pub fn windows(&self, context: usize) -> usize {
        let span = context + 1;
        self.len
            .saturating_sub(span)
            .saturating_add(usize::from(self.len >= span))
    }

    /// Read `count` tokens starting at token index `start`.
    ///
    /// # Errors
    ///
    /// If the range runs past the end of the corpus. Clamping would silently
    /// produce a short sequence, which downstream code would pad and then
    /// count as real tokens.
    pub fn window(&mut self, start: usize, count: usize) -> Result<Vec<u16>> {
        anyhow::ensure!(
            start + count <= self.len,
            "window [{start}, {}) runs past the {} tokens in {}",
            start + count,
            self.len,
            self.path.display()
        );

        match &mut self.source {
            Source::Memory(tokens) => Ok(tokens[start..start + count].to_vec()),
            Source::Streaming {
                file,
                scratch,
                reads_issued,
            } => {
                // One seek and one read per window: the fixed stride is what
                // buys that, and it is why the format has no header.
                scratch.resize(count * TOKEN_BYTES, 0);
                file.seek(SeekFrom::Start((start * TOKEN_BYTES) as u64))
                    .with_context(|| format!("seek {}", self.path.display()))?;
                file.read_exact(scratch)
                    .with_context(|| format!("read {}", self.path.display()))?;
                *reads_issued += 1;
                Ok(scratch
                    .chunks_exact(TOKEN_BYTES)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect())
            }
        }
    }

    /// The labels for the same window [`Self::window`] returns.
    ///
    /// Requires [`Self::open_labels`]; a corpus without labels is an error
    /// rather than a window of [`crate::antipattern::CLEAN`], because a run that asked for a
    /// penalty and got none would report success while training on the very
    /// patterns it was told to unlearn.
    pub fn window_labels(&mut self, start: usize, count: usize) -> Result<Vec<u8>> {
        anyhow::ensure!(
            start + count <= self.len,
            "window [{start}, {}) runs past the {} tokens in {}",
            start + count,
            self.len,
            self.path.display()
        );
        let labels = self.labels.as_mut().with_context(|| {
            format!(
                "{} has no labels open; call open_labels first",
                self.path.display()
            )
        })?;
        match labels {
            LabelSource::Memory(all) => Ok(all[start..start + count].to_vec()),
            LabelSource::Streaming { file, scratch } => {
                scratch.resize(count * LABEL_BYTES, 0);
                file.seek(SeekFrom::Start((start * LABEL_BYTES) as u64))
                    .with_context(|| format!("seek {}", labels_path(&self.path).display()))?;
                file.read_exact(scratch)
                    .with_context(|| format!("read {}", labels_path(&self.path).display()))?;
                if let Source::Streaming { reads_issued, .. } = &mut self.source {
                    *reads_issued += 1;
                }
                Ok(scratch.clone())
            }
        }
    }

    pub fn window_mask(&mut self, start: usize, count: usize) -> Result<Vec<u8>> {
        anyhow::ensure!(
            start + count <= self.len,
            "window [{start}, {}) runs past the {} tokens in {}",
            start + count,
            self.len,
            self.path.display()
        );
        let masks = self.masks.as_mut().with_context(|| {
            format!(
                "{} has no mask open; call open_mask first",
                self.path.display()
            )
        })?;
        match masks {
            MaskSource::Memory(all) => Ok(all[start..start + count].to_vec()),
            MaskSource::Streaming { file, scratch } => {
                scratch.resize(count, 0);
                file.seek(SeekFrom::Start(start as u64))
                    .with_context(|| format!("seek {}", mask_path(&self.path).display()))?;
                file.read_exact(scratch)
                    .with_context(|| format!("read {}", mask_path(&self.path).display()))?;
                if let Source::Streaming { reads_issued, .. } = &mut self.source {
                    *reads_issued += 1;
                }
                Ok(scratch.clone())
            }
        }
    }

    /// A window of grades, one per token, decoded and validated.
    ///
    /// # Errors
    ///
    /// If no sidecar is open, if the read fails or runs past the end, or if a
    /// byte is not one of the five quantized grades.
    pub fn window_grades(&mut self, start: usize, count: usize) -> Result<Vec<i8>> {
        let path = crate::grade::grades_path(&self.path);
        match &mut self.grades {
            None => Err(anyhow::anyhow!(
                "grades requested before open_grades on {}",
                self.path.display()
            )),
            Some(GradeSource::Memory(values)) => {
                anyhow::ensure!(
                    start.saturating_add(count) <= values.len(),
                    "grade window {start}+{count} runs past the {} grades of {}",
                    values.len(),
                    path.display()
                );
                Ok(values[start..start + count].to_vec())
            }
            Some(GradeSource::Streaming { file, scratch }) => {
                scratch.resize(count * crate::grade::GRADE_BYTES, 0);
                file.seek(SeekFrom::Start((start * crate::grade::GRADE_BYTES) as u64))
                    .with_context(|| format!("seek {}", path.display()))?;
                file.read_exact(scratch)
                    .with_context(|| format!("read {}", path.display()))?;
                if let Source::Streaming { reads_issued, .. } = &mut self.source {
                    *reads_issued += 1;
                }
                let plane = crate::grade::GradePlane::from_bytes(scratch)
                    .with_context(|| format!("decoding {}", path.display()))?;
                Ok(plane.as_i8().to_vec())
            }
        }
    }

    /// A batch of `batch_size` windows of length `context + 1`, sampled
    /// uniformly with replacement.
    ///
    /// Sampling with replacement rather than shuffling an epoch keeps the
    /// reader stateless and lets a corpus larger than memory be trained on
    /// without an index — the same choice [`crate::rawdata::RawImageDataset`]
    /// makes.
    pub fn sample_batch<R: rand::Rng>(
        &mut self,
        batch_size: usize,
        context: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<u16>>> {
        let span = context + 1;
        let starts = self.draw_starts(batch_size, span, rng)?;
        starts
            .into_iter()
            .map(|start| self.window(start, span))
            .collect()
    }

    /// Token windows, optionally paired with their per-token label bytes, mask
    /// bytes, and grade bytes. The side tables are `None` unless the matching
    /// `want_*` flag was set *and* the corpus was opened with that side
    /// (`open_labels`, `open_mask`, `open_grades`), so a caller that asks for a
    /// side it does not have gets an error rather than silent default data.
    pub fn sample_batch_with_sides<R: rand::Rng>(
        &mut self,
        batch_size: usize,
        context: usize,
        want_labels: bool,
        want_masks: bool,
        want_grades: bool,
        rng: &mut R,
    ) -> Result<BatchWithSides> {
        let span = context + 1;
        let has_labels = self.has_labels();
        let has_masks = self.has_mask();
        let has_grades = self.has_grades();
        let starts = if want_masks {
            self.draw_masked_starts(batch_size, span, rng)?
        } else {
            self.draw_starts(batch_size, span, rng)?
        };
        anyhow::ensure!(
            !want_labels || has_labels,
            "labels requested before open_labels"
        );
        anyhow::ensure!(!want_masks || has_masks, "masks requested before open_mask");
        anyhow::ensure!(
            !want_grades || has_grades,
            "grades requested before open_grades"
        );
        let mut tokens = Vec::with_capacity(batch_size);
        let mut labels = want_labels.then(|| Vec::with_capacity(batch_size));
        let mut masks = want_masks.then(|| Vec::with_capacity(batch_size));
        let mut grades = want_grades.then(|| Vec::with_capacity(batch_size));
        for start in starts {
            tokens.push(self.window(start, span)?);
            if let Some(rows) = labels.as_mut() {
                rows.push(self.window_labels(start, span)?);
            }
            if let Some(rows) = masks.as_mut() {
                rows.push(self.window_mask(start, span)?);
            }
            if let Some(rows) = grades.as_mut() {
                rows.push(self.window_grades(start, span)?);
            }
        }
        Ok((tokens, labels, masks, grades))
    }

    pub fn sample_batch_masked<R: rand::Rng>(
        &mut self,
        batch_size: usize,
        context: usize,
        rng: &mut R,
    ) -> Result<MaskedBatch> {
        let (tokens, _, masks, _) =
            self.sample_batch_with_sides(batch_size, context, false, true, false, rng)?;
        Ok((tokens, masks.unwrap_or_default()))
    }

    /// [`Self::sample_batch`] plus the labels of every window, drawn from the
    /// same random stream so a labeled and an unlabeled run with one seed
    /// visit the same windows.
    pub fn sample_batch_labeled<R: rand::Rng>(
        &mut self,
        batch_size: usize,
        context: usize,
        rng: &mut R,
    ) -> Result<LabeledBatch> {
        let span = context + 1;
        let starts = self.draw_starts(batch_size, span, rng)?;
        let mut tokens = Vec::with_capacity(batch_size);
        let mut labels = Vec::with_capacity(batch_size);
        for start in starts {
            tokens.push(self.window(start, span)?);
            labels.push(self.window_labels(start, span)?);
        }
        Ok((tokens, labels))
    }

    fn draw_masked_starts<R: rand::Rng>(
        &mut self,
        batch_size: usize,
        span: usize,
        rng: &mut R,
    ) -> Result<Vec<usize>> {
        anyhow::ensure!(self.has_mask(), "masked sampling requires an open mask");
        let mut starts = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut selected = None;
            for _ in 0..128 {
                let candidate = self.draw_starts(1, span, rng)?[0];
                if self
                    .window_mask(candidate, span)?
                    .iter()
                    .any(|value| *value != 0)
                {
                    selected = Some(candidate);
                    break;
                }
            }
            let candidate = selected.ok_or_else(|| {
                anyhow::anyhow!("no supervised window found in {}", self.path.display())
            })?;
            starts.push(candidate);
        }
        Ok(starts)
    }

    fn draw_starts<R: rand::Rng>(
        &self,
        batch_size: usize,
        span: usize,
        rng: &mut R,
    ) -> Result<Vec<usize>> {
        anyhow::ensure!(
            self.len >= span,
            "corpus has {} tokens, fewer than the {span} one window needs",
            self.len
        );
        let last_start = self.len - span;
        Ok((0..batch_size)
            .map(|_| {
                if last_start == 0 {
                    0
                } else {
                    rng.random_range(0..=last_start)
                }
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Cybercorpus normalizer (PLAN-NINA §2)
//
// The universal dataset contract for every field, not just cyber. Raw trees
// (`raw/<source>/…`) become: `dataset/all.jsonl` (one document per line),
// `dataset/<source>.jsonl` per source, `dataset/manifest.json` (counts and
// every skip/drop with its reason), and `text/<source>/…` plain-text files —
// the latter being what [`TokenCorpus::tokenize_file`] consumes to build a
// trainable corpus. The normalizer never tokenizes; it produces the layout
// the tokenizer reads.
//
// Every document's *content* passes through [`crate::cheat::scan_snippet`]:
// prose (`.txt`/`.md`) as [`crate::cheat::SnippetLanguage::Other`], JSONL rows
// declaring a `language` with that language's scanner. For paired rows
// (prompt+completion) only the completion half is scanned — the prompt half
// is instructions (ours, in the export case), and scanning instructions with
// a code scanner would flag the requirement text that names the forbidden
// patterns as if it violated them. This is where the secrets scan runs: a
// credential in a raw scrape is dropped, counted per class in the manifest,
// and never silently included. The same is true of the brokenness checks,
// which is deliberate: prose that starts a line with "here is" or "you can"
// reads like a pasted chat answer, and the manifest makes the drop visible
// rather than the document trusted.
// ---------------------------------------------------------------------------

/// Default ceiling on one document's size. Larger documents are skipped and
/// listed in the manifest: a 900 MB dump is a packaging mistake, not a
/// document, and reading it as one would silently dominate the corpus.
pub const DEFAULT_MAX_DOC_BYTES: usize = 4 * 1024 * 1024;

fn default_max_doc_bytes() -> usize {
    DEFAULT_MAX_DOC_BYTES
}

/// Configuration for [`build_cybercorpus`]. All fields carry serde defaults,
/// so a sidecar written by an older build still parses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CybercorpusConfig {
    /// Documents larger than this many bytes are skipped (listed in the
    /// manifest's `skipped_oversize`).
    #[serde(default = "default_max_doc_bytes")]
    pub max_doc_bytes: usize,
}

impl Default for CybercorpusConfig {
    fn default() -> Self {
        Self {
            max_doc_bytes: default_max_doc_bytes(),
        }
    }
}

/// One line of `dataset/all.jsonl` / `dataset/<source>.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetRecord {
    /// sha256 of `source`, the relative path and the text, first 16 hex
    /// chars. Content-addressed, so a rebuild over the same raw tree yields
    /// the same ids and a changed document is a different id.
    pub id: String,
    /// The first path component under `raw/` (`recon`, `methodology`, …);
    /// files directly at the raw root land in `misc`.
    pub source: String,
    /// The document's path relative to `raw/`; JSONL rows carry `#<line>`.
    pub path: String,
    /// `txt`, `md` or `jsonl`.
    pub format: String,
    /// The document text.
    pub text: String,
}

/// Per-source counts in [`CorpusManifest`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceManifest {
    /// Kept documents.
    pub documents: usize,
    /// Kept characters.
    pub chars: usize,
    /// Cheat-class name -> documents dropped because of it.
    #[serde(default)]
    pub dropped_by_class: BTreeMap<String, usize>,
    /// Files whose extension the normalizer does not know.
    #[serde(default)]
    pub skipped_unknown_format: Vec<String>,
    /// Documents over [`CybercorpusConfig::max_doc_bytes`].
    #[serde(default)]
    pub skipped_oversize: Vec<String>,
    /// JSONL rows that parsed as JSON but carried no text field.
    #[serde(default)]
    pub jsonl_rows_without_text: usize,
}

/// `dataset/manifest.json`: the provenance record of one normalizer run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// The raw tree that was read.
    pub raw_dir: PathBuf,
    /// The layout root that was written.
    pub out_dir: PathBuf,
    /// Per-source counts, keyed by source name (sorted).
    #[serde(default)]
    pub sources: BTreeMap<String, SourceManifest>,
    /// Kept documents across all sources.
    pub total_documents: usize,
    /// Kept characters across all sources.
    pub total_chars: usize,
    /// Cheat-class name -> documents dropped because of it, all sources.
    #[serde(default)]
    pub dropped_by_class: BTreeMap<String, usize>,
    /// Unknown-format files, all sources (relative paths, sorted).
    #[serde(default)]
    pub skipped_unknown_format: Vec<String>,
    /// Oversize documents, all sources (relative paths, sorted).
    #[serde(default)]
    pub skipped_oversize: Vec<String>,
    /// JSONL rows without extractable text, all sources.
    #[serde(default)]
    pub jsonl_rows_without_text: usize,
}

/// One document pulled out of the raw tree, before filtering.
struct RawDocument {
    /// First path component under `raw/`.
    source: String,
    /// Path relative to `raw/`, with `#<line>` for JSONL rows.
    rel_path: String,
    /// `txt`, `md` or `jsonl`.
    format: String,
    /// What lands in the dataset.
    text: String,
    /// What the cheat scanner sees (see [`RowParts`]).
    scan_text: String,
    /// Declared language of a JSONL row, if it carried one.
    language: Option<String>,
}

/// All files under `dir`, recursively, sorted for a deterministic walk.
fn walk_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&current)
            .with_context(|| format!("read directory {}", current.display()))?
            .map(|entry| {
                entry
                    .map(|e| e.path())
                    .with_context(|| format!("read an entry of {}", current.display()))
            })
            .collect::<Result<Vec<_>>>()?;
        entries.sort();
        for path in entries {
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The pieces of one JSONL row, by a fixed field order: `text`, then
/// `content`; then a `prompt`+`completion` or `instruction`+`response` pair
/// (the shape teacher-trace exports carry); then a lone `completion`,
/// `response` or `answer`. A row matching none of these is counted in the
/// manifest, never guessed at.
///
/// `scan_text` is the part the cheat scanner sees, and the distinction is
/// the contract: for a paired row the `prompt`/`instruction` half is
/// *instructions* — ours, in the export case, carrying the requirement text
/// that names the forbidden patterns — while the `completion`/`response`
/// half is the untrusted content. Scanning the instruction half with a code
/// scanner would flag the requirements that forbid `todo!()` as if they were
/// one. `language` is the row's declared language, when it carries one.
struct RowParts {
    /// What goes into the dataset (`text` field): the full row text.
    text: String,
    /// What [`crate::cheat::scan_snippet`] checks.
    scan_text: String,
    /// The row's `language` field, if any.
    language: Option<String>,
}

fn jsonl_row_text(value: &serde_json::Value) -> Option<RowParts> {
    let get = |key: &str| value.get(key).and_then(|v| v.as_str());
    let language = get("language").map(str::to_string);
    if let Some(text) = get("text").or_else(|| get("content")) {
        return Some(RowParts {
            text: text.to_string(),
            scan_text: text.to_string(),
            language,
        });
    }
    for (a, b) in [("prompt", "completion"), ("instruction", "response")] {
        if let (Some(first), Some(second)) = (get(a), get(b)) {
            return Some(RowParts {
                text: format!("{first}\n\n{second}"),
                scan_text: second.to_string(),
                language,
            });
        }
    }
    get("completion")
        .or_else(|| get("response"))
        .or_else(|| get("answer"))
        .map(|lone| RowParts {
            text: lone.to_string(),
            scan_text: lone.to_string(),
            language,
        })
}

/// The scanner language for a JSONL row's declared `language` field. The
/// student registry and the scanner name two families differently (`web` vs
/// `htmlcss`, `jsts` is known to both), so the row's name goes through
/// [`crate::student::CodeLanguage::parse`] first and the scanner's own parser
/// second; anything unknown scans as `Other`.
fn row_snippet_language(language: Option<&str>) -> crate::cheat::SnippetLanguage {
    match language {
        Some(name) => match crate::student::CodeLanguage::parse(name) {
            Ok(code) => crate::langdata::snippet_language(code),
            Err(_) => crate::cheat::SnippetLanguage::from_name(name),
        },
        None => crate::cheat::SnippetLanguage::Other,
    }
}

/// A document's stable id: 16 hex chars of sha256 over source, path and text.
fn document_id(source: &str, rel_path: &str, text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(b"\n");
    hasher.update(rel_path.as_bytes());
    hasher.update(b"\n");
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The plain-text file name for a document: its path under the source with
/// `/` flattened to `__`, plus `.txt`. JSONL rows add the row number so each
/// row is its own file.
fn text_file_name(rel_under_source: &str, row: Option<usize>) -> String {
    let flattened = rel_under_source.replace('/', "__");
    match row {
        Some(n) => format!("{flattened}.row{n:04}.txt"),
        None => format!("{flattened}.txt"),
    }
}

/// Normalize a raw tree into the cybercorpus layout at `out_dir`.
///
/// Reads `raw_dir` recursively: `.txt` and `.md` files are one document each,
/// `.jsonl` files are one document per row (the field-extraction order is on
/// `RowParts`), any other extension lands on the manifest's
/// `skipped_unknown_format` list.
/// Every extracted document is scanned with [`crate::cheat::scan_snippet`]
/// (as `SnippetLanguage::Other`); a document with any quarantine- or
/// reject-grade finding — a credential, an encoded blob, a pasted chat
/// artifact — is dropped and counted per cheat class in the manifest.
/// Output files are written atomically, and a rebuild over the same raw tree
/// is byte-identical: sorted walk, content-hash ids, sorted maps.
pub fn build_cybercorpus(
    raw_dir: &Path,
    out_dir: &Path,
    config: &CybercorpusConfig,
) -> Result<CorpusManifest> {
    anyhow::ensure!(
        config.max_doc_bytes > 0,
        "CybercorpusConfig.max_doc_bytes must be positive"
    );
    anyhow::ensure!(
        raw_dir.is_dir(),
        "cybercorpus raw directory {} does not exist; clone or export sources into it first",
        raw_dir.display()
    );
    let dataset_dir = out_dir.join("dataset");
    let text_dir = out_dir.join("text");
    std::fs::create_dir_all(&dataset_dir)
        .with_context(|| format!("create {}", dataset_dir.display()))?;
    std::fs::create_dir_all(&text_dir).with_context(|| format!("create {}", text_dir.display()))?;

    let mut manifest = CorpusManifest {
        raw_dir: raw_dir.to_path_buf(),
        out_dir: out_dir.to_path_buf(),
        ..CorpusManifest::default()
    };
    let mut documents: Vec<RawDocument> = Vec::new();
    for file in walk_files(raw_dir)? {
        let rel = file
            .strip_prefix(raw_dir)
            .with_context(|| format!("{} is not under {}", file.display(), raw_dir.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        let extension = file
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&file)
            .with_context(|| format!("read raw document {}", file.display()))?;
        match extension.as_str() {
            "txt" | "md" => {
                documents.push(RawDocument {
                    source: String::new(),
                    rel_path: rel.clone(),
                    format: extension,
                    scan_text: text.clone(),
                    text,
                    language: None,
                });
            }
            "jsonl" => {
                let mut without_text = 0usize;
                for (n, line) in text.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: serde_json::Value =
                        serde_json::from_str(line).with_context(|| {
                            format!("{} line {} is not valid JSON", file.display(), n + 1)
                        })?;
                    match jsonl_row_text(&value) {
                        Some(parts) => documents.push(RawDocument {
                            source: String::new(),
                            rel_path: format!("{rel}#{}", n + 1),
                            format: "jsonl".to_string(),
                            text: parts.text,
                            scan_text: parts.scan_text,
                            language: parts.language,
                        }),
                        None => without_text += 1,
                    }
                }
                if without_text > 0 {
                    let src = rel.split('/').next().unwrap_or("misc").to_string();
                    manifest
                        .sources
                        .entry(src)
                        .or_default()
                        .jsonl_rows_without_text += without_text;
                    manifest.jsonl_rows_without_text += without_text;
                }
            }
            _ => {
                let src = rel.split('/').next().unwrap_or("misc").to_string();
                manifest
                    .sources
                    .entry(src)
                    .or_default()
                    .skipped_unknown_format
                    .push(rel.clone());
                manifest.skipped_unknown_format.push(rel);
            }
        }
    }

    // Fill in each document's source (first path component, or `misc` for
    // files at the raw root) now that extraction is done.
    for document in &mut documents {
        let bare = document
            .rel_path
            .split('#')
            .next()
            .unwrap_or(&document.rel_path);
        document.source = if bare.contains('/') {
            bare.split('/').next().unwrap_or("misc").to_string()
        } else {
            "misc".to_string()
        };
    }

    let mut all_lines = String::new();
    let mut per_source_lines: BTreeMap<String, String> = BTreeMap::new();
    for document in &documents {
        let source_manifest = manifest.sources.entry(document.source.clone()).or_default();
        if document.text.len() > config.max_doc_bytes {
            source_manifest
                .skipped_oversize
                .push(document.rel_path.clone());
            manifest.skipped_oversize.push(document.rel_path.clone());
            continue;
        }
        let snippet_lang = row_snippet_language(document.language.as_deref());
        let report = crate::cheat::scan_snippet(&document.scan_text, snippet_lang);
        let drop_classes: Vec<String> = {
            let mut classes: Vec<String> = report
                .findings
                .iter()
                .filter(|f| f.severity >= crate::cheat::Severity::Evasion)
                .map(|f| f.class.as_str().to_string())
                .collect();
            classes.sort();
            classes.dedup();
            classes
        };
        if !drop_classes.is_empty() {
            for class in &drop_classes {
                *source_manifest
                    .dropped_by_class
                    .entry(class.clone())
                    .or_insert(0) += 1;
                *manifest.dropped_by_class.entry(class.clone()).or_insert(0) += 1;
            }
            eprintln!(
                "[corpus] dropped {} ({:?}): {}",
                document.rel_path,
                drop_classes,
                report
                    .findings
                    .iter()
                    .find(|f| f.severity >= crate::cheat::Severity::Evasion)
                    .map(|f| f.because.clone())
                    .unwrap_or_default()
            );
            continue;
        }
        let id = document_id(&document.source, &document.rel_path, &document.text);
        let record = DatasetRecord {
            id,
            source: document.source.clone(),
            path: document.rel_path.clone(),
            format: document.format.clone(),
            text: document.text.clone(),
        };
        let line = serde_json::to_string(&record).context("serialize dataset record")?;
        all_lines.push_str(&line);
        all_lines.push('\n');
        let source_lines = per_source_lines.entry(document.source.clone()).or_default();
        source_lines.push_str(&line);
        source_lines.push('\n');
        let bare = document
            .rel_path
            .split('#')
            .next()
            .unwrap_or(&document.rel_path);
        let under_source = bare
            .strip_prefix(&format!("{}/", document.source))
            .unwrap_or(bare);
        let row = document
            .rel_path
            .split('#')
            .nth(1)
            .and_then(|n| n.parse::<usize>().ok());
        let text_path = text_dir
            .join(&document.source)
            .join(text_file_name(under_source, row));
        if let Some(parent) = text_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        crate::nim::write_atomic(&text_path, document.text.as_bytes())?;
        source_manifest.documents += 1;
        source_manifest.chars += document.text.chars().count();
        manifest.total_documents += 1;
        manifest.total_chars += document.text.chars().count();
    }

    manifest.skipped_unknown_format.sort();
    manifest.skipped_oversize.sort();
    crate::nim::write_atomic(&dataset_dir.join("all.jsonl"), all_lines.as_bytes())?;
    for (source, lines) in &per_source_lines {
        crate::nim::write_atomic(
            &dataset_dir.join(format!("{source}.jsonl")),
            lines.as_bytes(),
        )?;
    }
    let manifest_json =
        serde_json::to_string_pretty(&manifest).context("serialize corpus manifest")?;
    crate::nim::write_atomic(&dataset_dir.join("manifest.json"), manifest_json.as_bytes())?;
    eprintln!(
        "[corpus] {}: {} documents ({} chars) from {} sources, {} dropped, {} unknown-format, -> {}",
        raw_dir.display(),
        manifest.total_documents,
        manifest.total_chars,
        manifest.sources.len(),
        manifest.dropped_by_class.values().sum::<usize>(),
        manifest.skipped_unknown_format.len(),
        dataset_dir.display()
    );
    Ok(manifest)
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
    use crate::antipattern::CLEAN;
    use crate::tokenizer::{Special, VOCAB_SIZE};
    use rand::{rngs::StdRng, SeedableRng};

    fn scratch_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("dblocks-corpus-tests");
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn corpus_path(name: &str) -> PathBuf {
        let path = scratch_dir().join(format!("{name}.bin"));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn test_a_corpus_round_trip_is_lossless() {
        // A tokenizer that cannot mangle its input is worth little if the
        // corpus format can. Every token in the vocabulary is exercised,
        // including the specials that live above the byte range and are the
        // reason the format is u16 rather than u8.
        let path = corpus_path("roundtrip");
        let tokens: Vec<u16> = (0..VOCAB_SIZE as u16).collect();
        TokenCorpus::write(&path, &tokens).unwrap();

        let mut corpus = TokenCorpus::in_memory(&path).unwrap();
        assert_eq!(corpus.len(), tokens.len());
        assert_eq!(corpus.window(0, tokens.len()).unwrap(), tokens);
    }

    #[test]
    fn test_streaming_and_memory_agree_exactly() {
        // The two readers exist so a corpus larger than memory can be trained
        // on. They are only interchangeable if they return the same tokens, so
        // that is checked window by window rather than assumed.
        let path = corpus_path("agreement");
        let tokens: Vec<u16> = (0..500u16).map(|i| i % VOCAB_SIZE as u16).collect();
        TokenCorpus::write(&path, &tokens).unwrap();

        let mut memory = TokenCorpus::in_memory(&path).unwrap();
        let mut streamed = TokenCorpus::streaming(&path).unwrap();
        assert!(streamed.is_streaming() && !memory.is_streaming());
        assert_eq!(memory.len(), streamed.len());

        for start in [0usize, 1, 17, 250, 500 - 9] {
            assert_eq!(
                memory.window(start, 9).unwrap(),
                streamed.window(start, 9).unwrap(),
                "window at {start} disagreed"
            );
        }
        assert_eq!(memory.reads_issued(), 0);
        assert_eq!(streamed.reads_issued(), 5, "one seek-and-read per window");
    }

    #[test]
    fn test_tokenizing_a_file_delimits_the_document() {
        let source = scratch_dir().join("source.txt");
        std::fs::write(&source, "hello corpus").unwrap();
        let path = corpus_path("tokenized");

        let count = TokenCorpus::tokenize_file(&source, &path).unwrap();
        assert_eq!(count, "hello corpus".len() + 2);

        let mut corpus = TokenCorpus::in_memory(&path).unwrap();
        let all = corpus.window(0, count).unwrap();
        assert_eq!(all.first(), Some(&Special::Bos.id()));
        assert_eq!(all.last(), Some(&Special::Eos.id()));
        assert_eq!(
            ByteTokenizer::new().decode(&all).as_deref(),
            Some("hello corpus")
        );
    }

    #[test]
    fn test_a_short_window_is_refused_not_padded() {
        // Clamping would hand back a short sequence that downstream code pads
        // and then counts as real tokens -- a silent corruption of the loss
        // denominator. Refusing is the honest failure.
        let path = corpus_path("short");
        TokenCorpus::write(&path, &[1, 2, 3, 4]).unwrap();
        let mut corpus = TokenCorpus::in_memory(&path).unwrap();

        assert!(corpus.window(0, 4).is_ok());
        let err = corpus.window(2, 4).unwrap_err().to_string();
        assert!(err.contains("runs past"), "unhelpful error: {err}");

        // ...and the same for a whole batch.
        let mut rng = StdRng::seed_from_u64(0);
        assert!(corpus.sample_batch(2, 8, &mut rng).is_err());
    }

    #[test]
    fn test_window_count_needs_a_target_for_the_last_position() {
        let path = corpus_path("counts");
        TokenCorpus::write(&path, &(0..10u16).collect::<Vec<_>>()).unwrap();
        let corpus = TokenCorpus::in_memory(&path).unwrap();

        // 10 tokens, context 4 -> windows of 5 -> starts 0..=5.
        assert_eq!(corpus.windows(4), 6);
        // Exactly one window fits.
        assert_eq!(corpus.windows(9), 1);
        // One token short: none, rather than a truncated one.
        assert_eq!(corpus.windows(10), 0);
    }

    #[test]
    fn test_batches_are_reproducible_and_correctly_shaped() {
        let path = corpus_path("batches");
        TokenCorpus::write(&path, &(0..200u16).collect::<Vec<_>>()).unwrap();
        let mut corpus = TokenCorpus::in_memory(&path).unwrap();

        let mut a = StdRng::seed_from_u64(11);
        let mut b = StdRng::seed_from_u64(11);
        let first = corpus.sample_batch(4, 7, &mut a).unwrap();
        let second = corpus.sample_batch(4, 7, &mut b).unwrap();

        assert_eq!(first, second, "the same seed must give the same batch");
        assert_eq!(first.len(), 4);
        for window in &first {
            assert_eq!(
                window.len(),
                8,
                "context + 1, so the last input has a target"
            );
            // Contiguity: the corpus is 0..200, so a window must be a run.
            for pair in window.windows(2) {
                assert_eq!(pair[1], pair[0] + 1, "windows must be contiguous");
            }
        }
    }

    #[test]
    fn test_a_truncated_file_is_rejected() {
        // An odd byte count means the file is not a whole number of tokens --
        // a truncated write, or the wrong file entirely. Reading it as if the
        // last token were fine would shift every token after the damage.
        let path = corpus_path("odd");
        std::fs::write(&path, [1u8, 0, 2]).unwrap();
        let err = TokenCorpus::in_memory(&path).unwrap_err().to_string();
        assert!(err.contains("not a whole number"), "unhelpful error: {err}");
        assert!(TokenCorpus::streaming(&path).is_err());
    }

    #[test]
    fn test_labels_land_on_the_tokens_they_were_computed_for() {
        // Label a corpus, then read tokens and labels back through both
        // readers and check that every labeled token decodes to the body of
        // the rule that flagged it -- across window boundaries included, since
        // labels are computed over the whole corpus before any window is cut.
        let source = scratch_dir().join("labeled.txt");
        std::fs::write(
            &source,
            "ok()\ntry:\n    f()\nexcept:\n    pass\ncatch (e) {}\n",
        )
        .unwrap();
        let path = corpus_path("labeled");
        TokenCorpus::tokenize_file(&source, &path).unwrap();

        let labeler = Labeler::builtin().expect("built-in rules");
        let manifest = TokenCorpus::label_file(&path, &labeler).unwrap();
        assert_eq!(manifest.labeled_tokens, ":pass}".len());
        assert!(labels_path(&path).exists() && manifest_path(&path).exists());

        let mut memory = TokenCorpus::in_memory(&path).unwrap();
        let mut streamed = TokenCorpus::streaming(&path).unwrap();
        assert!(!memory.has_labels());
        assert!(
            memory.window_labels(0, 4).is_err(),
            "labels must be opened explicitly"
        );
        memory.open_labels().unwrap();
        streamed.open_labels().unwrap();
        assert!(memory.has_labels() && streamed.has_labels());
        assert_eq!(memory.manifest().unwrap(), manifest);

        let len = memory.len();
        let tokens = memory.window(0, len).unwrap();
        let labels = memory.window_labels(0, len).unwrap();
        let flagged: String = tokens
            .iter()
            .zip(&labels)
            .filter(|(_, l)| **l != CLEAN)
            .map(|(t, _)| char::from(*t as u8))
            .collect();
        assert_eq!(flagged, ":pass}");

        // Windows of 7 cut through "pass": both readers must agree and the
        // labels must follow the tokens, not the window.
        for start in (0..len - 7).step_by(5) {
            let (t_mem, l_mem) = (
                memory.window(start, 7).unwrap(),
                memory.window_labels(start, 7).unwrap(),
            );
            let (t_str, l_str) = (
                streamed.window(start, 7).unwrap(),
                streamed.window_labels(start, 7).unwrap(),
            );
            assert_eq!(t_mem, t_str);
            assert_eq!(l_mem, l_str, "labels disagree at {start}");
            assert_eq!(l_mem, labels[start..start + 7].to_vec());
        }
        assert!(streamed.reads_issued() > 0);

        // The labeled batch draws the same windows as the plain one.
        let mut a = StdRng::seed_from_u64(5);
        let mut b = StdRng::seed_from_u64(5);
        let plain = memory.sample_batch(3, 6, &mut a).unwrap();
        let (labeled, batch_labels) = memory.sample_batch_labeled(3, 6, &mut b).unwrap();
        assert_eq!(plain, labeled);
        assert_eq!(batch_labels.len(), 3);
        assert!(batch_labels.iter().all(|l| l.len() == 7));
    }

    #[test]
    fn test_a_sidecar_from_another_corpus_is_refused() {
        // Right length is the only thing that keeps label i on token i. A
        // sidecar of the wrong length is a sidecar for a different file.
        let path = corpus_path("mismatch");
        TokenCorpus::write(&path, &[65, 66, 67, 68]).unwrap();
        TokenCorpus::write_labels(&labels_path(&path), &[0, 1, 0]).unwrap();
        let mut corpus = TokenCorpus::in_memory(&path).unwrap();
        let err = corpus.open_labels().unwrap_err().to_string();
        assert!(err.contains("different corpus"), "unhelpful error: {err}");

        let missing = corpus_path("unlabeled");
        TokenCorpus::write(&missing, &[65, 66]).unwrap();
        let err = TokenCorpus::in_memory(&missing)
            .unwrap()
            .open_labels()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("dblocks lm label"),
            "the error should say how to fix it: {err}"
        );
    }

    #[test]
    fn test_masks_follow_tokens_through_both_readers() {
        let path = corpus_path("masked");
        let tokens: Vec<u16> = (0..32u16).collect();
        let masks: Vec<u8> = (0..32).map(|i| u8::from(i % 3 == 0)).collect();
        TokenCorpus::write(&path, &tokens).unwrap();
        TokenCorpus::write_masks(&mask_path(&path), &masks).unwrap();
        let mut memory = TokenCorpus::in_memory(&path).unwrap();
        let mut streamed = TokenCorpus::streaming(&path).unwrap();
        assert!(!memory.has_mask());
        memory.open_mask().unwrap();
        streamed.open_mask().unwrap();
        assert!(memory.has_mask() && streamed.has_mask());
        assert_eq!(memory.window_mask(4, 9).unwrap(), masks[4..13].to_vec());
        assert_eq!(streamed.window_mask(4, 9).unwrap(), masks[4..13].to_vec());
        let mut rng = StdRng::seed_from_u64(9);
        let (batch, batch_masks) = memory.sample_batch_masked(3, 7, &mut rng).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch_masks.len(), 3);
        assert!(batch_masks
            .iter()
            .all(|row| row.len() == 8 && row.iter().any(|value| *value != 0)));
    }

    // -- cybercorpus normalizer --------------------------------------------

    /// A raw tree with three sources, one planted credential and one
    /// unknown-format file.
    fn cyber_fixture(root: &Path) -> PathBuf {
        let raw = root.join("raw");
        std::fs::create_dir_all(raw.join("alpha")).unwrap();
        std::fs::create_dir_all(raw.join("beta")).unwrap();
        std::fs::create_dir_all(raw.join("gamma")).unwrap();
        std::fs::write(
            raw.join("alpha/notes.txt"),
            "Network reconnaissance field notes.\n\nScan the perimeter hosts first.\n\
             Document every open port with its service banner.\n",
        )
        .unwrap();
        std::fs::write(
            raw.join("alpha/creds.txt"),
            "credentials\napi_key = \"nvapi-a1B2c3D4e5F6g7H8i9J0k1L2\"\n",
        )
        .unwrap();
        std::fs::write(
            raw.join("beta/guide.md"),
            "# Recon guide\n\n1. Enumerate hosts.\n2. Fingerprint services.\n",
        )
        .unwrap();
        std::fs::write(
            raw.join("beta/report.docx"),
            "not really a docx, but the extension is unknown\n",
        )
        .unwrap();
        std::fs::write(
            raw.join("gamma/rows.jsonl"),
            "{\"id\":\"g1\",\"text\":\"Kerberos pre-authentication is on by default in modern domains.\"}\n\
             {\"prompt\":\"Explain SMB signing.\",\"completion\":\"SMB signing requires packet signatures, which blocks relay attacks.\"}\n",
        )
        .unwrap();
        raw
    }

    #[test]
    fn cybercorpus_normalizer_counts_keeps_drops_and_skips() {
        let dir = corpus_path("cyber");
        let raw = cyber_fixture(&dir);
        let out = dir.join("out");
        let manifest = build_cybercorpus(&raw, &out, &CybercorpusConfig::default()).unwrap();

        assert_eq!(manifest.total_documents, 4);
        assert_eq!(manifest.sources["alpha"].documents, 1);
        assert_eq!(
            manifest.sources["alpha"]
                .dropped_by_class
                .get("data-secrets"),
            Some(&1),
            "the planted credential is dropped and counted"
        );
        assert_eq!(manifest.sources["beta"].documents, 1);
        assert_eq!(
            manifest.sources["beta"].skipped_unknown_format,
            vec!["beta/report.docx".to_string()]
        );
        assert_eq!(manifest.sources["gamma"].documents, 2);
        assert_eq!(manifest.dropped_by_class.get("data-secrets"), Some(&1));
        assert_eq!(
            manifest.skipped_unknown_format,
            vec!["beta/report.docx".to_string()]
        );
        assert_eq!(manifest.jsonl_rows_without_text, 0);
        assert!(manifest.total_chars > 0);

        // The dataset files hold exactly the kept documents; the secret text
        // appears in none of them.
        let all = std::fs::read_to_string(out.join("dataset/all.jsonl")).unwrap();
        assert_eq!(all.lines().count(), 4);
        assert!(
            !all.contains("nvapi-"),
            "the secret must not reach the dataset"
        );
        for line in all.lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            let id = record["id"].as_str().unwrap();
            assert_eq!(id.len(), 16, "16-hex id: {id}");
            assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
            for key in ["source", "path", "format", "text"] {
                assert!(record.get(key).is_some(), "record carries {key}");
            }
        }
        let gamma = std::fs::read_to_string(out.join("dataset/gamma.jsonl")).unwrap();
        assert_eq!(gamma.lines().count(), 2);
        assert!(out.join("text/alpha/notes.txt.txt").is_file());
        assert!(out.join("text/gamma/rows.jsonl.row0001.txt").is_file());
        assert!(out.join("text/gamma/rows.jsonl.row0002.txt").is_file());

        // The manifest round-trips through its own JSON.
        let stored: CorpusManifest = serde_json::from_str(
            &std::fs::read_to_string(out.join("dataset/manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(stored.total_documents, 4);
        assert_eq!(stored.dropped_by_class.get("data-secrets"), Some(&1));
    }

    #[test]
    fn cybercorpus_rebuild_is_byte_identical() {
        let dir = corpus_path("cyber-det");
        let raw = cyber_fixture(&dir);
        let out = dir.join("out");
        build_cybercorpus(&raw, &out, &CybercorpusConfig::default()).unwrap();
        let manifest_first = std::fs::read(out.join("dataset/manifest.json")).unwrap();
        let all_first = std::fs::read(out.join("dataset/all.jsonl")).unwrap();
        let manifest = build_cybercorpus(&raw, &out, &CybercorpusConfig::default()).unwrap();
        assert_eq!(manifest.total_documents, 4);
        assert_eq!(
            std::fs::read(out.join("dataset/manifest.json")).unwrap(),
            manifest_first
        );
        assert_eq!(
            std::fs::read(out.join("dataset/all.jsonl")).unwrap(),
            all_first
        );
    }

    #[test]
    fn cybercorpus_missing_raw_dir_is_a_named_error() {
        let dir = corpus_path("cyber-missing");
        let err = build_cybercorpus(
            &dir.join("absent"),
            &dir.join("out"),
            &CybercorpusConfig::default(),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("absent"));
    }
}
