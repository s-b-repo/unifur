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
}
