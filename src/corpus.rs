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

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::tokenizer::ByteTokenizer;

/// Bytes per token on disk.
pub const TOKEN_BYTES: usize = 2;

/// Where a corpus reads from.
#[derive(Debug)]
enum Source {
    /// The whole corpus, resident.
    Memory(Vec<u16>),
    /// A handle plus a reusable buffer; windows are read on demand.
    Streaming { file: File, scratch: Vec<u8>, reads_issued: usize },
}

/// A pre-tokenized corpus, addressed by token index.
#[derive(Debug)]
pub struct TokenCorpus {
    source: Source,
    path: PathBuf,
    /// Tokens in the file.
    len: usize,
}

impl TokenCorpus {
    /// Write `tokens` to `path` in the corpus format.
    pub fn write(path: &Path, tokens: &[u16]) -> Result<()> {
        let mut file =
            File::create(path).with_context(|| format!("create {}", path.display()))?;
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
        let text = std::fs::read_to_string(input)
            .with_context(|| format!("read {}", input.display()))?;
        let tokens = ByteTokenizer::new().encode_document(&text);
        Self::write(output, &tokens)?;
        Ok(tokens.len())
    }

    /// Read the whole corpus into memory.
    pub fn in_memory(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let len = Self::token_count(bytes.len(), path)?;
        let tokens = bytes
            .chunks_exact(TOKEN_BYTES)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Self { source: Source::Memory(tokens), path: path.to_path_buf(), len })
    }

    /// Read windows from disk on demand.
    pub fn streaming(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len = Self::token_count(file.metadata()?.len() as usize, path)?;
        Ok(Self {
            source: Source::Streaming { file, scratch: Vec::new(), reads_issued: 0 },
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
        self.len.saturating_sub(span).saturating_add(usize::from(self.len >= span))
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
            Source::Streaming { file, scratch, reads_issued } => {
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
        anyhow::ensure!(
            self.len >= span,
            "corpus has {} tokens, fewer than the {span} one window needs",
            self.len
        );
        let last_start = self.len - span;

        let mut out = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let start = if last_start == 0 {
                0
            } else {
                rng.random_range(0..=last_start)
            };
            out.push(self.window(start, span)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(ByteTokenizer::new().decode(&all).as_deref(), Some("hello corpus"));
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
            assert_eq!(window.len(), 8, "context + 1, so the last input has a target");
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
}
