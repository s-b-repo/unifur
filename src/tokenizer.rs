//! Byte-level tokenization (roadmap 19.1).
//!
//! A tokenizer with **no external dependency**: the vocabulary is the 256 byte
//! values plus a handful of specials. That is a real trade, so it is worth
//! stating both sides.
//!
//! *Against:* sequences are longer than with BPE — roughly 4x for English text
//! — and attention is quadratic in length, so the same context window costs
//! more.
//!
//! *For:* it is **lossless by construction** on every input, including
//! arbitrary binary; there is no vocabulary file to ship, version, or get out
//! of sync with a checkpoint; there are no unknown tokens, no normalization
//! surprises, and no merge table to train. For a crate whose stated design
//! constraint is a small dependency set and whose datasets are already raw
//! fixed-record binaries, that is the right side of the trade.
//!
//! Specials occupy ids at the *top* of the range rather than the bottom, so a
//! raw byte's id is exactly its value. `text.as_bytes()[i] == ids[i]` for plain
//! text, which makes a corpus file readable with `xxd` and a tokenization bug
//! visible by eye.

/// Number of raw byte tokens.
pub const BYTE_TOKENS: usize = 256;

/// Special tokens, appended above the byte range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    /// Beginning of a document.
    Bos,
    /// End of a document.
    Eos,
    /// Padding; masked out of the loss.
    Pad,
}

impl Special {
    pub const ALL: [Special; 3] = [Special::Bos, Special::Eos, Special::Pad];

    /// Token id.
    pub const fn id(&self) -> u16 {
        match self {
            Special::Bos => 256,
            Special::Eos => 257,
            Special::Pad => 258,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Special::Bos => "<bos>",
            Special::Eos => "<eos>",
            Special::Pad => "<pad>",
        }
    }

    pub fn from_id(id: u16) -> Option<Self> {
        Special::ALL.into_iter().find(|s| s.id() == id)
    }
}

/// Total vocabulary size: 256 bytes plus the specials.
pub const VOCAB_SIZE: usize = BYTE_TOKENS + Special::ALL.len();

/// A byte-level tokenizer.
///
/// Stateless — it exists as a type so the vocabulary size and the special ids
/// travel together with the code that uses them, rather than as loose constants
/// a caller has to keep consistent.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    /// Encode text to token ids. Never fails and never loses information.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        text.as_bytes().iter().map(|b| *b as u16).collect()
    }

    /// Encode with document markers, which is what a training corpus wants.
    pub fn encode_document(&self, text: &str) -> Vec<u16> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        ids.push(Special::Bos.id());
        ids.extend(text.as_bytes().iter().map(|b| *b as u16));
        ids.push(Special::Eos.id());
        ids
    }

    /// Decode ids back to bytes, dropping specials.
    ///
    /// Ids outside the vocabulary are dropped rather than panicking: a decode
    /// path fed by a *sampled* distribution should degrade rather than abort.
    pub fn decode_bytes(&self, ids: &[u16]) -> Vec<u8> {
        ids.iter()
            .filter(|id| **id < BYTE_TOKENS as u16)
            .map(|id| *id as u8)
            .collect()
    }

    /// Decode to a string.
    ///
    /// Returns `None` when the bytes are not valid UTF-8 — which a partially
    /// generated sequence often is not, since a multi-byte character can be cut
    /// in half. Callers streaming output should use [`Self::decode_lossy`].
    pub fn decode(&self, ids: &[u16]) -> Option<String> {
        String::from_utf8(self.decode_bytes(ids)).ok()
    }

    /// Decode to a string, replacing invalid sequences.
    pub fn decode_lossy(&self, ids: &[u16]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Whether `id` is a special rather than a byte.
    pub fn is_special(&self, id: u16) -> bool {
        Special::from_id(id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_is_lossless_for_arbitrary_text() {
        // The property that justifies choosing byte level at all: there is no
        // input this tokenizer can mangle.
        let tok = ByteTokenizer::new();
        for text in [
            "",
            "hello world",
            r#"fn main() { println!("hi"); }"#,
            "naïve café — em-dash, ünïcödé",
            "日本語のテキスト",
            "emoji: 🦀🔥",
            "\u{0}\u{1}\u{7f} control bytes",
        ] {
            let ids = tok.encode(text);
            assert_eq!(tok.decode(&ids).as_deref(), Some(text), "lost: {text:?}");
            assert_eq!(ids.len(), text.len(), "one token per byte");
        }
    }

    #[test]
    fn test_byte_ids_are_the_bytes_themselves() {
        // Specials sit above the byte range on purpose: a corpus file stays
        // readable with `xxd`, and a tokenization bug is visible by eye.
        let tok = ByteTokenizer::new();
        let ids = tok.encode("AZaz09");
        assert_eq!(ids, vec![65, 90, 97, 122, 48, 57]);
        for b in 0u16..256 {
            assert!(!tok.is_special(b), "byte {b} must not be special");
        }
    }

    #[test]
    fn test_specials_are_distinct_and_above_the_bytes() {
        let mut ids: Vec<u16> = Special::ALL.iter().map(Special::id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "special ids must be distinct");
        assert!(ids.iter().all(|id| *id >= BYTE_TOKENS as u16));
        assert_eq!(VOCAB_SIZE, BYTE_TOKENS + count);
        assert!(ids.iter().all(|id| (*id as usize) < VOCAB_SIZE));

        for s in Special::ALL {
            assert_eq!(Special::from_id(s.id()), Some(s));
        }
        assert_eq!(Special::from_id(0), None);
    }

    #[test]
    fn test_documents_are_delimited() {
        let tok = ByteTokenizer::new();
        let ids = tok.encode_document("hi");
        assert_eq!(ids.first(), Some(&Special::Bos.id()));
        assert_eq!(ids.last(), Some(&Special::Eos.id()));
        // The markers do not survive decoding as text.
        assert_eq!(tok.decode(&ids).as_deref(), Some("hi"));
    }

    #[test]
    fn test_decode_tolerates_partial_and_invalid_output() {
        let tok = ByteTokenizer::new();

        // A multi-byte character cut in half is not valid UTF-8. That is the
        // normal state of a partially generated sequence, so strict decoding
        // must report it rather than pretend.
        let full = tok.encode("é");
        assert_eq!(full.len(), 2);
        let half = &full[..1];
        assert_eq!(tok.decode(half), None);
        assert!(!tok.decode_lossy(half).is_empty(), "lossy decoding still yields something");

        // Ids past the vocabulary are dropped, not panicked on: a decode path
        // fed by a sampled distribution should degrade rather than abort.
        assert_eq!(tok.decode_bytes(&[65, 9999, 66]), vec![65, 66]);
    }
}
