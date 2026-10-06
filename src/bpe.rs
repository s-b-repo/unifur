//! BPE tokenizer compatible with HuggingFace `tokenizer.json` (Phase A of
//! `docs/Frontier-27B-Plan.md`), covering the GPT-2-family pipeline:
//! Split-regex pre-tokenization, byte-level alphabet, BPE merges, added
//! tokens, byte-level decoding, and the feasible normalizers.
//!
//! Loads `model.vocab` / `model.merges` / `added_tokens` plus the declared
//! `pre_tokenizer`, `decoder` and `normalizer`, and encodes over `u32` ids
//! (the 248K Qwen vocabulary overflows the crate's `u16` corpus ids, so this
//! module is deliberately decoupled from [`crate::corpus`] until Phase C
//! wires a `u32` corpus path).
//!
//! Honest scope limits (read before assuming HF parity):
//!
//! - The Split regex must be one of the three known GPT-2-family patterns
//!   ([`GPT2_ORIGINAL_PATTERN`], [`GPT2_GLM_PATTERN`], [`GPT2_QWEN_PATTERN`]);
//!   any other declared pattern is a loud load-time error naming it.
//!   `\p{L}` / `\p{M}` / `\p{N}` are exact ICU general categories (the
//!   tables already compile in-tree); case folding stays ASCII because every
//!   shipped `(?i:...)` group is ASCII-only.
//! - Normalizers covered: absent/null, Lowercase, Strip, Prepend,
//!   literal-String Replace, NFC/NFD/NFKC/NFKD (standard tables), and
//!   Sequences of those. Regex Replace stays a loud error: a pattern
//!   misread as a literal corrupts ids silently.
//! - Byte fallback (`<0xHH>` tokens) is supported on decode; characters with
//!   no merge path fall back to the file's `unk_token` when one exists, else
//!   error naming the character (never silent substitution).
//! - Merge entries accept `"a b"` strings and `["a", "b"]` arrays (both
//!   shipped in real files); `#version` headers are skipped, not parsed.
//! - Added-token ids may sit past the base vocabulary; the table spans both.
//! - Merge-rank order is load-bearing (late-ranked merges starve); the engine
//!   follows ranks faithfully, which the mini-pipeline test pins.
//!
//! Validated against real on-disk `tokenizer.json` files (154K vocab,
//! 321K merges, 36 added tokens): `test_real_world_file_loads_and_roundtrips`
//! loads, checks the declared pipeline, round-trips diverse text, confirms
//! merges fire and added tokens bypass. The Qwen-file parity gate lives in
//! `qwen_tokenizer_parity_against_reference`, `#[ignore]`d until
//! `QWEN_TOKENIZER_JSON` points at a downloaded file.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{ensure, Context, Result};
use icu_properties::{props::GeneralCategoryGroup, CodePointMapData};
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

/// How input text is chunked before the BPE merge loop runs on each chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pretokenizer {
    /// Split on whitespace runs, each run attached to the following word
    /// (`"hi  there"` -> `["hi", "  there"]`). Legacy default for files that
    /// declare no pre-tokenizer.
    #[default]
    Whitespace,
    /// The whole input is one chunk (deterministic, used by tests).
    Raw,
    /// GPT-2-family regex split plus the byte-level alphabet: matches what
    /// `ByteLevel` / `Sequence[Split, ByteLevel]` files declare. This is the
    /// file-derived default whenever the file declares it.
    Gpt2,
}

/// The original GPT-2 split pattern: contractions, then optionally
/// space-prefixed letters, numbers (unbounded), punctuation runs, then
/// whitespace runs (trailing-run lookahead included).
pub const GPT2_ORIGINAL_PATTERN: &str =
    "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/// The GPT-2-family variant shipped in current files (GLM and peers):
/// case-insensitive contractions, optional prefix punctuation on letters,
/// numbers capped at three digits, newline-aware punctuation and whitespace
/// runs. This is the exact string matched against a file's declared pattern.
pub const GPT2_GLM_PATTERN: &str =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// The Qwen variant (verified against a real Qwen3.8 `tokenizer.json`):
/// marks join the letter class everywhere (`[\p{L}\p{M}]`), and numbers are
/// single digits (`\p{N}` instead of `\p{N}{1,3}`). This is the exact string
/// matched against a file's declared pattern.
pub const GPT2_QWEN_PATTERN: &str =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/// Exact Unicode general-category lookups for the splitter classes.
/// `\p{L}` is `char::is_alphabetic` plus Nl (letter-numbers like Roman
/// numerals); `\p{M}` needs mark data no hand table carries honestly, so the
/// already-compiled ICU tables answer instead. Case folding stays ASCII
/// because every shipped pattern's `(?i:...)` group is ASCII-only.
fn general_category(c: char) -> icu_properties::props::GeneralCategory {
    use icu_properties::props::GeneralCategory;
    use icu_properties::CodePointMapDataBorrowed;
    static GC: OnceLock<CodePointMapDataBorrowed<'static, GeneralCategory>> = OnceLock::new();
    (*GC.get_or_init(CodePointMapData::<GeneralCategory>::new)).get(c)
}

/// Exact `\p{L}`.
fn is_letter(c: char) -> bool {
    GeneralCategoryGroup::Letter.contains(general_category(c))
}

/// Exact `\p{M}` (spacing, nonspacing and enclosing marks).
fn is_mark(c: char) -> bool {
    GeneralCategoryGroup::Mark.contains(general_category(c))
}

/// Exact `\p{N}`.
fn is_number(c: char) -> bool {
    GeneralCategoryGroup::Number.contains(general_category(c))
}

/// Byte-to-unicode table of the GPT-2 byte alphabet: printable ASCII and
/// Latin-1 ranges map to themselves, every other byte to `U+0100` and up,
/// so all 256 bytes are representable single characters and the map is a
/// bijection. Computed, not tabulated.
pub fn bytes_to_unicode_table() -> [char; 256] {
    let mut map = ['\u{fffd}'; 256];
    let mut extra = 0u32;
    for byte in 0u32..256 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
        {
            map[byte as usize] = char::from_u32(byte).unwrap_or('\u{fffd}');
        } else {
            map[byte as usize] = char::from_u32(256 + extra).unwrap_or('\u{fffd}');
            extra += 1;
        }
    }
    map
}

/// Inverse of [`bytes_to_unicode_table`]: mapped character -> byte.
pub fn unicode_to_byte(c: char, table: &[char; 256]) -> Option<u8> {
    table.iter().position(|m| *m == c).map(|i| i as u8)
}

/// Map raw text to the byte alphabet: UTF-8 bytes through
/// [`bytes_to_unicode_table`].
pub fn map_bytes_to_unicode(text: &str, table: &[char; 256]) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.as_bytes() {
        out.push(table[*byte as usize]);
    }
    out
}

/// Undo [`map_bytes_to_unicode`]: mapped characters back to bytes; any other
/// character to its UTF-8 encoding. Concatenated output decodes with
/// `String::from_utf8_lossy` at the end (a cut multi-byte sequence degrades
/// rather than aborts, matching the decoder contract).
pub fn unmap_unicode_to_bytes(text: &str, table: &[char; 256]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for c in text.chars() {
        match unicode_to_byte(c, table) {
            Some(byte) => out.push(byte),
            None => out.extend_from_slice(c.to_string().as_bytes()),
        }
    }
    out
}

/// Which GPT-2 split-pattern dialect a declared regex string matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gpt2Dialect {
    /// [`GPT2_ORIGINAL_PATTERN`]: space-prefixed words/numbers/punctuation,
    /// unbounded digit runs, case-sensitive contractions, no marks class.
    Original,
    /// [`GPT2_GLM_PATTERN`]: prefix punctuation, 1–3 digit runs,
    /// newline-aware runs, case-insensitive contractions, no marks class.
    GlmFamily,
    /// [`GPT2_QWEN_PATTERN`]: like GlmFamily but marks join the letter
    /// runs (`[\p{L}\p{M}]+`) and digits are single (`\p{N}`), while the
    /// prefix class stays `[^\r\n\p{L}\p{N}]` (a stray mark attaches
    /// forward as prefix punctuation).
    Qwen,
}

fn match_gpt2_pattern(pattern: &str) -> Option<Gpt2Dialect> {
    if pattern == GPT2_ORIGINAL_PATTERN {
        Some(Gpt2Dialect::Original)
    } else if pattern == GPT2_GLM_PATTERN {
        Some(Gpt2Dialect::GlmFamily)
    } else if pattern == GPT2_QWEN_PATTERN {
        Some(Gpt2Dialect::Qwen)
    } else {
        None
    }
}

/// How digit runs split in a dialect: the original is unbounded, GLM caps at
/// three, Qwen splits every digit singly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DigitsRule {
    Unlimited,
    UpTo3,
    Single,
}

impl Gpt2Dialect {
    fn digits(self) -> DigitsRule {
        match self {
            Self::Original => DigitsRule::Unlimited,
            Self::GlmFamily => DigitsRule::UpTo3,
            Self::Qwen => DigitsRule::Single,
        }
    }

    /// Whether marks join the letter class (Qwen's `[\p{L}\p{M}]`).
    fn marks_are_letters(self) -> bool {
        matches!(self, Self::Qwen)
    }

    /// Whether contractions fold case-insensitively (the `(?i:...)` group).
    fn contraction_folds_case(self) -> bool {
        !matches!(self, Self::Original)
    }
}

/// Longest contraction tail after `'`: `ll|ve|re` (2 chars) before
/// `s|t|m|d` (1 char). `case_insensitive` mirrors the `(?i:...)` group.
fn match_contraction(rest: &[char], case_insensitive: bool) -> Option<usize> {
    let lower: Vec<char> = rest
        .iter()
        .take(2)
        .map(|c| {
            if case_insensitive {
                c.to_ascii_lowercase()
            } else {
                *c
            }
        })
        .collect();
    if lower.len() == 2 {
        let tail: String = lower.iter().collect();
        if tail == "ll" || tail == "ve" || tail == "re" {
            return Some(2);
        }
    }
    match lower.first() {
        Some('s') | Some('t') | Some('m') | Some('d') => Some(1),
        _ => None,
    }
}

/// Hand scanner for the GPT-2 split-pattern family (no regex crate: the
/// crate keeps a small dependency set). Alternatives are tried in pattern
/// order at every position, so every byte of input is consumed exactly once;
/// reaching the end with no alternative matching is a bug, reported loudly
/// rather than looped on.
fn split_gpt2(text: &str, dialect: Gpt2Dialect) -> Result<Vec<String>> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    // Exact classes from the ICU tables (see above); `\s` is
    // `char::is_whitespace`, which is exactly White_Space.
    let is_space = |c: char| c.is_whitespace();
    let is_newline = |c: char| c == '\r' || c == '\n';
    // The letter class, plus marks in dialects whose pattern says
    // `[\p{L}\p{M}]`. Everything below reads through this closure so the
    // prefix class, the runs and the punctuation class stay consistent.
    let is_ll = |c: char| is_letter(c) || (dialect.marks_are_letters() && is_mark(c));
    let max_digits: Option<usize> = match dialect.digits() {
        DigitsRule::Unlimited => None,
        DigitsRule::UpTo3 => Some(3),
        DigitsRule::Single => Some(1),
    };
    // Only the original dialect lacks the newline-aware runs.
    let newline_aware = dialect != Gpt2Dialect::Original;
    while i < chars.len() {
        let start = i;
        // 1. Contractions: case-insensitive wherever the pattern has `(?i:)`.
        if chars[i] == '\'' {
            if let Some(len) = match_contraction(&chars[i + 1..], dialect.contraction_folds_case())
            {
                i += 1 + len;
                out.push(chars[start..i].iter().collect());
                continue;
            }
        }
        // 2. Letters, with optional prefix punctuation outside the original
        // dialect (`[^\r\n...]?`) or optional space in the original. A lone
        // prefix without following letters fails the alternative and falls
        // through (the space/punct is picked up by 4/6/7 below).
        let mut j = i;
        if dialect == Gpt2Dialect::Original {
            if j < chars.len() && chars[j] == ' ' {
                j += 1;
            }
        } else if j < chars.len()
            && chars[j] != '\r'
            && chars[j] != '\n'
            && !is_letter(chars[j])
            && !is_number(chars[j])
        {
            // Note: marks are NOT excluded here even in dialects whose runs
            // include them -- every shipped prefix class is `[^\r\n\p{L}\p{N}]`,
            // so a stray mark attaches forward as prefix punctuation.
            j += 1;
        }
        if j < chars.len() && is_ll(chars[j]) {
            while j < chars.len() && is_ll(chars[j]) {
                j += 1;
            }
            i = j;
            out.push(chars[start..i].iter().collect());
            continue;
        }
        // 3. Numbers: optional leading space in the original only, then the
        // dialect's digit cap.
        if dialect == Gpt2Dialect::Original && chars[i] == ' ' {
            j = i + 1;
        } else {
            j = i;
        }
        if j < chars.len() && is_number(chars[j]) {
            let mut count = 0;
            while j < chars.len() && is_number(chars[j]) && max_digits.is_none_or(|max| count < max)
            {
                j += 1;
                count += 1;
            }
            i = j;
            out.push(chars[start..i].iter().collect());
            continue;
        }
        // 4. Punctuation/symbol runs, optional leading space; the
        // newline-aware dialects append trailing newlines.
        j = i;
        if chars[j] == ' ' {
            j += 1;
        }
        if j < chars.len() && !is_space(chars[j]) && !is_ll(chars[j]) && !is_number(chars[j]) {
            while j < chars.len() && !is_space(chars[j]) && !is_ll(chars[j]) && !is_number(chars[j])
            {
                j += 1;
            }
            if newline_aware {
                while j < chars.len() && is_newline(chars[j]) {
                    j += 1;
                }
            }
            i = j;
            out.push(chars[start..i].iter().collect());
            continue;
        }
        // 5. Newline-aware dialects only: whitespace run containing a newline.
        if newline_aware {
            j = i;
            let mut seen_newline = false;
            while j < chars.len() && is_space(chars[j]) {
                seen_newline = seen_newline || is_newline(chars[j]);
                j += 1;
            }
            if seen_newline {
                i = j;
                out.push(chars[start..i].iter().collect());
                continue;
            }
        }
        // 6. Trailing whitespace (not followed by a non-space).
        j = i;
        while j < chars.len() && is_space(chars[j]) {
            j += 1;
        }
        if j > i && j == chars.len() {
            i = j;
            out.push(chars[start..i].iter().collect());
            continue;
        }
        // 7. Plain whitespace run.
        if j > i {
            i = j;
            out.push(chars[start..i].iter().collect());
            continue;
        }
        anyhow::bail!(
            "GPT-2 splitter stalled at {:?} (offset {i}); no alternative matched",
            chars[i].to_string()
        );
    }
    Ok(out)
}

/// A single added token from `tokenizer.json` (unknown fields ignored).
#[derive(Debug, Clone, Deserialize)]
struct AddedTokenRaw {
    id: u32,
    content: String,
}

/// One merge entry: `"a b"` string or `["a", "b"]` array (both shipped in
/// real files).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Joined(String),
    Pair(Vec<String>),
}

/// The `model` section of a BPE `tokenizer.json`.
#[derive(Debug, Clone, Deserialize)]
struct BpeModelRaw {
    #[serde(default)]
    vocab: HashMap<String, u32>,
    #[serde(default)]
    merges: Vec<MergeEntry>,
}

/// Minimal top-level shape of a HuggingFace `tokenizer.json`.
#[derive(Debug, Clone, Deserialize)]
struct TokenizerFile {
    #[serde(default)]
    model: Option<BpeModelRaw>,
    #[serde(default)]
    added_tokens: Vec<AddedTokenRaw>,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
    #[serde(default)]
    decoder: Option<serde_json::Value>,
    #[serde(default)]
    bos_token: Option<serde_json::Value>,
    #[serde(default)]
    eos_token: Option<serde_json::Value>,
    #[serde(default)]
    pad_token: Option<serde_json::Value>,
    #[serde(default)]
    unk_token: Option<serde_json::Value>,
}

/// Normalizers. The Unicode forms (NFC/NFD/NFKC/NFKD) run on the standard
/// decomposition tables via `unicode-normalization`; everything else is a
/// `std`-level string operation. Regex Replace stays a loud error: silently
/// treating a pattern as a literal (or vice versa) corrupts ids that look
/// right.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum Normalizer {
    #[default]
    None,
    Lowercase,
    Strip {
        left: bool,
        right: bool,
    },
    Prepend(String),
    Replace {
        from: String,
        to: String,
    },
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    fn parse(value: &Option<serde_json::Value>) -> Result<Self> {
        match value {
            None | Some(serde_json::Value::Null) => Ok(Self::None),
            Some(serde_json::Value::Object(map)) => {
                let kind = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<missing type>");
                match kind {
                    "Lowercase" => Ok(Self::Lowercase),
                    "NFC" => Ok(Self::Nfc),
                    "NFD" => Ok(Self::Nfd),
                    "NFKC" => Ok(Self::Nfkc),
                    "NFKD" => Ok(Self::Nfkd),
                    "Strip" => Ok(Self::Strip {
                        left: map
                            .get("strip_left")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                        right: map
                            .get("strip_right")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                    }),
                    "Prepend" => {
                        let s = map
                            .get("prepend")
                            .and_then(|v| v.as_str())
                            .context("Prepend normalizer needs a \"prepend\" string")?;
                        Ok(Self::Prepend(s.to_string()))
                    }
                    "Replace" => {
                        let pattern = map.get("pattern").unwrap_or(&serde_json::Value::Null);
                        let from = match pattern {
                            serde_json::Value::String(s) => Some(s.clone()),
                            serde_json::Value::Object(pat) => pat
                                .get("String")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            _ => None,
                        }
                        .context("Replace normalizer needs a literal String pattern (regex Replace is unsupported)")?;
                        let to = map.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        Ok(Self::Replace {
                            from,
                            to: to.to_string(),
                        })
                    }
                    "Sequence" => {
                        let items = map
                            .get("normalizers")
                            .and_then(|v| v.as_array())
                            .context("Sequence normalizer needs a \"normalizers\" array")?;
                        items
                            .iter()
                            .map(|v| Self::parse(&Some(v.clone())))
                            .collect::<Result<Vec<_>>>()
                            .map(Self::Sequence)
                    }
                    other => anyhow::bail!("unsupported normalizer type {other:?}"),
                }
            }
            _ => anyhow::bail!("unrecognized normalizer section shape"),
        }
    }

    fn apply(&self, text: &str) -> String {
        match self {
            Self::None => text.to_string(),
            Self::Lowercase => text.to_lowercase(),
            Self::Strip { left, right } => {
                let mut out = text;
                if *left {
                    out = out.trim_start();
                }
                if *right {
                    out = out.trim_end();
                }
                out.to_string()
            }
            Self::Prepend(prefix) => format!("{prefix}{text}"),
            Self::Replace { from, to } => text.replace(from.as_str(), to.as_str()),
            Self::Nfc => text.nfc().collect(),
            Self::Nfd => text.nfd().collect(),
            Self::Nfkc => text.nfkc().collect(),
            Self::Nfkd => text.nfkd().collect(),
            Self::Sequence(items) => {
                let mut out = text.to_string();
                for item in items {
                    out = item.apply(&out);
                }
                out
            }
        }
    }
}

/// A loaded BPE tokenizer: merge engine over `u32` ids.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string -> id, including added tokens (added win on collision).
    vocab: HashMap<String, u32>,
    /// Token id -> string. Dense: index `i` holds the string for id `i`.
    /// Gaps (should not happen in shipped files) hold `None`.
    decode_table: Vec<Option<String>>,
    /// Merge rank: `(left, right)` -> earliest position in the merges list.
    ranks: HashMap<(String, String), usize>,
    /// Added-token contents sorted longest-first for greedy splitting.
    added: Vec<(String, u32)>,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    /// Pre-tokenization declared by the file; [`Pretokenizer::Whitespace`]
    /// when the file declares none (the legacy behavior).
    default_strategy: Pretokenizer,
    /// Which GPT-2 split dialect the file declares (only meaningful with
    /// [`Pretokenizer::Gpt2`]).
    dialect: Gpt2Dialect,
    /// Whether text passes through the byte alphabet
    /// ([`bytes_to_unicode_table`]) before merging. True for ByteLevel
    /// pipelines; false keeps the legacy literal behavior.
    byte_level: bool,
    /// ByteLevel `add_prefix_space`: prepend one space when the input does
    /// not start with whitespace.
    prefix_space: bool,
    /// Byte table shared by encode and decode (built once at load).
    byte_table: [char; 256],
    /// File-declared normalizer, applied before added-token splitting so
    /// special tokens are protected from it (HF protection semantics).
    normalizer: Normalizer,
}

/// Derive `(strategy, dialect, byte_level, prefix_space)` from a file's
/// `pre_tokenizer` section. Unknown types and unrecognized regex strings are
/// loud errors: silently falling back to whitespace splitting would produce
/// ids that look right and are wrong.
fn interpret_pretokenizer(
    value: &Option<serde_json::Value>,
) -> Result<(Pretokenizer, Gpt2Dialect, bool, bool)> {
    // Gpt2Dialect has no Default; Original is only a fallback that callers
    // never observe unless the strategy is Gpt2 (enforced below).
    const FALLBACK: Gpt2Dialect = Gpt2Dialect::Original;
    match value {
        None | Some(serde_json::Value::Null) => {
            Ok((Pretokenizer::Whitespace, FALLBACK, false, false))
        }
        Some(serde_json::Value::Object(map)) => {
            match map.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                "WhitespaceSplit" | "Whitespace" => {
                    Ok((Pretokenizer::Whitespace, FALLBACK, false, false))
                }
                "ByteLevel" => {
                    let prefix = map
                        .get("add_prefix_space")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    Ok((Pretokenizer::Gpt2, Gpt2Dialect::Original, true, prefix))
                }
                "Split" => {
                    let pattern = map
                        .get("pattern")
                        .and_then(|v| v.get("Regex"))
                        .and_then(|v| v.as_str())
                        .context("Split pre-tokenizer needs a {\"Regex\": ...} pattern")?;
                    match match_gpt2_pattern(pattern) {
                        Some(dialect) => Ok((Pretokenizer::Gpt2, dialect, false, false)),
                        None => anyhow::bail!(
                            "unsupported Split regex {pattern:?}: only the GPT-2-family patterns are implemented"
                        ),
                    }
                }
                "Sequence" => {
                    let items = map
                        .get("pretokenizers")
                        .and_then(|v| v.as_array())
                        .context("Sequence pre-tokenizer needs a \"pretokenizers\" array")?;
                    let mut strategy = Pretokenizer::Whitespace;
                    let mut dialect = FALLBACK;
                    let mut byte_level = false;
                    let mut prefix = false;
                    let mut saw_split = false;
                    for item in items {
                        let part = item
                            .as_object()
                            .with_context(|| "pre-tokenizer item must be an object")?;
                        match part.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                            "Split" => {
                                let pattern = part
                                    .get("pattern")
                                    .and_then(|v| v.get("Regex"))
                                    .and_then(|v| v.as_str())
                                    .context(
                                        "Split pre-tokenizer needs a {\"Regex\": ...} pattern",
                                    )?;
                                dialect = match_gpt2_pattern(pattern).with_context(|| {
                                    format!("unsupported Split regex {pattern:?}: only the GPT-2-family patterns are implemented")
                                })?;
                                strategy = Pretokenizer::Gpt2;
                                saw_split = true;
                            }
                            "ByteLevel" => {
                                byte_level = true;
                                prefix = part
                                    .get("add_prefix_space")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                if !saw_split {
                                    strategy = Pretokenizer::Gpt2;
                                    dialect = Gpt2Dialect::Original;
                                }
                            }
                            other => anyhow::bail!(
                                "unsupported pre-tokenizer type {other:?} in Sequence"
                            ),
                        }
                    }
                    Ok((strategy, dialect, byte_level, prefix))
                }
                other => anyhow::bail!("unsupported pre-tokenizer type {other:?}"),
            }
        }
        _ => anyhow::bail!("unrecognized pre_tokenizer section shape"),
    }
}

fn token_content(value: &Option<serde_json::Value>) -> Option<String> {
    match value {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Object(map)) => map
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    }
}

impl BpeTokenizer {
    /// Load from the text of a `tokenizer.json` file.
    pub fn from_json(text: &str) -> Result<Self> {
        let file: TokenizerFile = serde_json::from_str(text).context("parse tokenizer.json")?;
        let normalizer = Normalizer::parse(&file.normalizer)?;
        let (default_strategy, dialect, mut byte_level, prefix_space) =
            interpret_pretokenizer(&file.pre_tokenizer)?;
        // The decoder confirms byte level independently: either side may
        // declare it.
        if let Some(serde_json::Value::Object(decoder)) = &file.decoder {
            if decoder.get("type").and_then(|v| v.as_str()) == Some("ByteLevel") {
                byte_level = true;
            }
        }
        let model = file.model.context("tokenizer.json has no model section")?;
        ensure!(!model.vocab.is_empty(), "tokenizer.json vocab is empty");

        // Added tokens commonly sit past the base vocabulary (e.g. base
        // 0..154819 with added id 154820+), so the table spans both.
        let size = model
            .vocab
            .values()
            .copied()
            .chain(file.added_tokens.iter().map(|t| t.id))
            .max()
            .unwrap_or(0)
            .saturating_add(1) as usize;
        let mut decode_table: Vec<Option<String>> = vec![None; size];
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(size);
        for (token, id) in &model.vocab {
            vocab.insert(token.clone(), *id);
            if let Some(slot) = decode_table.get_mut(*id as usize) {
                ensure!(slot.is_none(), "duplicate id {id} in tokenizer.json vocab");
                *slot = Some(token.clone());
            } else {
                anyhow::bail!("vocab id {id} out of range for table of {size}");
            }
        }

        // HF files occasionally carry a version header as the first line
        // (`#version: 0.2`); it is not a merge and must be skipped, not
        // parsed as a pair.
        let mut ranks: HashMap<(String, String), usize> = HashMap::new();
        for (rank, entry) in model.merges.iter().enumerate() {
            let pair: Option<(String, String)> = match entry {
                MergeEntry::Joined(line) => {
                    if line.starts_with('#') {
                        continue;
                    }
                    let mut parts = line.splitn(2, ' ');
                    match (parts.next(), parts.next()) {
                        (Some(left), Some(right)) => Some((left.to_string(), right.to_string())),
                        _ => None,
                    }
                }
                MergeEntry::Pair(pair) => {
                    if pair.len() == 2 {
                        Some((pair[0].clone(), pair[1].clone()))
                    } else {
                        None
                    }
                }
            };
            match pair {
                Some((left, right)) => {
                    ranks.entry((left, right)).or_insert(rank);
                }
                None => anyhow::bail!("malformed merge entry {rank}: {entry:?}"),
            }
        }

        let mut added: Vec<(String, u32)> = Vec::with_capacity(file.added_tokens.len());
        for tok in &file.added_tokens {
            vocab.insert(tok.content.clone(), tok.id);
            if let Some(slot) = decode_table.get_mut(tok.id as usize) {
                *slot = Some(tok.content.clone());
            } else {
                anyhow::bail!("added-token id {} out of range", tok.id);
            }
            added.push((tok.content.clone(), tok.id));
        }
        // Longest match first so `<|im_start|>` wins over `<|im`.
        added.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let resolve = |vocab: &HashMap<String, u32>,
                       value: &Option<serde_json::Value>|
         -> Result<Option<u32>> {
            match token_content(value) {
                None => Ok(None),
                Some(content) => vocab
                    .get(&content)
                    .copied()
                    .map(Some)
                    .with_context(|| format!("special token {content:?} not in vocab")),
            }
        };
        let bos = resolve(&vocab, &file.bos_token)?;
        let eos = resolve(&vocab, &file.eos_token)?;
        let pad = resolve(&vocab, &file.pad_token)?;
        let unk = resolve(&vocab, &file.unk_token)?;

        Ok(Self {
            vocab,
            decode_table,
            ranks,
            added,
            bos,
            eos,
            pad,
            unk,
            default_strategy,
            dialect,
            byte_level,
            prefix_space,
            byte_table: bytes_to_unicode_table(),
            normalizer,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.decode_table.len()
    }

    pub fn bos(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos(&self) -> Option<u32> {
        self.eos
    }

    pub fn pad(&self) -> Option<u32> {
        self.pad
    }

    pub fn unk(&self) -> Option<u32> {
        self.unk
    }

    /// Split input into (chunk, is_added_token) pieces. Added tokens match
    /// greedily longest-first and are never merged across.
    fn split_added(&self, text: &str) -> Vec<(String, bool)> {
        if self.added.is_empty() {
            return vec![(text.to_string(), false)];
        }
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut hit: Option<&(String, u32)> = None;
            for candidate in &self.added {
                if rest.starts_with(candidate.0.as_str()) {
                    hit = Some(candidate);
                    break;
                }
            }
            match hit {
                Some((content, _)) => {
                    out.push((content.clone(), true));
                    rest = &rest[content.len()..];
                }
                None => {
                    // Advance one character (not byte: never split UTF-8).
                    let next = rest
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| i)
                        .unwrap_or(rest.len());
                    let (head, tail) = rest.split_at(next);
                    match out.last_mut() {
                        Some((chunk, false)) => chunk.push_str(head),
                        _ => out.push((head.to_string(), false)),
                    }
                    rest = tail;
                }
            }
        }
        out
    }

    /// Whitespace splitter: runs attach to the following word; all-whitespace
    /// input is one chunk, not zero. Empty input yields no chunks.
    fn split_words(text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut pending_space = String::new();
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                pending_space.push(ch);
            } else {
                if !pending_space.is_empty() {
                    current.push_str(&pending_space);
                    pending_space.clear();
                }
                current.push(ch);
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        } else if !pending_space.is_empty() {
            chunks.push(pending_space);
        }
        chunks
    }

    /// BPE merge loop over one pre-tokenized chunk. Starts from Unicode
    /// scalar values; unknown single characters fall back to the file's
    /// `unk_token` when one exists, else error naming the character.
    fn encode_chunk(&self, chunk: &str) -> Result<Vec<u32>> {
        let mut symbols: Vec<String> = chunk.chars().map(|c| c.to_string()).collect();
        if symbols.is_empty() {
            return Ok(Vec::new());
        }
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = self
                    .ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.is_none_or(|(_, r)| *rank < r) {
                        best = Some((i, *rank));
                    }
                }
            }
            match best {
                None => break,
                Some((i, _)) => {
                    let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                    symbols.splice(i..=i + 1, [merged]);
                }
            }
        }
        symbols
            .iter()
            .map(|s| {
                self.vocab
                    .get(s)
                    .copied()
                    .or_else(|| {
                        // Byte fallback `<0xHH>`: decode-time convention shared
                        // with HF files that set byte_fallback.
                        byte_fallback_id(s).and_then(|id| {
                            if (id as usize) < self.decode_table.len() {
                                Some(id)
                            } else {
                                None
                            }
                        })
                    })
                    .or(self.unk)
                    .with_context(|| format!("no token for {s:?} and no unk_token"))
            })
            .collect()
    }

    /// Encode with an explicit pre-tokenization strategy, overriding the
    /// file's declared default. Byte mapping follows the *file* (byte-level
    /// files map every chunk; literal files map nothing), never the
    /// strategy, so overriding the splitter cannot silently produce
    /// unmatchable units.
    pub fn encode_with(&self, text: &str, strategy: Pretokenizer) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        // ByteLevel add_prefix_space applies once to the whole input, before
        // added-token protection (a leading added token simply follows it).
        let owned;
        let text = if strategy == Pretokenizer::Gpt2
            && self.prefix_space
            && !text.is_empty()
            && !text.starts_with(|c: char| c.is_whitespace())
        {
            owned = format!(" {text}");
            owned.as_str()
        } else {
            text
        };
        // Added tokens split first so normalization and merging never touch
        // them (HF protection semantics).
        for (piece, is_added) in self.split_added(text) {
            if is_added {
                let id = self
                    .vocab
                    .get(&piece)
                    .copied()
                    .with_context(|| format!("added token {piece:?} vanished from vocab"))?;
                ids.push(id);
                continue;
            }
            let normalized = self.normalizer.apply(&piece);
            let chunks: Vec<String> = match strategy {
                Pretokenizer::Raw => {
                    if normalized.is_empty() {
                        Vec::new()
                    } else {
                        vec![normalized]
                    }
                }
                Pretokenizer::Whitespace => Self::split_words(&normalized),
                Pretokenizer::Gpt2 => split_gpt2(&normalized, self.dialect)?,
            };
            for chunk in chunks {
                let unit = if self.byte_level {
                    map_bytes_to_unicode(&chunk, &self.byte_table)
                } else {
                    chunk
                };
                ids.extend(self.encode_chunk(&unit)?);
            }
        }
        Ok(ids)
    }

    /// Encode with the file's declared strategy ([`Pretokenizer::Whitespace`]
    /// when the file declares none).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_with(text, self.default_strategy)
    }

    /// Decode ids to text. Byte-level files concatenate bytes
    /// (`<0xHH>` tokens, mapped characters, literal UTF-8 as fallback) and
    /// decode once at the end; literal files push token strings. Unknown ids
    /// are an error (callers sampling from a distribution should filter first
    /// — this path refuses to guess).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if !self.byte_level {
            let mut out = String::new();
            let mut bytes: Vec<u8> = Vec::new();
            let flush = |out: &mut String, bytes: &mut Vec<u8>| {
                if !bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(bytes));
                    bytes.clear();
                }
            };
            for id in ids {
                let token = self
                    .decode_table
                    .get(*id as usize)
                    .and_then(|slot| slot.as_deref())
                    .with_context(|| {
                        format!("id {id} outside vocabulary of {}", self.decode_table.len())
                    })?;
                if let Some(byte) = parse_byte_fallback(token) {
                    bytes.push(byte);
                } else {
                    flush(&mut out, &mut bytes);
                    out.push_str(token);
                }
            }
            flush(&mut out, &mut bytes);
            return Ok(out);
        }
        let mut bytes: Vec<u8> = Vec::new();
        for id in ids {
            let token = self
                .decode_table
                .get(*id as usize)
                .and_then(|slot| slot.as_deref())
                .with_context(|| {
                    format!("id {id} outside vocabulary of {}", self.decode_table.len())
                })?;
            if let Some(byte) = parse_byte_fallback(token) {
                bytes.push(byte);
            } else {
                bytes.extend(unmap_unicode_to_bytes(token, &self.byte_table));
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// `<0xHH>` -> byte value, the HF byte-fallback spelling.
fn parse_byte_fallback(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() == 2 {
        u8::from_str_radix(inner, 16).ok()
    } else {
        None
    }
}

/// Mirror of [`parse_byte_fallback`] for the encode side.
fn byte_fallback_id(symbol: &str) -> Option<u32> {
    parse_byte_fallback(symbol).map(u32::from)
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

    /// Minimal Qwen-shaped fixture: merges fold `h e l l o`, added chat
    /// tokens bypass merging, byte fallback covers the rest.
    const FIXTURE: &str = r##"{
        "version": "1.0",
        "normalizer": null,
        "pre_tokenizer": null,
        "model": {
            "type": "BPE",
            "unk_token": null,
            "vocab": {
                "h": 0, "e": 1, "l": 2, "o": 3, " ": 4,
                "he": 5, "ll": 6, "hell": 7, "hello": 8,
                "<0x41>": 9, "<|im_start|>": 10, "<|im_end|>": 11
            },
            "merges": ["#version: 0.2", "h e", "l l", "he ll", "hell o"]
        },
        "added_tokens": [
            {"id": 10, "content": "<|im_start|>", "special": true},
            {"id": 11, "content": "<|im_end|>", "special": true}
        ],
        "bos_token": null,
        "eos_token": "<|im_end|>",
        "pad_token": null,
        "unk_token": null
    }"##;

    fn fixture() -> BpeTokenizer {
        BpeTokenizer::from_json(FIXTURE).expect("fixture must load")
    }

    #[test]
    fn test_merge_loop_folds_by_rank() {
        // Ranks: h+e=1, l+l=2, he+ll=3, hell+o=4. "hello" must fold fully
        // regardless of scan order because lowest rank always wins.
        let tok = fixture();
        assert_eq!(
            tok.encode_with("hello", Pretokenizer::Raw).unwrap(),
            vec![8]
        );
        assert_eq!(tok.encode_with("hell", Pretokenizer::Raw).unwrap(), vec![7]);
        // No applicable merge: stays split.
        assert_eq!(
            tok.encode_with("helo", Pretokenizer::Raw).unwrap(),
            vec![5, 2, 3]
        );
    }

    #[test]
    fn test_version_header_is_not_a_merge() {
        let tok = fixture();
        // If "#version: 0.2" were parsed as a pair, ranks would shift and
        // "hello" would fold differently. Rank order pins the header out.
        assert_eq!(
            tok.encode_with("hello", Pretokenizer::Raw).unwrap(),
            vec![8]
        );
    }

    #[test]
    fn test_added_tokens_bypass_merging() {
        let tok = fixture();
        let ids = tok
            .encode_with("<|im_start|>hello<|im_end|>", Pretokenizer::Raw)
            .unwrap();
        assert_eq!(ids, vec![10, 8, 11]);
        assert_eq!(tok.eos(), Some(11));
    }

    #[test]
    fn test_roundtrip_and_unknown_policy() {
        let tok = fixture();
        // Fixture lacks most letters, so only the mergeable subset
        // round-trips; unknown characters must error, not guess.
        let ids = tok.encode_with("hello", Pretokenizer::Raw).unwrap();
        assert_eq!(tok.decode(&ids).unwrap(), "hello");
        assert!(tok.encode_with("xyz", Pretokenizer::Raw).is_err());
        assert!(tok.encode_with("", Pretokenizer::Raw).unwrap().is_empty());
    }

    #[test]
    fn test_byte_fallback_decodes() {
        let tok = fixture();
        assert_eq!(tok.decode(&[9]).unwrap(), "A");
        assert_eq!(tok.decode(&[8, 9]).unwrap(), "helloA");
        assert!(tok.decode(&[999]).is_err());
    }

    #[test]
    fn test_whitespace_splitter_shape() {
        assert_eq!(
            BpeTokenizer::split_words("hi  there"),
            vec!["hi".to_string(), "  there".to_string()]
        );
        assert_eq!(BpeTokenizer::split_words("   "), vec!["   ".to_string()]);
        assert!(BpeTokenizer::split_words("").is_empty());
    }

    #[test]
    fn test_rejects_normalizer_and_bad_files() {
        assert!(BpeTokenizer::from_json(r#"{"model": {"vocab": {}, "merges": []}}"#).is_err());
        // NFC now loads (unicode-normalization tables); unknown types and
        // regex Replace still refuse loudly.
        assert!(BpeTokenizer::from_json(
            r#"{"normalizer": {"type": "NFC"}, "model": {"vocab": {"a": 0}, "merges": []}}"#
        )
        .is_ok());
        assert!(BpeTokenizer::from_json(
            r#"{"normalizer": {"type": "NFKC"}, "model": {"vocab": {"a": 0}, "merges": []}}"#
        )
        .is_ok());
        assert!(BpeTokenizer::from_json(
            r#"{"normalizer": {"type": "Mystery"}, "model": {"vocab": {"a": 0}, "merges": []}}"#
        )
        .is_err());
        assert!(
            BpeTokenizer::from_json(r#"{"model": {"vocab": {"a": 0}, "merges": ["lonely"]}}"#)
                .is_err()
        );
    }

    /// Real-file parity gate over a Qwen `tokenizer.json`: scale, declared
    /// pipeline (NFC + Qwen dialect + byte level), and the round-trip
    /// property, which any correct pipeline satisfies. Reference id vectors
    /// (from `transformers` `AutoTokenizer`) are still unrecorded. Skips
    /// with a note where no file exists; set `QWEN_TOKENIZER_JSON` to point
    /// at one.
    #[test]
    fn qwen_tokenizer_parity_against_reference() {
        let candidates = [
            std::env::var("QWEN_TOKENIZER_JSON").unwrap_or_default(),
            "/srv/m-sdd/unifur/tokenizers/qwen38-source/tokenizer.json".to_string(),
        ];
        let path = candidates
            .iter()
            .find(|p| !p.is_empty() && std::path::Path::new(p.as_str()).exists());
        let Some(path) = path else {
            eprintln!("skip: no Qwen tokenizer.json (set QWEN_TOKENIZER_JSON)");
            return;
        };
        let text = std::fs::read_to_string(path.as_str()).expect("read tokenizer.json");
        let tok = BpeTokenizer::from_json(&text).expect("load real tokenizer.json");
        assert!(tok.vocab_size() > 100_000, "expected a Qwen-scale vocab");
        assert_eq!(tok.default_strategy, Pretokenizer::Gpt2);
        assert!(tok.byte_level, "expected a ByteLevel pipeline");
        for sample in [
            "Hello, world!",
            "def fib(n):\n    return n",
            "emoji: \u{1f980} caf\u{e9}",
            "  spaced   out  ",
            "don't stop",
            "x=1+22-333*4444",
        ] {
            let ids = tok
                .encode(sample)
                .unwrap_or_else(|e| panic!("encode failed on {sample:?}: {e}"));
            let back = tok
                .decode(&ids)
                .unwrap_or_else(|e| panic!("decode failed on {sample:?}: {e}"));
            assert_eq!(back, sample, "round-trip broke on {sample:?}");
        }
    }

    #[test]
    fn test_byte_table_is_bijective_with_gpt2_values() {
        let table = bytes_to_unicode_table();
        // ASCII printables and Latin-1 map to themselves...
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b' ' as usize], '\u{120}'); // space -> G with dot
        assert_eq!(table[0], '\u{100}');
        assert_eq!(table[255], 'ÿ'); // 0xFF maps to itself (174..=255 range)
                                     // ...every byte invertible, no collisions.
        let mut seen = std::collections::HashSet::new();
        for (byte, ch) in table.iter().enumerate() {
            assert!(seen.insert(*ch), "collision at byte {byte}");
            assert_eq!(unicode_to_byte(*ch, &table), Some(byte as u8));
        }
        // Text through the map and back is the UTF-8 bytes, exactly.
        for text in ["hello", "caf\u{e9} \u{4e2d}\u{1f980}", "\0\t\n"] {
            let mapped = map_bytes_to_unicode(text, &table);
            assert_eq!(unmap_unicode_to_bytes(&mapped, &table), text.as_bytes());
        }
    }

    #[test]
    fn test_splitter_recognizes_all_three_dialects() {
        assert_eq!(
            match_gpt2_pattern(GPT2_ORIGINAL_PATTERN),
            Some(Gpt2Dialect::Original)
        );
        assert_eq!(
            match_gpt2_pattern(GPT2_GLM_PATTERN),
            Some(Gpt2Dialect::GlmFamily)
        );
        assert_eq!(
            match_gpt2_pattern(GPT2_QWEN_PATTERN),
            Some(Gpt2Dialect::Qwen)
        );
        assert_eq!(match_gpt2_pattern("(?i:'s)|\\p{L}+"), None);
    }

    #[test]
    fn test_qwen_dialect_marks_and_single_digits() {
        // e + COMBINING ACUTE (U+0301) + x, built from the scalar value
        // so no editor ever silently precomposes it: Qwen joins the mark
        // backward into the letter run; GLM (no marks class)
        // prefix-attaches it forward to the next letter instead.
        let mark = char::from_u32(0x301).expect("U+0301 exists");
        let decomposed = format!("e{mark}x");
        assert_eq!(
            split_gpt2(&decomposed, Gpt2Dialect::Qwen).unwrap(),
            vec![decomposed.clone()]
        );
        assert_eq!(
            split_gpt2(&decomposed, Gpt2Dialect::GlmFamily).unwrap(),
            vec!["e".to_string(), format!("{mark}x")]
        );
        assert_eq!(
            split_gpt2("x=1+22-333*4444", Gpt2Dialect::Qwen).unwrap(),
            vec!["x", "=", "1", "+", "2", "2", "-", "3", "3", "3", "*", "4", "4", "4", "4"]
        );
        assert_eq!(
            split_gpt2("x=1+22-333*4444", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["x", "=", "1", "+", "22", "-", "333", "*", "444", "4"]
        );
    }

    #[test]
    fn test_nfc_normalizer() {
        use serde_json::json;
        let parse = |v: serde_json::Value| Normalizer::parse(&Some(v)).unwrap();
        let acute = '\u{301}';
        let decomposed = format!("e{acute}");
        let composed = "é";
        // e + combining acute -> precomposed single char, and back.
        assert_eq!(parse(json!({"type": "NFC"})).apply(&decomposed), composed);
        assert_eq!(
            parse(json!({"type": "NFD"}))
                .apply(composed)
                .chars()
                .count(),
            2
        );
        assert_eq!(parse(json!({"type": "NFKC"})).apply(&decomposed), composed);
    }

    #[test]
    fn test_gpt2_splitter_both_dialects() {
        // GLM dialect: the space qualifies as the optional prefix punct, so
        // it attaches to the following word (exactly what the regex says);
        // digits cap at three, contractions fold, newlines group.
        assert_eq!(
            split_gpt2("Hello, world!", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["Hello", ",", " world", "!"]
        );
        assert_eq!(
            split_gpt2("don't", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["don", "'t"]
        );
        assert_eq!(
            split_gpt2("DON'T", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["DON", "'T"]
        );
        assert_eq!(
            split_gpt2("abc12345", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["abc", "123", "45"]
        );
        assert_eq!(
            split_gpt2("a\n\nb", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["a", "\n\n", "b"]
        );
        assert_eq!(
            split_gpt2("a ", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["a", " "]
        );
        assert_eq!(
            split_gpt2("!abc", Gpt2Dialect::GlmFamily).unwrap(),
            vec!["!abc"]
        );
        // Original dialect: leading spaces attach, digits unbounded,
        // case-sensitive contractions.
        assert_eq!(
            split_gpt2("Hello, world!", Gpt2Dialect::Original).unwrap(),
            vec!["Hello", ",", " world", "!"]
        );
        assert_eq!(
            split_gpt2("abc12345", Gpt2Dialect::Original).unwrap(),
            vec!["abc", "12345"]
        );
        assert_eq!(
            split_gpt2("DON'T", Gpt2Dialect::Original).unwrap(),
            vec!["DON", "'", "T"]
        );
        // Full consumption on adversarial input in both dialects.
        let nasty = "  a1! \t\nB-2.5x_9'end  \u{4e2d}\u{1f980}  ";
        for dialect in [Gpt2Dialect::Original, Gpt2Dialect::GlmFamily] {
            let pieces = split_gpt2(nasty, dialect).unwrap();
            assert_eq!(
                pieces.concat(),
                nasty,
                "splitter dropped text ({dialect:?})"
            );
        }
    }

    /// Byte-level fixture the way GPT-2-family files look. U+0120 is built
    /// with an unambiguous Rust escape, never a lookalike literal.
    fn mini_byte_fixture() -> BpeTokenizer {
        let gdot = '\u{120}';
        let template = r##"{
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false}
            ]},
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "GDOTPLACE": 4, "he": 5, "ll": 6, "hell": 7, "hello": 8, "GDOTPLACEhello": 10, "GDOTPLACEh": 11, "GDOTPLACEhe": 12, "GDOTPLACEhell": 13},
                "merges": ["GDOTPLACE h", "GDOTPLACEh e", "GDOTPLACEhe ll", "GDOTPLACEhell o", "h e", "l l", "he ll", "hell o"]
            },
            "added_tokens": [],
            "bos_token": null, "eos_token": null, "pad_token": null, "unk_token": null
        }"##;
        let file = template.replace("GDOTPLACE", &gdot.to_string());
        // Sanity: the placeholder scheme really produced U+0120 spellings.
        assert!(file.contains(gdot));
        BpeTokenizer::from_json(&file).unwrap()
    }

    #[test]
    fn test_byte_level_mini_pipeline() {
        let tok = mini_byte_fixture();
        assert_eq!(tok.default_strategy, Pretokenizer::Gpt2);
        assert!(tok.byte_level);
        // " hello" splits to [" hello"], maps to "U+0120 hello", and the
        // early-ranked space-merges (as in real files, where frequent
        // space-initial merges rank first) fold it fully to one id. Rank
        // order matters: space-merges ranked *after* the plain ones would
        // starve and strand ["U+0120", "hello"] -- the engine follows ranks
        // faithfully either way.
        let ids = tok.encode(" hello").unwrap();
        assert_eq!(ids, vec![10]);
        assert_eq!(tok.decode(&ids).unwrap(), " hello");
        assert_eq!(tok.decode(&tok.encode("he").unwrap()).unwrap(), "he");
    }

    #[test]
    fn test_normalizers() {
        use serde_json::json;
        let parse = |v: serde_json::Value| Normalizer::parse(&Some(v)).unwrap();
        assert_eq!(parse(json!({"type": "Lowercase"})).apply("AbC"), "abc");
        assert_eq!(
            parse(json!({"type": "Strip", "strip_left": true, "strip_right": false})).apply("  x "),
            "x "
        );
        assert_eq!(
            parse(json!({"type": "Prepend", "prepend": ">>"})).apply("x"),
            ">>x"
        );
        assert_eq!(
            parse(json!({"type": "Replace", "pattern": {"String": "a"}, "content": "b"}))
                .apply("aaa"),
            "bbb"
        );
        assert_eq!(
            parse(json!({"type": "Sequence", "normalizers": [
                {"type": "Strip", "strip_left": true, "strip_right": true},
                {"type": "Lowercase"}
            ]}))
            .apply("  AbC "),
            "abc"
        );
        assert!(Normalizer::parse(&Some(json!({"type": "NFKC"}))).is_ok());
        assert!(Normalizer::parse(&Some(json!({"type": "Mystery"}))).is_err());
        assert!(Normalizer::parse(&Some(
            json!({"type": "Replace", "pattern": {"Regex": "a+"}, "content": "b"})
        ))
        .is_err());
    }

    /// Real on-disk GPT-2-family files (e.g. `/srv` model mirrors on this
    /// machine): load, check the declared pipeline, round-trip diverse text,
    /// confirm merges fire and added tokens bypass. Skips with a note where
    /// no file exists so other machines stay green.
    #[test]
    fn test_real_world_file_loads_and_roundtrips() {
        let candidates = [
            std::env::var("GLM_TOKENIZER_JSON").unwrap_or_default(),
            "/srv/m-sdd/GLM-5.3-Flash-peregrine/tokenizer.json".to_string(),
            "/srv/m-sda/GLM-5.2-r5/tokenizer.json".to_string(),
        ];
        let path = candidates
            .iter()
            .find(|p| !p.is_empty() && std::path::Path::new(p.as_str()).exists());
        let Some(path) = path else {
            eprintln!("skip: no real tokenizer.json (set GLM_TOKENIZER_JSON)");
            return;
        };
        let text = std::fs::read_to_string(path.as_str()).expect("read real tokenizer.json");
        let tok = BpeTokenizer::from_json(&text).expect("load real tokenizer.json");
        assert!(
            tok.vocab_size() > 100_000,
            "expected a large vocab, got {}",
            tok.vocab_size()
        );
        assert_eq!(tok.default_strategy, Pretokenizer::Gpt2);
        assert!(tok.byte_level, "expected a ByteLevel pipeline");
        assert!(!tok.added.is_empty(), "expected added tokens");
        for sample in [
            "Hello, world!",
            "def fib(n):\n    return n",
            "fn main() { println!(\"hi\"); }",
            "caf\u{e9} na\u{ef}ve \u{4e2d}\u{6587} \u{1f980}",
            "  spaced   out  ",
            "don't stop",
            "x=1+22-333*4444",
            "a\n\nb\r\nc",
        ] {
            let ids = tok
                .encode(sample)
                .unwrap_or_else(|e| panic!("encode failed on {sample:?}: {e}"));
            assert!(!ids.is_empty() || sample.is_empty());
            let back = tok
                .decode(&ids)
                .unwrap_or_else(|e| panic!("decode failed on {sample:?}: {e}"));
            assert_eq!(back, sample, "round-trip broke on {sample:?}");
        }
        // Merges must fire on ordinary prose: fewer ids than characters.
        let prose = "the quick brown fox jumps over the lazy dog";
        let ids = tok.encode(prose).unwrap();
        assert!(ids.len() < prose.chars().count(), "no merge fired on prose");
        // Added tokens bypass merging, using the file's own first entry.
        let (content, id) = tok.added.first().expect("added tokens").clone();
        let wrapped = format!("hi{content}there");
        let ids = tok.encode(&wrapped).unwrap();
        assert!(
            ids.contains(&id),
            "added token {content:?} was merged through"
        );
        assert_eq!(tok.decode(&ids).unwrap(), wrapped);
    }
}
