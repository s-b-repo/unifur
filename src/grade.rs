//! Graded supervision: four points of feedback per token (roadmap 24.4).
//!
//! The label sidecar (roadmap 24) answers *"which bad pattern is this token?"*
//! and the penalty answers *"how hard should I push against it?"*. It cannot
//! express the other half of what a reviewer says: this span is **right**, and
//! *how* right. This module adds the missing scale — `+1` correct, `+0.5`
//! partially correct, `0` ungraded, `−0.5` partially wrong, `−1` wrong — and
//! the two per-token planes a graded training step needs to consume it.
//!
//! # What each point value does to the objective
//!
//! With `g` the grade of a target token, `s` its likelihood weight, `w` its
//! charge weight, and `α` the [`crate::lm::Unlikelihood`] strength:
//!
//! ```text
//! g > 0 :  loss += s · (−ln p)                        (an ordinary, scaled target)
//! g = 0 :  loss +=    (−ln p)                         (identical to g = +1)
//! g < 0 :  loss += α · |g| · (−ln(1 − p))             and the token leaves the
//!                                                      likelihood term entirely
//! ```
//!
//! `g = +1` is plain cross-entropy bit for bit, and `g = 0` (Neutral) is
//! defined to mean *ungraded*, so a corpus whose grade sidecar is all zeros
//! trains exactly as it did before this feature existed. That is not a
//! convenience: it is what lets a graded corpus be dropped into an ungraded
//! run without silently changing the objective, and the `grade` verification
//! group certifies it on every `dblocks verify`.
//!
//! Neutral is deliberately **not** "skip this token". Skipping is the
//! [`.mask`](crate::corpus) sidecar's job; a grade is an *opinion*, and an
//! absent opinion should cost nothing rather than remove data. Reading 0 as
//! "skip" would mean a freshly initialized sidecar silently deletes the
//! dataset.
//!
//! # Why a wrong token is charged instead of negatively trained
//!
//! Minimizing `-(-ln p)` = maximizing `ln p` without bound is the failure mode
//! [`crate::lm::Unlikelihood`] exists to prevent. A negative grade therefore
//! scales the *bounded* unlikelihood charge — its gradient is
//! `α · |g| · (1 − p)`, which vanishes as `p → 0` instead of exploding. The
//! point value scales the charge's magnitude; it never changes its sign.
//!
//! # Sidecar format
//!
//! `<corpus>.grades` is one signed byte per corpus token, in corpus order,
//! holding [`Grade::quantized`] in `−2..=2` (two's complement, so a
//! legitimately-empty file is all zero bytes). Any byte outside `−2..=2` is
//! refused on read: it means the file came from something else, and the first
//! bad byte would silently shift every later grade onto the wrong token.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

/// One signed byte per token.
pub const GRADE_BYTES: usize = 1;

/// The score thresholds that turn a continuous quality score into a point
/// value, high to low. Kept as data so the docs, the manifest, and the
/// implementation cannot drift apart.
pub const SCORE_THRESHOLDS: [(f32, Grade); 4] = [
    (0.90, Grade::Correct),
    (0.70, Grade::PartiallyCorrect),
    (0.40, Grade::Neutral),
    (0.20, Grade::PartiallyWrong),
];

/// The supervision point value of one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Grade {
    /// Actively harmful: charge it, do not learn it.
    Wrong,
    /// Right idea, wrong detail: charge it gently.
    PartiallyWrong,
    /// No opinion recorded. Trains as an ordinary supervised token.
    Neutral,
    /// Useful but incomplete: learn it at half weight.
    PartiallyCorrect,
    /// Fully correct: learn it at full weight.
    Correct,
}

impl Grade {
    /// The point value: `-1`, `-0.5`, `0`, `+0.5`, `+1`.
    #[must_use]
    pub const fn points(self) -> f64 {
        match self {
            Self::Wrong => -1.0,
            Self::PartiallyWrong => -0.5,
            Self::Neutral => 0.0,
            Self::PartiallyCorrect => 0.5,
            Self::Correct => 1.0,
        }
    }

    /// The sidecar byte as a signed value: `points × 2`, so the stored range
    /// is `−2..=2` and zero means Neutral.
    #[must_use]
    pub const fn quantized(self) -> i8 {
        match self {
            Self::Wrong => -2,
            Self::PartiallyWrong => -1,
            Self::Neutral => 0,
            Self::PartiallyCorrect => 1,
            Self::Correct => 2,
        }
    }

    /// Decode a sidecar byte. `None` for anything outside `−2..=2`.
    #[must_use]
    pub const fn from_quantized(value: i8) -> Option<Self> {
        match value {
            -2 => Some(Self::Wrong),
            -1 => Some(Self::PartiallyWrong),
            0 => Some(Self::Neutral),
            1 => Some(Self::PartiallyCorrect),
            2 => Some(Self::Correct),
            _ => None,
        }
    }

    /// Snap a real-valued point total to the nearest of the five levels.
    /// Anything outside `[-1.5, 1.5]` is a caller error and clamps.
    #[must_use]
    pub fn from_points(points: f64) -> Self {
        let scaled = (points * 2.0).round().clamp(-2.0, 2.0) as i8;
        Self::from_quantized(scaled).unwrap_or(Self::Neutral)
    }

    /// Quantize a continuous quality score in `[0, 1]` — a
    /// [`crate::codequality`] composite score, a test pass fraction, a judge's
    /// verdict — using [`SCORE_THRESHOLDS`]. Monotone by construction: a better
    /// score never earns fewer points.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        for &(threshold, grade) in &SCORE_THRESHOLDS {
            if score >= threshold {
                return grade;
            }
        }
        Self::Wrong
    }

    /// Parse a user-supplied grade: a name (`wrong`, `partially_wrong`,
    /// `neutral`, `partial`, `correct`), a point value (`-1`, `-0.5`, `0`,
    /// `0.5`, `1`), or `auto` for "no opinion". Hyphens, spaces, casing and
    /// case-folded abbreviations are all accepted, because this string comes
    /// from a terminal.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        let key = trimmed.to_lowercase().replace([' ', '-'], "_");
        let named = match key.as_str() {
            "wrong" | "negative" | "bad" => Some(Self::Wrong),
            "partially_wrong" | "partial_wrong" | "partiallywrong" | "half_wrong" => {
                Some(Self::PartiallyWrong)
            }
            "neutral" | "none" | "ungraded" | "auto" => Some(Self::Neutral),
            "partially_correct" | "partial_correct" | "partialcorrect" | "partial"
            | "half_correct" => Some(Self::PartiallyCorrect),
            "correct" | "positive" | "good" => Some(Self::Correct),
            _ => None,
        };
        // The numeric fallback reads the *unreplaced* text: the `-` → `_`
        // normalization above is for names like "partially-wrong" and would
        // eat the sign of "-1".
        named.or_else(|| trimmed.parse::<f64>().ok().map(Self::from_points))
    }

    /// A short, stable name for reports, manifests and rule files.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wrong => "wrong",
            Self::PartiallyWrong => "partially_wrong",
            Self::Neutral => "neutral",
            Self::PartiallyCorrect => "partially_correct",
            Self::Correct => "correct",
        }
    }

    /// Whether this grade charges an unlikelihood penalty.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        matches!(self, Self::Wrong | Self::PartiallyWrong)
    }

    /// Whether no opinion was recorded for this token.
    #[must_use]
    pub const fn is_ungraded(self) -> bool {
        matches!(self, Self::Neutral)
    }

    /// The value this grade writes into the loss-mask channel.
    ///
    /// This is *not* "how much this token is learned" — the weight channel
    /// already decides that. The mask says "this position is supervised", and
    /// [`crate::lm`] derives the charge as `(weight > 0) * mask`: a masked-off
    /// position cannot be charged at all. So a negative grade must keep the
    /// mask at full, or its penalty would silently vanish — the bug this
    /// comment exists to prevent.
    ///
    /// `Neutral`, `Correct` and the negative grades are all `1.0`;
    /// `PartiallyCorrect` is `0.5`, and because the denominator of the loss is
    /// the sum of the mask, that halves the token's share of the batch mean —
    /// a weighted mean, not a shrunken one.
    #[must_use]
    pub const fn supervision_scale(self) -> f32 {
        match self {
            Self::PartiallyCorrect => 0.5,
            Self::Wrong | Self::PartiallyWrong | Self::Neutral | Self::Correct => 1.0,
        }
    }

    /// The weight this grade puts on the unlikelihood charge. Zero for every
    /// non-negative grade, so an all-neutral corpus charges nothing.
    #[must_use]
    pub const fn charge_weight(self) -> f32 {
        match self {
            Self::Wrong => 1.0,
            Self::PartiallyWrong => 0.5,
            Self::Neutral | Self::PartiallyCorrect | Self::Correct => 0.0,
        }
    }

    /// Every level, worst to best — the order reports iterate in.
    pub const ALL: [Self; 5] = [
        Self::Wrong,
        Self::PartiallyWrong,
        Self::Neutral,
        Self::PartiallyCorrect,
        Self::Correct,
    ];
}

/// One grade per corpus token, in corpus order.
///
/// Mirrors the label plane: it is addressed by *target token index*, so
/// position `i` grades the token the model is being asked to predict from
/// window position `i - 1`. The same alignment the label and mask sidecars
/// use, which is what lets all three be sliced from one window read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradePlane {
    values: Vec<i8>,
}

impl GradePlane {
    /// Wrap decoded values, refusing any byte outside `−2..=2`.
    ///
    /// # Errors
    ///
    /// If any value is not a valid quantized grade.
    pub fn new(values: Vec<i8>) -> Result<Self> {
        if let Some(bad) = values
            .iter()
            .copied()
            .find(|v| Grade::from_quantized(*v).is_none())
        {
            return Err(anyhow!(
                "grade value {bad} is outside the valid range -2..=2"
            ));
        }
        Ok(Self { values })
    }

    /// A plane of `len` ungraded tokens — what a corpus with no opinions has.
    #[must_use]
    pub fn ungraded(len: usize) -> Self {
        Self {
            values: vec![Grade::Neutral.quantized(); len],
        }
    }

    /// Build from decoded grades.
    #[must_use]
    pub fn from_grades(grades: &[Grade]) -> Self {
        Self {
            values: grades.iter().map(|g| g.quantized()).collect(),
        }
    }

    /// Number of tokens graded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the plane covers no tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The grade of one token index.
    ///
    /// # Panics
    ///
    /// If `index` is out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Grade {
        self.get_or_neutral(index)
    }

    /// The grade of one token index, or Neutral past the end of the plane.
    ///
    /// Padding past a window's grades must read as *ungraded*, never panic:
    /// the caller is asking about a position the plane never claimed to cover.
    #[must_use]
    pub fn get_or_neutral(&self, index: usize) -> Grade {
        match self.values.get(index) {
            Some(value) => Grade::from_quantized(*value).unwrap_or(Grade::Neutral),
            None => Grade::Neutral,
        }
    }

    /// Set the grade of one token index.
    ///
    /// # Errors
    ///
    /// If `index` is out of range.
    pub fn set(&mut self, index: usize, grade: Grade) -> Result<()> {
        let len = self.values.len();
        let slot = self
            .values
            .get_mut(index)
            .with_context(|| format!("grade index {index} out of range for a {len}-token plane"))?;
        *slot = grade.quantized();
        Ok(())
    }

    /// Grade every token in `range`. Bounds are clamped to the plane, so an
    /// empty or reversed range grades nothing.
    pub fn fill(&mut self, range: std::ops::Range<usize>, grade: Grade) {
        let start = range.start.min(self.values.len());
        let end = range.end.min(self.values.len());
        for slot in &mut self.values[start..end] {
            *slot = grade.quantized();
        }
    }

    /// The raw quantized values.
    #[must_use]
    pub fn as_i8(&self) -> &[i8] {
        &self.values
    }

    /// Decode the plane from sidecar bytes.
    ///
    /// # Errors
    ///
    /// If any byte is outside `−2..=2`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::new(bytes.iter().map(|byte| *byte as i8).collect())
    }

    /// Mean point value over the whole plane: the one number that says whether
    /// a corpus mostly teaches, mostly warns, or says nothing.
    #[must_use]
    pub fn mean_points(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let total: i64 = self.values.iter().map(|value| i64::from(*value)).sum();
        let denominator =
            2i64.saturating_mul(i64::try_from(self.values.len()).unwrap_or(i64::MAX / 4));
        // Both sides are exact in f64 for any corpus that fits on disk, and the
        // quotient is a multiple of 0.5 / len — no drift to worry about.
        total as f64 / denominator as f64
    }

    /// How many tokens earned each grade, worst to best.
    #[must_use]
    pub fn counts(&self) -> Vec<(Grade, usize)> {
        Grade::ALL
            .iter()
            .map(|grade| {
                (
                    *grade,
                    self.values
                        .iter()
                        .filter(|value| Grade::from_quantized(**value) == Some(*grade))
                        .count(),
                )
            })
            .collect()
    }

    /// The sidecar bytes: the quantized values reinterpreted as unsigned,
    /// which is exactly the two's-complement encoding the file stores.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: every `i8` pattern is a valid `u8`, and both are one byte
        // with alignment 1, so this reinterprets without moving anything.
        unsafe { std::slice::from_raw_parts(self.values.as_ptr().cast::<u8>(), self.values.len()) }
    }

    /// Overwrite `path` with the sidecar encoding.
    ///
    /// # Errors
    ///
    /// If the file cannot be written.
    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.as_bytes())
            .with_context(|| format!("write grades to {}", path.display()))
    }

    /// Read a whole sidecar.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, or holds a byte outside `−2..=2`.
    pub fn read(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("read grades from {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("decoding {}", path.display()))
    }

    /// Read `len` grades starting at token `start`, seeking exactly once.
    ///
    /// Lets a streaming corpus take each sampled window's grades the same way
    /// it takes its tokens, without materializing the sidecar.
    ///
    /// # Errors
    ///
    /// If the seek or read fails, the file is shorter than asked, or a byte is
    /// outside `−2..=2`.
    pub fn read_at(path: &Path, start: usize, len: usize) -> Result<Self> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file =
            File::open(path).with_context(|| format!("open grades {}", path.display()))?;
        let offset = u64::try_from(start * GRADE_BYTES).unwrap_or(u64::MAX);
        file.seek(SeekFrom::Start(offset))
            .with_context(|| format!("seek to token {start} in {}", path.display()))?;
        let mut buf = vec![0u8; len * GRADE_BYTES];
        file.read_exact(&mut buf).with_context(|| {
            format!(
                "read {} grades from {} at token {start}",
                len,
                path.display()
            )
        })?;
        Self::from_bytes(&buf).with_context(|| format!("decoding {}", path.display()))
    }
}

/// The `.grades` sidecar path for a corpus.
#[must_use]
pub fn grades_path(corpus: &Path) -> PathBuf {
    let mut path = corpus.as_os_str().to_os_string();
    path.push(".grades");
    PathBuf::from(path)
}

/// The `.grades.json` manifest path for a corpus.
#[must_use]
pub fn manifest_path(corpus: &Path) -> PathBuf {
    let mut path = corpus.as_os_str().to_os_string();
    path.push(".grades.json");
    PathBuf::from(path)
}

/// The current sidecar format version. Reserved so a future widening (say
/// per-token confidence alongside the point value) can be refused rather than
/// silently mis-decoded.
pub const GRADES_FORMAT_VERSION: u32 = 1;

/// What a grade sidecar says about itself, written beside it.
///
/// The sidecar bytes carry no header, so this is where the *provenance* lives:
/// who graded the corpus, from what, and how the grades came out. A run that
/// trained on a graded corpus is only reproducible if the grading recipe is
/// recorded somewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradeManifest {
    pub format_version: u32,
    /// One per grade, worst to best, in [`Grade::ALL`] order.
    pub counts: Vec<GradeCount>,
    /// `points × 2` summed over the corpus divided by twice the token count.
    pub mean_points: f64,
    /// How the grades were produced, e.g. `codequality(score>=0.9 -> +1)`.
    pub graded_by: String,
    /// The corpus this was built from, so a mismatch is caught at train time.
    pub corpus_tokens: usize,
}

/// One grade's tally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradeCount {
    pub grade: Grade,
    pub tokens: usize,
}

impl GradeManifest {
    /// Summarize a plane, recording what produced it.
    #[must_use]
    pub fn for_plane(plane: &GradePlane, graded_by: &str, corpus_tokens: usize) -> Self {
        Self {
            format_version: GRADES_FORMAT_VERSION,
            counts: plane
                .counts()
                .into_iter()
                .map(|(grade, tokens)| GradeCount { grade, tokens })
                .collect(),
            mean_points: plane.mean_points(),
            graded_by: graded_by.to_string(),
            corpus_tokens,
        }
    }

    /// Tokens that earned a non-zero, non-neutral grade — the ones that change
    /// the objective at all.
    #[must_use]
    pub fn graded_tokens(&self) -> usize {
        self.counts
            .iter()
            .filter(|c| c.grade != Grade::Neutral)
            .map(|c| c.tokens)
            .sum()
    }

    /// Write the manifest as JSON.
    ///
    /// # Errors
    ///
    /// If serialization or the write fails.
    pub fn write(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).context("serialize grade manifest")?;
        std::fs::write(path, text)
            .with_context(|| format!("write grade manifest to {}", path.display()))
    }

    /// Read a manifest.
    ///
    /// # Errors
    ///
    /// If the file cannot be read or parsed.
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read grade manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse grade manifest {}", path.display()))?;
        anyhow::ensure!(
            manifest.format_version == GRADES_FORMAT_VERSION,
            "{} is grade format {} but this build reads {GRADES_FORMAT_VERSION}",
            path.display(),
            manifest.format_version
        );
        Ok(manifest)
    }

    /// A one-line summary for training logs.
    #[must_use]
    pub fn summary(&self) -> String {
        let parts: Vec<String> = self
            .counts
            .iter()
            .filter(|c| c.tokens > 0)
            .map(|c| format!("{}:{:+}×{}", c.grade.name(), c.grade.points(), c.tokens))
            .collect();
        format!(
            "{} tokens graded (mean {:+.3}) [{}]",
            self.corpus_tokens,
            self.mean_points,
            parts.join(" ")
        )
    }
}

/// The grade a token earns *because a rule fired on it*: a heavy rule is a
/// wrong token, a light one a partially-wrong token, and no rule at all is not
/// an opinion — an ungraded token trains as an ordinary supervised target.
#[must_use]
pub fn grade_from_rule_weight(weight: f32) -> Grade {
    if weight <= 0.0 {
        Grade::Neutral
    } else if weight < 0.5 {
        Grade::PartiallyWrong
    } else {
        Grade::Wrong
    }
}

impl GradePlane {
    /// Grade a label plane: every token a rule flagged with weight `w ≥ 0.5`
    /// is `−1`, one flagged more lightly is `−0.5`, and everything else is
    /// ungraded. `table` is the corpus's own label → weight table, so the
    /// thresholds match the numbers the penalty path already uses.
    #[must_use]
    pub fn from_label_ids(label_ids: &[u8], table: &[f32; 256]) -> Self {
        Self::from_grades(
            &label_ids
                .iter()
                .map(|id| grade_from_rule_weight(table[usize::from(*id)]))
                .collect::<Vec<_>>(),
        )
    }

    /// A plane where a contiguous span of `len` tokens starting at `start`
    /// earned `grade` and nothing else was graded — the shape of
    /// "one document, one verdict".
    #[must_use]
    pub fn one_document(len: usize, start: usize, span: usize, grade: Grade) -> Self {
        let mut plane = Self::ungraded(len);
        plane.fill(start..start.saturating_add(span), grade);
        plane
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Wrong => "-1",
            Self::PartiallyWrong => "-0.5",
            Self::Neutral => "0",
            Self::PartiallyCorrect => "+0.5",
            Self::Correct => "+1",
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

    fn scratch_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("dblocks-grade-tests");
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn sidecar_path(name: &str) -> PathBuf {
        let path = scratch_dir().join(format!("{name}.grades"));
        let _ = std::fs::remove_file(&path);
        path
    }

    /// Slice helper: read a window of a sidecar into a comparable plane.
    fn read_at(path: &Path, start: usize, len: usize) -> GradePlane {
        GradePlane::read_at(path, start, len).expect("windowed sidecar read")
    }

    #[test]
    fn test_the_five_levels_round_trip_through_one_byte() {
        // The sidecar is one byte per token. If any two levels shared a byte,
        // or a level had no byte, grading would silently lose information on
        // the way to disk — so every level must survive quantization exactly.
        for grade in Grade::ALL {
            let decoded = Grade::from_quantized(grade.quantized()).expect("every level has a byte");
            assert_eq!(decoded, grade, "{} quantized wrong", grade.name());
            assert_eq!(decoded.points(), grade.points());
        }
        // And the bytes must be distinct and contiguous, or the format wastes
        // space it will need later.
        let bytes: Vec<i8> = Grade::ALL.iter().map(|g| g.quantized()).collect();
        assert_eq!(bytes, vec![-2, -1, 0, 1, 2]);
        assert_eq!(Grade::from_quantized(3), None);
        assert_eq!(Grade::from_quantized(-3), None);
    }

    #[test]
    fn test_a_better_score_never_earns_fewer_points() {
        // Monotonicity is the whole reason a continuous score can be trusted
        // to produce a point value: sweeping the score upward must never walk
        // the grade down.
        let mut previous = f64::NEG_INFINITY;
        for step in 0..=1000 {
            let score = step as f32 / 1000.0;
            let points = Grade::from_score(score).points();
            assert!(
                points >= previous,
                "score {score} earned {points} after {previous}"
            );
            previous = points;
        }
        // The thresholds themselves, at the boundary, inclusive as documented.
        assert_eq!(Grade::from_score(0.90), Grade::Correct);
        assert_eq!(Grade::from_score(0.899), Grade::PartiallyCorrect);
        assert_eq!(Grade::from_score(0.70), Grade::PartiallyCorrect);
        assert_eq!(Grade::from_score(0.40), Grade::Neutral);
        assert_eq!(Grade::from_score(0.20), Grade::PartiallyWrong);
        assert_eq!(Grade::from_score(0.199), Grade::Wrong);
        assert_eq!(Grade::from_score(0.0), Grade::Wrong);
    }

    #[test]
    fn test_grades_parse_what_a_human_types() {
        for (text, expected) in [
            ("wrong", Grade::Wrong),
            ("Wrong", Grade::Wrong),
            ("-1", Grade::Wrong),
            ("-0.5", Grade::PartiallyWrong),
            ("partially_wrong", Grade::PartiallyWrong),
            ("partially-wrong", Grade::PartiallyWrong),
            ("0", Grade::Neutral),
            ("auto", Grade::Neutral),
            ("partial", Grade::PartiallyCorrect),
            ("+0.5", Grade::PartiallyCorrect),
            ("1", Grade::Correct),
            ("correct", Grade::Correct),
        ] {
            assert_eq!(Grade::parse(text), Some(expected), "{text} parsed wrong");
        }
        assert_eq!(Grade::parse("sometimes"), None);
        assert_eq!(Grade::parse(""), None);
    }

    #[test]
    fn test_a_plane_rejects_a_byte_that_is_not_a_grade() {
        assert!(GradePlane::new(vec![0, 1, 2, -1, -2]).is_ok());
        let err = GradePlane::new(vec![0, 7]).unwrap_err().to_string();
        assert!(err.contains('7'), "error did not name the bad byte: {err}");
        // Bytes arrive from disk as unsigned, so the same refusal must apply.
        assert!(
            GradePlane::from_bytes(&[0u8, 250]).is_err(),
            "250 as i8 is -6"
        );
    }

    #[test]
    fn test_fills_clamp_and_missing_indices_read_as_ungraded() {
        let mut plane = GradePlane::ungraded(10);
        plane.fill(2..5, Grade::Correct);
        plane.fill(8..100, Grade::Wrong);
        assert_eq!(plane.get(1), Grade::Neutral);
        assert_eq!(plane.get(2), Grade::Correct);
        assert_eq!(plane.get(4), Grade::Correct);
        assert_eq!(plane.get(9), Grade::Wrong);
        // Past the end is "no opinion", not a panic: a caller asking about a
        // position the plane never covered must get the honest answer.
        assert_eq!(plane.get_or_neutral(10), Grade::Neutral);
        assert!(plane.set(10, Grade::Correct).is_err());
        plane.set(0, Grade::PartiallyWrong).expect("in range");
        assert_eq!(plane.get(0), Grade::PartiallyWrong);
    }

    #[test]
    fn test_a_sidecar_round_trips_and_slices_like_the_original() {
        // Training reads one window at a time from a sidecar that may be
        // gigabytes. If a windowed read disagreed with the whole-file read, the
        // run would be graded by a different corpus than the one reported.
        let grades: Vec<Grade> = (0..64).map(|i| Grade::ALL[i % 5]).collect();
        let plane = GradePlane::from_grades(&grades);
        let path = sidecar_path("roundtrip");
        plane.write(&path).expect("write sidecar");

        let read = GradePlane::read(&path).expect("read sidecar");
        assert_eq!(read, plane);

        let window = read_at(&path, 17, 23);
        let expected = GradePlane::from_grades(
            &plane.as_i8()[17..40]
                .iter()
                .map(|value| Grade::from_quantized(*value).expect("written by this test"))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            window, expected,
            "a windowed read disagreed with the whole read"
        );

        // A short file must be an error, not a silent zero-fill.
        let short = sidecar_path("short");
        std::fs::write(&short, [0u8; 4]).expect("write short sidecar");
        assert!(GradePlane::read_at(&short, 0, 8).is_err());

        // The sidecar path hangs off the corpus path the way labels and masks do.
        assert_eq!(
            grades_path(Path::new("a.bin")),
            PathBuf::from("a.bin.grades")
        );
        assert_eq!(
            manifest_path(Path::new("a.bin")),
            PathBuf::from("a.bin.grades.json")
        );
    }

    #[test]
    fn test_mean_points_counts_tokens_not_windows() {
        let plane = GradePlane::from_grades(&[
            Grade::Wrong,
            Grade::Correct,
            Grade::Neutral,
            Grade::Neutral,
        ]);
        assert!(
            (plane.mean_points() - 0.0).abs() < 1e-12,
            "worst and best should cancel"
        );
        let plane =
            GradePlane::from_grades(&[Grade::Wrong, Grade::Wrong, Grade::Neutral, Grade::Neutral]);
        assert!((plane.mean_points() - -0.5).abs() < 1e-12);
        assert_eq!(GradePlane::ungraded(0).mean_points(), 0.0);
    }

    #[test]
    fn test_a_charged_token_is_never_masked_off() {
        // The bug this test exists to catch: the loss derives its charge as
        // `(weight > 0) * mask`, so a negative grade that masked its own token
        // off would charge nothing while *also* refusing to learn it — the
        // token would simply vanish from the objective.
        for grade in Grade::ALL {
            if grade.is_negative() {
                assert!(
                    grade.charge_weight() > 0.0,
                    "{} should charge",
                    grade.name()
                );
                assert_eq!(
                    grade.supervision_scale(),
                    1.0,
                    "{} must stay supervised to be charged",
                    grade.name()
                );
            } else {
                assert_eq!(
                    grade.charge_weight(),
                    0.0,
                    "{} should not charge",
                    grade.name()
                );
            }
            // Charging and learning are mutually exclusive per token.
            assert!(!(grade.is_negative() && grade.supervision_scale() < 1.0));
        }
        // Half credit is the only thing the mask channel is used for.
        assert_eq!(Grade::PartiallyCorrect.supervision_scale(), 0.5);
        assert_eq!(Grade::Correct.supervision_scale(), 1.0);
        assert_eq!(Grade::Neutral.supervision_scale(), 1.0);
    }

    #[test]
    fn test_rule_severity_becomes_a_grade_without_inventing_opinions() {
        assert_eq!(grade_from_rule_weight(0.0), Grade::Neutral);
        assert_eq!(grade_from_rule_weight(0.3), Grade::PartiallyWrong);
        assert_eq!(grade_from_rule_weight(0.5), Grade::Wrong);
        assert_eq!(grade_from_rule_weight(1.0), Grade::Wrong);

        let mut table = [0.0f32; 256];
        table[7] = 0.9;
        table[9] = 0.25;
        let plane = GradePlane::from_label_ids(&[0, 7, 0, 9], &table);
        assert_eq!(
            plane.counts(),
            vec![
                (Grade::Wrong, 1),
                (Grade::PartiallyWrong, 1),
                (Grade::Neutral, 2),
                (Grade::PartiallyCorrect, 0),
                (Grade::Correct, 0),
            ]
        );
    }

    #[test]
    fn test_a_document_grade_only_grades_that_document() {
        let plane = GradePlane::one_document(10, 3, 4, Grade::PartiallyCorrect);
        assert_eq!(plane.get(2), Grade::Neutral);
        assert_eq!(plane.get(3), Grade::PartiallyCorrect);
        assert_eq!(plane.get(6), Grade::PartiallyCorrect);
        assert_eq!(plane.get(7), Grade::Neutral);
        // A document that runs off the end of the plane grades what exists.
        let plane = GradePlane::one_document(5, 4, 100, Grade::Correct);
        assert_eq!(plane.get(4), Grade::Correct);
        assert_eq!(plane.len(), 5);
    }

    #[test]
    fn test_the_manifest_records_who_graded_what() {
        let plane = GradePlane::from_grades(&[
            Grade::Wrong,
            Grade::Neutral,
            Grade::Correct,
            Grade::Correct,
        ]);
        let manifest = GradeManifest::for_plane(&plane, "codequality(score>=0.9 -> +1)", 4);
        assert_eq!(
            manifest.graded_tokens(),
            3,
            "neutral is the absence of a grade"
        );
        assert_eq!(manifest.corpus_tokens, 4);

        let path = scratch_dir().join("manifest.grades.json");
        manifest.write(&path).expect("write manifest");
        let read = GradeManifest::read(&path).expect("read manifest");
        assert_eq!(read, manifest);

        let summary = read.summary();
        assert!(
            summary.contains("4 tokens graded"),
            "summary lost the token count: {summary}"
        );
        assert!(
            summary.contains("wrong:-1"),
            "summary lost the worst grade: {summary}"
        );

        // A format this build cannot decode must be refused, not guessed at.
        let foreign = r#"{"format_version":99,"counts":[],"mean_points":0.0,"graded_by":"x","corpus_tokens":0}"#;
        std::fs::write(&path, foreign).expect("write foreign manifest");
        let err = GradeManifest::read(&path).unwrap_err().to_string();
        assert!(
            err.contains("format 99"),
            "refusal did not name the version: {err}"
        );
    }
}
