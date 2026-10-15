//! Outcome-grounded adaptation for the reward-integrity scanner, in the shape
//! Harness-R1 gives an agent harness (arXiv:2608.02276).
//!
//! # The idea, and where it came from
//!
//! Harness-R1's observation is that a static detector is a fixed editor: it
//! proposes the same edit no matter what the last one got wrong. Its fix is to
//! separate three roles —
//!
//! * a **frozen target** that is never edited,
//! * an **engineer** that reads a batch of the target's *failures* and writes a
//!   patch, and
//! * an **outcome reward**: the patch is installed, the target **reruns the
//!   exact same items**, and the engineer is rewarded for the measured
//!   difference, not for sounding plausible.
//!
//! The load-bearing part is the third bullet. Every edit is scored by a real
//! rerun, so a patch that does not help gets zero and a patch that *regresses*
//! gets negative. There is no judge in the loop.
//!
//! # The mapping onto this crate
//!
//! | Harness-R1 | Here |
//! |---|---|
//! | frozen target agent | [`crate::cheat::scan_snippet_with`] — the pattern scanner, byte-for-byte unchanged |
//! | editable harness | [`HarnessProfile`] — the *operating point*: mixing thresholds and which classes are live |
//! | failure trajectories | [`FailurePacket`] — labeled cheats the scanner let through, and clean traces it wrongly dropped |
//! | harness engineer | [`propose`] — a deterministic rule-search proposer producing a group of `K` candidates |
//! | same-batch rerun | [`measure`] — rescans the identical corpus under baseline and candidate |
//! | outcome reward | [`Reward`] — caught-cheats gained minus false-drops incurred |
//! | regression protection | [`Acceptance`] — a candidate that drops a previously-kept clean trace is refused |
//! | versioned, reversible harness | [`Registry`] — every accepted profile, its reward, and its predecessor |
//!
//! # The invariant that makes this safe to ship
//!
//! [`crate::cheat::calibrate_mixing`] states a rule this module is built to
//! preserve: calibration **can only ever subtract**. Its rustdoc gives the
//! reason — "a self-modifying detector that could also loosen itself would be a
//! detector nobody should ship."
//!
//! A loop that *edits its own detector* is exactly where that property is most
//! likely to be lost, so it is enforced structurally rather than by convention:
//! [`accept`] refuses any candidate that is not
//! **monotone-tightening**. A candidate may only raise a mixing threshold, add
//! a class, or raise a class's severity. It may never lower a threshold,
//! disable a class, or lower a severity — and it may never admit a
//! `(declared, scored)` pair that [`crate::cheat::calibrate_mixing`] has not
//! already struck on measured evidence.
//!
//! The consequence is worth stating plainly: **this loop can make the scanner
//! stricter and it cannot make it laxer.** If the evidence says a rule is
//! costing more clean traces than it catches, this module cannot act on it —
//! that remains [`crate::cheat::calibrate_mixing`]'s job, on its own tally, with
//! its own floor. Splitting the two directions is deliberate: the relaxing
//! channel is human-auditable and batch-driven, and the tightening channel is
//! machine-driven and continuously measured.
//!
//! # Why the reward is not a single number
//!
//! A mean over a corpus hides a patch that catches ten cheats while dropping
//! nine clean traces. [`Reward`] reports caught-gained and false-drops-gained
//! as separate fields and [`Acceptance`] checks both, so that trade is visible
//! in the artifact rather than averaged into a scalar that reads as progress.
//! This is the same reason [`crate::cheat::calibrate_mixing`] requires
//! `MIN_CALIBRATION_DOCUMENTS` before it will say anything: a rate computed
//! from a handful of documents is a number that changes on the next run.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::cheat::{scan_snippet_with, CheatClass, MixingPolicy, Severity, SnippetLanguage};

/// How many candidates one proposal round emits. Harness-R1 samples `K = 8`
/// patches per failure packet and compares them as a group; the group is what
/// makes "better than the alternatives" a checkable statement rather than an
/// assertion about one draw.
pub const DEFAULT_CANDIDATES: usize = 8;

/// Observations below this are refused rather than adapted from, matching
/// [`crate::cheat::MIN_CALIBRATION_DOCUMENTS`]: a rate from a handful of
/// documents is not a rate.
pub const MIN_EVIDENCE_DOCUMENTS: usize = 32;

/// Per-class operating point: the severity this profile assigns, or `None` to
/// leave the scanner's own stated severity in force.
///
/// `None` is the unedited default and the only value that carries no authority,
/// so a profile that never names a class is byte-identical in behaviour to the
/// uncalibrated scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassGate {
    /// The severity to impose, overriding the scanner's own.
    pub severity: Option<Severity>,
    /// Whether this class is live at all. `false` is a *loosening* edit and is
    /// refused by [`accept`]; it exists so a profile can state
    /// the intent and be rejected on the record rather than being unrepresentable.
    pub enabled: bool,
}

impl Default for ClassGate {
    fn default() -> Self {
        Self {
            severity: None,
            enabled: true,
        }
    }
}

impl ClassGate {
    /// The severity this gate reports, given the severity the scanner found.
    pub fn effective(&self, scanner: Severity) -> Severity {
        if !self.enabled {
            return Severity::Note;
        }
        self.severity.unwrap_or(scanner)
    }
}

/// The editable object: the scanner's operating point, with the scanner itself
/// frozen.
///
/// This is deliberately a *profile* and not a rule set. The detection logic
/// stays where it is, in [`crate::cheat`]; what a run may move is where each
/// rule cuts. That is the Harness-R1 split — the runtime is fixed, the
/// configuration around it is learned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessProfile {
    /// Schema version, so a profile written by an older build is refused
    /// rather than half-read.
    pub version: u32,
    /// The language-mixing operating point, including any pairs
    /// [`crate::cheat::calibrate_mixing`] struck.
    pub mixing: MixingPolicy,
    /// Per-class overrides. A class absent from this map is unedited.
    pub gates: BTreeMap<String, ClassGate>,
    /// Where this profile came from and what it was measured on.
    pub provenance: String,
}

/// Current schema version for [`HarnessProfile`].
pub const PROFILE_VERSION: u32 = 1;

impl Default for HarnessProfile {
    fn default() -> Self {
        Self {
            version: PROFILE_VERSION,
            mixing: MixingPolicy::default(),
            gates: BTreeMap::new(),
            provenance: "unedited defaults: no class gates, uncalibrated mixing policy".to_string(),
        }
    }
}

/// A class whose key [`HarnessProfile::gates`] does not name.
///
/// The gate map is keyed by string because it is serialized, so a typo would
/// otherwise become a silently dead entry — the same failure
/// [`crate::cheat::MixingPolicy::load`] guards its pair set against.
fn class_key(class: CheatClass) -> &'static str {
    serde_json::to_value(class)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unserializable".to_string())
        .leak()
}

/// The key [`HarnessProfile::gates`] uses for `class`.
///
/// Public so a caller building a profile by hand, or a certificate enumerating
/// the class space, names a class exactly the way this module does.
pub fn class_key_for(class: CheatClass) -> &'static str {
    class_key(class)
}

/// Every [`CheatClass`], in declaration order, so the sweep is exhaustive
/// without depending on `serde`'s representation round-tripping.
const ALL_CLASSES: [CheatClass; 13] = [
    CheatClass::LintSuppression,
    CheatClass::TestEvasion,
    CheatClass::GateTampering,
    CheatClass::DataSuppression,
    CheatClass::DataStub,
    CheatClass::DataEvalGaming,
    CheatClass::DataBroken,
    CheatClass::DataSecrets,
    CheatClass::DataEncodingEvasion,
    CheatClass::DataDynamicEvasion,
    CheatClass::DataBehavioral,
    CheatClass::DataLanguageMixing,
    CheatClass::GateTamper,
];

impl HarnessProfile {
    /// The gate in force for `class`, or the default when unedited.
    pub fn gate(&self, class: CheatClass) -> ClassGate {
        self.gates
            .get(class_key(class))
            .copied()
            .unwrap_or_default()
    }

    /// Rewrite `report`'s severities under this profile.
    ///
    /// Elevating a severity is what makes a previously-recorded finding
    /// reject a trace, so this is where "tighten" actually takes effect.
    /// The scanner's classes are untouched; only severities move.
    pub fn apply(&self, report: &mut crate::cheat::CheatReport) {
        for finding in &mut report.findings {
            finding.severity = self.gate(finding.class).effective(finding.severity);
        }
    }

    /// Whether this profile is *no weaker* than `other` on every axis.
    ///
    /// This is the safety invariant. A candidate that lowers a mixing
    /// threshold, disables a class, lowers a severity, or admits a pair the
    /// baseline did not admit is rejected here, before any outcome is measured
    /// — a loosening edit is refused on the record even if it would have scored
    /// well, because a profile that improved the metric by becoming permissive is
    /// precisely the failure this crate's calibration rule exists to prevent.
    pub fn is_monotone_over(&self, other: &Self) -> bool {
        if self.mixing.min_score < other.mixing.min_score
            || self.mixing.min_margin < other.mixing.min_margin
        {
            return false;
        }
        // Admitting a pair the baseline struck is a loosening edit, whatever
        // else changed. Sets must be a subset.
        if !self
            .mixing
            .admitted_pairs
            .is_subset(&other.mixing.admitted_pairs)
        {
            return false;
        }
        for class in ALL_CLASSES {
            let mine = self.gate(class);
            let theirs = other.gate(class);
            if !mine.enabled && theirs.enabled {
                return false;
            }
            if let Some(s) = mine.severity {
                let floor = theirs.effective(Severity::Note);
                if s < floor {
                    return false;
                }
            }
        }
        true
    }

    /// A one-line statement of what changed, for the registry.
    pub fn diff_summary(&self, other: &Self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.mixing.min_score != other.mixing.min_score {
            parts.push(format!(
                "mix.min_score {}->{}",
                other.mixing.min_score, self.mixing.min_score
            ));
        }
        if self.mixing.min_margin != other.mixing.min_margin {
            parts.push(format!(
                "mix.min_margin {}->{}",
                other.mixing.min_margin, self.mixing.min_margin
            ));
        }
        for class in ALL_CLASSES {
            let mine = self.gate(class);
            let theirs = other.gate(class);
            if mine == theirs {
                continue;
            }
            let key = class_key(class);
            if !mine.enabled {
                parts.push(format!("{key}: disabled"));
            } else if mine.severity.is_some() && mine.severity != theirs.severity {
                parts.push(format!("{key}: severity ->{:?}", mine.severity));
            } else if theirs.severity.is_some() {
                parts.push(format!("{key}: severity gate removed"));
            }
        }
        if parts.is_empty() {
            "no change".to_string()
        } else {
            parts.join(", ")
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).context("serialize harness profile")?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
    }

    /// Refuse a profile that names a class this build does not have.
    ///
    /// The gate map is keyed by string because it is serialized, so a typo
    /// deserializes cleanly into an entry that [`Self::gate`] will never
    /// match — a suppression that reads as active and does nothing. This is the
    /// same hazard [`crate::cheat::MixingPolicy::load`] guards its pair set
    /// against, and the same reason: a profile that half-applies is worse than
    /// one that refuses to load.
    pub fn validate(&self) -> Result<()> {
        let known: BTreeSet<String> = ALL_CLASSES
            .iter()
            .map(|c| class_key(*c).to_string())
            .collect();
        for key in self.gates.keys() {
            if !known.contains(key) {
                bail!(
                    "harness profile names class `{key}`, which is not a CheatClass in this build; \
                     a gate that never matches is a suppression that reads as active and does \
                     nothing. Known classes: {}",
                    known.iter().cloned().collect::<Vec<_>>().join(", ")
                );
            }
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let p: Self =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        if p.version != PROFILE_VERSION {
            bail!(
                "{}: harness profile schema version {} but this build speaks {PROFILE_VERSION}",
                path.display(),
                p.version
            );
        }
        p.validate()?;
        Ok(p)
    }
}

/// One labeled item: the snippet and whether it is a cheat.
#[derive(Debug, Clone)]
pub struct Labeled {
    pub code: String,
    pub lang: SnippetLanguage,
    /// `true` when this snippet is known to be a cheat and must be caught.
    pub cheat: bool,
}

/// One line of the labeled corpus on disk.
///
/// A separate type from [`Labeled`] because [`SnippetLanguage`] is an enum with
/// a richer spelling than a serde derive would produce: the corpus is written by
/// hand and by other tools, so `"rust"`, `"rs"` and `"py"` all have to land on
/// the right variant rather than failing a derive.
#[derive(Debug, Clone, Deserialize)]
struct CorpusRow {
    code: String,
    lang: String,
    cheat: bool,
}

/// Load a labeled corpus from JSONL: one `{"code", "lang", "cheat"}` per line.
///
/// This is the evidence every number in this module stands on, so the loader is
/// strict in the two ways that matter: a line that does not parse names its line
/// number, and a row with no code is refused rather than scored as a trivially
/// clean snippet. An unknown language is **not** refused — it maps to
/// [`SnippetLanguage::Other`] exactly as the scanner's own parser does, so a
/// corpus row in a language the scanner has no patterns for still counts as a
/// clean trace rather than vanishing from the denominator.
pub fn load_corpus(path: &Path) -> Result<Vec<Labeled>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read corpus {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: CorpusRow = serde_json::from_str(line).with_context(|| {
            format!(
                "{}:{}: not a {{code, lang, cheat}} row",
                path.display(),
                i + 1
            )
        })?;
        if row.code.trim().is_empty() {
            bail!(
                "{}:{}: row carries no code; an empty snippet is clean by construction and would \
                 silently pad the clean side of every rate",
                path.display(),
                i + 1
            );
        }
        out.push(Labeled {
            code: row.code,
            lang: SnippetLanguage::from_name(&row.lang),
            cheat: row.cheat,
        });
    }
    if out.is_empty() {
        bail!(
            "{}: the corpus is empty; the evidence floor is {MIN_EVIDENCE_DOCUMENTS} labeled items \
             and there is nothing to measure",
            path.display()
        );
    }
    Ok(out)
}

/// The batch the engineer is allowed to look at.
///
/// Harness-R1's failure packet is "the episodes the target got wrong". Here
/// that is two sets, and both are required: cheats the scanner let through
/// (its misses) and clean traces it dropped (its false alarms). An adaptation
/// driven only by misses would raise recall by rejecting everything, which is
/// what the clean set exists to prevent.
#[derive(Debug, Clone, Default)]
pub struct FailurePacket {
    /// Labeled cheats, at least one of which the scanner did not catch.
    pub missed: Vec<Labeled>,
    /// Labeled clean traces, at least one of which the scanner wrongly dropped.
    pub false_drops: Vec<Labeled>,
}

impl FailurePacket {
    /// Build a packet by running `profile` over `corpus` and keeping the
    /// items it got wrong in each direction.
    ///
    /// The selection is the whole measurement: the same profile over the same
    /// corpus, and a packet containing exactly its errors. Changing the corpus
    /// between baseline and candidate is what Harness-R1's same-batch rerun
    /// exists to forbid, and [`measure`] re-scans `corpus` for that reason.
    pub fn mine(corpus: &[Labeled], profile: &HarnessProfile) -> Self {
        let mut packet = Self::default();
        for item in corpus {
            let mut report = scan_snippet_with(&item.code, item.lang, &profile.mixing);
            profile.apply(&mut report);
            let rejected = report.worst().is_some_and(|s| s >= Severity::Evasion);
            match (item.cheat, rejected) {
                (true, false) => packet.missed.push(item.clone()),
                (false, true) => packet.false_drops.push(item.clone()),
                _ => {}
            }
        }
        packet
    }

    /// Total labeled items the packet's selection stands on.
    pub fn evidence(&self) -> usize {
        self.missed.len() + self.false_drops.len()
    }
}

/// One count, and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tally {
    /// Labeled cheats the profile rejected.
    pub caught: usize,
    /// Labeled clean traces the profile rejected. Lower is better; zero is
    /// the only acceptable value for a shipped profile.
    pub false_drops: usize,
    /// Labeled items of each kind the measurement covers.
    pub cheats: usize,
    pub cleans: usize,
}

impl Tally {
    /// Recall over labeled cheats, or `None` when there are none.
    pub fn recall(&self) -> Option<f64> {
        (self.cheats > 0).then(|| self.caught as f64 / self.cheats as f64)
    }

    /// Fraction of labeled clean traces wrongly rejected.
    pub fn false_drop_rate(&self) -> Option<f64> {
        (self.cleans > 0).then(|| self.false_drops as f64 / self.cleans as f64)
    }
}

/// Write a labeled corpus back out, so a mined packet can become the next
/// round's evidence without a hand-written fixture.
pub fn save_corpus(path: &Path, items: &[Labeled]) -> Result<()> {
    let mut text = String::new();
    for item in items {
        let row = serde_json::json!({
            "code": item.code,
            "lang": item.lang.as_str(),
            "cheat": item.cheat,
        });
        text.push_str(&row.to_string());
        text.push('\n');
    }
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// The corpus the engineer proposed edits against, as a packet ready to be
/// replayed. Both halves travel together, so a saved packet cannot silently
/// become the misses-only corpus that rewards rejecting everything.
pub fn save_packet(path: &Path, packet: &FailurePacket) -> Result<()> {
    let mut items: Vec<Labeled> = packet.missed.clone();
    items.extend(packet.false_drops.iter().cloned());
    save_corpus(path, &items)
}

/// What one profile scored on the corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reward {
    pub baseline: Tally,
    pub candidate: Tally,
}

impl Reward {
    /// Cheats newly caught by the candidate.
    pub fn caught_gained(&self) -> i64 {
        self.candidate.caught as i64 - self.baseline.caught as i64
    }

    /// Clean traces newly dropped by the candidate. Negative is an improvement.
    pub fn false_drops_added(&self) -> i64 {
        self.candidate.false_drops as i64 - self.baseline.false_drops as i64
    }

    /// The signed improvement: cheats gained minus clean traces lost.
    ///
    /// Unsigned-by-intent: a patch that catches five and drops five scores zero,
    /// which is the correct answer and is why [`Acceptance`] also checks the
    /// two components separately.
    pub fn delta(&self) -> i64 {
        self.caught_gained() - self.false_drops_added()
    }
}

/// Scan `corpus` under `profile`.
pub fn tally(corpus: &[Labeled], profile: &HarnessProfile) -> Tally {
    let mut t = Tally {
        caught: 0,
        false_drops: 0,
        cheats: 0,
        cleans: 0,
    };
    for item in corpus {
        let mut report = scan_snippet_with(&item.code, item.lang, &profile.mixing);
        profile.apply(&mut report);
        let rejected = report.worst().is_some_and(|s| s >= Severity::Evasion);
        if item.cheat {
            t.cheats += 1;
            if rejected {
                t.caught += 1;
            }
        } else {
            t.cleans += 1;
            if rejected {
                t.false_drops += 1;
            }
        }
    }
    t
}

/// The Harness-R1 reward: rescan the **identical** corpus under baseline and
/// candidate and report the difference.
///
/// The same-batch constraint is the whole point. A reward computed on a
/// different corpus than the one the profile was chosen on measures the
/// difference between two corpora, not the difference between two profiles.
pub fn measure(
    corpus: &[Labeled],
    baseline: &HarnessProfile,
    candidate: &HarnessProfile,
) -> Reward {
    Reward {
        baseline: tally(corpus, baseline),
        candidate: tally(corpus, candidate),
    }
}

/// Why a candidate was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Refusal {
    /// The candidate loosened the profile. Refused before measurement: a
    /// detector that can relax itself is the failure this crate documents.
    Loosening,
    /// Fewer labeled items than [`MIN_EVIDENCE_DOCUMENTS`].
    TooLittleEvidence,
    /// No measurable improvement over the baseline.
    NoImprovement,
    /// Improved recall, but dropped a clean trace the baseline kept.
    Regression,
}

impl Refusal {
    pub fn as_str(self) -> &'static str {
        match self {
            Refusal::Loosening => "loosening",
            Refusal::TooLittleEvidence => "too-little-evidence",
            Refusal::NoImprovement => "no-improvement",
            Refusal::Regression => "regression",
        }
    }
}

/// Whether a candidate may replace the baseline, and why not if it may not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Acceptance {
    Accepted { delta: i64 },
    Refused { reason: Refusal },
}

impl Acceptance {
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted { .. })
    }
}

/// Decide a candidate's fate. The order of these checks is the policy:
///
/// 1. **Loosening** — refused on the record, before any number is computed.
/// 2. **Evidence** — a handful of items cannot say anything about a rate.
/// 3. **Improvement** — `delta() > 0`, with recall and false-drops as the
///    two terms rather than one opaque number.
/// 4. **Regression** — the candidate may not drop a clean trace the baseline
///    kept, even if the net delta is positive. A patch that buys recall by
///    discarding good data is the failure mode this whole module is for.
pub fn accept(
    baseline: &HarnessProfile,
    candidate: &HarnessProfile,
    reward: &Reward,
) -> Acceptance {
    if !candidate.is_monotone_over(baseline) {
        return Acceptance::Refused {
            reason: Refusal::Loosening,
        };
    }
    let evidence = reward.baseline.cheats + reward.baseline.cleans;
    if evidence < MIN_EVIDENCE_DOCUMENTS {
        return Acceptance::Refused {
            reason: Refusal::TooLittleEvidence,
        };
    }
    if reward.delta() <= 0 {
        return Acceptance::Refused {
            reason: Refusal::NoImprovement,
        };
    }
    if reward.false_drops_added() > 0 {
        return Acceptance::Refused {
            reason: Refusal::Regression,
        };
    }
    Acceptance::Accepted {
        delta: reward.delta(),
    }
}

/// One candidate from the engineer's group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub profile: HarnessProfile,
    pub reward: Reward,
    pub acceptance: Acceptance,
}

impl Candidate {
    /// A short statement of the edit and its measured outcome.
    pub fn summary(&self, baseline: &HarnessProfile) -> String {
        let verdict = match &self.acceptance {
            Acceptance::Accepted { delta } => format!("accepted delta=+{delta}"),
            Acceptance::Refused { reason } => format!("refused:{}", reason.as_str()),
        };
        let mut out = format!("{} | {verdict} | ", self.profile.diff_summary(baseline));
        out.push_str("caught ");
        out.push_str(&format!(
            "{b_caught}/{b_cheats} -> {c_caught}/{c_cheats}, false-drops {b_fp} -> {c_fp}",
            b_caught = self.reward.baseline.caught,
            b_cheats = self.reward.baseline.cheats,
            c_caught = self.reward.candidate.caught,
            c_cheats = self.reward.candidate.cheats,
            b_fp = self.reward.baseline.false_drops,
            c_fp = self.reward.candidate.false_drops,
        ));
        out
    }
}

/// The engineer: propose `count` candidate profiles and score every one by
/// rerun.
///
/// The proposer is a deterministic rule sweep, not a learned policy. Harness-R1
/// uses a 9B engineer because the editable surface is arbitrary Python; this
/// crate's editable surface is a closed set of severity and threshold moves,
/// and enumerating that set exhaustively is both cheaper and harder to fool
/// than a sampled proposal would be. The `K`-candidate group and the
/// same-batch reward are kept because they are what makes the comparison
/// checkable.
///
/// Every candidate here is monotone-tightening by construction, which is why
/// [`Refusal::Loosening`] should not fire in normal operation — it stays as the
/// guard for a caller who builds candidates by another route.
pub fn propose(
    corpus: &[Labeled],
    baseline: &HarnessProfile,
    count: usize,
) -> Result<Vec<Candidate>> {
    let count = count.max(1);
    let mut candidates: Vec<Candidate> = Vec::new();

    // Sweep 1: raise a class's severity one step at a time, in declaration
    // order, skipping classes the baseline already pins at the top grade.
    for class in ALL_CLASSES {
        if candidates.len() >= count {
            break;
        }
        let current = baseline.gate(class);
        let floor = current.effective(Severity::Note);
        let Some(next) = escalate(floor) else {
            continue;
        };
        let mut profile = baseline.clone();
        profile.gates.insert(
            class_key(class).to_string(),
            ClassGate {
                severity: Some(next),
                enabled: true,
            },
        );
        profile.provenance = format!(
            "cheatharness: escalate {} {floor:?}->{next:?}",
            class_key(class)
        );
        let reward = measure(corpus, baseline, &profile);
        let acceptance = accept(baseline, &profile, &reward);
        candidates.push(Candidate {
            profile,
            reward,
            acceptance,
        });
    }

    // Sweep 2: raise the mixing thresholds, which is the other tightening axis
    // and the one [`crate::cheat::calibrate_mixing`] shares.
    for (field, step) in [("min_score", 1u32), ("min_margin", 1u32)] {
        if candidates.len() >= count {
            break;
        }
        let mut profile = baseline.clone();
        let before = if field == "min_score" {
            profile.mixing.min_score
        } else {
            profile.mixing.min_margin
        };
        if field == "min_score" {
            profile.mixing.min_score = before.saturating_add(step);
        } else {
            profile.mixing.min_margin = before.saturating_add(step);
        }
        profile.provenance = format!("cheatharness: raise mix.{field} {before}+{step}");
        let reward = measure(corpus, baseline, &profile);
        let acceptance = accept(baseline, &profile, &reward);
        candidates.push(Candidate {
            profile,
            reward,
            acceptance,
        });
    }

    // Sweep 3: stack every escalation that individually helped. A group of one
    // is a weak comparison; stacking is where the measured deltas are additive
    // and the candidate is worth more than any single move.
    if candidates.len() < count {
        let helping: Vec<HarnessProfile> = candidates
            .iter()
            .filter(|c| c.acceptance.is_accepted())
            .map(|c| c.profile.clone())
            .collect();
        if helping.len() > 1 {
            let mut stacked = baseline.clone();
            for p in &helping {
                for (k, v) in &p.gates {
                    stacked.gates.insert(k.clone(), *v);
                }
                stacked.mixing.min_score = stacked.mixing.min_score.max(p.mixing.min_score);
                stacked.mixing.min_margin = stacked.mixing.min_margin.max(p.mixing.min_margin);
            }
            stacked.provenance =
                "cheatharness: stack of individually-measured escalations".to_string();
            let reward = measure(corpus, baseline, &stacked);
            let acceptance = accept(baseline, &stacked, &reward);
            candidates.push(Candidate {
                profile: stacked,
                reward,
                acceptance,
            });
        }
    }

    if candidates.is_empty() {
        bail!(
            "the engineer proposed no candidate: the baseline pins every class at fraud and both \
             mixing thresholds are already at their ceiling, so there is no tightening move left"
        );
    }
    Ok(candidates)
}

/// The next severity up, or `None` at the ceiling.
fn escalate(s: Severity) -> Option<Severity> {
    match s {
        Severity::Note => Some(Severity::Evasion),
        Severity::Evasion => Some(Severity::Suppression),
        Severity::Suppression => Some(Severity::Fraud),
        Severity::Fraud => None,
    }
}

/// One accepted step: the profile in force, what it changed, and what it
/// measured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// Monotonic version, starting at 1.
    pub version: u32,
    /// The profile this entry put in force.
    pub profile: HarnessProfile,
    /// What changed relative to the previous version.
    pub summary: String,
    /// The measured outcome that justified it.
    pub reward: Reward,
    /// Where the corpus came from, so a number is never floating free of its
    /// evidence.
    pub provenance: String,
}

/// The append-only chain of accepted profiles, with rollback.
///
/// Harness-R1's released harness is a versioned, reversible artifact; this is
/// the same, and the rollback is the point — the most recent accepted profile
/// can always be withdrawn without recomputing anything, because every prior
/// one is on file with the numbers that justified it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    /// The unedited scanner. Held separately so [`Registry::current`] can
    /// always name a profile even on an empty chain, and so "what did this
    /// loop never touch" is a field a reader can see rather than an invariant
    /// they have to trust.
    pub baseline: HarnessProfile,
    pub entries: Vec<Entry>,
}

impl Default for Registry {
    fn default() -> Self {
        Self::default_registry()
    }
}

impl Registry {
    pub fn default_registry() -> Self {
        let baseline = HarnessProfile::default();
        Self {
            entries: vec![Entry {
                version: 0,
                profile: baseline.clone(),
                summary: "baseline: unedited scanner".to_string(),
                reward: Reward {
                    baseline: Tally {
                        caught: 0,
                        false_drops: 0,
                        cheats: 0,
                        cleans: 0,
                    },
                    candidate: Tally {
                        caught: 0,
                        false_drops: 0,
                        cheats: 0,
                        cleans: 0,
                    },
                },
                provenance: "empty chain".to_string(),
            }],
            baseline,
        }
    }

    /// The profile currently in force.
    pub fn current(&self) -> &HarnessProfile {
        // A chain is never empty: `default_registry` seeds version 0 and
        // `accept` is the only way to append.
        self.entries
            .last()
            .map(|e| &e.profile)
            .unwrap_or(&self.baseline)
    }

    /// Append an accepted candidate. A refused one is not recorded, so the
    /// chain never contains a profile that lost its own acceptance test.
    pub fn commit(
        &mut self,
        baseline: &HarnessProfile,
        candidate: &Candidate,
        provenance: &str,
    ) -> Result<&Entry> {
        match &candidate.acceptance {
            Acceptance::Accepted { .. } => {}
            Acceptance::Refused { reason } => bail!(
                "refusing to commit a candidate the acceptance test rejected: {}",
                reason.as_str()
            ),
        }
        candidate.profile.validate()?;
        if !candidate.profile.is_monotone_over(baseline) {
            bail!(
                "refusing to commit a profile that is weaker than the baseline; the monotonicity \
                 check is re-run here so no caller can bypass it"
            );
        }
        let version = self.entries.len() as u32;
        self.entries.push(Entry {
            version,
            summary: candidate.profile.diff_summary(baseline),
            profile: candidate.profile.clone(),
            reward: candidate.reward,
            provenance: provenance.to_string(),
        });
        self.entries
            .last()
            .ok_or_else(|| anyhow!("just pushed an entry, but the chain reports empty"))
    }

    /// Withdraw the most recent accepted profile, returning it.
    ///
    /// Refuses to remove version 0: that is the unedited scanner, and a chain
    /// that cannot be restored to "no adaptation at all" is not reversible.
    pub fn rollback(&mut self) -> Result<Entry> {
        if self.entries.len() <= 1 {
            bail!(
                "rollback would remove the baseline entry; the chain always retains the unedited \
                 scanner"
            );
        }
        self.entries
            .pop()
            .ok_or_else(|| anyhow!("the chain reports non-empty but pop returned nothing"))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).context("serialize harness registry")?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let r: Self =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        Ok(r)
    }
}

/// One full round: mine, propose, measure, and report the group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Round {
    /// The packet the round was built from.
    pub missed: usize,
    pub false_drops: usize,
    /// The candidate group, most-improving first.
    pub candidates: Vec<Candidate>,
}

impl Round {
    /// The best accepted candidate, if any.
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates
            .iter()
            .filter(|c| c.acceptance.is_accepted())
            .max_by_key(|c| match c.acceptance {
                Acceptance::Accepted { delta } => delta,
                Acceptance::Refused { .. } => i64::MIN,
            })
    }
}

/// Run one round against `corpus` from the registry's current profile.
///
/// Returns the round without committing anything: acceptance is a separate,
/// inspectable step, so a caller can read the group before deciding.
pub fn run_round(corpus: &[Labeled], registry: &Registry, count: usize) -> Result<Round> {
    let baseline = registry.current().clone();
    let packet = FailurePacket::mine(corpus, &baseline);
    let mut candidates = propose(corpus, &baseline, count)?;
    // Rank by the acceptance decision first, then by measured delta, so the
    // head of the list is the one a reviewer should read.
    candidates.sort_by(|a, b| {
        b.acceptance
            .is_accepted()
            .cmp(&a.acceptance.is_accepted())
            .then(b.reward.delta().cmp(&a.reward.delta()))
            .then(b.reward.caught_gained().cmp(&a.reward.caught_gained()))
    });
    Ok(Round {
        missed: packet.missed.len(),
        false_drops: packet.false_drops.len(),
        candidates,
    })
}

/// Pairs an adaptation refused must state which direction it was in.
///
/// A strict helper for callers rendering an [`Acceptance`] into a report: the
/// loosening refusal is the one a reader needs to see flagged, because it is
/// the refusal that means "this loop was asked to do something it will not do".
pub fn is_safety_refusal(a: &Acceptance) -> bool {
    matches!(
        a,
        Acceptance::Refused {
            reason: Refusal::Loosening
        }
    )
}

/// Serialize a round for the append-only corpus log.
pub fn round_json(round: &Round) -> Result<String> {
    serde_json::to_string_pretty(round).context("serialize harness round")
}

/// Format the candidate group as a table.
pub fn render_round(round: &Round, baseline: &HarnessProfile) -> String {
    let mut out = format!(
        "packet: {} missed, {} false-dropped\n",
        round.missed, round.false_drops
    );
    for c in &round.candidates {
        out.push_str(&format!("  {}\n", c.summary(baseline)));
    }
    out
}

#[cfg(test)]
// A test says "this must have worked" with `unwrap`/`expect`, which is the
// right thing for a test to say. The grant is scoped to this module:
// production code in the same file is still denied it (see the `[lints]` table
// in `Cargo.toml` and the contract in the crate docs).
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
    clippy::exit
)]
mod tests {
    use super::*;

    fn cheat_snippet() -> Labeled {
        Labeled {
            code: "fn f() { let a = 1; #[allow(dead_code)]\n let b = 2; }".to_string(),
            lang: SnippetLanguage::Rust,
            cheat: true,
        }
    }

    fn clean_snippet() -> Labeled {
        Labeled {
            code: "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}".to_string(),
            lang: SnippetLanguage::Rust,
            cheat: false,
        }
    }

    /// A corpus with enough labeled items to clear the evidence floor.
    fn corpus() -> Vec<Labeled> {
        let mut v = Vec::new();
        for _ in 0..20 {
            v.push(clean_snippet());
        }
        for _ in 0..20 {
            v.push(Labeled {
                code: "fn g() { todo!() }".to_string(),
                lang: SnippetLanguage::Rust,
                cheat: true,
            });
        }
        v
    }

    /// A labeled cheat carrying a suppression with no written reason: the
    /// shape the learned layer in `mlcheat` exists to generalize.
    #[test]
    fn a_suppression_cheat_is_labeled_a_cheat() {
        assert!(cheat_snippet().cheat, "the fixture must carry its label");
        assert!(
            !clean_snippet().cheat,
            "the clean fixture must carry its label"
        );
    }

    #[test]
    fn default_profile_is_behaviourally_the_scanner() {
        let p = HarnessProfile::default();
        let c = corpus();
        let t = tally(&c, &p);
        // No gate is named, so every class keeps the scanner's own severity.
        assert!(t.caught > 0 || t.false_drops > 0 || t.cheats > 0);
        let named: usize = p.gates.len();
        assert_eq!(named, 0, "an unedited profile must name no class");
    }

    #[test]
    fn monotone_rejects_a_lower_threshold() {
        let base = HarnessProfile::default();
        let mut worse = base.clone();
        worse.mixing.min_score = base.mixing.min_score.saturating_sub(1);
        assert!(
            !worse.is_monotone_over(&base),
            "lowering a threshold is a loosening edit and must be refused"
        );
    }

    #[test]
    fn monotone_rejects_disabling_a_class() {
        let base = HarnessProfile::default();
        let mut worse = base.clone();
        worse.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: None,
                enabled: false,
            },
        );
        assert!(!worse.is_monotone_over(&base));
    }

    #[test]
    fn monotone_rejects_lowering_a_severity() {
        let mut base = HarnessProfile::default();
        base.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Fraud),
                enabled: true,
            },
        );
        let mut worse = base.clone();
        worse.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Note),
                enabled: true,
            },
        );
        assert!(!worse.is_monotone_over(&base));
    }

    #[test]
    fn monotone_accepts_an_escalation() {
        let base = HarnessProfile::default();
        let mut better = base.clone();
        better.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Fraud),
                enabled: true,
            },
        );
        assert!(better.is_monotone_over(&base));
    }

    #[test]
    fn a_loosening_candidate_is_refused_before_measurement() {
        let base = HarnessProfile::default();
        let mut worse = base.clone();
        worse.mixing.min_score = base.mixing.min_score.saturating_sub(1);
        // A reward that would otherwise look like a large win.
        let reward = Reward {
            baseline: Tally {
                caught: 0,
                false_drops: 0,
                cheats: 40,
                cleans: 40,
            },
            candidate: Tally {
                caught: 40,
                false_drops: 0,
                cheats: 40,
                cleans: 40,
            },
        };
        let a = accept(&base, &worse, &reward);
        assert_eq!(
            a,
            Acceptance::Refused {
                reason: Refusal::Loosening
            },
            "a profile that improves the metric by relaxing is the exact failure this refuses"
        );
        assert!(is_safety_refusal(&a));
    }

    #[test]
    fn a_regression_is_refused_even_with_positive_delta() {
        let base = HarnessProfile::default();
        let mut better = base.clone();
        better.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Fraud),
                enabled: true,
            },
        );
        // +8 caught, -4 false drops is a positive delta, but the clean count
        // went up; the candidate is refused on the clean side alone.
        let reward = Reward {
            baseline: Tally {
                caught: 2,
                false_drops: 2,
                cheats: 40,
                cleans: 40,
            },
            candidate: Tally {
                caught: 10,
                false_drops: 6,
                cheats: 40,
                cleans: 40,
            },
        };
        assert_eq!(reward.delta(), 4);
        assert_eq!(
            accept(&base, &better, &reward),
            Acceptance::Refused {
                reason: Refusal::Regression
            }
        );
    }

    #[test]
    fn too_little_evidence_is_refused() {
        let base = HarnessProfile::default();
        let mut better = base.clone();
        better.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Fraud),
                enabled: true,
            },
        );
        let reward = Reward {
            baseline: Tally {
                caught: 0,
                false_drops: 0,
                cheats: 2,
                cleans: 2,
            },
            candidate: Tally {
                caught: 2,
                false_drops: 0,
                cheats: 2,
                cleans: 2,
            },
        };
        assert_eq!(
            accept(&base, &better, &reward),
            Acceptance::Refused {
                reason: Refusal::TooLittleEvidence
            }
        );
    }

    #[test]
    fn no_improvement_is_refused() {
        let base = HarnessProfile::default();
        let mut better = base.clone();
        better.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Fraud),
                enabled: true,
            },
        );
        let reward = Reward {
            baseline: Tally {
                caught: 5,
                false_drops: 1,
                cheats: 40,
                cleans: 40,
            },
            candidate: Tally {
                caught: 5,
                false_drops: 1,
                cheats: 40,
                cleans: 40,
            },
        };
        assert_eq!(
            accept(&base, &better, &reward),
            Acceptance::Refused {
                reason: Refusal::NoImprovement
            }
        );
    }

    #[test]
    fn measure_reruns_the_same_corpus() {
        let base = HarnessProfile::default();
        let mut better = base.clone();
        better.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Fraud),
                enabled: true,
            },
        );
        let c = corpus();
        let r = measure(&c, &base, &better);
        assert_eq!(r.baseline.cheats + r.baseline.cleans, 40);
        assert_eq!(r.candidate.cheats + r.candidate.cleans, 40);
        // Escalating a class cannot make the corpus smaller or larger.
        assert_eq!(r.baseline.cheats, r.candidate.cheats);
    }

    #[test]
    fn mining_separates_what_the_scanner_caught_from_what_it_missed() {
        // A `todo!()` body is a DataStub the scanner grades itself, so a
        // labeled cheat carrying one must land in neither bucket: it was
        // caught, so there is nothing for the engineer to learn from it.
        let stub = Labeled {
            code: "fn g() { todo!() }".to_string(),
            lang: SnippetLanguage::Rust,
            cheat: true,
        };
        let clean = clean_snippet();
        let p = HarnessProfile::default();
        let packet = FailurePacket::mine(&[stub.clone(), clean.clone()], &p);

        let mut stub_report = scan_snippet_with(&stub.code, stub.lang, &p.mixing);
        p.apply(&mut stub_report);
        let stub_caught = stub_report.worst().is_some_and(|s| s >= Severity::Evasion);
        if stub_caught {
            assert!(
                !packet.missed.iter().any(|m| m.code == stub.code),
                "a cheat the scanner already catches must not appear as a miss"
            );
        }

        let mut clean_report = scan_snippet_with(&clean.code, clean.lang, &p.mixing);
        p.apply(&mut clean_report);
        let clean_dropped = clean_report.worst().is_some_and(|s| s >= Severity::Evasion);
        assert_eq!(
            packet.false_drops.iter().any(|m| m.code == clean.code),
            clean_dropped,
            "a clean trace belongs in the false-drop bucket exactly when the scanner drops it"
        );
    }

    #[test]
    fn packet_holds_both_directions_of_error() {
        let p = HarnessProfile::default();
        let c = corpus();
        let packet = FailurePacket::mine(&c, &p);
        // The packet cannot contain both directions unless the scanner made
        // both kinds of error; the assertion is that the split is computed,
        // not that either side is non-empty.
        assert_eq!(
            packet.evidence(),
            packet.missed.len() + packet.false_drops.len()
        );
        assert!(packet.evidence() <= c.len());
    }

    #[test]
    fn propose_only_emits_tightening_candidates() -> Result<()> {
        let base = HarnessProfile::default();
        let c = corpus();
        let group = propose(&c, &base, DEFAULT_CANDIDATES)?;
        assert!(!group.is_empty(), "the engineer must propose something");
        for cand in &group {
            assert!(
                cand.profile.is_monotone_over(&base),
                "candidate `{}` loosened the profile",
                cand.profile.diff_summary(&base)
            );
            assert!(
                !is_safety_refusal(&cand.acceptance),
                "a generated candidate must not trip the loosening refusal"
            );
        }
        Ok(())
    }

    #[test]
    fn registry_rollback_restores_the_unedited_scanner() -> Result<()> {
        let base = HarnessProfile::default();
        let c = corpus();
        let group = propose(&c, &base, DEFAULT_CANDIDATES)?;
        let mut reg = Registry::default_registry();
        let start = reg.current().clone();

        let mut committed = 0usize;
        for cand in &group {
            if cand.acceptance.is_accepted() {
                reg.commit(&start, cand, "test")?;
                committed += 1;
                break;
            }
        }
        if committed > 0 {
            assert_ne!(reg.current(), &start);
            reg.rollback()?;
            assert_eq!(
                reg.current(),
                &start,
                "rollback must restore the profile that was in force"
            );
        }
        // Rolling back past the baseline is refused.
        assert!(reg.rollback().is_err());
        Ok(())
    }

    #[test]
    fn registry_refuses_to_commit_a_refused_candidate() {
        let base = HarnessProfile::default();
        let mut worse = base.clone();
        worse.mixing.min_score = base.mixing.min_score.saturating_sub(1);
        let cand = Candidate {
            profile: worse,
            reward: Reward {
                baseline: Tally {
                    caught: 0,
                    false_drops: 0,
                    cheats: 40,
                    cleans: 40,
                },
                candidate: Tally {
                    caught: 0,
                    false_drops: 0,
                    cheats: 40,
                    cleans: 40,
                },
            },
            acceptance: Acceptance::Refused {
                reason: Refusal::Loosening,
            },
        };
        let mut reg = Registry::default_registry();
        assert!(reg.commit(&base, &cand, "test").is_err());
        assert_eq!(
            reg.entries.len(),
            1,
            "a refused candidate must not land on the chain"
        );
    }

    #[test]
    fn run_round_ranks_accepted_first() -> Result<()> {
        let c = corpus();
        let reg = Registry::default_registry();
        let round = run_round(&c, &reg, DEFAULT_CANDIDATES)?;
        // Accepted sorts first, so once one accepted candidate is seen every
        // later candidate must also be accepted.
        let mut seen_accepted = false;
        for cand in &round.candidates {
            if seen_accepted {
                assert!(
                    cand.acceptance.is_accepted(),
                    "a refused candidate sorted ahead of an accepted one: `{}`",
                    cand.summary(reg.current())
                );
            }
            if cand.acceptance.is_accepted() {
                seen_accepted = true;
            }
        }
        // The round is serializable, which is what the corpus log consumes.
        let text = round_json(&round)?;
        assert!(text.contains("candidates"));
        let rendered = render_round(&round, reg.current());
        assert!(rendered.contains("packet:"));
        Ok(())
    }

    /// The corpus loader is the only way evidence enters this module, so its
    /// refusals are part of the contract: a row with no code would otherwise be
    /// clean by construction and pad the denominator of every rate.
    #[test]
    fn the_corpus_loader_refuses_what_would_bias_a_rate() -> Result<()> {
        let dir = std::env::temp_dir();

        let empty_code = dir.join(format!("cheatharness-empty-{}.jsonl", std::process::id()));
        std::fs::write(
            &empty_code,
            "{\"code\":\"\",\"lang\":\"rust\",\"cheat\":false}\n",
        )?;
        let err = load_corpus(&empty_code).expect_err("an empty snippet must be refused");
        assert!(err.to_string().contains("no code"), "got: {err}");
        drop(std::fs::remove_file(&empty_code));

        let malformed = dir.join(format!("cheatharness-bad-{}.jsonl", std::process::id()));
        std::fs::write(&malformed, "not json\n")?;
        let err = load_corpus(&malformed).expect_err("a malformed row must be refused");
        assert!(
            err.to_string().contains("bad") || err.to_string().contains(":1:"),
            "the refusal must name the file and line, got: {err}"
        );
        drop(std::fs::remove_file(&malformed));

        let empty_file = dir.join(format!("cheatharness-none-{}.jsonl", std::process::id()));
        std::fs::write(&empty_file, "\n\n")?;
        let err = load_corpus(&empty_file).expect_err("an empty corpus must be refused");
        assert!(err.to_string().contains("empty"), "got: {err}");
        drop(std::fs::remove_file(&empty_file));

        let missing = dir.join("cheatharness-does-not-exist.jsonl");
        assert!(load_corpus(&missing).is_err(), "a missing file is an error");
        Ok(())
    }

    /// A corpus survives the disk round trip with its labels and languages
    /// intact — a label that flipped in transit would silently move an item
    /// between the two halves of every comparison.
    #[test]
    fn a_corpus_round_trips_through_disk() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("cheatharness-c-{}.jsonl", std::process::id()));
        let items = vec![
            Labeled {
                code: "#[allow(dead_code)] // audit-allow: mirrors the decoder\nfn f() {}"
                    .to_string(),
                lang: SnippetLanguage::Rust,
                cheat: true,
            },
            Labeled {
                code: "pub fn add(a: i64, b: i64) -> i64 { a + b }".to_string(),
                lang: SnippetLanguage::Rust,
                cheat: false,
            },
            Labeled {
                code: "def fib(n):\n    return n".to_string(),
                lang: SnippetLanguage::Python,
                cheat: false,
            },
        ];
        save_corpus(&path, &items)?;
        let back = load_corpus(&path)?;
        assert_eq!(back.len(), items.len());
        for (a, b) in items.iter().zip(back.iter()) {
            assert_eq!(a.code, b.code, "code must survive the round trip");
            assert_eq!(a.cheat, b.cheat, "the label must survive the round trip");
            assert_eq!(a.lang, b.lang, "the language must survive the round trip");
        }
        drop(std::fs::remove_file(&path));
        Ok(())
    }

    /// The two halves of a packet must travel together. A packet saved as only
    /// its misses would be a corpus where rejecting everything scores well.
    #[test]
    fn a_saved_packet_carries_both_halves() -> Result<()> {
        // A cheat the baseline misses (a reasoned suppression grades Note) and
        // a clean trace it drops, so both buckets are non-empty.
        let missed = Labeled {
            code: "#[allow(dead_code)] // audit-allow: mirrors the decoder\nfn f() {}".to_string(),
            lang: SnippetLanguage::Rust,
            cheat: true,
        };
        let dropped = Labeled {
            code: "pass".to_string(),
            lang: SnippetLanguage::Python,
            cheat: false,
        };
        let packet = FailurePacket {
            missed: vec![missed.clone()],
            false_drops: vec![dropped.clone()],
        };
        let path =
            std::env::temp_dir().join(format!("cheatharness-p-{}.jsonl", std::process::id()));
        save_packet(&path, &packet)?;
        let back = load_corpus(&path)?;
        assert_eq!(back.len(), 2, "both halves of the packet must be saved");
        assert!(
            back.iter().any(|i| i.cheat) && back.iter().any(|i| !i.cheat),
            "a saved packet that lost a half is a corpus where rejecting everything wins"
        );
        drop(std::fs::remove_file(&path));

        // And the empty half is simply absent, not fabricated.
        let only_misses = FailurePacket {
            missed: vec![missed],
            false_drops: Vec::new(),
        };
        save_packet(&path, &only_misses)?;
        assert_eq!(load_corpus(&path)?.len(), 1);
        drop(std::fs::remove_file(&path));
        Ok(())
    }

    /// The accept path, end to end, on a corpus where a reasoned suppression
    /// grades [`Severity::Note`] and therefore does not reject on its own.
    /// Escalating that class is a real edit with a real measured gain.
    #[test]
    fn a_tightening_edit_is_accepted_when_a_rerun_shows_it_helps() {
        let clean = "pub fn add(a: i64, b: i64) -> i64 { a + b }";
        let reasoned = "#[allow(dead_code)] // audit-allow: mirrors the decoder\nfn f() {}";
        let corpus: Vec<Labeled> = (0..20)
            .map(|_| Labeled {
                code: clean.to_string(),
                lang: SnippetLanguage::Rust,
                cheat: false,
            })
            .chain((0..20).map(|_| Labeled {
                code: reasoned.to_string(),
                lang: SnippetLanguage::Rust,
                cheat: true,
            }))
            .collect();

        let base = HarnessProfile::default();
        let before = tally(&corpus, &base);
        assert_eq!(
            before.caught, 0,
            "a reasoned suppression grades Note, so the unedited scanner keeps every one of them"
        );

        let mut tighter = base.clone();
        tighter.gates.insert(
            class_key(CheatClass::DataSuppression).to_string(),
            ClassGate {
                severity: Some(Severity::Evasion),
                enabled: true,
            },
        );
        let reward = measure(&corpus, &base, &tighter);
        assert_eq!(reward.caught_gained(), 20, "the edit must catch all twenty");
        assert_eq!(
            reward.false_drops_added(),
            0,
            "escalating one class must not cost a clean trace here"
        );
        assert!(matches!(
            accept(&base, &tighter, &reward),
            Acceptance::Accepted { delta: 20 }
        ));
    }

    #[test]
    fn escalate_stops_at_the_ceiling() {
        assert_eq!(escalate(Severity::Note), Some(Severity::Evasion));
        assert_eq!(escalate(Severity::Evasion), Some(Severity::Suppression));
        assert_eq!(escalate(Severity::Suppression), Some(Severity::Fraud));
        assert_eq!(escalate(Severity::Fraud), None);
    }

    #[test]
    fn profile_roundtrips_through_disk() -> Result<()> {
        let mut p = HarnessProfile::default();
        p.gates.insert(
            class_key(CheatClass::DataStub).to_string(),
            ClassGate {
                severity: Some(Severity::Suppression),
                enabled: true,
            },
        );
        let path = std::env::temp_dir().join(format!("cheatharness-{}.json", std::process::id()));
        p.save(&path)?;
        let back = HarnessProfile::load(&path)?;
        assert_eq!(p, back);
        drop(std::fs::remove_file(&path));
        Ok(())
    }

    #[test]
    fn a_gate_naming_an_unknown_class_is_refused() {
        // serde accepts any string key, so without `validate` a typo becomes
        // a gate that never matches: a suppression that reads as active and
        // does nothing. The refusal has to happen at load and at commit.
        let p = HarnessProfile {
            version: PROFILE_VERSION,
            mixing: MixingPolicy::default(),
            gates: [("data-stubb".to_string(), ClassGate::default())]
                .into_iter()
                .collect(),
            provenance: "typo".to_string(),
        };
        let err = p.validate().expect_err("a bogus class key must be refused");
        assert!(
            err.to_string().contains("data-stubb"),
            "the refusal must name the offending key, got: {err}"
        );

        let path =
            std::env::temp_dir().join(format!("cheatharness-bad-{}.json", std::process::id()));
        std::fs::write(
            &path,
            serde_json::to_string(&p).expect("the fixture serializes"),
        )
        .expect("writing the fixture is setup, not the assertion");
        let err = HarnessProfile::load(&path).expect_err("load must refuse the bogus key");
        assert!(err.to_string().contains("data-stubb"));
        drop(std::fs::remove_file(&path));

        // Every real class must survive the same check.
        let mut good = HarnessProfile::default();
        for class in ALL_CLASSES {
            good.gates
                .insert(class_key(class).to_string(), ClassGate::default());
        }
        good.validate()
            .expect("every real CheatClass must be a valid gate key");
    }
}
