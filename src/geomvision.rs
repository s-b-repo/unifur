//! Diagram grounding and the visual/symbolic closed loop (roadmap Phase 33,
//! audit fixes 3 and 5).
//!
//! A vision model reading a figure reports what the pixels look like. Pixels
//! are not premises: a figure drawn with a 1-degree wobble still has to yield
//! "these two lines are parallel" only when the problem *says* so, and a figure
//! whose artist was sloppy must not invent an intersection that the statement
//! denies. Grounding is therefore a pipeline with a ledger, not a caption:
//!
//! ```text
//! image -> stroke/arc detection -> intersection detection -> label association
//!       -> topological reconstruction -> canonical IR -> reasoner
//! ```
//!
//! and every observation it produces carries an [`Evidence`] strength. Weak
//! evidence never becomes a fact: a pair of strokes measured at 0.4 degrees of
//! wobble yields
//!
//! ```text
//! visual evidence: approximately parallel
//! logical state:   UNKNOWN
//! ```
//!
//! and the predicate stays on the graph's not-established ledger until the
//! statement or a deduction supplies it.
//!
//! The reverse direction closes the loop: the same IR renders back to SVG, so a
//! proof can be drawn, a counterexample can be drawn, and a rendered-then-
//! regrounded figure must reconstruct the graph it came from. That round-trip
//! check is the grounding test -- `image -> IR -> proof -> IR -> image` rather
//! than the model translating between pixels and prose in its head.
//!
//! # The rule, and where it lives in the type system
//!
//! The rule is that a graded observation can never *establish* a predicate,
//! and the module enforces it by making graded evidence the only thing the
//! topology pass is allowed to write. Concretely:
//!
//! - [`Evidence::Exact`] -- stated by the problem -- is the only evidence that
//!   is [`Evidence::is_trustworthy`]. A [`Evidence::Measured`] reading is
//!   untrustworthy *whatever* its wobble, including at exactly zero degrees:
//!   there is no threshold at which "close enough" quietly becomes a premise,
//!   because a threshold is precisely the place a sloppy drawing would enter
//!   the ledger as a fact. Its grade is [`MEASURED_CEILING`] / (1 + wobble) --
//!   strictly below 1.0 always.
//! - A *relation* between two strokes -- parallel, perpendicular, of equal
//!   length -- can only ever be [`suggest`]ed. No grade of visual evidence
//!   makes it a fact at any confidence; it goes onto the not-established
//!   ledger naming the measurement that raised it, and only a stated premise
//!   ([`record_stated`]) or a rule (the kernel's saturation, wrapped here as
//!   [`saturate`]) can supply the predicate.
//! - An *incidence* -- a detected crossing, an endpoint ordering, a point
//!   sitting exactly on a line -- is routed through [`record_visual`], which
//!   hands the kernel's own [`Fact::observed`] to [`SceneGraph::add_fact`] with
//!   the grade encoded in the `observed:<confidence>` operation string the
//!   kernel parses back. The exact predicate still decides whether the
//!   incidence is admitted at all, and it arrives strictly below certainty.
//!
//! So a figure drawn with 0.4 degrees of wobble yields
//!
//! ```text
//! visual evidence: approximately parallel(AB, CD) (0.4 deg of wobble, from hough-grid)
//! logical state:   UNKNOWN
//! ```
//!
//! on the ledger, and a derivation that consumed it is worth at most its
//! grade, multiplied along the dependency chain by [`SceneGraph::confidence_in`].
//!
//! # The reverse direction
//!
//! [`render_svg`] writes the same IR back out as a real SVG document -- exact
//! string building, no dependencies -- and [`parse_svg`] reads it back, so
//! [`round_trip`] can check that a figure which came out of a graph goes back
//! into the same graph. Stated honestly: the re-parser is a parser of this
//! module's own emitter, so the round trip validates the topology and
//! labelling path, not a real vision model's accuracy. What it does catch is
//! the failure that matters here -- a pass that loses a junction, double
//! counts a shared endpoint, or invents a point on the way through.

use crate::geomkernel::{
    Constraint, Fact, Frac, KCircle, KPoint, Provenance, Rule, SceneGraph, Segment, Q,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ----------------------------------------------------------------- constants --

/// The image lattice the kernel reasons over: a detected coordinate is snapped
/// to the nearest multiple of `1/65536` before it becomes a rational. This is
/// the *only* place a float enters the IR, and the grid is named so the error
/// is bounded by half a cell -- about one part in 131072 of the figure, well
/// under the stroke thickness a detector would have measured anyway.
pub const IMAGE_GRID: i128 = 65_536;

/// Two detected locations closer than this (in normalized image units) are the
/// same point of the figure. Below the weld tolerance they are indistinguishable
/// ink; above it they are two points the reasoner is allowed to keep apart.
///
/// Deliberately coarser than [`PROXIMITY_TOLERANCE`], and the two are meant to
/// agree: two lines close enough to be refused as a near miss are close enough
/// that their *ends* are the same points, because a figure that cannot tell the
/// lines apart cannot tell their endpoints apart either. A pass that welded
/// more tightly than it refused would happily keep four points for one line
/// drawn twice.
pub const WELD_TOLERANCE: f64 = 1.0 / 128.0;

/// Two strokes whose closest approach is below this do not intersect. Proximity
/// is not incidence: a figure whose lines pass within a thousandth of each other
/// has not drawn a junction, and reporting one would invent a point the problem
/// never mentioned.
pub const PROXIMITY_TOLERANCE: f64 = 1.0 / 256.0;

/// The shallowest crossing this module will call a junction: the sine of the
/// angle between the two strokes. Below it the strokes graze, and the crossing
/// point is a function of the wobble rather than of the geometry.
pub const MIN_CROSS_SINE: f64 = 0.05;

/// How far, in degrees, two stroke directions may differ and still *suggest*
/// parallelism or perpendicularity. The number only decides what lands on the
/// ledger as a measurement; it can never promote a suggestion to a fact.
pub const ANGLE_TOLERANCE_DEGREES: f64 = 0.5;

/// The relative length difference under which two strokes *suggest* equality of
/// length. Same standing as [`ANGLE_TOLERANCE_DEGREES`]: a suggestion only.
pub const LENGTH_TOLERANCE: f64 = 1.0 / 128.0;

/// The grade a *perfect* measurement can reach. Strictly below `1.0`, and that
/// strictness is the whole point: a detector that agrees with the figure to the
/// last bit it can resolve still has a systematic error the picture does not
/// contain, so its evidence can never be promoted to the kernel's certainty.
pub const MEASURED_CEILING: f64 = 0.99;

/// The grade of a bare categorical claim -- "these look parallel" from a
/// detector that reports no residual. A flat number rather than a function,
/// because a claim without an error bar has no error bar to grade: it sits
/// above a measurement so wobbly the measurement says nothing (past a wobble
/// of `MEASURED_CEILING / ASSERTED_CONFIDENCE - 1`, about 0.65 degrees, the
/// measured grade falls below this one) and far below any premise.
pub const ASSERTED_CONFIDENCE: f64 = 0.6;

/// A parameter this close to `0` or `1` is an endpoint, not an interior point:
/// a T-junction is a stroke that *ends* on another, and the classification
/// turns on exactly this.
pub const PARAMETER_EPSILON: f64 = 1.0e-9;

/// The sweep at or above which a detected arc is a whole circle rather than a
/// piece of one. A partial arc gives a centre but no circle object, because the
/// kernel's [`KCircle`] carries an identity and an exact squared radius, and a
/// quarter turn determines neither.
pub const FULL_SWEEP_DEGREES: f64 = 359.0;

/// The XML namespace an SVG document must declare, and the one [`render_svg`]
/// declares and the round-trip test checks.
pub const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

// ------------------------------------------------------------------ evidence --

/// How strongly an observation is supported.
///
/// The variant answers "who said this, and how do they know?", and it decides
/// exactly one thing that matters downstream: whether the observation may
/// *establish* a predicate. Only [`Evidence::Exact`] -- stated by the problem
/// -- may.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Evidence {
    /// Stated by the problem. The only trustworthy evidence there is.
    Exact,
    /// A numeric grade from a detector: the residual, in degrees, of the
    /// measurement this came from, and the detector's name.
    Measured { wobble: f64, source: String },
    /// A detector's categorical claim, with a name and no residual.
    Asserted { source: String },
}

impl Evidence {
    /// Whether this evidence may establish a predicate. Only [`Evidence::Exact`]
    /// may, and the threshold is zero: there is no wobble small enough.
    ///
    /// Why a threshold-free rule and not a tolerance: a tolerance is a place
    /// where "approximately" gets to mean "exactly", and a figure whose artist
    /// was sloppy would then enter the ledger as a premise the problem never
    /// gave. A measured reading is graded instead, and the grade is carried
    /// along every later step by [`SceneGraph::confidence_in`], so the
    /// uncertainty survives into the conclusion instead of being rounded off at
    /// the first step.
    pub fn is_trustworthy(&self) -> bool {
        matches!(self, Self::Exact)
    }

    /// The grade this evidence carries, in `[0, 1]`.
    ///
    /// The measured grade is [`MEASURED_CEILING`] / (1 + wobble): strictly
    /// below one for every finite wobble -- including `0.0` -- falling
    /// hyperbolically, so half a degree already costs a third of the ceiling
    /// and four degrees leaves a fifth of it. The asserted grade is the flat
    /// [`ASSERTED_CONFIDENCE`], because a claim with no residual has nothing
    /// to be a function of.
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Exact => 1.0,
            Self::Measured { wobble, .. } => MEASURED_CEILING / (1.0 + wobble.abs()),
            Self::Asserted { .. } => ASSERTED_CONFIDENCE,
        }
    }

    /// The measurement whose grade is exactly `confidence` -- the inverse of
    /// the rule above, so a pipeline can say "grade this claim `0.83`" and get
    /// the wobble that implies. A graded claim is still untrustworthy.
    pub fn graded(confidence: f64, source: &str) -> Self {
        let wanted = confidence.clamp(1.0e-3, MEASURED_CEILING);
        Self::Measured {
            wobble: MEASURED_CEILING / wanted - 1.0,
            source: source.to_string(),
        }
    }

    /// The weaker of two observations, which is what a claim resting on two
    /// strokes is worth. Two measurements add their wobbles -- the worst case
    /// for independent errors -- and a measurement beside a bare claim degrades
    /// to the bare claim, because once a categorical detector is in the chain
    /// the chain has no error bar to add up.
    pub fn weakest(first: &Self, second: &Self) -> Self {
        match (first, second) {
            (Self::Exact, Self::Exact) => Self::Exact,
            (Self::Exact, other) | (other, Self::Exact) => other.clone(),
            (
                Self::Measured {
                    wobble: a,
                    source: sa,
                },
                Self::Measured {
                    wobble: b,
                    source: sb,
                },
            ) => Self::Measured {
                wobble: a + b,
                source: format!("{sa}+{sb}"),
            },
            (Self::Asserted { source: sa }, Self::Asserted { source: sb }) => Self::Asserted {
                source: format!("{sa}+{sb}"),
            },
            (Self::Asserted { source: sa }, other) | (other, Self::Asserted { source: sa }) => {
                Self::Asserted {
                    source: format!("{sa}+{}", other.source()),
                }
            }
        }
    }

    /// The detector, or the problem, this came from -- for a report.
    pub fn source(&self) -> &str {
        match self {
            Self::Exact => "the problem statement",
            Self::Measured { source, .. } | Self::Asserted { source } => source,
        }
    }

    /// The evidence half of a report: what the picture said, in the words a
    /// caption would use. Never a verdict -- that is the other half of
    /// [`Evidence::report`], and keeping both in one function is what stops
    /// "looks parallel" from drifting into "is parallel".
    pub fn evidence_line(&self, claim: &str) -> String {
        match self {
            Self::Exact => format!("stated: {claim}"),
            Self::Measured { wobble, source } => {
                format!("approximately {claim} ({wobble} deg of wobble, from {source})")
            }
            Self::Asserted { source } => format!("{claim} (asserted by {source})"),
        }
    }

    /// The logical state this evidence leaves a claim in. `ESTABLISHED` only
    /// for [`Evidence::Exact`]; a measurement, however small its wobble, leaves
    /// the claim `UNKNOWN`.
    pub fn logical_state(&self) -> &'static str {
        if self.is_trustworthy() {
            "ESTABLISHED"
        } else {
            "UNKNOWN"
        }
    }

    /// The two-line report this module is about: the evidence, and the logical
    /// state it does *not* license.
    pub fn report(&self, claim: &str) -> String {
        format!(
            "visual evidence: {}\nlogical state:   {}",
            self.evidence_line(claim),
            self.logical_state()
        )
    }
}

impl fmt::Display for Evidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Self::Exact => "exact".to_string(),
            Self::Measured { source, .. } => format!("measured by {source}"),
            Self::Asserted { source } => format!("asserted by {source}"),
        })
    }
}

/// What became of a claim on its way through the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Stated by the problem: a premise, and the only verdict that licenses a
    /// derivation.
    Stated,
    /// Observed, and admitted by the kernel's exact predicate, at this grade.
    Observed { confidence: f64 },
    /// Not established: the claim is on the graph's not-established ledger.
    Unknown,
}

impl Verdict {
    /// The grade this verdict carries: `1.0` for a stated premise, the
    /// observation's own grade for an admitted observation, and `0.0` for a
    /// claim that stayed open.
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Stated => 1.0,
            Self::Observed { confidence } => *confidence,
            Self::Unknown => 0.0,
        }
    }

    /// Whether this verdict establishes the claim.
    pub fn is_established(&self) -> bool {
        self.confidence() >= 1.0
    }
}

// ---------------------------------------------------------------- detections --

/// A detected line segment: where a stroke's two ends are, how thick the ink
/// is, and what the detector knows about its own accuracy.
///
/// The coordinates are floats in normalized image space because that is what a
/// detector produces; [`reconstruct`] snaps them onto the kernel's lattice
/// before anything downstream is allowed to treat one as a fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stroke {
    pub from: (f64, f64),
    pub to: (f64, f64),
    /// The ink's width, in the same units. Carried because a detector's
    /// uncertainty is a function of it, and because a renderer needs it; this
    /// module does not yet turn it into a per-stroke wobble (see the honest
    /// limits in the module docs).
    pub thickness: f64,
    pub evidence: Evidence,
}

impl Stroke {
    /// A stroke with the given ends, thickness and evidence.
    pub fn new(from: (f64, f64), to: (f64, f64), thickness: f64, evidence: Evidence) -> Self {
        Self {
            from,
            to,
            thickness,
            evidence,
        }
    }

    /// The direction vector, `(dx, dy)`.
    pub fn direction(&self) -> (f64, f64) {
        (self.to.0 - self.from.0, self.to.1 - self.from.1)
    }

    /// The measured length.
    pub fn length(&self) -> f64 {
        let (dx, dy) = self.direction();
        dx.hypot(dy)
    }

    /// The direction in degrees, in `[0, 180)`: a stroke and the same stroke
    /// drawn backwards are the same line, and a parallel claim must not depend
    /// on which end the detector happened to list first.
    pub fn angle_degrees(&self) -> f64 {
        let (dx, dy) = self.direction();
        dy.atan2(dx).to_degrees().rem_euclid(180.0)
    }

    /// A stroke with coincident ends. It defines no direction, and every
    /// predicate that would consume one is refused rather than evaluated.
    pub fn is_degenerate(&self) -> bool {
        self.length() == 0.0
    }
}

/// A detected circular arc: a centre, a radius, and how far round it goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arc {
    pub center: (f64, f64),
    pub radius: f64,
    /// The swept angle, in degrees; negative sweeps clockwise.
    pub sweep_degrees: f64,
    pub evidence: Evidence,
}

impl Arc {
    /// Whether the arc closes on itself, which is what makes it a whole circle
    /// rather than a piece of one.
    pub fn is_closed(&self) -> bool {
        self.sweep_degrees.abs() >= FULL_SWEEP_DEGREES
    }

    /// Where the sweep starts, on the convention SVG uses (zero degrees at
    /// three o'clock, positive anticlockwise).
    pub fn start(&self) -> (f64, f64) {
        (self.center.0 + self.radius, self.center.1)
    }

    /// Where the sweep ends.
    pub fn end(&self) -> (f64, f64) {
        let radians = self.sweep_degrees.to_radians();
        (
            self.center.0 + self.radius * radians.cos(),
            self.center.1 + self.radius * radians.sin(),
        )
    }
}

/// A piece of OCR text and where it sits in the image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label {
    pub text: String,
    pub at: (f64, f64),
}

impl Label {
    /// A label reading `text`, anchored at `at`.
    pub fn new(text: &str, at: (f64, f64)) -> Self {
        Self {
            text: text.to_string(),
            at,
        }
    }

    /// The distance from the anchor to a detected location -- the number a
    /// label-to-entity association is decided on.
    pub fn distance_to(&self, at: (f64, f64)) -> f64 {
        (self.at.0 - at.0).hypot(self.at.1 - at.1)
    }
}

/// The geometric entity a label names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "entity", rename_all = "snake_case")]
pub enum EntityRef {
    /// A point of the figure -- the only association that can *name* one,
    /// because a point's identity is its name.
    Point(String),
    /// A segment, named by its two ends.
    Segment { from: String, to: String },
    /// A circle object.
    Circle(String),
}

impl EntityRef {
    /// The entity's name, where it has one; a segment's is its pair of ends.
    pub fn name(&self) -> String {
        match self {
            Self::Point(name) | Self::Circle(name) => name.clone(),
            Self::Segment { from, to } => format!("{from}{to}"),
        }
    }

    /// The encoding [`render_svg`] writes and [`parse_svg`] reads: one
    /// attribute value, no nesting, so the round trip needs no parser for the
    /// label grammar itself.
    pub fn encode(&self) -> String {
        match self {
            Self::Point(name) => format!("point:{name}"),
            Self::Circle(name) => format!("circle:{name}"),
            Self::Segment { from, to } => format!("segment:{from},{to}"),
        }
    }

    /// The inverse of [`EntityRef::encode`]. An entity this module cannot name
    /// -- or one named with nothing at all -- is a malformed file, not a silent
    /// default: an empty name would be a point that collides with every other
    /// unnamed point the figure has.
    pub fn decode(encoded: &str) -> anyhow::Result<Self> {
        let (kind, rest) = encoded
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("a label must name an entity, got '{encoded}'"))?;
        let named = |name: &str| -> anyhow::Result<String> {
            anyhow::ensure!(
                !name.is_empty(),
                "a label must name something, got '{encoded}'"
            );
            Ok(name.to_string())
        };
        match kind {
            "point" => Ok(Self::Point(named(rest)?)),
            "circle" => Ok(Self::Circle(named(rest)?)),
            "segment" => {
                let (from, to) = rest.split_once(',').ok_or_else(|| {
                    anyhow::anyhow!("a segment label needs two ends, got '{encoded}'")
                })?;
                Ok(Self::Segment {
                    from: named(from)?,
                    to: named(to)?,
                })
            }
            other => Err(anyhow::anyhow!("unknown label entity kind '{other}'")),
        }
    }
}

impl fmt::Display for EntityRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Self::Point(name) => format!("point {name}"),
            Self::Circle(name) => format!("circle {name}"),
            Self::Segment { from, to } => format!("segment {from}{to}"),
        })
    }
}

/// OCR text bound to a geometric entity, with the grade of the binding.
///
/// A name is not a premise: which point a label belongs to is a *detection*,
/// and a mis-read letter puts a name on the wrong point, which then poisons
/// every rule downstream. So the association carries its own confidence, and
/// the ledger records what the picture claimed about the name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelAssociation {
    pub label: Label,
    pub entity: EntityRef,
    /// The binding's grade -- the evidence's own [`Evidence::confidence`], kept
    /// as a field so a caller can read it without matching on the evidence.
    pub confidence: f64,
    pub evidence: Evidence,
}

impl LabelAssociation {
    /// An association whose confidence is its evidence's grade, which is the
    /// only way the two are built.
    pub fn new(label: Label, entity: EntityRef, evidence: Evidence) -> Self {
        let confidence = evidence.confidence();
        Self {
            label,
            entity,
            confidence,
            evidence,
        }
    }

    /// Whether this binding may name a point outright. Only an exact reading
    /// can, and no OCR ever is -- which is why a wrong label costs a fact its
    /// certainty rather than silently becoming one.
    pub fn is_trustworthy(&self) -> bool {
        self.evidence.is_trustworthy()
    }
}

// -------------------------------------------------------------- intersections --

/// What kind of junction two strokes make where they meet.
///
/// The three kinds are not decoration: "the lines cross", "a line stops on
/// another", and "two lines share a point" are different theorems with
/// different consequences, and a detector that reports all three as "an
/// intersection" cannot tell a transversal from a T.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntersectionKind {
    /// Both strokes pass through the junction: a proper crossing.
    Crossing,
    /// One stroke ends on the interior of the other: a stem on a through-line.
    TJunction,
    /// Both strokes end at the same point: a shared vertex.
    SharedEndpoint,
}

impl fmt::Display for IntersectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Crossing => "crossing",
            Self::TJunction => "T-junction",
            Self::SharedEndpoint => "shared endpoint",
        })
    }
}

/// A junction the topology pass found, with the parameters that classified it
/// and the evidence that put it there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Intersection {
    /// Index of the first stroke in the slice that was scanned.
    pub first: usize,
    /// Index of the second.
    pub second: usize,
    /// Where the two strokes meet.
    pub at: (f64, f64),
    /// Where along the first stroke, in `[0, 1]`: `0` is `from`, `1` is `to`.
    pub along_first: f64,
    /// The same for the second stroke.
    pub along_second: f64,
    pub kind: IntersectionKind,
    /// The weaker of the two strokes' evidence -- a junction is worth exactly
    /// what the shakier of the two strokes that make it is worth.
    pub evidence: Evidence,
}

/// Why a pair of strokes that a sloppier pass would have called a junction was
/// not called one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NearMissReason {
    /// One of the two strokes has coincident ends and defines no direction.
    Degenerate,
    /// The strokes come within [`PROXIMITY_TOLERANCE`] of each other without
    /// crossing: the same ink, or two lines drawn almost on top of each other.
    Proximity,
    /// The strokes do cross, but at a shallower angle than [`MIN_CROSS_SINE`],
    /// so the crossing point is a function of the wobble rather than of the
    /// geometry.
    Shallow,
}

/// A pair that came close to being a junction, and was not. Reported rather
/// than dropped: a figure whose two lines are a fifth of a degree apart is
/// exactly the case a reader should be told about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NearMiss {
    pub first: usize,
    pub second: usize,
    /// The point of closest approach.
    pub at: (f64, f64),
    pub distance: f64,
    pub reason: NearMissReason,
}

/// Every junction in a set of strokes, classified and graded.
///
/// Pairs that only come close are *not* here: see [`near_misses`]. That
/// separation is the whole content of the "topological reconstruction" step --
/// a figure's ink is a set of segments, and a segment set has junctions only
/// where two segments genuinely meet.
pub fn intersections(strokes: &[Stroke]) -> Vec<Intersection> {
    let mut found = Vec::new();
    for i in 0..strokes.len() {
        for j in (i + 1)..strokes.len() {
            if let Contact::Junction {
                at,
                along_first,
                along_second,
                kind,
            } = pair_contact(&strokes[i], &strokes[j])
            {
                found.push(Intersection {
                    first: i,
                    second: j,
                    at,
                    along_first,
                    along_second,
                    kind,
                    evidence: Evidence::weakest(&strokes[i].evidence, &strokes[j].evidence),
                });
            }
        }
    }
    found
}

/// Every pair of strokes that came close to being a junction, with the distance
/// and the reason. The mirror image of [`intersections`]: what the pass refused,
/// and why.
pub fn near_misses(strokes: &[Stroke]) -> Vec<NearMiss> {
    let mut missed = Vec::new();
    for i in 0..strokes.len() {
        for j in (i + 1)..strokes.len() {
            if let Some((reason, distance, at)) = pair_contact(&strokes[i], &strokes[j]).miss() {
                missed.push(NearMiss {
                    first: i,
                    second: j,
                    at,
                    distance,
                    reason,
                });
            }
        }
    }
    missed
}

/// What two strokes do to each other: meet, come close, or nothing.
enum Contact {
    Junction {
        at: (f64, f64),
        along_first: f64,
        along_second: f64,
        kind: IntersectionKind,
    },
    Miss(NearMissReason, f64, (f64, f64)),
    Nothing,
}

impl Contact {
    fn miss(self) -> Option<(NearMissReason, f64, (f64, f64))> {
        match self {
            Self::Miss(reason, distance, at) => Some((reason, distance, at)),
            Self::Junction { .. } | Self::Nothing => None,
        }
    }
}

/// Decide one pair. The geometry is the ordinary segment intersection, with
/// three refusals layered on it -- a degenerate stroke, a crossing too shallow
/// to locate, and a mere proximity. Each refusal is a *named* outcome rather
/// than a dropped pair, because "these two strokes do not meet" is a claim a
/// reader is entitled to check.
fn pair_contact(first: &Stroke, second: &Stroke) -> Contact {
    if first.is_degenerate() || second.is_degenerate() {
        return Contact::Miss(NearMissReason::Degenerate, 0.0, first.from);
    }
    let (u, v) = (first.direction(), second.direction());
    let sine = (u.0 * v.1 - u.1 * v.0).abs() / (first.length() * second.length());
    match crossing(first, second) {
        Some((along_first, along_second, at)) => {
            if sine < MIN_CROSS_SINE {
                return Contact::Miss(NearMissReason::Shallow, separation(first, second).0, at);
            }
            let interior =
                |value: f64| value > PARAMETER_EPSILON && value < 1.0 - PARAMETER_EPSILON;
            let kind = match (interior(along_first), interior(along_second)) {
                (true, true) => IntersectionKind::Crossing,
                (false, false) => IntersectionKind::SharedEndpoint,
                _ => IntersectionKind::TJunction,
            };
            Contact::Junction {
                at,
                along_first,
                along_second,
                kind,
            }
        }
        None => {
            // Parallel, or crossing outside the strokes: only proximity is
            // left, and proximity is not incidence.
            let (distance, at) = separation(first, second);
            if distance <= PROXIMITY_TOLERANCE {
                Contact::Miss(NearMissReason::Proximity, distance, at)
            } else {
                Contact::Nothing
            }
        }
    }
}

/// The two crossing parameters and the point they give, when two strokes cross
/// inside both of them. Raw geometry only: no angle test and no tolerance, so
/// that both the classifier and the distance measure can ask it the same
/// question without asking each other.
fn crossing(first: &Stroke, second: &Stroke) -> Option<(f64, f64, (f64, f64))> {
    let (p, r) = (first.from, second.from);
    let (u, v) = (first.direction(), second.direction());
    let (u_len, v_len) = (first.length(), second.length());
    if u_len == 0.0 || v_len == 0.0 {
        return None;
    }
    let denominator = u.0 * v.1 - u.1 * v.0;
    if denominator.abs() <= 1.0e-12 * u_len * v_len {
        return None;
    }
    let w = (r.0 - p.0, r.1 - p.1);
    let along_first = (w.0 * v.1 - w.1 * v.0) / denominator;
    let along_second = (w.0 * u.1 - w.1 * u.0) / denominator;
    let inside = |value: f64| (-PARAMETER_EPSILON..=1.0 + PARAMETER_EPSILON).contains(&value);
    if inside(along_first) && inside(along_second) {
        Some((
            along_first.clamp(0.0, 1.0),
            along_second.clamp(0.0, 1.0),
            (p.0 + along_first * u.0, p.1 + along_first * u.1),
        ))
    } else {
        None
    }
}

/// The least distance between two strokes, and where it happens. Zero when they
/// cross -- which is asked first, so a crossing never reports the distance to an
/// endpoint instead.
fn separation(first: &Stroke, second: &Stroke) -> (f64, (f64, f64)) {
    if let Some((_, _, at)) = crossing(first, second) {
        return (0.0, at);
    }
    let mut best = (f64::INFINITY, (0.0, 0.0));
    for (tip, host, on_host) in [
        (first.from, second, false),
        (first.to, second, false),
        (second.from, first, true),
        (second.to, first, true),
    ] {
        let (parameter, distance) = closest_on_segment(tip, host.from, host.to);
        let along = (
            host.from.0 + parameter * (host.to.0 - host.from.0),
            host.from.1 + parameter * (host.to.1 - host.from.1),
        );
        let at = if on_host { along } else { tip };
        if distance < best.0 {
            best = (distance, at);
        }
    }
    best
}

/// The point of the segment `from`-`to` nearest `tip`, and how far away it is.
fn closest_on_segment(tip: (f64, f64), from: (f64, f64), to: (f64, f64)) -> (f64, f64) {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let squared = dx * dx + dy * dy;
    if squared == 0.0 {
        return (0.0, (tip.0 - from.0).hypot(tip.1 - from.1));
    }
    let parameter = (((tip.0 - from.0) * dx + (tip.1 - from.1) * dy) / squared).clamp(0.0, 1.0);
    let at = (from.0 + parameter * dx, from.1 + parameter * dy);
    (parameter, (tip.0 - at.0).hypot(tip.1 - at.1))
}

// -------------------------------------------------------------------- ledger --

/// Write a line on the graph's not-established ledger, without repeats.
///
/// The single door to that ledger in this module. The kernel keeps its own
/// refusals there too (an exact predicate rejecting a statement), so the
/// ledger ends up holding both "nobody could decide this" and "the picture
/// suggested it, and here is the measurement" -- which is the honest state of
/// affairs for a diagram.
pub fn leave_open(scene: &mut SceneGraph, note: &str) {
    if !scene.not_established.iter().any(|line| line == note) {
        scene.not_established.push(note.to_string());
    }
}

/// Record a premise the *problem* stated -- the only route to
/// [`Provenance::Given`] and the only route to confidence `1.0` in this module.
///
/// The kernel still checks it against the reconstructed figure, and a premise
/// the drawing contradicts is refused with the reason on the ledger. That is
/// the intended behaviour, not a gap: when an artist's triangle is not the
/// triangle the problem states, the statement wins and the figure is reported
/// as disagreeing with it.
pub fn record_stated(scene: &mut SceneGraph, constraint: Constraint) -> Verdict {
    if scene.add_fact(constraint, Provenance::Given) {
        Verdict::Stated
    } else {
        Verdict::Unknown
    }
}

/// Record a *visual* observation: an incidence, an ordering, a midpoint a
/// detector believes it saw.
///
/// The grade rides in the operation string, in the kernel's own
/// `observed:<confidence>` convention, and the fact is built by the kernel's
/// [`Fact::observed`] so there is exactly one definition of "this is not
/// established". `SceneGraph::add_fact` then asks the exact predicate whether
/// the observation is even true of the reconstructed figure: a detector's claim
/// that a point sits on a line is admitted when it does and refused -- with the
/// reason on the ledger -- when it does not. Either way the fact lands below
/// certainty unless the evidence is [`Evidence::Exact`], and a graded
/// observation can never license a derivation.
pub fn record_visual(
    scene: &mut SceneGraph,
    constraint: Constraint,
    evidence: &Evidence,
) -> Verdict {
    let confidence = evidence.confidence();
    let observed = Fact::observed(constraint, &format!("observed:{confidence}"), confidence);
    if scene.add_fact(observed.constraint, observed.provenance) {
        Verdict::Observed { confidence }
    } else {
        Verdict::Unknown
    }
}

/// Record a *suggestion*: a relation between two strokes that the picture makes
/// plausible and that only a premise or a rule can establish.
///
/// Parallel, perpendicular and equal-length claims come through here and
/// through no other door. The rule is deliberately stricter than "below full
/// confidence": no grade of visual evidence makes them facts, because a figure
/// can only ever *look* like a parallelogram, and a model that lets the picture
/// supply the relation has replaced the problem's premise with a drawing. What
/// lands on the ledger is the measurement itself, so a reader can see how close
/// the figure came and decide.
pub fn suggest(scene: &mut SceneGraph, constraint: Constraint, evidence: &Evidence) -> Verdict {
    let claim = constraint.describe();
    leave_open(
        scene,
        &format!(
            "suggested: {claim} -- {}; logical state:   UNKNOWN (a relation is established by a premise or a rule, never by a figure)",
            evidence.evidence_line(&claim)
        ),
    );
    Verdict::Unknown
}

/// Run the kernel's deduction rules over a grounded graph until nothing new
/// fires.
///
/// This is where a grade starts multiplying, and where a subtlety lives worth
/// stating plainly. A *derived* fact is exact -- the rule is, and the kernel
/// stores it at confidence `1.0` -- even when every premise under it was graded
/// by a diagram reader. The parallelogram rule will happily derive a
/// parallelism from two midpoints a detector measured, and the parallelism
/// *is* true of the reconstructed figure. What is not established is that the
/// figure was worth believing, and the number that says so is
/// [`SceneGraph::confidence_in`], which walks the dependency chain and
/// multiplies: the derived parallelism comes back worth the product of the
/// grades underneath it, never `1.0`. So a consumer of a grounded graph must
/// read the chain, not the fact's own confidence -- and the rules themselves
/// are the only thing in the pipeline that establishes a relation, which is
/// exactly the rule the module docs state.
pub fn saturate(graph: &mut SceneGraph, max_rounds: usize) -> anyhow::Result<usize> {
    Rule::saturate(graph, max_rounds)
}

// ------------------------------------------------------------- reconstruction --

/// The reconstructed topology: the points of the figure on the kernel's
/// lattice, the segments between them, the junctions that were found, and the
/// [`SceneGraph`] they reconstruct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grounding {
    /// Every distinct point of the figure, welded from stroke endpoints, arc
    /// centres and detected crossings, and named by a label where one is close
    /// enough to bind.
    pub points: Vec<KPoint>,
    /// One segment per non-degenerate stroke.
    pub segments: Vec<Segment>,
    /// What the topology pass found, in stroke-pair order.
    pub intersections: Vec<Intersection>,
    /// The IR: points, graded and stated facts, and the not-established ledger.
    pub graph: SceneGraph,
}

impl Grounding {
    /// The point named `name`, if the figure has one.
    pub fn point(&self, name: &str) -> anyhow::Result<&KPoint> {
        self.graph.point(name)
    }

    /// Every point of the figure by name, sorted.
    pub fn point_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.points.iter().map(|p| p.name.clone()).collect();
        names.sort();
        names
    }
}

/// The pipeline's result: what the detectors saw, and the IR it reconstructs
/// from it.
///
/// The two halves are kept together deliberately. The detections are the
/// evidence, the graph is the reading, and a reader who is handed only the graph
/// cannot tell which of its facts the picture asserted and which the problem
/// did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundedScene {
    pub strokes: Vec<Stroke>,
    pub arcs: Vec<Arc>,
    pub labels: Vec<LabelAssociation>,
    /// The junctions the topology pass found. Recomputed by every
    /// [`GroundedScene::grounding`]; kept here so the result of the detection
    /// pass is inspectable without re-running the pass.
    pub intersections: Vec<Intersection>,
    /// The pairs that came close to being junctions and were refused, with the
    /// reason.
    pub near_misses: Vec<NearMiss>,
    /// The constraints the *problem* states. The only route to a fact at
    /// confidence `1.0`; see [`record_stated`].
    pub stated: Vec<Constraint>,
    /// The reconstructed IR.
    pub scene: SceneGraph,
}

impl GroundedScene {
    /// Run the whole pipeline: detect, weld, name, reconstruct.
    pub fn new(
        strokes: Vec<Stroke>,
        arcs: Vec<Arc>,
        labels: Vec<LabelAssociation>,
        stated: Vec<Constraint>,
    ) -> anyhow::Result<Self> {
        let grounding = ground(&strokes, &arcs, &labels, &stated)?;
        let misses = near_misses(&strokes);
        Ok(Self {
            strokes,
            arcs,
            labels,
            intersections: grounding.intersections.clone(),
            near_misses: misses,
            stated,
            scene: grounding.graph,
        })
    }

    /// The topology and the IR the detections reconstruct, rebuilt from scratch.
    ///
    /// Deliberately a *rebuild* rather than a getter: a stored graph nobody
    /// recomputed is a graph nobody can check, and this is what makes
    /// `scene.to_scene_graph()? == scene.scene` a meaningful identity rather
    /// than a tautology.
    pub fn grounding(&self) -> anyhow::Result<Grounding> {
        ground(&self.strokes, &self.arcs, &self.labels, &self.stated)
    }

    /// The kernel IR this scene reconstructs: the same graph, rebuilt.
    pub fn to_scene_graph(&self) -> anyhow::Result<SceneGraph> {
        Ok(self.grounding()?.graph)
    }

    /// The evidence ledger as a report: what the figure claimed, what is
    /// established, and what stayed open.
    pub fn report(&self) -> String {
        let mut lines = vec![format!(
            "figure: {} stroke(s), {} arc(s), {} label(s), {} premise(s)",
            self.strokes.len(),
            self.arcs.len(),
            self.labels.len(),
            self.stated.len()
        )];
        for hit in &self.intersections {
            lines.push(format!(
                "observed: {} of strokes {} and {} at ({}, {}) -- {}",
                hit.kind, hit.first, hit.second, hit.at.0, hit.at.1, hit.evidence
            ));
        }
        for miss in &self.near_misses {
            lines.push(format!(
                "refused: strokes {} and {} meet to within {} ({:?})",
                miss.first, miss.second, miss.distance, miss.reason
            ));
        }
        for fact in &self.scene.facts {
            if fact.is_established() {
                continue;
            }
            lines.push(format!(
                "supported, not established: {} (confidence {})",
                fact.constraint.describe(),
                fact.confidence
            ));
        }
        for line in &self.scene.not_established {
            lines.push(format!("not established: {line}"));
        }
        lines.join("\n")
    }
}

/// The topological pass: strokes, arcs and labels in; a [`Grounding`] out.
///
/// The order of the work is the order of the trust: place points (a placement
/// is not a claim), name them from labels (also not a claim), then write facts
/// -- stated premises through [`record_stated`], incidences through
/// [`record_visual`], and every measurable relation through [`suggest`]. The
/// last group can only ever reach the ledger.
pub fn reconstruct(
    strokes: &[Stroke],
    arcs: &[Arc],
    labels: &[LabelAssociation],
    stated: &[Constraint],
) -> anyhow::Result<GroundedScene> {
    GroundedScene::new(
        strokes.to_vec(),
        arcs.to_vec(),
        labels.to_vec(),
        stated.to_vec(),
    )
}

/// What a detected location turned out to be. It decides the name a figure
/// would give the point and nothing else -- a role is not a predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusterRole {
    /// A stroke end: a vertex of the figure.
    Endpoint,
    /// A point where two strokes cross. A figure rarely labels such a point, so
    /// it gets a name of its own that cannot be mistaken for a letter the
    /// problem used.
    Crossing,
    /// The centre of a detected arc.
    ArcCenter,
}

/// One welded point of the figure: where the detector first put it, which
/// lattice cell that became, and what the labels called it.
#[derive(Debug, Clone, PartialEq)]
struct Cluster {
    raw: (f64, f64),
    exact: (Q, Q),
    role: ClusterRole,
    /// Empty until a label or the name-mint supplies one.
    name: String,
}

/// Snap a detected coordinate onto the kernel's lattice: the nearest multiple
/// of `1/65536`, as an exact rational.
///
/// This is the module's only float-to-rational door, and it is a lossy one on
/// purpose -- bounded by half a cell, and named. A coordinate that is not a
/// number cannot be placed, so a detector that produced one is refused here
/// rather than allowed to poison the IR.
fn quantize(value: f64) -> anyhow::Result<Q> {
    anyhow::ensure!(
        value.is_finite(),
        "a detected coordinate is not a number: {value}"
    );
    let cells = (value * IMAGE_GRID as f64).round();
    anyhow::ensure!(
        cells.abs() < 1.0e15,
        "a detected coordinate is outside the image: {value}"
    );
    Q::new(cells as i128, IMAGE_GRID)
}

/// The cluster a detected location belongs to, if any: within [`WELD_TOLERANCE`]
/// in both coordinates. Chebyshev, not Euclidean, because a figure's ink merges
/// along an axis before it merges diagonally.
fn locate(clusters: &[Cluster], at: (f64, f64)) -> Option<usize> {
    clusters.iter().position(|cluster| {
        (cluster.raw.0 - at.0).abs() <= WELD_TOLERANCE
            && (cluster.raw.1 - at.1).abs() <= WELD_TOLERANCE
    })
}

/// The cluster for a detected location, welding it into an existing one when it
/// is the same ink and opening a new point when it is not.
fn claim(clusters: &mut Vec<Cluster>, at: (f64, f64), role: ClusterRole) -> anyhow::Result<usize> {
    if let Some(slot) = locate(clusters, at) {
        return Ok(slot);
    }
    let exact = (quantize(at.0)?, quantize(at.1)?);
    clusters.push(Cluster {
        raw: at,
        exact,
        role,
        name: String::new(),
    });
    Ok(clusters.len() - 1)
}

/// The name-mint. `V` for a vertex, `X` for a detected crossing, `O` for an arc
/// centre, `K` for a circle: letters a figure would use, prefixed so an
/// invented name can never be mistaken for one the problem gave. A name already
/// claimed by a label is skipped rather than reused.
#[derive(Debug, Default)]
struct Naming {
    vertex: usize,
    crossing: usize,
    centre: usize,
    circle: usize,
}

impl Naming {
    fn next_of(&mut self, role: ClusterRole) -> String {
        match role {
            ClusterRole::Endpoint => {
                self.vertex += 1;
                format!("V{}", self.vertex)
            }
            ClusterRole::Crossing => {
                self.crossing += 1;
                format!("X{}", self.crossing)
            }
            ClusterRole::ArcCenter => {
                self.centre += 1;
                format!("O{}", self.centre)
            }
        }
    }

    fn fresh(&mut self, role: ClusterRole, taken: &[String]) -> String {
        loop {
            let name = self.next_of(role);
            if !taken.contains(&name) {
                return name;
            }
        }
    }

    fn fresh_circle(&mut self, taken: &[String]) -> String {
        loop {
            self.circle += 1;
            let name = format!("K{}", self.circle);
            if !taken.contains(&name) {
                return name;
            }
        }
    }
}

/// The reconstruction, in the order of the trust.
///
/// 1. Every detected location is welded and placed on the lattice. A placement
///    is not a claim, which is why this can happen before anything is graded.
/// 2. Labels name the welded points they touch. A name is not a premise either:
///    the binding is a detection, so it is recorded on the ledger with its
///    confidence.
/// 3. Stated premises go in as premises -- the only route to confidence `1.0`.
/// 4. Every measurable relation between two strokes is *suggested*, and lands
///    on the ledger.
/// 5. Every detected incidence -- a crossing's collinearity, an endpoint
///    ordering, a junction that looks like the middle of a stroke -- is offered
///    to the kernel's exact predicate at its grade, and is admitted or refused
///    on its merits.
///
/// The pass is quadratic in the number of strokes, which is fine for a diagram
/// (tens of strokes) and named here rather than hidden: a real detector would
/// hand over its own adjacency, and the pairs worth comparing are the ones
/// sharing a junction.
#[allow(clippy::too_many_lines)]
fn ground(
    strokes: &[Stroke],
    arcs: &[Arc],
    labels: &[LabelAssociation],
    stated: &[Constraint],
) -> anyhow::Result<Grounding> {
    // A detector that produced a non-number has failed; refuse the figure
    // rather than let a NaN into the lattice.
    for (index, stroke) in strokes.iter().enumerate() {
        for at in [stroke.from, stroke.to] {
            anyhow::ensure!(
                at.0.is_finite() && at.1.is_finite(),
                "stroke {index} has a non-finite endpoint ({}, {})",
                at.0,
                at.1
            );
        }
    }
    for (index, arc) in arcs.iter().enumerate() {
        anyhow::ensure!(
            arc.radius.is_finite() && arc.radius > 0.0 && arc.sweep_degrees.is_finite(),
            "arc {index} needs a positive finite radius and a finite sweep, got r={} sweep={}",
            arc.radius,
            arc.sweep_degrees
        );
    }

    let found = intersections(strokes);
    let mut clusters: Vec<Cluster> = Vec::new();
    // Where each usable stroke's ends landed, which stroke that was, and a map
    // from stroke index to its position -- `usize::MAX` for a degenerate
    // stroke, which defines no direction and gets no segment.
    let mut ends: Vec<(usize, usize)> = Vec::new();
    let mut drawn: Vec<usize> = Vec::new();
    let mut seat: Vec<usize> = vec![usize::MAX; strokes.len()];
    for (index, stroke) in strokes.iter().enumerate() {
        if stroke.is_degenerate() {
            continue;
        }
        let from = claim(&mut clusters, stroke.from, ClusterRole::Endpoint)?;
        let to = claim(&mut clusters, stroke.to, ClusterRole::Endpoint)?;
        seat[index] = ends.len();
        ends.push((from, to));
        drawn.push(index);
    }
    let mut centres: Vec<usize> = Vec::with_capacity(arcs.len());
    for arc in arcs {
        centres.push(claim(&mut clusters, arc.center, ClusterRole::ArcCenter)?);
    }
    let mut junctions: Vec<usize> = Vec::with_capacity(found.len());
    for hit in &found {
        junctions.push(claim(&mut clusters, hit.at, ClusterRole::Crossing)?);
    }

    // (2) names, from the labels, with every binding accounted for.
    let mut taken: Vec<String> = Vec::new();
    let mut ledger: Vec<String> = Vec::new();
    for (index, stroke) in strokes.iter().enumerate() {
        if stroke.is_degenerate() {
            ledger.push(format!(
                "stroke {index} is degenerate: its ends coincide, so it defines no direction, no segment and no junction"
            ));
        }
    }
    for assoc in labels {
        match &assoc.entity {
            EntityRef::Point(wanted) => match locate(&clusters, assoc.label.at) {
                Some(slot) => {
                    let at = clusters[slot].raw;
                    if clusters[slot].name.is_empty() {
                        clusters[slot].name = wanted.clone();
                        taken.push(wanted.clone());
                        ledger.push(format!(
                            "label '{}' at ({}, {}) names the point at ({}, {}) '{}' (confidence {})",
                            assoc.label.text,
                            assoc.label.at.0,
                            assoc.label.at.1,
                            at.0,
                            at.1,
                            wanted,
                            assoc.confidence
                        ));
                    } else {
                        let held = clusters[slot].name.clone();
                        ledger.push(format!(
                            "label '{}' at ({}, {}) also claims the point at ({}, {}), already named '{}'",
                            assoc.label.text,
                            assoc.label.at.0,
                            assoc.label.at.1,
                            at.0,
                            at.1,
                            held
                        ));
                    }
                }
                None => ledger.push(format!(
                    "label '{}' at ({}, {}) names no detected point: nothing lies within the weld tolerance",
                    assoc.label.text,
                    assoc.label.at.0,
                    assoc.label.at.1
                )),
            },
            other => ledger.push(format!(
                "label '{}' names a {other}, an association the picture can suggest but not establish",
                assoc.label.text
            )),
        }
    }
    let mut naming = Naming::default();
    for cluster in clusters.iter_mut() {
        if cluster.name.is_empty() {
            cluster.name = naming.fresh(cluster.role, &taken);
            taken.push(cluster.name.clone());
        }
    }

    // (1) the welded points, placed on the lattice.
    let mut scene = crate::geomkernel::graph_from_points(Vec::new(), Vec::new());
    let mut points: Vec<KPoint> = Vec::with_capacity(clusters.len());
    for cluster in &clusters {
        let point = KPoint {
            name: cluster.name.clone(),
            x: Frac::from_q(cluster.exact.0),
            y: Frac::from_q(cluster.exact.1),
        };
        scene.add_point(point.clone())?;
        points.push(point);
    }

    // A closed arc is a circle object; a partial one is only a centre.
    for (index, arc) in arcs.iter().enumerate() {
        if !arc.is_closed() {
            ledger.push(format!(
                "arc {index} sweeps {} deg at ({}, {}): a centre, but no circle -- a partial arc determines no radius the kernel can hold",
                arc.sweep_degrees, arc.center.0, arc.center.1
            ));
            continue;
        }
        let centre = clusters[centres[index]].name.clone();
        let radius_sq = quantize(arc.radius * arc.radius)?;
        let circle = KCircle {
            name: naming.fresh_circle(&taken),
            center: centre.clone(),
            radius_sq: Frac::from_q(radius_sq),
        };
        scene.add_circle(circle.clone())?;
        record_visual(
            &mut scene,
            Constraint::Circle {
                name: circle.name,
                center: centre,
                radius_sq: circle.radius_sq,
            },
            &arc.evidence,
        );
    }

    // (3) the problem's premises: the only facts that may be established.
    for constraint in stated {
        record_stated(&mut scene, constraint.clone());
    }

    // (4) relations between strokes: suggestions, never facts.
    for left in 0..ends.len() {
        for right in (left + 1)..ends.len() {
            let first = &strokes[drawn[left]];
            let second = &strokes[drawn[right]];
            let evidence = Evidence::weakest(&first.evidence, &second.evidence);
            let pair = (
                Segment {
                    from: clusters[ends[left].0].name.clone(),
                    to: clusters[ends[left].1].name.clone(),
                },
                Segment {
                    from: clusters[ends[right].0].name.clone(),
                    to: clusters[ends[right].1].name.clone(),
                },
            );
            let difference = (first.angle_degrees() - second.angle_degrees())
                .abs()
                .rem_euclid(180.0);
            if difference.min(180.0 - difference) <= ANGLE_TOLERANCE_DEGREES {
                suggest(
                    &mut scene,
                    Constraint::Parallel {
                        first: pair.0.clone(),
                        second: pair.1.clone(),
                    },
                    &evidence,
                );
            }
            if (difference - 90.0).abs() <= ANGLE_TOLERANCE_DEGREES {
                suggest(
                    &mut scene,
                    Constraint::Perpendicular {
                        first: pair.0.clone(),
                        second: pair.1.clone(),
                    },
                    &evidence,
                );
            }
            let (one, other) = (first.length(), second.length());
            if (one - other).abs() <= LENGTH_TOLERANCE * one.max(other) {
                suggest(
                    &mut scene,
                    Constraint::EqualLength {
                        first: pair.0,
                        second: pair.1,
                    },
                    &evidence,
                );
            }
        }
    }

    // (5) the incidences each junction shows, at the junction's own grade.
    for (index, hit) in found.iter().enumerate() {
        let junction = clusters[junctions[index]].name.clone();
        for (stroke_index, along) in [(hit.first, hit.along_first), (hit.second, hit.along_second)]
        {
            // The junction *is* a stroke's endpoint when the parameter is at
            // either end: already one welded point, and that stroke has no
            // interior to speak of.
            if !(along > PARAMETER_EPSILON && along < 1.0 - PARAMETER_EPSILON) {
                continue;
            }
            let position = seat[stroke_index];
            if position == usize::MAX {
                continue;
            }
            let (from, to) = ends[position];
            let (a, b) = (clusters[from].name.clone(), clusters[to].name.clone());
            record_visual(
                &mut scene,
                Constraint::Collinear {
                    a: a.clone(),
                    b: junction.clone(),
                    c: b.clone(),
                },
                &hit.evidence,
            );
            record_visual(
                &mut scene,
                Constraint::Between {
                    a: a.clone(),
                    m: junction.clone(),
                    b: b.clone(),
                },
                &hit.evidence,
            );
            // A junction sitting in the middle of the stroke it splits is a
            // midpoint claim the figure is making. It goes to the exact
            // predicate like any other observation, so a wobbly figure that
            // only looks like a bisection is refused rather than believed.
            let whole = strokes[stroke_index].length();
            let (left, right) = (along * whole, (1.0 - along) * whole);
            if (left - right).abs() <= LENGTH_TOLERANCE * left.max(right) {
                record_visual(
                    &mut scene,
                    Constraint::MidpointOf {
                        p: junction.clone(),
                        a,
                        b,
                    },
                    &hit.evidence,
                );
            }
        }
    }

    for line in &ledger {
        leave_open(&mut scene, line);
    }
    let segments: Vec<Segment> = ends
        .iter()
        .map(|(from, to)| Segment {
            from: clusters[*from].name.clone(),
            to: clusters[*to].name.clone(),
        })
        .collect();
    Ok(Grounding {
        points,
        segments,
        intersections: found,
        graph: scene,
    })
}

// ---------------------------------------------------------------- rendering --

/// A coordinate as SVG text.
///
/// Rust's `{}` for `f64` is the shortest decimal that reads back as the same
/// bits, which is what makes the round trip exact rather than approximate. A
/// non-finite value cannot be drawn and is written as `0`; the detector that
/// produced it was already refused by [`reconstruct`], so this is a belt to a
/// pair of braces rather than a silent correction.
fn num(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "0".to_string()
    }
}

/// Escape a string for an XML attribute or text node.
fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

/// The inverse of [`xml_escape`]. An entity this module does not know is kept
/// verbatim rather than silently dropped, so a file with a numeric character
/// reference survives the round trip as text instead of as a hole.
fn xml_unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let mut decoded = None;
        for (entity, ch) in [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ] {
            if let Some(after) = rest.strip_prefix(entity) {
                decoded = Some((ch, after));
                break;
            }
        }
        match decoded {
            Some((ch, after)) => {
                out.push(ch);
                rest = after;
            }
            None => {
                let end = rest.find(';').map_or(rest.len(), |i| i + 1);
                out.push_str(&rest[..end]);
                rest = &rest[end..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The `data-evidence` attributes for an observation, so the evidence survives
/// the file. The grades are what make the round trip meaningful: a figure
/// re-read as perfectly certain is a figure that has quietly upgraded its own
/// wobble.
fn evidence_attributes(evidence: &Evidence) -> String {
    match evidence {
        Evidence::Exact => " data-evidence=\"exact\"".to_string(),
        Evidence::Measured { wobble, source } => format!(
            " data-evidence=\"measured\" data-wobble=\"{}\" data-source=\"{}\"",
            num(*wobble),
            xml_escape(source)
        ),
        Evidence::Asserted { source } => {
            format!(
                " data-evidence=\"asserted\" data-source=\"{}\"",
                xml_escape(source)
            )
        }
    }
}

/// Render the detections back to an SVG document.
///
/// Exact string building, no dependencies, and a real document: an XML
/// declaration, the SVG namespace, and balanced tags. Strokes become `<line>`,
/// arcs `<path>` (with the fitted centre, radius and sweep a reader needs),
/// labels `<text>` with the entity each one names.
///
/// The coordinates go out in the image's own frame, `y` increasing downwards,
/// which is the frame SVG draws in. Nothing in the round trip depends on the
/// orientation: the comparisons are angles, cross products and collinearities,
/// all of which a flip leaves alone.
pub fn render_svg(scene: &GroundedScene) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!(
        "<svg xmlns=\"{SVG_NAMESPACE}\" version=\"1.1\" viewBox=\"0 0 1 1\" width=\"512\" height=\"512\">\n"
    ));
    for (index, stroke) in scene.strokes.iter().enumerate() {
        out.push_str(&format!(
            "  <line id=\"s{index}\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#101010\" stroke-width=\"{}\"{}/>\n",
            num(stroke.from.0),
            num(stroke.from.1),
            num(stroke.to.0),
            num(stroke.to.1),
            num(stroke.thickness),
            evidence_attributes(&stroke.evidence)
        ));
    }
    for (index, arc) in scene.arcs.iter().enumerate() {
        let (start, end) = (arc.start(), arc.end());
        let large = i32::from(arc.sweep_degrees.abs() > 180.0);
        let clockwise = i32::from(arc.sweep_degrees < 0.0);
        out.push_str(&format!(
            "  <path id=\"a{index}\" d=\"M {} {} A {} {} 0 {large} {clockwise} {} {}\" fill=\"none\" stroke=\"#101010\" data-center-x=\"{}\" data-center-y=\"{}\" data-radius=\"{}\" data-sweep=\"{}\"{}/>\n",
            num(start.0),
            num(start.1),
            num(arc.radius),
            num(arc.radius),
            num(end.0),
            num(end.1),
            num(arc.center.0),
            num(arc.center.1),
            num(arc.radius),
            num(arc.sweep_degrees),
            evidence_attributes(&arc.evidence)
        ));
    }
    for (index, assoc) in scene.labels.iter().enumerate() {
        out.push_str(&format!(
            "  <text id=\"l{index}\" x=\"{}\" y=\"{}\" font-size=\"0.03\" data-entity=\"{}\"{}>{}</text>\n",
            num(assoc.label.at.0),
            num(assoc.label.at.1),
            xml_escape(&assoc.entity.encode()),
            evidence_attributes(&assoc.evidence),
            xml_escape(&assoc.label.text)
        ));
    }
    out.push_str("</svg>\n");
    out
}

/// One XML element, as much of it as this module needs.
#[derive(Debug, Clone, PartialEq)]
struct Element {
    name: String,
    attributes: Vec<(String, String)>,
    /// The text between the opening and closing tags, unescaped.
    text: String,
}

impl Element {
    fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }

    /// A numeric attribute. A missing or unparsable one is a malformed file,
    /// not a zero: a coordinate this module cannot read must not become the
    /// origin.
    fn number(&self, key: &str) -> anyhow::Result<f64> {
        let raw = self
            .attribute(key)
            .ok_or_else(|| anyhow::anyhow!("<{}> has no '{key}' attribute", self.name))?;
        let value: f64 = raw
            .parse()
            .map_err(|_| anyhow::anyhow!("<{}> has a non-numeric '{key}': '{raw}'", self.name))?;
        anyhow::ensure!(
            value.is_finite(),
            "<{}> has a non-finite '{key}': '{raw}'",
            self.name
        );
        Ok(value)
    }

    /// The observation this element was drawn with, from the `data-evidence`
    /// attributes. An element that says nothing about its evidence is read as
    /// an *assertion* -- a bare claim with no residual -- because reading an
    /// unlabelled element as exact is the one default that would let a file
    /// upgrade its own certainty.
    fn evidence(&self) -> anyhow::Result<Evidence> {
        let kind = self.attribute("data-evidence").unwrap_or("asserted");
        let source = self.attribute("data-source").unwrap_or("unknown detector");
        match kind {
            "exact" => Ok(Evidence::Exact),
            "measured" => Ok(Evidence::Measured {
                wobble: self.number("data-wobble")?,
                source: source.to_string(),
            }),
            "asserted" => Ok(Evidence::Asserted {
                source: source.to_string(),
            }),
            other => Err(anyhow::anyhow!(
                "unknown evidence kind '{other}' on <{}>",
                self.name
            )),
        }
    }
}

/// Every element of a document, in order, with nesting descended and the
/// `<text>` nodes' content kept.
///
/// A deliberately small reader: it understands the elements [`render_svg`]
/// writes, checks that every tag is closed by the tag it opened, and refuses a
/// document that is not well formed rather than guessing. It is a parser of this
/// module's own emitter, and the module docs say so -- what the round trip
/// proves is the topology and labelling path, not a vision model's accuracy.
fn elements(svg: &str) -> anyhow::Result<Vec<Element>> {
    let mut found: Vec<Element> = Vec::new();
    let mut open: Vec<String> = Vec::new();
    let mut rest = svg;
    while let Some(start) = rest.find('<') {
        rest = &rest[start..];
        if rest.starts_with("<?") || rest.starts_with("<!") {
            let end = rest
                .find('>')
                .ok_or_else(|| anyhow::anyhow!("an XML declaration or comment is unterminated"))?;
            rest = &rest[end + 1..];
            continue;
        }
        let close = rest
            .find('>')
            .ok_or_else(|| anyhow::anyhow!("an SVG tag is unterminated: {rest}"))?;
        let body = &rest[1..close];
        let after = &rest[close + 1..];
        if let Some(closing) = body.strip_prefix('/') {
            let name = closing.trim();
            anyhow::ensure!(
                open.pop().as_deref() == Some(name),
                "unbalanced SVG: </{name}> closes nothing"
            );
            rest = after;
            continue;
        }
        let empty = body.ends_with('/');
        let body = body.trim_end_matches('/');
        let mut parts = body.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or_default().to_string();
        anyhow::ensure!(!name.is_empty(), "an SVG tag has no name: <{body}>");
        let attributes = attributes_of(parts.next().unwrap_or(""))?;
        if empty {
            found.push(Element {
                name,
                attributes,
                text: String::new(),
            });
            rest = after;
            continue;
        }
        if name == "text" {
            // the one element whose content matters: the label it carries
            let tag = format!("</{name}>");
            let end = after
                .find(&tag)
                .ok_or_else(|| anyhow::anyhow!("an SVG <{name}> is never closed"))?;
            found.push(Element {
                name,
                attributes,
                text: xml_unescape(&after[..end]),
            });
            rest = &after[end + tag.len()..];
            continue;
        }
        open.push(name.clone());
        found.push(Element {
            name,
            attributes,
            text: String::new(),
        });
        rest = after;
    }
    anyhow::ensure!(
        open.is_empty(),
        "unbalanced SVG: {} tag(s) are never closed",
        open.len()
    );
    Ok(found)
}

/// `key="value"` pairs, in order. A bare or unquoted attribute is a malformed
/// file.
fn attributes_of(source: &str) -> anyhow::Result<Vec<(String, String)>> {
    let mut attributes = Vec::new();
    let mut rest = source.trim();
    while !rest.is_empty() {
        let (key, tail) = rest
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("an SVG attribute has no value: '{rest}'"))?;
        let quoted = tail.trim_start();
        let value_source = quoted
            .strip_prefix('"')
            .ok_or_else(|| anyhow::anyhow!("an SVG attribute value is not quoted: '{quoted}'"))?;
        let (value, tail) = value_source.split_once('"').ok_or_else(|| {
            anyhow::anyhow!("an SVG attribute value is unterminated: '{value_source}'")
        })?;
        attributes.push((key.trim().to_string(), xml_unescape(value)));
        rest = tail.trim_start();
    }
    Ok(attributes)
}

/// Re-read an SVG document as a [`GroundedScene`].
///
/// The inverse of [`render_svg`] over the elements that module writes, plus the
/// full pipeline: detections in, reconstructed graph out. Two things are worth
/// stating plainly. First, this is a parser of our own emitter, so it validates
/// the topology and labelling path and nothing about a real vision model's
/// accuracy. Second, what it reads is the *evidence* as well as the geometry --
/// the `data-evidence` attributes carry the wobble through the file, so a
/// figure cannot come back more certain than it went out. A label element must
/// say which entity it names (`data-entity`), because which point a letter
/// belongs to is part of what the round trip has to reproduce; a pipeline
/// testing its *binding* step would drop that attribute and let
/// [`reconstruct`] re-bind each label to the nearest welded point, which is
/// what a real one has to do.
pub fn parse_svg(svg: &str) -> anyhow::Result<GroundedScene> {
    let elements = elements(svg)?;
    anyhow::ensure!(
        elements.first().is_some_and(|root| root.name == "svg"),
        "the document has no <svg> root"
    );
    let mut strokes: Vec<Stroke> = Vec::new();
    let mut arcs: Vec<Arc> = Vec::new();
    let mut labels: Vec<LabelAssociation> = Vec::new();
    for element in &elements {
        match element.name.as_str() {
            "line" => strokes.push(Stroke {
                from: (element.number("x1")?, element.number("y1")?),
                to: (element.number("x2")?, element.number("y2")?),
                thickness: element.number("stroke-width")?,
                evidence: element.evidence()?,
            }),
            "path" => arcs.push(Arc {
                center: (
                    element.number("data-center-x")?,
                    element.number("data-center-y")?,
                ),
                radius: element.number("data-radius")?,
                sweep_degrees: element.number("data-sweep")?,
                evidence: element.evidence()?,
            }),
            "text" => {
                let anchor = (element.number("x")?, element.number("y")?);
                let encoded = element.attribute("data-entity").ok_or_else(|| {
                    anyhow::anyhow!("a label element must say which entity it names")
                })?;
                let evidence = element.evidence()?;
                // The binding's confidence is the evidence's own grade, so the
                // file has one source of truth rather than two that can drift.
                labels.push(LabelAssociation::new(
                    Label::new(&element.text, anchor),
                    EntityRef::decode(encoded)?,
                    evidence,
                ));
            }
            _ => {}
        }
    }
    GroundedScene::new(strokes, arcs, labels, Vec::new())
}

/// The grounding test, as a function: render the figure, read it back, and
/// reconstruct the graph from the re-read pixels.
///
/// A figure rendered and then re-grounded must come back as the graph it came
/// from -- same points, same segments, same labels, same facts and the same
/// ledger. That is the check this module exists for: a topological pass that
/// loses a junction, double counts a shared endpoint, or invents a point on the
/// way through fails here and nowhere else.
///
/// The problem's premises are re-attached rather than read back, because a
/// figure does not carry its statement: the ink went out, the words did not.
/// Everything the *picture* asserted comes back on its own.
pub fn round_trip(scene: &GroundedScene) -> anyhow::Result<GroundedScene> {
    let regrounded = parse_svg(&render_svg(scene))?;
    reconstruct(
        &regrounded.strokes,
        &regrounded.arcs,
        &regrounded.labels,
        &scene.stated,
    )
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
    use approx::assert_relative_eq;

    /// A detector's measured stroke, with the residual it reported.
    fn measured(wobble: f64) -> Evidence {
        Evidence::Measured {
            wobble,
            source: "hough-grid".to_string(),
        }
    }

    /// A detector's bare categorical claim.
    fn asserted() -> Evidence {
        Evidence::Asserted {
            source: "tesseract".to_string(),
        }
    }

    fn stroke(from: (f64, f64), to: (f64, f64), evidence: Evidence) -> Stroke {
        Stroke::new(from, to, 0.004, evidence)
    }

    fn point_label(text: &str, at: (f64, f64), name: &str) -> LabelAssociation {
        LabelAssociation::new(
            Label::new(text, at),
            EntityRef::Point(name.to_string()),
            asserted(),
        )
    }

    fn ground_bare(strokes: Vec<Stroke>) -> anyhow::Result<GroundedScene> {
        GroundedScene::new(strokes, Vec::new(), Vec::new(), Vec::new())
    }

    /// Two nearly parallel strokes: the case the module is about.
    fn wobbly_pair() -> Vec<Stroke> {
        vec![
            stroke((0.2, 0.2), (0.8, 0.2), measured(0.4)),
            stroke((0.2, 0.6), (0.8, 0.6042), measured(0.4)),
        ]
    }

    /// A crossing on the exact lattice: a horizontal and a vertical stroke
    /// meeting at (1/2, 1/2), which the kernel can certify.
    fn exact_crossing(evidence: Evidence) -> Vec<Stroke> {
        vec![
            stroke((0.25, 0.5), (0.75, 0.5), evidence.clone()),
            stroke((0.5, 0.25), (0.5, 0.75), evidence),
        ]
    }

    #[test]
    fn test_exact_evidence_is_the_only_trustworthy_evidence() {
        assert!(Evidence::Exact.is_trustworthy());
        assert_eq!(Evidence::Exact.confidence(), 1.0);
        assert_eq!(Evidence::Exact.logical_state(), "ESTABLISHED");
        // A measurement is untrustworthy at every wobble, including none at all,
        // and so is a bare categorical claim.
        for evidence in [
            measured(0.0),
            measured(0.4),
            measured(4.0),
            asserted(),
            Evidence::graded(0.83, "grid"),
        ] {
            assert!(
                !evidence.is_trustworthy(),
                "{evidence} must not establish a predicate"
            );
            assert_eq!(
                evidence.logical_state(),
                "UNKNOWN",
                "{evidence} must leave the claim open"
            );
        }
    }

    #[test]
    fn test_measured_grades_stay_strictly_below_one() {
        // a perfect measurement still tops out strictly below the kernel's
        // certainty, because a detector has a systematic error the picture
        // does not contain
        assert_eq!(measured(0.0).confidence(), MEASURED_CEILING);
        assert!(measured(0.0).confidence() < 1.0);
        assert!(measured(1.0e-9).confidence() < 1.0);
        assert_relative_eq!(
            measured(0.4).confidence(),
            MEASURED_CEILING / 1.4,
            epsilon = 1.0e-12
        );
        // hyperbolic: a bigger residual is always worth less
        assert!(measured(0.4).confidence() < measured(0.2).confidence());
        assert!(measured(4.0).confidence() < measured(0.4).confidence());
        assert_relative_eq!(
            measured(4.0).confidence(),
            MEASURED_CEILING / 5.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn test_graded_inverts_the_grade_rule() {
        let evidence = Evidence::graded(0.83, "hough-grid");
        assert_relative_eq!(evidence.confidence(), 0.83, epsilon = 1.0e-12);
        assert!(!evidence.is_trustworthy());
        // asking for more than the ceiling gets the ceiling, not a certainty
        assert_eq!(Evidence::graded(2.0, "x").confidence(), MEASURED_CEILING);
        assert_eq!(Evidence::graded(0.0, "x").confidence(), 1.0e-3);
    }

    #[test]
    fn test_asserted_evidence_is_flat_and_below_any_measurement() {
        assert_eq!(asserted().confidence(), ASSERTED_CONFIDENCE);
        // A measurement that bounds its own error outranks a claim with no
        // residual -- until it is so wobbly that the bound is worthless, which
        // is the honest crossover rather than a claim that a number always
        // beats a word.
        assert!(measured(0.4).confidence() > asserted().confidence());
        assert!(measured(4.0).confidence() < asserted().confidence());
        assert_eq!(asserted().source(), "tesseract");
        assert_eq!(Evidence::Exact.source(), "the problem statement");
    }

    #[test]
    fn test_weakest_evidence_combines_two_observations() {
        let wobbles = Evidence::weakest(&measured(0.4), &measured(0.2));
        assert_relative_eq!(
            wobbles.confidence(),
            MEASURED_CEILING / 1.6,
            epsilon = 1.0e-12
        );
        assert_eq!(wobbles.source(), "hough-grid+hough-grid");
        assert_eq!(
            Evidence::weakest(&Evidence::Exact, &measured(0.4)),
            measured(0.4)
        );
        assert_eq!(
            Evidence::weakest(&measured(0.4), &Evidence::Exact),
            measured(0.4)
        );
        assert_eq!(
            Evidence::weakest(&Evidence::Exact, &Evidence::Exact),
            Evidence::Exact
        );
        // a categorical detector in the chain leaves nothing to add up
        assert_eq!(
            Evidence::weakest(&asserted(), &measured(0.4)),
            Evidence::Asserted {
                source: "tesseract+hough-grid".to_string()
            }
        );
    }

    #[test]
    fn test_evidence_reports_evidence_and_state_separately() {
        let report = measured(0.4).report("parallel(AB, CD)");
        assert_eq!(
            report,
            "visual evidence: approximately parallel(AB, CD) (0.4 deg of wobble, from hough-grid)\nlogical state:   UNKNOWN"
        );
        // the two halves are separate on purpose: the evidence line is what the
        // picture said, the state is what it did not license
        assert!(!report
            .lines()
            .next()
            .unwrap_or_default()
            .contains("logical state"));
    }

    #[test]
    fn test_a_nearly_parallel_pair_never_becomes_a_parallel_fact() -> anyhow::Result<()> {
        let scene = ground_bare(wobbly_pair())?;
        let names = scene.grounding()?.point_names();
        assert_eq!(
            names,
            vec![
                "V1".to_string(),
                "V2".to_string(),
                "V3".to_string(),
                "V4".to_string()
            ]
        );
        let claim = Constraint::Parallel {
            first: Segment {
                from: names[0].clone(),
                to: names[1].clone(),
            },
            second: Segment {
                from: names[2].clone(),
                to: names[3].clone(),
            },
        };
        assert!(
            !scene.scene.has_fact(&claim),
            "0.14 degrees of wobble is not a premise"
        );
        let ledger = &scene.scene.not_established;
        assert!(
            ledger
                .iter()
                .any(|line| line.contains("suggested: parallel(")
                // the two strokes' wobbles add: a claim resting on both is
                // worth the sum, which is 0.4 + 0.4 here
                && line.contains("0.8 deg of wobble")
                && line.contains("logical state:   UNKNOWN")),
            "the suggestion must name its measurement: {ledger:?}"
        );
        assert!(scene.scene.facts.iter().all(|fact| !fact.is_established()));
        Ok(())
    }

    #[test]
    fn test_an_exactly_parallel_pair_is_still_only_suggested() -> anyhow::Result<()> {
        // the rule is about the *kind* of evidence, not about an epsilon: even
        // ink that is exactly parallel cannot assert the relation, because then
        // the picture would be supplying the problem's premise
        let scene = ground_bare(vec![
            stroke((0.2, 0.2), (0.8, 0.2), Evidence::Exact),
            stroke((0.2, 0.6), (0.8, 0.6), Evidence::Exact),
        ])?;
        let claim = Constraint::Parallel {
            first: Segment {
                from: "V1".to_string(),
                to: "V2".to_string(),
            },
            second: Segment {
                from: "V3".to_string(),
                to: "V4".to_string(),
            },
        };
        assert!(!scene.scene.has_fact(&claim));
        assert!(scene
            .scene
            .not_established
            .iter()
            .any(|line| line.contains("suggested: parallel(")));
        assert!(
            scene.scene.facts.is_empty(),
            "an exact figure still states nothing by itself"
        );
        Ok(())
    }

    #[test]
    fn test_perpendicular_and_equal_length_are_suggested_too() -> anyhow::Result<()> {
        let scene = ground_bare(exact_crossing(measured(0.3)))?;
        let ledger = &scene.scene.not_established;
        assert!(
            ledger.iter().any(|line| line.contains("suggested: perp(")),
            "{ledger:?}"
        );
        assert!(
            ledger.iter().any(|line| line.contains("suggested: len(")),
            "{ledger:?}"
        );
        assert!(scene.scene.facts.iter().all(|fact| !fact.is_established()));
        Ok(())
    }

    #[test]
    fn test_a_crossing_is_a_crossing() {
        let strokes = exact_crossing(measured(0.2));
        let found = intersections(&strokes);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, IntersectionKind::Crossing);
        assert_eq!(found[0].at, (0.5, 0.5));
        assert_eq!(found[0].first, 0);
        assert_eq!(found[0].second, 1);
        // a crossing is interior to both strokes: that is what makes it one
        assert_eq!(found[0].along_first, 0.5);
        assert_eq!(found[0].along_second, 0.5);
        assert_eq!(
            found[0].evidence.confidence(),
            Evidence::weakest(&measured(0.2), &measured(0.2)).confidence()
        );
    }

    #[test]
    fn test_a_t_junction_is_not_a_crossing() -> anyhow::Result<()> {
        // a stem that stops on a through-line: the same two segments as the
        // crossing case, with one endpoint moved onto the middle of the other
        let strokes = vec![
            stroke((0.25, 0.5), (0.75, 0.5), measured(0.2)),
            stroke((0.5, 0.25), (0.5, 0.5), measured(0.2)),
        ];
        let found = intersections(&strokes);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, IntersectionKind::TJunction);
        assert_eq!(found[0].at, (0.5, 0.5));
        // the through-line is split; the stem ends there
        assert_eq!(found[0].along_first, 0.5);
        assert_eq!(found[0].along_second, 1.0);
        // and the T is one point of the figure: the stem's end welds into the
        // junction rather than opening a second one
        let grounding = ground_bare(strokes)?.grounding()?;
        assert_eq!(grounding.points.len(), 4);
        assert_eq!(grounding.intersections[0].kind, IntersectionKind::TJunction);
        Ok(())
    }

    #[test]
    fn test_a_shared_endpoint_is_reported_once() -> anyhow::Result<()> {
        let strokes = vec![
            stroke((0.25, 0.5), (0.5, 0.5), measured(0.2)),
            stroke((0.5, 0.5), (0.75, 0.25), measured(0.2)),
        ];
        let found = intersections(&strokes);
        assert_eq!(found.len(), 1, "one pair, one junction -- not two");
        assert_eq!(found[0].kind, IntersectionKind::SharedEndpoint);
        assert_eq!(found[0].at, (0.5, 0.5));
        assert_eq!(found[0].along_first, 1.0);
        assert_eq!(found[0].along_second, 0.0);
        // the shared vertex is welded into one point of the figure
        let grounding = ground_bare(strokes)?.grounding()?;
        assert_eq!(grounding.points.len(), 3);
        assert_eq!(grounding.segments.len(), 2);
        Ok(())
    }

    #[test]
    fn test_three_strokes_at_one_point_make_one_point() -> anyhow::Result<()> {
        let strokes = vec![
            stroke((0.25, 0.5), (0.75, 0.5), measured(0.2)),
            stroke((0.5, 0.25), (0.5, 0.75), measured(0.2)),
            stroke((0.5, 0.5), (0.75, 0.75), measured(0.2)),
        ];
        let scene = GroundedScene::new(strokes, Vec::new(), Vec::new(), Vec::new())?;
        // three pairs meet at (1/2, 1/2) and all three are reported -- pairwise
        // reporting is what a consumer needs -- but they are one point
        assert_eq!(scene.intersections.len(), 3);
        assert!(scene.intersections.iter().all(|hit| hit.at == (0.5, 0.5)));
        let grounding = scene.grounding()?;
        assert_eq!(grounding.points.len(), 6);
        assert_eq!(
            scene.near_misses.len(),
            0,
            "every pair here really does meet"
        );
        Ok(())
    }

    #[test]
    fn test_proximity_is_not_incidence() -> anyhow::Result<()> {
        // two horizontal strokes a 512th of the figure apart: closer than the
        // proximity tolerance, and therefore not a junction
        let gap = 1.0 / 512.0;
        let strokes = vec![
            stroke((0.2, 0.5), (0.8, 0.5), measured(0.2)),
            stroke((0.2, 0.5 + gap), (0.8, 0.5 + gap), measured(0.2)),
        ];
        assert!(intersections(&strokes).is_empty());
        let misses = near_misses(&strokes);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].reason, NearMissReason::Proximity);
        assert_eq!(misses[0].distance, gap);
        // and the reconstruction invents no junction: the welder even collapses
        // the two lines' ends into one pair of points, because at that
        // separation the ink is one line drawn twice
        let scene = ground_bare(strokes)?;
        let grounding = scene.grounding()?;
        assert!(grounding.intersections.is_empty());
        assert_eq!(grounding.points.len(), 2);
        assert_eq!(scene.near_misses.len(), 1);
        Ok(())
    }

    #[test]
    fn test_a_grazing_crossing_is_refused() -> anyhow::Result<()> {
        // the two strokes do cross, but at about a two-thousandth of a radian:
        // the crossing point is a function of the wobble, not of the geometry
        let strokes = vec![
            stroke((0.2, 0.5), (0.8, 0.5), measured(0.05)),
            stroke((0.2, 0.4999), (0.8, 0.5001), measured(0.05)),
        ];
        assert!(
            intersections(&strokes).is_empty(),
            "a graze is not a junction"
        );
        let misses = near_misses(&strokes);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].reason, NearMissReason::Shallow);
        assert!(misses[0].distance < 1.0e-3);
        assert!(ground_bare(strokes)?.grounding()?.intersections.is_empty());
        Ok(())
    }

    #[test]
    fn test_a_degenerate_stroke_is_reported_not_a_junction() -> anyhow::Result<()> {
        let strokes = vec![
            stroke((0.4, 0.4), (0.6, 0.6), measured(0.2)),
            stroke((0.5, 0.5), (0.5, 0.5), measured(0.2)),
        ];
        assert!(intersections(&strokes).is_empty());
        let misses = near_misses(&strokes);
        assert_eq!(misses.len(), 1);
        assert_eq!(misses[0].reason, NearMissReason::Degenerate);
        let scene = ground_bare(strokes)?;
        let grounding = scene.grounding()?;
        // a stroke whose ends coincide contributes no segment and no point
        assert_eq!(grounding.segments.len(), 1);
        assert_eq!(grounding.points.len(), 2);
        assert!(scene
            .scene
            .not_established
            .iter()
            .any(|line| line.contains("stroke 1 is degenerate")));
        Ok(())
    }

    #[test]
    fn test_labels_name_the_points_they_touch() -> anyhow::Result<()> {
        let strokes = vec![
            stroke((0.25, 0.25), (0.75, 0.25), measured(0.2)),
            stroke((0.75, 0.25), (0.75, 0.75), measured(0.2)),
            stroke((0.75, 0.75), (0.25, 0.75), measured(0.2)),
        ];
        let labels = vec![
            point_label("A", (0.25, 0.25), "A"),
            point_label("B", (0.75, 0.25), "B"),
            point_label("C", (0.75, 0.75), "C"),
        ];
        let scene = GroundedScene::new(strokes, Vec::new(), labels, Vec::new())?;
        let names = scene.grounding()?.point_names();
        // three sides, four corners: the three the labels reach are named by the
        // figure, the fourth by the mint
        assert_eq!(
            names,
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "V1".to_string()
            ]
        );
        // the binding is recorded as what it is: a detection, with its grade
        let ledger = &scene.scene.not_established;
        assert!(
            ledger
                .iter()
                .any(|line| line.contains("label 'A' at (0.25, 0.25) names the point")),
            "{ledger:?}"
        );
        // and a corner nobody labelled is still named, by the mint
        let scene = GroundedScene::new(
            scene.strokes.clone(),
            Vec::new(),
            vec![point_label("A", (0.25, 0.25), "A")],
            Vec::new(),
        )?;
        let names = scene.grounding()?.point_names();
        assert_eq!(
            names,
            vec![
                "A".to_string(),
                "V1".to_string(),
                "V2".to_string(),
                "V3".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn test_a_label_that_names_nothing_stays_open() -> anyhow::Result<()> {
        let scene = GroundedScene::new(
            vec![stroke((0.25, 0.25), (0.75, 0.25), measured(0.2))],
            Vec::new(),
            vec![point_label("Z", (0.95, 0.95), "Z")],
            Vec::new(),
        )?;
        // the name is not invented, and the failed binding is on the ledger
        let names = scene.grounding()?.point_names();
        assert_eq!(names, vec!["V1".to_string(), "V2".to_string()]);
        assert!(scene
            .scene
            .not_established
            .iter()
            .any(|line| { line.contains("label 'Z' at (0.95, 0.95) names no detected point") }));
        Ok(())
    }

    #[test]
    fn test_a_label_naming_a_segment_goes_to_the_ledger() -> anyhow::Result<()> {
        // "AB" beside a side of a triangle is an association about a *relation*,
        // which a picture can suggest and only the problem can state
        let assoc = LabelAssociation::new(
            Label::new("AB", (0.3, 0.2)),
            EntityRef::Segment {
                from: "A".to_string(),
                to: "B".to_string(),
            },
            asserted(),
        );
        let scene = GroundedScene::new(
            vec![stroke((0.25, 0.25), (0.75, 0.25), measured(0.2))],
            Vec::new(),
            vec![assoc],
            Vec::new(),
        )?;
        let names = scene.grounding()?.point_names();
        assert_eq!(names, vec!["V1".to_string(), "V2".to_string()]);
        let ledger = &scene.scene.not_established;
        assert!(
            ledger
                .iter()
                .any(|line| line.contains("label 'AB' names a segment AB")),
            "{ledger:?}"
        );
        assert!(
            !ledger.iter().any(|line| line.contains("names the point")),
            "a segment label names no point"
        );
        Ok(())
    }

    #[test]
    fn test_a_stated_parallelism_is_the_only_established_fact() -> anyhow::Result<()> {
        let claim = Constraint::Parallel {
            first: Segment {
                from: "A".to_string(),
                to: "B".to_string(),
            },
            second: Segment {
                from: "C".to_string(),
                to: "D".to_string(),
            },
        };
        let scene = GroundedScene::new(
            vec![
                stroke((0.2, 0.2), (0.8, 0.2), measured(0.4)),
                stroke((0.2, 0.6), (0.8, 0.6), measured(0.4)),
            ],
            Vec::new(),
            vec![
                point_label("A", (0.2, 0.2), "A"),
                point_label("B", (0.8, 0.2), "B"),
                point_label("C", (0.2, 0.6), "C"),
                point_label("D", (0.8, 0.6), "D"),
            ],
            vec![claim.clone()],
        )?;
        assert!(scene.scene.has_fact(&claim));
        let fact = scene
            .scene
            .facts
            .iter()
            .find(|fact| fact.constraint == claim)
            .ok_or_else(|| anyhow::anyhow!("the stated premise is missing from the ledger"))?;
        assert_eq!(fact.confidence, 1.0);
        assert!(fact.is_established());
        assert!(matches!(fact.provenance, Provenance::Given));
        assert_eq!(scene.scene.confidence_in(&claim)?, 1.0);
        // the same ink still only *suggests* it, at a grade, on the ledger
        assert!(scene
            .scene
            .not_established
            .iter()
            .any(|line| line.contains("suggested: parallel(AB, CD)")));
        Ok(())
    }

    #[test]
    fn test_a_premise_the_figure_contradicts_is_refused() -> anyhow::Result<()> {
        // the statement says the two lines are parallel; the drawing says they
        // are not. The statement is the premise and the drawing is evidence, so
        // the premise is refused *by the kernel's own exact predicate* and the
        // disagreement is reported rather than papered over
        let claim = Constraint::Parallel {
            first: Segment {
                from: "A".to_string(),
                to: "B".to_string(),
            },
            second: Segment {
                from: "C".to_string(),
                to: "D".to_string(),
            },
        };
        let scene = GroundedScene::new(
            vec![
                stroke((0.2, 0.2), (0.8, 0.2), measured(0.4)),
                stroke((0.2, 0.6), (0.8, 0.64), measured(0.4)),
            ],
            Vec::new(),
            vec![
                point_label("A", (0.2, 0.2), "A"),
                point_label("B", (0.8, 0.2), "B"),
                point_label("C", (0.2, 0.6), "C"),
                point_label("D", (0.8, 0.64), "D"),
            ],
            vec![claim.clone()],
        )?;
        assert!(
            !scene.scene.has_fact(&claim),
            "the kernel refuses a false premise"
        );
        assert!(scene.scene.facts.iter().all(|fact| !fact.is_established()));
        let ledger = &scene.scene.not_established;
        assert!(
            ledger.iter().any(
                |line| line.contains("refused: parallel(AB, CD) is false of the stated figure")
            ),
            "{ledger:?}"
        );
        Ok(())
    }

    #[test]
    fn test_exact_incidence_is_graded_never_established() -> anyhow::Result<()> {
        // the crossing at (1/2, 1/2) is *exactly* collinear with both strokes on
        // the lattice, so the kernel admits the incidence -- at a grade
        let scene = ground_bare(exact_crossing(measured(0.4)))?;
        let incidence = Constraint::Collinear {
            a: "V1".to_string(),
            b: "X1".to_string(),
            c: "V2".to_string(),
        };
        assert!(
            scene.scene.has_fact(&incidence),
            "the exact predicate admits it"
        );
        let fact = scene
            .scene
            .facts
            .iter()
            .find(|fact| fact.constraint == incidence)
            .ok_or_else(|| anyhow::anyhow!("the graded incidence is missing"))?;
        // graded at the junction's evidence: the two strokes' wobbles add
        assert_relative_eq!(fact.confidence, MEASURED_CEILING / 1.8, epsilon = 1.0e-12);
        assert!(!fact.is_established());
        assert!(!scene.scene.facts.iter().any(|fact| fact.is_established()));
        // the ordering and the bisection the junction shows are graded the same
        // way, and the exact predicate certified all three
        assert!(scene.scene.has_fact(&Constraint::Between {
            a: "V1".to_string(),
            m: "X1".to_string(),
            b: "V2".to_string()
        }));
        assert!(scene.scene.has_fact(&Constraint::MidpointOf {
            p: "X1".to_string(),
            a: "V1".to_string(),
            b: "V2".to_string()
        }));
        Ok(())
    }

    #[test]
    fn test_a_wobbly_crossing_is_refused_by_the_exact_predicate() -> anyhow::Result<()> {
        // a tilted through-line and an exactly vertical stem that cross: the
        // detector saw a junction, and the kernel splits that junction's claims
        // in two -- the vertical half is exactly collinear and is admitted at a
        // grade, the tilted half is not, and it is refused rather than rounded
        // into place
        let scene = ground_bare(vec![
            stroke((0.25, 0.5), (0.75, 0.52), measured(0.1)),
            stroke((0.5, 0.25), (0.5, 0.75), measured(0.1)),
        ])?;
        assert_eq!(
            scene.intersections.len(),
            1,
            "the detector did see a crossing"
        );
        assert_eq!(scene.intersections[0].kind, IntersectionKind::Crossing);
        // the junction is still a point of the figure -- a placement is not a
        // claim -- but the incidence the wobble spoils is not asserted
        assert!(scene.scene.point("X1").is_ok());
        let spoiled = Constraint::Collinear {
            a: "V1".to_string(),
            b: "X1".to_string(),
            c: "V2".to_string(),
        };
        let exact_half = Constraint::Collinear {
            a: "V3".to_string(),
            b: "X1".to_string(),
            c: "V4".to_string(),
        };
        assert!(!scene.scene.has_fact(&spoiled));
        assert!(scene.scene.has_fact(&exact_half));
        assert!(scene.scene.facts.iter().all(|fact| !fact.is_established()));
        assert!(
            scene
                .scene
                .not_established
                .iter()
                .any(|line| line
                    .contains("refused: collinear(V1,X1,V2) is false of the stated figure")),
            "{:?}",
            scene.scene.not_established
        );
        Ok(())
    }

    #[test]
    fn test_confidence_propagates_along_a_dependency_chain() -> anyhow::Result<()> {
        // one stroke read off an exact IR, one measured at 0.83: the junction
        // inherits the weaker grade, and everything derived from it is worth no
        // more than that
        let scene = GroundedScene::new(
            vec![
                stroke((0.25, 0.5), (0.75, 0.5), Evidence::Exact),
                stroke(
                    (0.5, 0.25),
                    (0.5, 0.75),
                    Evidence::graded(0.83, "hough-grid"),
                ),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        let mut graph = scene.scene.clone();
        let midpoint = Constraint::MidpointOf {
            p: "X1".to_string(),
            a: "V1".to_string(),
            b: "V2".to_string(),
        };
        assert!(graph.has_fact(&midpoint));
        let graded = graph
            .facts
            .iter()
            .find(|fact| fact.constraint == midpoint)
            .ok_or_else(|| anyhow::anyhow!("the graded midpoint is missing"))?;
        assert_relative_eq!(graded.confidence, 0.83, epsilon = 1.0e-12);
        // the kernel's own rule fires off that graded premise...
        assert!(saturate(&mut graph, 4)? >= 1);
        // ...and the conclusion it licenses is worth exactly the grade, however
        // exact the rule that produced it
        let conclusion = Constraint::EqualLength {
            first: Segment {
                from: "V1".to_string(),
                to: "X1".to_string(),
            },
            second: Segment {
                from: "X1".to_string(),
                to: "V2".to_string(),
            },
        };
        assert!(graph.has_fact(&conclusion));
        let worth = graph.confidence_in(&conclusion)?;
        // the comparison is exact up to one rounding of a float: a grade of
        // 0.83 comes back as 0.8300000000000001, never as more than itself
        assert!(
            worth <= 0.83 + 1.0e-12,
            "a conclusion cannot be worth more than its premise: {worth}"
        );
        assert_relative_eq!(worth, 0.83, epsilon = 1.0e-12);
        // and the grade lives in the *chain*, not in the derived fact: the rule
        // that produced the conclusion was exact, and only the walk recovers
        // the weakness underneath it
        let derived = graph
            .facts
            .iter()
            .find(|fact| fact.constraint == conclusion)
            .ok_or_else(|| anyhow::anyhow!("the derived conclusion is missing"))?;
        assert_eq!(derived.confidence, 1.0);
        assert!(derived.is_established());
        Ok(())
    }

    #[test]
    fn test_a_measured_figure_establishes_nothing() -> anyhow::Result<()> {
        // the whole point, checked end to end: a measured triangle with a
        // measured median, saturated. Note what the rules are allowed to do --
        // the parallelogram rule fires off the two graded midpoints and derives
        // a *parallelism* at confidence 1.0, because the rule is exact. The
        // figure established nothing: the rule did, and the chain is what says
        // what it is worth.
        let scene = GroundedScene::new(
            vec![
                stroke((0.25, 0.25), (0.75, 0.25), measured(0.3)),
                stroke((0.75, 0.25), (0.75, 0.75), measured(0.3)),
                stroke((0.25, 0.25), (0.75, 0.75), measured(0.3)),
                stroke((0.5, 0.25), (0.5, 0.75), measured(0.3)),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        // the median crosses the hypotenuse at (1/2, 1/2), exactly on both
        assert!(scene
            .intersections
            .iter()
            .any(|hit| hit.at == (0.5, 0.5) && hit.kind == IntersectionKind::Crossing));
        let mut graph = scene.scene.clone();
        saturate(&mut graph, 6)?;
        assert!(!graph.facts.is_empty(), "the figure did produce incidences");
        // no fact in a measured figure is a premise
        assert!(graph.assumptions().is_empty());
        // and nothing in it is worth full certainty, however it was produced:
        // the walk down the dependency chain is the number that matters
        for fact in &graph.facts {
            let worth = graph.confidence_in(&fact.constraint)?;
            assert!(
                worth < 1.0,
                "{} is worth {worth}",
                fact.constraint.describe()
            );
        }
        assert!(graph
            .facts
            .iter()
            .any(|fact| matches!(fact.provenance, Provenance::Derived { .. })));
        Ok(())
    }

    /// A labelled square with a diagonal, a median and a closed circle: corners,
    /// a crossing, shared endpoints, a label on every corner, and an arc -- the
    /// whole pipeline in one figure. Every coordinate is a dyadic rational, so
    /// the round trip has nothing to hide behind. The median meets the diagonal
    /// at the circle's centre, so that one point is simultaneously a crossing
    /// and an arc centre, and the weld has to get that right in both
    /// directions.
    fn worked_example() -> anyhow::Result<GroundedScene> {
        let strokes = vec![
            stroke((0.25, 0.25), (0.75, 0.25), measured(0.2)),
            stroke((0.75, 0.25), (0.75, 0.75), measured(0.2)),
            stroke((0.75, 0.75), (0.25, 0.75), measured(0.2)),
            stroke((0.25, 0.75), (0.25, 0.25), measured(0.2)),
            stroke((0.25, 0.25), (0.75, 0.75), Evidence::Exact),
            stroke((0.5, 0.25), (0.5, 0.75), measured(0.2)),
        ];
        let labels = vec![
            point_label("A", (0.25, 0.25), "A"),
            point_label("B", (0.75, 0.25), "B"),
            point_label("C", (0.75, 0.75), "C"),
            point_label("D", (0.25, 0.75), "D"),
        ];
        let arcs = vec![Arc {
            center: (0.5, 0.5),
            radius: 0.125,
            sweep_degrees: 360.0,
            evidence: measured(0.3),
        }];
        GroundedScene::new(strokes, arcs, labels, Vec::new())
    }

    #[test]
    fn test_the_svg_is_well_formed() -> anyhow::Result<()> {
        let svg = render_svg(&worked_example()?);
        // an XML declaration, then the SVG namespace: a document that calls
        // itself SVG and does not declare it is not SVG
        assert!(
            svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"),
            "{svg}"
        );
        assert!(svg.contains(&format!("xmlns=\"{SVG_NAMESPACE}\"")), "{svg}");
        assert!(svg.trim_end().ends_with("</svg>"));
        // balanced tags: every opening tag is either self-closing or closed by
        // the tag it opened, and no '<' or '>' is left over
        let mut stack: Vec<&str> = Vec::new();
        let mut rest = svg.as_str();
        while let Some(open) = rest.find('<') {
            rest = &rest[open..];
            let close = rest
                .find('>')
                .ok_or_else(|| anyhow::anyhow!("an unterminated tag"))?;
            let body = &rest[1..close];
            rest = &rest[close + 1..];
            if body.starts_with('?') || body.starts_with('!') {
                continue;
            }
            if let Some(name) = body.strip_prefix('/') {
                assert_eq!(stack.pop(), Some(name.trim()), "unbalanced SVG: </{name}>");
            } else if !body.ends_with('/') {
                stack.push(body.split_whitespace().next().unwrap_or_default());
            }
        }
        assert!(stack.is_empty(), "unclosed tags: {stack:?}");
        assert_eq!(svg.matches('<').count(), svg.matches('>').count());
        // every attribute value is quoted, so no quote is left dangling
        assert_eq!(
            svg.matches('"').count() % 2,
            0,
            "an attribute value is unterminated"
        );
        Ok(())
    }

    #[test]
    fn test_the_svg_carries_the_evidence_through_the_file() -> anyhow::Result<()> {
        let scene = worked_example()?;
        let svg = render_svg(&scene);
        assert!(svg.contains("data-evidence=\"measured\""), "{svg}");
        assert!(svg.contains("data-evidence=\"exact\""), "{svg}");
        assert!(svg.contains("data-evidence=\"asserted\""), "{svg}");
        assert!(svg.contains("data-wobble=\"0.2\""), "{svg}");
        assert!(svg.contains("data-source=\"hough-grid\""), "{svg}");
        assert!(svg.contains("data-source=\"tesseract\""), "{svg}");
        assert!(svg.contains("data-entity=\"point:A\""), "{svg}");
        // the arc carries the parameters a reader needs, and a real path
        assert!(svg.contains("data-radius=\"0.125\""), "{svg}");
        assert!(svg.contains("data-sweep=\"360\""), "{svg}");
        assert!(svg.contains("<path id=\"a0\" d=\"M "), "{svg}");
        Ok(())
    }

    #[test]
    fn test_label_text_is_escaped_and_reads_back() -> anyhow::Result<()> {
        let tricky = "A & B < C> \"D\"";
        let scene = GroundedScene::new(
            vec![stroke((0.25, 0.25), (0.75, 0.25), measured(0.2))],
            Vec::new(),
            vec![LabelAssociation::new(
                Label::new(tricky, (0.25, 0.25)),
                EntityRef::Point("A".to_string()),
                asserted(),
            )],
            Vec::new(),
        )?;
        let svg = render_svg(&scene);
        assert!(svg.contains("A &amp; B &lt; C&gt;"), "{svg}");
        assert!(
            !svg.contains("< C>"),
            "raw angle brackets in a text node: {svg}"
        );
        let regrounded = round_trip(&scene)?;
        assert_eq!(regrounded.labels[0].label.text, tricky);
        assert_eq!(regrounded.scene, scene.scene);
        Ok(())
    }

    #[test]
    fn test_the_round_trip_reconstructs_the_graph_it_came_from() -> anyhow::Result<()> {
        // the grounding test: a figure rendered to SVG and read back must
        // reconstruct the graph it came from -- points, segments, labels, facts
        // and the whole ledger
        let scene = worked_example()?;
        let regrounded = round_trip(&scene)?;
        let (before, after) = (scene.grounding()?, regrounded.grounding()?);
        assert_eq!(before.points, after.points);
        assert_eq!(before.segments, after.segments);
        assert_eq!(before.intersections, after.intersections);
        assert_eq!(scene.labels, regrounded.labels);
        assert_eq!(scene.arcs, regrounded.arcs);
        assert_eq!(scene.strokes, regrounded.strokes);
        assert_eq!(scene.near_misses, regrounded.near_misses);
        assert_eq!(before.graph, after.graph);
        // the corners came back named by the figure, not by the mint, and the
        // circle's centre -- which is also the crossing -- came back as one
        // point, not two
        assert_eq!(
            after.point_names(),
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
                "O1".to_string(),
                "V1".to_string(),
                "V2".to_string()
            ]
        );
        // and the crossing, the shared corners and the circle all survived
        assert!(after
            .intersections
            .iter()
            .any(|hit| hit.at == (0.5, 0.5) && hit.kind == IntersectionKind::Crossing));
        assert!(after
            .intersections
            .iter()
            .any(|hit| hit.kind == IntersectionKind::SharedEndpoint));
        assert_eq!(after.graph.circles.len(), 1);
        Ok(())
    }

    #[test]
    fn test_evidence_survives_the_round_trip_unweakened() -> anyhow::Result<()> {
        // a figure must not come back more certain than it went out: the grades
        // ride through the file in the data-evidence attributes
        let scene = worked_example()?;
        let regrounded = round_trip(&scene)?;
        assert_eq!(regrounded.strokes[0].evidence, measured(0.2));
        assert_eq!(regrounded.strokes[4].evidence, Evidence::Exact);
        assert_eq!(regrounded.arcs[0].evidence, measured(0.3));
        assert_eq!(regrounded.labels[0].evidence, asserted());
        assert!(regrounded
            .strokes
            .iter()
            .all(|stroke| !stroke.evidence.is_trustworthy() || stroke.evidence == Evidence::Exact));
        // the graded incidences come back at the same grade, not at 1.0. The
        // junction at (1/2, 1/2) is also the circle's centre, so the point is
        // named O1 even though it is a crossing.
        let incidence = Constraint::Collinear {
            a: "A".to_string(),
            b: "O1".to_string(),
            c: "C".to_string(),
        };
        let worth = scene.scene.confidence_in(&incidence)?;
        assert!(
            worth < 1.0,
            "the graded incidence came back at certainty: {worth}"
        );
        assert_relative_eq!(worth, MEASURED_CEILING / 1.2, epsilon = 1.0e-12);
        assert_relative_eq!(
            regrounded.scene.confidence_in(&incidence)?,
            worth,
            epsilon = 1.0e-12
        );
        Ok(())
    }

    #[test]
    fn test_a_figure_carries_no_statement() -> anyhow::Result<()> {
        // the ink went out, the words did not: the premises are re-attached by
        // the round trip rather than read back, and a parser on its own
        // recovers a figure with no premises at all
        let claim = Constraint::Parallel {
            first: Segment {
                from: "A".to_string(),
                to: "B".to_string(),
            },
            second: Segment {
                from: "D".to_string(),
                to: "C".to_string(),
            },
        };
        let scene = GroundedScene::new(
            vec![
                stroke((0.25, 0.25), (0.75, 0.25), measured(0.2)),
                stroke((0.25, 0.75), (0.75, 0.75), measured(0.2)),
            ],
            Vec::new(),
            vec![
                point_label("A", (0.25, 0.25), "A"),
                point_label("B", (0.75, 0.25), "B"),
                point_label("C", (0.75, 0.75), "C"),
                point_label("D", (0.25, 0.75), "D"),
            ],
            vec![claim.clone()],
        )?;
        let parsed = parse_svg(&render_svg(&scene))?;
        assert!(
            parsed.stated.is_empty(),
            "a drawing does not state its premises"
        );
        assert!(!parsed.scene.has_fact(&claim));
        let regrounded = round_trip(&scene)?;
        assert_eq!(regrounded.stated, scene.stated);
        assert_eq!(regrounded.scene, scene.scene);
        assert!(regrounded.scene.has_fact(&claim));
        assert_eq!(regrounded.scene.confidence_in(&claim)?, 1.0);
        Ok(())
    }

    #[test]
    fn test_arcs_become_circles_only_when_they_close() -> anyhow::Result<()> {
        let open = Arc {
            center: (0.5, 0.5),
            radius: 0.125,
            sweep_degrees: 90.0,
            evidence: measured(0.3),
        };
        let closed = Arc {
            sweep_degrees: 360.0,
            ..open.clone()
        };
        assert!(!open.is_closed() && closed.is_closed());
        // a quarter turn gives a centre and no circle: the kernel's KCircle
        // carries an exact squared radius, and a partial arc determines none
        let scene = GroundedScene::new(Vec::new(), vec![open.clone()], Vec::new(), Vec::new())?;
        assert!(scene.scene.circles.is_empty());
        assert_eq!(scene.grounding()?.point_names(), vec!["O1".to_string()]);
        assert!(scene
            .scene
            .not_established
            .iter()
            .any(|line| line.contains("no circle")));
        // a closed one does
        let scene = GroundedScene::new(Vec::new(), vec![closed.clone()], Vec::new(), Vec::new())?;
        assert_eq!(scene.scene.circles.len(), 1);
        let circle = scene.scene.circles[0].clone();
        assert_eq!(circle.center, "O1");
        assert_eq!(
            (circle.radius_sq.num, circle.radius_sq.den),
            (1, 64),
            "r^2 = 1/64"
        );
        assert!(scene.scene.has_fact(&Constraint::Circle {
            name: circle.name.clone(),
            center: circle.center.clone(),
            radius_sq: circle.radius_sq
        }));
        // and the object itself is not a certainty: it is an observation
        let fact = scene
            .scene
            .facts
            .iter()
            .find(|fact| matches!(&fact.constraint, Constraint::Circle { name, .. } if *name == circle.name))
            .ok_or_else(|| anyhow::anyhow!("the circle observation is missing"))?;
        assert!(!fact.is_established());
        Ok(())
    }

    #[test]
    fn test_to_scene_graph_rebuilds_the_stored_graph() -> anyhow::Result<()> {
        // not a getter: a rebuild, so the identity is a check on the
        // reconstruction rather than a tautology
        let scene = worked_example()?;
        assert_eq!(scene.to_scene_graph()?, scene.scene);
        assert_eq!(scene.scene.geometry, "euclidean");
        assert!(!scene.scene.points.is_empty());
        Ok(())
    }

    #[test]
    fn test_a_figure_with_a_non_finite_coordinate_is_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let strokes = vec![stroke((bad, 0.5), (0.8, 0.5), measured(0.2))];
            let failure = GroundedScene::new(strokes, Vec::new(), Vec::new(), Vec::new());
            assert!(
                failure.is_err(),
                "a coordinate that is not a number must be refused"
            );
        }
        // and an arc with no radius is refused too
        let arcs = vec![Arc {
            center: (0.5, 0.5),
            radius: 0.0,
            sweep_degrees: 360.0,
            evidence: measured(0.2),
        }];
        assert!(GroundedScene::new(Vec::new(), arcs, Vec::new(), Vec::new()).is_err());
    }

    #[test]
    fn test_an_empty_figure_grounds_to_an_empty_graph() -> anyhow::Result<()> {
        // no strokes is a figure with nothing in it, not a failure -- and not a
        // figure with invented points either
        let scene = GroundedScene::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())?;
        assert!(scene.scene.points.is_empty());
        assert!(scene.scene.facts.is_empty());
        assert!(scene.scene.circles.is_empty());
        assert!(scene.scene.not_established.is_empty());
        assert_eq!(render_svg(&scene), "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" viewBox=\"0 0 1 1\" width=\"512\" height=\"512\">\n</svg>\n");
        assert_eq!(round_trip(&scene)?.scene, scene.scene);
        Ok(())
    }

    #[test]
    fn test_entity_references_survive_the_file_encoding() -> anyhow::Result<()> {
        for entity in [
            EntityRef::Point("A".to_string()),
            EntityRef::Circle("K1".to_string()),
            EntityRef::Segment {
                from: "A".to_string(),
                to: "B".to_string(),
            },
        ] {
            assert_eq!(EntityRef::decode(&entity.encode())?, entity);
        }
        assert_eq!(EntityRef::Point("A".to_string()).name(), "A");
        assert_eq!(
            EntityRef::Segment {
                from: "A".to_string(),
                to: "B".to_string()
            }
            .name(),
            "AB"
        );
        // a malformed binding is an error, never a default
        assert!(EntityRef::decode("A").is_err());
        assert!(EntityRef::decode("point:").is_err());
        assert!(EntityRef::decode("wormhole:A").is_err());
        assert!(EntityRef::decode("segment:A").is_err());
        Ok(())
    }

    #[test]
    fn test_the_report_names_what_the_picture_only_suggested() -> anyhow::Result<()> {
        let report = ground_bare(wobbly_pair())?.report();
        assert!(
            report.contains("not established: suggested: parallel("),
            "{report}"
        );
        assert!(report.contains("logical state:   UNKNOWN"), "{report}");
        assert!(report.contains("figure: 2 stroke(s)"), "{report}");
        // a crossing the detector did find is reported as an observation, with
        // the detector that made it
        let report = ground_bare(exact_crossing(measured(0.4)))?.report();
        assert!(
            report.contains("observed: crossing of strokes 0 and 1"),
            "{report}"
        );
        assert!(
            report.contains("measured by hough-grid+hough-grid"),
            "{report}"
        );
        assert!(
            report.contains("supported, not established: collinear(V1,X1,V2)"),
            "{report}"
        );
        Ok(())
    }
}
